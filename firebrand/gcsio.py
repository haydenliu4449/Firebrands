"""Cloud Storage paths, everywhere a local path used to work.

Anything in the kit that takes a path now also takes `gs://bucket/key`. Videos
are downloaded to local disk first and cached -- OpenCV cannot decode from a
network stream, and even if it could, seeking over HTTP would be far slower
than one sequential download.

Two backends, tried in order:
  1. `google-cloud-storage` (pip install google-cloud-storage) -- preferred.
  2. the `gcloud storage` CLI -- already present on every Google Cloud VM, so
     the kit works out of the box on Workbench with nothing installed.

Authentication is never handled here. On a Google Cloud VM the attached service
account is picked up automatically; on your laptop run
`gcloud auth application-default login` once. If you find yourself downloading
a JSON key file, stop -- you almost certainly do not need one, and a key file in
a repo is the single most common way a cloud project gets compromised.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

GCS_PREFIX = "gs://"


def is_gcs(path) -> bool:
    return str(path).startswith(GCS_PREFIX)


def split(path):
    """gs://bucket/a/b.mp4 -> ('bucket', 'a/b.mp4')"""
    rest = str(path)[len(GCS_PREFIX):]
    bucket, _, key = rest.partition("/")
    return bucket, key


# ---------------------------------------------------------------------------
# backend selection
# ---------------------------------------------------------------------------

def _client():
    try:
        from google.cloud import storage  # noqa: F401
    except ImportError:
        return None
    from google.cloud import storage
    if not hasattr(_client, "_c"):
        _client._c = storage.Client()
    return _client._c


def _gcloud() -> str | None:
    return shutil.which("gcloud") or shutil.which("gsutil")


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise IOError(f"{' '.join(cmd)}\n{r.stderr.strip()}")
    return r.stdout


def available() -> str:
    """Which backend will be used, for diagnostics."""
    if _client() is not None:
        return "google-cloud-storage"
    if _gcloud():
        return "gcloud CLI"
    return "none"


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------

def exists(path) -> bool:
    if not is_gcs(path):
        return Path(path).exists()
    c = _client()
    if c is not None:
        b, k = split(path)
        return c.bucket(b).blob(k).exists()
    g = _gcloud()
    if not g:
        raise RuntimeError(_no_backend_msg())
    return subprocess.run([g, "storage", "ls", str(path)],
                          capture_output=True).returncode == 0


def listdir(prefix):
    """List objects under a gs:// prefix (or files in a local directory)."""
    if not is_gcs(prefix):
        return sorted(str(p) for p in Path(prefix).iterdir())
    c = _client()
    if c is not None:
        b, k = split(prefix)
        return [f"{GCS_PREFIX}{b}/{o.name}"
                for o in c.list_blobs(b, prefix=k.rstrip("/") + "/")]
    g = _gcloud()
    if not g:
        raise RuntimeError(_no_backend_msg())
    out = _run([g, "storage", "ls", str(prefix).rstrip("/") + "/"])
    return [l.strip() for l in out.splitlines() if l.strip()]


def download(path, dest) -> str:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    c = _client()
    if c is not None:
        b, k = split(path)
        c.bucket(b).blob(k).download_to_filename(str(dest))
        return str(dest)
    g = _gcloud()
    if not g:
        raise RuntimeError(_no_backend_msg())
    _run([g, "storage", "cp", str(path), str(dest)])
    return str(dest)


def upload(local, path) -> str:
    c = _client()
    if c is not None:
        b, k = split(path)
        c.bucket(b).blob(k).upload_from_filename(str(local))
        return str(path)
    g = _gcloud()
    if not g:
        raise RuntimeError(_no_backend_msg())
    _run([g, "storage", "cp", str(local), str(path)])
    return str(path)


def upload_dir(local_dir, gs_prefix) -> int:
    """Mirror a local directory up to a gs:// prefix. Returns file count."""
    local_dir = Path(local_dir)
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    if not files:
        return 0
    g = _gcloud()
    if g:                                   # one recursive copy beats N round-trips
        _run([g, "storage", "cp", "-r", f"{local_dir}/.",
              str(gs_prefix).rstrip("/") + "/"])
        return len(files)
    for p in files:
        rel = p.relative_to(local_dir).as_posix()
        upload(p, f"{str(gs_prefix).rstrip('/')}/{rel}")
    return len(files)


# ---------------------------------------------------------------------------
# the function everything else calls
# ---------------------------------------------------------------------------

def cache_dir() -> Path:
    d = Path(os.environ.get("FIREBRAND_CACHE", Path.home() / ".firebrand_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def localize(path, verbose=True) -> str:
    """Return a local path for `path`, downloading from GCS if needed.

    Cached by bucket+key hash, so re-running a notebook cell does not re-download
    a 2 GB clip. Delete ~/.firebrand_cache (or set FIREBRAND_CACHE) to clear.
    """
    if not is_gcs(path):
        return str(path)

    b, k = split(path)
    tag = hashlib.sha1(f"{b}/{k}".encode()).hexdigest()[:12]
    dest = cache_dir() / f"{tag}_{Path(k).name}"
    if dest.exists() and dest.stat().st_size > 0:
        if verbose:
            print(f"  cached: {path} -> {dest}")
        return str(dest)

    if verbose:
        print(f"  downloading {path} ...", end=" ", flush=True)
    download(path, dest)
    if verbose:
        print(f"{dest.stat().st_size / 1e6:.0f} MB")
    return str(dest)


def _no_backend_msg():
    return ("No Cloud Storage backend available.\n"
            "  Either:  pip install google-cloud-storage\n"
            "  or:      install the gcloud CLI (already present on Google Cloud VMs)\n"
            "Then authenticate once with:  gcloud auth application-default login")


def check(verbose=True) -> bool:
    """Print a short diagnosis of the GCS setup. Call this first on a new VM."""
    backend = available()
    if verbose:
        print(f"Cloud Storage backend: {backend}")
    if backend == "none":
        if verbose:
            print(_no_backend_msg())
        return False
    try:
        proj = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        if not proj and _gcloud():
            proj = _run([_gcloud(), "config", "get-value", "project"]).strip()
        if verbose:
            print(f"project: {proj or '(not set -- gcloud config set project YOUR_ID)'}")
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"could not read project: {e}", file=sys.stderr)
    return True
