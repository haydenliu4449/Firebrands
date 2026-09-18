"""Decoding, frame sources, ROI masks, and the sanity checks that must pass
before anything downstream is trustworthy.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import gcsio


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def probe(video_path) -> dict:
    """Container metadata. Read this before you believe any velocity."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate,nb_frames,width,height,pix_fmt,bit_rate,codec_name",
        "-of", "json", str(video_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]

    def _rate(x):
        if not x or x == "0/0":
            return None
        n, d = x.split("/")
        return float(n) / float(d) if float(d) else None

    return {
        "codec": s.get("codec_name"),
        "width": int(s["width"]),
        "height": int(s["height"]),
        "pix_fmt": s.get("pix_fmt"),
        "bit_rate": int(s["bit_rate"]) if s.get("bit_rate") else None,
        "nb_frames": int(s["nb_frames"]) if s.get("nb_frames") else None,
        "r_frame_rate": _rate(s.get("r_frame_rate")),
        "avg_frame_rate": _rate(s.get("avg_frame_rate")),
    }


def decode_to_frames(video_path, out_dir, ext="png"):
    """Extract exactly the frames in the container -- no re-timing, no drops.

    `-vsync 0` matters: the default (`cfr`) invents or discards frames to hit a
    constant rate, which silently corrupts every timing measurement you make
    downstream.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
           "-vsync", "0", "-start_number", "0", str(out_dir / f"%06d.{ext}")]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob(f"*.{ext}"))


# ---------------------------------------------------------------------------
# frame sources
# ---------------------------------------------------------------------------

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".dav", ".asf", ".ts", ".wmv")


def _explain_missing(path) -> str:
    """A path error should say which of the three things went wrong: the file
    is not there, the folder is not there, or the file is there and the codec
    is unreadable. 'cannot open X' says none of them."""
    if gcsio.is_gcs(path):
        return (f"cannot open {path}\n"
                f"  This is a Cloud Storage path. Check that it exists:\n"
                f"    gcloud storage ls {path}\n"
                f"  and that you are authenticated:\n"
                f"    gcloud auth application-default login\n"
                f"  backend in use: {gcsio.available()}")
    p = Path(path)
    lines = [f"cannot open {p}",
             f"  working directory: {Path.cwd()}",
             f"  resolved to:       {p if p.is_absolute() else (Path.cwd() / p)}"]

    if p.exists():
        lines.append("  The file EXISTS but OpenCV could not decode it. Likely an")
        lines.append("  unsupported codec (some DVRs write .dav/H.265). Convert first:")
        lines.append(f'    ffmpeg -i "{p}" -c:v libx264 -crf 18 -an "{p.with_suffix(".mp4").name}"')
        return "\n".join(lines)

    if not p.parent.exists():
        lines.append(f"  The folder '{p.parent}' does not exist either.")
        lines.append(f"    mkdir -p '{p.parent}'    # then put your video files in it")
    else:
        there = sorted(q.name for q in p.parent.iterdir() if q.is_file())[:12]
        lines.append(f"  '{p.parent}' exists but has no file named '{p.name}'.")
        lines.append(f"  It contains: {', '.join(there) if there else '(nothing)'}")

    found, seen = [], set()
    for root in (Path.cwd(), Path.cwd().parent, Path.home() / "Downloads",
                 Path.home() / "downloads", Path.home() / "Desktop"):
        try:
            if not root.exists():
                continue
            for q in root.rglob("*"):
                if q.is_file() and q.suffix.lower() in VIDEO_EXTS:
                    key = q.resolve()
                    if key in seen:
                        continue
                    seen.add(key)
                    found.append(q)
                    if len(found) >= 8:
                        break
        except (PermissionError, OSError):
            continue
        if len(found) >= 8:
            break

    if found:
        lines.append("")
        lines.append("  Video files I can see nearby -- pass one of these instead:")
        for q in found:
            lines.append(f'    "{q}"')
    else:
        lines.append("")
        lines.append("  No video files found nearby. Copy your clips into a folder")
        lines.append("  first, then pass the path to one of them.")
    return "\n".join(lines)


class VideoSource:
    """Iterate BGR frames from a video file, local or gs://.

    GCS objects are downloaded to a local cache first. OpenCV cannot decode
    from a network stream, and a sequential download is faster than seeking
    over HTTP would be even if it could.
    """

    def __init__(self, path):
        self.uri = str(path)
        self.path = gcsio.localize(path) if gcsio.is_gcs(path) else str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(_explain_missing(path))
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise IOError(_explain_missing(path))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or None
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        self.shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cap.release()

    def __iter__(self):
        cap = cv2.VideoCapture(self.path)
        try:
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield f
        finally:
            cap.release()

    def __len__(self):
        return self.n or 0


class FrameDirSource:
    """Iterate BGR frames from a directory of images."""

    def __init__(self, directory, pattern="*.png"):
        self.paths = sorted(Path(directory).glob(pattern))
        if not self.paths:
            raise IOError(f"no frames matching {pattern} in {directory}")
        first = cv2.imread(str(self.paths[0]))
        self.shape = first.shape[:2]
        self.fps = None
        self.n = len(self.paths)

    def __iter__(self):
        for p in self.paths:
            yield cv2.imread(str(p))

    def __len__(self):
        return self.n


class ArraySource:
    """Iterate an in-memory list/array of BGR frames (used by the tests)."""

    def __init__(self, frames, fps=None):
        self.frames = frames
        self.shape = frames[0].shape[:2]
        self.fps = fps
        self.n = len(frames)

    def __iter__(self):
        return iter(self.frames)

    def __len__(self):
        return self.n


# ---------------------------------------------------------------------------
# sanity checks
# ---------------------------------------------------------------------------

def is_ir(frame, tol=2.0) -> bool:
    """True if the frame is effectively monochrome (IR night mode).

    Colour separation is the whole basis of the blue-channel trick. If the
    camera has flipped to IR the three channels carry the same signal and the
    detector must fall back to grayscale contrast.
    """
    b, g, r = frame[..., 0].astype(np.float32), frame[..., 1].astype(np.float32), frame[..., 2].astype(np.float32)
    return float(np.mean(np.abs(b - r)) + np.mean(np.abs(b - g))) < tol


def duplicate_frame_fraction(source, max_frames=300, tol=0.05) -> float:
    """Fraction of consecutive frame pairs that are (near-)identical.

    Many DVRs advertise 30 fps but repeat frames from a 15 fps sensor. If this
    comes back near 0.5, your true capture rate is half what the container
    says, and every velocity you compute is wrong by 2x.
    """
    prev, dup, total = None, 0, 0
    for i, f in enumerate(source):
        if i >= max_frames:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            total += 1
            if float(np.mean(np.abs(g.astype(np.int16) - prev.astype(np.int16)))) < tol:
                dup += 1
        prev = g
    return dup / total if total else 0.0


def channel_separability(frame, top_pct=0.05) -> dict:
    """Streak-to-background distance in background sigmas, per channel.

    Reproduces the numbers in the approach note on any frame you hand it. Use
    it to confirm the blue channel is still the right choice on your clips.
    """
    a = frame.astype(np.float32)
    lum = a.mean(2)
    thr = np.percentile(lum, 100 - top_pct)
    fg, bg = lum > thr, lum <= thr
    if fg.sum() < 20:
        return {}
    chans = {
        "blue": a[..., 0], "green": a[..., 1], "red": a[..., 2],
        "gray": lum, "minrgb": a.min(2),
        "blue-0.15red": a[..., 0] - 0.15 * a[..., 2],
    }
    out = {}
    for k, v in chans.items():
        s = v[bg].std()
        out[k] = float((v[fg].mean() - v[bg].mean()) / s) if s > 1e-6 else 0.0
    return out


def audit(source, name="clip") -> dict:
    """Run every pre-flight check and return a report. Read it before trusting
    anything the pipeline produces."""
    frames = []
    for i, f in enumerate(source):
        if i >= 300:
            break
        frames.append(f)
    if not frames:
        raise IOError("no frames")

    rep = {
        "name": name,
        "n_frames_checked": len(frames),
        "shape": frames[0].shape[:2],
        "ir_night_mode": bool(is_ir(frames[0])),
        "duplicate_fraction": duplicate_frame_fraction(ArraySource(frames)),
        "separability": channel_separability(frames[len(frames) // 2]),
    }
    sep = rep["separability"]
    if sep:
        rep["best_channel"] = max(sep, key=sep.get)
    rep["warnings"] = []
    if rep["ir_night_mode"]:
        rep["warnings"].append(
            "Monochrome/IR frames: blue-channel advantage is gone, "
            "set DetectConfig.channel='gray' and expect lower separability.")
    if rep["duplicate_fraction"] > 0.15:
        rep["warnings"].append(
            f"{rep['duplicate_fraction']:.0%} of frame pairs are duplicates: the true "
            "capture rate is below the container fps. Fix TrackConfig.fps or "
            "every speed you measure will be scaled wrong.")
    if sep.get("blue", 0) < 6:
        rep["warnings"].append(
            "Blue-channel separability is low on this clip; check exposure and "
            "whether the scene is glow-lit.")
    return rep


# ---------------------------------------------------------------------------
# ROI masks
# ---------------------------------------------------------------------------

def make_roi_mask(shape, exclude_boxes=(), exclude_polys=(), normalized=False) -> np.ndarray:
    """255 = analyse, 0 = ignore.

    Coordinates may be absolute pixels or, with `normalized=True`, fractions of
    width/height -- which is what you want for anything you intend to reuse,
    since clips from the same camera are not always exactly 1920x1080 (the
    sample frame is 1917x1079) and an exact-size check would silently skip the
    mask entirely.

    Always exclude:
      * the burned-in timestamp and camera label -- permanent maximum-brightness
        objects that will anchor anything you train,
      * glass and painted metal that reflects the sky -- reflections produce
        mirrored duplicate firebrands, which double-count flux and generate
        physically impossible trajectories.
    """
    H, W = shape[:2]
    sx, sy = (W, H) if normalized else (1, 1)
    m = np.full((H, W), 255, np.uint8)
    for (x0, y0, x1, y1) in exclude_boxes:
        m[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)] = 0
    for poly in exclude_polys:
        pts = np.asarray([(p[0] * sx, p[1] * sy) for p in poly], np.int32)
        cv2.fillPoly(m, [pts], 0)
    return m


# The burned-in overlays on the 'Left Front' camera, in fractions of the frame
# so they survive a resize. These two are measured off the sample frame and are
# correct.
#
# NOTE: no reflection polygons are shipped. An earlier version guessed at the
# glass and hood; rendered over the real frame the guesses covered driveway
# rather than glass, which would have thrown away good detections while leaving
# the reflections in. Draw your own with `python make_mask.py <clip>` -- it
# takes two minutes and it is the one manual step in the whole pipeline that
# nobody can do for you.
LEFT_FRONT_OVERLAYS = dict(
    normalized=True,
    exclude_boxes=[
        (0.010, 0.037, 0.271, 0.102),   # 01-08-2025 Wed 00:05:36
        (0.714, 0.713, 0.844, 0.769),   # "Left Front"
    ],
    exclude_polys=[],
)

# Backwards-compatible alias.
LEFT_FRONT_1080P = LEFT_FRONT_OVERLAYS

# The driveway on the 'Left Front' camera: the ground plane where saltation
# contacts happen, and the region a mask must not eat into. Used by
# review.mask_cost to report what a candidate mask actually costs.
DRIVEWAY_LEFT_FRONT = [
    (0.125, 0.40), (0.19, 0.345), (0.30, 0.295), (0.55, 0.305), (1.0, 0.278),
    (1.0, 1.0), (0.215, 1.0), (0.20, 0.62),
]


def save_mask_spec(path, spec):
    if gcsio.is_gcs(path):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(spec, fh, indent=2)
            tmp = fh.name
        gcsio.upload(tmp, path)
        Path(tmp).unlink(missing_ok=True)
        return
    json.dump(spec, open(path, "w"), indent=2)


def load_mask_spec(path):
    local = gcsio.localize(path, verbose=False) if gcsio.is_gcs(path) else path
    return json.load(open(local))


def mask_from_spec(shape, spec):
    return make_roi_mask(shape,
                         exclude_boxes=spec.get("exclude_boxes", []),
                         exclude_polys=spec.get("exclude_polys", []),
                         normalized=spec.get("normalized", False))


def overlay_mask(frame, mask, alpha=0.45):
    """Visualise a mask so you can check it before running 40 000 frames."""
    vis = frame.copy()
    red = np.zeros_like(vis)
    red[..., 2] = 255
    off = mask == 0
    vis[off] = (alpha * red[off] + (1 - alpha) * vis[off]).astype(np.uint8)
    return vis
