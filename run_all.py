#!/usr/bin/env python3
"""The whole pipeline as a resumable script. No notebook, no kernel to lose.

    python run_all.py --clips clips/ --work work --mask masks/left_front.json

Runs detect -> tiles -> train -> eval. **Every stage writes its result to disk
and is skipped if that result already exists**, so a crash, a dropped
connection, or a machine you turned off costs you only the stage that was
running -- never the hours before it.

Run it detached so nothing in your browser can kill it:

    nohup python run_all.py --clips clips/ --work work > work/log.txt 2>&1 &
    tail -f work/log.txt          # watch it; Ctrl-C only stops watching

Or under tmux on the VM, which also survives closing the tab:

    tmux new -s fb
    python run_all.py --clips clips/ --work work
    # Ctrl-B then D to detach; `tmux attach -t fb` to come back

WHY THIS EXISTS

A Jupyter kernel holds everything in memory. When it reconnects you lose
`frames`, `X`, `Y`, every track you detected -- and there is no way to get them
back except re-running from the top. That is a bad fit for stages that take
tens of minutes, and it is the wrong tool for the compute half of this project.

Use the notebook for LOOKING at results (plots, contact sheets, diagnostics)
and this script for PRODUCING them. Each stage here leaves files the notebook
can load in seconds.

STAGES

  detect   each clip -> tracks.csv, detections.csv, contact sheets, heatmap
  tiles    verified tracks -> fitted generator -> data/tiles.npz
  train    tiles -> runs/streak/best.pt
  eval     model vs classical detector -> runs/streak/report.json

Between `detect` and `tiles`, review the contact sheets and write the track IDs
that are NOT firebrands into work/clips/<clip>/reject.txt (commas or newlines).
Re-run with --stage tiles afterwards.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from firebrand import Config, frames as F, detect as D, track as T, review as R
from firebrand import synth as S, dataset as DS, evaluate as E, gcsio

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


class _Gen:
    """Re-iterable frame source over a generator factory.

    The pipeline's consumers each want to walk the clip from the start, and a
    plain generator is exhausted after one pass. This re-opens the stream per
    iteration so nothing has to hold the frames.
    """

    def __init__(self, factory):
        self.factory = factory

    def __iter__(self):
        return iter(self.factory())


# ---------------------------------------------------------------------------
# logging: timestamped and flushed, so `tail -f` is actually live
# ---------------------------------------------------------------------------

def log(msg, *, rule=False):
    if rule:
        print("\n" + "=" * 68, flush=True)
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
    if rule:
        print("=" * 68, flush=True)


class Timer:
    def __init__(self, what):
        self.what = what

    def __enter__(self):
        self.t0 = time.time()
        log(f"{self.what} ...")
        return self

    def __exit__(self, *a):
        log(f"{self.what} done in {time.time() - self.t0:.0f}s")


def progress_printer(total, every_s=20.0):
    """Sparse progress for a log file. A carriage-return bar is useless in
    nohup output -- it just writes one enormous line."""
    state = {"last": 0.0, "t0": time.time()}

    def p(k):
        now = time.time()
        if now - state["last"] < every_s and k + 1 < total:
            return
        state["last"] = now
        el = now - state["t0"]
        rate = (k + 1) / max(el, 1e-6)
        log(f"    frame {k+1}/{total}  {rate:.1f} fps  "
            f"eta {(total - k - 1)/max(rate,1e-6)/60:.1f} min")
    return p


# ---------------------------------------------------------------------------
# stage: detect
# ---------------------------------------------------------------------------

def list_clips(spec):
    if gcsio.is_gcs(spec):
        return [o for o in gcsio.listdir(spec)
                if Path(o).suffix.lower() in VIDEO_EXTS]
    p = Path(spec)
    if p.is_file():
        return [str(p)]
    return sorted(str(q) for q in p.iterdir()
                  if q.suffix.lower() in VIDEO_EXTS)


def stage_detect(args, cfg):
    clips = list_clips(args.clips)
    if not clips:
        raise SystemExit(f"no video files found in {args.clips}")
    log(f"{len(clips)} clip(s) to process")

    for clip in clips:
        name = Path(clip).stem
        out = Path(args.work) / "clips" / name
        flag = out / "done.json"
        if flag.exists() and not args.force:
            log(f"  {name}: already done, skipping (use --force to redo)")
            continue

        out.mkdir(parents=True, exist_ok=True)

        # STREAMED, never materialised. At 3840x2160 a frame is 25 MB, so a
        # 300-frame list is 7.5 GB -- and the audit used to keep a second copy.
        # Detection only ever needs an 11-frame rolling window (275 MB), and
        # every other consumer here can take a second pass over the file.
        src = F.VideoSource(clip)

        def stream(limit=args.max_frames):
            for i, f in enumerate(src):
                if limit and i >= limit:
                    return
                yield f

        first = next(iter(stream(1)), None)
        if first is None:
            log(f"  {name}: no frames decoded, skipping")
            continue
        H, W = first.shape[:2]
        n_frames = min(len(src) or 10 ** 9, args.max_frames or 10 ** 9)

        c = Config()
        c.detect.__dict__.update(cfg.detect.__dict__)
        c.track.__dict__.update(cfg.track.__dict__)
        c.track.fps = float(src.fps or 15)
        c.detect.tophat_ksize = max(5, int(round(15 * W / 1920)) | 1)
        if W * H > 2_500_000 and c.detect.bg_refresh == 1:
            c.detect.bg_refresh = 4

        with Timer(f"  {name}: auditing"):
            audit = F.audit(_Gen(stream), name)
        json.dump(audit, open(out / "audit.json", "w"), indent=2, default=str)
        log(f"  {name}: {n_frames} frames {W}x{H} @ {c.track.fps:.1f}fps, "
            f"blue sep {audit['separability'].get('blue', 0):.1f}σ, "
            f"dup {audit['duplicate_fraction']:.0%}")
        for w in audit["warnings"]:
            log(f"    !! {w}")
        if audit["ir_night_mode"]:
            c.detect.channel = "gray"

        roi = None
        if args.mask:
            roi = F.mask_from_spec((H, W), F.load_mask_spec(args.mask))
        elif not args.no_mask:
            roi = F.make_roi_mask((H, W), **F.LEFT_FRONT_OVERLAYS)
        if roi is not None:
            cv2.imwrite(str(out / "roi_mask.png"), F.overlay_mask(first, roi))

        with Timer(f"  {name}: detecting"):
            dets = D.detect_source(_Gen(stream), c.detect, roi,
                                   progress=progress_printer(n_frames))

        ratio, diag = T.estimate_step_ratio(dets, c.track, return_diagnostics=True)
        c.track.step_ratio = ratio
        log(f"  {name}: step_ratio {ratio:.2f} (sharpness {diag['sharpness']:.1f})")
        tracks = T.link(dets, c.track, step_ratio=ratio)
        acc = T.accept(tracks, c.track)
        rej = T.rejected(tracks, acc)
        log(f"  {name}: {sum(len(v) for v in dets.values())} candidates -> "
            f"{len(acc)} accepted tracks, {len(rej)} rejected")

        pd.DataFrame(T.summarize(acc, c.track)).to_csv(out / "tracks.csv", index=False)
        pd.DataFrame([{**d.as_dict(), "track_id": t.id} for t in acc for d in t.dets]) \
            .to_csv(out / "detections.csv", index=False)
        if rej:
            pd.DataFrame(T.summarize(rej, c.track)).to_csv(out / "negatives.csv",
                                                           index=False)
        with open(out / "tracks.pkl", "wb") as fh:
            pickle.dump({"accepted": acc, "rejected": rej}, fh)

        # Second pass: contact-sheet crops, the heatmap frame, background
        # frames and the overlay video, all from one stream.
        with Timer(f"  {name}: review artefacts"):
            R.make_contact_sheets_streaming(acc, _Gen(stream), out / "review")

            bgdir = out / "bg"
            bgdir.mkdir(exist_ok=True)
            by = {}
            for tr in acc:
                for d in tr.dets:
                    by.setdefault(d.frame, []).append(d)
            vw = None
            if args.overlay:
                vw = cv2.VideoWriter(str(out / "overlay.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     c.track.fps, (W, H))
            mid = n_frames // 2
            for i, f in enumerate(stream()):
                if i < 12:
                    cv2.imwrite(str(bgdir / f"{i:03d}.png"), f)
                if i == mid:
                    cv2.imwrite(str(out / "fp_heatmap.png"),
                                R.fp_heatmap(rej, acc, f))
                if vw is not None:
                    vw.write(D.draw(f, by.get(i, [])))
            if vw is not None:
                vw.release()

        c.save(out / "config.json")
        json.dump({"clip": clip, "n_frames": n_frames, "shape": [H, W],
                   "fps": c.track.fps, "n_accepted": len(acc),
                   "n_rejected": len(rej), "step_ratio": ratio},
                  open(flag, "w"), indent=2)

    log("detect stage complete", rule=True)
    log("Review the contact sheets in work/clips/*/review/, then put the track")
    log("IDs that are NOT firebrands in work/clips/<clip>/reject.txt")


# ---------------------------------------------------------------------------
# stage: tiles
# ---------------------------------------------------------------------------

def read_rejects(d: Path):
    f = d / "reject.txt"
    if not f.exists():
        return set()
    txt = f.read_text().replace(",", " ").split()
    return {int(x) for x in txt if x.strip().lstrip("-").isdigit()}


def stage_tiles(args, cfg):
    work = Path(args.work)
    tiles_path = work / "data" / "tiles.npz"
    if tiles_path.exists() and not args.force:
        log(f"tiles already built ({tiles_path}), skipping")
        return

    clip_dirs = sorted(d for d in (work / "clips").iterdir() if d.is_dir()) \
        if (work / "clips").exists() else []
    if not clip_dirs:
        raise SystemExit("no detect output found -- run --stage detect first")

    verified_rows, bg_frames, ratios = [], [], []
    n_rejected_manual = 0
    for d in clip_dirs:
        if not (d / "tracks.pkl").exists():
            continue
        with open(d / "tracks.pkl", "rb") as fh:
            blob = pickle.load(fh)
        rej_ids = read_rejects(d)
        n_rejected_manual += len(rej_ids)
        keep = [t for t in blob["accepted"] if t.id not in rej_ids]
        meta = json.load(open(d / "done.json"))
        c = Config.load(d / "config.json")
        verified_rows += T.summarize(keep, c.track)
        ratios.append(meta.get("step_ratio"))
        if (d / "bg").exists() and len(bg_frames) < 40:
            bg_frames += [cv2.imread(str(q)) for q in sorted((d / "bg").glob("*.png"))]
        log(f"  {d.name}: {len(keep)} verified "
            f"({len(rej_ids)} manually rejected)")

    if not verified_rows:
        raise SystemExit("no verified tracks -- check the detect stage output")
    if not n_rejected_manual:
        log("  !! no reject.txt anywhere. That is fine to start, but the")
        log("     generator is being fitted to unreviewed tracks.")

    cfg.synth = S.fit_config_to_tracks(
        verified_rows, cfg.synth,
        step_ratio=float(np.median([r for r in ratios if r])) if ratios else None)
    log(f"  fitted speed_px {tuple(round(v,1) for v in cfg.synth.speed_px)}, "
        f"peak_srgb {tuple(round(v,1) for v in cfg.synth.peak_srgb)}, "
        f"exposure_duty {cfg.synth.exposure_duty:.3f}")

    bg = bg_frames[:40] if bg_frames else None
    if bg is None:
        raise SystemExit("no background frames saved -- re-run detect")

    H, W = bg[0].shape[:2]
    cfg.detect.tophat_ksize = max(5, int(round(15 * W / 1920)) | 1)
    roi = F.mask_from_spec((H, W), F.load_mask_spec(args.mask)) if args.mask else None

    with Timer(f"  rendering {args.synth_frames} synthetic frames"):
        X, Y, _sf, _sg = DS.build_from_synthetic(bg, args.synth_frames, cfg,
                                                 seed=0, roi_mask=roi)
    log(f"  {DS.tile_stats(X, Y)}")
    tiles_path.parent.mkdir(parents=True, exist_ok=True)
    DS.save_tiles(tiles_path, X, Y)
    cfg.save(work / "config.json")
    log(f"tiles written to {tiles_path}", rule=True)


# ---------------------------------------------------------------------------
# stage: train
# ---------------------------------------------------------------------------

def stage_train(args, cfg):
    work = Path(args.work)
    ck = work / "runs" / "streak" / "best.pt"
    if ck.exists() and not args.force:
        log(f"checkpoint already exists ({ck}), skipping")
        return
    tiles_path = work / "data" / "tiles.npz"
    if not tiles_path.exists():
        raise SystemExit("no tiles -- run --stage tiles first")

    from firebrand import train as TR
    # as_float=False keeps masks uint8; see load_tiles. With float masks the
    # array pair is ~3.4 GB instead of ~1.9 GB at 7k tiles.
    X, Y = DS.load_tiles(tiles_path, as_float=False)
    log(f"  {len(X)} tiles, {X.shape[1:]} each, "
        f"{(X.nbytes + Y.nbytes)/1e9:.2f} GB resident")
    cfg.train.epochs = args.epochs
    with Timer(f"  training {args.epochs} epochs"):
        TR.train(X, Y, cfg.train, out_dir=work / "runs" / "streak", log_every=1)
    log(f"checkpoint at {ck}", rule=True)


# ---------------------------------------------------------------------------
# stage: eval
# ---------------------------------------------------------------------------

def stage_eval(args, cfg):
    work = Path(args.work)
    ck = work / "runs" / "streak" / "best.pt"
    if not ck.exists():
        raise SystemExit("no checkpoint -- run --stage train first")
    report_path = work / "runs" / "streak" / "report.json"
    if report_path.exists() and not args.force:
        log(f"report already exists ({report_path}), skipping")
        return

    import torch
    from firebrand import train as TR, infer as I

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = TR.load(ck, device=dev)

    clip_dirs = sorted(d for d in (work / "clips").iterdir() if d.is_dir())
    bgd = next((d / "bg" for d in clip_dirs if (d / "bg").exists()), None)
    if bgd is None:
        raise SystemExit("no background frames for the held-out set")
    bg = [cv2.imread(str(q)) for q in sorted(bgd.glob("*.png"))]

    H, W = bg[0].shape[:2]
    cfg.detect.tophat_ksize = max(5, int(round(15 * W / 1920)) | 1)
    roi = F.mask_from_spec((H, W), F.load_mask_spec(args.mask)) if args.mask else None

    # Seed 999: never used to build the training tiles, so this is held out.
    with Timer("  rendering held-out evaluation clip"):
        tf, tg = S.make_clip(bg, 60, cfg.synth, seed=999, roi_mask=roi)
    gtf = E.filter_gt(tg, min_L=cfg.detect.min_length, min_peak_srgb=25.0)

    with Timer("  threshold sweep"):
        sweep = I.sweep_threshold(model, F.ArraySource(tf), cfg, gtf, roi, device=dev)
    best = max(sweep, key=lambda r: r["f1"])
    log(f"  best threshold {best['threshold']} (F1 {best['f1']:.3f})")

    with Timer("  learned model"):
        m_dets, m_tracks = I.run_clip(model, F.ArraySource(tf), cfg, roi,
                                      best["threshold"], dev)
    with Timer("  classical detector"):
        c_dets = D.detect_source(F.ArraySource(tf), cfg.detect, roi)
        c_acc = T.accept(T.link(c_dets, cfg.track,
                                step_ratio=T.estimate_step_ratio(c_dets, cfg.track)),
                         cfg.track)

    rep = {"threshold": best["threshold"], "sweep": sweep,
           "learned": E.report(m_dets, m_tracks, gtf),
           "classical": E.report(c_dets, c_acc, gtf)}
    json.dump(rep, open(report_path, "w"), indent=2, default=float)

    log("", rule=True)
    lt = rep["learned"]["trk_track_recall"]
    ct = rep["classical"]["trk_track_recall"]
    log(f"  track recall   classical {ct:.3f}   learned {lt:.3f}")
    log(f"  detection P/R  classical {rep['classical']['det_precision']:.3f}/"
        f"{rep['classical']['det_recall']:.3f}   learned "
        f"{rep['learned']['det_precision']:.3f}/{rep['learned']['det_recall']:.3f}")
    if lt > ct:
        log(f"  The model beats the classical detector by {lt - ct:+.3f} track recall.")
    else:
        log("  The model does NOT beat the classical detector yet. The fix is")
        log("  almost always more or better-matched synthetic data (raise")
        log("  --synth-frames, and verify more tracks), not a bigger model.")
    log(f"  full report: {report_path}")


# ---------------------------------------------------------------------------

STAGES = {"detect": stage_detect, "tiles": stage_tiles,
          "train": stage_train, "eval": stage_eval}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="clips/",
                    help="folder of videos, one video, or a gs:// prefix")
    ap.add_argument("--work", default="work",
                    help="local output directory (push it to a bucket afterwards "
                         "with: bash cloud/upload.sh work/)")
    ap.add_argument("--mask", help="mask JSON from make_mask.py")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--stage", default="all",
                    choices=["all", "detect", "tiles", "train", "eval"])
    ap.add_argument("--force", action="store_true",
                    help="redo stages whose output already exists")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--synth-frames", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()

    Path(args.work).mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.work) / "config.json"
    cfg = Config.load(cfg_path) if cfg_path.exists() else Config()

    order = ["detect", "tiles", "train", "eval"] if args.stage == "all" else [args.stage]
    log(f"stages: {' -> '.join(order)}   work={args.work}", rule=True)

    failed = None
    for st in order:
        log(f"STAGE: {st}", rule=True)
        try:
            STAGES[st](args, cfg)
        except SystemExit as e:
            log(f"STAGE {st} stopped: {e}")
            failed = st
            break
        except Exception:
            log(f"STAGE {st} FAILED:")
            traceback.print_exc()
            failed = st
            break

    log("", rule=True)
    if failed:
        log(f"Stopped at '{failed}'. Everything before it is saved in {args.work} --")
        log(f"fix the problem and re-run; completed stages are skipped.")
        return 1
    log("All stages complete.")
    log("  Push the results to your bucket:  bash cloud/upload.sh " + str(args.work))
    log(f"  tracks   {args.work}/clips/*/tracks.csv")
    log(f"  tiles    {args.work}/data/tiles.npz")
    log(f"  model    {args.work}/runs/streak/best.pt")
    log(f"  report   {args.work}/runs/streak/report.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted -- completed stages are saved and will be skipped on re-run")
        raise SystemExit(130)
