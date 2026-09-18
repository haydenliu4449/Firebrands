#!/usr/bin/env python3
"""Stage 1 CLI: clip -> auto-labels + contact sheets, with no training involved.

    python run_pipeline.py clips/eaton_01.mp4 --out work/eaton_01

Produces, in the output directory:
    audit.json          pre-flight checks -- READ THIS FIRST
    tracks.csv          one row per accepted track
    detections.csv      one row per detection
    review/sheet_*.png  contact sheets for verification
    labels/*.txt        YOLO-format boxes, one file per frame
    config.json         exact parameters used
    overlay.mp4         detections drawn on the clip (with --overlay)

Then open the contact sheets, note the IDs that are not firebrands, and run:

    python run_pipeline.py clips/eaton_01.mp4 --out work/eaton_01 \
        --reject 12,47,51,88

which re-exports labels with those tracks removed and writes negatives.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from firebrand import Config, frames as F, detect as D, track as T, review as R
from firebrand import gcsio


class Progress:
    """Frame counter with a rate and an ETA, refreshed on a wall-clock timer.

    The previous version printed every 100th frame with a bare \r and no flush,
    so at 4K -- where a frame takes about a second -- it showed "frame 0" and
    then nothing for minutes. That is indistinguishable from a hang, and it is
    a reporting bug rather than a performance one.
    """

    def __init__(self, total, every_s=0.5, label="  "):
        self.total, self.every, self.label = total, every_s, label
        self.t0 = self.last = time.time()

    def __call__(self, k):
        now = time.time()
        if now - self.last < self.every and k + 1 < self.total:
            return
        self.last = now
        el = now - self.t0
        rate = (k + 1) / max(el, 1e-6)
        eta = (self.total - k - 1) / max(rate, 1e-6)
        bar_n = int(28 * (k + 1) / max(self.total, 1))
        bar = "#" * bar_n + "." * (28 - bar_n)
        sys.stdout.write(f"\r{self.label}[{bar}] {k+1}/{self.total}  "
                         f"{rate:.1f} fps  eta {eta/60:4.1f} min   ")
        sys.stdout.flush()

    def done(self):
        sys.stdout.write("\n"); sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="config.json to load (else defaults)")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--exposure", type=float, default=None,
                    help="exposure time in seconds, if you know it")
    ap.add_argument("--mask", help="mask JSON from make_mask.py (strongly recommended)")
    ap.add_argument("--no-mask", action="store_true",
                    help="run completely unmasked, not even the burned-in overlays")
    ap.add_argument("--reject", default="", help="comma-separated track IDs to drop")
    ap.add_argument("--overlay", action="store_true", help="write overlay.mp4")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--check-mask", action="store_true",
                    help="report what the mask costs in accepted tracks, then exit")
    args = ap.parse_args()

    # gs:// output: build everything on local disk, then mirror up once at the
    # end. Writing 200 small files to object storage one at a time is slow and,
    # if the run dies halfway, leaves a half-populated prefix that looks like a
    # completed run.
    gs_out = args.out if gcsio.is_gcs(args.out) else None
    if gs_out:
        out = Path(tempfile.mkdtemp(prefix="firebrand_out_"))
        print(f"staging locally in {out}, will upload to {gs_out}")
    else:
        out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(args.config) if args.config else Config()
    if args.fps:
        cfg.track.fps = args.fps
    if args.exposure:
        cfg.track.exposure_s = args.exposure

    # ---- load -----------------------------------------------------------
    src = F.VideoSource(args.clip)
    all_frames = []
    for i, f in enumerate(src):
        if args.max_frames and i >= args.max_frames:
            break
        all_frames.append(f)
    if not all_frames:
        raise SystemExit("no frames decoded")
    H, W = all_frames[0].shape[:2]
    print(f"{len(all_frames)} frames  {W}x{H}  container fps={src.fps}")

    if not args.fps and src.fps:
        cfg.track.fps = float(src.fps)

    # scale the top-hat kernel with resolution (tuned at 1920 wide)
    cfg.detect.tophat_ksize = max(5, int(round(15 * W / 1920)) | 1)

    # Large frames: recompute the background median less often. It is ~490 ms
    # per frame at 4K and the background is quasi-static, so this is the single
    # biggest saving available and it costs nothing measurable on the benchmark.
    if cfg.detect.bg_refresh == 1 and W * H > 2_500_000:
        cfg.detect.bg_refresh = 4
        print(f"  large frame ({W}x{H}): background refresh every "
              f"{cfg.detect.bg_refresh} frames, stabilisation at 1/"
              f"{max(1, round(W/1920))} resolution")

    # ---- audit ----------------------------------------------------------
    audit = F.audit(F.ArraySource(all_frames), Path(args.clip).name)
    json.dump(audit, open(out / "audit.json", "w"), indent=2, default=str)
    print("\naudit:")
    print(f"  best channel        {audit.get('best_channel')}")
    print(f"  blue separability   {audit['separability'].get('blue', 0):.1f} sigma")
    print(f"  duplicate frames    {audit['duplicate_fraction']:.1%}")
    for w in audit["warnings"]:
        print(f"  !! {w}")
    if audit["ir_night_mode"]:
        cfg.detect.channel = "gray"

    # ---- mask -----------------------------------------------------------
    if args.no_mask:
        roi = None
        print("\n  running unmasked")
    elif args.mask:
        spec = F.load_mask_spec(args.mask)
        roi = F.mask_from_spec((H, W), spec)
        print(f"\n  mask {args.mask}: {(roi == 0).mean():.1%} of the frame excluded")
    else:
        # Only the burned-in overlays, which are certain. Reflections are not
        # guessed at -- see make_mask.py.
        roi = F.make_roi_mask((H, W), **F.LEFT_FRONT_OVERLAYS)
        print(f"\n  mask: burned-in overlays only ({(roi == 0).mean():.1%} excluded)")
        print("  !! No reflection mask. If this camera sees glass or painted metal,")
        print("     run `python make_mask.py <clip> --out masks/<name>.json` first --")
        print("     reflections are mirrored duplicate firebrands and they will")
        print("     double-count flux and create impossible trajectories.")
    if roi is not None:
        cv2.imwrite(str(out / "roi_mask.png"), F.overlay_mask(all_frames[0], roi))

    # ---- what is this mask costing? --------------------------------------
    if args.check_mask:
        if roi is None:
            raise SystemExit("--check-mask needs a mask; drop --no-mask")
        print("\nmask cost (percent of frame is not the number that matters):")
        R.print_mask_cost(R.mask_cost(all_frames, roi, cfg,
                                      ground_plane_poly=F.DRIVEWAY_LEFT_FRONT))
        return

    # ---- detect + track -------------------------------------------------
    print("\ndetecting...")
    prog = Progress(len(all_frames))
    dets = D.detect_source(F.ArraySource(all_frames), cfg.detect, roi, progress=prog)
    prog.done()
    n_det = sum(len(v) for v in dets.values())
    print(f"  {n_det} candidates over {len(dets)} frames "
          f"({n_det/max(len(dets),1):.1f}/frame)")

    ratio, diag = T.estimate_step_ratio(dets, cfg.track, return_diagnostics=True)
    cfg.track.step_ratio = ratio
    print(f"  step_ratio {ratio:.2f}  (from {diag['n_pairs']} pairs, "
          f"sharpness {diag['sharpness']:.1f})")
    if diag["sharpness"] < 2:
        print("  !! low sharpness: the ratio estimate is unreliable. "
              "Measure the camera exposure time and pass --exposure.")

    tracks = T.link(dets, cfg.track, step_ratio=ratio)
    acc = T.accept(tracks, cfg.track)
    rej = T.rejected(tracks, acc)
    print(f"  {len(tracks)} tracks -> {len(acc)} accepted, {len(rej)} rejected")

    # ---- manual rejections ----------------------------------------------
    reject_ids = [int(x) for x in args.reject.split(",") if x.strip()]
    if reject_ids:
        acc, dropped = R.apply_rejections(acc, reject_ids)
        rej = rej + dropped
        R.save_review(out / "review.json", acc, rej,
                      dict(clip=args.clip, n_rejected_manual=len(reject_ids)))
        print(f"  manual review: dropped {len(dropped)}, kept {len(acc)}")

    # ---- outputs --------------------------------------------------------
    rows = T.summarize(acc, cfg.track)
    pd.DataFrame(rows).to_csv(out / "tracks.csv", index=False)
    pd.DataFrame([{**d.as_dict(), "track_id": t.id} for t in acc for d in t.dets]) \
        .to_csv(out / "detections.csv", index=False)
    if rej:
        pd.DataFrame(T.summarize(rej, cfg.track)).to_csv(out / "negatives.csv", index=False)

    lab_dir = out / "labels"; lab_dir.mkdir(exist_ok=True)
    for fi, boxes in T.to_yolo_labels(acc, (H, W)).items():
        with open(lab_dir / f"{fi:06d}.txt", "w") as fh:
            for c, xc, yc, bw, bh in boxes:
                fh.write(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    sheets = R.make_contact_sheets(acc, all_frames, out / "review")
    cv2.imwrite(str(out / "fp_heatmap.png"),
                R.fp_heatmap(rej, acc, all_frames[len(all_frames) // 2]))
    cfg.save(out / "config.json")

    if args.overlay:
        vw = cv2.VideoWriter(str(out / "overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), cfg.track.fps, (W, H))
        by_frame = {}
        for t in acc:
            for d in t.dets:
                by_frame.setdefault(d.frame, []).append(d)
        for i, f in enumerate(all_frames):
            vw.write(D.draw(f, by_frame.get(i, [])))
        vw.release()

    if rows:
        res = np.array([r["ballistic_rms_px"] for r in rows])
        print(f"\n  ballistic fit residual: median {np.median(res):.2f} px, "
              f"p90 {np.percentile(res, 90):.2f} px")
        print("    (tracks with a large residual are usually ID switches -- "
              "sort tracks.csv by this column and check the worst ones)")

    if gs_out:
        n_up = gcsio.upload_dir(out, gs_out)
        print(f"\nuploaded {n_up} files to {gs_out}/")
        print(f"  browse:   gcloud storage ls {gs_out}/")
        print(f"  fetch:    gcloud storage cp -r {gs_out}/review .")
        out = gs_out

    print(f"\nwrote {out}/  --  {len(sheets)} contact sheets to review")
    print("Open them, note the IDs that are NOT firebrands, then re-run with "
          "--reject a,b,c")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, IOError) as e:
        # A path problem is a user mistake, not a crash. Print the diagnosis,
        # not a stack trace pointing into library code.
        import sys
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1)
