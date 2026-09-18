#!/usr/bin/env python3
"""End-to-end benchmark on synthetic firebrands composited onto a real frame.

This is the pipeline's regression test AND its ablation study. Run it after any
change to detect.py or track.py -- the numbers below are what the shipped
defaults produce, and a change that moves them should be deliberate.

    python tests/benchmark.py            # defaults
    python tests/benchmark.py --ablate   # also run the ablation table

Why synthetic: real firebrand footage has no ground truth, so there is no
honest way to measure recall on it. Compositing onto a real background frame
keeps the hard part real (pavement texture, compression, glow gradient) while
making the answer exactly known.

What it does NOT prove: performance on real embers, whose brightness and shape
distributions differ from the generator's priors. Treat these numbers as a
lower bound on the mechanics working, not as expected field accuracy.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firebrand import Config, frames as F, detect as D, track as T, synth as S, evaluate as E  # noqa: E402

HERE = Path(__file__).resolve().parent


def make_bench(width=960, n_frames=50, seed=1):
    bg = cv2.imread(str(HERE / "sample_frame.png"))
    if bg is None:
        raise SystemExit("tests/sample_frame.png missing")
    h = int(bg.shape[0] * width / bg.shape[1])
    bg = cv2.resize(bg, (width, h))

    cfg = Config()
    cfg.track.fps = 15.0
    cfg.detect.tophat_ksize = max(5, int(round(15 * width / 1920)) | 1)
    cfg.synth.n_particles_per_frame = (3, 10)
    cfg.synth.speed_px = (8.0, 40.0)
    cfg.synth.exposure_duty = 0.35
    cfg.synth.reencode_h264 = False          # keep the benchmark fast

    frames, gt = S.make_clip(bg, n_frames, cfg.synth, seed=seed,
                             flicker_pct=6.0, shake_px=0.8)
    return cfg, frames, gt


def run(cfg, frames, gt, label, stabilize=True, edge_suppress=None, verbose=True):
    c = Config()
    c.detect.__dict__.update(cfg.detect.__dict__)
    c.track.__dict__.update(cfg.track.__dict__)
    c.detect.stabilize = stabilize
    if edge_suppress is not None:
        c.detect.edge_suppress = edge_suppress

    t0 = time.time()
    dets = D.detect_source(F.ArraySource(frames), c.detect)
    gtf = E.filter_gt(gt, min_L=c.detect.min_length, min_peak_srgb=25.0)

    raw = E.detection_metrics(dets, gtf, 6.0, gt_for_precision=gt)
    ratio = T.estimate_step_ratio(dets, c.track)
    acc = T.accept(T.link(dets, c.track, step_ratio=ratio), c.track)

    acc_by_frame = {}
    for tr in acc:
        for d in tr.dets:
            acc_by_frame.setdefault(d.frame, []).append(d)
    fin = E.detection_metrics(acc_by_frame, gtf, 6.0, gt_for_precision=gt)
    tm = E.track_metrics(acc, gtf, 6.0)
    secs = time.time() - t0

    if verbose:
        print(f"{label:30s} raw P={raw['precision']:.2f} R={raw['recall']:.2f}"
              f" ({raw['tp']+raw['fp']:5d} det) | after tracking P={fin['precision']:.2f}"
              f" R={fin['recall']:.2f} | tracks={len(acc):3d} recall={tm['track_recall']:.2f}"
              f" frag={tm['fragmentation']:.2f} idsw={tm['id_switches']}  [{secs:.0f}s]")
    return dict(label=label, raw=raw, final=fin, track=tm, ratio=ratio, secs=secs)


def channel_sweep(width=960, n_frames=45):
    """Where does the blue channel actually earn its keep?

    Answer: not on bright embers -- background subtraction already handles
    those, and grayscale matches it. Only at the faint end, which is where the
    ground-contact phase of saltation lives.
    """
    bg = cv2.imread(str(HERE / "sample_frame.png"))
    h = int(bg.shape[0] * width / bg.shape[1])
    bg = cv2.resize(bg, (width, h))

    print("\nchannel comparison by ember brightness:\n")
    print(f"  {'peak sRGB':<14}{'blue R':>9}{'gray R':>9}"
          f"{'blue trkR':>12}{'gray trkR':>12}")
    for peak, name in [((70, 255), "bright"), ((35, 90), "medium"), ((20, 55), "faint")]:
        cfg = Config()
        cfg.track.fps = 15.0
        cfg.detect.tophat_ksize = max(5, int(round(15 * width / 1920)) | 1)
        cfg.synth.n_particles_per_frame = (3, 10)
        cfg.synth.speed_px = (8.0, 40.0)
        cfg.synth.exposure_duty = 0.35
        cfg.synth.peak_srgb = peak
        cfg.synth.reencode_h264 = False
        frames, gt = S.make_clip(bg, n_frames, cfg.synth, seed=2,
                                 flicker_pct=6.0, shake_px=0.8)
        gtf = E.filter_gt(gt, min_L=cfg.detect.min_length,
                          min_peak_srgb=peak[0] * 0.9)
        row = {}
        for ch in ("blue", "gray"):
            c = Config()
            c.detect.__dict__.update(cfg.detect.__dict__)
            c.detect.channel = ch
            dets = D.detect_source(F.ArraySource(frames), c.detect)
            m = E.detection_metrics(dets, gtf, 6.0, gt_for_precision=gt)
            acc = T.accept(T.link(dets, c.track,
                                  step_ratio=T.estimate_step_ratio(dets, c.track)),
                           c.track)
            row[ch] = (m["recall"], E.track_metrics(acc, gtf, 6.0)["track_recall"])
        print(f"  {name+' '+str(peak):<14}{row['blue'][0]:>9.2f}{row['gray'][0]:>9.2f}"
              f"{row['blue'][1]:>12.2f}{row['gray'][1]:>12.2f}")
    print("\n  Blue and grayscale are equivalent on bright embers -- background")
    print("  subtraction already removes the orange ambient. The blue channel is")
    print("  the difference between working and not working on faint ones, which")
    print("  is the population that matters: an ember is dimmest while it is on")
    print("  the ground, and ground contact is the saltation event itself.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--channels", action="store_true")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--width", type=int, default=960)
    args = ap.parse_args()

    cfg, frames, gt = make_bench(args.width, args.frames)
    n_gt = sum(len(g) for g in gt)
    n_det = sum(len(g) for g in E.filter_gt(gt, min_L=cfg.detect.min_length,
                                            min_peak_srgb=25.0))
    print(f"benchmark clip: {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}, "
          f"{n_gt} ground-truth streaks ({n_det} above the detectability floor)\n")

    res = [run(cfg, frames, gt, "shipped defaults")]

    if args.ablate:
        print("\nablation -- each row removes one thing:\n")
        run(cfg, frames, gt, "  no stabilisation", stabilize=False)
        run(cfg, frames, gt, "  no edge suppression", edge_suppress=0.0)
        run(cfg, frames, gt, "  neither", stabilize=False, edge_suppress=0.0)

        c2 = Config()
        c2.detect.__dict__.update(cfg.detect.__dict__)
        c2.track.__dict__.update(cfg.track.__dict__)
        c2.detect.channel = "gray"
        run(c2, frames, gt, "  grayscale not blue")

    if args.channels:
        channel_sweep(args.width)

    print("\nWhat to read from this:")
    print("  * precision after tracking should be ~1.0 -- the acceptance rule is")
    print("    what makes auto-labels trustworthy, and it is doing its job.")
    print("  * raw recall > final recall by design: requiring 3 coherent frames")
    print("    costs detections and buys precision. The learned model recovers")
    print("    that recall, which is the reason to train one at all.")
    print("  * id_switches should be 0. Anything above ~2 means the step_ratio")
    print("    estimate is off; check its sharpness or measure the exposure.")
    return res


if __name__ == "__main__":
    main()
