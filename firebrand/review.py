"""Contact-sheet review: turn hours of annotation into minutes of verification.

Drawing a box takes about 4 seconds. Saying yes/no to something already drawn
takes about one. And a track of six detections is *one* decision instead of
six. That is roughly a 25x multiplier before you write any model code, which is
what makes a real supervised set reachable in an afternoon.

Anything more elaborate -- CVAT, Label Studio -- costs more setup time than the
labelling itself at this scale, and neither of them reviews *tracks*.

Workflow:
    sheets = make_contact_sheets(tracks, frames, "review/")
    # open review/sheet_000.png, note the IDs that are not firebrands
    keep, reject = apply_rejections(tracks, [12, 47, 51, ...])
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def track_strip(track, frames, crop=36, max_cells=10, scale=2):
    """One track as a horizontal strip of crops, in time order.

    Crops are centred on the detection, so a real firebrand shows a bright
    streak sitting still in the middle of every cell while the background slides
    past. A false positive looks like static texture in every cell. The
    difference is obvious at a glance, which is the whole point.
    """
    step = max(1, int(np.ceil(len(track.dets) / max_cells)))
    dets = track.dets[::step][:max_cells]
    h = crop * scale
    cells = []
    for d in dets:
        f = frames[d.frame]
        x0, y0 = int(d.x - crop // 2), int(d.y - crop // 2)
        x0 = int(np.clip(x0, 0, f.shape[1] - crop))
        y0 = int(np.clip(y0, 0, f.shape[0] - crop))
        c = f[y0:y0 + crop, x0:x0 + crop].copy()
        if c.shape[:2] != (crop, crop):
            c = cv2.copyMakeBorder(c, 0, crop - c.shape[0], 0, crop - c.shape[1],
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        c = cv2.resize(c, (h, h), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(c, (0, 0), (h - 1, h - 1), (60, 60, 60), 1)
        cells.append(c)
    while len(cells) < max_cells:
        cells.append(np.zeros((h, h, 3), np.uint8))
    return np.hstack(cells)


def make_contact_sheets(tracks, frames, out_dir, per_sheet=24, crop=36,
                        max_cells=10, scale=2, label_w=110):
    """Render every track as a labelled strip, `per_sheet` strips per PNG.

    Returns the list of written paths. A sheet of 24 takes ~30 s to scan, so
    800 tracks is about half an hour.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    order = sorted(tracks, key=lambda t: -len(t.dets))

    for s in range(0, len(order), per_sheet):
        chunk = order[s:s + per_sheet]
        rows = []
        for t in chunk:
            strip = track_strip(t, frames, crop, max_cells, scale)
            pad = np.zeros((strip.shape[0], label_w, 3), np.uint8)
            cv2.putText(pad, f"#{t.id}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(pad, f"n={len(t.dets)}", (6, 46), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (170, 170, 170), 1, cv2.LINE_AA)
            cv2.putText(pad, f"f{t.frames[0]}", (6, 64), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (170, 170, 170), 1, cv2.LINE_AA)
            rows.append(np.hstack([pad, strip]))
        sheet = np.vstack(rows)
        head = np.zeros((34, sheet.shape[1], 3), np.uint8)
        cv2.putText(head, f"sheet {s // per_sheet}   "
                          f"{len(chunk)} tracks, longest first   "
                          f"note the IDs that are NOT firebrands",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        p = out_dir / f"sheet_{s // per_sheet:03d}.png"
        cv2.imwrite(str(p), np.vstack([head, sheet]))
        paths.append(p)
    return paths


def fp_heatmap(rejected, accepted, frame, sigma=25, alpha=0.55):
    """Where the false positives live, drawn on the frame.

    Run the pipeline once *unmasked*, then look at this. Rejected candidates
    cluster on whatever in the scene generates streak-shaped signal that does
    not move like an ember -- wind-blown vegetation, specular metal, a swaying
    branch. Those clusters are your mask, measured rather than guessed.

    Red = density of rejected candidates (mask these).
    Green dots = accepted tracks (do NOT mask these -- if a region has both,
    leave it in and let the 3-frame rule do the work).

    This is the honest way round: draw polygons around what the data says is
    noisy, not around what looks suspicious in a still frame.
    """
    H, W = frame.shape[:2]
    acc = np.zeros((H, W), np.float32)
    for t in rejected:
        for d in t.dets:
            y, x = int(np.clip(d.y, 0, H - 1)), int(np.clip(d.x, 0, W - 1))
            acc[y, x] += 1.0
    if acc.max() > 0:
        acc = cv2.GaussianBlur(acc, (0, 0), sigma)
        acc /= acc.max()

    heat = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    strength = np.clip(acc * 2.2, 0, 1)[..., None] * alpha
    vis = (frame * (1 - strength) + heat * strength).astype(np.uint8)

    for t in accepted:
        for d in t.dets:
            cv2.circle(vis, (int(d.x), int(d.y)), 3, (120, 255, 120), -1, cv2.LINE_AA)

    cv2.rectangle(vis, (0, 0), (W, 34), (0, 0, 0), -1)
    cv2.putText(vis, f"red = {sum(len(t.dets) for t in rejected)} rejected candidates (mask the hot zones)   "
                     f"green = {sum(len(t.dets) for t in accepted)} accepted detections (keep these regions)",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    return vis


def mask_cost(frames, mask, cfg, ground_plane_poly=None):
    """What a mask actually costs you. Run this before committing to one.

    'What percent of the frame should I mask?' is the wrong question -- the
    fraction of *pixels* removed says nothing, because the regions worth cutting
    are rarely the regions where the measurement happens. Two numbers matter:

      tracks_lost_frac  -- accepted tracks the mask removes. This is real data,
                           and it is the cost.
      ground_plane_kept -- fraction of the surface where saltation contacts
                           occur that survives. This is what must stay near 1.0,
                           and it is what a percentage-of-frame target ignores.

    A mask that cuts 44% of the frame while keeping 98% of the driveway is fine.
    One that cuts 15% but clips a corner of the driveway is not, because it
    biases which hops you observe -- and a biased sample of hop lengths is worse
    than a smaller unbiased one.
    """
    from . import detect as D
    from . import track as T
    from .frames import ArraySource

    def run(m):
        dets = D.detect_source(ArraySource(frames), cfg.detect, m)
        acc = T.accept(T.link(dets, cfg.track,
                              step_ratio=T.estimate_step_ratio(dets, cfg.track)),
                       cfg.track)
        return sum(len(v) for v in dets.values()), len(acc)

    n_open, trk_open = run(None)
    n_mask, trk_mask = run(mask)

    rep = dict(
        frame_cut_frac=float((mask == 0).mean()),
        candidates_unmasked=n_open, candidates_masked=n_mask,
        candidates_cut_frac=1 - n_mask / max(n_open, 1),
        tracks_unmasked=trk_open, tracks_masked=trk_mask,
        tracks_lost_frac=1 - trk_mask / max(trk_open, 1),
    )
    if ground_plane_poly is not None:
        H, W = mask.shape[:2]
        gp = np.zeros((H, W), np.uint8)
        pts = np.array([(int(x * W), int(y * H)) for x, y in ground_plane_poly], np.int32)
        cv2.fillPoly(gp, [pts], 255)
        rep["ground_plane_frac"] = float((gp > 0).mean())
        rep["ground_plane_kept"] = float(((gp > 0) & (mask > 0)).sum() / max((gp > 0).sum(), 1))
    return rep


def print_mask_cost(rep):
    print(f"  frame cut            {rep['frame_cut_frac']:.1%}")
    print(f"  candidates           {rep['candidates_unmasked']} -> {rep['candidates_masked']}"
          f"  ({rep['candidates_cut_frac']:.1%} cut)")
    print(f"  ACCEPTED TRACKS      {rep['tracks_unmasked']} -> {rep['tracks_masked']}"
          f"  ({rep['tracks_lost_frac']:.1%} lost)   <- the real cost")
    if "ground_plane_kept" in rep:
        print(f"  ground plane kept    {rep['ground_plane_kept']:.1%}"
              f"   <- keep this above ~95%")
    if rep["tracks_lost_frac"] > 0.25:
        print("  !! This mask is removing a quarter of your real tracks. Check the")
        print("     heatmap -- you are probably masking a region embers fly through.")


def apply_rejections(tracks, rejected_ids):
    """Split tracks into (verified, rejected).

    Keep the rejected list. Every one is a labelled *negative* mined from your
    own footage -- a paver joint, a compression block, a reflection -- and hard
    negatives are worth far more to the model than random background crops,
    because they are exactly the mistakes it would otherwise make.
    """
    rej = set(int(i) for i in rejected_ids)
    keep = [t for t in tracks if t.id not in rej]
    drop = [t for t in tracks if t.id in rej]
    return keep, drop


def save_review(path, verified, rejected, meta=None):
    json.dump({
        "verified_ids": [t.id for t in verified],
        "rejected_ids": [t.id for t in rejected],
        "meta": meta or {},
    }, open(path, "w"), indent=2)


def load_review(path):
    d = json.load(open(path))
    return set(d["verified_ids"]), set(d["rejected_ids"])


def crops_from_tracks(tracks, frames, crop=32, label=1):
    """Extract per-detection crops for training or for fitting the synthetic
    generator. Returns (crops, labels) as arrays."""
    out = []
    for t in tracks:
        for d in t.dets:
            x0 = int(np.clip(d.x - crop // 2, 0, frames[d.frame].shape[1] - crop))
            y0 = int(np.clip(d.y - crop // 2, 0, frames[d.frame].shape[0] - crop))
            c = frames[d.frame][y0:y0 + crop, x0:x0 + crop]
            if c.shape[:2] == (crop, crop):
                out.append(c)
    return np.array(out), np.full(len(out), label, np.int64)
