"""Metrics that tell the truth at 7 pixels.

The default object-detection metric will mislead you here. mAP@0.5 requires
IoU >= 0.5; for a 7 px object a one-pixel centroid error already drops IoU below
that. You would be tuning against quantisation noise, and a genuinely good
detector would score as a failure.

Everything below matches on centre distance instead, and scores tracks rather
than frames -- because a tracker that splits one ember into four tracks ruins
your saltation statistics while looking fine on any per-frame metric.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def filter_gt(gt_by_frame, min_L=3.0, min_peak_srgb=25.0, roi_mask=None):
    """Drop ground truth that is not physically detectable, before scoring.

    A synthetic particle at the apex of a hop can have a sub-pixel streak, and
    a cooling one can sit below the noise floor. Counting those as misses
    measures the sensor, not the pipeline, and will send you tuning thresholds
    down into the noise chasing recall you cannot have.

    Report both filtered and unfiltered numbers -- the gap between them is your
    detectability limit, which is itself worth knowing.
    """
    from .synth import srgb_to_linear
    lin_floor = float(srgb_to_linear(min_peak_srgb))
    out = []
    for gs in gt_by_frame:
        keep = []
        for g in gs:
            if g.get("L", 0) < min_L:
                continue
            if g.get("peak_lin", 1.0) < lin_floor:
                continue
            if roi_mask is not None:
                yi = int(np.clip(g["y"], 0, roi_mask.shape[0] - 1))
                xi = int(np.clip(g["x"], 0, roi_mask.shape[1] - 1))
                if roi_mask[yi, xi] == 0:
                    continue
            keep.append(g)
        out.append(keep)
    return out


# ---------------------------------------------------------------------------
# detection-level
# ---------------------------------------------------------------------------

def match_frame(pred_xy, gt_xy, max_dist=5.0):
    """Optimal one-to-one matching within a frame. Returns (pairs, n_tp)."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return [], 0
    P, G = np.asarray(pred_xy, float), np.asarray(gt_xy, float)
    D = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    C = np.where(D <= max_dist, D, 1e6)
    ri, ci = linear_sum_assignment(C)
    pairs = [(int(i), int(j)) for i, j in zip(ri, ci) if C[i, j] < 1e5]
    return pairs, len(pairs)


def detection_metrics(pred_by_frame, gt_by_frame, max_dist=5.0,
                      gt_for_precision=None):
    """Precision / recall / F1 on centre distance.

    `pred_by_frame`: {frame -> [Detection]} or {frame -> [(x, y), ...]}
    `gt_by_frame`:   list or dict of [{x, y, ...}] per frame

    `gt_for_precision`: if you scored recall against a *detectable* subset of
    ground truth (see `filter_gt`), pass the unfiltered ground truth here.
    Otherwise a detection that correctly found a marginal object you excluded
    from the recall denominator gets counted as a false positive, and precision
    is understated -- badly, when the marginal population is large.
    """
    tp = fp = fn = 0
    dists = []
    frames = sorted(set(pred_by_frame) | set(
        range(len(gt_by_frame)) if isinstance(gt_by_frame, list) else gt_by_frame))
    for f in frames:
        pd = pred_by_frame.get(f, [])
        gd = gt_by_frame[f] if isinstance(gt_by_frame, list) and f < len(gt_by_frame) \
            else (gt_by_frame.get(f, []) if isinstance(gt_by_frame, dict) else [])
        p_xy = [(d.x, d.y) if hasattr(d, "x") else (d[0], d[1]) for d in pd]
        g_xy = [(g["x"], g["y"]) if isinstance(g, dict) else (g[0], g[1]) for g in gd]
        pairs, n = match_frame(p_xy, g_xy, max_dist)
        for i, j in pairs:
            dists.append(float(np.hypot(p_xy[i][0] - g_xy[j][0], p_xy[i][1] - g_xy[j][1])))
        tp += n
        fn += len(g_xy) - n

        if gt_for_precision is None:
            fp += len(p_xy) - n
        else:
            gp = gt_for_precision[f] if isinstance(gt_for_precision, list) and f < len(gt_for_precision) \
                else (gt_for_precision.get(f, []) if isinstance(gt_for_precision, dict) else [])
            gp_xy = [(g["x"], g["y"]) if isinstance(g, dict) else (g[0], g[1]) for g in gp]
            _, n_all = match_frame(p_xy, gp_xy, max_dist)
            fp += len(p_xy) - n_all

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                tp=tp, fp=fp, fn=fn,
                mean_loc_err_px=float(np.mean(dists)) if dists else float("nan"))


# ---------------------------------------------------------------------------
# track-level
# ---------------------------------------------------------------------------

def track_metrics(tracks, gt_by_frame, max_dist=5.0, min_overlap=0.5):
    """Track recall, purity, fragmentation and ID switches.

    - track_recall  : GT tracks with >= min_overlap of their detections covered
                      by a single predicted track. This is the number that
                      decides whether your saltation statistics are usable.
    - fragmentation : predicted tracks per covered GT track. 1.0 is perfect;
                      2.0 means every ember is being cut in half.
    - id_switches   : predicted tracks that change which GT track they follow.
    """
    # frame -> [(gt_track_id, x, y)]
    gt_frames = {}
    gt_len = {}
    for f, gs in enumerate(gt_by_frame):
        gt_frames[f] = [(g["track_id"], g["x"], g["y"]) for g in gs]
        for g in gs:
            gt_len[g["track_id"]] = gt_len.get(g["track_id"], 0) + 1

    hits = {}         # gt_id -> {pred_id: count}
    switches = 0
    for t in tracks:
        seq = []
        for d in t.dets:
            cands = gt_frames.get(d.frame, [])
            if not cands:
                seq.append(None); continue
            arr = np.array([[c[1], c[2]] for c in cands], float)
            dd = np.linalg.norm(arr - np.array([d.x, d.y]), axis=1)
            k = int(np.argmin(dd))
            if dd[k] <= max_dist:
                gid = cands[k][0]
                seq.append(gid)
                hits.setdefault(gid, {})
                hits[gid][t.id] = hits[gid].get(t.id, 0) + 1
            else:
                seq.append(None)
        real = [s for s in seq if s is not None]
        switches += sum(1 for a, b in zip(real, real[1:]) if a != b)

    covered, frags = 0, []
    for gid, n_total in gt_len.items():
        if n_total < 3:
            continue                      # too short to be a fair target
        h = hits.get(gid, {})
        if not h:
            continue
        best = max(h.values())
        if best / n_total >= min_overlap:
            covered += 1
            frags.append(len([v for v in h.values() if v >= 2]) or 1)

    eligible = sum(1 for n in gt_len.values() if n >= 3)
    return dict(
        gt_tracks=eligible,
        pred_tracks=len(tracks),
        track_recall=covered / eligible if eligible else 0.0,
        fragmentation=float(np.mean(frags)) if frags else float("nan"),
        id_switches=switches,
    )


def report(pred_by_frame, tracks, gt_by_frame, max_dist=5.0):
    d = detection_metrics(pred_by_frame, gt_by_frame, max_dist)
    t = track_metrics(tracks, gt_by_frame, max_dist)
    return {**{f"det_{k}": v for k, v in d.items()},
            **{f"trk_{k}": v for k, v in t.items()}}


def print_report(rep, title="evaluation"):
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in rep.items():
        print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
