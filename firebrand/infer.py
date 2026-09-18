"""Running the trained model on a clip, and turning its output back into tracks.

The output of the network is a streak probability map. Everything downstream --
connected components, streak fitting, linking, acceptance -- is the same code
the classical detector uses, so the model is a drop-in replacement for the
thresholding stage, not a separate pipeline. That means you can compare them on
identical terms, which is the only way to know whether training helped.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from . import detect as D
from . import track as T
from .config import Config
from .dataset import residual_stacks, tile_positions


@torch.no_grad()
def predict_map(model, stack, tile=256, stride=192, device="cpu", batch=8):
    """Sliding-window inference over a full frame -> probability map.

    Tiles at *native resolution*, never a resized frame. Resizing 1920x1080 to
    640 turns a 7 px streak into 2.3 px, which is below what any standard
    detection head can represent -- it is the third reason the first attempt
    stalled, and it is entirely avoidable.

    Overlapping predictions are averaged with a cosine window so tile seams do
    not create phantom edges that the streak fitter would happily detect.
    """
    C, H, W = stack.shape
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)

    win1 = np.hanning(tile).astype(np.float32)
    win = np.outer(win1, win1) + 1e-3

    pos = tile_positions((H, W), tile, stride)
    for s in range(0, len(pos), batch):
        chunk = pos[s:s + batch]
        xb = np.stack([stack[:, y:y + tile, x:x + tile] for (y, x) in chunk])
        xb = torch.from_numpy(np.clip(xb.astype(np.float32), 0, 255) / 64.0).to(device)
        pr = torch.sigmoid(model(xb)).cpu().numpy()[:, 0]
        for (y, x), p in zip(chunk, pr):
            acc[y:y + tile, x:x + tile] += p * win
            wsum[y:y + tile, x:x + tile] += win
    return acc / np.maximum(wsum, 1e-6)


def detections_from_map(prob, frame_index, cfg: Config, thresh=0.5, roi_mask=None):
    """Probability map -> Detection objects, using the same streak fitting as
    the classical path so the two are directly comparable."""
    p8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    if roi_mask is not None:
        p8 = cv2.bitwise_and(p8, p8, mask=roi_mask)

    dcfg = cfg.detect
    bw = (p8 > int(thresh * 255)).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)

    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (dcfg.min_area <= area <= dcfg.max_area):
            continue
        ys, xs = np.where(lab == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        if len(pts) >= 5:
            (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
            L, W = max(w, h), max(min(w, h), 1e-3)
            theta = np.deg2rad(ang if w >= h else ang + 90.0)
        else:
            cx, cy = float(xs.mean()), float(ys.mean())
            L, W, theta = float(max(stats[i, cv2.CC_STAT_WIDTH],
                                    stats[i, cv2.CC_STAT_HEIGHT])), 1.0, 0.0
        if L < dcfg.min_length or L > dcfg.max_length:
            continue
        v = p8[ys, xs].astype(np.float32)
        out.append(D.Detection(frame=frame_index, x=float(cx), y=float(cy),
                               L=float(L), W=float(W), theta=float(theta),
                               peak=float(v.max()), area=area, flux=float(v.sum())))
    return out


def run_clip(model, source, cfg: Config, roi_mask=None, thresh=0.5,
             device="cpu", progress=None):
    """Full model inference over a clip -> (detections_by_frame, tracks).

    Returns accepted tracks, using the same link/accept as the classical path.
    Note that acceptance still applies: even a good model produces isolated
    false positives, and the 3-frame coherence rule is the cheapest way to
    remove them.
    """
    dets = {}
    for k, (i, stack) in enumerate(residual_stacks(source, cfg.detect, roi_mask,
                                                   cfg.train.n_frames_stack)):
        prob = predict_map(model, stack, cfg.train.tile, cfg.train.stride, device)
        dets[i] = detections_from_map(prob, i, cfg, thresh, roi_mask)
        if progress and k % 25 == 0:
            progress(k)

    ratio = T.estimate_step_ratio(dets, cfg.track)
    tracks = T.link(dets, cfg.track, step_ratio=ratio)
    return dets, T.accept(tracks, cfg.track)


def sweep_threshold(model, source, cfg, gt_by_frame, roi_mask=None,
                    thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), device="cpu"):
    """Pick the operating threshold on held-out data, not by eye.

    Cache the probability maps once and re-threshold, rather than re-running the
    network per threshold -- otherwise this is the slowest thing in the project.
    """
    from .evaluate import detection_metrics

    maps = {}
    for i, stack in residual_stacks(source, cfg.detect, roi_mask,
                                    cfg.train.n_frames_stack):
        maps[i] = predict_map(model, stack, cfg.train.tile, cfg.train.stride, device)

    rows = []
    for th in thresholds:
        dets = {i: detections_from_map(p, i, cfg, th, roi_mask)
                for i, p in maps.items()}
        m = detection_metrics(dets, gt_by_frame, max_dist=5.0)
        rows.append(dict(threshold=th, **m))
    return rows
