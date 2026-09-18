"""Turning clips into training tiles.

The tile builder is deliberately free of torch so it can run anywhere and be
tested on its own; only the Dataset wrapper at the bottom imports torch.

Input convention throughout: a sample is `n_frames_stack` consecutive
*residual* frames (background-subtracted, see detect.ResidualEngine) stacked as
channels, and the target is a binary streak mask for the centre frame.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import detect as D
from . import synth as S
from .config import DetectConfig, TrainConfig


# ---------------------------------------------------------------------------
# residual stacks
# ---------------------------------------------------------------------------

def residual_stacks(source, dcfg: DetectConfig, roi_mask=None, n_stack=3):
    """Yield (centre_frame_index, stack) where stack is (n_stack, H, W) uint8.

    n_stack must be odd; the centre channel is the frame being labelled.
    """
    assert n_stack % 2 == 1, "n_frames_stack must be odd"
    half = n_stack // 2
    buf, idx = [], []
    for i, res, _edge in D.residual_source(source, dcfg, roi_mask):
        buf.append(res); idx.append(i)
        if len(buf) > n_stack:
            buf.pop(0); idx.pop(0)
        if len(buf) == n_stack:
            yield idx[half], np.stack(buf, 0)


# ---------------------------------------------------------------------------
# tiling
# ---------------------------------------------------------------------------

def tile_positions(shape, tile, stride):
    H, W = shape[:2]
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if ys and ys[-1] + tile < H:
        ys.append(H - tile)
    if xs and xs[-1] + tile < W:
        xs.append(W - tile)
    return [(y, x) for y in ys for x in xs]


def masks_from_detections(dets, shape, width=3):
    """Draw a streak mask from Detection objects (for real, verified tracks).

    Uses each detection's fitted length and orientation, so the mask is the
    streak itself rather than a blob at its centre -- which is what lets the
    network learn orientation, and what makes its output directly usable for
    velocity.
    """
    m = np.zeros(shape[:2], np.float32)
    for d in dets:
        dx, dy = np.cos(d.theta) * d.L / 2, np.sin(d.theta) * d.L / 2
        cv2.line(m, (int(round(d.x - dx)), int(round(d.y - dy))),
                 (int(round(d.x + dx)), int(round(d.y + dy))),
                 1.0, thickness=width, lineType=cv2.LINE_AA)
    return np.clip(m, 0, 1)


def build_tiles(source, masks_by_frame, dcfg: DetectConfig, tcfg: TrainConfig,
                roi_mask=None, keep_empty_frac=0.15, rng=None, max_tiles=None,
                max_per_frame=48):
    """Residual stacks + masks -> training tiles.

    `keep_empty_frac` controls how many all-negative tiles survive. Keeping
    *some* is essential -- they are what teaches the model to reject clean
    pavement -- but keeping all of them makes 99% of the set empty and the
    model converges to predicting nothing. 0.15 is a reasonable balance;
    hard negatives (from rejected tracks) should be added on top rather than
    relied on from this stream.
    """
    rng = rng or np.random.default_rng(tcfg.seed)
    X, Y = [], []
    for i, stack in residual_stacks(source, dcfg, roi_mask, tcfg.n_frames_stack):
        m = masks_by_frame.get(i)
        if m is None:
            continue
        # A 4K frame has 220 tile positions against 60 at 1080p, and the extra
        # ones are overwhelmingly empty background. Shuffling and capping keeps
        # the set from being dominated by one resolution's geometry.
        pos = tile_positions(stack.shape[1:], tcfg.tile, tcfg.stride)
        if max_per_frame and len(pos) > max_per_frame:
            rng.shuffle(pos)
        kept_here = 0
        for (y, x) in pos:
            if max_per_frame and kept_here >= max_per_frame:
                break
            mt = m[y:y + tcfg.tile, x:x + tcfg.tile]
            if mt.max() < 0.2 and rng.random() > keep_empty_frac:
                continue
            X.append(stack[:, y:y + tcfg.tile, x:x + tcfg.tile])
            Y.append(mt)
            kept_here += 1
            if max_tiles and len(X) >= max_tiles:
                return np.array(X, np.uint8), np.array(Y, np.float32)
    if not X:
        return np.zeros((0, tcfg.n_frames_stack, tcfg.tile, tcfg.tile), np.uint8), \
               np.zeros((0, tcfg.tile, tcfg.tile), np.float32)
    return np.array(X, np.uint8), np.array(Y, np.float32)


def build_from_synthetic(background, n_frames, cfg, seed=0, max_tiles=None,
                         roi_mask=None, chunk=40, progress=True):
    """Render synthetic clips and tile them, in bounded memory.

    GENERATED IN CHUNKS ON PURPOSE. The obvious implementation renders all
    n_frames, then re-encodes them, then tiles them -- and at 3840x2160 one
    frame is 25 MB, so 200 frames is 5 GB live, and the re-encode briefly holds
    both copies at 10 GB. That does not fit in a 16 GB VM alongside everything
    else, and the swapping turns eleven minutes of real work into over an hour.
    Chunking caps the frame buffer at `chunk` frames (40 x 25 MB = 1 GB) and
    keeps only the tiles, which are small.

    Each chunk is an independent short clip, so particles do not continue
    across a chunk boundary. That costs nothing here: tiles are per-frame, and
    the 3-frame residual stack never spans a boundary.

    Returns (X, Y, sample_frames, sample_gt) -- the samples are from the FIRST
    chunk only, for eyeballing; the full clip is never held in memory.
    """
    from .frames import ArraySource

    rng_seed = seed
    Xs, Ys = [], []
    sample_frames, sample_gt = None, None
    n_done = 0
    total_tiles = 0

    # residual_stacks needs median_window frames of context before it yields
    # anything, so a chunk smaller than that produces nothing at all.
    min_chunk = cfg.detect.median_window + cfg.train.n_frames_stack + 2
    chunk = max(int(chunk), min_chunk)

    while n_done < n_frames:
        n = min(chunk, n_frames - n_done)
        if n < min_chunk and n_done > 0:
            break                                  # trailing stub yields nothing
        frames, gt = S.make_clip(background, n, cfg.synth, seed=rng_seed,
                                 roi_mask=roi_mask)
        if cfg.synth.reencode_h264:
            frames = S.reencode(frames, fps=int(cfg.track.fps))

        shape = frames[0].shape
        masks = {i: S.gt_masks(g, shape) for i, g in enumerate(gt)}
        x, y = build_tiles(ArraySource(frames), masks, cfg.detect, cfg.train,
                           roi_mask=roi_mask,
                           rng=np.random.default_rng(rng_seed),
                           max_tiles=None if max_tiles is None
                           else max(max_tiles - total_tiles, 0))
        if len(x):
            Xs.append(x); Ys.append(y)
            total_tiles += len(x)

        if sample_frames is None:
            sample_frames, sample_gt = frames[:6], gt[:6]

        n_done += n
        rng_seed += 1
        if progress:
            print(f"    synth {n_done}/{n_frames} frames, {total_tiles} tiles",
                  flush=True)
        del frames, masks                          # let the chunk go before the next
        if max_tiles and total_tiles >= max_tiles:
            break

    if not Xs:
        empty_x = np.zeros((0, cfg.train.n_frames_stack, cfg.train.tile,
                            cfg.train.tile), np.uint8)
        return empty_x, np.zeros((0, cfg.train.tile, cfg.train.tile), np.float32), \
            sample_frames or [], sample_gt or []

    return (np.concatenate(Xs), np.concatenate(Ys),
            sample_frames, sample_gt)


def save_tiles(path, X, Y):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, Y=(Y * 255).astype(np.uint8))


def load_tiles(path):
    d = np.load(path)
    return d["X"], d["Y"].astype(np.float32) / 255.0


def suggested_pos_weight(Y, thresh=0.2):
    """A pos_weight matched to the actual class imbalance in your tiles.

    Streak pixels come out around 1 in 1500 on this footage. Fully inverting
    that (pos_weight = 1500) makes the model fire everywhere; ignoring it
    (pos_weight = 1) makes it predict nothing, and the loss curve will look
    excellent while it does. The geometric mean is the usual compromise and is
    what this returns -- roughly 40 for these tiles.

    Set TrainConfig.pos_weight from this rather than from the default, because
    the imbalance depends on your ember density, which varies a lot between
    clips.
    """
    p = float((Y > thresh).mean())
    if p <= 0:
        return 1.0
    return float(np.clip((1.0 / p) ** 0.5, 1.0, 200.0))


def tile_stats(X, Y):
    pos = float((Y > 0.2).mean())
    return dict(n_tiles=len(X), tile=X.shape[-1], channels=X.shape[1],
                positive_pixel_frac=pos,
                tiles_with_signal=float((Y.reshape(len(Y), -1).max(1) > 0.2).mean()),
                residual_mean=float(X.mean()), residual_p99=float(np.percentile(X, 99)))


# ---------------------------------------------------------------------------
# torch Dataset
# ---------------------------------------------------------------------------

class StreakTiles:
    """torch Dataset over tiles produced above.

    Augmentation is limited to transforms that preserve the physics: flips and
    90-degree rotations (a firebrand can travel any direction), mild gain
    jitter (exposure varies), and additive noise. Deliberately NOT included:
    scale/zoom augmentation, which changes streak length -- the quantity the
    model is meant to measure -- and mosaic-style augmentation, which
    downsamples and erases objects this small.
    """

    def __init__(self, X, Y, train=True, gain_jitter=0.25, noise=2.0, seed=0):
        import torch  # noqa: F401  (import here so tile building stays torch-free)
        self.X, self.Y = X, Y
        self.train = train
        self.gain_jitter = gain_jitter
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        import torch
        x = self.X[i].astype(np.float32)
        y = self.Y[i].astype(np.float32)[None]

        if self.train:
            k = int(self.rng.integers(4))
            if k:
                x = np.rot90(x, k, axes=(1, 2)).copy()
                y = np.rot90(y, k, axes=(1, 2)).copy()
            if self.rng.random() < 0.5:
                x, y = x[:, :, ::-1].copy(), y[:, :, ::-1].copy()
            if self.rng.random() < 0.5:
                x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
            if self.gain_jitter:
                x = x * (1.0 + self.rng.normal(0, self.gain_jitter))
            if self.noise:
                x = x + self.rng.normal(0, self.noise, x.shape)

        x = np.clip(x, 0, 255) / 64.0        # residuals are small; keep O(1)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)
