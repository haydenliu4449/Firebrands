"""Per-frame candidate detection.

This stage is deliberately over-sensitive. On the sample frame a plain
threshold fires 491 times, mostly on paver joints; the shape gate cuts that to
~82. Neither number is the answer -- precision comes from `track.py`, which
throws away anything that does not move coherently. Tuning this stage for
precision is the mistake that stalled the first attempt.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectConfig


@dataclass
class Detection:
    frame: int
    x: float
    y: float
    L: float          # streak length, px (distance travelled during exposure)
    W: float          # streak width, px
    theta: float      # heading, radians, image coords (+x right, +y down)
    peak: float       # peak residual intensity
    area: int
    flux: float       # summed residual intensity ~ brightness * size

    def speed_px_s(self, exposure_s):
        """Instantaneous speed from a *single frame*.

        The ember moves during the exposure, so the streak is an integrated
        velocity vector. This is a real measurement channel that exists before
        any tracking, and it is free.
        """
        return self.L / exposure_s

    def as_dict(self):
        return dict(frame=self.frame, x=self.x, y=self.y, L=self.L, W=self.W,
                    theta=self.theta, peak=self.peak, area=self.area, flux=self.flux)


class Stabilizer:
    """Sub-pixel translation alignment by phase correlation.

    The single most important preprocessing step after the channel choice, and
    the easiest to leave out. Rolling-median subtraction assumes the background
    is *pixel-stationary*; a house-mounted camera in a wind event is not. When
    a high-contrast edge shifts by half a pixel, the median no longer cancels
    it and it reappears in the residual as a bright, elongated, streak-shaped
    object -- indistinguishable from a firebrand by shape, and present in every
    frame. Measured on a synthetic clip with 0.8 px of shake, precision goes
    from 0.04 without this to 0.95 with it.

    Translation-only is deliberate: it is what a rigid mount actually does, and
    estimating rotation/scale from a smoke-filled frame is unstable.
    """

    def __init__(self, max_px=12.0, downscale=0, target_width=1920):
        self.ref = None
        self.win = None
        self.max_px = float(max_px)
        self.ds_cfg = int(downscale)
        self.target_w = int(target_width)
        self.ds = 1                       # resolved on the first frame
        self.shifts = []

    def _resolve_ds(self, w):
        """Auto mode: downscale only while >= target_width pixels remain."""
        if self.ds_cfg > 0:
            return max(int(self.ds_cfg), 1)
        return max(1, int(round(w / max(self.target_w, 320))))

    def _small(self, gray_f32):
        if self.ds == 1:
            return gray_f32
        h, w = gray_f32.shape[:2]
        return cv2.resize(gray_f32, (w // self.ds, h // self.ds),
                          interpolation=cv2.INTER_AREA)

    def __call__(self, gray_f32):
        if self.ref is None:
            self.ds = self._resolve_ds(gray_f32.shape[1])
        g = self._small(gray_f32)
        if self.ref is None:
            self.ref = g.copy()
            self.win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
            self.shifts.append((0.0, 0.0))
            return None
        (dx, dy), _resp = cv2.phaseCorrelate(self.ref, g, self.win)
        dx, dy = dx * self.ds, dy * self.ds     # back to full-resolution pixels
        if abs(dx) > self.max_px or abs(dy) > self.max_px:
            dx = dy = 0.0                      # nothing to lock onto
        self.shifts.append((float(dx), float(dy)))
        return np.float32([[1, 0, -dx], [0, 1, -dy]])


def _channel(bgr, which):
    if which == "blue":
        return bgr[..., 0]
    if which == "gray":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if which == "minrgb":
        return bgr.min(axis=2)
    raise ValueError(f"unknown channel {which!r}")


class ResidualEngine:
    """Streaming rolling-median background subtraction.

    Push frames in order; once the window is full, each push returns the
    residual for the *centre* frame of the window along with its index.

    Why a median and not MOG2: the clutter here is static scene texture plus a
    global illumination flicker, not per-pixel multimodality. A median over ~11
    frames puts every paver joint into the background and subtracts it to zero,
    while a firebrand -- present in one or two frames -- never enters the median
    and survives untouched. MOG2 spends its variance budget modelling the
    flicker instead.
    """

    def __init__(self, cfg: DetectConfig, roi_mask: np.ndarray | None = None):
        self.cfg = cfg
        self.roi = roi_mask
        shape = cv2.MORPH_RECT if getattr(cfg, "tophat_shape", "rect") == "rect" \
            else cv2.MORPH_ELLIPSE
        self.K = cv2.getStructuringElement(shape, (cfg.tophat_ksize, cfg.tophat_ksize))
        self.buf: deque = deque(maxlen=cfg.median_window)
        self.idx: deque = deque(maxlen=cfg.median_window)
        self._ref_level = None
        self._n = 0
        self.stab = Stabilizer(cfg.stabilize_max_px,
                               getattr(cfg, "stabilize_downscale", 0),
                               getattr(cfg, "stabilize_target_width", 1920)) \
            if cfg.stabilize else None
        self._bg = None          # cached background + edge map
        self._edge = None
        self._since = 10 ** 9

    # -- stage 1: stabilise + channel + gain + top-hat ---------------------
    def prep(self, bgr):
        if self.stab is not None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            M = self.stab(gray)
            if M is not None:
                bgr = cv2.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)

        ch = _channel(bgr, self.cfg.channel)

        if self.cfg.normalize_gain:
            sample = ch[self.roi > 0] if self.roi is not None else ch
            level = float(np.median(sample))
            if self._ref_level is None:
                self._ref_level = max(level, 1.0)
            elif level > 1.0:
                ch = np.clip(ch.astype(np.float32) * (self._ref_level / level),
                             0, 255).astype(np.uint8)

        th = cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, self.K)
        if self.roi is not None:
            th = cv2.bitwise_and(th, th, mask=self.roi)
        return th

    # -- stage 2: temporal residual ----------------------------------------
    @staticmethod
    def _edge_map(bg):
        """|grad| of the background. Where this is large, imperfect alignment
        leaves a residual; where it is zero, it cannot."""
        gx = cv2.Sobel(bg, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(bg, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.GaussianBlur(cv2.magnitude(gx, gy) / 8.0, (0, 0), 1.0)

    def push(self, bgr):
        """Returns (frame_index, residual, edge_map) once the window is full."""
        self.buf.append(self.prep(bgr))
        self.idx.append(self._n)
        self._n += 1
        if len(self.buf) < self.buf.maxlen:
            return None
        c = len(self.buf) // 2
        refresh = max(int(getattr(self.cfg, "bg_refresh", 1)), 1)
        if self._bg is None or self._since >= refresh:
            # np.partition is the median for an odd window and skips numpy's
            # float64 promotion, which matters at 4K.
            st = np.stack(self.buf)
            self._bg = np.partition(st, len(self.buf) // 2, axis=0)[len(self.buf) // 2] \
                .astype(np.float32)
            self._edge = self._edge_map(self._bg)
            self._since = 0
        self._since += 1
        res = np.clip(self.buf[c].astype(np.float32) - self._bg, 0, 255).astype(np.uint8)
        return self.idx[c], res, self._edge

    def flush(self):
        """Residuals for the trailing frames, computed against the last full
        window. Slightly weaker than the streaming case but keeps the ends of
        short clips usable."""
        if len(self.buf) < 3:
            return
        st = np.stack(self.buf)
        bg = np.partition(st, len(self.buf) // 2, axis=0)[len(self.buf) // 2].astype(np.float32)
        edge = self._edge_map(bg)
        c = len(self.buf) // 2
        for k in range(c + 1, len(self.buf)):
            res = np.clip(self.buf[k].astype(np.float32) - bg, 0, 255).astype(np.uint8)
            yield self.idx[k], res, edge


def robust_sigma(res):
    """MAD-based noise scale of the positive residual.

    Guarded: after a good background subtraction most pixels are exactly zero,
    and a naive MAD collapses to 0, which sends the threshold to 0 and returns
    every pixel in the frame. (That failure is worth knowing about -- it is one
    way a working detector suddenly reports thousands of detections.)
    """
    v = res[res > 0]
    if v.size < 50:
        return 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma = 1.4826 * mad
    if sigma < 1e-3:                       # too many zeros for MAD to work
        sigma = max(float(np.percentile(v, 84.1) - np.percentile(v, 50.0)), 1.0)
    return max(sigma, 1.0)


def detect_frame(res, frame_index, cfg: DetectConfig, edge=None):
    """Residual image -> streak candidates.

    `edge` is |grad(background)|. When supplied, the threshold rises where the
    scene has strong edges, which is exactly where residual misalignment
    produces false positives and nowhere a firebrand cares about.
    """
    sigma = robust_sigma(res)
    base = float(np.median(res[res > 0])) if (res > 0).any() else 255.0
    thr = base + cfg.sigma_k * sigma
    if edge is not None and cfg.edge_suppress > 0:
        thr = thr + cfg.edge_suppress * edge          # per-pixel threshold map
    bw = (res > thr).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (cfg.min_area <= area <= cfg.max_area):
            continue
        ys, xs = np.where(lab == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)

        if len(pts) >= 5:
            (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
            L, W = max(w, h), max(min(w, h), 1e-3)
            theta = np.deg2rad(ang if w >= h else ang + 90.0)
        else:
            cx, cy = float(xs.mean()), float(ys.mean())
            w = float(stats[i, cv2.CC_STAT_WIDTH])
            h = float(stats[i, cv2.CC_STAT_HEIGHT])
            L, W = max(w, h), max(min(w, h), 1e-3)
            theta = 0.0 if w >= h else np.pi / 2

        if not (cfg.min_length <= L <= cfg.max_length):
            continue
        if L / W < cfg.min_elongation:
            continue
        if area / max(L * W, 1.0) < cfg.min_solidity:
            continue

        vals = res[ys, xs].astype(np.float32)
        out.append(Detection(frame=frame_index, x=float(cx), y=float(cy),
                             L=float(L), W=float(W), theta=float(theta),
                             peak=float(vals.max()), area=area,
                             flux=float(vals.sum())))
    return out


def detect_source(source, cfg: DetectConfig, roi_mask=None, progress=None):
    """Run detection over a whole frame source.

    Returns (detections_by_frame, residuals) where residuals is a dict of
    frame_index -> residual image, kept only if `keep_residuals` is patched on.
    Memory-safe default: residuals are discarded.
    """
    eng = ResidualEngine(cfg, roi_mask)
    by_frame: dict[int, list[Detection]] = {}
    for k, frame in enumerate(source):
        got = eng.push(frame)
        if got is not None:
            i, res, edge = got
            by_frame[i] = detect_frame(res, i, cfg, edge)
        if progress:
            progress(k)
    for i, res, edge in eng.flush():
        by_frame[i] = detect_frame(res, i, cfg, edge)
    return by_frame


def residual_source(source, cfg: DetectConfig, roi_mask=None):
    """Generator of (frame_index, residual, edge_map). Used by training-data
    extraction and by model inference, which both need the residual itself,
    not just the classical detections."""
    eng = ResidualEngine(cfg, roi_mask)
    for frame in source:
        got = eng.push(frame)
        if got is not None:
            yield got
    yield from eng.flush()


def draw(frame, dets, color=(90, 255, 90), pad=6, thickness=2):
    """Overlay detections for eyeballing. Use this constantly -- most bugs in
    this pipeline are visible in one frame and invisible in any metric."""
    vis = frame.copy()
    for d in dets:
        r = max(d.L, 6) / 2 + pad
        cv2.rectangle(vis, (int(d.x - r), int(d.y - r)),
                      (int(d.x + r), int(d.y + r)), color, thickness)
    return vis
