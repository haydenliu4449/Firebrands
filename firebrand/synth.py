"""Synthetic firebrands: the way to get a training set in an afternoon.

A firebrand is a moving point emitter integrated over the exposure. That is a
simple and *physically correct* appearance model, which means the rendering is
not an approximation of the real thing -- it is the same forward model, with
parameters you sample. That is why this works far better here than synthetic
data usually does.

Three details decide whether the model trained on this transfers:

  1. Real backgrounds. Composite onto frames from your own camera (or the
     rolling median of them). Synthetic backgrounds lack your pavement texture,
     which is the exact confuser the model must learn to reject.
  2. Additive, in linear light. A glowing object adds photons; it does not
     alpha-blend. Compositing in gamma space gives streaks the wrong
     brightness profile and the model learns the wrong edge statistics.
  3. Re-encode through the same codec. Otherwise the model learns "real embers
     have compression ringing, synthetic ones do not" and collapses on real
     footage.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import SynthConfig

GAMMA = 2.2


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def srgb_to_linear(v):
    return (np.asarray(v, np.float32) / 255.0) ** GAMMA


def flux_for_peak(peak_lin, length_px, psf_sigma):
    """Total linear flux that produces `peak_lin` at the streak core.

    A streak of total flux A spread evenly over length L and convolved with a
    2-D Gaussian of width sigma has a cross-sectional peak of
    (A / L) / (sqrt(2*pi) * sigma). Inverting that lets the generator be
    parameterised by peak brightness, which is directly measurable on real
    detections, instead of by flux, which is not.
    """
    return float(peak_lin) * max(float(length_px), 1.0) * np.sqrt(2 * np.pi) * max(float(psf_sigma), 0.35)


def render_streak(shape, x0, y0, x1, y1, amp=None, psf_sigma=1.1, curvature=0.0,
                  supersample=4, peak_lin=None):
    """Line integral of a point emitter travelling (x0,y0)->(x1,y1).

    Returns a float32 image in linear light. Give either `amp` (total linear
    flux) or `peak_lin` (peak linear intensity at the core); `peak_lin` is
    usually what you want, because it is what you measure on real streaks.

    `curvature` bends the path slightly, as drag does over a real exposure.
    Supersampling along the path avoids the dashed look you get from stepping
    one pixel at a time.
    """
    h, w = shape[:2]
    canvas = np.zeros((h, w), np.float32)
    length = float(np.hypot(x1 - x0, y1 - y0))
    if amp is None:
        amp = flux_for_peak(peak_lin if peak_lin is not None else 0.3,
                            length, psf_sigma)
    n = max(int(length * supersample), 8)

    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    nx, ny = -(y1 - y0), (x1 - x0)                 # normal to the path
    nrm = max(np.hypot(nx, ny), 1e-6)
    nx, ny = nx / nrm, ny / nrm

    ts = np.linspace(0.0, 1.0, n)
    bow = curvature * length * (ts * (1 - ts) * 4.0)      # 0 at ends, max mid
    xs = x0 + (x1 - x0) * ts + nx * bow
    ys = y0 + (y1 - y0) * ts + ny * bow

    per = amp / n
    xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if not ok.any():
        return canvas
    np.add.at(canvas, (yi[ok], xi[ok]), per)

    if psf_sigma > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), psf_sigma)
    return canvas


def composite(bg_bgr, layers, emitter_bgr=(0.92, 0.97, 1.0)):
    """Add glow layers to a background, additively, in linear light.

    `emitter_bgr` is the colour of the emitter. Firebrands measure near
    (250, 241, 227) in sRGB on this camera -- slightly warm white -- which is
    what makes the blue channel such a good discriminator against the orange
    ambient.
    """
    lin = (bg_bgr.astype(np.float32) / 255.0) ** GAMMA
    glow = np.zeros(lin.shape[:2], np.float32)
    for l in layers:
        glow += l
    lin += glow[..., None] * np.asarray(emitter_bgr, np.float32)[None, None, :]
    return (np.clip(lin, 0, 1) ** (1.0 / GAMMA) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# motion
# ---------------------------------------------------------------------------

@dataclass
class Particle:
    x: float; y: float; vx: float; vy: float
    peak_lin: float; psf: float; curv: float; tid: int
    alive: bool = True


class SaltationSim:
    """2-D image-space particle motion with gravity, drag and bouncing.

    This is *not* a physical saltation model -- it is uncalibrated image space,
    and its only job is to produce motion statistics realistic enough to
    (a) exercise the tracker and (b) teach the network what coherent motion
    looks like. The real dynamics work happens after calibration, on real
    tracks.
    """

    def __init__(self, shape, cfg: SynthConfig, rng=None):
        self.h, self.w = shape[:2]
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(0)
        self.ground_y = self.h * cfg.ground_y_frac
        self.parts: list[Particle] = []
        self._next_id = 0

    def _spawn(self):
        r = self.rng
        speed = r.uniform(*self.cfg.speed_px)
        # Ember cast on this camera runs broadly left-to-right and downward;
        # keep the heading distribution wide so the tracker is not tuned to it.
        ang = r.uniform(np.deg2rad(-40), np.deg2rad(85))
        p = Particle(
            x=r.uniform(-40, self.w * 0.55),
            y=r.uniform(0, self.ground_y),
            vx=speed * np.cos(ang), vy=speed * np.sin(ang),
            peak_lin=float(srgb_to_linear(r.uniform(*self.cfg.peak_srgb))),
            psf=r.uniform(*self.cfg.psf_sigma),
            curv=r.uniform(*self.cfg.curvature),
            tid=self._next_id,
        )
        self._next_id += 1
        self.parts.append(p)
        return p

    def step(self):
        """Advance one frame. Returns per-particle (start, end) segments, which
        are exactly what the exposure integrates over."""
        c = self.rng
        target = c.integers(*self.cfg.n_particles_per_frame)
        while len([p for p in self.parts if p.alive]) < target:
            self._spawn()

        duty = float(np.clip(self.cfg.exposure_duty, 0.02, 1.0))
        segs = []
        for p in self.parts:
            if not p.alive:
                continue
            x0, y0 = p.x, p.y
            sp = np.hypot(p.vx, p.vy)
            p.vx -= self.cfg.drag * p.vx * sp / max(sp, 1e-6)
            p.vy += self.cfg.gravity_px - self.cfg.drag * p.vy * sp / max(sp, 1e-6)
            p.x += p.vx
            p.y += p.vy

            if p.y >= self.ground_y and p.vy > 0:            # saltation contact
                e = self.rng.uniform(*self.cfg.restitution)
                p.y = self.ground_y - (p.y - self.ground_y) * e
                p.vy = -abs(p.vy) * e
                p.vx *= 0.9
                p.peak_lin *= 0.75                            # cools on impact

            if not (-60 <= p.x <= self.w + 60) or p.y > self.h + 60 or p.peak_lin < 0.015:
                p.alive = False
                continue
            # The shutter is open for `duty` of the interval: the streak covers
            # only that much of the travel, centred on the frame timestamp.
            mx, my = (x0 + p.x) / 2.0, (y0 + p.y) / 2.0
            hx, hy = (p.x - x0) * duty / 2.0, (p.y - y0) * duty / 2.0
            segs.append((p, (mx - hx, my - hy), (mx + hx, my + hy)))
        self.parts = [p for p in self.parts if p.alive]
        return segs


# ---------------------------------------------------------------------------
# clip generation
# ---------------------------------------------------------------------------

def make_clip(background, n_frames, cfg: SynthConfig, seed=0,
              noise_sigma=1.2, roi_mask=None, flicker_pct=6.0, shake_px=0.8):
    """Render a synthetic clip with exact ground truth.

    `background` is a real BGR frame from your camera, or a list of them (a
    real background sequence is strongly preferred -- it carries the camera's
    own noise, flicker and micro-shake for free).

    `flicker_pct` and `shake_px` simulate what a single repeated still cannot:
    the fire glow pulsing (which the AGC chases) and the camera trembling in
    the wind. A pipeline validated only against a perfectly static background
    will look far better here than it performs on real footage, so these are
    on by default. Set both to 0 to isolate a bug.

    Returns (frames, gt) where gt is a list, one entry per frame, of dicts:
        {track_id, x, y, L, theta, peak_lin}
    with (x, y) the streak centre -- the same convention `detect.Detection`
    uses, so they compare directly.
    """
    rng = np.random.default_rng(seed)
    bgs = background if isinstance(background, (list, tuple)) else [background]
    shape = bgs[0].shape
    sim = SaltationSim(shape, cfg, rng)

    frames, gt = [], []
    for k in range(n_frames):
        bg = bgs[k % len(bgs)]
        if flicker_pct or shake_px:
            bg = bg.astype(np.float32)
            if flicker_pct:
                bg = bg * (1.0 + rng.normal(0, flicker_pct / 100.0))
            if shake_px:
                dx, dy = rng.normal(0, shake_px, 2)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                bg = cv2.warpAffine(bg, M, (shape[1], shape[0]),
                                    borderMode=cv2.BORDER_REFLECT)
            bg = np.clip(bg, 0, 255).astype(np.uint8)
        segs = sim.step()
        layers, truth = [], []
        for p, (x0, y0), (x1, y1) in segs:
            if roi_mask is not None:
                xi, yi = int(np.clip((x0 + x1) / 2, 0, shape[1] - 1)), int(np.clip((y0 + y1) / 2, 0, shape[0] - 1))
                if roi_mask[yi, xi] == 0:
                    continue
            layers.append(render_streak(shape, x0, y0, x1, y1, amp=None,
                                        psf_sigma=p.psf, curvature=p.curv,
                                        peak_lin=p.peak_lin))
            truth.append(dict(track_id=p.tid,
                              x=(x0 + x1) / 2.0, y=(y0 + y1) / 2.0,
                              L=float(np.hypot(x1 - x0, y1 - y0)),
                              theta=float(np.arctan2(y1 - y0, x1 - x0)),
                              peak_lin=float(p.peak_lin)))
        f = composite(bg, layers)
        if noise_sigma > 0:
            f = np.clip(f.astype(np.float32) +
                        rng.normal(0, noise_sigma, f.shape), 0, 255).astype(np.uint8)
        frames.append(f)
        gt.append(truth)
    return frames, gt


def reencode(frames, fps=15, crf=23, preset="medium"):
    """Round-trip frames through H.264 so synthetic data carries the same
    compression artifacts as the originals.

    Match `crf` to your source bitrate. If you skip this step the model will
    learn to key on ringing and blocking that only real footage has.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, f in enumerate(frames):
            cv2.imwrite(str(td / f"{i:06d}.png"), f)
        vid = td / "clip.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", str(td / "%06d.png"), "-c:v", "libx264", "-preset", preset,
             "-crf", str(crf), "-pix_fmt", "yuv420p", str(vid)], check=True)
        cap = cv2.VideoCapture(str(vid))
        out = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            out.append(f)
        cap.release()
    return out


def gt_masks(gt_frame, shape, width=2.0):
    """Ground-truth streak mask for one frame -- the training target.

    A mask rather than a box: regressing four box coordinates from a 7 px
    object is numerically unstable, while a mask is a dense, well-conditioned
    target and hands you the streak's length and orientation directly, which
    you need for the physics anyway.
    """
    m = np.zeros(shape[:2], np.float32)
    for g in gt_frame:
        dx, dy = np.cos(g["theta"]) * g["L"] / 2, np.sin(g["theta"]) * g["L"] / 2
        cv2.line(m, (int(round(g["x"] - dx)), int(round(g["y"] - dy))),
                 (int(round(g["x"] + dx)), int(round(g["y"] + dy))),
                 1.0, thickness=max(int(round(width)), 1), lineType=cv2.LINE_AA)
    return np.clip(m, 0, 1)


# ---------------------------------------------------------------------------
# matching real statistics
# ---------------------------------------------------------------------------

def fit_config_to_tracks(track_rows, base: SynthConfig, step_ratio=None) -> SynthConfig:
    """Retune the synthetic parameters to match verified real tracks.

    This is what the contact-sheet review is really for: 800 verified tracks
    are not a big training set, but they are an excellent *parameter estimate*
    for a generator that can then produce an unlimited one. Sampling from
    uniforms instead of your measured distributions is the most common way
    synthetic-data pipelines quietly fail.
    """
    import copy
    cfg = copy.deepcopy(base)
    if not track_rows:
        return cfg
    L = np.array([r["mean_streak_L"] for r in track_rows], float)
    P = np.array([r["mean_peak"] for r in track_rows], float)
    STEP = np.array([r.get("mean_step_px", np.nan) for r in track_rows], float)

    # SynthConfig.speed_px is travel *between frames*; mean_streak_L is travel
    # during the exposure only. Setting one from the other without dividing by
    # the duty cycle understates ember speed by 2-4x, and then every synthetic
    # streak is too short and too slow to look like the real thing.
    if step_ratio is None and np.isfinite(STEP).sum() >= 10:
        good = np.isfinite(STEP) & (L > 0.5)
        step_ratio = float(np.median(STEP[good] / L[good])) if good.sum() else None
    if step_ratio and step_ratio > 0:
        cfg.exposure_duty = float(np.clip(1.0 / step_ratio, 0.02, 1.0))

    if len(STEP) >= 10 and np.isfinite(STEP).all():
        cfg.speed_px = (float(np.percentile(STEP, 5)), float(np.percentile(STEP, 95)))
    elif len(L) >= 10:
        duty = max(cfg.exposure_duty, 1e-3)
        cfg.speed_px = (float(np.percentile(L, 5) / duty),
                        float(np.percentile(L, 95) / duty))
    if len(P) >= 10:
        # Detection.peak is measured on the residual, which is already
        # background-subtracted -- the same quantity peak_srgb describes.
        cfg.peak_srgb = (float(np.clip(np.percentile(P, 5), 20, 255)),
                         float(np.clip(np.percentile(P, 97), 40, 255)))
    return cfg
