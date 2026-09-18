"""Linking, and the acceptance rule that makes the tracker the detector.

The pitch: run an over-sensitive per-frame detector, then keep only what moves
coherently. A paver joint is bright and elongated in every frame but never
travels; an ember travels smoothly. Nothing in a single frame separates them,
so the decision is made here.

Note on off-the-shelf trackers: SORT / ByteTrack / DeepSORT all associate on
IoU. A 7 px object moving 40 px between frames has *zero* box overlap, so every
association score is zero and they produce nothing. Association here is on
predicted position, with the streak geometry supplying the prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackConfig
from .detect import Detection


@dataclass
class Track:
    id: int
    dets: list = field(default_factory=list)

    # -- geometry ----------------------------------------------------------
    @property
    def frames(self):
        return [d.frame for d in self.dets]

    @property
    def xy(self):
        return np.array([[d.x, d.y] for d in self.dets], float)

    @property
    def steps(self):
        """Inter-frame displacement magnitudes, normalised by the frame gap."""
        p, f = self.xy, np.array(self.frames, float)
        if len(p) < 2:
            return np.array([])
        return np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(f), 1)

    @property
    def headings(self):
        p = self.xy
        if len(p) < 2:
            return np.array([])
        d = np.diff(p, axis=0)
        return np.arctan2(d[:, 1], d[:, 0])

    def last(self):
        return self.dets[-1]

    def predict(self, cfg: TrackConfig, step_ratio: float, next_frame: int):
        """Where this track should appear at `next_frame`.

        Two frames in, use the observed velocity (a constant-velocity model is
        right for an ember between contacts). With only one detection, fall
        back to the streak itself: length x step_ratio is the expected
        displacement, and theta is the direction.
        """
        gap = max(next_frame - self.last().frame, 1)
        if len(self.dets) >= 2:
            a, b = self.dets[-2], self.dets[-1]
            dt = max(b.frame - a.frame, 1)
            vx, vy = (b.x - a.x) / dt, (b.y - a.y) / dt
            step = float(np.hypot(vx, vy))
            return b.x + vx * gap, b.y + vy * gap, max(step, 1.0)

        d = self.last()
        step = min(d.L * step_ratio, cfg.max_step_px)
        # A streak's orientation is defined mod pi -- the raw angle cannot say
        # which end is the head. With one detection we have to try both; the
        # gate below is generous enough that the wrong sign is usually rejected
        # on the next frame by the heading-consistency check.
        return d.x + step * np.cos(d.theta) * gap, d.y + step * np.sin(d.theta) * gap, max(step, 1.0)


# ---------------------------------------------------------------------------
# linking
# ---------------------------------------------------------------------------

def link(dets_by_frame: dict[int, list[Detection]], cfg: TrackConfig,
         step_ratio: float | None = None) -> list[Track]:
    """Frame-to-frame assignment, solved optimally per frame pair.

    linear_sum_assignment gives the globally best matching *within* a frame
    pair, which is much better than greedy nearest-neighbour when several
    embers fly close together -- exactly the case in dense ember cast.
    """
    if step_ratio is None:
        step_ratio = resolve_step_ratio(cfg)

    frames = sorted(dets_by_frame)
    live: list[Track] = []
    done: list[Track] = []
    next_id = 0

    for fi in frames:
        dets = dets_by_frame[fi]

        # retire tracks that have gone quiet
        still, retired = [], []
        for t in live:
            (still if fi - t.last().frame <= cfg.max_gap + 1 else retired).append(t)
        live, done = still, done + retired

        if not dets:
            continue
        if not live:
            for d in dets:
                live.append(Track(next_id, [d])); next_id += 1
            continue

        # cost = distance from prediction, gated per-track
        C = np.full((len(live), len(dets)), 1e6, float)
        for i, t in enumerate(live):
            px, py, step = t.predict(cfg, step_ratio, fi)
            # The gate must grow with the gap: predicting two frames ahead has
            # roughly twice the error of predicting one. Without this, max_gap
            # is silently inoperative -- the track survives but can never match.
            gap = max(fi - t.last().frame, 1)
            gate = max(cfg.gate_frac * step * gap, 6.0)
            for j, d in enumerate(dets):
                dist = float(np.hypot(d.x - px, d.y - py))
                if dist <= gate:
                    C[i, j] = dist

        ri, ci = linear_sum_assignment(C)
        used = set()
        for i, j in zip(ri, ci):
            if C[i, j] < 1e5:
                live[i].dets.append(dets[j])
                used.add(j)
        for j, d in enumerate(dets):
            if j not in used:
                live.append(Track(next_id, [d])); next_id += 1

    return done + live


def resolve_step_ratio(cfg: TrackConfig) -> float:
    """displacement between frames / streak length.

    Exactly (1/fps) / exposure_s. A 1/500 s exposure at 15 fps gives 33, i.e.
    the ember travels 33 streak-lengths between frames. If exposure is unknown
    the caller should use `estimate_step_ratio` instead of this fallback.
    """
    if cfg.step_ratio is not None:
        return float(cfg.step_ratio)
    if cfg.exposure_s:
        return float((1.0 / cfg.fps) / cfg.exposure_s)
    return 8.0


def estimate_step_ratio(dets_by_frame, cfg: TrackConfig, angle_tol_deg=25.0,
                        min_pairs=40, return_diagnostics=False):
    """Self-calibrate the ratio when the camera will not report its exposure.

    The trick is that a streak already tells you which *direction* the ember
    travelled (up to a 180-degree ambiguity). So for every detection, look at
    the next frame's detections that lie along that direction, and record
    distance / streak_length. Genuine successor pairs all give the same value --
    the true (1/fps) / exposure_s -- while accidental pairs scatter. The
    histogram peak is the answer.

    Doing it this way rather than by trying candidate ratios and keeping
    whichever produces the most tracks matters: that objective is maximised by
    ratios so large the association gate swallows the whole frame, which
    produces many long tracks that are all ID switches. Ask the geometry
    directly instead.

    Returns the ratio, or (ratio, diagnostics) if requested. Diagnostics carry
    `n_pairs` and `sharpness` (peak height over median bin height); sharpness
    below ~2 means the estimate is not trustworthy and you should measure the
    exposure time instead.
    """
    tol = np.deg2rad(angle_tol_deg)
    ratios = []
    frames = sorted(dets_by_frame)
    for a, b in zip(frames, frames[1:]):
        if b - a != 1:
            continue
        A, B = dets_by_frame[a], dets_by_frame[b]
        if not A or not B:
            continue
        Bxy = np.array([[d.x, d.y] for d in B], float)
        for d in A:
            v = Bxy - np.array([d.x, d.y])
            dist = np.linalg.norm(v, axis=1)
            ok = (dist > 1e-6) & (dist <= cfg.max_step_px)
            if not ok.any():
                continue
            ang = np.arctan2(v[:, 1], v[:, 0])
            dth = np.abs(np.angle(np.exp(1j * (ang - d.theta))))
            dth = np.minimum(dth, np.pi - dth)        # streak angle is mod pi
            m = ok & (dth <= tol)
            if m.any():
                k = int(np.argmin(np.where(m, dist, np.inf)))
                ratios.append(dist[k] / max(d.L, 1e-3))

    fallback = resolve_step_ratio(cfg)
    if len(ratios) < min_pairs:
        return (fallback, dict(n_pairs=len(ratios), sharpness=0.0)) \
            if return_diagnostics else fallback

    r = np.asarray(ratios, float)
    r = r[(r > 0.2) & (r < 200)]
    lo, hi = np.log10(0.2), np.log10(200)
    hist, edges = np.histogram(np.log10(r), bins=40, range=(lo, hi))
    k = int(np.argmax(hist))
    peak = 10 ** ((edges[k] + edges[k + 1]) / 2)
    near = r[(r > peak / 1.6) & (r < peak * 1.6)]
    est = float(np.median(near)) if near.size else float(peak)
    sharp = float(hist[k] / max(np.median(hist[hist > 0]), 1))

    return (est, dict(n_pairs=int(r.size), sharpness=sharp)) \
        if return_diagnostics else est


def stitch(tracks: list[Track], cfg: TrackConfig, max_frame_gap=4,
           pos_tol=1.4, ang_tol_deg=30.0) -> list[Track]:
    """Second pass: rejoin tracks that are two halves of one ember.

    Frame-to-frame linking cannot bridge a run of missed detections, so a
    single ember that dims below threshold for three frames comes out as two
    tracks. That inflates your hop count and halves your measured hop lengths,
    which is a much worse error for saltation statistics than a missed
    detection would be.

    Two fragments are joined when extrapolating the first with its own velocity
    lands close to the start of the second, and the headings agree.
    """
    order = sorted(tracks, key=lambda t: t.frames[0])
    used, out = set(), []

    for i, t in enumerate(order):
        if id(t) in used:
            continue
        cur = t
        used.add(id(cur))
        merged = True
        while merged:
            merged = False
            end = cur.dets[-1]
            if len(cur.dets) < 2:
                break
            a, b = cur.dets[-2], cur.dets[-1]
            dt = max(b.frame - a.frame, 1)
            vx, vy = (b.x - a.x) / dt, (b.y - a.y) / dt
            speed = float(np.hypot(vx, vy))
            h0 = np.arctan2(vy, vx)

            best, best_d = None, np.inf
            for u in order:
                if id(u) in used:
                    continue
                gap = u.frames[0] - end.frame
                if not (1 <= gap <= max_frame_gap):
                    continue
                px, py = end.x + vx * gap, end.y + vy * gap
                d = float(np.hypot(u.dets[0].x - px, u.dets[0].y - py))
                if d > pos_tol * speed * gap:
                    continue
                if len(u.headings):
                    dh = abs(np.angle(np.exp(1j * (u.headings[0] - h0))))
                    if np.rad2deg(dh) > ang_tol_deg:
                        continue
                if d < best_d:
                    best, best_d = u, d
            if best is not None:
                cur.dets.extend(best.dets)
                cur.dets.sort(key=lambda d: d.frame)
                used.add(id(best))
                merged = True
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# acceptance -- this is the labeller
# ---------------------------------------------------------------------------

def _heading_ok(t: Track, max_deg):
    h = t.headings
    if len(h) < 2:
        return True
    d = np.abs(np.diff(np.unwrap(h)))
    return bool(np.all(np.rad2deg(d) <= max_deg))


def accept(tracks: list[Track], cfg: TrackConfig) -> list[Track]:
    """Keep only tracks that behave like a physical particle.

    Each rule kills a specific failure:
      length      -- static texture that flickered above threshold once
      heading     -- associations that jump between unrelated detections
      brightness  -- a stitch between a bright ember and a faint artifact
      speed CV    -- a stitch between two particles at different speeds
    """
    out = []
    for t in tracks:
        if len(t.dets) < cfg.min_track_len:
            continue
        if not _heading_ok(t, cfg.max_heading_change_deg):
            continue
        peaks = np.array([d.peak for d in t.dets], float)
        if peaks.min() <= 0 or peaks.max() / peaks.min() > cfg.max_brightness_ratio:
            continue
        s = t.steps
        if len(s) >= 2 and s.mean() > 1e-6 and s.std() / s.mean() > cfg.max_speed_cv:
            continue
        out.append(t)
    return out


def rejected(tracks: list[Track], accepted: list[Track]) -> list[Track]:
    """Everything acceptance threw away.

    Do not discard these. Each one is a labelled *negative* -- a paver joint, a
    compression block, a reflection -- and hard negatives mined from your own
    footage are worth far more to the model than random background crops.
    """
    keep = {id(t) for t in accepted}
    return [t for t in tracks if id(t) not in keep]


# ---------------------------------------------------------------------------
# physics-based validation (needs no labels at all)
# ---------------------------------------------------------------------------

def ballistic_residual(t: Track) -> float:
    """RMS residual of a constant-acceleration fit to the track, in px.

    A real ember between contacts follows gravity plus drag, which over a few
    frames is well approximated by a quadratic in time. A track that is
    actually an ID switch between two different particles cannot be fit by one,
    and shows up here as a large residual.

    This is a validation signal that costs nothing and needs no ground truth --
    most tracking projects do not have one. Monitor it continuously, not just
    at the end.
    """
    f = np.array(t.frames, float)
    p = t.xy
    if len(f) < 4:
        return 0.0
    f = f - f.mean()
    A = np.stack([np.ones_like(f), f, f ** 2], 1)
    res = 0.0
    for k in range(2):
        coef, *_ = np.linalg.lstsq(A, p[:, k], rcond=None)
        res += float(np.mean((A @ coef - p[:, k]) ** 2))
    return float(np.sqrt(res))


def summarize(tracks: list[Track], cfg: TrackConfig) -> "list[dict]":
    """One row per track, ready for pandas."""
    rows = []
    for t in tracks:
        s = t.steps
        rows.append(dict(
            track_id=t.id,
            n_det=len(t.dets),
            frame_start=t.frames[0], frame_end=t.frames[-1],
            x0=t.dets[0].x, y0=t.dets[0].y,
            x1=t.dets[-1].x, y1=t.dets[-1].y,
            mean_step_px=float(s.mean()) if len(s) else 0.0,
            mean_streak_L=float(np.mean([d.L for d in t.dets])),
            mean_peak=float(np.mean([d.peak for d in t.dets])),
            total_flux=float(np.sum([d.flux for d in t.dets])),
            heading_deg=float(np.rad2deg(np.mean(t.headings))) if len(t.headings) else 0.0,
            ballistic_rms_px=ballistic_residual(t),
        ))
    return rows


def to_yolo_labels(tracks: list[Track], shape, box_pad=4.0):
    """Export accepted tracks as YOLO-format boxes, one file per frame.

    Returns {frame_index: [(cls, xc, yc, w, h), ...]} normalised to [0, 1].

    These are the auto-labels -- but unlike the first attempt's, they were
    filtered by motion, so their errors are not the single-frame detector's
    errors baked into a training set.
    """
    H, W = shape[:2]
    per_frame: dict[int, list] = {}
    for t in tracks:
        for d in t.dets:
            side = max(d.L, 6.0) + 2 * box_pad
            per_frame.setdefault(d.frame, []).append(
                (0, d.x / W, d.y / H, side / W, side / H))
    return per_frame
