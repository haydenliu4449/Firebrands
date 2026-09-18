"""Every tunable in one place.

Defaults are calibrated on the Ranch Top Rd 'Left Front' camera
(2025-01-08 00:05:36). Re-run `firebrand.calibrate` on your own clips before
trusting the photometric numbers.
"""
from dataclasses import dataclass, field, asdict
import json


@dataclass
class DetectConfig:
    # --- channel -----------------------------------------------------------
    channel: str = "blue"
    """'blue' | 'gray' | 'minrgb'.

    On this footage the background is fire-glow orange, mean RGB (154, 87, 23),
    and firebrands are near-blackbody white, (250, 241, 227). Measured
    separability (streak-to-background distance in background sigmas):

        red 3.6 | grayscale 9.0 | green 9.1 | blue 19.7 | blue-0.15*red 22.1

    Important caveat, measured rather than assumed: that 2.2x advantage is a
    *single-frame* property, and most of it is redundant once background
    subtraction is doing its job. On the synthetic benchmark, blue and
    grayscale are equivalent for bright embers (recall 0.73 vs 0.72). The
    advantage appears entirely at the faint end:

        peak brightness    blue recall   grayscale recall
        bright (70-255)       0.73            0.72
        medium (35-90)        0.80            0.51
        faint  (20-55)        0.71            0.19   (track recall 0.64 vs 0.00)

    Which is exactly the population that matters for saltation: an ember is
    dimmest while it is on the ground losing heat, and ground contact is the
    event you are trying to measure. Keep 'blue'.

    If a clip switches to monochrome IR night mode the three channels become
    identical and 'blue' silently degrades to 'gray'. `frames.is_ir()` detects
    that; the pipeline warns and carries on.
    """

    tophat_ksize: int = 15
    """Morphological top-hat kernel. Must be wider than the widest streak and
    narrower than the smallest real scene structure. 15 px works for 1920x1080
    on this camera; scale it with resolution."""

    # --- temporal ----------------------------------------------------------
    median_window: int = 11
    """Frames in the rolling background median (odd). Larger = cleaner
    background but a firebrand that lingers >window/2 frames starts erasing
    itself. 11 at 15-30 fps is a good default for saltating embers."""

    stabilize: bool = True
    """Sub-pixel translation alignment to a reference frame, before anything
    else.

    Do not skip this. These cameras are mounted on houses in a wind event, and
    a *sub-pixel* tremor is enough to break rolling-median subtraction: every
    high-contrast edge in the scene leaves a bright residual when it moves,
    so paver joints reappear in the residual as streak-shaped false positives.
    In testing, adding 0.8 px of camera shake to a synthetic clip dropped
    detection precision from 0.95 to 0.04 -- stabilisation restores it.
    """

    stabilize_max_px: float = 12.0
    """Reject implausible alignment estimates (occurs when a frame is mostly
    smoke and phase correlation has nothing to lock onto)."""

    stabilize_downscale: int = 0
    """Estimate the shift on a downscaled copy, apply it at full resolution.
    0 = auto (recommended), 1 = never downscale, N = fixed factor.

    Phase correlation is an FFT over the whole frame and is the most expensive
    step in the pipeline at high resolution -- ~430 ms per frame at 3840x2160.
    Downscaling first is nearly free in accuracy *if you keep enough pixels*.
    Measured against a known 0.83/-0.47 px shift on a 4K frame:

        full res   err 0.291 px    572 ms
        1/2 res    err 0.258 px     62 ms      <- as accurate, 9x faster
        1/4 res    err 0.328 px     12 ms

    But this does NOT generalise downward, and assuming it did cost a
    regression: fixing downscale=2 dropped benchmark precision from 1.00 to
    0.39 at 960x540, because 480x270 is too little signal to localise a 0.8 px
    shake, and a bad shift estimate is worse than no stabilisation at all.

    So 'auto' targets a working resolution near 1920 wide -- factor 2 at 4K,
    factor 1 at 1080p and below -- which keeps the 4K speedup and the accuracy
    at every size."""

    stabilize_target_width: int = 1920
    """Working width that `stabilize_downscale=0` aims for."""

    tophat_shape: str = "rect"
    """'rect' or 'ellipse'. A rectangular structuring element is separable, so
    OpenCV runs it in two passes instead of one 2-D pass: at 4K with a 31 px
    kernel that is 17 ms against 132 ms. Detection quality is unchanged on the
    benchmark -- this filter only flattens the illumination gradient, and it
    does not care about the corners of the kernel."""

    bg_refresh: int = 1
    """Recompute the rolling background median every N frames instead of every
    frame, reusing it in between.

    The median over an 11-frame stack is ~490 ms at 4K, and the background is
    quasi-static by construction, so recomputing it every frame is mostly
    wasted work. N=4 costs a little accuracy at illumination steps and buys
    most of that time back. Left at 1 by default so behaviour is exact;
    run_pipeline raises it automatically for large frames and says so."""

    normalize_gain: bool = True
    """Rescale each frame so the robust background level is constant. The fire
    glow pulses and the camera AGC chases it; without this the median
    subtraction produces a full-frame residual on every brightness step."""

    # --- thresholding ------------------------------------------------------
    sigma_k: float = 6.0
    """Detection threshold in robust (MAD-derived) noise sigmas. Deliberately
    permissive: this stage should over-detect. Precision comes from tracking."""

    min_area: int = 5
    max_area: int = 1200
    min_length: float = 2.5
    max_length: float = 90.0
    min_elongation: float = 1.25
    """Streak shape gate. Firebrands smear during the exposure; most noise
    blobs and compression artifacts do not. Median real streak on this camera
    is 7 px long at ~3:1."""

    min_solidity: float = 0.30
    """component area / (L * W). Rejects wispy edge fragments."""

    edge_suppress: float = 1.0
    """Per-pixel threshold surcharge proportional to the background gradient.

    Stabilisation is never perfect -- a fraction of a pixel of residual
    misalignment always remains, and the residual it leaves at an edge is
    proportional to that edge's gradient. Firebrands are additive light and do
    not care where the scene's edges are, so raising the threshold in
    proportion to |grad(background)| costs almost no real detections while
    removing the shake-driven false positives that survive alignment.

    Measured on a synthetic clip with 0.8 px shake: precision 0.05 raw,
    0.13 with stabilisation alone, 0.63 with stabilisation + edge suppression,
    at essentially unchanged recall. Set to 0 to disable.
    """


@dataclass
class TrackConfig:
    fps: float = 15.0
    exposure_s: float | None = None
    """Camera exposure time in seconds. If known, streak length converts
    directly to speed (px/s = L / exposure_s) and predicts the inter-frame
    displacement exactly. If None, `track.estimate_step_ratio` infers the
    equivalent ratio from the data."""

    step_ratio: float | None = None
    """displacement_between_frames / streak_length. Equals
    (1/fps) / exposure_s. Set directly, or leave None to auto-estimate."""

    max_step_px: float = 120.0
    """Hard cap on inter-frame displacement, used to bound the search and as
    the fallback gate when step_ratio is unknown."""

    gate_frac: float = 0.55
    """Association gate as a fraction of the predicted step."""

    max_gap: int = 2
    """Frames a track may go undetected and still be continued. Embers do
    briefly dim below threshold."""

    # --- acceptance: this is what replaces the auto-labeller ---------------
    min_track_len: int = 3
    """Detections required before a candidate is called a firebrand. The whole
    design rests on this: single-frame appearance cannot separate an ember from
    a paver joint, but a paver joint never moves coherently for 3 frames."""

    max_heading_change_deg: float = 25.0
    """Per-frame turn limit. Embers have inertia; noise associations do not."""

    max_brightness_ratio: float = 5.0
    """Peak intensity may vary along a track, but not by more than this."""

    max_speed_cv: float = 0.8
    """Coefficient of variation of inter-frame step length. Catches
    associations that stitch two different particles together."""


@dataclass
class SynthConfig:
    n_particles_per_frame: tuple = (2, 14)
    speed_px: tuple = (6.0, 55.0)
    """Distance travelled between frames, in px."""

    exposure_duty: float = 0.35
    """exposure_time / frame_interval.

    The streak records only the part of the frame interval the shutter was
    open, so streak_length = speed_px * exposure_duty and the particle jumps
    the rest of the way in the dark. This is the reciprocal of
    TrackConfig.step_ratio, and setting it below 1 is what makes synthetic
    clips exercise the tracker the way real footage does. A camera at 1/60 s
    and 15 fps has a duty of 0.25.
    """
    peak_srgb: tuple = (70.0, 255.0)
    """Peak streak brightness in 8-bit sRGB, above background.

    Parameterised by peak rather than total flux because peak is what you can
    read straight off a real detection (`Detection.peak`), so
    `synth.fit_config_to_tracks` can match it to measured data without a
    conversion you would have to trust. Real streaks on this camera peak near
    226 in the blue channel against a background of 23.
    """
    psf_sigma: tuple = (0.6, 1.3)
    curvature: tuple = (-0.08, 0.08)
    gravity_px: float = 0.9
    """Downward acceleration in px per frame^2 (image space, uncalibrated)."""
    drag: float = 0.02
    restitution: tuple = (0.25, 0.6)
    """Rebound speed / impact speed at a ground contact."""
    ground_y_frac: float = 0.62
    """Where the ground plane sits in the frame, as a fraction of height.
    Only used to make synthetic motion saltation-like."""
    reencode_h264: bool = True
    """Round-trip synthetic clips through the same codec as the originals.
    Without this the model learns 'real embers have compression ringing,
    synthetic ones do not' and collapses on real footage."""


@dataclass
class TrainConfig:
    tile: int = 256
    stride: int = 192
    n_frames_stack: int = 3
    """Residual frames fed as input channels: t-1, t, t+1. This is the single
    biggest accuracy lever. A network given one frame is being asked to decide
    'is this bright thing static?' from an image that cannot answer it."""

    batch_size: int = 16
    lr: float = 3e-4
    epochs: int = 30
    pos_weight: float = 20.0
    """Firebrand pixels are ~0.1% of the frame. Without reweighting, predicting
    all-zero is a near-optimal loss."""
    dice_weight: float = 0.5
    base_ch: int = 16
    """U-Net width. 16 gives ~0.5M params, which is plenty: this is a
    low-level texture task, not a semantic one."""
    val_frac: float = 0.15
    seed: int = 0
    amp: bool = True


@dataclass
class Config:
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(
            detect=DetectConfig(**d.get("detect", {})),
            track=TrackConfig(**d.get("track", {})),
            synth=SynthConfig(**{k: tuple(v) if isinstance(v, list) else v
                                 for k, v in d.get("synth", {}).items()}),
            train=TrainConfig(**d.get("train", {})),
        )
