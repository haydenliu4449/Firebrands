#!/usr/bin/env python3
"""Build the Colab notebook.

nbformat stores `source` as a list of lines that the client joins with NO
separator, so every line must keep its own trailing newline. Splitting on "\n"
(which drops them) produces a file that looks fine in the JSON and collapses to
one line when opened. Validate by joining with "" -- the way Jupyter does.
"""
import ast
import json
import pathlib


def lines(src):
    """Text -> nbformat source list, newlines preserved."""
    return src.strip("\n").splitlines(keepends=True)


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines(src)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(src)}


cells = []

cells.append(md("""
# Firebrand detection — end to end

Single-camera saltation tracking for the Berkeley Fire Lab firebrand project.

**The one idea:** a firebrand is defined by its *motion*, not its appearance. A
7 px ember and a 7 px crack in the concrete are identical in one frame, so
detection has to be temporal. This notebook runs a deliberately over-sensitive
per-frame detector and lets a 3-frame coherence rule decide what was real —
which is what makes the auto-labels good enough to train on.

For the training section: **Runtime → Change runtime type → T4 GPU**.

Everything before section 6 runs fine on CPU.
"""))

# ---------------------------------------------------------------- 0. setup
cells.append(md("## 0. Setup"))
cells.append(code("""
# On a Vertex AI Workbench instance, open a terminal and clone or upload the
# kit into your home directory, then cd into it here:
# %cd ~/firebrand-kit
#
# On Colab instead:
# from google.colab import drive; drive.mount('/content/drive')
# %cd /content/drive/MyDrive/firebrand-kit

!pip -q install opencv-python-headless scipy pandas google-cloud-storage

import sys, os
sys.path.insert(0, os.getcwd())

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from firebrand import Config
from firebrand import frames as F
from firebrand import detect as D
from firebrand import track as T
from firebrand import synth as S
from firebrand import review as R
from firebrand import evaluate as E
from firebrand import dataset as DS
from firebrand import gcsio

print('imports ok')
print('Cloud Storage backend:', gcsio.available())
"""))

# ---------------------------------------------------------------- 1. clip
cells.append(md("""
## 1. Load a clip

Set `CLIP` to your own footage. If that path does not exist the cell builds a
synthetic stand-in from the bundled sample frame, so the whole notebook is
runnable before you have uploaded anything — useful for checking the install.
"""))
cells.append(code("""
# Set BUCKET to your bucket name (from cloud/setup.sh), or leave it None to
# work with local files. Both work -- every path in the kit accepts gs://.
BUCKET = os.environ.get('FIREBRAND_BUCKET') or None

CLIP = f'gs://{BUCKET}/clips/eaton_01.mp4' if BUCKET else 'clips/eaton_01.mp4'
OUT = f'gs://{BUCKET}/work/eaton_01' if BUCKET else 'work/eaton_01'
MAX_FRAMES = 300                 # start small; raise once one clip works

# gs:// clips are downloaded to ~/.firebrand_cache once, then reused.
if gcsio.is_gcs(CLIP):
    have = gcsio.exists(CLIP)
    if not have:
        print(f'{CLIP} not found in the bucket. What is there:')
        try:
            for o in gcsio.listdir(f'gs://{BUCKET}/clips/')[:20]:
                print('  ', o)
        except Exception as e:
            print('  could not list:', e)
elif os.path.exists(CLIP):
    have = True
else:
    have = False

if have:
    src = F.VideoSource(CLIP)
    frames = [f for _, f in zip(range(MAX_FRAMES), src)]
    fps = float(src.fps or 15)
    print(f'loaded {len(frames)} frames from {CLIP}')
else:
    print('No clip available -- building a synthetic demo clip instead.')
    print('Replace CLIP with a real path when you have one.')
    bg = cv2.resize(cv2.imread('tests/sample_frame.png'), (1280, 720))
    demo = Config()
    demo.synth.n_particles_per_frame = (4, 12)
    demo.synth.speed_px = (10.0, 45.0)
    demo.synth.exposure_duty = 0.35
    demo.synth.reencode_h264 = False
    frames, _demo_gt = S.make_clip(bg, 90, demo.synth, seed=1,
                                   flicker_pct=6.0, shake_px=0.9)
    fps = 15.0

H, W = frames[0].shape[:2]
print(f'{len(frames)} frames, {W}x{H}, fps={fps:.2f}')
"""))

# ---------------------------------------------------------------- 2. audit
cells.append(md("""
## 2. Audit — do this before anything else

Three things silently corrupt results downstream:

- **IR night mode** — the three colour channels collapse to one and the blue
  channel stops helping (recall on faint embers falls from 0.71 to 0.19).
- **Duplicate frames** — many DVRs advertise 30 fps by repeating frames from a
  15 fps sensor. Near 0.5 here means every velocity you compute is wrong by 2×.
- **Channel separability** on *your* clips, not on the one frame the defaults
  were tuned on.
"""))
cells.append(code("""
audit = F.audit(F.ArraySource(frames), os.path.basename(CLIP))

print('best channel      :', audit['best_channel'])
print('blue separability :', round(audit['separability']['blue'], 1), 'sigma')
print('duplicate frames  :', f"{audit['duplicate_fraction']:.1%}")
print('IR night mode     :', audit['ir_night_mode'])
for w in audit['warnings']:
    print('!!', w)
"""))

# ---------------------------------------------------------------- 3. config
cells.append(md("""
## 3. Configure and mask

The top-hat kernel scales with resolution. At 4K the pipeline also drops the
background-median refresh rate and estimates camera shake at half resolution —
without that a 3840×2160 clip runs at about a second per frame.

The mask must exclude the burned-in overlays and anything that produces
streak-shaped signal which is not an ember: **wind-blown vegetation** above all
(real motion, so neither background subtraction nor stabilisation removes it),
plus specular metal. Draw your own with `python make_mask.py <clip>`.
"""))
cells.append(code("""
cfg = Config()
cfg.track.fps = fps
cfg.detect.tophat_ksize = max(5, int(round(15 * W / 1920)) | 1)

if W * H > 2_500_000:
    cfg.detect.bg_refresh = 4     # ~490 ms/frame median at 4K, background is static
    print(f'large frame: background refresh every {cfg.detect.bg_refresh} frames')

if audit['ir_night_mode']:
    cfg.detect.channel = 'gray'

MASK = (f'gs://{BUCKET}/masks/left_front.json' if BUCKET
        else 'masks/left_front_starter.json')
if not (gcsio.exists(MASK) if gcsio.is_gcs(MASK) else os.path.exists(MASK)):
    MASK = 'masks/left_front_starter.json'      # bundled fallback

if os.path.exists(MASK) or gcsio.is_gcs(MASK):
    roi = F.mask_from_spec((H, W), F.load_mask_spec(MASK))
    print(f'{MASK}: {(roi == 0).mean():.1%} of the frame excluded')
else:
    roi = F.make_roi_mask((H, W), **F.LEFT_FRONT_OVERLAYS)
    print(f'burned-in overlays only: {(roi == 0).mean():.1%} excluded')

plt.figure(figsize=(12, 7))
plt.axis('off')
plt.imshow(cv2.cvtColor(F.overlay_mask(frames[0], roi), cv2.COLOR_BGR2RGB))
plt.title('red = excluded')
plt.show()
"""))

# ---------------------------------------------------------------- 4. mask cost
cells.append(md("""
### What is this mask costing?

Percent-of-frame is the wrong number. What matters is how many real tracks the
mask removes, and how much of the **ground plane** survives — saltation
contacts happen on the driveway, and a mask that clips a corner of it biases
which hops you observe. A biased sample of hop lengths is worse than a smaller
unbiased one.
"""))
cells.append(code("""
cost = R.mask_cost(frames[:60], roi, cfg,
                   ground_plane_poly=F.DRIVEWAY_LEFT_FRONT)
R.print_mask_cost(cost)
"""))

# ---------------------------------------------------------------- 5. detect
cells.append(md("""
## 4. Detect and track

The detector over-detects on purpose. Precision comes from the acceptance rule:
a candidate is kept only if it joins a track of ≥3 frames with a smooth heading,
consistent brightness and a stable speed.
"""))
cells.append(code("""
import time

t0 = time.time()
dets = D.detect_source(F.ArraySource(frames), cfg.detect, roi)
n = sum(len(v) for v in dets.values())
print(f'{n} candidates over {len(dets)} frames '
      f'({n / max(len(dets), 1):.1f} per frame) in {time.time() - t0:.0f}s')

ratio, diag = T.estimate_step_ratio(dets, cfg.track, return_diagnostics=True)
print(f'step_ratio {ratio:.2f} from {diag["n_pairs"]} pairs, '
      f'sharpness {diag["sharpness"]:.1f}')
if diag['sharpness'] < 2:
    print('!! unreliable -- measure the camera exposure and set cfg.track.exposure_s')

tracks = T.link(dets, cfg.track, step_ratio=ratio)
acc = T.accept(tracks, cfg.track)
rej = T.rejected(tracks, acc)
print(f'{len(tracks)} tracks -> {len(acc)} accepted, {len(rej)} rejected')
print('(keep the rejected ones -- they are labelled hard negatives)')
"""))

cells.append(md("""
### Look at it

Most bugs in this pipeline are obvious in one frame and invisible in every
metric. Look before you trust a number.
"""))
cells.append(code("""
by_frame = {}
for t in acc:
    for d in t.dets:
        by_frame.setdefault(d.frame, []).append(d)

if by_frame:
    k = max(by_frame, key=lambda i: len(by_frame[i]))
    plt.figure(figsize=(14, 8))
    plt.axis('off')
    plt.imshow(cv2.cvtColor(D.draw(frames[k], by_frame[k]), cv2.COLOR_BGR2RGB))
    plt.title(f'frame {k}: {len(by_frame[k])} accepted detections')
    plt.show()
else:
    print('no accepted tracks -- check the mask and the audit warnings')
"""))

cells.append(md("""
### Where the false positives live

Red is the density of *rejected* candidates; green dots are accepted tracks.
Mask where red is hot and green is absent. Where a region has both, leave it in
and let the tracker do the work — this is how to draw a mask from evidence
rather than from intuition about a still frame.
"""))
cells.append(code("""
heat = R.fp_heatmap(rej, acc, frames[len(frames) // 2])
plt.figure(figsize=(15, 9))
plt.axis('off')
plt.imshow(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
plt.show()
"""))

cells.append(md("""
### Physics check — free, and needs no labels

Fit gravity + drag to each track. An ID switch between two different embers
cannot be fit by one parabola and shows up as a large residual. Sort by it and
inspect the worst.
"""))
cells.append(code("""
rows = T.summarize(acc, cfg.track)
if rows:
    df = pd.DataFrame(rows).sort_values('ballistic_rms_px', ascending=False)
    print(f'median residual {df.ballistic_rms_px.median():.2f} px, '
          f'p90 {df.ballistic_rms_px.quantile(0.9):.2f} px')
    cols = ['track_id', 'n_det', 'mean_step_px', 'mean_streak_L',
            'heading_deg', 'ballistic_rms_px']
    display(df.head(10)[cols])
else:
    print('no tracks to summarise')
"""))

# ---------------------------------------------------------------- 6. review
cells.append(md("""
## 5. Verify by contact sheet

One decision per *track*, not per box — roughly 25× faster than annotating. A
real firebrand sits still in the centre of every crop while the background
slides past; a false positive is static texture in all of them.
"""))
cells.append(code("""
sheets = R.make_contact_sheets(acc, frames, 'review/')
print(f'{len(sheets)} sheets written to review/')

if sheets:
    plt.figure(figsize=(15, 18))
    plt.axis('off')
    plt.imshow(cv2.cvtColor(cv2.imread(str(sheets[0])), cv2.COLOR_BGR2RGB))
    plt.show()
"""))
cells.append(code("""
REJECT = []      # <-- the track IDs that are NOT firebrands

verified, dropped = R.apply_rejections(acc, REJECT)
negatives = rej + dropped
R.save_review('review/review.json', verified, negatives, {'clip': CLIP})
print(f'{len(verified)} verified, {len(negatives)} negatives')
"""))

# ---------------------------------------------------------------- 7. synth
cells.append(md("""
## 6. Synthetic training data

Verified tracks are not a large training set — they are a *parameter estimate*
for a generator that can make an unlimited one. A firebrand is a moving point
emitter integrated over the exposure, so the rendering uses the same forward
model as the real thing rather than an approximation of it.

Sampling from your measured distributions instead of uniforms is the step people
skip, and it is the usual reason synthetic pipelines fail to transfer.
"""))
cells.append(code("""
if verified:
    cfg.synth = S.fit_config_to_tracks(T.summarize(verified, cfg.track),
                                       cfg.synth, step_ratio=ratio)
    print('fitted speed_px     :', tuple(round(v, 1) for v in cfg.synth.speed_px))
    print('fitted peak_srgb    :', tuple(round(v, 1) for v in cfg.synth.peak_srgb))
    print('fitted exposure_duty:', round(cfg.synth.exposure_duty, 3))
else:
    print('no verified tracks yet -- using default synthesis parameters')

# real background frames carry the camera's own noise and flicker for free
bg_frames = frames[:40]

N_SYNTH_FRAMES = 200          # raise for a bigger training set
X, Y, sframes, sgt = DS.build_from_synthetic(bg_frames, N_SYNTH_FRAMES, cfg,
                                             seed=0, roi_mask=roi)
print(DS.tile_stats(X, Y))

os.makedirs('data', exist_ok=True)
DS.save_tiles('data/tiles.npz', X, Y)
print('saved data/tiles.npz')
"""))

cells.append(md("**Always look at a few samples before a long training run.**"))
cells.append(code("""
k = int(np.argmax(Y.reshape(len(Y), -1).max(1)))
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for c in range(3):
    ax[c].imshow(X[k][c], cmap='gray', vmax=40)
    ax[c].set_title(f'residual t{c - 1:+d}')
    ax[c].axis('off')
ax[3].imshow(Y[k], cmap='magma')
ax[3].set_title('target mask')
ax[3].axis('off')
plt.show()
"""))

# ---------------------------------------------------------------- 8. train
cells.append(md("""
## 7. Train

Two decisions do most of the work, and neither is the architecture.

**The input is time, not colour** — residual frames at t−1, t, t+1 as channels.
A network given one frame is being asked "is this bright thing static?" from an
image that cannot answer it.

**The output is a mask, not a box** — regressing four coordinates to sub-pixel
precision from a 7 px object is unstable, and the mask gives you streak length
and orientation directly, which the physics needs anyway.

Watch `val_f1`, not loss. On a 1-in-1500 class the loss is dominated by
background and keeps improving while detection quality does not.
"""))
cells.append(code("""
import torch
from firebrand import train as TR

print('cuda:', torch.cuda.is_available())

cfg.train.epochs = 30
cfg.train.batch_size = 16

model, hist = TR.train(X, Y, cfg.train, out_dir='runs/streak')
print('parameters:', model.n_params)
"""))
cells.append(code("""
h = pd.DataFrame(hist)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(h.epoch, h.train_loss, label='train')
ax[0].plot(h.epoch, h.val_loss, label='val')
ax[0].set_title('loss (do not judge the model by this)')
ax[0].set_xlabel('epoch')
ax[0].legend()

ax[1].plot(h.epoch, h.val_f1, label='F1')
ax[1].plot(h.epoch, h.val_precision, '--', label='precision')
ax[1].plot(h.epoch, h.val_recall, '--', label='recall')
ax[1].set_title('pixel metrics (judge it by these)')
ax[1].set_xlabel('epoch')
ax[1].legend()

plt.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------- 9. compare
cells.append(md("""
## 8. Does it beat the classical detector?

The model replaces only the thresholding stage — connected components, streak
fitting, linking and acceptance are the same code — so the comparison is on
identical terms. That is the only way to know whether training helped.
"""))
cells.append(code("""
from firebrand import infer as I

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

test_frames, test_gt = S.make_clip(bg_frames, 60, cfg.synth, seed=999, roi_mask=roi)
gtf = E.filter_gt(test_gt, min_L=cfg.detect.min_length, min_peak_srgb=25.0)

sweep = I.sweep_threshold(model, F.ArraySource(test_frames), cfg, gtf, roi,
                          device=dev)
display(pd.DataFrame(sweep)[['threshold', 'precision', 'recall', 'f1',
                             'mean_loc_err_px']])
"""))
cells.append(code("""
best_th = max(sweep, key=lambda r: r['f1'])['threshold']
print('operating threshold:', best_th)

m_dets, m_tracks = I.run_clip(model, F.ArraySource(test_frames), cfg, roi,
                              best_th, dev)

c_dets = D.detect_source(F.ArraySource(test_frames), cfg.detect, roi)
c_acc = T.accept(
    T.link(c_dets, cfg.track,
           step_ratio=T.estimate_step_ratio(c_dets, cfg.track)),
    cfg.track)

for name, dd, tt in [('classical', c_dets, c_acc), ('learned', m_dets, m_tracks)]:
    E.print_report(E.report(dd, tt, gtf), name)
"""))

# ---------------------------------------------------------------- 10. next
cells.append(md("""
### Save the results back to the bucket

The Workbench disk disappears when you delete the instance. The bucket does
not. Anything you want to keep goes to `gs://`.
"""))
cells.append(code("""
if BUCKET:
    import shutil
    stage = 'work_local'
    os.makedirs(stage, exist_ok=True)
    for src_path in ['review', 'data', 'runs']:
        if os.path.isdir(src_path):
            shutil.copytree(src_path, os.path.join(stage, src_path),
                            dirs_exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(stage, 'tracks.csv'), index=False)
    n = gcsio.upload_dir(stage, OUT)
    print(f'uploaded {n} files to {OUT}')
else:
    print('BUCKET not set -- results stay on local disk')
"""))

cells.append(md("""
### Before you close this tab

**Stop the instance.** A running T4 Workbench costs about $0.55/hour, roughly
$400/month if it is left on. A stopped one costs only its disk, about $15/month.

Idle shutdown is set to 30 minutes by default in `cloud/workbench.sh`, but it
watches *kernel activity* rather than CPU, so do not rely on it as your only
guard. From your laptop:

    bash cloud/workbench.sh stop
"""))
cells.append(code("""
# Or stop it from inside the notebook -- this kills the kernel, which is the point.
# !gcloud compute instances stop $(hostname) --zone=$(curl -s -H Metadata-Flavor:Google \
#     http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4) --quiet
print('remember to stop the instance when you are done')
"""))

cells.append(md("""
## 9. Next: pixels to metres

The pavers are a regular grid of known-size rectangles lying in the ground
plane, so `cv2.findHomography` gives a ground-plane calibration with no
checkerboard. Undistort the fisheye first — the straight paver joints also let
you self-calibrate the distortion from line-straightness alone.

Saltation contacts happen *on* that plane, so impacts and rebounds convert to
real metres exactly. With one camera the hops between need an assumption:
constrain each hop to a vertical plane containing the local wind direction from
your WindNinja run, anchored at the two contact points.

Report hop lengths (exact) separately from apex heights (assumption-dependent).

Per hop, log: length, apex height, impact and rebound angle, restitution
e = v_out / v_in, residence time, and integrated brightness as a size proxy.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = pathlib.Path('firebrand-kit/notebooks/Firebrand_Colab.ipynb')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))

# --- validate the way Jupyter actually reads it: join with "" -------------
bad = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])                      # <- no separator
    clean = "\n".join(l for l in src.split("\n")
                      if not l.strip().startswith(("!", "%")))
    try:
        ast.parse(clean)
    except SyntaxError as e:
        bad += 1
        print(f"cell {i}: {e.msg} (line {e.lineno})")
print(f"{len(nb['cells'])} cells, {bad} syntax errors, {out.stat().st_size} bytes")
