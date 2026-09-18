# firebrand-kit

Detecting and tracking firebrands in single-view security footage, for
saltation trajectory measurement.

Built for the UC Berkeley Fire Research Lab firebrand project (Ranch Top Rd,
Pasadena, 2025-01-07 Eaton Fire footage). Single camera, no overlapping views.

---

## The one idea

**A firebrand is defined by its motion, not its appearance.**

In a single frame, a 7-pixel glowing ember and a 7-pixel sunlit crack in the
concrete are the same object: a small, bright, elongated thing. No appearance
feature separates them, so no per-frame threshold can, and a detector trained
on per-frame thresholds inherits every mistake the threshold made.

So this pipeline runs a deliberately **over-sensitive** per-frame detector and
lets **temporal coherence** decide what was real. A candidate becomes a
firebrand only if it joins a track of three or more frames with a smooth
heading and consistent brightness. Static texture never does; embers always do.

That inversion — the tracker is the detector — is what makes the auto-labels
trustworthy enough to train on, and it is the difference from the first
attempt, where OpenCV thresholds generated labels that YOLO then learned to
imitate, mistakes included.

---

## Measured results

Benchmark: 50 frames of synthetic firebrands composited additively (in linear
light) onto a real frame from your camera, with 6% brightness flicker and 0.8 px
of camera shake. Ground truth exact by construction.

```
                              raw detector          after tracking       tracks
shipped defaults          P=1.00  R=0.75         P=1.00  R=0.45      recall 0.62, frag 1.12, 0 ID switches
```

Reproduce with `python tests/benchmark.py --ablate --channels`.

### What each piece is worth

```
  no stabilisation          P=0.08     (2877 detections, 12x too many)
  no edge suppression       P=0.17     (1392 detections)
  neither                   P=0.06     (4128 detections)
  shipped defaults          P=1.00     ( 237 detections)
```

**Stabilisation is the single most important step after the channel choice, and
the easiest to leave out.** These cameras are mounted on houses during a wind
event. A *sub-pixel* tremor is enough to break rolling-median background
subtraction: when a high-contrast edge moves half a pixel, the median stops
cancelling it and it reappears in the residual as a bright, elongated,
streak-shaped object — indistinguishable from a firebrand by shape, and present
in every frame. Phase-correlation alignment plus a gradient-proportional
threshold surcharge (`edge_suppress`) removes them.

### Where the blue channel actually matters

The background is fire-glow orange (mean RGB 154, 87, 23); firebrands are
near-blackbody white (250, 241, 227). Single-frame separability, measured on
your frame, in background sigmas:

```
  red 3.6  |  grayscale 9.0  |  green 9.1  |  blue 19.7  |  blue-0.15*red 22.1
```

But most of that advantage is redundant once background subtraction is working.
Measured, rather than assumed:

```
  ember brightness      blue recall   gray recall      blue trkR   gray trkR
  bright (70-255)          0.73          0.72             0.42        0.42
  medium (35-90)           0.80          0.51             0.60        0.40
  faint  (20-55)           0.71          0.19             0.64        0.00
```

Blue and grayscale are equivalent on bright embers. Blue is the difference
between working and not working on faint ones — which is exactly the population
that matters here, because an ember is dimmest while it is on the ground losing
heat, and **ground contact is the saltation event you are trying to measure.**

---

## Running on Google Cloud

### On Windows, run the cloud/ scripts in Cloud Shell

Click the terminal icon (`>_`) in the Cloud Console toolbar. That is Cloud
Shell: a browser terminal with `gcloud` preinstalled and already authenticated
as your console account. No install, no `gcloud auth login`, no MSYS path
conversion, no component prompts — the entire class of Windows tooling problems
goes away.

Upload the kit with the `⋮` menu (it is ~1.5 MB), then:

```bash
unzip firebrand-kit.zip && cd firebrand-kit
PROJECT_ID=your-project bash cloud/setup.sh
```

Free, 50 hours/week, 5 GB persistent `$HOME`, 40-minute idle timeout, 12-hour
session cap, and `$HOME` is deleted after 120 days unused.

**Do not upload your clips through Cloud Shell** — 5 GB of home directory is not
the place to stage footage. Drag them into the bucket in the console instead, or
run `gcloud storage cp` from the laptop where the files already live.

The split that works: **Cloud Shell** for everything in `cloud/`, **the console**
for uploading footage, **your laptop** for the Python work — which never calls
`gcloud` at all.

---

Every path in the kit accepts `gs://bucket/key` wherever a local path works, so
the same commands run on your laptop and on a cloud VM.

```bash
bash cloud/setup.sh                    # project + bucket + budget alert (once)
bash cloud/upload.sh clips/            # push footage up (free, resumable)
bash cloud/workbench.sh create         # GPU notebook VM
bash cloud/workbench.sh url            # open JupyterLab
bash cloud/workbench.sh stop           # <- the command that decides your bill
```

Then in the notebook, one line:

```python
os.environ['FIREBRAND_BUCKET'] = 'your-bucket-name'
```

Or from any shell:

```bash
python run_pipeline.py gs://$BUCKET/clips/eaton_01.mp4 \
    --out gs://$BUCKET/work/eaton_01 --mask gs://$BUCKET/masks/left_front.json
```

Clips are downloaded to `~/.firebrand_cache` once and reused. Outputs are built
on local disk and mirrored up in one pass at the end — writing hundreds of small
objects one at a time is slow, and a run that dies halfway would otherwise leave
a prefix that looks complete but is not.

**Cost, in one line:** a running T4 Workbench is about $0.55/hour (~$400/month
if left on); stopped, it is about $15/month for its disk; 10 GB of storage is
20 cents. `bash cloud/stop.sh` stops everything hourly-billed in the project and
is safe to run whenever you are unsure. Budget alerts *email* you — they do not
stop anything. See the setup guide for the full walkthrough.

### Masks without a display

`make_mask.py` needs a window, and cloud VMs have none — it now detects that and
tells you the two ways through instead of crashing:

```bash
python make_mask.py CLIP --export-frame frame.png   # needs no display
```

That writes the frame plus a max-projection (showing where embers actually fly),
which you drop into the browser mask editor. Or just draw the mask on your
laptop and upload the JSON — masks are stored in normalized coordinates, so one
file works at any resolution and moves between machines unchanged.

---

## Install

```bash
pip install -r requirements.txt      # numpy, opencv, scipy, pandas
pip install torch                    # only needed for the learned model
```

`ffmpeg` must be on PATH for decoding and for the synthetic re-encode step.

---

## Stage 0 — check the install, before touching your data

```bash
python tests/benchmark.py --ablate --channels
```

Self-contained (uses the bundled sample frame), takes about a minute, and
reproduces every number above. If `shipped defaults` comes back at P = 1.00 with
0 ID switches, the install is good.

---

## Where your clips go

There is no upload step — everything runs on your own machine, on files already
on your disk. Paths in these examples like `clips/eaton_01.mp4` are placeholders
for *your* filenames.

Either make a folder inside the kit and copy clips into it:

```bash
mkdir clips          # then copy your .mp4 files in
ls clips             # confirm what they are actually called
```

or leave them where they are and pass the full path. On Windows in Git Bash,
the C: drive is `/c/`, and **paths containing spaces need quotes**:

```bash
python make_mask.py "/c/Users/Hayden Liu/Downloads/eaton footage/cam1.mp4" \
    --out masks/left_front.json
```

If a path is wrong, the error tells you the working directory, what the folder
actually contains, and lists video files it can find nearby.

Supported: whatever OpenCV can decode — `.mp4`, `.avi`, `.mov`, `.mkv`. Some
DVRs export `.dav` or H.265, which OpenCV often cannot read; convert first and
keep the quality high, since compression is what destroys these objects:

```bash
ffmpeg -i input.dav -c:v libx264 -crf 18 -an clips/cam1.mp4
```

---

## Stage 1 — draw the mask, once per camera angle

```bash
python make_mask.py clips/YOUR_CLIP.mp4 --out masks/left_front.json
```

Click to add points, ENTER to close a polygon, `s` to save. Press `p` for a
max-projection view, which shows where embers actually fly so you can see
whether a polygon is about to throw away the interesting region.

**Exclude glass and painted metal that reflects the sky.** A reflection is a
mirrored duplicate of a real firebrand: it double-counts flux and generates
trajectories that are physically impossible, which quietly poisons every
saltation statistic downstream. This is the one manual step nobody can do for
you — an earlier version of this kit shipped guessed polygons for the Left Front
camera, and rendered over the real frame they covered driveway instead of glass.
The two burned-in overlay boxes (timestamp, camera label) *are* measured and are
pre-loaded.

### What to draw polygons around

For the Ranch Top Rd 'Left Front' camera, in order of how much damage skipping
each one does. A starter mask is in `masks/left_front_starter.json` — treat it
as a first draft, not an answer.

**A. Vegetation — the hedge and lawn strip across the middle.** The important
one, and the least obvious. Wind-blown leaves are *real motion*, so neither
background subtraction nor stabilisation removes them: the median can't cancel
something that genuinely moves, and stabilisation only corrects the camera. A
swaying leaf edge catching the glow reads as a short bright streak, frame after
frame, from a slightly different place each time — which can even survive the
3-frame coherence rule, because the motion is smooth. This is the one confuser
in the scene that defeats every other defence in the pipeline.

**B. The left wall, light fixture, and the metal handrail.** The rail is
specular, permanently bright, and already streak-shaped. It's static, so the
median handles it in principle — but it sits at maximum contrast, which is
exactly where sub-pixel shake leaves the largest residual.

**C. The street beyond the hedge.** Optional. Embers there are far away and
below your resolution limit, and passing headlights are large moving bright
objects. Cutting it also speeds up every run.

**Do not exclude the driveway.** Pavement texture does generate false positives,
but that is what stabilisation, edge suppression and the 3-frame rule are for —
and the ground plane is where saltation contacts happen, so masking it removes
the measurement you are here to make.

### How much should you mask?

There is no target percentage, and aiming for one will mislead you — the
fraction of *pixels* removed says nothing about what you lost, because the
regions worth cutting are rarely the regions where the measurement happens.

Measured on the sample frame, the driveway is **57% of the frame**, and every
mask variant below keeps ~98% of it:

```
mask                        % frame cut   % of driveway kept   % of single-frame candidates cut
overlays only                      2.4%                98.7%                              6.6%
B + overlays                      18.9%                97.8%                             22.4%
B + A + overlays                  28.6%                97.7%                             36.3%
everything (starter mask)         44.2%                97.7%                             78.9%
```

So "44% of the frame" sounds alarming and costs 2% of the ground plane. The
number that moved was candidates, and almost all of that is region C — the
street holds 47% of the single-frame candidates on its own, because that is
where the airborne ember cast is. Those are real embers; they are just not
*saltating*, and they are past the distance where you can resolve a trajectory.

Two numbers to watch instead of a percentage:

- **accepted tracks lost** — the actual cost. Keep it low.
- **ground plane kept** — above ~95%. A mask that clips a corner of the driveway
  biases *which* hops you observe, and a biased sample of hop lengths is worse
  than a smaller unbiased one.

Both are reported by:

```bash
python run_pipeline.py clips/YOUR_CLIP.mp4 --out work/check \
    --mask masks/left_front_starter.json --check-mask
```

Rough guidance for this camera: **~19% (B + overlays) is a sane starting point.**
Add the vegetation (~29%) if the heatmap shows the hedge is hot. Only go to 44%
if you are strictly measuring ground contacts and are content to discard the
airborne population — and note that the shipped starter mask, run through
`--check-mask`, flags itself as removing about a quarter of accepted tracks.

### Better: let the data draw it

Every run writes `fp_heatmap.png` — rejected candidates as a red density map,
accepted tracks as green dots, both on your frame. Run once with `--no-mask`,
open it, and mask where the red is hot and the green is absent. Where a region
has both, leave it in and let the tracker do the work.

That ordering matters. Vegetation is a genuine judgement call: masking the hedge
throws away any ember that flies in front of it, and on a windy night that may
be a lot of them. The heatmap tells you which cost is larger *on your footage*
instead of making you guess from a still frame — which is how I put the
reflection polygons in the wrong place in the first version of this kit.

Mask specs are stored in normalized coordinates, so one mask works across
resolutions (the sample frame is 1917×1079, not 1920×1080 — an exact-size check
would have silently skipped the mask entirely).

---

## Stage 2 — auto-labels, no training

```bash
python run_pipeline.py clips/eaton_01.mp4 --out work/eaton_01 \
    --mask masks/left_front.json --overlay --max-frames 300
```

Start with `--max-frames 300`. Get one clip right before running forty.

Writes to `work/eaton_01/`:

| file | what it is |
|---|---|
| `audit.json` | **read this first** — IR mode, duplicate frames, channel separability |
| `tracks.csv` | one row per accepted track, with a ballistic-fit residual column |
| `detections.csv` | one row per detection: x, y, streak length, orientation, flux |
| `review/sheet_*.png` | contact sheets for verification |
| `labels/*.txt` | YOLO-format boxes, one file per frame |
| `overlay.mp4` | detections drawn on the clip |

**Look at `overlay.mp4` before anything else.** Most bugs in this pipeline are
obvious in one frame and invisible in every metric.

### Pre-flight checks that actually matter

`audit.json` reports three things that will silently corrupt your results:

- **IR night mode.** If the camera flips to monochrome, the blue channel stops
  helping and the pipeline falls back to grayscale (with the recall cost above).
- **Duplicate frames.** Many DVRs advertise 30 fps by repeating frames from a
  15 fps sensor. If the duplicate fraction is near 0.5, your true capture rate
  is half what the container says and **every velocity you measure is wrong by
  2x**.
- **Channel separability** on your actual clips, not on the one frame these
  defaults were tuned on.

### Exposure time

If you know the camera's exposure time, pass `--exposure 0.002`. Then streak
length converts directly to speed: **speed = L / t_exposure**, an instantaneous
velocity per detection from a single frame, before any tracking.

If you don't, `track.estimate_step_ratio` recovers the equivalent ratio from the
data by asking, for each streak, how far along its own direction the next frame's
detections lie. Note the ratio it returns is defined against *measured* streak
length, which includes PSF broadening, so it is self-consistent for prediction
but biased low as an estimate of `(1/fps)/exposure`. Getting the real exposure
from the camera is worth the effort — it is the only thing standing between you
and absolute speeds.

---

## Stage 3 — verification (a few hours of your time, total)

Drawing a box takes ~4 seconds. Saying yes/no to one already drawn takes ~1. And
a track of six detections is **one** decision instead of six — roughly a 25x
multiplier before you write any model code.

1. Open `review/sheet_*.png`. Each row is one track, shown as a strip of crops
   in time order. A real firebrand sits still in the middle of every cell while
   the background slides past; a false positive is static texture in all of them.
2. Note the IDs that are not firebrands.
3. Re-run with `--reject 12,47,51,88`.

Keep `negatives.csv`. Every rejected track is a **labelled hard negative** mined
from your own footage — a paver joint, a compression block, a reflection — and
those are worth far more to the model than random background crops, because they
are exactly the mistakes it would otherwise make.

At 24 tracks per sheet and ~30 s per sheet, 800 tracks is about half an hour.

---

## Stage 4 — synthetic training data

800 verified tracks are not a large training set. They are an excellent
*parameter estimate* for a generator that can produce an unlimited one.

A firebrand is a moving point emitter integrated over the exposure. That is not
an approximation — it is the same forward model, with parameters you sample,
which is why synthetic data works far better here than it usually does.

```python
from firebrand import Config, synth as S, dataset as DS
import pandas as pd

cfg  = Config()
rows = pd.read_csv("work/eaton_01/tracks.csv").to_dict("records")
cfg.synth = S.fit_config_to_tracks(rows, cfg.synth)     # match YOUR distributions

X, Y, frames, gt = DS.build_from_synthetic(background_frame, 400, cfg)
DS.save_tiles("data/tiles.npz", X, Y)
```

Three details decide whether the model transfers:

1. **Real backgrounds.** Composite onto frames from your own camera. Synthetic
   backgrounds lack your pavement texture, which is the exact confuser the model
   must learn to reject.
2. **Additive, in linear light.** A glowing object adds photons; it does not
   alpha-blend. Compositing in gamma space gives the wrong brightness profile
   and the model learns the wrong edge statistics.
3. **Re-encode through the same codec** (`reencode_h264=True`). Otherwise the
   model learns "real embers have compression ringing, synthetic ones don't" and
   collapses on real footage.

And: sample from your *measured* distributions, not from uniforms. That is the
most common way synthetic-data pipelines quietly fail.

---

## Stage 5 — the model

### Smoke-test first

```bash
python tests/smoke_train.py          # ~15 s on a GPU, ~60 s on CPU
```

Runs the entire torch path at toy scale: 60 tiny tiles, model construction, a
loss-decreases check, 2 epochs, checkpoint save and reload, sliding-window
inference, and tracks out the far end. Eight stages, each reporting PASS/FAIL
independently so a failure names the broken piece instead of handing you a
traceback.

Run it on every new machine **before** starting a long run. A Workbench
instance bills from the moment it starts, so finding a broken import 20 minutes
into training costs money and momentum.

It also matters for a specific reason: PyTorch could not be installed in the
environment where this kit was written, so `model.py`, `train.py` and `infer.py`
have never actually executed. Their tensor shapes and parameter count are
verified arithmetically and the data feeding them is tested end-to-end, but the
first real execution is on your machine.


```python
from firebrand import train as TR
model, hist = TR.train(X, Y, cfg.train, out_dir="runs/streak")
```

Two decisions do most of the work, and neither is the architecture.

**The input is time, not colour.** Channels are residual frames at t−1, t, t+1
rather than R, G, B. A network given one frame is being asked "is this bright
thing static?" from an image that cannot answer it — the question isn't hard, it
is underdetermined. This is a dataloader change and it is the single biggest
lever in the pipeline.

**The output is a mask, not a box.** Regressing four box coordinates to
sub-pixel precision from a 7-pixel object is numerically unstable, and IoU ≥ 0.5
is meaningless at that size — a one-pixel error already fails it. A dense mask
is well-conditioned, needs less data, and gives you streak length and
orientation directly, which you need for the physics anyway.

The network is a 3-level U-Net, ~0.48M parameters at `base_ch=16`. That is
plenty: this is a low-level texture task, not a semantic one. Bigger backbones
overfit a small real set and train slower for no gain.

**Colab:** ~8k tiles of 256 px at batch 16 fits a free-tier T4 and runs 30
epochs in 15–25 minutes. On CPU, drop to `tile=192, epochs=12`.

Watch `val_f1`, not loss — loss on a 1-in-1500 class is dominated by background
and will keep improving while detection quality doesn't. `pos_weight` is set
automatically from the actual imbalance in your tiles (~40 here); leaving it at
1 makes the model predict nothing while showing an excellent loss curve.

### If you have to use YOLO

The lab's README commits to YOLO. It can work, but not out of the box:

- **P2 head** (stride 4, e.g. `yolo11-p2.yaml`). Default `imgsz=640` letterboxes
  1920×1080 by 3×, turning a 7 px streak into 2.3 px — a quarter of one stride-8
  grid cell, below what the architecture can represent.
- **512×512 tiles at native resolution**, SAHI-style tiling at inference. Never
  resize the frame.
- `mosaic=0.0` — mosaic augmentation downsamples and erases objects this small.
- Feed the **3-frame residual stack** in place of RGB. Change the dataloader,
  not the architecture.

---

## Stage 6 — evaluation

```python
from firebrand import evaluate as E
E.print_report(E.report(dets, tracks, gt))
```

**Do not use mAP.** At 7 px, a one-pixel centroid error drops IoU below 0.5, so
`mAP@0.5` reports a genuinely good detector as a failure and you end up tuning
against quantisation noise.

Instead:

- Match on **centre distance ≤ 5 px**.
- Score **tracks**, not frames. A tracker that splits one ember into four scores
  fine on any per-frame metric while ruining your saltation statistics —
  `fragmentation` is the column to watch.
- **Ballistic residual** (`tracks.csv`) is a validation signal that needs no
  labels at all: fit gravity + drag to each track and report RMS residual. ID
  switches between two different embers can't be fit by one parabola and show up
  as large residuals. Most tracking projects don't have a free ground-truth
  proxy like this. Sort by it and inspect the worst.

---

## Stage 7 — pixels to metres

Not implemented here, but the pipeline outputs what it needs.

**The driveway pavers are your calibration target.** They are a regular grid of
known-size rectangles lying in the ground plane — measure a few on site and
`cv2.findHomography` gives you a ground-plane calibration with no checkerboard.
Undistort the fisheye first; the straight paver joints also let you
self-calibrate the distortion from line-straightness alone.

This matters because **saltation contacts happen on the ground plane**, so every
impact and rebound point converts to real metres exactly, with no depth
ambiguity. With one camera the hops in between need an assumption: constrain
each hop to a vertical plane containing the local wind direction (your WindNinja
run gives you that), anchored at the two ground-contact points you can locate
exactly.

Per-hop quantities to log: hop length, apex height, impact and rebound angles,
restitution *e* = v_out/v_in, surface residence time, and integrated brightness
as a size/temperature proxy.

---

## Layout

```
firebrand/
  config.py     every tunable, with the reasoning for each default
  frames.py     decode, frame sources, ROI masks, pre-flight audit
  detect.py     stabilise -> blue channel -> top-hat -> rolling median -> streaks
  track.py      linking, step-ratio self-calibration, acceptance, ballistic check
  review.py     contact sheets and rejection handling
  synth.py      physically-grounded streak rendering and clip generation
  dataset.py    residual stacks, tiling, augmentation (torch-free builder)
  model.py      TinyUNet + weighted BCE/Dice loss
  train.py      training loop
  infer.py      sliding-window inference -> detections -> tracks
  evaluate.py   distance matching, track metrics, detectability filtering
make_mask.py     click-to-draw exclusion mask editor
run_pipeline.py  CLI for stages 1-3
tests/benchmark.py  end-to-end benchmark and ablation
notebooks/Firebrand_Colab.ipynb
```

---

## Known limits

- **Recall after acceptance is ~0.45** on the benchmark, against ~0.75 raw. That
  is by design — requiring three coherent frames costs detections and buys
  precision — and it is the reason to train the model at all: the network sees
  three frames at once and can recover faint, short streaks the shape gate
  rejects.
- **Track stitching is off by default.** `track.stitch()` exists and rejoins
  fragments, but on the benchmark (fragmentation already 1.12) it lost more
  tracks than it merged. Turn it on if your fragmentation exceeds ~1.3.
- **The benchmark is synthetic.** It proves the mechanics work and makes the
  ablations honest. It does not predict field accuracy, because real ember
  brightness and shape distributions differ from the generator's priors. Fit the
  generator to verified real tracks before trusting the numbers.
- **Everything photometric was tuned on one frame** (Left Front, 00:05:36).
  Re-run the audit on your own clips.
