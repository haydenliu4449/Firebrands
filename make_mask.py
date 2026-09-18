#!/usr/bin/env python3
"""Draw the exclusion mask for a camera. Two minutes, once per camera angle.

    python make_mask.py clips/eaton_01.mp4 --out masks/left_front.json

Controls
    left click      add a point to the current polygon
    right click     undo the last point
    ENTER / n       close the current polygon and start a new one
    u               undo the last completed polygon
    p               toggle a max-projection view (shows where embers actually fly)
    s               save and quit
    q / ESC         quit without saving

Then use it:

    python run_pipeline.py clips/eaton_01.mp4 --out work/e1 --mask masks/left_front.json

WHAT TO EXCLUDE, in order of how much damage it does if you skip it:

  1. Glass and painted metal that reflects the sky. A reflection is a mirrored
     duplicate of a real firebrand -- it double-counts flux, and it produces
     trajectories that are physically impossible, which will quietly poison any
     saltation statistic you compute. This is the important one.
  2. The burned-in timestamp and camera label. Permanent, maximum-brightness,
     streak-shaped objects. They are pre-loaded for you (press 'u' if you want
     to redraw them).
  3. Sky and anything above the roofline, if you only care about ground-level
     saltation. Optional -- excluding it speeds things up and removes distant
     ember cast you cannot resolve anyway.

DO NOT exclude the driveway. Pavement texture generates false positives, but
that is what stabilisation, edge suppression and the 3-frame rule are for --
and the ground plane is where saltation contacts happen, so masking it would
remove the measurement you are here to make.

If you have no display (a remote box, a container), use the headless helper
instead:

    from firebrand.frames import make_roi_mask, save_mask_spec, overlay_mask
    spec = {"normalized": True,
            "exclude_boxes": [...],
            "exclude_polys": [[(0.05, 0.20), (0.30, 0.18), ...]]}
    save_mask_spec("masks/left_front.json", spec)
    # then render overlay_mask(frame, mask_from_spec(frame.shape, spec)) to a
    # PNG and look at it before running anything long.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from firebrand import frames as F


def _has_display() -> bool:
    """cv2.imshow needs a window server. Cloud VMs and plain SSH have none, and
    the failure OpenCV gives is a cryptic abort rather than an explanation."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def load_frame(path, index=0, projection_frames=60):
    """Return (single frame, max-projection over the first N frames).

    The max projection is genuinely useful for mask drawing: it shows where
    embers actually travel in this scene, so you can see immediately whether a
    polygon you drew is about to throw away the interesting region.
    """
    p = Path(path)
    if p.is_dir():
        src = F.FrameDirSource(p)
    elif p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        f = cv2.imread(str(p))
        return f, f
    else:
        src = F.VideoSource(p)

    frames = []
    for i, f in enumerate(src):
        frames.append(f)
        if i >= projection_frames:
            break
    if not frames:
        raise SystemExit(f"no frames in {path}")
    idx = min(index, len(frames) - 1)
    proj = np.max(np.stack(frames), axis=0)
    return frames[idx], proj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", help="video, frame directory, or a single image")
    ap.add_argument("--out", default="mask.json")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280, help="display width")
    ap.add_argument("--export-frame", metavar="PNG",
                    help="write a frame + max-projection PNG and exit "
                         "(for drawing the mask elsewhere; needs no display)")
    ap.add_argument("--blank", action="store_true",
                    help="start empty instead of with the burned-in overlays")
    args = ap.parse_args()

    frame, proj = load_frame(args.clip, args.frame)
    H, W = frame.shape[:2]

    if args.export_frame:
        cv2.imwrite(args.export_frame, frame)
        base = Path(args.export_frame).with_name(
            Path(args.export_frame).stem + "_maxproj.png")
        cv2.imwrite(str(base), proj)
        print(f"wrote {args.export_frame} and {base}")
        print("Open the browser mask editor, drop these in, draw your polygons,")
        print("and save the JSON it gives you. No display needed on this machine.")
        return

    if not _has_display():
        raise SystemExit(
            "No display available -- cv2.imshow cannot open a window here.\n"
            "This is normal on a cloud VM or over plain SSH.\n\n"
            "Export the frames instead and draw the mask in a browser:\n"
            f"    python make_mask.py {args.clip} --export-frame frame.png\n\n"
            "Then use the browser mask editor (see README), or draw the polygons\n"
            "on your laptop where a display exists and upload the JSON:\n"
            "    gcloud storage cp masks/left_front.json gs://YOUR_BUCKET/masks/")
    scale = args.width / W
    disp_size = (args.width, int(H * scale))

    polys: list[list[tuple[float, float]]] = []
    boxes: list[tuple[float, float, float, float]] = []
    if not args.blank:
        boxes = list(F.LEFT_FRONT_OVERLAYS["exclude_boxes"])
        print("pre-loaded the two burned-in overlay boxes; press 'u' to drop the last item")

    current: list[tuple[int, int]] = []
    show_proj = [False]

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    win = "mask  |  click=add  right=undo pt  ENTER=close poly  u=undo poly  p=projection  s=save  q=quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        base = proj if show_proj[0] else frame
        canvas = cv2.resize(base, disp_size)

        spec = {"normalized": True, "exclude_boxes": boxes, "exclude_polys": polys}
        mask = F.mask_from_spec((disp_size[1], disp_size[0]), spec)
        canvas = F.overlay_mask(canvas, mask)

        for i, pt in enumerate(current):
            cv2.circle(canvas, pt, 4, (0, 255, 255), -1)
            if i:
                cv2.line(canvas, current[i - 1], pt, (0, 255, 255), 1)
        if len(current) > 2:
            cv2.line(canvas, current[-1], current[0], (0, 200, 200), 1, cv2.LINE_AA)

        hud = (f"{len(polys)} polygons, {len(boxes)} boxes, "
               f"{len(current)} points in progress"
               f"{'   [MAX PROJECTION]' if show_proj[0] else ''}")
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(canvas, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.imshow(win, canvas)

        k = cv2.waitKey(20) & 0xFF
        if k in (13, ord("n")) and len(current) >= 3:
            polys.append([(x / disp_size[0], y / disp_size[1]) for x, y in current])
            current.clear()
        elif k == ord("u"):
            if polys:
                polys.pop()
            elif boxes:
                boxes.pop()
        elif k == ord("p"):
            show_proj[0] = not show_proj[0]
        elif k == ord("s"):
            if len(current) >= 3:
                polys.append([(x / disp_size[0], y / disp_size[1]) for x, y in current])
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            F.save_mask_spec(out, {"normalized": True,
                                   "exclude_boxes": boxes,
                                   "exclude_polys": polys})
            full = F.mask_from_spec((H, W), {"normalized": True,
                                             "exclude_boxes": boxes,
                                             "exclude_polys": polys})
            png = out.with_suffix(".png")
            cv2.imwrite(str(png), F.overlay_mask(frame, full))
            print(f"saved {out}  ({(full == 0).mean():.1%} of the frame excluded)")
            print(f"preview {png}")
            break
        elif k in (ord("q"), 27):
            print("quit without saving")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, IOError) as e:
        # A path problem is a user mistake, not a crash. Print the diagnosis,
        # not a stack trace pointing into library code.
        import sys
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1)
