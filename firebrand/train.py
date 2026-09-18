"""Training loop.

Colab notes: a run on ~8k tiles of 256px with base_ch=16 fits comfortably in a
free-tier T4 session at batch_size 16, and takes roughly 15-25 minutes for 30
epochs. On CPU the same run is a few hours -- reduce tile to 192 and epochs to
12 if that is what you have.

The validation metric here is not loss. Loss on a 1-in-1500 class is dominated
by the background and will keep improving while detection quality does not.
Watch `val_f1` (pixel-level, at the operating threshold) and, more importantly,
run `evaluate.py` on a held-out clip -- pixel F1 still does not tell you whether
tracks come out intact.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import StreakTiles, suggested_pos_weight
from .model import StreakLoss, build


def split(X, Y, val_frac=0.15, seed=0):
    """Split tiles into train/val.

    NOTE: this splits tiles at random, which leaks between overlapping tiles
    from the same frame. It is fine for watching convergence, but for a number
    you would put in a paper, hold out whole *clips* instead and evaluate with
    evaluate.report().
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    v, t = idx[:n_val], idx[n_val:]
    return (X[t], Y[t]), (X[v], Y[v])


@torch.no_grad()
def pixel_scores(model, loader, device, thresh=0.5):
    model.eval()
    tp = fp = fn = 0
    loss_sum, n = 0.0, 0
    lossf = StreakLoss().to(device)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += float(lossf(logits, yb)) * len(xb)
        n += len(xb)
        p = (torch.sigmoid(logits) > thresh)
        t = yb > 0.2
        tp += int((p & t).sum()); fp += int((p & ~t).sum()); fn += int((~p & t).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(loss=loss_sum / max(n, 1), precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def train(X, Y, cfg: TrainConfig, out_dir="runs/streak", device=None,
          auto_pos_weight=True, log_every=1):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if auto_pos_weight:
        cfg.pos_weight = suggested_pos_weight(Y)

    (Xt, Yt), (Xv, Yv) = split(X, Y, cfg.val_frac, cfg.seed)
    dl_t = DataLoader(StreakTiles(Xt, Yt, train=True, seed=cfg.seed),
                      batch_size=cfg.batch_size, shuffle=True, num_workers=2,
                      drop_last=len(Xt) > cfg.batch_size)
    dl_v = DataLoader(StreakTiles(Xv, Yv, train=False), batch_size=cfg.batch_size)

    model = build(cfg).to(device)
    lossf = StreakLoss(cfg.pos_weight, cfg.dice_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = bool(cfg.amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"device={device}  params={model.n_params/1e6:.2f}M  "
          f"train={len(Xt)} val={len(Xv)}  pos_weight={cfg.pos_weight:.1f}")

    hist, best = [], -1.0
    for ep in range(cfg.epochs):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for xb, yb in dl_t:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = lossf(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += float(loss) * len(xb); n += len(xb)
        sched.step()

        v = pixel_scores(model, dl_v, device)
        rec = dict(epoch=ep, train_loss=run / max(n, 1),
                   secs=round(time.time() - t0, 1), **{f"val_{k}": v[k] for k in v})
        hist.append(rec)
        if ep % log_every == 0 or ep == cfg.epochs - 1:
            print(f"  ep{ep:3d}  train {rec['train_loss']:.4f}  "
                  f"val {rec['val_loss']:.4f}  P {v['precision']:.3f} "
                  f"R {v['recall']:.3f}  F1 {v['f1']:.3f}  ({rec['secs']}s)")

        if v["f1"] > best:
            best = v["f1"]
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "val_f1": best, "epoch": ep}, out / "best.pt")

    json.dump(hist, open(out / "history.json", "w"), indent=2)
    print(f"best val pixel-F1 {best:.3f} -> {out/'best.pt'}")
    return model, hist


def load(checkpoint, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = TrainConfig(**ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    model = build(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg
