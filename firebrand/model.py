"""The network: a small U-Net that segments streaks from a stack of residuals.

Two decisions do most of the work here, and neither is about the architecture.

1. THE INPUT IS TIME, NOT COLOUR.
   Channels are residual frames at t-1, t, t+1 rather than R, G, B. A network
   given a single frame is being asked "is this bright thing static?" from an
   image that cannot answer it -- the question is not hard, it is
   underdetermined. Three frames make it trivial. This is a dataloader change,
   and it is the single biggest lever in the whole pipeline.

2. THE OUTPUT IS A MASK, NOT A BOX.
   Regressing four box coordinates to sub-pixel precision from a 7-pixel object
   is numerically unstable, and the standard detection metric (IoU >= 0.5) is
   meaningless at that size -- a one-pixel error already fails it. A dense mask
   is a well-conditioned target, needs less data, and hands you the streak's
   length and orientation directly, which you need for the physics anyway.

Size: base_ch=16 gives ~0.5M parameters. That is plenty. This is a low-level
texture task, not a semantic one; there is no object category to learn, only
"additive line that moved". Bigger backbones overfit your small real set and
train slower for no gain.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """3-level U-Net. Input (B, in_ch, H, W) -> logits (B, 1, H, W).

    H and W must be divisible by 8.
    """

    def __init__(self, in_ch=3, base=16):
        super().__init__()
        b = base
        self.e1 = ConvBlock(in_ch, b)
        self.e2 = ConvBlock(b, b * 2)
        self.e3 = ConvBlock(b * 2, b * 4)
        self.bott = ConvBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)

        self.u3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.d3 = ConvBlock(b * 8, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d2 = ConvBlock(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d1 = ConvBlock(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)                    # H
        e2 = self.e2(self.pool(e1))        # H/2
        e3 = self.e3(self.pool(e2))        # H/4
        bo = self.bott(self.pool(e3))      # H/8

        d3 = self.d3(torch.cat([self.u3(bo), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


class StreakLoss(nn.Module):
    """Weighted BCE + soft Dice.

    Firebrand pixels are on the order of 0.1% of a frame, so plain BCE is
    minimised by predicting all-zero -- the model will converge fast to a
    perfect-looking loss and detect nothing. `pos_weight` fixes the gradient
    imbalance; Dice additionally optimises overlap directly, which matters when
    the positive class is a handful of pixels wide.
    """

    def __init__(self, pos_weight=20.0, dice_weight=0.5):
        super().__init__()
        self.register_buffer("pw", torch.tensor(float(pos_weight)))
        self.dw = float(dice_weight)

    def forward(self, logits, target):
        bce = Fn.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pw)
        return bce + self.dw * dice_loss(logits, target)


def build(cfg, in_ch=None):
    return TinyUNet(in_ch=in_ch or cfg.n_frames_stack, base=cfg.base_ch)
