"""Segmentation losses plus HAND-based physical regularizers.

All losses take logits (B,1,H,W), labels (B,1,H,W) in {-1,0,1} and, for the
HAND terms, HAND in metres (B,1,H,W) with NaN where unknown.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from floodseg.data import IGNORE


def bce_dice(logits: torch.Tensor, y: torch.Tensor, pos_weight: float = 1.0, eps: float = 1.0) -> torch.Tensor:
    valid = y != IGNORE
    target = (y == 1).float()
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device), reduction="none"
    )
    bce = bce[valid].mean() if valid.any() else logits.sum() * 0
    p = torch.sigmoid(logits) * valid
    t = target * valid
    dice = 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return bce + dice


def hand_monotonicity(
    logits: torch.Tensor, hand: torch.Tensor, margin: float = 0.05, min_delta: float = 1.0, stride: int = 1
) -> torch.Tensor:
    """Penalize a pixel for being predicted wetter than a pixel `stride` cells away that sits lower above drainage.

    For pixel pairs (a, b) `stride` apart along rows and columns with HAND_a - HAND_b >= min_delta metres,
    add relu(p_a - p_b - margin). Averaged over qualifying pairs. Pairs with unknown HAND are skipped.
    Labels are not used: this is a physical prior on the prediction, applied to labelled and unlabelled
    pixels alike.

    HAND is a 30 m product resampled to 10 m, so with stride 1 most neighbour pairs lie inside one 30 m
    cell or differ by less than min_delta and the term sees few pairs; stride 3 matches the
    `metrics.plausibility` diagnostic. stride=1 is what the first sweep (sar_mono, sar_hand_mono) used.
    """
    p = torch.sigmoid(logits)
    total = logits.new_zeros(())
    count = logits.new_zeros(())
    for dim in (-1, -2):
        n = p.shape[dim] - stride
        if n <= 0:
            continue
        pa, pb = p.narrow(dim, 0, n), p.narrow(dim, stride, n)
        ha, hb = hand.narrow(dim, 0, n), hand.narrow(dim, stride, n)
        delta = torch.nan_to_num(ha - hb, nan=0.0)
        a_higher = delta >= min_delta
        b_higher = -delta >= min_delta
        total = total + (F.relu(pa - pb - margin) * a_higher).sum() + (F.relu(pb - pa - margin) * b_higher).sum()
        count = count + a_higher.sum() + b_higher.sum()
    return total / count.clamp_min(1)


def hand_ceiling(logits: torch.Tensor, hand: torch.Tensor, threshold: float = 15.0) -> torch.Tensor:
    """Mean flood probability on pixels far above the drainage network (HAND > threshold metres)."""
    high = torch.nan_to_num(hand, nan=0.0) > threshold
    if not high.any():
        return logits.sum() * 0
    return torch.sigmoid(logits)[high].mean()
