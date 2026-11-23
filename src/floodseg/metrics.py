"""Segmentation metrics and hydrologic plausibility diagnostics.

Segmentation metrics are accumulated as pixel confusion counts, reported both
pooled over all pixels ("total") and averaged per chip / per event, since the
Sen1Floods11 literature uses both and they can disagree sharply.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage

from floodseg.data import IGNORE


def confusion(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    valid = y != IGNORE
    p, t = pred[valid].astype(bool), y[valid] == 1
    return np.array([(p & t).sum(), (p & ~t).sum(), (~p & t).sum(), (~p & ~t).sum()], dtype=np.int64)


def scores(cm: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = (float(v) for v in cm)
    div = lambda a, b: a / b if b else float("nan")
    return {
        "iou": div(tp, tp + fp + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn),
        "accuracy": div(tp + tn, tp + fp + fn + tn),
    }


def plausibility(
    pred: np.ndarray, hand: np.ndarray, hand_threshold: float = 15.0, min_delta: float = 1.0, stride: int = 3
) -> dict[str, float]:
    """Physics diagnostics of a binary prediction. Needs no labels.

    Monotonicity compares pixels `stride` cells apart: HAND comes from a 30 m
    product resampled to 10 m, so immediate neighbours almost never differ by
    `min_delta` and a stride of 1 makes the metric blind.
    """
    pred = pred.astype(bool)
    out: dict[str, float] = {"flood_fraction": float(pred.mean())}
    known = np.isfinite(hand)
    flood_known = pred & known
    # share of predicted flood sitting far above drainage
    out["high_hand_flood_fraction"] = float((flood_known & (hand > hand_threshold)).sum() / max(flood_known.sum(), 1))
    # share of HAND-ordered neighbour pairs where the higher pixel is flooded but the lower is not
    violations = pairs = 0
    for axis in (0, 1):
        a = [slice(None)] * 2
        b = [slice(None)] * 2
        a[axis], b[axis] = slice(0, -stride), slice(stride, None)
        ha, hb, pa, pb = hand[tuple(a)], hand[tuple(b)], pred[tuple(a)], pred[tuple(b)]
        with np.errstate(invalid="ignore"):
            a_hi, b_hi = ha - hb >= min_delta, hb - ha >= min_delta
        pairs += int(a_hi.sum() + b_hi.sum())
        violations += int((a_hi & pa & ~pb).sum() + (b_hi & pb & ~pa).sum())
    out["monotonicity_violation_rate"] = violations / pairs if pairs else float("nan")
    # fragmentation: share of flood pixels in tiny isolated components (< 10 px = 0.1 ha)
    labels, n = ndimage.label(pred)
    if n:
        sizes = np.bincount(labels.ravel())[1:]
        out["small_component_fraction"] = float(sizes[sizes < 10].sum() / sizes.sum())
    else:
        out["small_component_fraction"] = 0.0
    out["n_components"] = float(n)
    return out


class Tracker:
    """Accumulates per-chip results and summarizes them pooled, per chip and per event."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, chip: str, event: str, pred: np.ndarray, y: np.ndarray, hand: np.ndarray) -> None:
        cm = confusion(pred, y)
        row = {"chip": chip, "event": event, "cm": cm, **scores(cm)}
        row["has_flood"] = bool(cm[0] + cm[2] > 0)
        row.update(plausibility(pred, hand))
        self.rows.append(row)

    def summary(self) -> dict:
        total_cm = sum(r["cm"] for r in self.rows)
        by_event: dict[str, list[dict]] = defaultdict(list)
        for r in self.rows:
            by_event[r["event"]].append(r)
        phys_keys = ("high_hand_flood_fraction", "monotonicity_violation_rate", "small_component_fraction")
        mean = lambda rows, k: float(np.nanmean([r[k] for r in rows])) if rows else float("nan")
        return {
            "total": scores(total_cm),
            "chip_mean_iou": mean(self.rows, "iou"),
            "event": {e: {**scores(sum(r["cm"] for r in rs)), "n_chips": len(rs)} for e, rs in sorted(by_event.items())},
            "physics": {k: mean(self.rows, k) for k in phys_keys},
            # false alarms on chips whose label has no flood at all
            "dry_chip_false_positive_rate": mean([r for r in self.rows if not r["has_flood"]], "flood_fraction"),
            "n_chips": len(self.rows),
        }

    def per_chip(self) -> list[dict]:
        return [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in self.rows]
