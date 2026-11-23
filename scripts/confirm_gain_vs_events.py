"""Pre-registered test (PREREGISTRATIONS.md, 2026-11-12): at a fixed 500 training tiles, is the HAND gain larger with
20-40 training events than with all 159?

Uses only the new draws (seeds 3-5 by default) of runs/geoid_tm/e{20,40,159}. Per test event:
    contrast = mean over e20 and e40 draws of (sar_hand - sar_only)  -  mean over e159 draws of the same
One-sided Wilcoxon over test events (H1: contrast > 0) and a bootstrap 95% CI.

    python scripts/confirm_gain_vs_events.py --runs /scratch/$USER/floodseg/runs --name geoid_test_flood

--metric ap (post hoc, not pre-registered; added 2026-11-17 after the IoU test was null) uses per-event
average precision from scripts/eval_prob_hist.py instead of IoU at the 0.5 threshold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from external_report import load  # noqa: E402
from eval_prob_hist import average_precision  # noqa: E402
from eval_prob_hist import load as load_hist  # noqa: E402
from learning_curve_report import event_iou, training_events, unseen_only  # noqa: E402


def gain(path: Path, name: str, seeds: set[str], train_list: Path | None) -> pd.Series:
    ev = event_iou(unseen_only(load(path, name), train_list))
    ev = ev[ev.seed.isin(seeds)]
    t = ev.pivot_table(index=["event", "seed"], columns="variant", values="iou").dropna()
    found = set(t.index.get_level_values("seed"))
    if found != seeds:
        raise SystemExit(f"{path}: expected paired seeds {sorted(seeds)}, found {sorted(found)}")
    return (t.sar_hand - t.sar_only).groupby(level="event").mean()


def gain_ap(path: Path, name: str, seeds: set[str], train_list: Path | None) -> pd.Series:
    data = load_hist(path, name)
    rows = []
    for (variant, seed), (events, pos, neg) in data.items():
        if seed in seeds:
            rows += [{"variant": variant, "seed": seed, "event": e, "ap": average_precision(pos[i], neg[i])}
                     for i, e in enumerate(events) if pos[i].sum() > 0]
    ev = pd.DataFrame(rows)
    if train_list:
        ev = ev[~ev.event.isin(training_events(train_list))]
    t = ev.pivot_table(index=["event", "seed"], columns="variant", values="ap").dropna()
    found = set(t.index.get_level_values("seed"))
    if found != seeds:
        raise SystemExit(f"{path}: expected paired seeds {sorted(seeds)}, found {sorted(found)}")
    return (t.sar_hand - t.sar_only).groupby(level="event").mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--unseen-only", type=Path, default=None, help="see learning_curve_report.py (primary analysis uses it)")
    parser.add_argument("--seeds", nargs="+", default=["seed3", "seed4", "seed5"])
    parser.add_argument("--metric", choices=["iou", "ap"], default="iou", help="ap: post hoc, see docstring")
    args = parser.parse_args()
    fn = gain if args.metric == "iou" else gain_ap
    seeds = set(args.seeds)
    tm = args.runs / "geoid_tm"
    g = pd.DataFrame({f"e{n}": fn(tm / f"e{n}", args.name, seeds, args.unseen_only) for n in (20, 40, 159)}).dropna()
    contrast = (g.e20 + g.e40) / 2 - g.e159
    boot = np.random.default_rng(0).choice(contrast.to_numpy(), (10000, len(contrast))).mean(axis=1)
    print(f"test = {args.name}, metric {args.metric}, draws {sorted(seeds)}, {len(g)} test events")
    print("mean HAND gain:", "  ".join(f"{c} {g[c].mean():+.4f}" for c in g))
    print(f"contrast (e20/e40 minus e159): {contrast.mean():+.4f}  95% CI [{np.percentile(boot, 2.5):+.4f}, "
          f"{np.percentile(boot, 97.5):+.4f}]  one-sided Wilcoxon p = {wilcoxon(contrast, alternative='greater').pvalue:.4f}"
          f"  ({(contrast > 0).sum()}/{len(contrast)} events > 0)")


if __name__ == "__main__":
    main()
