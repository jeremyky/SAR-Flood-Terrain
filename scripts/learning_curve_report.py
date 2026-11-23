"""HAND gain vs number of training events (runs from scripts/make_event_subsets.py lists).

Expects run dirs <runs>/geoid_lc/e<size>/<variant>/seed<k> (seed k trained on subset draw k) and the
full-data runs <runs>/geoid/<variant>/seed<k>, each scored by scripts/eval_external.py under --name.
For every size: event IoU per (variant, draw), paired sar_hand - sar_only within each draw (same
training events), then averaged over draws; CIs by bootstrap over test events (ci) and over draws and test events (ci2,
pooled).

    python scripts/learning_curve_report.py --runs /scratch/$USER/floodseg/runs --name geoid_test_flood
    # tile-matched control (500 tiles per subset, no full-data point):
    python scripts/learning_curve_report.py --runs /scratch/$USER/floodseg/runs --name geoid_test_flood --curve-dir geoid_tm --full-dir ""
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


def training_events(train_list: Path) -> set[str]:
    """Activations (event ids) present in a training list; test events among them are not unseen."""
    return {c.rsplit("_", 1)[0] for c in train_list.read_text().split()}


def unseen_only(chips: pd.DataFrame, train_list: Path | None) -> pd.DataFrame:
    return chips if train_list is None else chips[~chips.event.isin(training_events(train_list))]


def shared_only(chips: pd.DataFrame, train_list: Path | None) -> pd.DataFrame:
    return chips if train_list is None else chips[chips.event.isin(training_events(train_list))]


def hier_boot(per_draw: pd.Series, rng: np.random.Generator, n: int = 10000) -> np.ndarray:
    """Bootstrap the mean paired delta resampling training draws and test events (two-way)."""
    m = per_draw.unstack("seed").to_numpy()  # events x draws
    out = np.empty(n)
    for b in range(n):
        i = rng.integers(0, m.shape[0], m.shape[0])
        j = rng.integers(0, m.shape[1], m.shape[1])
        out[b] = np.nanmean(np.nanmean(m[np.ix_(i, j)], axis=1))
    return out


def pooled_delta(chips: pd.DataFrame, base: str, variant: str, events: list[str], rng: np.random.Generator,
                 n: int = 5000) -> tuple[float, float, float, float, float]:
    """Pooled IoU (confusion counts summed over test events) per draw, averaged over draws, for base and
    variant; and the mean within-draw pooled delta with a two-way 95% CI resampling draws and test events
    (as scripts/hand_gate.compare and eval_geoid_protocol)."""
    c = chips[chips.event.isin(events)].groupby(["variant", "seed", "event"])[["tp", "fp", "fn"]].sum()
    seeds = sorted(set(c.loc[base].index.get_level_values("seed")) & set(c.loc[variant].index.get_level_values("seed")))
    arr = {v: np.stack([c.loc[(v, s)].reindex(events).fillna(0).to_numpy() for s in seeds]) for v in (base, variant)}
    f = lambda a: a[..., 0] / a.sum(-1)  # noqa: E731  (tp / (tp + fp + fn))
    pb, pv = f(arr[base].sum(1)), f(arr[variant].sum(1))  # per draw
    eidx = rng.integers(0, len(events), (n, len(events)))
    sidx = rng.integers(0, len(seeds), (n, len(seeds)))
    boot = np.array([(f(arr[variant][sidx[k]][:, eidx[k]].sum(1)) - f(arr[base][sidx[k]][:, eidx[k]].sum(1))).mean()
                     for k in range(n)])
    return pb.mean(), pv.mean(), (pv - pb).mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def event_iou(chips: pd.DataFrame) -> pd.DataFrame:
    ev = chips.groupby(["variant", "seed", "event"])[["tp", "fp", "fn"]].sum().reset_index()
    ev = ev[(ev.tp + ev.fn) > 0]
    ev["iou"] = ev.tp / (ev.tp + ev.fp + ev.fn)
    return ev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base", default="sar_only")
    parser.add_argument("--variant", default="sar_hand")
    parser.add_argument("--curve-dir", default="geoid_lc", help="subdirectory of --runs holding e<size>/ runs")
    parser.add_argument("--full-dir", default="geoid", help="full-data runs under --runs ('' to skip)")
    parser.add_argument("--full-events", type=int, default=159)
    parser.add_argument("--unseen-only", type=Path, default=None,
                        help="training list (e.g. <geoid_trainval>/splits/train.txt): drop test events whose activation is in it")
    parser.add_argument("--shared-only", type=Path, default=None,
                        help="training list: keep only test events whose activation is in it (complement of --unseen-only)")
    parser.add_argument("--draws", nargs="+", default=None,
                        help="restrict to these seeds/draws, e.g. seed0 seed1 seed2 (exploratory draws, excluding the confirmation draws 3-5)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    sources = {int(p.name[1:]): p for p in sorted((args.runs / args.curve_dir).glob("e*"))}
    if args.full_dir:
        sources.setdefault(args.full_events, args.runs / args.full_dir)
    rng = np.random.default_rng(0)
    rows = []
    for size, path in sorted(sources.items()):
        try:
            chips = shared_only(unseen_only(load(path, args.name), args.unseen_only), args.shared_only)
            if args.draws:
                chips = chips[chips.seed.isin(args.draws)]
            ev = event_iou(chips)
        except ValueError:  # no finished evaluations yet
            continue
        t = ev.pivot_table(index=["event", "seed"], columns="variant", values="iou")
        if args.base not in t or args.variant not in t:
            continue
        per_draw = (t[args.variant] - t[args.base]).dropna()
        d = per_draw.groupby("event").mean()  # average the paired delta over draws
        boot = rng.choice(d.to_numpy(), (10000, len(d))).mean(axis=1)
        hboot = hier_boot(per_draw, rng) if per_draw.index.get_level_values("seed").nunique() > 1 else boot
        draw_means = per_draw.groupby("seed").mean()
        chips = chips[chips.seed.isin(set(per_draw.index.get_level_values("seed")))]
        pooled = pooled_delta(chips, args.base, args.variant, sorted(d.index), rng)
        rows.append({
            "train_events": size, "draws": per_draw.index.get_level_values("seed").nunique(), "test_events": len(d),
            f"iou_{args.base}": t[args.base].groupby("event").mean().mean(),
            f"iou_{args.variant}": t[args.variant].groupby("event").mean().mean(),
            "delta": d.mean(), "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
            "ci2_lo": np.percentile(hboot, 2.5), "ci2_hi": np.percentile(hboot, 97.5),
            "wilcoxon_p": wilcoxon(d).pvalue, "improved": f"{(d > 0).sum()}/{len(d)}",
            "pooled_base": pooled[0], "pooled_variant": pooled[1], "pooled_delta": pooled[2],
            "pooled_ci_lo": pooled[3], "pooled_ci_hi": pooled[4],
            "delta_by_draw": " ".join(f"{v:+.3f}" for v in draw_means),
        })
    table = pd.DataFrame(rows)
    print(f"HAND gain vs number of training events, test = {args.name}"
          + (" (unseen activations only)" if args.unseen_only else "")
          + (" (shared activations only)" if args.shared_only else "")
          + "\nci = bootstrap over test events; ci2 = two-way bootstrap over draws and test events;"
          " pooled = IoU from confusion counts summed over test events, per draw, then averaged over draws;"
          " pooled CI two-way")
    print(table.round(3).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
