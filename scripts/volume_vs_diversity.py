"""Does the HAND gain shrink with the number of training events or with the number of training tiles?

In the learning curve (geoid_lc) both grow together. The tile-matched control (geoid_tm) holds tiles
at 500 while the number of events varies, so pooling the two separates them. Each training draw is
one point: its tile count and event count (read from the run's training list) and its mean over test
events of the paired delta (sar_hand - sar_only, same draw). Fit

    y = b0 + b_events * log2(events) + b_tiles * log2(tiles)

and get 95% CIs by a two-way bootstrap that resamples training draws and test events. The same fit is
reported for SAR-only and SAR + HAND IoU, to show what each kind of data buys the baseline.

    python scripts/volume_vs_diversity.py --runs /scratch/$USER/floodseg/runs --name geoid_test_flood
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from external_report import load  # noqa: E402
from learning_curve_report import event_iou, unseen_only  # noqa: E402


def training_set(run: Path) -> tuple[int, int]:
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    name = cfg.get("split", {}).get("train_list") or "train"
    chips = (Path(cfg["data"]["root"]) / "splits" / f"{name}.txt").read_text().split()
    return len(chips), len({c.rsplit("_", 1)[0] for c in chips})


def points(runs: Path, name: str, dirs: list[str], train_list: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """meta: one row per training draw; per_event: (draw x test event) table of iou_base, iou_hand."""
    meta, frames = [], []
    for d in dirs:
        for path in _sources(runs / d):
            try:
                ev = event_iou(unseen_only(load(path, name), train_list))
            except ValueError:
                continue
            t = ev.pivot_table(index=["seed", "event"], columns="variant", values="iou")
            if not {"sar_only", "sar_hand"} <= set(t.columns):
                continue
            for seed, g in t.dropna().groupby(level="seed"):
                tiles, events = training_set(path / "sar_hand" / seed)
                assert training_set(path / "sar_only" / seed) == (tiles, events), f"unpaired draw {path}/{seed}"
                key = f"{path.relative_to(runs)}/{seed}"
                meta.append({"draw": key, "source": d, "events": events, "tiles": tiles})
                g = g.droplevel("seed")
                frames.append(pd.DataFrame({"draw": key, "event": g.index, "base": g.sar_only.values, "hand": g.sar_hand.values}))
    return pd.DataFrame(meta), pd.concat(frames)


def _sources(path: Path) -> list[Path]:
    subs = sorted(path.glob("e*"))
    return subs if subs else [path]


def fit(meta: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(meta)), np.log2(meta.events), np.log2(meta.tiles)])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dirs", nargs="+", default=["geoid_lc", "geoid_tm", "geoid"])
    parser.add_argument("--boot", type=int, default=5000)
    parser.add_argument("--min-events", type=int, default=0, help="drop draws with fewer training events (sensitivity)")
    parser.add_argument("--unseen-only", type=Path, default=None, help="see learning_curve_report.py")
    parser.add_argument("--exclude-draws", nargs="*", default=[],
                        help="draw keys to drop, e.g. geoid_tm/e20/seed3 (the pre-registered confirmation draws)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    meta, per_event = points(args.runs, args.name, args.dirs, args.unseen_only)
    meta = meta[(meta.events >= args.min_events) & ~meta.draw.isin(args.exclude_draws)].reset_index(drop=True)
    per_event = per_event[per_event.draw.isin(meta.draw)]
    per_event = per_event[per_event.event.isin(set.intersection(*per_event.groupby("draw").event.apply(set)))]
    base = per_event.pivot(index="draw", columns="event", values="base").loc[meta.draw]
    hand = per_event.pivot(index="draw", columns="event", values="hand").loc[meta.draw]
    outcomes = {"hand_gain": hand - base, "iou_sar_only": base, "iou_sar_hand": hand}
    print(f"test = {args.name}: {len(meta)} training draws, {base.shape[1]} test events")
    print(meta.groupby("source").agg(draws=("draw", "size"), events=("events", lambda s: sorted(set(s))),
                                     tiles=("tiles", lambda s: f"{s.min()}-{s.max()}")).to_string())
    print(f"corr(log events, log tiles) = {np.corrcoef(np.log(meta.events), np.log(meta.tiles))[0, 1]:+.2f}")

    rng = np.random.default_rng(0)
    n_draw, n_ev = base.shape
    rows = []
    for label, table in outcomes.items():
        m = table.to_numpy()
        coef = fit(meta, m.mean(axis=1))
        boot = np.empty((args.boot, 3))
        for b in range(args.boot):
            i, j = rng.integers(0, n_draw, n_draw), rng.integers(0, n_ev, n_ev)
            boot[b] = fit(meta.iloc[i], m[np.ix_(i, j)].mean(axis=1))
        for k, term in enumerate(("intercept", "per doubling of events", "per doubling of tiles")):
            lo, hi = np.percentile(boot[:, k], [2.5, 97.5])
            rows.append({"outcome": label, "term": term, "coef": coef[k], "ci_lo": lo, "ci_hi": hi,
                         "p_sign": 2 * min((boot[:, k] <= 0).mean(), (boot[:, k] >= 0).mean())})
    table = pd.DataFrame(rows)
    print("\nOLS over training draws, two-way bootstrap (draws x test events) 95% CI:")
    print(table[table.term != "intercept"].round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        meta.assign(hand_gain=(hand - base).mean(axis=1).values, iou_sar_only=base.mean(axis=1).values).to_csv(
            args.out.with_name(args.out.stem + "_draws.csv"), index=False)


if __name__ == "__main__":
    main()
