"""Geographic-cluster sensitivity of the headline transfer result (PREREGISTRATIONS.md #8).

Test events from the same country may share weather systems, landscapes and mapping teams, so they are not
independent. This recomputes the CI of the SAR + HAND minus SAR-only difference for the models trained on
the 11 Sen1Floods11 events, resampling countries (with all their events) and seeds instead of events and
seeds. Event-mean IoU: mean over events of the seed-averaged paired difference.

    python scripts/cluster_bootstrap.py --runs $S/runs/all --events $S/external_explain_events.csv \
        --seeds seed0 seed1 seed2 --out $S/reports/cluster_bootstrap.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hand_gate import load  # noqa: E402

NAMES = {"geoid_test_flood": "flooded only", "geoid_test": "all water"}
PAIRS = [("sar_hand", "sar_only"), ("r34_sar_hand", "r34_sar_only")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True, help="CSV with event, country, continent")
    p.add_argument("--cluster", default="country", choices=["country", "continent"])
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--out", type=Path)
    p.add_argument("--seeds", nargs="+", help="seed dirs to use (default: all); registered with seed0 seed1 seed2")
    args = p.parse_args()
    meta = pd.read_csv(args.events).set_index("event")[args.cluster]
    rng = np.random.default_rng(0)
    rows = []
    for name, label in NAMES.items():
        df = load(args.runs, name, args.seeds)
        ev = df.groupby(["variant", "seed", "event"])[["tp", "fp", "fn"]].sum()
        ev = ev[(ev.tp + ev.fn).groupby(level="event").transform("max") > 0]
        iou = (ev.tp / (ev.tp + ev.fp + ev.fn)).rename("iou").reset_index()
        for x, y in PAIRS:
            a = iou[iou.variant == x].pivot(index="event", columns="seed", values="iou")
            b = iou[iou.variant == y].pivot(index="event", columns="seed", values="iou")
            seeds = sorted(set(a.columns) & set(b.columns))
            delta = (a[seeds] - b[seeds]).dropna()                       # events x seeds
            clusters = meta.reindex(delta.index).fillna("unknown")
            groups = [np.flatnonzero(clusters.to_numpy() == c) for c in clusters.unique()]
            m = delta.to_numpy()
            boot = np.empty(args.boot)
            for k in range(args.boot):
                ci = rng.integers(0, len(groups), len(groups))
                rows_idx = np.concatenate([groups[c] for c in ci])
                cols = rng.integers(0, m.shape[1], m.shape[1])
                boot[k] = m[np.ix_(rows_idx, cols)].mean()
            ev_boot = np.array([m[rng.integers(0, m.shape[0], m.shape[0])][:, rng.integers(0, m.shape[1], m.shape[1])].mean()
                                for _ in range(args.boot)])
            rows.append({"test": label, "comparison": f"{x} - {y}", "events": m.shape[0], "seeds": m.shape[1],
                         "clusters": len(groups), "cluster_by": args.cluster, "event_mean": m.mean(),
                         "ci_events_lo": np.percentile(ev_boot, 2.5), "ci_events_hi": np.percentile(ev_boot, 97.5),
                         "ci_clusters_lo": np.percentile(boot, 2.5), "ci_clusters_hi": np.percentile(boot, 97.5)})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"{args.runs}: event-mean IoU difference, CI resampling events x seeds vs {args.cluster} clusters x seeds")
    print(table.round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
