"""Same tiles, same metric: HAND gain when the tile's event was in training vs held out (PREREGISTRATIONS.md #9).

The 90 tiles of Sen1Floods11's official test split come from 10 of its 11 events (Bolivia has none). Each is scored
by two kinds of model:
  same event  official-split models (runs/<variant>/seed*), trained on the official train split, which holds
              other tiles of every one of these 10 events;
  held out    leave-one-event-out models (runs/loeo/<event>/<variant>/seed*) whose held-out event is the tile's event.
Per event, IoU is pooled over that event's official test tiles; the HAND gain is SAR + HAND minus SAR only from the
same seed. Primary: difference of event-mean gains (held out minus same event), two-way bootstrap that resamples
events (shared by both settings, since the tiles are the same) and, independently within each setting, seeds.
Secondary: the same difference in IoU pooled over all 90 tiles.

    python scripts/matched_split_report.py --runs $S/runs --out $S/reports/matched_split.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = ("sar_only", "sar_hand")


def counts(csv: Path, chips: set[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv)
    if chips is not None:
        df = df[df.chip.isin(chips)]
    df[["tp", "fp", "fn", "tn"]] = np.array([json.loads(c) for c in df.cm])
    return df.groupby("event")[["tp", "fp", "fn"]].sum()


def load(runs: Path) -> tuple[dict, dict, list[str]]:
    """Per setting: {variant: {seed: DataFrame(event -> tp, fp, fn)}}, on the official test tiles."""
    test = pd.read_csv(next((runs / "sar_only").glob("seed*/chips_test.csv")))
    chips, events = set(test.chip), sorted(test.event.unique())
    same = {v: {d.name: counts(d / "chips_test.csv") for d in sorted((runs / v).glob("seed*"))} for v in VARIANTS}
    held: dict = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        seeds = sorted({d.name for e in events for d in (runs / "loeo" / e / v).glob("seed*")})
        for s in seeds:
            parts = [counts(runs / "loeo" / e / v / s / "chips_test.csv", chips) for e in events
                     if (runs / "loeo" / e / v / s / "chips_test.csv").exists()]
            if len(parts) == len(events):  # a seed counts only if every held-out event has it
                held[v][s] = pd.concat(parts)
    for setting in (same, held):
        for v in VARIANTS:
            for s, df in setting[v].items():
                assert sorted(df.index) == events, (v, s)
    return same, held, events


def arrays(setting: dict, events: list[str]) -> np.ndarray:
    """seeds x events x {sar_only, sar_hand} x (tp, fp, fn), seeds present for both variants."""
    seeds = sorted(set(setting[VARIANTS[0]]) & set(setting[VARIANTS[1]]))
    return np.stack([np.stack([setting[v][s].loc[events].to_numpy(float) for v in VARIANTS], 1) for s in seeds]), seeds


def iou(c: np.ndarray) -> np.ndarray:
    return c[..., 0] / np.maximum(c.sum(-1), 1)


def stats(a: np.ndarray, ei: np.ndarray, si: np.ndarray) -> tuple[float, float]:
    """(event-mean gain, pooled gain) for seed and event index arrays."""
    sub = a[si][:, ei]                                   # seeds x events x variants x 3
    ev = (iou(sub[:, :, 1]) - iou(sub[:, :, 0])).mean()  # same-seed gain per event, averaged
    pooled = sub.sum(1)                                  # seeds x variants x 3
    return ev, (iou(pooled[:, 1]) - iou(pooled[:, 0])).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    same, held, events = load(args.runs)
    a_same, s_same = arrays(same, events)
    a_held, s_held = arrays(held, events)
    n_e = len(events)
    all_e = np.arange(n_e)
    point = {"same": stats(a_same, all_e, np.arange(len(s_same))), "held": stats(a_held, all_e, np.arange(len(s_held)))}

    rng = np.random.default_rng(0)
    boot = np.empty((args.boot, 2, 2))
    for k in range(args.boot):
        ei = rng.integers(0, n_e, n_e)  # shared: the same tiles are scored in both settings
        boot[k, 0] = stats(a_same, ei, rng.integers(0, len(s_same), len(s_same)))
        boot[k, 1] = stats(a_held, ei, rng.integers(0, len(s_held), len(s_held)))
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    dlo, dhi = np.percentile(boot[:, 1] - boot[:, 0], [2.5, 97.5], axis=0)

    print(f"{len(events)} events, {int(sum(a_same[0, :, 0].sum(-1) > 0))} with labelled pixels; official test tiles only")
    print(f"seeds: same event {s_same}; held out {s_held}")
    rows = []
    for j, metric in enumerate(("event-mean IoU", "pooled IoU")):
        for i, (key, label) in enumerate((("same", "same event (official split)"), ("held", "event held out (LOEO)"))):
            rows.append({"metric": metric, "quantity": f"HAND gain, {label}", "value": point[key][j],
                         "lo": lo[i, j], "hi": hi[i, j]})
        rows.append({"metric": metric, "quantity": "held out minus same event", "value": point["held"][j] - point["same"][j],
                     "lo": dlo[j], "hi": dhi[j]})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))

    per = pd.DataFrame({
        "tiles": pd.read_csv(next((args.runs / "sar_only").glob("seed*/chips_test.csv"))).event.value_counts().reindex(events),
        "same: SAR": iou(a_same[:, :, 0]).mean(0), "same: +HAND": iou(a_same[:, :, 1]).mean(0),
        "held: SAR": iou(a_held[:, :, 0]).mean(0), "held: +HAND": iou(a_held[:, :, 1]).mean(0)}, index=events)
    per["gain same"], per["gain held"] = per["same: +HAND"] - per["same: SAR"], per["held: +HAND"] - per["held: SAR"]
    print("\nPer event (mean over seeds):")
    print(per.round(3).to_string())
    print(f"\nevents where the held-out gain exceeds the same-event gain: {(per['gain held'] > per['gain same']).sum()}/{n_e}")
    if args.out:
        table.to_csv(args.out, index=False)
        per.to_csv(args.out.with_name(args.out.stem + "_events.csv"))


if __name__ == "__main__":
    main()
