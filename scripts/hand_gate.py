"""Relief gate for HAND: on chips where nearly all ground is close to drainage, HAND carries no information
and pushes the model toward flood everywhere (Somalia in Sen1Floods11, EMSR419 in GEOID). The gate uses the
SAR-only model's prediction on those chips and the HAND model's elsewhere. No retraining: it recombines the
per-chip confusion counts both models already wrote.

Rule (fixed before outcomes were looked at, PREREGISTRATIONS.md, 2026-11-21): a chip is "flat" when the 90th percentile of
its valid HAND pixels is below 2 m; flat chips take the SAR-only prediction of the same seed / draw.

    # 1. per-chip HAND statistics (no outcomes)
    python scripts/hand_gate.py stats --data-root $D/geoid_test_flood --out $S/reports/hand_stats_geoid_test.csv
    # 2. gated vs ungated on finished runs (external eval names or protocol names)
    python scripts/hand_gate.py report --runs $S/runs/geoid_lc/e20 --stats ... --name geoid_test_flood \
        --train-list $D/geoid_trainval/splits/train.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).parent))
from learning_curve_report import hier_boot, training_events  # noqa: E402

PAIRS = {"sar_hand": "sar_only", "cd_sar_hand": "cd_sar", "fm_sar_hand": "fm_sar"}


def stats(args: argparse.Namespace) -> None:
    rows = []
    for tif in sorted((args.data_root / "terrain").glob("*_HAND.tif")):
        with rasterio.open(tif) as src:
            h = src.read(1).astype(np.float32)
        h = h[np.isfinite(h)]
        chip = tif.name[: -len("_HAND.tif")]
        q = np.percentile(h, [50, 90]) if h.size else (np.nan, np.nan)
        rows.append({"chip": chip, "hand_p50": q[0], "hand_p90": q[1], "frac_below_2m": float((h < 2).mean()) if h.size else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} chips; flat (p90 < {args.threshold} m): {(df.hand_p90 < args.threshold).mean():.3f}")


def load(runs: Path, name: str) -> pd.DataFrame:
    frames = []
    for csv in sorted(runs.glob(f"*/seed*/external_{name}/chips.csv")):
        df = pd.read_csv(csv)
        if "cm" in df:  # external eval: one row per chip
            df[["tp", "fp", "fn", "tn"]] = np.array([json.loads(c) for c in df.cm])
            df["metric"] = "label"
        df["variant"], df["seed"] = csv.parent.parent.parent.name, csv.parent.parent.name
        frames.append(df[["chip", "event", "metric", "variant", "seed", "tp", "fp", "fn"]])
    if not frames:
        raise SystemExit(f"no {name} results under {runs}")
    return pd.concat(frames)


def gate(df: pd.DataFrame, flat: set[str], hand: str, base: str) -> pd.DataFrame:
    a = df[df.variant == hand].set_index(["metric", "seed", "chip"])
    b = df[df.variant == base].set_index(["metric", "seed", "chip"])
    a = a[a.index.isin(b.index)]
    use_b = a.index.get_level_values("chip").isin(flat)
    g = a.copy()
    g.loc[use_b, ["tp", "fp", "fn"]] = b.loc[a.index[use_b], ["tp", "fp", "fn"]].to_numpy()
    g["variant"] = f"{hand}_gate"
    return g.reset_index()


def compare(d: pd.DataFrame, x: str, y: str, rng: np.random.Generator, n: int) -> dict:
    """x - y: event-mean IoU (two-way CI, draws x events) and pooled IoU (two-way CI)."""
    ev = d.groupby(["variant", "seed", "event"])[["tp", "fp", "fn"]].sum()
    pos = (ev.tp + ev.fn).groupby("event").max() > 0
    events = pos[pos].index
    ev = ev[ev.index.get_level_values("event").isin(events)]
    seeds = sorted(set(ev.loc[x].index.get_level_values("seed")) & set(ev.loc[y].index.get_level_values("seed")))
    iou = ev.tp / (ev.tp + ev.fp + ev.fn)
    delta = (iou.loc[x] - iou.loc[y]).dropna()
    delta = delta[delta.index.get_level_values("seed").isin(seeds)]
    eb = hier_boot(delta, rng, n)
    by_event = delta.groupby("event").mean()
    per = {v: np.stack([ev.loc[(v, s)].reindex(events, fill_value=0).to_numpy() for s in seeds]) for v in (x, y)}
    f = lambda c: c[..., 0] / c.sum(-1)  # noqa: E731
    pooled = (f(per[x].sum(1)) - f(per[y].sum(1))).mean()
    eidx = rng.integers(0, len(events), (n, len(events)))
    sidx = rng.integers(0, len(seeds), (n, len(seeds)))
    boot = np.array([f(per[x][sidx[k]][:, eidx[k]].sum((0, 1))) - f(per[y][sidx[k]][:, eidx[k]].sum((0, 1))) for k in range(n)])
    return {"events": len(events), "seeds": len(seeds),
            "event_mean": delta.groupby("seed").mean().mean(), "ev_lo": np.percentile(eb, 2.5), "ev_hi": np.percentile(eb, 97.5),
            "improved": int((by_event > 0).sum()), "worse": int((by_event < 0).sum()),
            "pooled": pooled, "pool_lo": np.percentile(boot, 2.5), "pool_hi": np.percentile(boot, 97.5)}


def report(args: argparse.Namespace) -> None:
    st = pd.read_csv(args.stats)
    flat = set(st.chip[st.hand_p90 < args.threshold])
    df = load(args.runs, args.name)
    metrics = ["label"] if "label" in set(df.metric) else ["flood_diff"]
    df = df[df.metric.isin(metrics)]
    subsets = {"all": df}
    if args.train_list:
        seen = training_events(args.train_list)
        subsets["unseen"] = df[~df.event.isin(seen)]
        subsets["shared"] = df[df.event.isin(seen)]
    rng = np.random.default_rng(0)
    rows = []
    for hand, base in PAIRS.items():
        if not {hand, base} <= set(df.variant):
            continue
        for subset, d in subsets.items():
            if d.empty:
                continue
            g = pd.concat([d, gate(d, flat, hand, base)])
            n_flat = d[(d.variant == hand) & d.chip.isin(flat)].chip.nunique()
            for x, y in ((f"{hand}_gate", base), (hand, base), (f"{hand}_gate", hand)):
                rows.append({"subset": subset, "comparison": f"{x} - {y}", "flat_chips": n_flat,
                             "chips": d[d.variant == hand].chip.nunique(), **compare(g, x, y, rng, args.boot)})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"{args.runs} / {args.name}: flat = HAND p90 < {args.threshold} m; event-mean and pooled IoU deltas, "
          "two-way 95% CIs (draws/seeds x events)")
    print(table.round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--data-root", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--threshold", type=float, default=2.0)
    r = sub.add_parser("report")
    r.add_argument("--runs", type=Path, required=True)
    r.add_argument("--stats", type=Path, required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--train-list", type=Path)
    r.add_argument("--threshold", type=float, default=2.0)
    r.add_argument("--boot", type=int, default=4000)
    r.add_argument("--out", type=Path)
    args = p.parse_args()
    stats(args) if args.cmd == "stats" else report(args)


if __name__ == "__main__":
    main()
