"""Pre-registered #11: GEOID-Flood-trained results with "unseen" = activation absent from train (as reported) next to
"unseen" = absent from train and validation. Reads two `slurm/paper_reports.sbatch` output directories and writes the
rows of supplement Table S14 (LaTeX) and a CSV.

    python scripts/trainvalid_sensitivity.py reports/paper_20261124 reports/paper_trainvalid \
        --out reports/trainvalid_sensitivity.csv
"""
from __future__ import annotations

import argparse
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd


def fmt(x: float) -> str:
    d = Decimal(repr(float(x))).quantize(Decimal("0.001"), ROUND_HALF_UP)
    return ("−" if d < 0 or (d == 0 and x < 0) else "+") + f"{abs(d):.3f}"


def cell(est, lo, hi) -> str:
    return f"{fmt(est)} [{fmt(lo)}, {fmt(hi)}]"


def curve(d: Path, name: str, events: int, col: str = "event") -> tuple:
    r = pd.read_csv(d / name).set_index("train_events").loc[events]
    if col == "event":
        return r.delta, r.ci2_lo, r.ci2_hi, r.test_events
    return r.pooled_delta, r.pooled_ci_lo, r.pooled_ci_hi, r.test_events


def proto(d: Path, name: str, comparison: str) -> tuple:
    df = pd.read_csv(d / name)
    r = df[(df.subset == "unseen") & (df.metric == "flood_diff") & (df.variant == comparison)].iloc[0]
    return r.pooled_iou, r.ci2_lo, r.ci2_hi, r.events


def gate(d: Path, name: str) -> tuple:
    df = pd.read_csv(d / name)
    r = df[(df.subset == "unseen") & (df.comparison == "sar_hand_gate - sar_hand")].iloc[0]
    return r.pooled, r.pool_lo, r.pool_hi, r.events


def dropout(d: Path, name: str) -> tuple:
    df = pd.read_csv(d / name)
    r = df[(df.test == "flooded only") & (df.comparison == "sar_hand_md - sar_hand")].iloc[0]
    return r.event_mean, r.ev_lo, r.ev_hi, r.events


def confirmation(d: Path) -> tuple:
    text = (d / "report.txt").read_text()
    block = text.split("pre-registered confirmation test (draws 3-5), unseen, geoid_test_flood")[1]
    m = re.search(r"contrast .*?: ([-+][\d.]+)\s+95% CI \[([-+][\d.]+), ([-+][\d.]+)\]\s+one-sided Wilcoxon p = ([\d.]+)",
                  block)
    events = int(re.search(r"(\d+) test events", block).group(1))
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), events, float(m.group(4))


ROWS = [  # label, function, args
    ("159 events (Table~\\ref{tab:geoid}), flooded, event-mean", curve, ("lc_unseen_geoid_test_flood.csv", 159)),
    ("159 events, all water, event-mean", curve, ("lc_unseen_geoid_test.csv", 159)),
    ("159 events, flooded, pooled", curve, ("lc_unseen_geoid_test_flood.csv", 159, "pooled")),
    ("159 events, protocol pooled", proto, ("proto_geoid_test.csv", "sar_hand - sar_only")),
    ("10 events, flooded, event-mean", curve, ("lc_unseen_geoid_test_flood.csv", 10)),
    ("20 events, flooded, event-mean", curve, ("lc_unseen_geoid_test_flood.csv", 20)),
    ("40 events, flooded, event-mean", curve, ("lc_unseen_geoid_test_flood.csv", 40)),
    ("80 events, flooded, event-mean", curve, ("lc_unseen_geoid_test_flood.csv", 80)),
    ("20 events, protocol pooled", proto, ("proto_geoid_lc_e20_test.csv", "sar_hand - sar_only")),
    ("20 events, change detection, protocol pooled", proto, ("proto_geoid_lc_e20_test.csv", "cd_sar_hand - cd_sar")),
    ("159 events, change detection, protocol pooled", proto, ("proto_geoid_test.csv", "cd_sar_hand - cd_sar")),
    ("20 events, TerraMind, protocol pooled", proto, ("proto_geoid_lc_e20_test.csv", "fm_sar_hand - fm_sar")),
    ("20 events, HAND vs DEM + slope, flooded", curve, ("ablate_geoid_test_flood_sar_hand_vs_sar_dem_slope.csv", 20)),
    ("159 events, JRC vs SAR only, flooded", curve, ("ablate_geoid_test_flood_sar_jrc_vs_sar_only.csv", 159)),
    ("Relief gate minus no gate, 20 events, protocol", gate, ("gate_geoid_lc_e20_geoid_proto_test.csv",)),
    ("Relief gate, 159 events, protocol", gate, ("gate_geoid_geoid_proto_test.csv",)),
    ("HAND dropout minus HAND, 20 events, flooded", dropout, ("dropout_geoid_lc_e20.csv",)),
    ("HAND dropout, 159 events, flooded", dropout, ("dropout_geoid.csv",)),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("train_dir", type=Path, help="reports with unseen = absent from train")
    p.add_argument("trainvalid_dir", type=Path, help="reports with unseen = absent from train and validation")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    out = []
    for label, fn, fargs in ROWS:
        a, b = fn(args.train_dir, *fargs), fn(args.trainvalid_dir, *fargs)
        out.append({"result": label, "train": a[0], "train_lo": a[1], "train_hi": a[2], "train_events": a[3],
                    "trainvalid": b[0], "trainvalid_lo": b[1], "trainvalid_hi": b[2], "trainvalid_events": b[3]})
        print(f"{label} & {cell(*a[:3])} & {cell(*b[:3])} \\\\")
    a, b = confirmation(args.train_dir), confirmation(args.trainvalid_dir)
    out.append({"result": "confirmation contrast", "train": a[0], "train_lo": a[1], "train_hi": a[2], "train_events": a[3],
                "trainvalid": b[0], "trainvalid_lo": b[1], "trainvalid_hi": b[2], "trainvalid_events": b[3],
                "train_p": a[4], "trainvalid_p": b[4]})
    c4 = lambda x: f"{x:+.4f}".replace("-", "−")  # the paper quotes this contrast to four decimals
    print(f"Confirmation test (\\#1), contrast & {c4(a[0])} [{fmt(a[1])}, {fmt(a[2])}], $p = {a[4]:.2f}$ & "
          f"{c4(b[0])} [{fmt(b[1])}, {fmt(b[2])}], $p = {b[4]:.2f}$ \\\\")
    if args.out:
        pd.DataFrame(out).to_csv(args.out, index=False)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
