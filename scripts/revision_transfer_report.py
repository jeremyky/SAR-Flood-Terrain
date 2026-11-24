"""Sen1Floods11 -> GEOID-Flood transfer: physics losses, HAND post-masks and TerraMind learning rates
(PREREGISTRATIONS.md #10 and #12), and the five-seed headline (#15: Table III, Fig. 1). Same estimands as scripts/ablation_all_report.py: all 77/79 GEOID test events,
event-mean IoU (flooded only, all water) and pooled IoU_flood under the GEOID-Flood protocol, two-way 95% CIs over
seeds x events; each comparison uses the seeds both variants have.

    python scripts/revision_transfer_report.py losses --runs $S/runs/all --out $S/reports/transfer_losses.csv
    python scripts/revision_transfer_report.py masks  --runs $S/runs/all --out $S/reports/transfer_masks.csv
    python scripts/revision_transfer_report.py fm-lr  --runs $S/runs/all --out $S/reports/transfer_fm_lr.csv
    python scripts/revision_transfer_report.py headline --runs $S/runs/all --out $S/reports/transfer_headline.csv
    python scripts/revision_transfer_report.py input-vs-mask --runs $S/runs/all \
        --out $S/reports/transfer_input_vs_mask_exploratory.csv   # exploratory, not pre-registered
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hand_gate import compare, load  # noqa: E402

TESTS = {"geoid_test_flood": "flooded only", "geoid_test": "all water", "geoid_proto_test": "protocol flood_diff"}
LOSSES = ["sar_mono", "sar_ceil", "sar_mono3_w1", "sar_mono3_w10"]
MASKS = [5, 10, 15, 20]
FM_LRS = {"1e-5": "_lr1e-5", "2e-5": "_lr2e-5", "5e-5": "", "1e-4": "_lr1e-4", "2e-4": "_lr2e-4"}  # 5e-5: existing runs


def frame(runs: Path, name: str, suffix: str = "") -> pd.DataFrame:
    df = load(runs, name)
    df = df[df.metric.isin(["label", "flood_diff"])].copy()
    df["variant"] = df.variant + suffix
    return df


def table(runs: Path, pairs: list[tuple[str, str]], masks: list[int] = (), boot: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for name, label in TESTS.items():
        parts = [frame(runs, name)] + [frame(runs, f"{name}_mask{t}", f"+mask{t}") for t in masks]
        df = pd.concat(parts)
        for x, y in pairs:
            if {x, y} <= set(df.variant):
                rows.append({"test": label, "comparison": f"{x} - {y}", **compare(df, x, y, rng, boot)})
            else:
                print(f"missing for {label}: {sorted({x, y} - set(df.variant))}")
    return pd.DataFrame(rows)


def val_iou(runs: Path, variant: str) -> list[float]:
    return [json.loads((d / "metrics.json").read_text())["best_val_iou"] for d in sorted((runs / variant).glob("seed*"))
            if (d / "metrics.json").exists()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["losses", "masks", "fm-lr", "headline", "input-vs-mask"])
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--boot", type=int, default=4000)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    pd.set_option("display.width", 250)

    if args.what == "losses":
        pairs = [(v, "sar_only") for v in LOSSES] + [("sar_hand", "sar_only")]
        out = table(args.runs, pairs, boot=args.boot)
    elif args.what == "headline":
        out = table(args.runs, [("sar_hand", "sar_only"), ("r34_sar_hand", "r34_sar_only")], boot=args.boot)
    elif args.what == "input-vs-mask":
        out = table(args.runs, [("sar_hand", f"sar_only+mask{t}") for t in MASKS], masks=MASKS, boot=args.boot)
    elif args.what == "masks":
        pairs = [(f"sar_only+mask{t}", "sar_only") for t in MASKS] + [(f"sar_hand+mask{t}", "sar_hand") for t in MASKS]
        pairs += [("sar_hand", "sar_only")]
        out = table(args.runs, pairs, masks=MASKS, boot=args.boot)
    else:
        sel = {}
        print("Sen1Floods11 validation IoU (best checkpoint), mean over seeds, by learning rate:")
        for base in ("fm_sar", "fm_sar_hand"):
            vals = {lr: val_iou(args.runs, base + sfx) for lr, sfx in FM_LRS.items()}
            print(f"  {base}: " + ", ".join(f"{lr} {np.mean(v):.4f} (n={len(v)})" for lr, v in vals.items() if v))
            done = {lr: v for lr, v in vals.items() if len(v) >= 3}
            sel[base] = base + FM_LRS[max(done, key=lambda lr: np.mean(done[lr]))]
            print(f"  -> selected {sel[base]}")
        a, h = sel["fm_sar"], sel["fm_sar_hand"]
        pairs = [(h, a), (a, "sar_only"), ("sar_hand", a)]
        pairs += [(f"fm_sar_hand{s}", f"fm_sar{s}") for s in FM_LRS.values() if f"fm_sar{s}" != a or f"fm_sar_hand{s}" != h]
        out = table(args.runs, pairs, boot=args.boot)
    print(f"{args.runs}: all GEOID test events; two-way 95% CIs (seeds x events)")
    print(out.round(4).to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
