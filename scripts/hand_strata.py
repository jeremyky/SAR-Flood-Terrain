"""Confusion counts per HAND stratum for LOEO test predictions.

For every leave-one-event-out run of the given variants, re-runs the saved
best.pt on the held-out event and accumulates TP/FP/FN/TN per HAND bin.
Output: one CSV row per (variant, seed, event, hand_bin).

    python scripts/hand_strata.py --runs /scratch/$USER/floodseg/runs --variants sar_only sar_hand --out strata.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from floodseg import data as D
from floodseg.model import build_model
from floodseg.train import build_splits

BINS = [0, 1, 2, 5, 10, 20, np.inf]
LABELS = ["0-1", "1-2", "2-5", "5-10", "10-20", ">20"]


def strata_for_run(run_dir: Path, device: torch.device) -> list[dict]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    root = Path(cfg["data"]["root"])
    channels = tuple(cfg["data"]["channels"])
    chips = build_splits(cfg, root)["test"]
    model = build_model(len(channels), cfg.get("model", {})).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.eval()
    threshold = cfg["eval"].get("threshold", 0.5)

    counts = np.zeros((len(LABELS), 4), dtype=np.int64)  # tp, fp, fn, tn
    loader = DataLoader(D.Sen1Floods11(root, chips, channels), batch_size=4, num_workers=4)
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                probs = torch.sigmoid(model(batch["x"].to(device)).float()).cpu().numpy()[:, 0]
            y, hand = batch["y"][:, 0].numpy(), batch["hand"][:, 0].numpy()
            valid = (y != D.IGNORE) & np.isfinite(hand)
            pred, truth = probs >= threshold, y == 1
            bins = np.digitize(hand, BINS[1:-1])
            for b in range(len(LABELS)):
                m = valid & (bins == b)
                counts[b] += [(pred & truth & m).sum(), (pred & ~truth & m).sum(), (~pred & truth & m).sum(), (~pred & ~truth & m).sum()]
    return [
        {"tp": c[0], "fp": c[1], "fn": c[2], "tn": c[3], "hand_bin": LABELS[b]}
        for b, c in enumerate(counts)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["sar_only", "sar_hand"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for variant in args.variants:
        for run_dir in sorted(args.runs.glob(f"loeo/*/{variant}/seed*")):
            if not (run_dir / "best.pt").exists() or not (run_dir / "metrics.json").exists():
                continue  # unfinished run
            event, seed = run_dir.parent.parent.name, run_dir.name
            for row in strata_for_run(run_dir, device):
                rows.append({"variant": variant, "event": event, "seed": seed, **row})
            print(variant, event, seed, flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
