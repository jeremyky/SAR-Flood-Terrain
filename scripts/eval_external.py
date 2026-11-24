"""Evaluate trained runs on an external dataset in the Sen1Floods11 layout (e.g. GEOID-Flood test).

For every run directory given (containing config.yaml + best.pt), scores all chips under
--data-root and writes <run>/external_<name>/metrics.json (per-event and pooled summaries,
raw and HAND-masked) and chips.csv. With --post-mask T the scored prediction is the HAND post-mask itself (flood
removed wherever HAND > T metres), so chips.csv holds the masked counts; use a distinct --name for it.

    python scripts/eval_external.py --data-root /scratch/$USER/floodseg/data/geoid_test --name geoid_test \
        /scratch/$USER/floodseg/runs/all/*/seed*
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np  # noqa: F401  (import before torch: the conda libstdc++ must load first)
import torch
import yaml
from torch.utils.data import DataLoader

from floodseg import data as D
from floodseg.model import build_model
from floodseg.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--exclude-events", nargs="*", default=[], help="events to leave out (e.g. floods also in the training set)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--hand-perturb", default=None,
                        help="perturb the HAND input at test time: shift:<px>, blur:<px> or swap (see floodseg.data); "
                             "default: the run's own data.hand_perturb, if any")
    parser.add_argument("--post-mask", type=float, default=None,
                        help="score the HAND post-mask at this threshold (m) instead of the raw prediction")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chips = [c for c in D.all_chips(args.data_root) if c.event not in args.exclude_events]
    print(f"{len(chips)} chips, {len({c.event for c in chips})} events")

    for run in args.runs:
        out = run / f"external_{args.name}"
        if (out / "metrics.json").exists():
            print("skip", run)
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        channels = tuple(cfg["data"]["channels"])
        model = build_model(len(channels), cfg.get("model", {})).to(device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        perturb = args.hand_perturb or cfg["data"].get("hand_perturb")
        loader = DataLoader(D.Sen1Floods11(args.data_root, chips, channels, hand_perturb=perturb),
                            batch_size=args.batch_size, num_workers=args.workers)
        if args.post_mask is not None:
            masked = evaluate(model, loader, device, cfg["eval"].get("threshold", 0.5), hand_mask=args.post_mask)
            trackers = {"raw": masked["hand_mask"]}
        else:
            trackers = evaluate(model, loader, device, cfg["eval"].get("threshold", 0.5), hand_mask=cfg["eval"].get("hand_mask"))
        out.mkdir(exist_ok=True)
        (out / "metrics.json").write_text(json.dumps({k: t.summary() for k, t in trackers.items() if t.rows}, indent=1))
        rows = trackers["raw"].per_chip()
        with open(out / "chips.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(run, json.dumps(trackers["raw"].summary()["total"]), flush=True)


if __name__ == "__main__":
    main()
