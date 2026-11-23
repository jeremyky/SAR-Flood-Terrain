"""Train a U-Net on Sen1Floods11 and evaluate it on held-out chips/events.

    python -m floodseg.train --config configs/base.yaml [key.sub=value ...]

Model selection uses validation IoU only; test splits are scored once, with
the selected checkpoint, at the end of the run.
"""
from __future__ import annotations

import argparse
import os
import csv
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from floodseg import data as D
from floodseg.losses import bce_dice, hand_ceiling, hand_monotonicity
from floodseg.metrics import Tracker
from floodseg.model import build_model


def _expand(node):
    """Expand $VARS in string values (paths), e.g. ${FLOODSEG_ROOT}/runs."""
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    return os.path.expandvars(node) if isinstance(node, str) else node


def load_config(path: str, overrides: list[str]) -> dict:
    os.environ.setdefault("FLOODSEG_ROOT", f"/scratch/{os.environ.get('USER', 'user')}/floodseg")
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
        if isinstance(node[leaf], str):  # YAML 1.1 reads 5e-5 (no decimal point) as a string
            try:
                node[leaf] = float(node[leaf])
            except ValueError:
                pass
    return _expand(cfg)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_splits(cfg: dict, root: Path) -> dict[str, list[D.Chip]]:
    split = cfg["split"]
    if split["mode"] == "official":
        return {name: D.official_split(root, name) for name in ("train", "valid", "test", "bolivia")}
    if split["mode"] == "loeo":
        return D.leave_one_event_out(root, split["held_out"], split.get("val_fraction", 0.15), cfg["seed"])
    if split["mode"] == "lists":
        return D.listed_split(root, split.get("train_list", "train"))
    if split["mode"] == "all":
        return D.all_events(root, split.get("val_fraction", 0.15), cfg["seed"])
    raise ValueError(split["mode"])


def compute_loss(cfg: dict, logits: torch.Tensor, y: torch.Tensor, hand: torch.Tensor) -> dict[str, torch.Tensor]:
    lc = cfg["loss"]
    terms = {"seg": bce_dice(logits.float(), y, pos_weight=lc.get("pos_weight", 1.0))}
    mono = lc.get("hand_monotonicity", {})
    if mono.get("weight", 0) > 0:
        terms["mono"] = mono["weight"] * hand_monotonicity(logits.float(), hand, mono.get("margin", 0.05), mono.get("min_delta", 1.0),
                                                           mono.get("stride", 1))
    ceil = lc.get("hand_ceiling", {})
    if ceil.get("weight", 0) > 0:
        terms["ceil"] = ceil["weight"] * hand_ceiling(logits.float(), hand, ceil.get("threshold", 15.0))
    terms["total"] = sum(terms.values())
    return terms


@torch.no_grad()
def evaluate(model, loader, device, threshold: float, hand_mask: float | None = None) -> dict[str, Tracker]:
    model.eval()
    trackers = {"raw": Tracker(), "hand_mask": Tracker()}
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            probs = torch.sigmoid(model(batch["x"].to(device)).float()).cpu().numpy()
        for i in range(len(probs)):
            pred = probs[i, 0] >= threshold
            y, hand = batch["y"][i, 0].numpy(), batch["hand"][i, 0].numpy()
            trackers["raw"].add(batch["chip"][i], batch["event"][i], pred, y, hand)
            if hand_mask is not None:
                masked = pred & ~(np.nan_to_num(hand, nan=0.0) > hand_mask)
                trackers["hand_mask"].add(batch["chip"][i], batch["event"][i], masked, y, hand)
    return trackers


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotted key=value overrides, e.g. train.epochs=5")
    parser.add_argument("--eval-only", action="store_true", help="re-score an existing run's best.pt (pass its saved config.yaml)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg["seed"])

    root = Path(cfg["data"]["root"])
    run_dir = Path(cfg["output_root"]) / cfg["name"] / f"seed{cfg['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "git_commit": git_commit()}, sort_keys=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = tuple(cfg["data"]["channels"])
    splits = build_splits(cfg, root)
    tc = cfg["train"]
    workers = tc.get("num_workers", 4)
    train_loader = DataLoader(
        D.Sen1Floods11(root, splits["train"], channels, crop=tc.get("crop", 256), augment=True,
                        hand_dropout=cfg["data"].get("hand_dropout", 0.0),
                        hand_perturb=cfg["data"].get("hand_perturb")),
        batch_size=tc["batch_size"], shuffle=True, num_workers=workers, drop_last=True, persistent_workers=workers > 0,
    )
    eval_loader = lambda chips: DataLoader(D.Sen1Floods11(root, chips, channels, hand_perturb=cfg["data"].get("hand_perturb")),
                                           batch_size=4, num_workers=workers)
    val_loader = eval_loader(splits["valid"])

    model = build_model(len(channels), cfg.get("model", {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 1e-4))
    # `train.steps` fixes the optimizer budget regardless of training-set size (epochs derived from it);
    # `train.num_validations` limits how often the validation set is scored.
    epochs = math.ceil(tc["steps"] / len(train_loader)) if tc.get("steps") else tc["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=epochs * len(train_loader))
    threshold = cfg["eval"].get("threshold", 0.5)
    val_every = max(1, epochs // tc.get("num_validations", epochs))

    best_iou, history = -1.0, []
    if args.eval_only:
        history = json.loads((run_dir / "history.json").read_text())
        best_iou = max(row["val_iou"] for row in history)
    for epoch in range(0 if args.eval_only else epochs):
        model.train()
        t0, sums = time.time(), {}
        for batch in train_loader:
            x, y, hand = (batch[k].to(device, non_blocking=True) for k in ("x", "y", "hand"))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
            terms = compute_loss(cfg, logits, y, hand)
            opt.zero_grad(set_to_none=True)
            terms["total"].backward()
            opt.step()
            sched.step()
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + v.item()
        if (epoch + 1) % val_every and epoch != epochs - 1:
            continue
        val = evaluate(model, val_loader, device, threshold)["raw"].summary()
        row = {"epoch": epoch, **{f"loss_{k}": v / len(train_loader) for k, v in sums.items()},
               "val_iou": val["total"]["iou"], "val_chip_iou": val["chip_mean_iou"], "sec": time.time() - t0}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["total"]["iou"] > best_iou:
            best_iou = val["total"]["iou"]
            torch.save(model.state_dict(), run_dir / "best.pt")

    if not args.eval_only:
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    results = {"best_val_iou": best_iou, "splits": {}}
    for name, chips in splits.items():
        if name == "train":
            continue
        trackers = evaluate(model, eval_loader(chips), device, threshold, hand_mask=cfg["eval"].get("hand_mask"))
        results["splits"][name] = {k: t.summary() for k, t in trackers.items() if t.rows}
        with open(run_dir / f"chips_{name}.csv", "w", newline="") as f:
            rows = trackers["raw"].per_chip()
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({s: r["raw"]["total"] for s, r in results["splits"].items()}, indent=1))


if __name__ == "__main__":
    main()
