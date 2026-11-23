"""Score runs on GEOID-Flood test the way the GEOID-Flood paper does (Chiriaco et al. 2026, arXiv 2608.02315).

The paper evaluates single-image models by running them on the pre- and the post-event S1 image and calling
flood the pixels that are water after but not before ("post-hoc three-class map"); permanent water is a
separate class, so flood predicted on it is a false positive. Metrics are accumulated over all windows
(pooled pixel counts) and IoU_flood is their primary flood metric.

Inputs (built by slurm/prepare_geoid_protocol.sbatch):
  --post  data root with the post-event S1 and three-class labels (0 background, 1 permanent, 2 flood)
  --pre   data root with the latest pre-event S1 for the same chips (labels identical)

Per run it writes <run>/external_<name>/chips.csv with confusion counts, per chip, for
  flood_diff  predicted flood = water(post) & ~water(pre)  vs  label == 2   (the paper's protocol)
  flood_post  predicted flood = water(post)                vs  label == 2   (no pre image; permanent water = FP)
  water_post  predicted water = water(post)                vs  label in {1, 2}
Pixels count only where the label is valid and both images are complete.
Change-detection runs (inputs vv_pre/vh_pre, trained on flooded water vs rest; scripts/sweep_geoid.sh cd) see both
images at once, so their output is the flood map itself: flood_diff = their prediction vs label == 2, on the same
chips and pixels (flood_post and water_post do not apply). --post must then hold S1PreHand (prepare_geoid_cd.sbatch).

    python scripts/eval_geoid_protocol.py eval --post $D/geoid_test_raw --pre $D/geoid_test_pre --name geoid_proto runs/geoid/*/seed*
    python scripts/eval_geoid_protocol.py report --runs runs/geoid --name geoid_proto [--train-list .../train.txt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (import before torch: the conda libstdc++ must load first)
import pandas as pd

METRICS = ("flood_diff", "flood_post", "water_post")


def counts(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> list[int]:
    p, g = pred[valid], gt[valid]
    return [int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum()), int((~p & ~g).sum())]


def evaluate_runs(args: argparse.Namespace) -> None:
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from floodseg import data as D
    from floodseg.model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_ids = {c.chip_id for c in D.all_chips(args.pre)}
    chips = [c for c in D.all_chips(args.post) if c.chip_id in pre_ids]
    print(f"{len(chips)} chips with a pre-event image ({len(D.all_chips(args.post)) - len(chips)} without), "
          f"{len({c.event for c in chips})} events", flush=True)
    for run in args.runs:
        out = run / f"external_{args.name}"
        if (out / "chips.csv").exists():
            print("skip", run)
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        channels = tuple(cfg["data"]["channels"])
        thr = cfg["eval"].get("threshold", 0.5)
        model = build_model(len(channels), cfg.get("model", {})).to(device).eval()
        model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        change = any(ch.endswith("_pre") for ch in channels)  # pre + post inputs: the output is the flood map
        # for a change model the pre loader only supplies the pre-image validity mask (no forward pass)
        loaders = [DataLoader(D.Sen1Floods11(root, chips, ch), batch_size=args.batch_size, num_workers=args.workers)
                   for root, ch in ((args.post, channels), (args.pre, ("vv", "vh") if change else channels))]
        rows = []
        with torch.no_grad():
            for post, pre in zip(*loaders):
                assert list(post["chip"]) == list(pre["chip"])
                water = []
                for batch in (post,) if change else (post, pre):
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        prob = torch.sigmoid(model(batch["x"].to(device)).float()).cpu().numpy()[:, 0]
                    water.append(prob >= thr)
                y_post, y_pre = post["y"][:, 0].numpy(), pre["y"][:, 0].numpy()
                for i, chip in enumerate(post["chip"]):
                    valid = (y_post[i] >= 0) & (y_pre[i] >= 0)  # label valid and both images complete
                    flood, perm_or_flood = y_post[i] == 2, y_post[i] >= 1
                    if change:
                        scored = (("flood_diff", water[0][i], flood),)
                    else:
                        w_post, w_pre = water[0][i], water[1][i]
                        scored = (("flood_diff", w_post & ~w_pre, flood), ("flood_post", w_post, flood),
                                  ("water_post", w_post, perm_or_flood))
                    for metric, pred, gt in scored:
                        rows.append([chip, post["event"][i], metric, *counts(pred, gt, valid)])
        out.mkdir(exist_ok=True)
        df = pd.DataFrame(rows, columns=["chip", "event", "metric", "tp", "fp", "fn", "tn"])
        df.to_csv(out / "chips.csv", index=False)
        pooled = df.groupby("metric")[["tp", "fp", "fn"]].sum()
        print(run, (pooled.tp / (pooled.tp + pooled.fp + pooled.fn)).round(4).to_dict(), flush=True)


def load(runs: Path, name: str) -> pd.DataFrame:
    frames = []
    for csv in sorted(runs.glob(f"*/seed*/external_{name}/chips.csv")):
        df = pd.read_csv(csv)
        df["variant"], df["seed"] = csv.parent.parent.parent.name, csv.parent.parent.name
        frames.append(df)
    if not frames:
        raise SystemExit(f"no {name} results under {runs}")
    return pd.concat(frames)


def iou(t: pd.DataFrame) -> pd.Series:
    return t.tp / (t.tp + t.fp + t.fn)


def report(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from learning_curve_report import training_events

    pd.set_option("display.width", 250)
    df = load(args.runs, args.name)
    subsets = {"all": df}
    if args.train_list:
        seen = training_events(args.train_list)
        subsets["unseen"] = df[~df.event.isin(seen)]
        subsets["shared"] = df[df.event.isin(seen)]
    rng = np.random.default_rng(0)
    rows = []
    for subset, d in subsets.items():
        if d.empty:  # e.g. the held-out events share no activation with train
            continue
        ev = d.groupby(["metric", "variant", "seed", "event"])[["tp", "fp", "fn"]].sum()
        for metric in METRICS:
            if metric not in ev.index.get_level_values("metric"):
                continue
            m = ev.loc[metric]
            has_pos = (m.tp + m.fn).groupby("event").max() > 0
            events = has_pos[has_pos].index
            m = m[m.index.get_level_values("event").isin(events)]
            pooled = m.groupby(["variant", "seed"]).sum()
            pooled = iou(pooled).groupby("variant").agg(["mean", "std"])
            evmean = iou(m).groupby(["variant", "seed"]).mean().groupby("variant").mean()
            for v in pooled.index:
                rows.append({"subset": subset, "metric": metric, "variant": v, "events": len(events),
                             "seeds": m.loc[v].index.get_level_values("seed").nunique(),
                             "pooled_iou": pooled.loc[v, "mean"], "pooled_sd": pooled.loc[v, "std"], "event_mean_iou": evmean[v]})
            for a, b in (p.split(":") for p in args.pairs):
                if a not in m.index.get_level_values("variant") or b not in m.index.get_level_values("variant"):
                    continue
                # pooled delta over seeds; CI by resampling test events (pooled counts recomputed)
                cnt = {v: m.loc[v].groupby("event").sum() / m.loc[v].index.get_level_values("seed").nunique() for v in (a, b)}
                ca, cb = cnt[a].loc[events].to_numpy(), cnt[b].loc[events].to_numpy()
                f = lambda c: c[:, 0] / (c[:, 0] + c[:, 1] + c[:, 2])  # noqa: E731
                delta = f(ca.sum(0, keepdims=True))[0] - f(cb.sum(0, keepdims=True))[0]
                idx = rng.integers(0, len(events), (args.boot, len(events)))
                boot = f(ca[idx].sum(1)) - f(cb[idx].sum(1))
                lo, hi = np.percentile(boot, [2.5, 97.5])
                # two-way: also resample the paired seeds / event-subset draws (matters when draws differ in events)
                seeds = sorted(set(m.loc[a].index.get_level_values("seed")) & set(m.loc[b].index.get_level_values("seed")))
                per = {v: np.stack([m.loc[(v, sd)].reindex(events, fill_value=0).to_numpy() for sd in seeds]) for v in (a, b)}
                sidx = rng.integers(0, len(seeds), (args.boot, len(seeds)))
                boot2 = np.empty(args.boot)
                for k in range(args.boot):
                    sa, sb = per[a][sidx[k]][:, idx[k]].sum((0, 1)), per[b][sidx[k]][:, idx[k]].sum((0, 1))
                    boot2[k] = f(sa[None])[0] - f(sb[None])[0]
                lo2, hi2 = np.percentile(boot2, [2.5, 97.5])
                rows.append({"subset": subset, "metric": metric, "variant": f"{a} - {b}", "events": len(events),
                             "seeds": len(seeds), "pooled_iou": delta, "ci_lo": lo, "ci_hi": hi, "ci2_lo": lo2, "ci2_hi": hi2})
    table = pd.DataFrame(rows)
    print(f"{args.runs} / {args.name}: pooled IoU = from summed confusion counts over test events (mean ± sd over seeds);"
          " deltas: pooled over seeds, 95% CI resampling test events (ci) or paired seeds/draws x test events (ci2)")
    print(table.round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("runs", nargs="+", type=Path)
    e.add_argument("--post", type=Path, required=True)
    e.add_argument("--pre", type=Path, required=True)
    e.add_argument("--name", default="geoid_proto")
    e.add_argument("--batch-size", type=int, default=2)
    e.add_argument("--workers", type=int, default=6)
    r = sub.add_parser("report")
    r.add_argument("--runs", type=Path, required=True)
    r.add_argument("--name", default="geoid_proto")
    r.add_argument("--train-list", type=Path, help="GEOID train list: also report unseen / shared test activations")
    r.add_argument("--pairs", nargs="+", default=["sar_hand:sar_only", "r34_sar_hand:r34_sar_only", "cd_sar_hand:cd_sar",
                                                  "cd_sar:sar_only", "cd_sar_hand:sar_hand", "fm_sar_hand:fm_sar", "fm_sar:sar_only"])
    r.add_argument("--boot", type=int, default=4000)
    r.add_argument("--out", type=Path)
    args = parser.parse_args()
    evaluate_runs(args) if args.cmd == "eval" else report(args)


if __name__ == "__main__":
    main()
