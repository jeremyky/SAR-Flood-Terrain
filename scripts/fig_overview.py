"""Overview figure (manuscript Fig. 1): models trained on the 11 Sen1Floods11 events.

(a) HAND gain (SAR + HAND minus SAR only, same seed) in event-mean water IoU, by test setting:
    Sen1Floods11's 90 official test tiles scored by official-split models (their events are in training) and
    by the leave-one-event-out models that held each tile's event out (PREREGISTRATIONS.md #9); the Bolivia
    hold-out (one event, so event-mean = pooled IoU); 79 external GEOID-Flood events. Dots: seeds; diamonds:
    means; bars: two-way 95% CIs over seeds and events (none for Bolivia, a single event).
(b) Gain over the SAR-only U-Net in event-mean flood IoU on 77 GEOID-Flood events for terrain inputs and for
    TerraMind with and without HAND (two-way 95% CIs). + HAND uses every seed (0-4, #15); the other rows use
    the seeds of Tables IV and VI (0-2).

    python scripts/fig_overview.py --scratch /scratch/$USER/floodseg --out figures/fig_overview.png

Reads reports/official_split.csv (scripts/official_report.py), reports/matched_split.csv
(scripts/matched_split_report.py), reports/transfer_headline.csv (scripts/revision_transfer_report.py headline)
and reports/ablation_all.csv (scripts/ablation_all_report.py --seeds seed0 seed1 seed2; or --fm-report); only the
TerraMind + HAND row, which no table reports, is computed here with the same bootstrap (hand_gate.compare).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import matched_split_report as msr  # noqa: E402
from hand_gate import compare, load  # noqa: E402
from make_figures import COLORS, INK2, SURFACE, plt, save  # noqa: E402

GREY = "#8a8984"


def seed_gains(d: pd.DataFrame, x: str, y: str) -> pd.Series:
    """Per-seed event-mean IoU of x minus y (events with labelled pixels), as hand_gate.compare."""
    ev = d.groupby(["variant", "seed", "event"])[["tp", "fp", "fn"]].sum()
    pos = (ev.tp + ev.fn).groupby("event").max() > 0
    ev = ev[ev.index.get_level_values("event").isin(pos[pos].index)]
    iou = ev.tp / (ev.tp + ev.fp + ev.fn)
    return (iou.loc[x] - iou.loc[y]).dropna().groupby("seed").mean()


def row(ax, i: int, seeds, mean: float, ci: tuple[float, float] | None, color: str) -> None:
    if ci is not None:
        ax.plot(ci, [i, i], color=color, lw=2, solid_capstyle="round")
    ax.plot(seeds, [i + 0.28] * len(seeds), "o", ms=3.5, color=color, alpha=0.55, mec="none")  # seeds, just below
    ax.plot(mean, i, "D", ms=7, color=color, mec=SURFACE, mew=1.2)
    right = max(max(seeds), ci[1] if ci is not None else -np.inf)
    ax.annotate(f"{mean:+.3f}".replace("-", "−"), (right, i), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=7.5, color=INK2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("figures/fig_overview.png"))
    p.add_argument("--fm", nargs=2, default=["fm_sar", "fm_sar_hand"], metavar=("TM", "TM_HAND"),
                   help="TerraMind run dirs under runs/all (the learning rates selected in PREREGISTRATIONS.md #12)")
    p.add_argument("--fm-report", default="ablation_all.csv",
                   help="report under reports/ holding '<TM> - sar_only' (Table VI): ablation_all.csv or transfer_fm_lr.csv")
    args = p.parse_args()
    rep, runs = args.scratch / "reports", args.scratch / "runs"
    c = COLORS["sar_hand"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.45), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.95})

    # (a) HAND gain by test setting, event-mean water IoU
    m = pd.read_csv(rep / "matched_split.csv")
    m = m[m.metric == "event-mean IoU"].set_index("quantity")
    same, held, events = msr.load(runs)
    labels = []
    for i, (setting, q, label) in enumerate(((same, "HAND gain, same event (official split)",
                                              "official test tiles,\nevent in training"),
                                             (held, "HAND gain, event held out (LOEO)",
                                              "same tiles,\nevent held out"))):
        arr, _ = msr.arrays(setting, events)
        g = (msr.iou(arr[:, :, 1]) - msr.iou(arr[:, :, 0])).mean(1)
        row(a, i, g, m.loc[q, "value"], (m.loc[q, "lo"], m.loc[q, "hi"]), c)
        labels.append(label)
    off = pd.read_csv(rep / "official_split.csv").set_index(["split", "variant"]).loc[("bolivia", "sar_hand")]
    bol = [float(s) for s in off.delta_by_seed.split()]
    row(a, 2, bol, float(np.mean(bol)), None, c)
    labels.append("Bolivia\n(held out, 1 event)")
    head = pd.read_csv(rep / "transfer_headline.csv").set_index(["test", "comparison"])
    r = head.loc[("all water", "sar_hand - sar_only")]
    water = load(runs / "all", "geoid_test")
    row(a, 3, seed_gains(water, "sar_hand", "sar_only"), r.event_mean, (r.ev_lo, r.ev_hi), c)
    labels.append(f"{int(r.events)} GEOID-Flood\nevents (external)")
    a.set_yticks(range(4), labels)
    a.set_xlabel("HAND gain\n(event-mean water IoU)")
    a.set_title("(a) HAND gain by test setting", loc="left")

    # (b) what helps on new floods
    abl = pd.read_csv(rep / "ablation_all.csv").set_index(["test", "comparison"])
    flood = load(runs / "all", "geoid_test_flood")
    tm, tmh = args.fm
    fmr = pd.read_csv(rep / args.fm_report).set_index(["test", "comparison"])
    fl = flood.assign(variant=flood.variant.replace({tm: "TM", tmh: "TM_HAND"}))
    items = [("+ DEM + slope", abl.loc[("flooded only", "sar_dem_slope - sar_only")], GREY),
             ("+ HAND", head.loc[("flooded only", "sar_hand - sar_only")], c),
             ("+ HAND, DEM, slope", abl.loc[("flooded only", "sar_terrain - sar_only")], c),
             ("TerraMind (101 M)", fmr.loc[("flooded only", f"{tm} - sar_only")], COLORS["sar_only"]),
             ("TerraMind + HAND", pd.Series(compare(fl, "TM_HAND", "sar_only", np.random.default_rng(0), 10000)), COLORS["sar_only"])]
    for i, (label, r, col) in enumerate(items):
        b.plot([r.ev_lo, r.ev_hi], [i, i], color=col, lw=2, solid_capstyle="round")
        b.plot(r.event_mean, i, "o", ms=7, color=col, mec=SURFACE, mew=1.2)
        b.annotate(f"{r.event_mean:+.3f}".replace("-", "−"), (r.ev_hi, i), xytext=(6, 0),
                   textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    b.set_yticks(range(len(items)), [it[0] for it in items])
    b.set_xlabel("Gain over SAR-only U-Net (7.8 M)\n(event-mean flood IoU)")
    b.set_title(f"(b) Gains on {int(items[1][1]['events'])} new floods", loc="left")

    for ax in (a, b):
        ax.axvline(0, color=INK2, lw=0.8)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    a.set_xlim(-0.05, 0.24)
    b.set_xlim(-0.08, 0.10)
    fig.suptitle("Models trained on the 11 Sen1Floods11 events", x=0.01, ha="left", fontsize=10)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.84, bottom=0.25)
    save(fig, args.out)
    print({k: {q: round(float(it[1][q]), 4) for q in ("event_mean", "ev_lo", "ev_hi", "seeds")}
           for k, it in zip([i[0] for i in items], items)})


if __name__ == "__main__":
    main()
