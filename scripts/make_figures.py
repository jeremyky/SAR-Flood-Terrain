"""Paper figures from finished runs. Re-run after new runs land; it reads results directly.

    python scripts/make_figures.py --runs /scratch/$USER/floodseg/runs --scratch /scratch/$USER/floodseg --out figures

fig_learning_curve.png   IoU and HAND gain vs number of GEOID training events (both label modes)
fig_tile_control.png     same, with every subset cut to 500 tiles (event count vs data volume)
fig_gain_vs_terrain.png  per-event HAND gain on 79 GEOID test events vs median slope (Sen1Floods11-trained)
fig_backbones.png        HAND gain under the GEOID-Flood protocol for U-Net, change-detection U-Net and
                         TerraMind at 20 and 159 training events (reads reports/paper_20261121/proto_*.csv,
                         from slurm/paper_reports.sbatch)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from external_report import load  # noqa: E402
from learning_curve_report import event_iou, hier_boot, unseen_only  # noqa: E402

# reference palette (dataviz skill), validated as a pair: slot 1 blue, slot 2 orange
COLORS = {"sar_only": "#2a78d6", "sar_hand": "#eb6834"}
LABELS = {"sar_only": "SAR only", "sar_hand": "SAR + HAND"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABEL_MODES = {"geoid_test_flood": "Flooded water only", "geoid_test": "All water (flood + permanent)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def save(fig, path: Path) -> None:
    """PNG for the repo / markdown draft; vector PDF next to it for the manuscript, without the figure
    title (the caption carries it) and cropped to content."""
    fig.savefig(path, dpi=200)
    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)


def curve(runs: Path, curve_dir: str, name: str, full_dir: str | None, full_events: int = 159,
          train_list: Path | None = None, draws: list[str] | None = None) -> pd.DataFrame:
    """Per (size, variant, draw): mean event IoU; plus per (size): paired event deltas with a two-way CI
    (training draws x test events). train_list: keep only test events whose activation is not in it."""
    sources = {int(p.name[1:]): p for p in sorted((runs / curve_dir).glob("e*"))}
    if full_dir:
        sources.setdefault(full_events, runs / full_dir)
    rows = []
    for size, path in sorted(sources.items()):
        try:
            chips = unseen_only(load(path, name), train_list)
            if draws:
                chips = chips[chips.seed.isin(draws)]
            ev = event_iou(chips)
        except ValueError:
            continue
        t = ev.pivot_table(index=["event", "seed"], columns="variant", values="iou")
        if not {"sar_only", "sar_hand"} <= set(t.columns):
            continue
        t = t.dropna()
        for seed, g in t.groupby(level="seed"):
            for v in ("sar_only", "sar_hand"):
                rows.append({"size": size, "variant": v, "seed": seed, "iou": g[v].mean()})
        per_draw = t.sar_hand - t.sar_only  # (event, seed)
        boot = hier_boot(per_draw, np.random.default_rng(0), n=4000)
        mean = per_draw.groupby(level="event").mean().mean()
        rows.append({"size": size, "variant": "delta", "iou": mean, "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)})
    return pd.DataFrame(rows)


def plot_curves(data: dict[str, pd.DataFrame], title: str, path: Path, xlabel: str) -> None:
    fig, axes = plt.subplots(2, len(data), figsize=(3.6 * len(data), 5.2), sharex=True, height_ratios=[1.6, 1])
    axes = np.atleast_2d(axes).reshape(2, len(data))
    for col, (name, df) in enumerate(data.items()):
        top, bottom = axes[0, col], axes[1, col]
        sizes = sorted(df[df.variant != "delta"]["size"].unique())
        ends: dict[str, float] = {}
        for v in ("sar_only", "sar_hand"):
            g = df[df.variant == v].groupby("size").iou
            mean, lo, hi = g.mean(), g.min(), g.max()
            top.fill_between(sizes, lo.loc[sizes], hi.loc[sizes], color=COLORS[v], alpha=0.15, linewidth=0)
            top.plot(sizes, mean.loc[sizes], color=COLORS[v], linewidth=2, marker="o", markersize=5,
                     markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[v])
            ends[v] = mean.loc[sizes[-1]]
        # direct labels at the line ends, pushed apart when the lines finish close together
        lo_y, hi_y = top.get_ylim()
        gap = 0.07 * (hi_y - lo_y)
        y_only, y_hand = ends["sar_only"], ends["sar_hand"]
        if abs(y_hand - y_only) < gap:
            mid = (y_hand + y_only) / 2
            y_hand, y_only = (mid + gap / 2, mid - gap / 2) if y_hand >= y_only else (mid - gap / 2, mid + gap / 2)
        for v, y in (("sar_only", y_only), ("sar_hand", y_hand)):
            top.annotate(LABELS[v], (sizes[-1], ends[v]), xytext=(sizes[-1] * 1.12, y), textcoords="data",
                         va="center", fontsize=8, color=INK2)
        top.set_title(LABEL_MODES[name], loc="left")
        top.set_ylabel("Mean event IoU" if col == 0 else "")
        d = df[df.variant == "delta"].set_index("size").loc[sizes]
        bottom.axhline(0, color=INK2, linewidth=0.8)
        bottom.errorbar(sizes, d.iou, yerr=[d.iou - d.lo, d.hi - d.iou], color=INK, linewidth=2, marker="o",
                        markersize=5, capsize=3, markeredgecolor=SURFACE, markeredgewidth=1.5)
        bottom.set_ylabel("HAND gain (Δ IoU)" if col == 0 else "")
        bottom.set_xlabel(xlabel)
        bottom.set_xscale("log")
        bottom.set_xticks(sizes, [str(s) for s in sizes])
        bottom.minorticks_off()
        top.set_xlim(sizes[0] / 1.25, sizes[-1] * 1.9)  # room for direct labels
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.text(0.01, 0.005, "Bands: range over random event draws. Error bars: 95% two-way bootstrap CI (training draws x test events).",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, path)
    plt.close(fig)


def plot_terrain(explain_csv: Path, path: Path) -> None:
    df = pd.read_csv(explain_csv)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharex=True)
    for ax, (col, title) in zip(axes, (("d_water_unet", "All water"), ("d_flood_unet", "Flooded water only"))):
        m = df[["slope_median", col]].dropna()
        rho, p = spearmanr(m.slope_median, m[col])
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.scatter(m.slope_median, m[col], s=22, color=COLORS["sar_hand"], edgecolor=SURFACE, linewidth=1)
        ax.set_title(f"{title}\nSpearman ρ = {rho:+.2f}, p = {p:.1g}, n = {len(m)}", loc="left", fontsize=9)
        ax.set_xlabel("Median slope of the event's tiles (°)")
    axes[0].set_ylabel("HAND gain per event (Δ IoU)")
    fig.suptitle("HAND gain on 79 GEOID-Flood test events vs terrain (U-Net trained on Sen1Floods11)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)


PAPER_REPORTS = "paper_20261121"  # slurm/paper_reports.sbatch output that the manuscript quotes
BACKBONES = [("sar_hand - sar_only", "U-Net, single image"),
             ("cd_sar_hand - cd_sar", "U-Net, change detection\n(pre + post image)"),
             ("fm_sar_hand - fm_sar", "TerraMind-1.0-base,\nfine-tuned (101 M)")]
SIZES = [("geoid_lc_e20", "20 training events (5 draws)", "o", INK), ("geoid", "159 training events (3 seeds)", "s", "#8a8984")]


def plot_backbones(reports: Path, path: Path) -> None:
    """Forest plot: HAND gain (pooled IoU_flood, GEOID-Flood protocol, two-way CI) per backbone and size."""
    panels = [("test", "unseen", "Test, unseen activations (38 events)"),
              ("heldout", "unseen", "Held-out 2026 events (9)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, (split, subset, title) in zip(axes, panels):
        for j, (size, label, marker, color) in enumerate(SIZES):
            for i, (pair, _) in enumerate(BACKBONES):
                f = reports / PAPER_REPORTS / f"proto_{size}_{split}.csv"
                if not f.exists():
                    continue
                d = pd.read_csv(f)
                r = d[(d.subset == subset) & (d.metric == "flood_diff") & (d.variant == pair)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = i + (-0.17 if j == 0 else 0.17)  # y axis is inverted: 20 events on top
                ax.plot([r.ci2_lo, r.ci2_hi], [y, y], color=color, lw=2, solid_capstyle="round")
                ax.plot(r.pooled_iou, y, marker=marker, ms=8, color=color, mec=SURFACE, mew=1.5,
                        ls="none", label=label if i == 0 else None)
                ax.annotate(f"{r.pooled_iou:+.3f}".replace("-", "\u2212"), (r.ci2_hi, y), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color=INK2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("HAND gain in pooled IoU$_{flood}$ (95% CI)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-0.07, 0.15)
    axes[0].set_yticks(range(len(BACKBONES)), [b[1] for b in BACKBONES])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.02))
    fig.suptitle("HAND gain by model under the GEOID-Flood protocol",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    save(fig, path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--train-list", type=Path, default=None,
                        help="GEOID train list; learning-curve figures then use unseen test activations only")
    parser.add_argument("--tm-draws", nargs="+", default=["seed0", "seed1", "seed2"],
                        help="tile-control draws to plot (default: the exploratory ones, not the confirmation draws)")
    args = parser.parse_args()
    tl = args.train_list
    scope = "38–39 unseen GEOID test activations" if tl else "77–79 GEOID test events"
    args.out.mkdir(exist_ok=True)

    lc = {n: curve(args.runs, "geoid_lc", n, "geoid", train_list=tl) for n in LABEL_MODES}
    plot_curves(lc, f"HAND gain vs number of training events\n(GEOID-Flood train subsets; tested on {scope})",
                args.out / "fig_learning_curve.png", "Training events (log scale)")
    tm = {n: curve(args.runs, "geoid_tm", n, None, train_list=tl, draws=args.tm_draws) for n in LABEL_MODES}
    if all(len(d) for d in tm.values()):
        plot_curves(tm, f"Control: 500 training tiles, drawn from different numbers of events\n(tested on {scope})",
                    args.out / "fig_tile_control.png", "Training events in the 500 tiles (log scale)")
    explain = args.scratch / "external_explain_events.csv"
    if explain.exists():
        plot_terrain(explain, args.out / "fig_gain_vs_terrain.png")
    if (args.scratch / "reports").exists():
        plot_backbones(args.scratch / "reports", args.out / "fig_backbones.png")
    print("wrote", sorted(p.name for p in args.out.glob("*.png")))


if __name__ == "__main__":
    main()
