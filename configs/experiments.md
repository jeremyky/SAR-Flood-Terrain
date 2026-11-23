# Experiment variants

Every variant is `configs/base.yaml` (Sen1Floods11) or `configs/geoid.yaml` (GEOID-Flood) plus command-line
overrides. The authoritative definitions are in `scripts/sweep.sh` (`OVERRIDES`, Sen1Floods11 training) and
`scripts/sweep_geoid.sh` (`CH`, `VX`, GEOID-Flood training); `REPRODUCE.md` maps each paper table and figure
to the sweep that produced it.

| name | what changes |
|---|---|
| sar_only | (none) |
| sar_hand | `data.channels=[vv,vh,hand]` |
| sar_dem_slope | `data.channels=[vv,vh,dem,slope]` |
| sar_terrain | `data.channels=[vv,vh,dem,hand,slope]` |
| sar_jrc, sar_hand_jrc | JRC permanent-water layer added (without / with HAND) |
| r34_* | ResNet-34 encoder (`model.encoder=resnet34`) |
| fm_sar, fm_sar_hand | TerraMind-1.0-base (needs the TerraMind env, `env/README-fm.md`) |
| sar_mono*, sar_ceil, sar_hand_mono*, sar_hand_ceil | HAND physics losses (monotonicity, ceiling) |
| sar_hand_md | HAND modality dropout (`data.hand_dropout=0.5`) |
| sar_hand_swap | HAND from another event in training and test (`data.hand_perturb=swap`) |

The HAND post-mask baseline needs no training: every run also reports `hand_mask` metrics next to `raw`.
Test-time HAND perturbations (`shift:<px>`, `blur:<px>`, `swap`) are given to `scripts/eval_external.py`
with `--hand-perturb`.
