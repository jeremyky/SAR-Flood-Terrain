#!/bin/bash
# Submit the experiment matrix (configs/experiments.md) as independent SLURM jobs.
#   bash scripts/sweep.sh official "0 1 2"          # official split, seeds 0-2
#   bash scripts/sweep.sh loeo "0 1 2"              # leave-one-event-out over all 11 events
#   VARIANTS="sar_only sar_hand" bash scripts/sweep.sh loeo 0
#   bash scripts/sweep.sh all "0 1 2"               # train on all 11 events (for external test sets)
set -euo pipefail
MODE=${1:?official|loeo|all}
SEEDS=${2:-0}
ACCOUNT=${ACCOUNT:-uva-mira}
EVENTS="Bolivia Ghana India Mekong Nigeria Pakistan Paraguay Somalia Spain Sri-Lanka USA"

declare -A OVERRIDES=(
  [sar_only]=""
  [sar_hand]="data.channels=[vv,vh,hand]"
  [sar_terrain]="data.channels=[vv,vh,dem,hand,slope]"
  [sar_dem_slope]="data.channels=[vv,vh,dem,slope]"   # terrain without HAND (ablation, 2026-11-21)
  # TerraMind-1.0-base (needs FLOODSEG_ENV=$HOME/.conda/envs/floodseg-fm, env/README-fm.md); LR from the GEOID pilot
  [fm_sar]="model.arch=terramind model.bands=[VV,VH] train.lr=5.0e-5"
  [fm_sar_hand]="model.arch=terramind model.bands=[VV,VH,HAND] data.channels=[vv,vh,hand] train.lr=5.0e-5"
  [sar_mono]="loss.hand_monotonicity.weight=1.0"
  [sar_ceil]="loss.hand_ceiling.weight=1.0"
  [sar_hand_mono]="data.channels=[vv,vh,hand] loss.hand_monotonicity.weight=1.0"
  [sar_hand_ceil]="data.channels=[vv,vh,hand] loss.hand_ceiling.weight=1.0"
  # monotonicity at 30 m (stride 3), no margin, two weights (2026-11 revision; first sweep used stride 1, margin 0.05)
  [sar_mono3_w1]="loss.hand_monotonicity.weight=1.0 loss.hand_monotonicity.stride=3 loss.hand_monotonicity.margin=0.0"
  [sar_mono3_w10]="loss.hand_monotonicity.weight=10.0 loss.hand_monotonicity.stride=3 loss.hand_monotonicity.margin=0.0"
  [sar_hand_mono3_w1]="data.channels=[vv,vh,hand] loss.hand_monotonicity.weight=1.0 loss.hand_monotonicity.stride=3 loss.hand_monotonicity.margin=0.0"
  [sar_hand_mono3_w10]="data.channels=[vv,vh,hand] loss.hand_monotonicity.weight=10.0 loss.hand_monotonicity.stride=3 loss.hand_monotonicity.margin=0.0"
  # JRC GSW permanent water (occurrence >= 50%, scripts/build_jrc.py) on a root whose JRC layer is built the
  # same way as GEOID-Flood's, so train and external test use one definition (2026-11 revision)
  [sar_jrc]="data.channels=[vv,vh,jrc] data.root=${FLOODSEG_ROOT:-/scratch/$USER/floodseg}/data/sen1floods11_gsw"
  [sar_hand_jrc]="data.channels=[vv,vh,hand,jrc] data.root=${FLOODSEG_ROOT:-/scratch/$USER/floodseg}/data/sen1floods11_gsw"
  [r34_sar_only]="model.encoder=resnet34"
  [r34_sar_hand]="model.encoder=resnet34 data.channels=[vv,vh,hand]"
  # HAND modality dropout: HAND replaced by -1 on a random 50% of training chips (2026-11-21)
  [sar_hand_md]="data.channels=[vv,vh,hand] data.hand_dropout=0.5"
  # HAND from a tile of another event, in training and test (mechanism control, PREREGISTRATIONS.md #6)
  [sar_hand_swap]="data.channels=[vv,vh,hand] data.hand_perturb=swap"
)
VARIANTS=${VARIANTS:-"sar_only sar_hand sar_terrain sar_mono sar_ceil sar_hand_mono"}

submit() {  # name, extra overrides...
  local name=$1; shift
  sbatch --account="$ACCOUNT" --job-name="${name//\//-}" slurm/train.sbatch configs/base.yaml "name=$name" "$@"
}

for v in $VARIANTS; do
  for s in $SEEDS; do
    if [ "$MODE" = official ]; then
      submit "$v" seed=$s ${OVERRIDES[$v]}
    elif [ "$MODE" = all ]; then
      submit "all/$v" seed=$s split.mode=all ${OVERRIDES[$v]}
    else
      for e in $EVENTS; do
        submit "loeo/$e/$v" seed=$s split.mode=loeo split.held_out=$e ${OVERRIDES[$v]}
      done
    fi
  done
done
