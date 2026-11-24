#!/bin/bash
# Third-review experiments (PREREGISTRATIONS.md #9-#15), Sen1Floods11 training. Every training job is followed by
# the evaluations its analysis needs. Run from the repository root:
#   bash scripts/sweep_revision.sh seeds    # #9 and #15: official-split and all-11-event models, seeds 3-4
#                                           #   (U-Net, ResNet-34; SAR only and SAR + HAND)
#   bash scripts/sweep_revision.sh losses   # #10: physics losses trained on all 11 events, seeds 0-2
#   bash scripts/sweep_revision.sh masks    # #10: HAND post-masks (5, 10, 15, 20 m) on SAR only and SAR + HAND,
#                                           #   seeds 0-4; DEPS=<job ids> to wait for the seeds 3-4 training
#   bash scripts/sweep_revision.sh fm-lr    # #12: TerraMind +/- HAND at four more learning rates, seeds 0-2
# The fixed-epoch learning curve (#14) is `bash scripts/sweep_geoid.sh epochs`.
set -euo pipefail
MODE=${1:?seeds|losses|masks|fm-lr}
ACCOUNT=${ACCOUNT:-uva-mira}
ROOT=${FLOODSEG_ROOT:-/scratch/$USER/floodseg}
R=$ROOT/runs; D=$ROOT/data

train() {  # name seed [overrides...] -> job id
  local name=$1 seed=$2; shift 2
  sbatch --parsable --account="$ACCOUNT" --job-name="${name//\//-}" slurm/train.sbatch configs/base.yaml \
    "name=$name" "seed=$seed" "$@"
}
evaluate() {  # dependency ("" for none) suffix [eval options...] -- run-dirs...
  local deps=$1 sfx=$2; shift 2
  local opts=()
  while [ "$1" != -- ]; do opts+=("$1"); shift; done; shift
  local dep=(); [ -n "$deps" ] && dep=(--dependency=afterok:$deps)
  for n in geoid_test geoid_test_flood; do
    sbatch --parsable --account="$ACCOUNT" "${dep[@]}" --job-name="ext-$n$sfx" slurm/eval_external.sbatch \
      "$D/$n" "$n$sfx" "${opts[@]}" "$@"
  done
  for set in test heldout; do
    [ -n "$sfx" ] && [ $set = heldout ] && continue   # masks: test set only
    sbatch --parsable --account="$ACCOUNT" "${dep[@]}" --job-name="proto-$set$sfx" slurm/eval_geoid_protocol.sbatch \
      $set --name "geoid_proto_$set$sfx" "${opts[@]}" "$@"
  done
}
group() {  # variants seeds [overrides...]: train all on the 11 events, then evaluate them
  local variants=$1 seeds=$2; shift 2
  local ids="" dirs=""
  for v in $variants; do for s in $seeds; do
    ids="$ids:$(train "all/$v" $s split.mode=all ${OVR[$v]} "$@")"; dirs="$dirs $R/all/$v/seed$s"
  done; done
  evaluate "${ids#:}" "" -- $dirs
  echo "${ids#:}"
}

declare -A OVR=(
  [sar_only]="" [sar_hand]="data.channels=[vv,vh,hand]"
  [r34_sar_only]="model.encoder=resnet34" [r34_sar_hand]="model.encoder=resnet34 data.channels=[vv,vh,hand]"
  [sar_mono]="loss.hand_monotonicity.weight=1.0" [sar_ceil]="loss.hand_ceiling.weight=1.0"
  [sar_mono3_w1]="loss.hand_monotonicity.weight=1.0 loss.hand_monotonicity.stride=3 loss.hand_monotonicity.margin=0.0"
  [sar_mono3_w10]="loss.hand_monotonicity.weight=10.0 loss.hand_monotonicity.stride=3 loss.hand_monotonicity.margin=0.0"
)
for lr in 1e-5 2e-5 1e-4 2e-4; do
  OVR[fm_sar_lr$lr]="model.arch=terramind model.bands=[VV,VH] train.lr=$lr"
  OVR[fm_sar_hand_lr$lr]="model.arch=terramind model.bands=[VV,VH,HAND] data.channels=[vv,vh,hand] train.lr=$lr"
done

case $MODE in
  seeds)
    for v in sar_only sar_hand; do for s in 3 4; do train "$v" $s ${OVR[$v]} >/dev/null; done; done   # official split
    group "sar_only sar_hand r34_sar_only r34_sar_hand" "3 4" ;;
  losses) group "sar_mono sar_ceil sar_mono3_w1 sar_mono3_w10" "0 1 2" ;;
  masks)
    dirs=""; for v in sar_only sar_hand; do for s in 0 1 2 3 4; do dirs="$dirs $R/all/$v/seed$s"; done; done
    for t in 5 10 15 20; do evaluate "${DEPS:-}" "_mask$t" --post-mask $t -- $dirs; done ;;
  fm-lr)
    export FLOODSEG_ENV=$HOME/.conda/envs/floodseg-fm
    vs=""; for lr in 1e-5 2e-5 1e-4 2e-4; do vs="$vs fm_sar_lr$lr fm_sar_hand_lr$lr"; done
    group "$vs" "0 1 2" ;;
esac
