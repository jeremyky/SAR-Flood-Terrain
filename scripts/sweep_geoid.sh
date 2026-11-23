#!/bin/bash
# GEOID-Flood experiments, as run for the paper (the exact submissions are in experiments/slurm_history.tsv).
#   bash scripts/sweep_geoid.sh full          # 159 events, seeds 0-2            -> runs/geoid/
#   bash scripts/sweep_geoid.sh curve         # 10/20/40/80 events, draws 0-4    -> runs/geoid_lc/
#   bash scripts/sweep_geoid.sh tiles         # 500 tiles from 10..159 events, draws 0-2 (+3-5 at 20/40/159) -> runs/geoid_tm/
#   bash scripts/sweep_geoid.sh ablate        # terrain ablations (DEM + slope without HAND; all three) at 159 events
#                                             # (seeds 0-2) and 20 events (draws 0-4)  -> runs/geoid/, runs/geoid_lc/
#   bash scripts/sweep_geoid.sh jrc           # JRC permanent water (needs slurm/build_jrc.sbatch), same layout as ablate
#   bash scripts/sweep_geoid.sh cd            # change detection: pre + post S1 -> flooded water (vs permanent water and
#                                             # background), +/- HAND; needs slurm/prepare_geoid_cd.sbatch; same layout as ablate
#   bash scripts/sweep_geoid.sh dropout       # HAND modality dropout (p = 0.5, training only), same layout as ablate;
#                                             # scored with both external eval and the GEOID-Flood protocol
#   FM_LR=5e-5 bash scripts/sweep_geoid.sh fm # TerraMind-1.0-base (S1GRD) +/- HAND, same layout as ablate; needs the
#                                             # floodseg-fm env (env/README-fm.md); FM_LR from the validation-only pilot
# Each training job is followed by external evaluation on GEOID test (all water and flood only); change-detection
# runs are scored with the GEOID-Flood protocol instead (scripts/eval_geoid_protocol.py, test and held-out).
# Event subsets must exist first:
#   python scripts/make_event_subsets.py --root $FLOODSEG_ROOT/data/geoid_trainval --sizes 10 20 40 80 --draws 5
#   python scripts/make_event_subsets.py --root ... --sizes 10 20 40 80 159 --draws 3 --tiles 500
#   python scripts/make_event_subsets.py --root ... --sizes 20 40 159 --draws 3 --first-draw 3 --tiles 500
set -euo pipefail
MODE=${1:?full|curve|tiles|ablate|jrc|cd|fm|dropout}
ACCOUNT=${ACCOUNT:-uva-mira}
ROOT=${FLOODSEG_ROOT:-/scratch/$USER/floodseg}
R=$ROOT/runs; D=$ROOT/data
STEPS="train.steps=32670 train.num_validations=30"   # = 30 epochs of the full 8,712-tile train set at batch 8
declare -A CH=([sar_only]="[vv,vh]" [sar_hand]="[vv,vh,hand]" [sar_dem_slope]="[vv,vh,dem,slope]" [sar_terrain]="[vv,vh,dem,hand,slope]"
          [sar_jrc]="[vv,vh,jrc]" [sar_hand_jrc]="[vv,vh,hand,jrc]"
          [cd_sar]="[vv,vh,vv_pre,vh_pre]" [cd_sar_hand]="[vv,vh,vv_pre,vh_pre,hand]"
          [fm_sar]="[vv,vh]" [fm_sar_hand]="[vv,vh,hand]" [sar_hand_md]="[vv,vh,hand]")
declare -A VX=([fm_sar]="model.arch=terramind model.bands=[VV,VH]" [fm_sar_hand]="model.arch=terramind model.bands=[VV,VH,HAND]"
          [sar_hand_md]="data.hand_dropout=0.5")
EXTRA=""   # extra training overrides for every run of the mode
EVAL=external
VARIANTS=${VARIANTS:-"sar_only sar_hand"}

train() {  # name seed [overrides...]
  local name=$1 seed=$2; shift 2
  # 64G: three 32G GEOID jobs failed forking dataloader workers (ENOMEM); memory does not change results
  sbatch --parsable --account="$ACCOUNT" --time=12:00:00 --mem=64G --job-name="${name//\//-}" \
    slurm/train.sbatch configs/geoid.yaml "name=$name" "seed=$seed" "$@"
}
evaluate() {  # dependency-ids run-dirs...
  local deps=$1; shift
  if [ "$EVAL" = protocol ] || [ "$EVAL" = both ]; then
    for set in test heldout; do
      sbatch --parsable --account="$ACCOUNT" --dependency=afterok:$deps slurm/eval_geoid_protocol.sbatch $set "$@" >/dev/null
    done
    [ "$EVAL" = both ] || return 0
  fi
  for n in geoid_test geoid_test_flood; do
    sbatch --parsable --account="$ACCOUNT" --dependency=afterok:$deps slurm/eval_external.sbatch "$D/$n" "$n" "$@" >/dev/null
  done
}
group() {  # dir size list-prefix draws...
  local dir=$1 size=$2 prefix=$3; shift 3
  local ids="" dirs=""
  for r in "$@"; do for v in $VARIANTS; do
    if [ "$dir" = geoid ]; then ids="$ids:$(train geoid/$v $r data.channels=${CH[$v]} ${VX[$v]:-} $EXTRA)"; dirs="$dirs $R/geoid/$v/seed$r"
    else ids="$ids:$(train $dir/e$size/$v $r split.train_list=${prefix}_e${size}_r$r $STEPS data.channels=${CH[$v]} ${VX[$v]:-} $EXTRA)"
         dirs="$dirs $R/$dir/e$size/$v/seed$r"; fi
  done; done
  evaluate "${ids#:}" $dirs
}

case $MODE in
  full)  group geoid 159 train 0 1 2 ;;
  curve) for s in 10 20 40 80; do group geoid_lc $s train 0 1 2 3 4; done ;;
  tiles) for s in 10 20 40 80 159; do group geoid_tm $s train_t500 0 1 2; done
         for s in 20 40 159; do group geoid_tm $s train_t500 3 4 5; done ;;
  ablate) VARIANTS="sar_dem_slope sar_terrain"; group geoid 159 train 0 1 2; group geoid_lc 20 train 0 1 2 3 4 ;;
  jrc)    VARIANTS="sar_jrc sar_hand_jrc"; group geoid 159 train 0 1 2; group geoid_lc 20 train 0 1 2 3 4 ;;
  cd)     VARIANTS="cd_sar cd_sar_hand"; EXTRA="data.root=$D/geoid_trainval_cd"; EVAL=protocol
          group geoid 159 train 0 1 2; group geoid_lc 20 train 0 1 2 3 4 ;;
  fm)     VARIANTS="fm_sar fm_sar_hand"; EXTRA="train.lr=${FM_LR:?set FM_LR}"
          export FLOODSEG_ENV=$HOME/.conda/envs/floodseg-fm
          group geoid 159 train 0 1 2; group geoid_lc 20 train 0 1 2 3 4 ;;
  dropout) VARIANTS="sar_hand_md"; EVAL=both
          group geoid 159 train 0 1 2; group geoid_lc 20 train 0 1 2 3 4 ;;
esac
