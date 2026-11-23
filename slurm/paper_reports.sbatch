#!/bin/bash
# Regenerate the report tables that the manuscript quotes (Sections V-B to V-F) (learning curve, equal tiles, protocol), into one dated folder.
# All scripts use fixed bootstrap seeds, so re-running reproduces the numbers exactly.
#   sbatch slurm/paper_reports.sbatch
#SBATCH --job-name=floodseg-paper-reports
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/floodseg/logs/%x-%j.out

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FLOODSEG_ENV:-$HOME/.conda/envs/floodseg}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this checkout's floodseg
S=/scratch/$USER/floodseg
TRAIN=$S/data/geoid_trainval/splits/train.txt
OUT=${OUT:-$S/reports/paper_$(date +%Y%m%d)}
mkdir -p $OUT
CONFIRM=$(echo geoid_tm/e{20,40,159}/seed{3,4,5})  # pre-registered confirmation draws (§5.4)
lc() { python scripts/learning_curve_report.py --runs $S/runs "$@"; }
{
  echo "# paper report tables, $(date), commit $(git rev-parse --short HEAD)"
  for name in geoid_test_flood geoid_test; do
    echo; echo "## §5.3/5.4 learning curve (exploratory draws 0-4, full 159 seeds 0-2), unseen, $name"
    lc --name $name --unseen-only $TRAIN --out $OUT/lc_unseen_$name.csv
    echo; echo "## §5.4 equal-tiles control (500 tiles, exploratory draws 0-2), unseen, $name"
    lc --name $name --curve-dir geoid_tm --full-dir "" --unseen-only $TRAIN --draws seed0 seed1 seed2 \
       --out $OUT/tiles_unseen_$name.csv
    echo; echo "## §5.4 events vs tiles (exploratory draws), unseen, $name"
    python scripts/volume_vs_diversity.py --runs $S/runs --name $name --unseen-only $TRAIN \
       --exclude-draws $CONFIRM --out $OUT/volume_vs_diversity_unseen_$name.csv
  done
  for name in geoid_test_flood geoid_test; do
    for pair in sar_dem_slope:sar_only sar_hand:sar_only sar_terrain:sar_only sar_hand:sar_dem_slope sar_terrain:sar_hand \
                sar_jrc:sar_only sar_hand:sar_jrc sar_hand_jrc:sar_hand; do
      echo; echo "## §5.4 terrain / JRC ablation, unseen, $name, ${pair%:*} vs ${pair#*:}"
      lc --name $name --unseen-only $TRAIN --variant ${pair%:*} --base ${pair#*:} --out $OUT/ablate_${name}_${pair/:/_vs_}.csv
    done
  done
  echo; echo "## §5.3 shared activations, geoid_test_flood"
  lc --name geoid_test_flood --shared-only $TRAIN --out $OUT/lc_shared_geoid_test_flood.csv
  echo; echo "## §5.3 held-out 2026, geoid_heldout_flood"
  lc --name geoid_heldout_flood --out $OUT/lc_heldout_geoid_heldout_flood.csv
  for runs in all geoid geoid_lc/e20; do
    for split in test heldout; do
      echo; echo "## §5.2/5.3/5.5 GEOID-Flood protocol, runs/$runs, $split"
      python scripts/eval_geoid_protocol.py report --runs $S/runs/$runs --name geoid_proto_$split --train-list $TRAIN \
         --out $OUT/proto_${runs//\//_}_$split.csv
    done
  done
} > $OUT/report.txt 2>&1
echo "wrote $OUT"
