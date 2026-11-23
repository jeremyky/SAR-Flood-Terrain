#!/bin/bash
# JRC GSW permanent-water layer (occurrence >= 50%, calibrated against Sen1Floods11's shipped layer: IoU 0.91)
# for the GEOID-Flood roots; the *_flood label-mode roots share the S1 grid and get a symlink.
#SBATCH --job-name=floodseg-build-jrc
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/floodseg/logs/%x-%j.out

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FLOODSEG_ENV:-$HOME/.conda/envs/floodseg}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this checkout's floodseg
D=/scratch/$USER/floodseg/data
H=data/flood_events/HandLabeled
for root in geoid_trainval geoid_test geoid_heldout; do
  python scripts/build_jrc.py build --root $D/$root --threshold 50 --workers 32 | awk 'NR % 500 == 0'
done
for pair in geoid_test_flood:geoid_test geoid_heldout_flood:geoid_heldout; do
  dst=${pair%:*}; src=${pair#*:}
  [ -e $D/$dst/$H/JRCWaterHand ] || ln -s $D/$src/$H/JRCWaterHand $D/$dst/$H/JRCWaterHand
done
for root in geoid_trainval geoid_test geoid_heldout; do echo "$root $(ls $D/$root/$H/JRCWaterHand | wc -l) / $(ls $D/$root/$H/S1Hand | wc -l)"; done
echo "JRC DONE"
