#!/bin/bash
# Convert GEOID-Flood test tiles to the Sen1Floods11 layout and build terrain for them.
#SBATCH --job-name=floodseg-geoid-prep
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/floodseg/logs/%x-%j.out

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FLOODSEG_ENV:-$HOME/.conda/envs/floodseg}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this checkout's floodseg
S=/scratch/$USER/floodseg
OUT=$S/data/geoid_test
# EMSR388 (Spain, Sept 2019) is the same flood as Sen1Floods11's Spain event
python scripts/prepare_geoid.py --geoid $S/geoid_data/geoid-flood --catalog $S/geoid_hf/geoid-flood/tile_catalog.parquet \
  --split test --out $OUT --exclude EMSR388 --workers 16
python scripts/build_terrain.py --root $OUT --workers 16 > $S/logs/geoid_terrain.log
grep -vc "nan(dem,hand)=0.000,0.000" $S/logs/geoid_terrain.log || true
# flood-only labels (permanent water -> ignore), sharing S1 and terrain with geoid_test
OF=$S/data/geoid_test_flood
mkdir -p $OF/data/flood_events/HandLabeled
ln -sfn $OUT/data/flood_events/HandLabeled/S1Hand $OF/data/flood_events/HandLabeled/S1Hand
ln -sfn $OUT/terrain $OF/terrain
python scripts/prepare_geoid.py --geoid $S/geoid_data/geoid-flood --catalog $S/geoid_hf/geoid-flood/tile_catalog.parquet \
  --split test --out $OF --exclude EMSR388 --label-mode flood --workers 16
echo "PREP DONE"
