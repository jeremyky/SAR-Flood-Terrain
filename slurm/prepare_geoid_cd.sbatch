#!/bin/bash
# Inputs for the change-detection baseline (pre + post S1 -> flooded water; scripts/sweep_geoid.sh cd).
#   geoid_trainval_pre  latest pre-event S1 for the GEOID train/val tiles
#   geoid_trainval_cd   post S1 + pre S1 (S1PreHand) + terrain, labels flooded water = 1, permanent water and
#                       background = 0 (the GEOID-Flood protocol); split lists = the geoid_trainval lists
#                       restricted to tiles that have a pre-event image
#   geoid_{test,heldout}_raw/.../S1PreHand -> the pre-event images already built by prepare_geoid_protocol.sbatch
# Needs prepare_geoid_trainval.sbatch, prepare_geoid_protocol.sbatch and make_event_subsets.py first.
#SBATCH --job-name=floodseg-geoid-cd-prep
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/floodseg/logs/%x-%j.out

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FLOODSEG_ENV:-$HOME/.conda/envs/floodseg}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this checkout's floodseg
S=/scratch/$USER/floodseg
HL=data/flood_events/HandLabeled
TV=$S/data/geoid_trainval; PRE=$S/data/geoid_trainval_pre; CD=$S/data/geoid_trainval_cd
GEO="--geoid $S/geoid_data/geoid-flood --catalog $S/geoid_hf/geoid-flood/tile_catalog.parquet --workers 32"
mkdir -p $CD/$HL $CD/splits
ln -sfn $TV/$HL/S1Hand $CD/$HL/S1Hand
ln -sfn $PRE/$HL/S1Hand $CD/$HL/S1PreHand
ln -sfn $TV/terrain $CD/terrain
for split in train val; do
  python scripts/prepare_geoid.py $GEO --split $split --out $PRE --image pre
  python scripts/prepare_geoid.py $GEO --split $split --out $CD --label-mode flood_vs_rest
done
python - "$TV" "$CD" <<'PY'
import sys
from pathlib import Path
tv, cd = map(Path, sys.argv[1:])
have = {p.name.removesuffix("_S1PreHand.tif") for p in (cd / "data/flood_events/HandLabeled/S1PreHand").iterdir()}
have = {h.removesuffix("_S1Hand.tif") for h in have}
for lst in sorted((tv / "splits").glob("*.txt")):
    chips = [c for c in lst.read_text().split() if c]
    kept = [c for c in chips if c in have]
    ev = lambda cs: {c.rsplit("_", 1)[0] for c in cs}
    (cd / "splits" / lst.name).write_text("\n".join(kept) + "\n")
    print(f"{lst.name}: {len(kept)}/{len(chips)} tiles, {len(ev(kept))}/{len(ev(chips))} events")
PY
for set in test heldout; do ln -sfn $S/data/geoid_${set}_pre/$HL/S1Hand $S/data/geoid_${set}_raw/$HL/S1PreHand; done
echo "CD PREP DONE"
