#!/bin/bash
# Download GEOID-Flood (Chiriaco et al., arXiv 2608.02315; CC BY 4.0) as used in this study.
# Uses the dataset's own downloader at the code commit we used; the Hugging Face dataset revision
# we downloaded was 868407460bf3db492f50730a57585916baa71dc6 (get_data.py fetches the current one:
# check the README of links-ads/geoid-flood for changes if numbers differ).
#
#   FLOODSEG_ROOT=/scratch/$USER/floodseg bash scripts/download_geoid.sh     # ~190 GB
set -euo pipefail
ROOT=${FLOODSEG_ROOT:-/scratch/$USER/floodseg}
cd "$ROOT"
mkdir -p logs geoid_hf
[ -d geoid-flood-code ] || git clone -q https://github.com/links-ads/geoid-flood.git geoid-flood-code
git -C geoid-flood-code checkout -q 3a1647ed11aea0c81d5f59a8022e9cb8548bddab

# catalogs (tile_catalog.parquet per tree) and README
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="links-ads/geoid-flood", repo_type="dataset",
                  allow_patterns=["*.md", "*.json", "*/tile_catalog.parquet", "*/*.csv"], local_dir="geoid_hf")
EOF

G=geoid-flood-code/scripts/get_data.py
python $G --dest geoid_data --tree geoid-flood --split test      --layer s1grd label --workers 6 > logs/geoid_download_test.log 2>&1
python $G --dest geoid_data --tree geoid-flood --split train val --layer s1grd label --workers 6 > logs/geoid_download_trainval.log 2>&1
python $G --dest geoid_data --tree geoid-flood-heldout           --layer s1grd label --workers 6 > logs/geoid_download_heldout.log 2>&1
echo "done; next: sbatch slurm/prepare_geoid.sbatch and slurm/prepare_geoid_trainval.sbatch (see REPRODUCE.md)"
