#!/bin/bash
# Learning-curve + tile-control reports and figures, once their runs are scored.
# Submit with --dependency=afterany:<job ids>.
#SBATCH --job-name=floodseg-lc-report
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/floodseg/logs/%x-%j.out

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FLOODSEG_ENV:-$HOME/.conda/envs/floodseg}"
cd "$SLURM_SUBMIT_DIR"
S=/scratch/$USER/floodseg
OUT=$S/reports/lc_$(date +%Y%m%d_%H%M)
mkdir -p $OUT
{
  echo "# learning curve + tile control, $(date)"
  sacct -u $USER -S 2026-11-02 -X -n -P -o JobName,State | grep -E '^(tm-|lc-|floodseg-external)' | cut -d'|' -f2 | sort | uniq -c
  for name in geoid_test_flood geoid_test; do
    echo; echo "## learning curve (all draws), $name"
    python scripts/learning_curve_report.py --runs $S/runs --name $name --out $OUT/learning_curve_$name.csv
    echo; echo "## tile-matched control (500 tiles), $name"
    python scripts/learning_curve_report.py --runs $S/runs --name $name --curve-dir geoid_tm --full-dir "" --out $OUT/tile_control_$name.csv
    echo; echo "## events vs tiles (pooled draws), $name"
    python scripts/volume_vs_diversity.py --runs $S/runs --name $name --out $OUT/volume_vs_diversity_$name.csv
    echo; echo "## same, >= 20 training events (post hoc sensitivity)"
    python scripts/volume_vs_diversity.py --runs $S/runs --name $name --min-events 20 | tail -7
  done
} > $OUT/report.txt 2>&1
python scripts/make_figures.py --runs $S/runs --scratch $S --out $OUT >> $OUT/report.txt 2>&1
echo "wrote $OUT"
