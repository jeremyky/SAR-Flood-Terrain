# Terrain helps SAR flood mapping on new floods

Code and paper for an event-level study of whether terrain helps Sentinel-1 flood segmentation models on
flood events they were not trained on.

Jeremy Ky, Bao Duong, Rich Nguyen (University of Virginia).
Preprint (paper + supplement): [`paper/preprint.pdf`](paper/preprint.pdf). LaTeX source: `paper/jstars/`.

## Findings

Models trained on the 11 events of Sen1Floods11, tested on 79 external Copernicus emergency activations
from GEOID-Flood:

- **On new floods, terrain helps clearly.** Adding height above nearest drainage (HAND) as an input
  raises mean event IoU on the 79 external floods by +0.063 for water (0.572 → 0.635, 68 of 79 improved,
  5 seeds) and by +0.032 for flooded water on the 77 with flood. The gain holds without a threshold and
  under GEOID-Flood's own pre/post protocol. On Sen1Floods11's official test tiles, whose events are in
  training, it is +0.008; scored by models that held each tile's event out, the same tiles gain more on 8
  of 10 events, but ten events cannot resolve the difference.
- **It is HAND's drainage structure, not relief alone, and not known water.** Elevation and slope recover
  a quarter of the event-level flood gain; HAND shifted by 500 m or taken from another flood, and a
  permanent-water map, recover none of it.
- **With few training events, a foundation model does not replace terrain.** HAND helps a fine-tuned
  TerraMind (101 M parameters) about as much as a 7.8 M-parameter U-Net in event-mean flood IoU, and the
  small U-Net with HAND beats TerraMind without it.

Also: as a post-processing mask HAND recovers about a third of the input gain on new floods, and as a
physics loss at most a quarter of the flood gain (one of four variants); within Sen1Floods11 neither does
anything. With 159 training events the flood-extent gain is no longer detectable; two remedies for HAND's
failure cases (gating on flat ground, HAND dropout) do not help. Fifteen analyses were pre-registered
(`PREREGISTRATIONS.md`).

## Setup

```bash
conda env create -p ~/.conda/envs/floodseg -f env/environment.lock.yml   # exact versions used
conda activate ~/.conda/envs/floodseg
pip install -e .
pytest -q tests

python scripts/download_sen1floods11.py --root /scratch/$USER/floodseg/data/sen1floods11   # ~0.7 GB
python scripts/build_terrain.py        --root /scratch/$USER/floodseg/data/sen1floods11   # GLO-30 DEM, HAND, slope
```

GEOID-Flood preparation, the TerraMind environment and every experiment are in `REPRODUCE.md`.

## Running

```bash
# one model (SLURM); variants are command-line overrides of the config
sbatch slurm/train.sbatch configs/base.yaml name=sar_hand seed=0 data.channels=[vv,vh,hand]

# leave one event out
sbatch slurm/train.sbatch configs/base.yaml name=loeo/Bolivia/sar_only split.mode=loeo split.held_out=Bolivia
```

Outputs go to `<output_root>/<name>/seed<k>/`: `best.pt` (selected on validation IoU), `config.yaml`
(fully resolved), `history.json`, `metrics.json` and per-chip confusion counts.

## Layout

| path | contents |
|---|---|
| `src/floodseg/` | data loading (Sen1Floods11 / GEOID-Flood tiles + terrain), U-Net and TerraMind models, losses, metrics, training |
| `configs/` | base configs; variants are overrides (`configs/experiments.md`) |
| `scripts/` | data download and preparation, sweeps, evaluation, report and figure scripts |
| `slurm/` | SLURM jobs for training, evaluation, reports, figures and the archive |
| `env/` | locked environments (main and TerraMind) |
| `experiments/slurm_history.tsv` | every SLURM submission of the study, with the exact command |
| `figures/` | paper figures (PNG and PDF) |
| `paper/jstars/` | manuscript and supplement (LaTeX) |
| `tests/` | unit tests for losses, metrics, model and HAND dropout |

## Data

All public: Sen1Floods11 v1.1 (`gs://sen1floods11`), GEOID-Flood (CC BY 4.0), Copernicus GLO-30 DEM
(`s3://copernicus-dem-30m`), ASF GLO-30 HAND (`s3://glo-30-hand`), ESA WorldCover 2021, JRC Global Surface
Water. Terrain is resampled bilinearly onto each tile's 10-m grid, so every channel lines up pixel for
pixel with the labels.

## License

Code: MIT (`LICENSE`). Data are third-party and keep their own licenses; this repository does not
redistribute them.
