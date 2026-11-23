# Terrain helps SAR flood mapping on new floods, and same-event splits hide it

Code and paper for an event-level study of whether terrain helps Sentinel-1 flood segmentation models on
flood events they were not trained on.

Jeremy Ky, Bao Duong, Rich Nguyen (University of Virginia).
Manuscript: `paper/jstars/` (IEEE two-column), with supplementary material.

## Findings

Models trained on the 11 events of Sen1Floods11, tested on 79 external Copernicus emergency activations
from GEOID-Flood:

- **Same-event splits hide the value of terrain.** Adding height above nearest drainage (HAND) as an input
  gains +0.007 IoU on Sen1Floods11's official test split (tiles from the training events), but +0.128 on
  its held-out Bolivia event and +0.061 mean event IoU on the 79 external floods (69 of 79 improved,
  p < 1e-4). The gain holds without a threshold, under GEOID-Flood's own pre/post protocol, and in event
  IoU for a second backbone.
- **It is HAND's drainage structure, not relief alone, and not known water.** Elevation and slope recover
  a quarter of the event-level flood gain; a permanent-water map none of it.
- **A foundation model does not replace terrain.** HAND helps a fine-tuned TerraMind (101 M parameters)
  about as much as a 7.8 M-parameter U-Net in event-mean flood IoU, and the small U-Net with HAND beats
  TerraMind without it.

Also: HAND as a physics loss or post-processing mask does nothing; with 159 training events the
flood-extent gain is no longer detectable; two remedies for HAND's failure cases (gating on flat ground,
HAND dropout) do not help. Five analyses were pre-registered (`PREREGISTRATIONS.md`).

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
