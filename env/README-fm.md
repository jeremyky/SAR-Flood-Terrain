# Foundation-model env (TerraMind baseline)

TerraMind runs need terratorch, kept out of the main env so the U-Net results stay on the locked env.

    conda create --clone floodseg -p ~/.conda/envs/floodseg-fm
    ~/.conda/envs/floodseg-fm/bin/pip install terratorch==1.2.11     # leaves torch/numpy/timm unchanged
    # full pip freeze: env/requirements-fm.lock.txt

Weights (`ibm-esa-geospatial/TerraMind-1.0-base`, Apache-2.0) download to $HF_HOME (default /scratch/$USER/hf)
on first use. Jobs select the env with `FLOODSEG_ENV=$HOME/.conda/envs/floodseg-fm` (see scripts/sweep_geoid.sh tm).
