# End-to-end learning of national-scale river discharge simulation over Japan

Code, trained weights and instructions to reproduce the experiments of the paper

> Hascoet, T., Pellet, V., Oishi, S., & Miyoshi, T. **Leveraging Japan's
> streamflow records for end-to-end learning of hydrological simulations at
> national scale.**

The model simulates daily river discharge at the outlet of each of the 8,893
catchments (mean area 36.5 km²) of a river network covering Japan. It has three
parts, trained jointly end-to-end against observed discharge:

1. a **runoff module**: an LSTM that maps each catchment's forcing (in-situ
   precipitation and air temperature in 4-hour bins) and static descriptors
   (topography, land use, seasonality) to the runoff at the catchment outlet;
2. a **routing-parameter module**: an MLP that maps each river reach's channel
   length to the time constant of its routing kernel;
3. a **differentiable river-routing scheme**
   ([DiffRoute](https://github.com/TristHas/DiffRoute), linear-storage kernel)
   that carries the runoff downstream through the network.

The code is built on [DiffHydro](https://github.com/TristHas/DiffHydro), which
provides the river-network graph, the coupled runoff–routing model and its
training and inference pipelines, with DiffRoute for the routing and
[xtensor](https://github.com/TristHas/xtensor) for the labelled tensors;
`requirements.txt` installs the three at the versions used for the paper.

It is trained and evaluated on the **D3P-JP** dataset; its simulation over the
whole network is released as **D3P-JP-Q**:

| | Hugging Face (fastest, most recent version) | Zenodo (permanent snapshot of the paper) |
|---|---|---|
| D3P-JP, the dataset | <https://huggingface.co/datasets/tristhas/D3P-JP> | <https://doi.org/10.5281/zenodo.24212554> |
| D3P-JP-Q, the simulation | <https://huggingface.co/datasets/tristhas/D3P-JP-Q> | <https://doi.org/10.5281/zenodo.24212556> |

This repository contains what is needed to download the dataset and, from it,
to train the model of the released simulation (section 4), to evaluate the
model in the four settings of the paper (PGB, PGB-T, PUB, PUB-T; section 5)
and to run the sensitivity experiments (section 6).

---

## 1. Quick start

Requirements: Linux, Python 3.11–3.13, `git`, disk space for the dataset
(section 3), and an NVIDIA GPU: 16 GB of GPU memory is enough to run the
trained models (the paper's training runs used 80 GB A100s). 

```bash
# 1. code
git clone https://github.com/TristHas/jpdh.git
cd jpdh

# 2. environment (conda shown; a plain `python -m venv .venv` works the same way)
conda create -n jpdh python=3.13 -y
conda activate jpdh
# install PyTorch for your CUDA driver first (see `nvidia-smi`), e.g. CUDA 12.8:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 3. data (into ./data; see section 3)
python scripts/download_data.py
python scripts/verify_data.py

# 4. run the trained model over the whole river network, 2009-2020
python scripts/simulate.py                 # -> results/simulation.zarr
```

The shipped model, `weights/full_data_seed00.pt`, is the model of the released
simulation D3P-JP-Q: the default configuration trained on all 332 training
gauges over 2009–2020, with no river basin and no year withheld. Step 4 runs it
over the 8,893 catchments in about a minute on one A100 GPU (13 GB of GPU
memory with the default 12 groups of basins; a larger `--n_groups` needs less)
and writes the daily discharge at every catchment outlet: the D3P-JP-Q
simulation. To compare it with the observations at a gauge:

```python
import xarray as xr
sim = xr.open_zarr("results/simulation.zarr")["discharge"]       # (time, spatial), m3/s, every catchment
obs = xr.open_zarr("data/daily/discharge.zarr")["discharge"]     # (time, spatial), m3/s, the 979 gauges
gauge = int(obs.spatial[0])                                      # node_id of a gauged catchment
pair = xr.Dataset({"simulated": sim.sel(spatial=gauge),
                   "observed": obs.sel(spatial=gauge, time=sim.time)}).to_dataframe()
print(pair[["simulated", "observed"]].head())
```

---

## 2. What is in this repository

```
train.py                 train and evaluate a configuration over the 10 PUB folds
src/                     the code (put on the path by every entry point; nothing to install)
  config.py              input variables, default configuration, dataset location
  data.py                network, folds, temporal blocks, forcing and discharge loading
  model.py               coupled model: LSTM runoff module, routing MLP, routing kernels
  training.py            loss, schedules, SWA weights, per-fold training loop
  lumped.py              lumped training of the runoff module (two-stage recipe)
  inference.py           continuous forward passes over the whole period, basin by basin
  weights.py             model files: save / load a trained model with its configuration
  metrics.py             per-gauge scores (NSE, KGE' and components, nRMSD, FDC signatures)
  provenance.py          library versions recorded in every result
experiments/             one launcher per experiment of the paper (sections 4-6)
scripts/
  download_data.py       fetch, verify and extract D3P-JP
  verify_data.py         check a downloaded dataset
  simulate.py            run a model file over the river network
  evaluate_pgbt.py       PGB-T scores from the PUB-T runs
  table_settings.py      Table 3: the four evaluation settings
  table_sensitivity.py   the numbers of Figure 9: the sensitivity experiments
  collect_runs.py        series and per-gauge scores of several seeds' runs
  export_weights.py      training-run pickle -> model files
  merge_folds.py         merge folds trained in parallel into one run
weights/                 trained models of the default configuration (section 7)
tests/                   unit tests on a synthetic river network (`pytest -q`)
```

### The default configuration

`train.py`'s defaults are the default configuration of the paper (Table 2):

| Component | Setting |
|---|---|
| Dynamic inputs | in-situ gauge-interpolated precipitation + JMA MSM analysis air temperature, six 4-hour bins of each 12Z day |
| Static inputs | elevation mean and standard deviation, catchment area, JAXA land-cover fractions, monthly climatology of the MSM analysis precipitation |
| Runoff module | 2-layer LSTM, 256 hidden units, softplus output (mm per day), output bias initialised at the training gauges' mean specific runoff |
| Routing | DiffRoute linear-storage kernel; a catchment's own runoff is not routed through its own reach |
| Routing parameters | MLP (3 × 256, ReLU) on the z-scored channel length; time constant τ = 0.01 + 9.89·sigmoid(·) hours, initialised at 4.7 % of the range |
| Optimisation | AdamW, 2,400 gradient updates (120 epochs × 20 windows of 365 warm-up + 100 scored days), cosine decay to zero, learning rates 1e-3 (runoff) and 1e-4 (routing), gradient norm clipped at 1 |
| Loss | per-station 1 − NSE, each station's term clipped at 5 |
| Reported weights | stochastic weight averaging over the last 25 % of the training steps |

The schedule is fixed and no model is selected on validation scores: under
PUB the validation gauges of a fold lie in its held-out river basins.

One `train.py` run trains the 10 folds of one seed (about 45 minutes on one
A100) and writes `results/<seed>_<name>.pkl` with the per-gauge scores and,
with `--save_series` / `--save_weights`, the series and the weights:

```bash
python train.py --exp_name default --save_series --save_weights
python train.py --exp_name quick --max_folds 1 --n_epoch 5      # a short check
python train.py --help                                          # every option
```

Folds are seeded by their index, so they can run on several GPUs
(`--folds 0,1,2` …) and be merged with `scripts/merge_folds.py`. Training is
not bit-reproducible on a GPU: two runs of the same seed differ by a few
thousandths of NSE at a gauge, like two different seeds.

---

## 3. Dataset

`scripts/download_data.py` fetches the archives of D3P-JP from its Hugging
Face repository, checks their SHA-256 against the repository's `MANIFEST.json`
and extracts them.

| Archive | Option | Size | Contents |
|---|---|---|---|
| core | (always) | 1.2 GB | river network and catchment geometries, static descriptors, inputs of the default configuration, quality-controlled daily discharge at 979 gauges, records of 116 dams, PUB folds and temporal blocks, quality-control tables |
| panels | `--panels` | 13.3 GB | inputs of the sensitivity experiments: four other precipitation products, 1-, 2-, 6- and 12-hour binnings, MSM wind, pressure and cloud cover |

Extracted layout (`$JPDH_DATA_ROOT`, default `./data`):

```
network/   nodes.parquet (node table: coordinates, areas, channel length, gauge/dam flags)
           edges.parquet (downstream edges), catchments.parquet, channels.parquet (geometries)
           stations.parquet (gauges and dams: names, river, metadata, status of each discharge record)
statics/   runoff_statics.parquet, routing_statics.parquet
daily/     forcing.zarr, [forcing_panels.zarr], discharge.zarr, dams.zarr
splits/    pub_folds_10.json, time_blocks.json
qc/        station_decisions.parquet, discharge_exclusions.parquet
checksums/ one SHA-256 list per archive (checked by verify_data.py)
```

**Time convention.** Daily values are 12Z days: the label `YYYY-MM-DD 12:00`
(UTC) covers 12:00 UTC of that day to 12:00 UTC of the next (21:00 to 21:00
JST). Forcing and discharge share one axis, 2008-01-01 to 2020-12-30; the first
365 days are the models' warm-up and scores are computed over 2009-01-01 to
2020-12-30.

**Folds.** `splits/pub_folds_10.json` holds the 10 folds of whole river
basins. The 979 evaluation gauges are the quality-controlled gauges with at
least 364 valid days in 2009–2020; the 332 training gauges are those of them
with no dam upstream. Per fold, `train` is the training gauges of the other
nine folds, `val` the training gauges of the held-out fold (monitoring only),
and `test` every evaluation gauge of the held-out fold, so each evaluation
gauge is tested exactly once. `time_blocks.json` defines the three 4-year
blocks B1 2009–2012, B2 2013–2016 and B3 2017–2020.

The `README.md` of the dataset describes every variable, unit and source.

---

## 4. Training the model of the released simulation

D3P-JP-Q is the simulation of the default configuration trained on all 332
training gauges over 2009–2020, with no river basin and no year withheld
(seed 0):

```bash
python experiments/full_data.py --seeds 0-0                       # about 6 minutes on one A100
python scripts/export_weights.py results/full_data/00_full_data.pkl --out results/weights --name full_data_seed00
python scripts/simulate.py --model results/weights/full_data_seed00.pt --runoff --out results/simulation.zarr
```

`simulate.py` writes `discharge(time, spatial)` in m³ s⁻¹ and, with `--runoff`,
the catchments' own runoff in mm day⁻¹, for every catchment and every day of
2009–2020. `weights/full_data_seed00.pt` is the model file of D3P-JP-Q itself:
`python scripts/simulate.py --runoff` reproduces the released store.

---

## 5. The four evaluation settings (Table 3)

| Setting | Withheld from training | Runs |
|---|---|---|
| PGB | nothing: the released simulation scored at the gauges | `experiments/full_data.py`, 10 seeds |
| PGB-T | the scored four-year block; the gauge's river basin is seen | the PUB-T runs, scored by `scripts/evaluate_pgbt.py` |
| PUB | the gauge's river basin; all twelve years are used | `experiments/settings.py --cells pub`, 10 seeds |
| PUB-T | both | `experiments/settings.py --cells pubt`, 10 seeds × 3 blocks |

```bash
python experiments/settings.py --cells pub --seeds 0-9        # 10 runs of about 45 minutes
python experiments/settings.py --cells pubt --seeds 0-9       # 30 runs of about 45 minutes
python experiments/full_data.py --seeds 0-9                   # 10 runs of about 6 minutes
python scripts/evaluate_pgbt.py results/settings/pubt --out results/settings/pgbt   # 30 runs of about 2 minutes
python scripts/table_settings.py
```

`table_settings.py` prints the table and writes it to `results/`: per setting,
the median NSE and KGE' over all 979 evaluation gauges and over the 332
unregulated ones, computed on each four-year block (gauges with at least 364
valid days in the block), averaged over the three blocks, as mean ± standard
deviation over the seeds. A setting whose runs are missing is left out, so the
table can be built on fewer seeds or settings.

Every launcher of `experiments/` runs its configurations as `train.py`
commands, seed by seed, into `results/<experiment>/`, skipping runs whose
output exists (so re-running resumes). `--dry_run` prints the commands (e.g.
to submit them to a scheduler), `--shard i/n` splits them across `n`
processes, `--seeds` selects the seeds, and any unknown flag is passed on to
`train.py`.

`scripts/collect_runs.py results/settings/pub --out results/collected/pub`
gathers the simulated series of the seeds in one store with their per-gauge
scores (for PUB-T runs, the three block runs of a seed are stitched so that
every day comes from the run that withheld it).

---

## 6. Sensitivity experiments (Figure 9)

Each launcher varies one factor of the default configuration, in the PUB
setting; the default configuration itself is the `pub` cell of section 5.

| Panel | Launcher | Factor | Runs |
|---|---|---|---|
| a | `precipitation.py` | precipitation product: gauge-adjusted radar, radar, MSM analysis, ERA5-Land | 4 × 3 seeds |
| b | `temporal_resolution.py` | temporal binning of the inputs: 1, 2, 6, 12 hours | 4 × 3 seeds |
| c | `dynamic_inputs.py` | additional MSM inputs: cloud cover, wind, surface pressure, all | 4 × 3 seeds |
| d | `routing_scheme.py` | routing kernel: Nash cascade, Muskingum, pure lag, Hayami | 4 × 3 seeds |
| e | `runoff_statics.py` | runoff static descriptor groups used in isolation | 13 × 3 seeds |
| g, h | `training_recipe.py` | two-stage training (lumped runoff module, then routing calibrated alone) and no routing | 3 arms × 10 seeds |

```bash
python scripts/download_data.py --panels                        # inputs of panels a-c
python experiments/precipitation.py --seeds 0-2
python experiments/training_recipe.py --arms lumped --seeds 0-9
python experiments/training_recipe.py --arms two_stage,noroute --seeds 0-9
python scripts/table_sensitivity.py
```

`table_sensitivity.py` prints, for every configuration found under
`results/`, the median NSE over the 979 evaluation gauges (2009–2020 as one
series; mean ± standard deviation over the seeds) and, for the training
recipes, the scores by tercile of drainage area (panel h).

Panel f (resolution of the river network) compares datasets built on other
meshes; D3P-JP is distributed on the 40 km² mesh only, so this panel cannot be
reproduced from it.

The paper's computations ran on NVIDIA A100 GPUs (80 GB) with Python 3.13,
PyTorch 2.9 (CUDA 12.8), numpy 2.4, pandas 2.3, xarray 2025.12 and zarr 3.1.

---

## 7. Trained models

| File | Model |
|---|---|
| `weights/full_data_seed00.pt` | default configuration trained on all 332 training gauges over 2009–2020, nothing withheld (seed 0): the model of D3P-JP-Q |

The models of the evaluation settings and of the sensitivity experiments are
not shipped: the launchers of sections 5 and 6 train them, and
`scripts/export_weights.py` turns any training run made with `--save_weights`
into model files.

A model file is a dictionary saved with `torch.save` (load it with
`weights_only=True`): `state_dict` (weights), `spec` (the configuration that
rebuilds the model: kernel, descriptors, bounds, inputs, window lengths),
`norm` (the training run's normalisation constants, checked against the
dataset when the model is used) and `meta` (seed, training and test gauges,
library versions). In Python:

```python
import sys; sys.path.insert(0, "src")
from weights import load_model
model, payload = load_model("weights/full_data_seed00.pt", device="cuda:0")
```

---

## 8. Tests

```bash
pip install pytest
pytest -q tests          # synthetic 20-node network, about a minute; two tests need a GPU
```

---

## Data sources and licences

The code is released under the MIT License (`LICENSE`). D3P-JP and D3P-JP-Q
are distributed under the CC BY-NC 4.0 licence; the dataset combines river
discharge and dam records of the Ministry of Land, Infrastructure, Transport
and Tourism (MLIT, Water Information System); the in-situ gauge-interpolated
precipitation of Hatono et al. (2022); JMA radar, gauge-adjusted radar and MSM
analyses; ERA5-Land (Muñoz-Sabater et al., 2021); JAXA land use and land cover;
J-FlwDir (Yamazaki et al., 2020) and HydroATLAS (Linke et al., 2019). The
`LICENSE.md` of the dataset gives the terms of each source and the credit it
requires.

## Citation

Please cite the paper, the dataset and, when you use the code, its archive on
Zenodo: <https://doi.org/10.5281/zenodo.24238059> (see `CITATION.cff`).
