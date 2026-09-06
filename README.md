# RADL

RADL (Regime-Aligned Dependency Learning) is a multivariate time-series
forecasting framework built around a Graph WaveNet-style backbone with
per-window dependency learning and anchored reversible instance normalization.

The repository contains the reusable training and evaluation framework. Local
experiment runners, logs, generated results, checkpoints, caches, and backup
copies are intentionally excluded from version control.

## Included capabilities

- Chronological train/validation/test splitting with training-only scaling.
- Per-window latent dependency graphs.
- Optional learned static adjacency and variable gating.
- Configurable attention initialization gain.
- Optional lag-aware graph edges with delayed messages.
- Batch or channel-wise backbone normalization.
- Optional target MAE loss, movement gate, checkpoint metric selection, and
  final train-plus-validation refit.
- Unit tests for normalization and lag-graph invariants.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run a forecast

The checked-in CSV files under `data/` provide example input data. A custom
CSV or Excel file can be supplied with `-data_path`; the first column is read
as the index and the remaining columns are variables.

```powershell
python RADL.py -data_path .\data\GDEA.csv -cuda false
```

Outputs are written to `output/` and checkpoints to `save_dict/`. Both paths
can be changed with `-output_dir` and `-save_dir`.

The main implementation options are exposed by `python RADL.py -h`, including
`-att_init_gain`, `-use_lag_graph`, `-graph_lags`, `-backbone_norm`,
`-gcn_order`, `-weight_average_k`, and `-refit_train_valid`.

## Run framework tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Repository layout

- `RADL.py`: training entry point.
- `test.py`: checkpoint evaluation entry point.
- `config.py`: command-line configuration.
- `load_data.py`: loading, scaling, and chronological windows.
- `model/gwnet_att.py`: RADL backbone and optional graph extensions.
- `tool.py`: training, checkpointing, testing, and metrics.
- `evaluate.py`: regression and Diebold-Mariano metrics.
- `tests/`: framework-level invariant tests.
