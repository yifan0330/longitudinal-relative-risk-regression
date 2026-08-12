# Longitudinal relative risk regression

**Full project name:** Longitudinal relative risk regression for binary-valued brain-lesion data

This repository contains code and generated simulation results for longitudinal relative-risk regression analyses using GEE and penalized GEE estimators.

## Environment setup

Install [uv](https://docs.astral.sh/uv/) and create the locked Python
environment from the repository root:

```bash
uv sync --locked
```

The project requires Python 3.11 or newer. Run analysis commands through the
same environment with `uv run`; uv will use the versions recorded in
`uv.lock`:

```bash
uv run python -m RR_OR_GEE_PGEE --help
uv run python -m UKB_validation --help
```

## Repository Structure

- `RR_OR_GEE_PGEE/`: simulation code, post-processing scripts, manuscript table generators, and generated simulation results comparing RR-GEE, RR-PGEE, OR-GEE, and OR-PGEE estimators.
- `RR_OR_GEE_PGEE/results/`: generated simulation summaries, LaTeX tables, diagnostic CSV files, and figures.
- `UKB_validation/`: UK Biobank real-data analysis scripts for tables and figures.

The real-data input directory `UKB_validation/UKB/` is intentionally excluded from Git because it contains large and sensitive analysis inputs and model outputs.

## Main Analyses

### Simulation Study

The simulation study is implemented under `RR_OR_GEE_PGEE/`. Key scripts include:

- `run_simulation.py`: runs simulation replications.
- `postprocess.py`: builds summary tables from replication-level outputs.
- `plot_figureB1.py`: plots the BEC threshold histogram for the simulation study.
- `plot_figure1.py` and `plot_pp.py`: generate simulation diagnostic and probability-probability plots.

Generated outputs are stored under `RR_OR_GEE_PGEE/results/`.

### Real-Data UKB Analysis

The real-data analysis scripts are under `UKB_validation/`. Key outputs include:

- `figure3/`: empirical lesion incidence maps.
- `figure4/`: voxel-wise significance maps.
- `figure5/`: estimated intra-subject correlation and dispersion maps.
- `figure6/`: relative-risk maps.
- `figure7/`: relative-risk comparison plots.
- `figureB3/`: UKB BEC threshold histogram.
- `table5/`, `table6/`, and `table7/`: manuscript table generators and LaTeX outputs.

## Data Policy

Large UKB input files, intermediate model outputs, and binary neuroimaging/data files are not tracked in this repository. The ignore rules protect files such as `UKB_validation/UKB/`, `*.npz`, `*.npy`, `*.pkl`, and NIfTI files.

## Reproducibility Notes

Most scripts are intended to be run from the repository root. For example:

```bash
python -m RR_OR_GEE_PGEE.plot_figureB1
python UKB_validation/figureB3/plot_figureB3.py
```

## Troubleshooting

- If a script cannot find data, check the absolute path constants at the top of that script.
- If local imports fail, run commands from the repository root.

Generated simulation results are tracked under `RR_OR_GEE_PGEE/results/`; real-data
inputs are not included and must be available locally to rerun the UKB analyses.
