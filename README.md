# GEE for Relative Risk Regression

This repository contains code and generated simulation results for longitudinal relative-risk regression analyses using GEE and penalized GEE estimators.

## Repository Structure

- `or_pgee_comparison/`: simulation code, post-processing scripts, manuscript table generators, and generated simulation results comparing RR-GEE, RR-PGEE, OR-GEE, and OR-PGEE estimators.
- `or_pgee_comparison/results/`: generated simulation summaries, LaTeX tables, diagnostic CSV files, and figures.
- `real_dset_experiment/`: UK Biobank real-data analysis scripts for tables and figures.

The real-data input directory `real_dset_experiment/UKB/` is intentionally excluded from Git because it contains large and sensitive analysis inputs and model outputs.

## Main Analyses

### Simulation Study

The simulation study is implemented under `or_pgee_comparison/`. Key scripts include:

- `run_simulation.py`: runs simulation replications.
- `postprocess.py`: builds summary tables from replication-level outputs.
- `plot_figureB1.py`: plots the BEC threshold histogram for the simulation study.
- `plot_figure1.py` and `plot_pp.py`: generate simulation diagnostic and probability-probability plots.

Generated outputs are stored under `or_pgee_comparison/results/`.

### Real-Data UKB Analysis

The real-data analysis scripts are under `real_dset_experiment/`. Key outputs include:

- `figure3/`: empirical lesion incidence maps.
- `figure4/`: voxel-wise significance maps.
- `figure5/`: estimated intra-subject correlation and dispersion maps.
- `figure6/`: relative-risk maps.
- `figure7/`: relative-risk comparison plots.
- `figureB3/`: UKB BEC threshold histogram.
- `table5/`, `table6/`, and `table7/`: manuscript table generators and LaTeX outputs.

## Data Policy

Large UKB input files, intermediate model outputs, and binary neuroimaging/data files are not tracked in this repository. The ignore rules protect files such as `real_dset_experiment/UKB/`, `*.npz`, `*.npy`, `*.pkl`, and NIfTI files.

## Reproducibility Notes

Most scripts are intended to be run from the repository root. For example:

```bash
python -m or_pgee_comparison.plot_figureB1
python real_dset_experiment/figureB3/plot_figureB3.py
```

Generated simulation results are tracked under `or_pgee_comparison/results/`; real-data inputs are not included and must be available locally to rerun the UKB analyses.
