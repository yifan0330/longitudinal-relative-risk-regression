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

This creates a local `.venv` from `pyproject.toml` with the Python packages needed by the analysis scripts. Run commands through that environment, for example:

```bash
uv run python GEE_tests/GEE_run.py
```

Some cluster job scripts load `R/3.6.2-foss-2019b` because the original workflow used R-era `.RData` files and HPC module environments.

## Data and path assumptions

Most scripts resolve project inputs relative to the repository root. Sibling data directories are referenced relative to the repository parent, so the project can be moved between `/well` and `/gpfs3` locations without editing hard-coded user paths.

Common inputs include:

- `prelim/temp/df_visits.dat`
- `prelim/temp/lesions_atleast6.RData`
- `bianca_1vis_2vis_overlap.txt`
- NIfTI lesion image directories referenced by each analysis script
- Per-analysis temporary/output folders such as `GEE_tests/temp_*`, `PGEE_Mondol/temp_atleast6`, `Apr2021_GEE/temp_*`, `CVRanalysis/temp_*`, and `Simulations/Mar23_results`

Python-generated intermediate files should use Python-native suffixes such as `.pkl` or `.npz`. Scripts that read native R `.RData` files generally require `pyreadr` or `rdata` for compatibility with original inputs.

## Running analyses

Run commands from the repository root unless a script says otherwise.

### GEE analyses

Run the full voxel-wise GEE script:

```bash
python GEE_tests/GEE_run.py
```

Run one subset for log-Poisson GEE:

```bash
python GEE_tests/GEE_logPoisson_run.py <n_cores> <subset_index>
```

Run one subset for dispersed log-Poisson GEE:

```bash
python GEE_tests/GEE_logPoisson_dispersed_run.py <n_cores> <subset_index>
```

Run one subset for penalised log-Poisson GEE:

```bash
python GEE_tests/GEE_penalty_logPoisson_run.py <n_cores> <subset_index>
```

Example:

```bash
python GEE_tests/GEE_logPoisson_dispersed_run.py 1 1
```

Inspect or aggregate outputs:

```bash
python GEE_tests/gee_results.py
python GEE_tests/dispersion_check.py
```

Submit the corresponding Sun Grid Engine jobs:

```bash
qsub GEE_tests/GEE.sh
qsub GEE_tests/logPoisson.sh
qsub GEE_tests/penalty_logPoisson.sh
```

### PGEE analysis

Run the voxel-wise PGEE analysis directly:

```bash
python PGEE_Mondol/PGEE_run.py
```

Or submit the SGE job:

```bash
qsub PGEE_Mondol/PGEE.sh
```

`PGEE_Mondol/PGEE_run.py` writes subset outputs to `PGEE_Mondol/temp_atleast6/PGEE_NAs_<subset>.RData`.

### April 2021 GEE/PGEE analyses

Representative direct run:

```bash
python Apr2021_GEE/code/GEE_logPoisson_interaction_run.py <n_cores> <subset_index>
```

Cluster jobs:

```bash
qsub Apr2021_GEE/logPoisson_gee.sh
qsub Apr2021_GEE/logPoisson_pgee.sh
qsub Apr2021_GEE/logistic_ORpgee.sh
```

Summarise results with:

```bash
python Apr2021_GEE/gee_results.py
```

### CVR analyses

Run a CVR subset directly:

```bash
python CVRanalysis/code/GEE_logPoisson_interaction_run.py <n_cores> <subset_index>
python CVRanalysis/code/PGEE_logPoisson_interaction_run.py <n_cores> <subset_index>
python CVRanalysis/code/OR-PGEE_logistic_interaction_run.py <n_cores> <subset_index>
```

Or submit the SGE jobs:

```bash
qsub CVRanalysis/gee.sh
qsub CVRanalysis/pgee.sh
qsub CVRanalysis/logistic_ORpgee.sh
```

Summarise or explore CVR outputs:

```bash
python CVRanalysis/pgee_results.py
python CVRanalysis/CVRexploratory.py
```

### Basel data analyses

Run Basel GEE analyses directly:

```bash
python Basel_data/GEE/GEE_run.py
python Basel_data/GEE/GEE_vis.py
python Basel_data/GEE/GEE_sex.py
python Basel_data/GEE/GEE_MStype.py
```

Or submit the SGE jobs:

```bash
qsub Basel_data/GEE/GEE.sh
qsub Basel_data/GEE/GEE_vis.sh
qsub Basel_data/GEE/GEE_sex.sh
qsub Basel_data/GEE/GEE_MStype.sh
```

## Running simulations

Run one March 2023 simulation configuration:

```bash
python Simulations/code/Mar23_rep_sims.py <beta0> <beta1> <beta2> <n_clusters> <cluster_size> <event_probability> <rho> <n_sim> <p> <output_file>
```

Example:

```bash
python Simulations/code/Mar23_rep_sims.py -4 1.6 0.2 50 4 0.2 0.4 1000 3 Simulations/Mar23_results/example.RData
```

Submit SLURM array jobs for March 2023 parameter sweeps:

```bash
sbatch Simulations/code/Mar23_sims.sh
sbatch Simulations/code/Mar23_sims_beta0.sh
sbatch Simulations/code/Mar23_sims_betaB.sh
sbatch Simulations/code/Mar23_sims_gamma.sh
sbatch Simulations/code/Mar23_sims_N.sh
```

Submit older SGE simulation jobs:

```bash
qsub Simulations/code/sims.sh
qsub Simulations/code/sims_beta0.sh
qsub Simulations/code/sims_betaB.sh
qsub Simulations/code/sims_gamma.sh
qsub Simulations/code/sims_N.sh
```

Summarise simulation outputs:

```bash
python Simulations/code/Mar23_Simulations_results.py
python Simulations/code/Simulations_results.py
```

## Pre-processing and exploratory scripts

Examples:

```bash
python prelim/prelim.py
python prelim/CVR_prep.py
python prelim/RE_run.py
python Basel_data/Exploratory.py
python CVRanalysis/CVRexploratory.py
```

The Funpack extraction scripts in `funpack/` are shell scripts intended for the original UK Biobank extraction workflow.

## Troubleshooting

- If a script cannot find data, check the absolute path constants at the top of that script.
- If `.RData` loading fails, install `pyreadr`/`rdata` or convert the input to the format expected by the specific script.
- If local imports fail, run from the repository root or from the script's directory so neighbouring modules are on `sys.path`.
- If cluster jobs fail immediately, create the referenced output directories first, for example `GEE_tests/output`, `Simulations/code/Mar23_output`, or the relevant `temp_*` directory.

Generated simulation results are tracked under `or_pgee_comparison/results/`; real-data
inputs are not included and must be available locally to rerun the UKB analyses.
