# Longitudinal

Research code for longitudinal lesion modelling in neuroimaging data. The repository contains Python ports of voxel-wise GEE, penalised GEE (PGEE), log-Poisson GEE/PGEE, CVR analyses, Basel cohort analyses, and simulation studies.

The code is organised as analysis scripts rather than as an installable Python package. Many scripts reproduce a specific cluster/HPC workflow and expect the original data layout.

## Repository layout

| Path | Contents |
| --- | --- |
| `prelim/` | Pre-processing, exploratory analyses, CVR preparation, lesion probability, and random-effects scripts. |
| `GEE_tests/` | Main UK Biobank voxel-wise GEE and log-Poisson GEE scripts, helper functions, result summaries, and SGE job scripts. |
| `PGEE_Mondol/` | Penalised GEE implementation and voxel-wise PGEE runner. |
| `Apr2021_GEE/` | Earlier GEE/PGEE interaction and dispersed-model analyses plus result aggregation code. |
| `CVRanalysis/` | CVR-specific GEE/PGEE analysis scripts and plotting/result summaries. |
| `Basel_data/` | Basel cohort exploratory scripts and GEE analyses. |
| `Simulations/code/` | Simulation data generation, repeated simulation runners, summary scripts, and SGE/SLURM array jobs. |
| `funpack/` | Shell scripts for extracting UK Biobank fields with Funpack. |

## Requirements

Use Python 3 with the scientific Python stack:

```bash
python -m pip install numpy pandas scipy statsmodels matplotlib seaborn nibabel pyreadr rdata
```

Some cluster job scripts load `R/3.6.2-foss-2019b` because the original workflow used R-era `.RData` files and HPC module environments.

## Data and path assumptions

Most scripts contain absolute paths under `/well/nichols/users/...` or `/gpfs3/well/nichols/users/...`. Before running on another machine, update the path constants near the top of the script you want to run, or run in an environment where those paths and data files exist.

Common inputs include:

- `prelim/temp/df_visits.dat`
- `prelim/temp/lesions_atleast6.RData`
- `bianca_1vis_2vis_overlap.txt`
- NIfTI lesion image directories referenced by each analysis script
- Per-analysis temporary/output folders such as `GEE_tests/temp_*`, `PGEE_Mondol/temp_atleast6`, `Apr2021_GEE/temp_*`, `CVRanalysis/temp_*`, and `Simulations/Mar23_results`

Several Python scripts save pickled Python objects with historical `.RData` filenames. Scripts that read native R `.RData` files generally require `pyreadr` or `rdata`; some newer ports expect pickle, `.npz`, or MATLAB `.mat` content stored under the original `.RData` filename.

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
