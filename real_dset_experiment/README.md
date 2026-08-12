# Real-data experiment

This package reproduces the UK Biobank analyses and manuscript outputs. It keeps
the historical input, cache, and output locations stable while exposing one
workflow entry point:

```bash
python -m real_dset_experiment figures
python -m real_dset_experiment tables
python -m real_dset_experiment all
```

The default workflow reuses cached model maps. Add `--rerun-models` to refit
the models, and `--n-jobs N` to control fitting parallelism.

## Manuscript output catalogue

All outputs are committed beside the code in the directories below. Each
directory contains publication-ready PDF files, PNG previews, and (where
applicable) CSV/LaTeX tables.

| Manuscript item | Reproduction module | Outputs |
| --- | --- | --- |
| Figure 3 | `figure3.plot_ukb_empirical_maps` | [figure3](figure3/) |
| Figure 4 | `figure4.plot_rr_pgee_significance_maps` | [figure4](figure4/) |
| Figure 5 | `figure5.plot_rr_alpha_phi_maps` | [figure5](figure5/) |
| Figure 6 | `figure6.plot_rr_pgee_relative_risk_maps` | [figure6](figure6/) |
| Figure 7 | `figure7.plot_age_relative_risk_comparison` | [figure7](figure7/) |
| Figure B3 | `figureB3.plot_figureB3` | [figureB3](figureB3/) |
| Table 5 | `table5.calculate_ukb_characteristics` | [table5](table5/) |
| Table 6 | `table6.calculate_significant_coefficients`, `calculate_or_significant_coefficients` | [table6](table6/) |
| Table 7 | `table7.calculate_rr_relative_risk_by_incidence` | [table7](table7/) |

The generated files are intentionally kept next to their generating modules so
that a GitHub reader can inspect every original figure and table without
running the UKB model-fitting stage. The `UKB/` directory contains the input
data and cached model results; it is not recreated by the plotting commands.

## Package layout

* `paths.py` defines the canonical filesystem configuration.
* `ukb_python_experiment.py` fits the four longitudinal models.
* `figure3`–`figure7` and `figureB3` generate figures.
* `table5`–`table7` generate manuscript tables.
* `cli.py` and `__main__.py` provide the reproducible command-line workflow.

Each plotting or table module remains directly executable with `python -m`,
which makes individual manuscript items easy to rerun and debug.
