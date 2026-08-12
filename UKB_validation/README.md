# Real-data experiment

This package reproduces the UK Biobank analyses and manuscript outputs. It keeps
the historical input, cache, and output locations stable while exposing one
workflow entry point:

```bash
python -m UKB_validation figures
python -m UKB_validation tables
python -m UKB_validation all
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
* `plot_figure3.py`–`plot_figure7.py` and `plot_figure_b3.py` are the
  recommended figure entry points.
* `table_5.py`, `table6_rr.py`, `table6_or.py`, and `table_7.py` are the
  recommended table entry points.
* The historical `figure*/` and `table*/` modules remain available as
  compatibility entry points and retain their original output directories.
* `cli.py` and `__main__.py` provide the reproducible command-line workflow.

For example, run `python -m UKB_validation.plot_figure3` or
`python -m UKB_validation.table6_rr` to rerun one manuscript item.
Each historical plotting or table module also remains directly executable
with `python -m`.
