# OR-GEE/OR-PGEE comparison simulation extension

This folder extends the existing Python simulation code without modifying or rerunning the published Table 2/3/4 scripts.

## Reuse

- Standard data generation imports `Simulations/code/source_simdata.py::gen_dataPP`.
- RR-GEE imports `Simulations/code/gee_logPoisson_dispersion_fn.py::gee_dispersion_run`.
- RR-PGEE imports `Simulations/code/Sept21_pgee_logPoisson_dispersion_fn.py::gee_penalty_run`.
- OR-GEE and OR-PGEE import `PGEE_Mondol/Mar23_PGEE_source.py::geefirth`; OR-GEE uses the same logistic GEE score without the Firth penalty, while OR-PGEE mirrors the published penalized call: `geefirth(y, x=X[:, 1:3], id=data["id"], ar=False)`.

Replication seeds are one-based (`1..R`) to match `Mar23_rep_sims.py`, so new OR-GEE/OR-PGEE fits are paired to the same simulated datasets as the existing RR-GEE/RR-PGEE runs.

## Zhang-Yu transformation and intervals

The transformed OR-GEE/OR-PGEE point estimate is

```text
RR_hat = exp(beta_b*) / ((1 - p0_hat) + p0_hat * exp(beta_b*)).
```

The default `p0_hat` estimator is the observed event proportion among unexposed observations (`X1 = 0`). The 95% CI is built on the log-OR scale, exponentiated to an OR interval, and then transformed endpoint-wise with the same plug-in `p0_hat`. This intentionally ignores uncertainty in `p0_hat`, matching common Zhang-Yu practice.

RR-GEE and RR-PGEE use sandwich Wald intervals on the log-RR scale, then exponentiate endpoints. OR-GEE and OR-PGEE use sandwich Wald intervals on the log-OR scale, then transform endpoints to the RR scale. Coverage is computed only for converged, finite fits; the denominator is reported as `coverage_n`.

## Confounded scenario

The confounded scenario keeps the base parameters but replaces the time-varying covariate used in the linear predictor with

```text
X2_ij = max(x2_floor, base_x2_j + confounding_shift * (X1_i - c)).
```

The default smoke and full confounded scenario uses `confounding_shift = 0.50`. This creates positive dependence between `X1` and `X2` while keeping the same log-link DGP and Qaqish-style correlated binary outcome generation. The run writes empirical `corr(X1, X2)`, `X2` clipping, and probability clipping diagnostics.

## Smoke test

Run the R=100 smoke test across the full scenario grid before any full-scale queue submission:

```bash
uv run python -m or_pgee_comparison.run_simulation --mode smoke
```

This uses the same scenario names as the full run, but with 100 replications per scenario. Expected outputs under `or_pgee_comparison/results/smoke/`:

- `replications.csv`
- `diagnostics.csv`
- `table2_convergence.{csv,md,tex}`
- `table3_bias_mse.{csv,md,tex}`
- `table3_beta_bias_mse.{csv,md,tex}`
- `table4_variance_coverage.{csv,md,tex}`
- `confounded_summary.{csv,md,tex}`

## Figure 1

Generate the Figure 1-style RR-GEE vs RR-PGEE z-statistic scatter plot from
the full simulation results:

```bash
uv run python -m or_pgee_comparison.plot_figure1
```

By default this uses the base scenario and paired replications where both RR-GEE
and RR-PGEE converged with finite estimates, plotting RR-GEE
`beta_hat / SE(beta_hat)` on the x-axis against RR-PGEE
`beta_hat / SE(beta_hat)` on the y-axis. The outputs are written as
`figure1_rr_zscore_scatter.{pdf,png}` under
`or_pgee_comparison/results/full/figure1/`. Each plot uses the same data-driven
x- and y-axis limits for its metric so the dashed `y=x` reference line is
shown on comparable scales.

To restrict to the converged finite paired replications where RR-GEE has
`BEC > 10`, run:

```bash
uv run python -m or_pgee_comparison.plot_figure1 --boundary-warnings --stem figure1_rr_zscore_scatter_boundary
```

The same paired scatter format can be generated for the coefficient estimate
and sandwich SE:

```bash
uv run python -m or_pgee_comparison.plot_figure1 --metric beta
uv run python -m or_pgee_comparison.plot_figure1 --metric se
```

To generate the same three subfigures for OR-GEE vs OR-PGEE, run:

```bash
uv run python -m or_pgee_comparison.plot_figure1 --pair or --metric all
```

## PP-plots

Generate one p-value PP-plot for every full simulation setting:

```bash
uv run python -m or_pgee_comparison.plot_pp
```

The plots compare ordered two-sided Wald p-values with uniform probabilities
on a `-log10(p)` scale and are written as PDF and PNG files under
`or_pgee_comparison/results/full/pp_plots/`. Every method uses the same subset
of replications for which all four methods converged and returned finite
estimates with positive SEs. Each displayed curve is the ordered p-value
sequence, with a pointwise beta order-statistic band under the uniform null.
The confidence level can be changed with `--confidence-level`. Each legend
entry also reports coverage probability (CP) and mean RR-scale CI width from
`table4/table4_variance_coverage.csv`. P-values are centred on the simulated
true effect. For OR-GEE and OR-PGEE, the true RR is mapped back to the log-OR
scale by inverting the Zhang-Yu transformation using each replication's
`p0_hat`.

Full-run manuscript tables are grouped by output under
`or_pgee_comparison/results/full/table1/`, `table2/`, `table3/`, `table4/`,
and `confounded_summary/`; `replications.csv` and `diagnostics.csv` remain in
the full-run root as shared simulation inputs.

## Full run

The full configuration uses `R = 1000`, `epsilon = 1e-4`, `K = 25`, and `phi = 1` with one scenario per job, mirroring the existing parallelisation pattern. To run one full scenario:

```bash
uv run python -m or_pgee_comparison.run_simulation --mode full --scenario base
```

To run all configured scenarios in one process:

```bash
uv run python -m or_pgee_comparison.run_simulation --mode full
```

To parallelize over independent simulation replications within a scenario:

```bash
uv run python -m or_pgee_comparison.run_simulation --mode full --scenario base --workers 8
```

The default estimator engine remains the validated IRLS implementation. For
manuscript results, use `--workers` with the original one-seed-per-replication
data generation and IRLS estimators. Serial IRLS and parallel IRLS write the
same replication and diagnostic rows; the parallel path only changes wall-clock
execution.

To compare serial IRLS, parallel IRLS, loop-based data generation, broadcast
data generation, and optionally the L-BFGS OR-GEE benchmark path:

```bash
uv run python -m or_pgee_comparison.benchmark_speed --replications 20 --workers 8 --include-lbfgs
```

This writes `or_pgee_comparison/results/benchmark/speed_comparison.csv`.
The benchmark labels L-BFGS and broadcast generation as exploratory: L-BFGS
does not reproduce the validated OR-GEE outcomes exactly, and broadcast
generation does not reproduce the historical one-seed-per-replication streams.

For the cluster, submit one job per scenario name returned by `config.full_scenarios()`. The full grid currently contains 23 scenario jobs: the base case, one-at-a-time sweeps for `beta0`, `beta_b`, `c`, `rho`, `N`, and one confounded scenario. Each scenario job runs all four methods on the same `R = 1000` seeded datasets.

The confounding strength is configurable:

```bash
uv run python -m or_pgee_comparison.run_simulation --mode full --confounding-shift 1.00 --scenario confounded_shift_1_00
```

Use this knob if the default `0.50` shift does not produce a strong enough separation between transformed OR-PGEE and RR-PGEE in the final replication count.

## Post-hoc coverage from published outputs

`coverage.py::coverage_from_saved_payload` computes RR-GEE/RR-PGEE coverage from saved per-replication arrays (`coefs_*`, `SEs_*`, `conv_*`) using the sandwich SEs. Use it for existing published-result payloads when they are present; do not rerun RR-GEE/RR-PGEE for those shared scenarios solely to compute coverage. In the current checked-out tree no `Simulations/**/*.RData`, `.Rdata`, `.pkl`, or `.pickle` payloads were present, so the smoke test generated fresh paired outputs only under `or_pgee_comparison/results/smoke/`.
