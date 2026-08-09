# OR-PGEE simulation extension inventory

## Confirmed repository and language

- `REPO_ROOT`: `/gpfs3/well/nichols/users/pra123/Longitudinal`
- Existing Table 2/3/4 simulation code is Python, with Bash SLURM/SGE launch scripts. Some job scripts load an R module for historical compatibility, but the simulation generator, model fitters, repeated-run drivers, and result summaries are Python.

## Existing data-generating process

- The simulation generator is `Simulations/code/source_simdata.py`.
- `simulate_correlated_bernoulli(probs, corr, rng)` approximates the original Qaqish (2003) / `binarySimCLF` correlated-binary step with a Gaussian-copula Bernoulli simulator and nearest-correlation projection (`source_simdata.py:36-49`).
- `gen_dataPP(beta, nc, cl_size, p=None, rho=0.0, prop=None, rng=None)` is the dataset generator (`source_simdata.py:52-99`):
  - Builds `X = [intercept, X1i, obstime]`.
  - Draws subject-level binary `X1i` with probability `p` / `prop`.
  - Uses visit times `obstime = 0.2, 0.4, ...`.
  - Uses a log-link DGP: `probs = exp(X @ beta)`.
  - Applies exchangeable within-subject correlation via `xch(size, rho)`.
  - Returns a `pandas.DataFrame` with `id`, `yij`, `intercept`, `X1i`, and `obstime`.
- `model_matrix(df)` in `Simulations/code/simulation_helpers.py:42-45` rebuilds the fitting matrix as `[1, X1i, obstime]`.

## Existing method-fitting routines

### RR-GEE

- Function: `Simulations/code/gee_logPoisson_dispersion_fn.py::gee_dispersion_run`.
- Implementation: `_gee_core(..., penalty="none")` (`gee_logPoisson_dispersion_fn.py:32-172`).
- Key arguments used by simulation drivers: `y`, `X`, `n_subj`, `n_visits`, `covariance="Exchangeable"`, `tol=1e-4`, `max_iter`, and `phi_est=True`.
- Returns a dictionary with:
  - `beta`
  - `beta_se_model`
  - `beta_se_model_trace`
  - `beta_se_sandwich`
  - `alpha`
  - `phi`
  - `iterations`
  - `conv`

### RR-PGEE

- Function currently used by simulation drivers: `Simulations/code/Sept21_pgee_logPoisson_dispersion_fn.py::gee_penalty_run`.
- Implementation: wraps `_gee_core(..., penalty="ik")` (`Sept21_pgee_logPoisson_dispersion_fn.py:11-23`).
- Return keys are the same as RR-GEE, plus `_gee_core` can include `H` for penalized fits.
- There is also `Simulations/code/pgee_logPoisson_dispersion_fn.py::gee_penalty_run`, which wraps `_gee_core(..., penalty="qr")`; the current `rep_sims.py` and `Mar23_rep_sims.py` import the September 2021 `penalty="ik"` variant.

### OR-PGEE

- Existing implementation is in `PGEE_Mondol/PGEE_source.py`, reused through `PGEE_Mondol/Mar23_PGEE_source.py`.
- `Mar23_PGEE_source.py::geefirth(y, x, id, ar=True)` calls `_geefirth_impl(..., init_offset=0.0001, keep_trace=True)` (`Mar23_PGEE_source.py:13-14`).
- `_geefirth_impl` returns a list (`PGEE_source.py:270-279`):
  - `[0]`: robust/sandwich coefficient table `est_sw`, with columns including `coefficients` and `std.err`.
  - `[1]`: model-based coefficient table `est_swm`.
  - `[2]`: estimated correlation.
  - `[3]`: iteration counter.
  - `[4]`: dispersion `phi`.
  - `[5]`: model SE vector, when `keep_trace=True`.
  - `[6]`: model SE trace, when `keep_trace=True`.
- Existing March 2023 simulation code already fits OR-PGEE in `Simulations/code/Mar23_rep_sims.py:98-122` and saves:
  - `coefs_ORpgee`
  - `SEs_ORpgee`
  - `SEs_model_ratio_ORpgee`
  - `alpha_ORpgee`
  - `phi_ORpgee`
  - `eta_ORpgee`
  - `iter_ORpgee`
  - `conv_ORpgee`
- Current OR-PGEE convergence bookkeeping is simple: failed fit gives `NaN` arrays and `conv_ORpgee=False`; otherwise `conv_ORpgee=True`.

## Scenario grid and fixed parameters

The March 2023 scripts are the best match for the requested Table 2/3/4 extension because they already include OR-PGEE.

Base scenario:

| Parameter | Value | Source |
| --- | --- | --- |
| `beta = (beta1, beta_b, beta_c)` | `(-4, 1.6, 0.2)` | `Mar23_Simulations_results.py:16`, `Mar23_sims.sh:32` |
| `N` / `nc` | `50` | `Mar23_sims.sh:32` |
| `T` / `cl_size` | `4` | `Mar23_sims.sh:32` |
| `c` / `prop` | `0.2` | `Mar23_sims.sh:32` |
| `alpha` / `rho` | `0.4` in base; alpha sweep varies it | `Mar23_sims.sh:32` |
| `R` / `n_sim` | `1000` | `Mar23_sims.sh:32` |
| number of coefficients `p` | `3` | `Mar23_sims.sh:32` |
| tolerance `epsilon` | `1e-4` default | `gee_logPoisson_dispersion_fn.py:37-40` |
| max iterations `K` | `25` in March 2023 runner | `Mar23_rep_sims.py:77,93` |
| true dispersion `phi` | initialized/estimated around `1.0`; `phi_est=True` | `gee_logPoisson_dispersion_fn.py:57`, `Mar23_rep_sims.py:78,94` |

One-at-a-time variations:

- `alpha` / `rho`: `0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8` (`Mar23_sims.sh:28-32`, `Mar23_Simulations_results.py:17`).
- `beta_b`: `1.2, 1.4, 1.6, 1.8, 2.0` (`Mar23_sims_betaB.sh:28-32`, `Mar23_Simulations_results.py:18`).
- `c` / `prop`: `0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8` (`Mar23_sims_gamma.sh:28-32`, `Mar23_Simulations_results.py:21`).
- `N`: `25, 50, 75, 100, 500, 1000` in the SLURM launcher (`Mar23_sims_N.sh:28-32`); `Mar23_Simulations_results.py:19` currently lists `25, 50, 75, 100`.
- `beta1` / intercept: launcher uses `-4, -3, -2, -1` (`Mar23_sims_beta0.sh:28-32`); `Mar23_Simulations_results.py:20` currently lists `-4, -3, -2`.

## Saved outputs and whether coverage can be computed post hoc

- Repeated simulations are saved as Python pickles, even when the filename suffix is `.RData`.
- `Simulations/code/simulation_helpers.py:48-52` writes with `pickle.dump`.
- `Simulations/code/rep_sims.py:87-95` saves RR-GEE/RR-PGEE per-replication arrays.
- `Simulations/code/Mar23_rep_sims.py:126-142` saves RR-GEE, RR-PGEE, and OR-PGEE per-replication arrays.
- Existing per-replication outputs include the required ingredients for post-hoc RR-GEE and RR-PGEE coverage:
  - `coefs_geePK`, `SEs_geePK`, `conv_geePK`
  - `coefs_PgeePK`, `SEs_PgeePK`, `conv_PgeePK`
  - `SEs_model_ratio_geePK` and `SEs_model_ratio_PgeePK` for BEC-style filtering.
- March 2023 outputs also include OR-PGEE coefficient and SE arrays:
  - `coefs_ORpgee`, `SEs_ORpgee`, `conv_ORpgee`, `iter_ORpgee`, and `SEs_model_ratio_ORpgee`.
- Existing result loaders are in `Simulations/code/Simulations_results.py:31-51` and can read pickle-backed `.RData` files.
- Therefore:
  - If the relevant `Simulations/Mar23_results/*.RData` files already exist, coverage can be computed post hoc for all three methods.
  - If only older `rep_sims.py` outputs exist for a scenario, RR-GEE/RR-PGEE coverage can be computed post hoc, while OR-PGEE must be run on paired regenerated datasets using the same replicate seeds.

## Seeds and pairing

- Both repeated simulation drivers seed each replication with the replication index:
  - `rep_sims.py:38-47`: `np.random.default_rng(i)`.
  - `Mar23_rep_sims.py:54-62`: `np.random.default_rng(i)`.
- All methods within a replication use the same generated dataset, so comparisons are paired by replication index.
- New shared-grid OR-PGEE runs should reuse the same `rng=np.random.default_rng(i)` data-generation rule to pair with existing RR-GEE/RR-PGEE outputs if saved OR-PGEE outputs are unavailable.

## Parallelization pattern

- March 2023 full runs use SLURM arrays:
  - `Mar23_sims.sh` for alpha/rho sweep.
  - `Mar23_sims_beta0.sh` for intercept sweep.
  - `Mar23_sims_betaB.sh` for binary-covariate effect sweep.
  - `Mar23_sims_gamma.sh` for prevalence/proportion sweep.
  - `Mar23_sims_N.sh` for sample-size sweep.
- Each array task runs one scenario with `R=1000` and writes one result file under `Simulations/Mar23_results/`.
- Older simulation scripts have SGE equivalents (`sims*.sh`) with the same one-scenario-per-task pattern.
- The new folder should mirror the one-scenario-per-command pattern and provide SLURM-friendly arguments, plus a smoke-test config with `R=20`.

## Existing code to reuse

- Data generation for published independent-covariate scenarios: `Simulations/code/source_simdata.py::gen_dataPP`.
- Model matrix helper: `Simulations/code/simulation_helpers.py::model_matrix`.
- RR-GEE fitter: `Simulations/code/gee_logPoisson_dispersion_fn.py::gee_dispersion_run`.
- RR-PGEE fitter used by current simulation drivers: `Simulations/code/Sept21_pgee_logPoisson_dispersion_fn.py::gee_penalty_run`.
- OR-PGEE fitter: `PGEE_Mondol/Mar23_PGEE_source.py::geefirth`.
- Existing result-file conventions and keys from `Simulations/code/simulation_helpers.py::ModelResults` and `Mar23_rep_sims.py`.
- Existing scenario values from the `Mar23_sims*.sh` scripts and `Mar23_Simulations_results.py`.

## New work needed

1. Build a new, self-contained `or_pgee_comparison/` implementation without modifying published-result scripts.
2. Add a config layer that exposes:
   - the shared Table 2/3/4-style scenario grid,
   - a smoke-test subset,
   - output locations,
   - max iterations/tolerance,
   - and the `p0` estimator choice for Zhang-Yu transformation.
3. Add thin method wrappers that call the existing RR-GEE, RR-PGEE, and OR-PGEE routines and normalize their return values.
4. Implement Zhang-Yu OR-to-RR transformation with configurable `p0_hat`; default should be the observed event proportion among `X1 = 0`.
5. Implement RR-scale bias, MSE, coverage, convergence/failure, BEC, and variance summaries from per-replication arrays.
6. Add a confounded DGP wrapper that reuses the existing correlated Bernoulli response generation but induces dependence between `X1` and `X2`; expose the strength and report the induced correlation.
7. Add orchestration for one scenario per job plus smoke-test execution.
8. Add postprocessing that writes CSV and Markdown/LaTeX table outputs.

## Judgment calls to document in the implementation README

- `p0_hat` default: observed event proportion among observations with `X1 = 0`.
- OR-PGEE interval construction: Wald interval on log-OR scale, exponentiate, then apply Zhang-Yu endpoint transformation with fixed plug-in `p0_hat`.
- Confounding parameterization: still to be implemented; recommended approach is shifting subject-level/visit-level `X2` by `X1` or making `P(X1=1)` depend on a subject-level `X2` summary, with the induced `corr(X1, X2)` saved per scenario.
- Coverage denominators: conditional on finite converged estimates for each method; also provide an unconditional RR-PGEE variant if cheap.

## Plan before bulk coding

- Keep all existing simulation and published-result scripts read-only.
- Implement new code only in `or_pgee_comparison/`.
- First create smoke-testable wrappers and coverage utilities using `R=20`.
- Validate the smoke test end-to-end before preparing full-scale run instructions.
- Use saved March 2023 outputs for post-hoc coverage when available; otherwise regenerate paired datasets with the same replicate-index seeds only for the missing method/scenario.
