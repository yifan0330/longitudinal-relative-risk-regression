# Code Development Conventions

These conventions keep the Longitudinal codebase clean, professional, reproducible, and efficient while preserving its role as research analysis code rather than an installable Python package.

## Core principles

- Keep analysis intent clear: each script should make it obvious which cohort, model, simulation, or result summary it supports.
- Prefer small, testable functions over long procedural blocks. Put reusable model, simulation, path, and IO logic in helper functions instead of copying it between scripts.
- Preserve scientific reproducibility. Record model assumptions, parameter choices, random seeds, input paths, and output file names close to the code that uses them.
- Make changes surgical. Do not rewrite unrelated historical scripts unless the change is needed for correctness, reproducibility, or maintainability.

## Repository structure

- Keep new code near the workflow it supports:
  - `GEE_tests/` for main UK Biobank GEE workflows.
  - `PGEE_Mondol/` for PGEE implementations and runners.
  - `Apr2021_GEE/` for April 2021 analysis variants.
  - `CVRanalysis/` for CVR-specific analyses.
  - `Basel_data/` for Basel cohort analyses.
  - `Simulations/code/` for simulation generation, repeated simulation runs, and simulation summaries.
  - `prelim/` for exploratory or preprocessing work.
- Avoid adding large generated outputs, logs, images, intermediate tables, or copied datasets to git. Use ignored `temp*`, `output`, or `results*` directories for generated files.
- If a helper becomes useful across multiple workflows, prefer a shared module with a clear name rather than duplicating the helper in several directories.

## Python style

- Target Python 3.11 or later, matching `pyproject.toml`.
- Use `pathlib.Path` for paths in new or substantially modified code.
- Use descriptive names for model inputs and outputs, for example `n_subj`, `n_visits`, `covariance`, `beta_se_sandwich`, and `subset_index`.
- Add type hints to new public helpers and functions with non-obvious return values.
- Keep imports at the top of the file and remove unused imports.
- Prefer explicit keyword arguments for model-fitting and simulation functions when positional arguments would be ambiguous.
- Avoid hidden import-time side effects. Expensive analyses, file writes, and cluster-job behavior should run behind a `main()` function or an `if __name__ == "__main__":` guard.
- Use concise comments only when they explain model reasoning, data assumptions, numerical safeguards, or non-obvious HPC behavior.

## Data and IO

- Resolve project-relative paths from the repository root or the script location. Avoid hard-coded user-specific absolute paths when a relative path can work.
- Use Python-native formats such as `.pkl` or `.npz` for new Python-generated intermediates unless compatibility with existing R-era workflows requires `.RData`.
- Make IO helpers create parent directories explicitly before writing outputs.
- Keep data-loading assumptions visible: expected columns, array shapes, visit ordering, subject ordering, and missing-value behavior should be documented in code or function docstrings.
- Do not silently continue after missing required inputs. Raise a clear error that names the missing path or malformed column.

## Numerical and modelling code

- Validate shapes before fitting models. Check that response vectors, design matrices, subject counts, visit counts, and covariance matrices are compatible.
- Prefer NumPy and pandas vectorized operations for voxel-wise and simulation-heavy work, but keep expressions readable.
- Guard numerical operations that can become unstable, such as divisions, logarithms, matrix inversions, and variance estimates.
- Return structured dictionaries or lightweight result objects with stable keys for coefficients, standard errors, convergence flags, dispersion, correlation, and iteration counts.
- Keep convergence handling explicit. Record failed fits as failed results instead of mixing them with successful estimates.
- For stochastic simulations, accept an explicit random number generator or seed and document the default behavior.

## Command-line and cluster scripts

- New Python runners should parse command-line arguments clearly, validate required values, and print concise progress messages suitable for HPC logs.
- Keep SGE/SLURM scripts minimal: load the required environment, create required output directories, and call the Python runner with explicit arguments.
- Use repository-relative paths in job scripts when possible so jobs can run from both `/well` and `/gpfs3` checkouts.
- Do not depend on interactive prompts in analysis or cluster jobs.

## Testing and validation

- Add or update unit tests when changing reusable model, simulation, IO, or argument-parsing logic.
- Prefer small toy datasets in tests so model code can be validated without private data or long HPC jobs.
- Run the project tests from the repository root with:

```bash
uv run python -m unittest discover -s test -p 'test_*.py'
```

- For script-only changes that cannot be unit-tested directly, at least validate argument parsing, path construction, or helper functions separately.
- Do not require private datasets, large generated outputs, or cluster schedulers for unit tests.

## Dependencies

- Manage Python dependencies in `pyproject.toml`.
- Avoid adding dependencies for functionality already available from the existing scientific stack.
- If a new dependency is necessary, choose a maintained package, document why it is needed, and keep imports localized when only one workflow needs it.

## Documentation

- Update `README.md` when adding a new top-level workflow, important command, dependency, or data assumption.
- Document expected inputs and outputs in scripts that are intended to be rerun by others.
- Prefer Markdown tables or concise command examples for operational documentation.
- Keep historical context when it explains why a workflow differs from newer code, especially for R-era `.RData` compatibility.

## Code review checklist

Before considering a change complete, confirm that:

- The change is limited to the intended workflow or helper.
- Names, paths, and output files are clear and consistent with the surrounding code.
- Required inputs fail with clear errors.
- Generated data, logs, figures, and large outputs are not staged.
- New reusable behavior has a focused test or a documented validation command.
- The relevant unit tests or lightweight validation command pass.
