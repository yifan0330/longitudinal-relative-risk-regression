"""OR-PGEE comparison simulation extension.

Public API
----------
Configuration
    Scenario, BASE_SCENARIO, full_scenarios, smoke_scenarios, RESULTS_DIR

Data generation
    generate_dataset, generate_standard_datasets_broadcast

Fitting
    fit_all_methods, fit_rr_method, fit_or_method
    METHOD_RR_GEE, METHOD_RR_PGEE, METHOD_OR_GEE, METHOD_OR_PGEE
    METHOD_ORDER, OR_METHODS

Coverage utilities
    add_interval_columns, summarize_coverage

Post-processing
    build_tables

Simulation orchestration
    run_scenarios
"""

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "Scenario": ".config",
    "BASE_SCENARIO": ".config",
    "RESULTS_DIR": ".config",
    "full_scenarios": ".config",
    "smoke_scenarios": ".config",
    "generate_dataset": ".data_generation",
    "generate_standard_datasets_broadcast": ".data_generation",
    "fit_all_methods": ".methods",
    "fit_rr_method": ".methods",
    "fit_or_method": ".methods",
    "METHOD_RR_GEE": ".methods",
    "METHOD_RR_PGEE": ".methods",
    "METHOD_OR_GEE": ".methods",
    "METHOD_OR_PGEE": ".methods",
    "METHOD_ORDER": ".methods",
    "OR_METHODS": ".methods",
    "add_interval_columns": ".coverage",
    "summarize_coverage": ".coverage",
    "build_tables": ".postprocess",
    "run_scenarios": ".run_simulation",
}


def __getattr__(name: str) -> Any:
    """Load public objects on demand so lightweight CLI commands stay fast."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    # config
    "Scenario",
    "BASE_SCENARIO",
    "RESULTS_DIR",
    "full_scenarios",
    "smoke_scenarios",
    # data generation
    "generate_dataset",
    "generate_standard_datasets_broadcast",
    # methods
    "fit_all_methods",
    "fit_rr_method",
    "fit_or_method",
    "METHOD_RR_GEE",
    "METHOD_RR_PGEE",
    "METHOD_OR_GEE",
    "METHOD_OR_PGEE",
    "METHOD_ORDER",
    "OR_METHODS",
    # coverage
    "add_interval_columns",
    "summarize_coverage",
    # postprocess
    "build_tables",
    # simulation
    "run_scenarios",
]
