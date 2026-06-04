#!/usr/bin/env python3
"""Run the translated penalized log-Poisson GEE interaction model."""
from __future__ import annotations
import sys
from pathlib import Path
from Sept21_pgee_logPoisson_dispersion_fn import GEEDIR, run_subset

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Wrong number of arguments.")
    run_subset(int(float(sys.argv[1])), int(float(sys.argv[2])), GEEDIR / "temp_Sept_pgee_interaction", "pgee", True)
