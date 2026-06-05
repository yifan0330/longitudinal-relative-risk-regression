"""March 2023 variant of the penalized logistic GEE routines."""

from __future__ import annotations

try:
    from .PGEE_source import *  # noqa: F401,F403
    from .PGEE_source import _geefirth_impl
except ImportError:
    from PGEE_source import *  # noqa: F401,F403
    from PGEE_source import _geefirth_impl


def geefirth(y, x, id, ar=True):
    return _geefirth_impl(y, x, id, ar=ar, init_offset=0.0001, keep_trace=True)
