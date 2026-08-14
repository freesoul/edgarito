"""Compatibility facade for deterministic operating forecasts.

The implementation lives in :mod:`edgarito.services.operating._forecast`.
This module intentionally keeps the historical import path stable for callers
and for the operating-integration boundary.  The old module also exposed its
imported schema helpers, so they are explicitly rebound here instead of being
silently lost when the calculation stages move.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Any

from edgarito.config.operating import OPERATING_VOCABULARY
from edgarito.schemas.operating import (
    CompanyOperatingForecast,
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverForecast,
    OperatingDriverObservation,
    OperatingSegment,
    SegmentRevenueForecast,
    canonical_operating_segment_id,
    operating_periods_compatible,
    operating_units_compatible,
)
from edgarito.services.operating._forecast import service as _implementation
from edgarito.services.operating._forecast.contracts import (
    _HIGH_DRIVER_COVERAGE,
    _HIGH_RECONSTRUCTION_ERROR,
)
from edgarito.services.operating._forecast.service import (
    ARCHETYPE_FORMULAS,
    FORMULA_REGISTRY,
    ArchetypeFormulaRegistry,
    DeterministicOperatingForecastService,
    OperatingForecastEngine,
    OperatingForecastService,
    normalize_company_historical_revenue,
)

# These are deliberate compatibility exports, not unused implementation
# imports.  They were all importable from this module before the split.
# ruff: noqa: F401, F811

__all__ = [
    "ARCHETYPE_FORMULAS",
    "ArchetypeFormulaRegistry",
    "CompanyOperatingForecast",
    "DeterministicOperatingForecastService",
    "FORMULA_REGISTRY",
    "_HIGH_DRIVER_COVERAGE",
    "_HIGH_RECONSTRUCTION_ERROR",
    "OperatingForecastEngine",
    "OperatingForecastService",
    "SegmentRevenueForecast",
    "normalize_company_historical_revenue",
]


def __getattr__(name: str):
    """Resolve legacy private helper imports from the moved implementation."""

    return getattr(_implementation, name)


def __dir__():
    return sorted({*globals(), *vars(_implementation)})
