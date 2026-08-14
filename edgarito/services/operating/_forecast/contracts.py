"""Internal value objects shared by the operating forecast components."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from edgarito.schemas.operating import (
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
)

_INDEPENDENT_SOURCE = "independent_operating"
_MANAGEMENT_SOURCE = "management_guidance"
_HISTORICAL_SOURCE = "normalized_historical"
_UNAVAILABLE_SOURCE = "unavailable"
_CONSOLIDATION_SOURCE = "mixed"
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_YEAR_MIN = 1900
_YEAR_MAX = 2200
_HIGH_DRIVER_COVERAGE = Decimal("0.80")
_MEDIUM_DRIVER_COVERAGE = Decimal("0.50")
_HIGH_RECONSTRUCTION_ERROR = Decimal("0.05")
_MEDIUM_RECONSTRUCTION_ERROR = Decimal("0.20")


@dataclass(frozen=True)
class _HistoricalRevenue:
    company: dict[int, Decimal]
    by_segment: dict[str, dict[int, Decimal]]


@dataclass(frozen=True)
class _SelectedObservation:
    observation: OperatingDriverObservation
    value: Decimal
    constraint: str | None = None

    @property
    def source(self) -> str:
        return _observation_source(self.observation)

    @property
    def confidence(self) -> str:
        return self.observation.confidence

    @property
    def provenance(self) -> Any:
        return self.observation.provenance or self.observation.evidence

    @property
    def period(self) -> str:
        return self.observation.fiscal_period

    @property
    def period_key(self) -> str | None:
        return self.observation.period_key


@dataclass(frozen=True)
class _FormulaResult:
    definition: OperatingDriverDefinition
    value: Decimal
    source: str
    confidence: str
    provenance: Any
    method: str
    constraint: str | None
    inputs: tuple[_SelectedObservation, ...]


@dataclass(frozen=True)
class _ReconstructionAudit:
    """Deterministic validation of driver formulas against reported revenue."""

    coverage: Decimal | None
    error: Decimal | None
    error_by_year: dict[int, Decimal]
    supported_years: tuple[int, ...]
    confidence: str
    warnings: tuple[str, ...]
    genuine_coverage: Decimal | None = None
    derived_reconstruction_years: tuple[int, ...] = ()


@dataclass(frozen=True)
class _ConsolidationSelection:
    """Non-overlapping segment scopes eligible for company aggregation."""

    segments: tuple[OperatingSegment, ...]
    warnings: tuple[str, ...] = ()


def _observation_source(observation: OperatingDriverObservation) -> str:
    if observation.origin == "management_guidance":
        return _MANAGEMENT_SOURCE
    return observation.origin


def _worst_confidence(confidences: tuple[str, ...] | list[str]) -> str:
    values = tuple(confidences)
    if not values:
        return "low"
    return min(values, key=lambda value: _CONFIDENCE_RANK[value])


__all__ = [
    "_CONSOLIDATION_SOURCE",
    "_CONFIDENCE_RANK",
    "_FormulaResult",
    "_HistoricalRevenue",
    "_ConsolidationSelection",
    "_ReconstructionAudit",
    "_SelectedObservation",
    "_INDEPENDENT_SOURCE",
    "_MANAGEMENT_SOURCE",
    "_HISTORICAL_SOURCE",
    "_UNAVAILABLE_SOURCE",
    "_YEAR_MIN",
    "_YEAR_MAX",
    "_HIGH_DRIVER_COVERAGE",
    "_MEDIUM_DRIVER_COVERAGE",
    "_HIGH_RECONSTRUCTION_ERROR",
    "_MEDIUM_RECONSTRUCTION_ERROR",
]
