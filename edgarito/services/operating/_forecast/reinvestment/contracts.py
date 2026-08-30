"""Small internal contracts shared by reinvestment selectors and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from edgarito.schemas.forecasting import ForecastStrategy, ForecastValueBasis
from edgarito.schemas.operating import EvidenceReference

_DA = "depreciation_and_amortization"
_CAPEX = "capital_expenditures"
_OWC = "operating_working_capital"
_DELTA = "change_in_operating_working_capital"
_FCFF = "fcff"
_REVENUE = "revenue"
_MONEY_UNITS = frozenset(
    {"usd", "eur", "gbp", "jpy", "cny", "cad", "aud", "chf", "currency"}
)
_RATE_UNITS = frozenset(
    {
        "%",
        "percent",
        "percentage",
        "percentage_points",
        "ratio",
        "rate",
        "decimal",
        "fraction",
        "bps",
        "bp",
    }
)
_DIRECT_ORIGINS = frozenset(
    {
        "reported",
        "first_party_observation",
        "extracted_evidence",
        "derived",
        "forward_evidence",
    }
)
_METRICS = frozenset({_DA, _CAPEX, _OWC})
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_UNAVAILABLE = "unavailable"
_HUNDRED = Decimal(100)


@dataclass(frozen=True)
class _Candidate:
    value: Decimal
    source: str
    method: str
    confidence: str
    provenance: Any = None
    references: tuple[EvidenceReference, ...] = ()
    audit: tuple[str, ...] = ()
    ratio: Decimal | None = None
    historical_years: tuple[int, ...] = ()
    provenance_chain: tuple[Any, ...] = ()
    segment_id: str | None = None
    is_component: bool = False
    exhaustive: bool = False
    residual: bool = False
    historical_amount: Decimal | None = None


@dataclass(frozen=True)
class _Path:
    values: tuple[Decimal, ...]
    strategy: ForecastStrategy
    basis: ForecastValueBasis
    provenance: Any = None
    references: tuple[EvidenceReference, ...] = ()
    residual: bool = False


__all__ = [
    "_CAPEX",
    "_Candidate",
    "_CONFIDENCE_RANK",
    "_DA",
    "_DELTA",
    "_DIRECT_ORIGINS",
    "_FCFF",
    "_HUNDRED",
    "_METRICS",
    "_MONEY_UNITS",
    "_OWC",
    "_Path",
    "_RATE_UNITS",
    "_REVENUE",
    "_UNAVAILABLE",
]
