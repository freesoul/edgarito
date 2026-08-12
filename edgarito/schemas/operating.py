"""Provider-neutral contracts for an independent operating forecast.

This module intentionally contains data contracts only.  Discovery, evidence
extraction, deterministic driver formulas, and valuation integration belong to
separate services and should consume these models without importing providers.
"""

from __future__ import annotations

import datetime
import re
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgarito.schemas.valuation.assumptions import AssumptionProvenance

_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")
_MIN_FISCAL_YEAR = 1900
_MAX_FISCAL_YEAR = 2200
_CONFIDENCE_LEVELS = {"high", "medium", "low"}


class OperatingArchetype(str, Enum):
    """Reusable economic relationship for a forecastable operating segment."""

    VOLUME_PRICE = "volume_price"
    SUBSCRIBERS_ARPU = "subscribers_arpu"
    CAPACITY_UTILIZATION_PRICE = "capacity_utilization_price"
    TRANSACTIONS_TAKE_RATE = "transactions_take_rate"
    BACKLOG_CONVERSION = "backlog_conversion"
    STORE_COUNT_SALES_PER_STORE = "store_count_sales_per_store"
    GENERIC_SEGMENT_GROWTH = "generic_segment_growth"


class EvidenceReference(BaseModel):
    """Minimal source pointer retained with extracted operating evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    accession: str | None = None
    filing_date: datetime.date | None = None
    document_name: str | None = None
    source_text_hash: str | None = None
    supporting_text: str | None = None

    @field_validator(
        "provider", "accession", "document_name", "source_text_hash", "supporting_text"
    )
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Evidence reference text fields cannot be blank")
        return normalized

    @property
    def accession_number(self) -> str | None:
        """Compatibility name used by the normalized filing schemas."""

        return self.accession


class OperatingSegment(BaseModel):
    """A stable, forecastable economic unit within a company."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segment_id: str
    name: str
    parent_id: str | None = None
    scope: Literal["consolidated", "segment", "geography", "product"] = "segment"
    currency: str | None = None
    dimensions: dict[str, str] = Field(default_factory=dict)

    @field_validator("segment_id", "name", "parent_id")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Segment text")

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not _CURRENCY_PATTERN.fullmatch(normalized):
            raise ValueError("Segment currency must be a three-letter ISO code")
        return normalized

    @field_validator("dimensions")
    @classmethod
    def normalize_dimensions(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, item in value.items():
            normalized_key = _normalize_required_text(key, "Segment dimension key")
            normalized[normalized_key] = _normalize_required_text(
                item, "Segment dimension value"
            )
        return normalized


class OperatingDriverDefinition(BaseModel):
    """Definition of a deterministic driver relationship without forecast data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    driver_id: str
    archetype: OperatingArchetype
    segment_id: str
    output_metric: Literal["revenue", "volume", "price", "margin"]
    input_metrics: tuple[str, ...]
    units: dict[str, str]
    formula_id: str
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...] = ()

    @field_validator("driver_id", "segment_id", "formula_id")
    @classmethod
    def normalize_identifiers(cls, value: str) -> str:
        return _normalize_required_text(value, "Driver identifier")

    @field_validator("input_metrics", "required_inputs", "optional_inputs")
    @classmethod
    def normalize_metric_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            _normalize_required_text(item, "Driver metric") for item in value
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError("Driver metric names must be unique")
        return normalized

    @field_validator("units")
    @classmethod
    def normalize_units(cls, value: dict[str, str]) -> dict[str, str]:
        return {
            _normalize_required_text(
                metric, "Driver unit metric"
            ): _normalize_required_text(unit, "Driver unit")
            for metric, unit in value.items()
        }

    @model_validator(mode="after")
    def validate_inputs(self) -> "OperatingDriverDefinition":
        if not self.input_metrics:
            raise ValueError("Operating driver requires at least one input metric")
        if not self.required_inputs:
            raise ValueError("Operating driver requires at least one required input")
        if not set(self.required_inputs).issubset(self.input_metrics):
            raise ValueError("Required driver inputs must be listed in input_metrics")
        if not set(self.optional_inputs).issubset(self.input_metrics):
            raise ValueError("Optional driver inputs must be listed in input_metrics")
        if set(self.required_inputs) & set(self.optional_inputs):
            raise ValueError("Required and optional driver inputs must be disjoint")
        missing_units = set(self.input_metrics) - set(self.units)
        if missing_units:
            raise ValueError(
                "Operating driver units are missing for: "
                + ", ".join(sorted(missing_units))
            )
        return self


class OperatingDriverObservation(BaseModel):
    """A reported or extracted operating-driver observation for one fiscal period."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segment_id: str
    driver_id: str
    fiscal_year: int = Field(ge=_MIN_FISCAL_YEAR, le=_MAX_FISCAL_YEAR)
    fiscal_period: str = "FY"
    value: Decimal | None = Field(default=None, allow_inf_nan=True)
    low: Decimal | None = Field(default=None, allow_inf_nan=True)
    high: Decimal | None = Field(default=None, allow_inf_nan=True)
    unit: str
    currency: str | None = None
    basis: str | None = None
    origin: Literal["reported", "management_guidance", "derived", "extracted_evidence"]
    confidence: Literal["high", "medium", "low"]
    provenance: AssumptionProvenance | None = None
    evidence: EvidenceReference | None = None

    @field_validator("segment_id", "driver_id")
    @classmethod
    def normalize_identifiers(cls, value: str) -> str:
        return _normalize_required_text(value, "Operating observation identifier")

    @field_validator("fiscal_period", "unit", "basis")
    @classmethod
    def normalize_period_and_units(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Operating observation text")

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not _CURRENCY_PATTERN.fullmatch(normalized):
            raise ValueError(
                "Operating observation currency must be a three-letter ISO code"
            )
        return normalized

    @field_validator("origin", "confidence", mode="before")
    @classmethod
    def normalize_choices(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @field_validator("value", "low", "high")
    @classmethod
    def validate_decimal(cls, value: Decimal | None) -> Decimal | None:
        return _finite_decimal(value, "Operating observations")

    @model_validator(mode="after")
    def validate_range(self) -> "OperatingDriverObservation":
        if self.value is None and self.low is None and self.high is None:
            raise ValueError("Operating observation requires a value or range")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("Operating observation low cannot exceed high")
        if self.value is not None:
            if self.low is not None and self.value < self.low:
                raise ValueError("Operating observation value cannot be below low")
            if self.high is not None and self.value > self.high:
                raise ValueError("Operating observation value cannot exceed high")
        return self


class OperatingDriverForecast(BaseModel):
    """A deterministic forecast value for one operating driver and fiscal year."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segment_id: str
    driver_id: str
    fiscal_year: int = Field(ge=_MIN_FISCAL_YEAR, le=_MAX_FISCAL_YEAR)
    value: Decimal
    unit: str
    source: str
    method: str
    confidence: Literal["high", "medium", "low"]
    constraint: str | None = None
    provenance: AssumptionProvenance | EvidenceReference | None = None

    @field_validator("segment_id", "driver_id")
    @classmethod
    def normalize_identifiers(cls, value: str) -> str:
        return _normalize_required_text(value, "Operating forecast identifier")

    @field_validator("unit", "source", "method", "constraint")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Operating forecast text")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: Decimal) -> Decimal:
        return _finite_decimal(value, "Operating driver forecasts")


class SegmentRevenueForecast(BaseModel):
    """Selected revenue path and driver audit for one operating segment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segment: OperatingSegment
    fiscal_years: tuple[int, ...]
    revenue: tuple[Decimal, ...]
    revenue_growth: tuple[Decimal | None, ...]
    driver_forecasts: tuple[OperatingDriverForecast, ...] = ()
    explicit_years: tuple[int, ...] = ()
    source_by_year: dict[int, str] = Field(default_factory=dict)
    confidence_by_year: dict[int, str] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    unit: str = "currency"

    @field_validator("unit")
    @classmethod
    def normalize_unit(cls, value: str) -> str:
        return _normalize_required_text(value, "Segment revenue unit")

    @field_validator("fiscal_years")
    @classmethod
    def validate_year_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        _validate_year_sequence(value, "Segment forecast years")
        return value

    @field_validator("explicit_years")
    @classmethod
    def validate_explicit_year_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value:
            _validate_year_sequence(value, "Segment explicit years")
        return value

    @field_validator("revenue")
    @classmethod
    def validate_revenue(cls, value: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
        return tuple(
            _non_negative_decimal(item, "Segment revenue values") for item in value
        )

    @field_validator("revenue_growth")
    @classmethod
    def validate_growth_values(
        cls, value: tuple[Decimal | None, ...]
    ) -> tuple[Decimal | None, ...]:
        return tuple(_finite_decimal(item, "Segment revenue growth") for item in value)

    @field_validator("source_by_year", mode="before")
    @classmethod
    def normalize_sources(cls, value: dict[int, str] | None) -> dict[int, str]:
        return {
            int(year): _normalize_required_text(
                getattr(source, "value", source), "Segment forecast source"
            )
            for year, source in (value or {}).items()
        }

    @field_validator("confidence_by_year", mode="before")
    @classmethod
    def normalize_confidences(cls, value: dict[int, str] | None) -> dict[int, str]:
        normalized = {
            int(year): str(getattr(confidence, "value", confidence)).strip().casefold()
            for year, confidence in (value or {}).items()
        }
        invalid = set(normalized.values()) - _CONFIDENCE_LEVELS
        if invalid:
            raise ValueError("Forecast confidence must be high, medium, or low")
        return normalized

    @field_validator("warnings")
    @classmethod
    def normalize_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            _normalize_required_text(item, "Segment forecast warning") for item in value
        )

    @model_validator(mode="after")
    def validate_forecast(self) -> "SegmentRevenueForecast":
        _validate_year_sequence(self.fiscal_years, "Segment forecast years")
        if len(self.revenue) != len(self.fiscal_years):
            raise ValueError(
                "Segment revenue and fiscal-year paths must have equal length"
            )
        if len(self.revenue_growth) != len(self.fiscal_years):
            raise ValueError(
                "Segment revenue-growth and fiscal-year paths must have equal length"
            )
        _validate_growth_consistency(
            self.revenue, self.revenue_growth, "Segment revenue growth"
        )
        _validate_subset(self.explicit_years, self.fiscal_years, "explicit years")
        _validate_year_map(self.source_by_year, self.fiscal_years, "source_by_year")
        _validate_year_map(
            self.confidence_by_year, self.fiscal_years, "confidence_by_year"
        )
        keys = [
            (item.segment_id, item.driver_id, item.fiscal_year)
            for item in self.driver_forecasts
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "Segment driver forecasts must be unique by driver and year"
            )
        for item in self.driver_forecasts:
            if item.segment_id != self.segment.segment_id:
                raise ValueError("Driver forecast segment_id must match the segment")
            if item.fiscal_year not in self.fiscal_years:
                raise ValueError("Driver forecast year must be in fiscal_years")
        return self


class CompanyOperatingForecast(BaseModel):
    """Consolidated provider-neutral operating revenue forecast."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    company_id: str
    fiscal_years: tuple[int, ...]
    segment_forecasts: tuple[SegmentRevenueForecast, ...] = ()
    consolidated_revenue: tuple[Decimal, ...]
    consolidated_growth: tuple[Decimal | None, ...]
    explicit_years: tuple[int, ...] = ()
    transition_start_year: int | None = Field(
        default=None, ge=_MIN_FISCAL_YEAR, le=_MAX_FISCAL_YEAR
    )
    source_by_year: dict[int, str] = Field(default_factory=dict)
    confidence_by_year: dict[int, str] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    unit: str = "currency"

    @field_validator("company_id")
    @classmethod
    def normalize_company_id(cls, value: str) -> str:
        return _normalize_required_text(value, "Company identifier")

    @field_validator("unit")
    @classmethod
    def normalize_unit(cls, value: str) -> str:
        return _normalize_required_text(value, "Company revenue unit")

    @field_validator("fiscal_years")
    @classmethod
    def validate_year_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        _validate_year_sequence(value, "Company forecast years")
        return value

    @field_validator("explicit_years")
    @classmethod
    def validate_explicit_year_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value:
            _validate_year_sequence(value, "Company explicit years")
        return value

    @field_validator("consolidated_revenue")
    @classmethod
    def validate_revenue(cls, value: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
        return tuple(
            _non_negative_decimal(item, "Consolidated revenue values") for item in value
        )

    @field_validator("consolidated_growth")
    @classmethod
    def validate_growth_values(
        cls, value: tuple[Decimal | None, ...]
    ) -> tuple[Decimal | None, ...]:
        return tuple(
            _finite_decimal(item, "Consolidated revenue growth") for item in value
        )

    @field_validator("source_by_year", mode="before")
    @classmethod
    def normalize_sources(cls, value: dict[int, str] | None) -> dict[int, str]:
        return {
            int(year): _normalize_required_text(
                getattr(source, "value", source), "Company forecast source"
            )
            for year, source in (value or {}).items()
        }

    @field_validator("confidence_by_year", mode="before")
    @classmethod
    def normalize_confidences(cls, value: dict[int, str] | None) -> dict[int, str]:
        normalized = {
            int(year): str(getattr(confidence, "value", confidence)).strip().casefold()
            for year, confidence in (value or {}).items()
        }
        invalid = set(normalized.values()) - _CONFIDENCE_LEVELS
        if invalid:
            raise ValueError("Forecast confidence must be high, medium, or low")
        return normalized

    @field_validator("warnings")
    @classmethod
    def normalize_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            _normalize_required_text(item, "Company forecast warning") for item in value
        )

    @model_validator(mode="after")
    def validate_forecast(self) -> "CompanyOperatingForecast":
        _validate_year_sequence(self.fiscal_years, "Company forecast years")
        if len(self.consolidated_revenue) != len(self.fiscal_years):
            raise ValueError(
                "Consolidated revenue and fiscal-year paths must have equal length"
            )
        if len(self.consolidated_growth) != len(self.fiscal_years):
            raise ValueError(
                "Consolidated growth and fiscal-year paths must have equal length"
            )
        _validate_growth_consistency(
            self.consolidated_revenue,
            self.consolidated_growth,
            "Consolidated revenue growth",
        )
        _validate_subset(self.explicit_years, self.fiscal_years, "explicit years")
        _validate_year_map(self.source_by_year, self.fiscal_years, "source_by_year")
        _validate_year_map(
            self.confidence_by_year, self.fiscal_years, "confidence_by_year"
        )
        segment_ids = [item.segment.segment_id for item in self.segment_forecasts]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("Company segment forecasts must have unique segment IDs")
        for item in self.segment_forecasts:
            if item.fiscal_years != self.fiscal_years:
                raise ValueError("Segment and company forecast years must match")
        if (
            self.transition_start_year is not None
            and self.explicit_years
            and self.transition_start_year <= max(self.explicit_years)
        ):
            raise ValueError(
                "Transition must start after the last explicit forecast year"
            )
        return self


def _normalize_required_text(value: str, label: str) -> str:
    normalized = str(getattr(value, "value", value)).strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized


def _normalize_optional_text(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    return _normalize_required_text(value, label)


def _finite_decimal(value: Decimal | None, label: str) -> Decimal | None:
    if value is not None and not value.is_finite():
        raise ValueError(f"{label} must be finite")
    return value


def _non_negative_decimal(value: Decimal, label: str) -> Decimal:
    if not value.is_finite():
        raise ValueError(f"{label} must be finite")
    if value < 0:
        raise ValueError(f"{label} cannot be negative")
    return value


def _validate_year_sequence(years: tuple[int, ...], label: str) -> None:
    if not years:
        raise ValueError(f"{label} cannot be empty")
    if any(year < _MIN_FISCAL_YEAR or year > _MAX_FISCAL_YEAR for year in years):
        raise ValueError(
            f"{label} must be between {_MIN_FISCAL_YEAR} and {_MAX_FISCAL_YEAR}"
        )
    if tuple(sorted(years)) != years:
        raise ValueError(f"{label} must be in ascending order")
    if len(years) != len(set(years)):
        raise ValueError(f"{label} cannot contain duplicate years")


def _validate_subset(
    values: tuple[int, ...], universe: tuple[int, ...], label: str
) -> None:
    if tuple(sorted(values)) != values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be sorted and unique")
    if not set(values).issubset(universe):
        raise ValueError(f"{label} must be contained in fiscal_years")


def _validate_year_map(
    values: dict[int, str], years: tuple[int, ...], label: str
) -> None:
    if not set(values).issubset(years):
        raise ValueError(f"{label} contains a year outside fiscal_years")


def _validate_growth_consistency(
    revenue: tuple[Decimal, ...],
    growth: tuple[Decimal | None, ...],
    label: str,
) -> None:
    """Validate percentage-point growth where an absolute prior is available."""

    for index in range(1, len(revenue)):
        observed_growth = growth[index]
        if observed_growth is None:
            continue
        previous = revenue[index - 1]
        current = revenue[index]
        if previous == 0:
            if current != 0:
                raise ValueError(f"{label} cannot be derived from zero prior revenue")
            expected = Decimal(0)
        else:
            expected = (current / previous - Decimal(1)) * Decimal(100)
        scale = max(abs(expected), abs(observed_growth), Decimal(1))
        if abs(expected - observed_growth) > scale * Decimal("1e-18"):
            raise ValueError(
                f"{label} must match growth derived from absolute revenue values"
            )


__all__ = [
    "CompanyOperatingForecast",
    "EvidenceReference",
    "OperatingArchetype",
    "OperatingDriverDefinition",
    "OperatingDriverForecast",
    "OperatingDriverObservation",
    "OperatingSegment",
    "SegmentRevenueForecast",
]
