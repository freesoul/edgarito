"""Provider-neutral contracts for an independent operating forecast.

This module intentionally contains data contracts only.  Discovery, evidence
extraction, deterministic driver formulas, and valuation integration belong to
separate services and should consume these models without importing providers.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Mapping
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    WithJsonSchema,
    field_validator,
    model_serializer,
    model_validator,
)

from edgarito.schemas.forecasting import ForecastAssumptionSource, ForecastProvenance
from edgarito.schemas.operating_normalization import (
    _CONFIDENCE_LEVELS,
    _MAX_FISCAL_YEAR,
    _MIN_FISCAL_YEAR,
    _OPERATING_PERIODS,  # noqa: F401
    _archetype_metrics,
    _canonical_metric_name,  # noqa: F401
    _canonical_segment_display,  # noqa: F401
    _canonical_segment_text,  # noqa: F401
    _coerce_extracted_text_map,
    _coerce_operating_archetype,
    _coerce_operating_period_fields,
    _finite_decimal,
    _non_negative_decimal,
    _normalize_optional_text,
    _normalize_required_text,
    _observation_scaled_value,
    _operating_unit_scale_aliases,  # noqa: F401
    _period_key,  # noqa: F401
    _scale_value,  # noqa: F401
    _unit_dimension,  # noqa: F401
    _validate_growth_consistency,
    _validate_subset,
    _validate_year_map,
    _validate_year_sequence,
    canonical_operating_segment_id,
    canonical_operating_segment_identity,
    normalize_operating_fiscal_period,
    normalize_operating_unit,
    operating_periods_compatible,
    operating_units_compatible,
)
from edgarito.schemas.valuation.assumptions import AssumptionProvenance

_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")


class ResolvedRevenueYear(BaseModel):
    """The selected revenue evidence for one fiscal year.

    ``independent_revenue`` and ``consensus_revenue`` are retained even when
    they lose precedence.  That makes the selection auditable without making
    the FCFF/DCF layer aware of operating-forecast details.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fiscal_year: int = Field(ge=_MIN_FISCAL_YEAR, le=_MAX_FISCAL_YEAR)
    revenue: Decimal
    source: str
    confidence: str
    independent_revenue: Decimal | None = None
    independent_source: str | None = None
    independent_confidence: str | None = None
    consensus_revenue: Decimal | None = None
    consensus_source: str | None = None
    consensus_confidence: str | None = None
    historical_revenue: Decimal | None = None
    explicit_revenue: Decimal | None = None
    management_revenue: Decimal | None = None
    # Percentage variance of consensus against the independent value.
    variance: Decimal | None = None

    @field_validator(
        "revenue",
        "independent_revenue",
        "consensus_revenue",
        "historical_revenue",
        "explicit_revenue",
        "management_revenue",
    )
    @classmethod
    def validate_revenue_values(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and (not value.is_finite() or value < 0):
            raise ValueError("Resolved revenue values must be finite and non-negative")
        return value

    @field_validator("variance")
    @classmethod
    def validate_variance(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and not value.is_finite():
            raise ValueError("Resolved revenue variance must be finite")
        return value

    @field_validator("source", "independent_source", "consensus_source")
    @classmethod
    def normalize_sources(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(getattr(value, "value", value)).strip()
        return normalized or None

    @field_validator(
        "confidence",
        "independent_confidence",
        "consensus_confidence",
        mode="before",
    )
    @classmethod
    def normalize_confidences(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(getattr(value, "value", value)).strip().casefold()
        if normalized not in _CONFIDENCE_LEVELS:
            raise ValueError("Revenue confidence must be high, medium, or low")
        return normalized

    @property
    def selected_revenue(self) -> Decimal:
        """Descriptive alias for the selected absolute revenue."""

        return self.revenue

# OpenAI Structured Outputs does not accept an object whose
# ``additionalProperties`` value is another schema.  Pydantic emits exactly
# that shape for ``dict[str, str]``.  Keep mappings in the provider-neutral
# Python contract, but expose a strict list-of-pairs shape to the model.  The
# before validators below also accept the mapping shape used by existing local
# fixtures and callers.
_TEXT_MAP_JSON_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["key", "value"],
        "additionalProperties": False,
    },
}
ExtractedTextMap = Annotated[
    dict[str, str],
    WithJsonSchema(_TEXT_MAP_JSON_SCHEMA),
]
ExtractedRevenueMetric = Annotated[
    Literal["revenue"],
    WithJsonSchema({"type": "string", "enum": ["revenue"]}),
]


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
    source: str = "extracted_evidence"
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence: EvidenceReference | None = None

    @field_validator("name")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Segment text")

    @field_validator("segment_id", "parent_id")
    @classmethod
    def normalize_segment_ids(cls, value: str | None) -> str | None:
        return canonical_operating_segment_id(value)

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

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return _normalize_required_text(value, "Segment source")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @model_validator(mode="before")
    @classmethod
    def canonicalize_identity_input(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        segment_id, name = canonical_operating_segment_identity(
            data.get("segment_id", ""), data.get("name")
        )
        data["segment_id"] = segment_id
        data["name"] = name
        data["parent_id"] = canonical_operating_segment_id(data.get("parent_id"))
        return data


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
    source: str = "extracted_evidence"
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence: EvidenceReference | None = None

    @field_validator("driver_id", "formula_id")
    @classmethod
    def normalize_identifiers(cls, value: str) -> str:
        return _normalize_required_text(value, "Driver identifier")

    @field_validator("segment_id")
    @classmethod
    def normalize_definition_segment_id(cls, value: str) -> str:
        return canonical_operating_segment_id(value) or value

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

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return _normalize_required_text(value, "Driver source")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @model_validator(mode="before")
    @classmethod
    def canonicalize_segment_input(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        data["segment_id"] = canonical_operating_segment_id(data.get("segment_id"))
        return data

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
    period_key: str | None = None
    value: Decimal | None = Field(default=None, allow_inf_nan=True)
    low: Decimal | None = Field(default=None, allow_inf_nan=True)
    high: Decimal | None = Field(default=None, allow_inf_nan=True)
    unit: str
    currency: str | None = None
    basis: str | None = None
    # Generic scope metadata keeps cross-document driver joins auditable without
    # introducing issuer-specific scope vocabularies.
    scope: str | None = None
    scope_evidence: str | None = None
    is_total: bool = False
    is_component: bool = False
    exhaustive: bool = False
    scale: Decimal = Decimal(1)
    # ``unit`` and ``scale`` are canonical formula inputs.  These fields retain
    # the source declaration so cross-filing normalization remains auditable.
    original_unit: str | None = None
    original_scale: Decimal = Decimal(1)
    method: str | None = None
    origin: Literal[
        "reported",
        "first_party_observation",
        "management_guidance",
        "derived",
        "extracted_evidence",
    ]
    confidence: Literal["high", "medium", "low"]
    provenance: AssumptionProvenance | EvidenceReference | None = None
    evidence: EvidenceReference | None = None
    # A derived observation may combine compatible facts from multiple SEC
    # documents. Keep every source instead of reducing the audit trail to the
    # first evidence pointer.
    source_provenance: tuple[EvidenceReference, ...] = ()

    @field_validator("driver_id")
    @classmethod
    def normalize_identifiers(cls, value: str) -> str:
        return _normalize_required_text(value, "Operating observation identifier")

    @field_validator("segment_id")
    @classmethod
    def normalize_observation_segment_id(cls, value: str) -> str:
        return canonical_operating_segment_id(value) or value

    @field_validator("fiscal_period")
    @classmethod
    def normalize_period(cls, value: str) -> str:
        return normalize_operating_fiscal_period(value)

    @field_validator("period_key")
    @classmethod
    def normalize_period_key(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Operating observation period key")

    @field_validator(
        "unit", "original_unit", "basis", "scope", "scope_evidence", "method"
    )
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

    @model_validator(mode="before")
    @classmethod
    def canonicalize_segment_input(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if data.get("segment_id") is not None:
            data["segment_id"] = canonical_operating_segment_id(data["segment_id"])
        return data

    @model_validator(mode="before")
    @classmethod
    def canonicalize_period_input(cls, value: Any) -> Any:
        data = _coerce_operating_period_fields(value)
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        if data.get("segment_id") is not None:
            data["segment_id"] = canonical_operating_segment_id(data["segment_id"])
        raw_unit = data.get("original_unit") or data.get("unit", "unit")
        raw_scale = data.get("original_scale", data.get("scale", 1))
        data["original_unit"] = raw_unit
        data["original_scale"] = Decimal(str(raw_scale))
        unit, unit_scale = normalize_operating_unit(
            data.get("unit", "unit"), data.get("driver_id")
        )
        data["unit"] = unit
        data["scale"] = Decimal(str(data.get("scale", 1))) * unit_scale
        return data

    @field_validator("value", "low", "high")
    @classmethod
    def validate_decimal(cls, value: Decimal | None) -> Decimal | None:
        return _finite_decimal(value, "Operating observations")

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, value: Decimal) -> Decimal:
        normalized = _finite_decimal(value, "Operating observation scale")
        if normalized is None or normalized <= 0:
            raise ValueError("Operating observation scale must be positive")
        return normalized

    @field_validator("original_scale")
    @classmethod
    def validate_original_scale(cls, value: Decimal) -> Decimal:
        normalized = _finite_decimal(value, "Original operating observation scale")
        if normalized is None or normalized <= 0:
            raise ValueError("Original operating observation scale must be positive")
        return normalized

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

    @property
    def normalized_value(self) -> Decimal:
        """Return the value in the canonical base unit used by formulas."""

        if self.value is not None:
            return _observation_scaled_value(self.value, self.scale)
        if self.low is not None and self.high is not None:
            return (self.low + self.high) / Decimal(2) * self.scale
        if self.low is not None:
            return self.low * self.scale
        if self.high is not None:
            return self.high * self.scale
        raise ValueError("Operating observation has no usable value")

    @property
    def source_unit(self) -> str | None:
        """Compatibility alias for the unit declared by the source."""

        return self.original_unit

    @property
    def source_scale(self) -> Decimal:
        """Compatibility alias for the scale declared by the source."""

        return self.original_scale


class OperatingEvidenceGap(BaseModel):
    """A provider-neutral missing operating fact for one period."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    segment_id: str = Field(validation_alias=AliasChoices("segment_id", "segment"))
    metric: str = Field(
        validation_alias=AliasChoices("metric", "driver_id", "required_metric")
    )
    fiscal_year: int | None = Field(
        default=None,
        validation_alias=AliasChoices("fiscal_year", "year"),
        ge=_MIN_FISCAL_YEAR,
        le=_MAX_FISCAL_YEAR,
    )
    fiscal_period: str = Field(
        default="FY", validation_alias=AliasChoices("fiscal_period", "period")
    )
    period_key: str | None = None
    reason: str = Field(
        default="missing_required_input",
        validation_alias=AliasChoices("reason", "gap_type"),
    )
    status: str = "unresolved"
    source_documents: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def infer_quarter_key(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        period = data.get("fiscal_period", data.get("period"))
        if isinstance(period, str) and period.strip().upper() in {
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        }:
            data.setdefault("period_key", period.strip().upper())
        return data

    @field_validator("segment_id")
    @classmethod
    def normalize_segment_id(cls, value: str) -> str:
        return canonical_operating_segment_id(value) or value.strip()

    @field_validator("metric", "reason", "status")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Operating evidence gap text cannot be blank")
        return normalized

    @field_validator("fiscal_period")
    @classmethod
    def normalize_period(cls, value: str) -> str:
        normalized = str(value).strip().upper()
        if normalized in {"Q1", "Q2", "Q3", "Q4"}:
            return "FQ"
        if normalized not in {"FY", "FQ", "YTD", "LTM"}:
            raise ValueError("Operating evidence gap period is not supported")
        return normalized

    @field_validator("period_key")
    @classmethod
    def normalize_period_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().upper()
        return normalized or None

    @field_validator("source_documents", "diagnostics")
    @classmethod
    def normalize_texts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(str(item).strip() for item in value if str(item).strip())
        )

    @property
    def driver_id(self) -> str:
        """Compatibility alias for callers using driver terminology."""

        return self.metric

    @property
    def key(self) -> tuple[str, str, int | None, str, str | None]:
        return (
            self.segment_id,
            self.metric.casefold(),
            self.fiscal_year,
            self.fiscal_period,
            self.period_key,
        )

    @property
    def label(self) -> str:
        year = f"FY{self.fiscal_year}" if self.fiscal_year is not None else "FY?"
        period = self.fiscal_period
        if self.period_key:
            period = f"{period}/{self.period_key}"
        return f"{self.segment_id}/{self.metric}/{year}/{period}"

    @property
    def gap_type(self) -> str:
        return self.reason

    @property
    def period(self) -> str:
        return self.fiscal_period


class OperatingInvestmentProgram(BaseModel):
    """A first-party investment or capacity program fact.

    Investment programs are retained as evidence for audit and downstream
    consumers.  They are deliberately not converted into revenue, growth, or
    cash-flow forecasts by the operating discovery layer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    program_id: str
    name: str
    segment_id: str | None = None
    fiscal_year: int | None = Field(
        default=None, ge=_MIN_FISCAL_YEAR, le=_MAX_FISCAL_YEAR
    )
    fiscal_period: str = "FY"
    period_key: str | None = None
    scale: Decimal = Decimal(1)
    value: Decimal | None = Field(default=None, allow_inf_nan=True)
    low: Decimal | None = Field(default=None, allow_inf_nan=True)
    high: Decimal | None = Field(default=None, allow_inf_nan=True)
    unit: str = "unspecified"
    currency: str | None = None
    status: Literal[
        "announced",
        "planned",
        "in_progress",
        "under_construction",
        "completed",
        "reported",
        "unknown",
    ] = "reported"
    purpose: str | None = None
    source: str = "first_party_filing"
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence: EvidenceReference | None = None

    @field_validator("program_id", "name", "fiscal_period", "unit", "purpose")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        normalized = _normalize_optional_text(value, "Investment program text")
        return normalized or ("FY" if value == "" else normalized)

    @field_validator("segment_id")
    @classmethod
    def normalize_program_segment_id(cls, value: str | None) -> str | None:
        return canonical_operating_segment_id(value)

    @field_validator("fiscal_period")
    @classmethod
    def normalize_period(cls, value: str) -> str:
        return normalize_operating_fiscal_period(value)

    @field_validator("period_key")
    @classmethod
    def normalize_period_key(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Investment program period key")

    @model_validator(mode="before")
    @classmethod
    def canonicalize_period_input(cls, value: Any) -> Any:
        data = _coerce_operating_period_fields(value)
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        if data.get("segment_id") is not None:
            data["segment_id"] = canonical_operating_segment_id(data["segment_id"])
        return data

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not _CURRENCY_PATTERN.fullmatch(normalized):
            raise ValueError(
                "Investment program currency must be a three-letter ISO code"
            )
        return normalized

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, value: Decimal) -> Decimal:
        normalized = _finite_decimal(value, "Investment program scale")
        if normalized is None or normalized <= 0:
            raise ValueError("Investment program scale must be positive")
        return normalized

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return _normalize_required_text(value, "Investment program source")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @field_validator("value", "low", "high")
    @classmethod
    def validate_decimal(cls, value: Decimal | None) -> Decimal | None:
        return _finite_decimal(value, "Investment program values")

    @model_validator(mode="after")
    def validate_range(self) -> "OperatingInvestmentProgram":
        if self.value is None and self.low is None and self.high is None:
            # A program can be a qualitative first-party fact, such as an
            # announced facility, without a disclosed amount or capacity.
            return self
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("Investment program low cannot exceed high")
        if self.value is not None:
            if self.low is not None and self.value < self.low:
                raise ValueError("Investment program value cannot be below low")
            if self.high is not None and self.value > self.high:
                raise ValueError("Investment program value cannot exceed high")
        return self

    @property
    def amount(self) -> Decimal | None:
        """Compatibility name for programs disclosed as monetary amounts."""

        return self.value


class ExtractedOperatingSegment(BaseModel):
    """Untrusted structured output for one first-party operating segment."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str
    name: str
    parent_id: str | None = None
    scope: Literal["consolidated", "segment", "geography", "product"] = "segment"
    currency: str | None = None
    dimensions: ExtractedTextMap = Field(default_factory=dict)
    supporting_text: str
    confidence: Literal["high", "medium", "low"] = "medium"

    @field_validator("segment_id", "name", "parent_id", "supporting_text")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Extracted operating segment text")

    @field_validator("dimensions", mode="before")
    @classmethod
    def coerce_dimensions(cls, value: Any) -> dict[str, str]:
        return _coerce_extracted_text_map(value, "dimension")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not _CURRENCY_PATTERN.fullmatch(normalized):
            raise ValueError(
                "Extracted operating segment currency must be a three-letter ISO code"
            )
        return normalized

    @field_validator("dimensions")
    @classmethod
    def normalize_dimensions(cls, value: dict[str, str]) -> dict[str, str]:
        return {
            _normalize_required_text(
                key, "Extracted segment dimension key"
            ): _normalize_required_text(item, "Extracted segment dimension value")
            for key, item in value.items()
        }

    @model_validator(mode="before")
    @classmethod
    def canonicalize_identity_input(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        segment_id, name = canonical_operating_segment_identity(
            data.get("segment_id", ""), data.get("name")
        )
        data["segment_id"] = segment_id
        data["name"] = name
        data["parent_id"] = canonical_operating_segment_id(data.get("parent_id"))
        return data


class ExtractedOperatingDriverDefinition(BaseModel):
    """Untrusted structured output for an archetype mapping.

    This schema intentionally has no forecast values.  The model may describe
    a reported economic relationship, but it cannot return revenue, growth, or
    any other forecast path.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    driver_id: str
    archetype: OperatingArchetype
    segment_id: str
    output_metric: ExtractedRevenueMetric = "revenue"
    input_metrics: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("input_metrics", "inputs"),
    )
    units: ExtractedTextMap = Field(default_factory=dict)
    required_inputs: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("required_inputs", "required_metrics"),
    )
    optional_inputs: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("optional_inputs", "optional_metrics"),
    )
    formula_id: str | None = None
    supporting_text: str
    confidence: Literal["high", "medium", "low"] = "medium"

    @field_validator("archetype", mode="before")
    @classmethod
    def normalize_archetype(cls, value: OperatingArchetype | str) -> OperatingArchetype:
        return _coerce_operating_archetype(value)

    @field_validator("driver_id", "segment_id", "formula_id", "supporting_text")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Extracted operating definition text")

    @field_validator("units", mode="before")
    @classmethod
    def coerce_units(cls, value: Any) -> dict[str, str]:
        return _coerce_extracted_text_map(value, "unit")

    @field_validator("input_metrics", "required_inputs", "optional_inputs")
    @classmethod
    def normalize_metrics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            _normalize_required_text(item, "Extracted operating metric")
            for item in value
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError("Extracted operating metric names must be unique")
        return normalized

    @field_validator("units")
    @classmethod
    def normalize_units(cls, value: dict[str, str]) -> dict[str, str]:
        return {
            _normalize_required_text(
                metric, "Extracted operating unit metric"
            ): _normalize_required_text(unit, "Extracted operating unit")
            for metric, unit in value.items()
        }

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @model_validator(mode="before")
    @classmethod
    def populate_archetype_defaults(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if data.get("segment_id") is not None:
            data["segment_id"] = canonical_operating_segment_id(data["segment_id"])
        if "inputs" in data and "input_metrics" not in data:
            data["input_metrics"] = data.pop("inputs")
        if "required_metrics" in data and "required_inputs" not in data:
            data["required_inputs"] = data.pop("required_metrics")
        if "optional_metrics" in data and "optional_inputs" not in data:
            data["optional_inputs"] = data.pop("optional_metrics")
        if data.get("archetype") is None:
            return data
        archetype = _coerce_operating_archetype(data["archetype"])
        metrics = list(data.get("input_metrics") or _archetype_metrics(archetype))
        data.setdefault("input_metrics", metrics)
        data.setdefault("required_inputs", list(data.get("required_inputs") or metrics))
        data.setdefault(
            "units",
            {metric: "unspecified" for metric in metrics},
        )
        data.setdefault("formula_id", archetype.value)
        return data

    @model_validator(mode="after")
    def validate_definition_shape(self) -> "ExtractedOperatingDriverDefinition":
        input_metrics = self.input_metrics or _archetype_metrics(self.archetype)
        required = self.required_inputs or input_metrics
        # The extraction model sometimes reports an archetype's canonical
        # optional input (for example utilization) without listing it in its
        # input_metrics array. Treat the canonical archetype contract as the
        # boundary rather than rejecting otherwise usable evidence.
        canonical_metrics = set(_archetype_metrics(self.archetype))
        input_metrics = tuple(
            dict.fromkeys((*input_metrics, *canonical_metrics, *self.optional_inputs))
        )
        input_metrics = tuple(dict.fromkeys((*input_metrics, *required)))
        optional_inputs = tuple(
            item for item in self.optional_inputs if item not in required
        )
        input_metrics = tuple(dict.fromkeys((*input_metrics, *optional_inputs)))
        if set(required) & set(optional_inputs):
            raise ValueError("Extracted required and optional inputs must be disjoint")
        units = {
            metric: self.units.get(metric, "unspecified") for metric in input_metrics
        }
        self.input_metrics = list(input_metrics)
        self.required_inputs = list(required)
        self.optional_inputs = list(optional_inputs)
        self.units = units
        return self


class ExtractedOperatingObservation(BaseModel):
    """Untrusted structured output for a reported operating fact.

    Only observations are allowed here.  There is deliberately no forecast,
    growth-path, or consensus field in this contract.
    """

    model_config = ConfigDict(extra="forbid")

    segment_id: str
    driver_id: str
    fiscal_year: int = Field(ge=_MIN_FISCAL_YEAR, le=_MAX_FISCAL_YEAR)
    fiscal_period: str = "FY"
    period_key: str | None = None
    value: float | None = None
    low: float | None = None
    high: float | None = None
    unit: str
    scale: float = 1
    original_unit: str | None = None
    original_scale: float = 1
    currency: str | None = None
    basis: str | None = None
    scope: str | None = None
    scope_evidence: str | None = None
    is_total: bool = False
    is_component: bool = False
    exhaustive: bool = False
    origin: Literal["reported", "first_party_observation", "management_guidance"] = (
        "reported"
    )
    supporting_text: str
    confidence: Literal["high", "medium", "low"] = "medium"

    @field_validator(
        "segment_id",
        "driver_id",
        "unit",
        "original_unit",
        "basis",
        "scope",
        "scope_evidence",
        "supporting_text",
    )
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Extracted operating observation text")

    @field_validator("fiscal_period")
    @classmethod
    def normalize_period(cls, value: str) -> str:
        return normalize_operating_fiscal_period(value)

    @field_validator("period_key")
    @classmethod
    def normalize_period_key(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Extracted observation period key")

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not _CURRENCY_PATTERN.fullmatch(normalized):
            raise ValueError(
                "Extracted operating observation currency must be a three-letter ISO code"
            )
        return normalized

    @field_validator("value", "low", "high")
    @classmethod
    def validate_number(cls, value: float | None) -> float | None:
        if value is not None:
            _finite_decimal(Decimal(str(value)), "Extracted operating observations")
        return value

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, value: float) -> float:
        if value <= 0 or not Decimal(str(value)).is_finite():
            raise ValueError("Extracted operating observation scale must be positive")
        return value

    @field_validator("original_scale")
    @classmethod
    def validate_original_scale(cls, value: float) -> float:
        if value <= 0 or not Decimal(str(value)).is_finite():
            raise ValueError(
                "Extracted original operating observation scale must be positive"
            )
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @model_validator(mode="before")
    @classmethod
    def canonicalize_segment_input(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if data.get("segment_id") is not None:
            data["segment_id"] = canonical_operating_segment_id(data["segment_id"])
        return data

    @model_validator(mode="before")
    @classmethod
    def canonicalize_period_input(cls, value: Any) -> Any:
        data = _coerce_operating_period_fields(value)
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        data["original_unit"] = data.get("original_unit") or data.get("unit", "unit")
        data["original_scale"] = data.get("original_scale", data.get("scale", 1))
        unit, unit_scale = normalize_operating_unit(
            data.get("unit", "unit"), data.get("driver_id")
        )
        data["unit"] = unit
        data["scale"] = float(Decimal(str(data.get("scale", 1))) * unit_scale)
        return data

    @model_validator(mode="after")
    def require_value(self) -> "ExtractedOperatingObservation":
        if self.value is None and self.low is None and self.high is None:
            raise ValueError(
                "Extracted operating observation requires a value or range"
            )
        metric = self.driver_id.strip().casefold().replace("-", "_").replace(" ", "_")
        margin_metric = metric in {
            "gross_margin",
            "gross_profit_margin",
            "gross_margin_percent",
            "gross_margin_percentage",
            "gross_margin_rate",
            "gross_margin_pct",
        }
        gross_profit_metric = metric in {
            "gross_profit",
            "gross_profit_amount",
            "gross_income",
            "gross_income_amount",
        }
        if not margin_metric and not gross_profit_metric and any(
            value is not None and value < 0
            for value in (self.value, self.low, self.high)
        ):
            raise ValueError("Extracted operating observations cannot be negative")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("Extracted operating observation low cannot exceed high")
        if self.value is not None:
            if self.low is not None and self.value < self.low:
                raise ValueError(
                    "Extracted operating observation value cannot be below low"
                )
            if self.high is not None and self.value > self.high:
                raise ValueError(
                    "Extracted operating observation value cannot exceed high"
                )
        return self


class ExtractedOperatingInvestmentProgram(BaseModel):
    """Untrusted output for a first-party investment-program fact.

    A program may be announced or planned and may describe capacity or spend,
    but it is never a revenue forecast.
    """

    model_config = ConfigDict(extra="forbid")

    program_id: str
    name: str
    segment_id: str | None = None
    fiscal_year: int | None = Field(
        default=None, ge=_MIN_FISCAL_YEAR, le=_MAX_FISCAL_YEAR
    )
    fiscal_period: str = "FY"
    period_key: str | None = None
    value: float | None = None
    low: float | None = None
    high: float | None = None
    scale: float = 1
    unit: str = "unspecified"
    currency: str | None = None
    status: Literal[
        "announced",
        "planned",
        "in_progress",
        "under_construction",
        "completed",
        "reported",
        "unknown",
    ] = "reported"
    purpose: str | None = None
    supporting_text: str
    confidence: Literal["high", "medium", "low"] = "medium"

    @field_validator("fiscal_period", mode="before")
    @classmethod
    def normalize_fiscal_period(cls, value: str | None) -> str:
        return normalize_operating_fiscal_period(value)

    @field_validator("period_key")
    @classmethod
    def normalize_period_key(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Extracted investment period key")

    @model_validator(mode="before")
    @classmethod
    def canonicalize_period_input(cls, value: Any) -> Any:
        return _coerce_operating_period_fields(value)

    @field_validator(
        "program_id", "name", "segment_id", "unit", "purpose", "supporting_text"
    )
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Extracted investment-program text")

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not _CURRENCY_PATTERN.fullmatch(normalized):
            raise ValueError(
                "Extracted investment-program currency must be a three-letter ISO code"
            )
        return normalized

    @field_validator("value", "low", "high")
    @classmethod
    def validate_number(cls, value: float | None) -> float | None:
        if value is not None:
            _finite_decimal(Decimal(str(value)), "Extracted investment-program values")
        return value

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, value: float) -> float:
        if value <= 0 or not Decimal(str(value)).is_finite():
            raise ValueError("Extracted investment-program scale must be positive")
        return value

    @field_validator("value", "low", "high")
    @classmethod
    def reject_negative_values(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("Extracted investment-program values cannot be negative")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @model_validator(mode="after")
    def validate_range(self) -> "ExtractedOperatingInvestmentProgram":
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("Extracted investment-program low cannot exceed high")
        if self.value is not None:
            if self.low is not None and self.value < self.low:
                raise ValueError(
                    "Extracted investment-program value cannot be below low"
                )
            if self.high is not None and self.value > self.high:
                raise ValueError(
                    "Extracted investment-program value cannot exceed high"
                )
        return self


class ExtractedOperatingEvidenceResponse(BaseModel):
    """Structured OpenAI response containing evidence only.

    ``extra='forbid'`` is a hard boundary: a response containing a forecast,
    consensus estimate, or other unapproved section cannot validate.
    """

    # The Responses API returns the parsed JSON object itself.  Some OpenAI
    # model/provider combinations use the more explicit collection names
    # below, while the original local contract used the shorter names.  Keep
    # the JSON schema emitted to OpenAI on the canonical names, but accept the
    # compatible response aliases at the trust boundary.  ``extra=forbid`` is
    # intentionally retained so forecast/consensus collections cannot pass
    # through this compatibility layer.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    segments: list[ExtractedOperatingSegment] = Field(
        default_factory=list,
        validation_alias=AliasChoices("segments", "operating_segments"),
    )
    definitions: list[ExtractedOperatingDriverDefinition] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "definitions", "driver_definitions", "operating_definitions"
        ),
    )
    observations: list[ExtractedOperatingObservation] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "observations", "driver_observations", "operating_observations"
        ),
    )
    investment_programs: list[ExtractedOperatingInvestmentProgram] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "investment_programs", "investment_program_facts"
        ),
    )


class OperatingEvidenceRejection(BaseModel):
    """Audit record for an extracted item rejected by deterministic checks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_type: Literal[
        "segment", "definition", "observation", "investment_program", "response"
    ]
    reason: str
    item: (
        ExtractedOperatingSegment
        | ExtractedOperatingDriverDefinition
        | ExtractedOperatingObservation
        | ExtractedOperatingInvestmentProgram
        | None
    ) = None
    unsupported_evidence: bool = False
    missing_evidence: bool = False
    source: str | None = None
    confidence: Literal["high", "medium", "low"] | None = None

    @field_validator("reason", "source")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Operating evidence rejection text")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(getattr(value, "value", value)).strip().casefold()


class OperatingDocumentAudit(BaseModel):
    """Content-free diagnostics for one inspected SEC document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    filing_form: str
    filing_date: datetime.date
    accession_number: str
    filename: str
    document_type: str
    is_primary: bool = False
    cleaned_size: int = Field(default=0, ge=0)
    bounded_context_size: int = Field(default=0, ge=0)
    keyword_hits: dict[str, int] = Field(default_factory=dict)
    accepted_segments: int = Field(default=0, ge=0)
    accepted_definitions: int = Field(default=0, ge=0)
    accepted_observations: int = Field(default=0, ge=0)
    accepted_investment_programs: int = Field(default=0, ge=0)
    rejected_records: int = Field(default=0, ge=0)
    unsupported_evidence: int = Field(default=0, ge=0)
    missing_evidence: int = Field(default=0, ge=0)
    unusable_evidence: int = Field(default=0, ge=0)

    @field_validator("filing_form", "accession_number", "filename", "document_type")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return _normalize_required_text(value, "Operating document audit text")

    @field_validator("keyword_hits")
    @classmethod
    def validate_keyword_hits(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("Operating keyword hit counts cannot be negative")
        return dict(value)

    @property
    def segments(self) -> int:
        return self.accepted_segments

    @property
    def definitions(self) -> int:
        return self.accepted_definitions

    @property
    def observations(self) -> int:
        return self.accepted_observations

    @property
    def investment_programs(self) -> int:
        return self.accepted_investment_programs


class OperatingEvidenceAuditRecord(BaseModel):
    """Concise, content-free debug record for normalized operating evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_type: Literal["segment", "definition", "observation", "investment_program"]
    segment_id: str | None = None
    segment_name: str | None = None
    driver_id: str | None = None
    archetype: OperatingArchetype | None = None
    fiscal_year: int | None = Field(
        default=None, ge=_MIN_FISCAL_YEAR, le=_MAX_FISCAL_YEAR
    )
    fiscal_period: str | None = None
    period_key: str | None = None
    values: dict[str, Decimal] = Field(default_factory=dict)
    unit: str | None = None
    source: str
    confidence: Literal["high", "medium", "low"]
    status: Literal["accepted", "rejected", "unusable"] = "accepted"
    reason: str | None = None
    method: str | None = None

    @field_validator(
        "segment_id",
        "segment_name",
        "driver_id",
        "unit",
        "source",
        "reason",
        "method",
    )
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Operating evidence audit text")

    @field_validator("fiscal_period")
    @classmethod
    def normalize_period(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_operating_fiscal_period(value)

    @field_validator("period_key")
    @classmethod
    def normalize_period_key(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Operating evidence audit period key")

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: dict[str, Decimal]) -> dict[str, Decimal]:
        normalized: dict[str, Decimal] = {}
        for key, item in value.items():
            normalized[_normalize_required_text(key, "Operating audit value name")] = (
                _finite_decimal(item, "Operating audit values")
            )
        return normalized

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()


class OperatingEvidenceExtractionResult(BaseModel):
    """Provider-neutral normalized evidence returned by one extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segments: tuple[OperatingSegment, ...] = ()
    definitions: tuple[OperatingDriverDefinition, ...] = ()
    observations: tuple[OperatingDriverObservation, ...] = ()
    investment_programs: tuple[OperatingInvestmentProgram, ...] = ()
    audit_records: tuple[OperatingEvidenceAuditRecord, ...] = ()
    rejected: tuple[OperatingEvidenceRejection, ...] = ()
    unsupported_evidence: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    unusable_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @field_validator(
        "unsupported_evidence", "missing_evidence", "unusable_reasons", "warnings"
    )
    @classmethod
    def normalize_diagnostics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            _normalize_required_text(item, "Operating extraction diagnostic")
            for item in value
        )

    @property
    def extracted_records(self) -> int:
        return (
            len(self.segments)
            + len(self.definitions)
            + len(self.observations)
            + len(self.investment_programs)
        )

    @property
    def rejected_records(self) -> int:
        return len(self.rejected)

    @property
    def audits(self) -> tuple[OperatingEvidenceAuditRecord, ...]:
        """Concise alias used by audit-oriented callers."""

        return self.audit_records

    @property
    def unusable_evidence(self) -> tuple[str, ...]:
        return self.unusable_reasons


class OperatingExtractionCacheEntry(OperatingEvidenceExtractionResult):
    """Versioned, deterministic post-validation extraction artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    extracted_at: datetime.datetime
    model: str
    reasoning_effort: str
    prompt_version: str
    schema_version: str
    content_hash: str
    accession_number: str
    document_filename: str

    @field_validator(
        "model",
        "reasoning_effort",
        "prompt_version",
        "schema_version",
        "content_hash",
        "accession_number",
        "document_filename",
    )
    @classmethod
    def normalize_cache_text(cls, value: str) -> str:
        return _normalize_required_text(value, "Operating extraction cache text")


# Descriptive aliases keep the extraction boundary discoverable without
# creating parallel schema implementations.
OperatingEvidenceExtractionResponse = ExtractedOperatingEvidenceResponse
OperatingExtractionResult = OperatingEvidenceExtractionResult
OperatingEvidenceAudit = OperatingDocumentAudit
OperatingEvidenceResult = OperatingEvidenceExtractionResult
OperatingDriverExtractionResult = OperatingEvidenceExtractionResult
ExtractedOperatingResponse = ExtractedOperatingEvidenceResponse
ExtractedOperatingDriverObservation = ExtractedOperatingObservation
ExtractedInvestmentProgram = ExtractedOperatingInvestmentProgram
OperatingExtractionRejection = OperatingEvidenceRejection


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


class OperatingEconomicsMetricDiagnostics(BaseModel):
    """Independent coverage and validation diagnostics for one economics metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: Literal["gross_margin", "gross_profit"]
    coverage: Decimal | None = Field(default=None, allow_inf_nan=True)
    supported_years: tuple[int, ...] = ()
    confidence: Literal["high", "medium", "low"] = "low"
    reconstruction_error: Decimal | None = Field(default=None, allow_inf_nan=True)
    completeness: Decimal | None = Field(default=None, allow_inf_nan=True)
    warnings: tuple[str, ...] = ()
    identity_warnings: tuple[str, ...] = ()

    @field_validator("coverage", "reconstruction_error", "completeness")
    @classmethod
    def validate_metrics(cls, value: Decimal | None) -> Decimal | None:
        return _finite_decimal(value, "Operating economics diagnostics")

    @field_validator("supported_years")
    @classmethod
    def validate_years(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value:
            _validate_year_sequence(value, "Operating economics supported years")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @field_validator("warnings", "identity_warnings")
    @classmethod
    def normalize_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            _normalize_required_text(item, "Operating economics diagnostic warning")
            for item in value
        )

    @model_validator(mode="after")
    def validate_coverage(self) -> "OperatingEconomicsMetricDiagnostics":
        if self.reconstruction_error is not None and self.reconstruction_error < 0:
            raise ValueError(
                "Operating economics reconstruction error cannot be negative"
            )
        for value, label in (
            (self.coverage, "coverage"),
            (self.completeness, "completeness"),
        ):
            if value is not None and not Decimal(0) <= value <= Decimal(1):
                raise ValueError(f"Operating economics {label} must be between 0 and 1")
        return self


class OperatingEconomicsDiagnostics(BaseModel):
    """Metric-specific diagnostics retained independently from revenue quality."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gross_margin: OperatingEconomicsMetricDiagnostics = Field(
        default_factory=lambda: OperatingEconomicsMetricDiagnostics(metric="gross_margin")
    )
    gross_profit: OperatingEconomicsMetricDiagnostics = Field(
        default_factory=lambda: OperatingEconomicsMetricDiagnostics(metric="gross_profit")
    )
    completeness: Decimal | None = Field(default=None, allow_inf_nan=True)
    identity_warnings: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @field_validator("completeness")
    @classmethod
    def validate_completeness(cls, value: Decimal | None) -> Decimal | None:
        value = _finite_decimal(value, "Operating economics completeness")
        if value is not None and not Decimal(0) <= value <= Decimal(1):
            raise ValueError("Operating economics completeness must be between 0 and 1")
        return value

    @field_validator("identity_warnings", "warnings")
    @classmethod
    def normalize_diagnostic_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            _normalize_required_text(item, "Operating economics warning")
            for item in value
        )

    @property
    def margin(self) -> OperatingEconomicsMetricDiagnostics:
        """Short alias for the gross-margin diagnostics."""

        return self.gross_margin

    @property
    def profit(self) -> OperatingEconomicsMetricDiagnostics:
        """Short alias for the gross-profit diagnostics."""

        return self.gross_profit


class OperatingEconomicsYear(BaseModel):
    """Selected segment economics for one fiscal year."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fiscal_year: int = Field(ge=_MIN_FISCAL_YEAR, le=_MAX_FISCAL_YEAR)
    fiscal_period: str = "FY"
    period_key: str | None = None
    revenue: Decimal | None = None
    gross_margin: Decimal | None = None
    gross_profit: Decimal | None = None
    source: str = "unavailable"
    confidence: Literal["high", "medium", "low"] = "low"
    provenance: AssumptionProvenance | EvidenceReference | ForecastProvenance | str | None = None
    provenance_chain: tuple[AssumptionProvenance | EvidenceReference | ForecastProvenance | str, ...] = ()
    source_provenance: tuple[EvidenceReference, ...] = ()
    gross_margin_provenance: AssumptionProvenance | EvidenceReference | ForecastProvenance | str | None = None
    gross_margin_provenance_chain: tuple[AssumptionProvenance | EvidenceReference | ForecastProvenance | str, ...] = ()
    gross_margin_source_provenance: tuple[EvidenceReference, ...] = ()
    gross_profit_provenance: AssumptionProvenance | EvidenceReference | ForecastProvenance | str | None = None
    gross_profit_provenance_chain: tuple[AssumptionProvenance | EvidenceReference | ForecastProvenance | str, ...] = ()
    gross_profit_source_provenance: tuple[EvidenceReference, ...] = ()
    method: str = "unavailable"
    audit: tuple[str, ...] = ()
    expected_gross_profit: Decimal | None = None
    identity_error: Decimal | None = None

    @field_validator("fiscal_period")
    @classmethod
    def normalize_period(cls, value: str) -> str:
        return normalize_operating_fiscal_period(value)

    @field_validator("period_key")
    @classmethod
    def normalize_period_key(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Operating economics period key")

    @field_validator("revenue", "gross_margin", "gross_profit", "expected_gross_profit", "identity_error")
    @classmethod
    def validate_values(cls, value: Decimal | None) -> Decimal | None:
        return _finite_decimal(value, "Operating economics values")

    @field_validator("source", "method")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return _normalize_required_text(value, "Operating economics provenance")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @field_validator("audit")
    @classmethod
    def normalize_audit(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_normalize_required_text(item, "Operating economics audit") for item in value)

    @model_validator(mode="after")
    def validate_identity_error(self) -> "OperatingEconomicsYear":
        if self.identity_error is not None and self.identity_error < 0:
            raise ValueError("Operating economics identity error cannot be negative")
        return self


class SegmentOperatingEconomicsForecast(BaseModel):
    """Gross-margin and gross-profit path paired with one revenue segment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segment: OperatingSegment
    fiscal_years: tuple[int, ...]
    revenue: tuple[Decimal | None, ...]
    gross_margin: tuple[Decimal | None, ...]
    gross_profit: tuple[Decimal | None, ...]
    years: tuple[OperatingEconomicsYear, ...] = ()
    source_by_year: dict[int, str] = Field(default_factory=dict)
    confidence_by_year: dict[int, str] = Field(default_factory=dict)
    provenance_by_year: dict[int, AssumptionProvenance | EvidenceReference | ForecastProvenance | str] = Field(default_factory=dict)
    provenance_chain_by_year: dict[int, tuple[AssumptionProvenance | EvidenceReference | ForecastProvenance | str, ...]] = Field(default_factory=dict)
    source_provenance_by_year: dict[int, tuple[EvidenceReference, ...]] = Field(default_factory=dict)
    gross_margin_provenance_by_year: dict[int, AssumptionProvenance | EvidenceReference | ForecastProvenance | str] = Field(default_factory=dict)
    gross_margin_provenance_chain_by_year: dict[int, tuple[AssumptionProvenance | EvidenceReference | ForecastProvenance | str, ...]] = Field(default_factory=dict)
    gross_margin_source_provenance_by_year: dict[int, tuple[EvidenceReference, ...]] = Field(default_factory=dict)
    gross_profit_provenance_by_year: dict[int, AssumptionProvenance | EvidenceReference | ForecastProvenance | str] = Field(default_factory=dict)
    gross_profit_provenance_chain_by_year: dict[int, tuple[AssumptionProvenance | EvidenceReference | ForecastProvenance | str, ...]] = Field(default_factory=dict)
    gross_profit_source_provenance_by_year: dict[int, tuple[EvidenceReference, ...]] = Field(default_factory=dict)
    method_by_year: dict[int, str] = Field(default_factory=dict)
    audit_by_year: dict[int, tuple[str, ...]] = Field(default_factory=dict)
    diagnostics: OperatingEconomicsDiagnostics = Field(
        default_factory=OperatingEconomicsDiagnostics
    )
    warnings: tuple[str, ...] = ()
    unit: str = "currency"

    @field_validator("fiscal_years")
    @classmethod
    def validate_year_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        _validate_year_sequence(value, "Segment economics years")
        return value

    @field_validator("revenue")
    @classmethod
    def validate_revenue_values(cls, value: tuple[Decimal | None, ...]) -> tuple[Decimal | None, ...]:
        return tuple(
            None
            if item is None
            else _non_negative_decimal(item, "Segment economics revenue")
            for item in value
        )

    @field_validator("gross_margin", "gross_profit")
    @classmethod
    def validate_economics_values(cls, value: tuple[Decimal | None, ...]) -> tuple[Decimal | None, ...]:
        return tuple(_finite_decimal(item, "Segment economics values") for item in value)

    @field_validator("unit")
    @classmethod
    def normalize_unit(cls, value: str) -> str:
        return _normalize_required_text(value, "Segment economics unit")

    @field_validator("source_by_year", "confidence_by_year", "method_by_year", mode="before")
    @classmethod
    def normalize_year_maps(cls, value: dict[int, str] | None) -> dict[int, str]:
        return {int(year): str(getattr(item, "value", item)).strip() for year, item in (value or {}).items()}

    @field_validator("confidence_by_year")
    @classmethod
    def validate_confidence_map(cls, value: dict[int, str]) -> dict[int, str]:
        normalized = {year: confidence.casefold() for year, confidence in value.items()}
        if set(normalized.values()) - _CONFIDENCE_LEVELS:
            raise ValueError("Operating economics confidence must be high, medium, or low")
        return normalized

    @field_validator("warnings")
    @classmethod
    def normalize_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_normalize_required_text(item, "Segment economics warning") for item in value)

    @model_validator(mode="after")
    def validate_forecast(self) -> "SegmentOperatingEconomicsForecast":
        length = len(self.fiscal_years)
        if any(len(path) != length for path in (self.revenue, self.gross_margin, self.gross_profit)):
            raise ValueError("Segment economics paths must match fiscal_years")
        if self.years and tuple(item.fiscal_year for item in self.years) != self.fiscal_years:
            raise ValueError("Segment economics year records must match fiscal_years")
        for mapping, label in (
            (self.source_by_year, "source_by_year"),
            (self.confidence_by_year, "confidence_by_year"),
            (self.provenance_by_year, "provenance_by_year"),
            (self.provenance_chain_by_year, "provenance_chain_by_year"),
            (self.source_provenance_by_year, "source_provenance_by_year"),
            (self.gross_margin_provenance_by_year, "gross_margin_provenance_by_year"),
            (
                self.gross_margin_provenance_chain_by_year,
                "gross_margin_provenance_chain_by_year",
            ),
            (
                self.gross_margin_source_provenance_by_year,
                "gross_margin_source_provenance_by_year",
            ),
            (self.gross_profit_provenance_by_year, "gross_profit_provenance_by_year"),
            (
                self.gross_profit_provenance_chain_by_year,
                "gross_profit_provenance_chain_by_year",
            ),
            (
                self.gross_profit_source_provenance_by_year,
                "gross_profit_source_provenance_by_year",
            ),
            (self.method_by_year, "method_by_year"),
            (self.audit_by_year, "audit_by_year"),
        ):
            if not set(mapping).issubset(self.fiscal_years):
                raise ValueError(f"Segment economics {label} contains an unknown year")
        if not self.years:
            object.__setattr__(
                self,
                "years",
                tuple(
                    OperatingEconomicsYear(
                        fiscal_year=year,
                        revenue=revenue,
                        gross_margin=margin,
                        gross_profit=profit,
                        source=self.source_by_year.get(year, "unavailable"),
                        confidence=self.confidence_by_year.get(year, "low"),
                        provenance=self.provenance_by_year.get(year),
                        provenance_chain=self.provenance_chain_by_year.get(year, ()),
                        source_provenance=self.source_provenance_by_year.get(year, ()),
                        gross_margin_provenance=self.gross_margin_provenance_by_year.get(year),
                        gross_margin_provenance_chain=self.gross_margin_provenance_chain_by_year.get(year, ()),
                        gross_margin_source_provenance=self.gross_margin_source_provenance_by_year.get(year, ()),
                        gross_profit_provenance=self.gross_profit_provenance_by_year.get(year),
                        gross_profit_provenance_chain=self.gross_profit_provenance_chain_by_year.get(year, ()),
                        gross_profit_source_provenance=self.gross_profit_source_provenance_by_year.get(year, ()),
                        method=self.method_by_year.get(year, "unavailable"),
                        audit=self.audit_by_year.get(year, ()),
                    )
                    for year, revenue, margin, profit in zip(
                        self.fiscal_years, self.revenue, self.gross_margin, self.gross_profit, strict=True
                    )
                ),
            )
        return self

    @property
    def gross_margin_by_year(self) -> dict[int, Decimal | None]:
        return dict(zip(self.fiscal_years, self.gross_margin, strict=True))

    @property
    def gross_profit_by_year(self) -> dict[int, Decimal | None]:
        return dict(zip(self.fiscal_years, self.gross_profit, strict=True))

    @property
    def margin_provenance(self):
        return self.gross_margin_provenance_by_year

    @property
    def profit_provenance(self):
        return self.gross_profit_provenance_by_year

    @property
    def margin_source_provenance(self):
        return self.gross_margin_source_provenance_by_year

    @property
    def profit_source_provenance(self):
        return self.gross_profit_source_provenance_by_year

    @property
    def margin_diagnostics(self) -> OperatingEconomicsMetricDiagnostics:
        return self.diagnostics.gross_margin

    @property
    def profit_diagnostics(self) -> OperatingEconomicsMetricDiagnostics:
        return self.diagnostics.gross_profit

    @property
    def gross_margin_coverage(self) -> Decimal | None:
        return self.diagnostics.gross_margin.coverage

    @property
    def gross_profit_coverage(self) -> Decimal | None:
        return self.diagnostics.gross_profit.coverage

    @property
    def gross_margin_supported_years(self) -> tuple[int, ...]:
        return self.diagnostics.gross_margin.supported_years

    @property
    def gross_profit_supported_years(self) -> tuple[int, ...]:
        return self.diagnostics.gross_profit.supported_years


class OperatingEconomicsForecastConfig(BaseModel):
    """Small deterministic policy surface for gross-economics normalization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    historical_window: int = Field(default=3, ge=1, le=10)
    normalization_method: Literal["median", "weighted_recent"] = "median"
    gross_margin_min: Decimal = Decimal("-100")
    gross_margin_max: Decimal = Decimal("100")

    @field_validator("gross_margin_min", "gross_margin_max")
    @classmethod
    def validate_bounds(cls, value: Decimal) -> Decimal:
        return _finite_decimal(value, "Gross-margin bounds")

    @model_validator(mode="after")
    def validate_margin_bounds(self) -> "OperatingEconomicsForecastConfig":
        if self.gross_margin_min >= self.gross_margin_max:
            raise ValueError("Gross-margin minimum must be below maximum")
        if self.gross_margin_max > Decimal("100"):
            raise ValueError("Gross-margin maximum cannot exceed 100 percentage points")
        return self


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
    # Historical driver validation is deliberately an audit result rather
    # than another forecast input.  ``None`` means that no reported segment
    # revenue history was supplied and therefore no reconstruction could be
    # tested.  A zero coverage value means history was supplied but none of
    # its years had a complete driver reconstruction.
    driver_coverage: Decimal | None = Field(default=None, allow_inf_nan=True)
    modeled_revenue_share: Decimal | None = Field(default=None, allow_inf_nan=True)
    genuine_coverage: Decimal | None = Field(default=None, allow_inf_nan=True)
    reconstruction_error: Decimal | None = Field(default=None, allow_inf_nan=True)
    reconstruction_error_by_year: dict[int, Decimal] = Field(default_factory=dict)
    derived_reconstruction_years: tuple[int, ...] = ()
    supported_years: tuple[int, ...] = ()
    # Forward years for which this segment supplied its own usable operating
    # path.  ``supported_years`` is reserved for the historical reconstruction
    # audit above; keeping the two sets separate makes reconciliation audits
    # unambiguous.
    own_supported_years: tuple[int, ...] = ()
    confidence: Literal["high", "medium", "low"] = "low"
    # Optional sibling output.  It is omitted from serialization when absent so
    # existing revenue-only dumps remain byte/model-dump compatible.
    operating_economics: SegmentOperatingEconomicsForecast | None = None

    @model_serializer(mode="wrap")
    def serialize_revenue_forecast(self, handler):
        data = handler(self)
        if self.operating_economics is None:
            data.pop("operating_economics", None)
        return data

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

    @field_validator(
        "driver_coverage",
        "modeled_revenue_share",
        "genuine_coverage",
        "reconstruction_error",
    )
    @classmethod
    def validate_reconstruction_metric(cls, value: Decimal | None) -> Decimal | None:
        return _finite_decimal(value, "Segment reconstruction audit")

    @field_validator("reconstruction_error_by_year")
    @classmethod
    def validate_reconstruction_error_by_year(
        cls, value: dict[int, Decimal]
    ) -> dict[int, Decimal]:
        normalized: dict[int, Decimal] = {}
        for year, error in value.items():
            normalized_year = int(year)
            normalized_error = _non_negative_decimal(
                error, "Segment reconstruction errors"
            )
            normalized[normalized_year] = normalized_error
        return normalized

    @field_validator("supported_years")
    @classmethod
    def validate_supported_year_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value:
            _validate_year_sequence(value, "Segment supported years")
        return value

    @field_validator("derived_reconstruction_years")
    @classmethod
    def validate_derived_reconstruction_years(
        cls, value: tuple[int, ...]
    ) -> tuple[int, ...]:
        if value:
            _validate_year_sequence(value, "Segment derived reconstruction years")
        return value

    @field_validator("own_supported_years")
    @classmethod
    def validate_own_supported_year_values(
        cls, value: tuple[int, ...]
    ) -> tuple[int, ...]:
        if value:
            _validate_year_sequence(value, "Segment own-supported years")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_reconstruction_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

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
        _validate_subset(
            self.own_supported_years,
            self.fiscal_years,
            "own-supported years",
        )
        _validate_year_map(self.source_by_year, self.fiscal_years, "source_by_year")
        _validate_year_map(
            self.confidence_by_year, self.fiscal_years, "confidence_by_year"
        )
        if self.driver_coverage is not None and not (
            Decimal(0) <= self.driver_coverage <= Decimal(1)
        ):
            raise ValueError("Segment driver coverage must be between 0 and 1")
        for metric, label in (
            (self.modeled_revenue_share, "Segment modeled revenue share"),
            (self.genuine_coverage, "Segment genuine coverage"),
        ):
            if metric is not None and not Decimal(0) <= metric <= Decimal(1):
                raise ValueError(f"{label} must be between 0 and 1")
        if set(self.reconstruction_error_by_year) - set(self.supported_years):
            raise ValueError(
                "Segment reconstruction errors must be reported for supported years"
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

    @property
    def driver_confidence(self) -> str:
        """Compatibility alias for the reconstruction confidence field."""

        return self.confidence

    @property
    def coverage_ratio(self) -> Decimal | None:
        """Compatibility/readability alias for ``driver_coverage``."""

        return self.driver_coverage

    @property
    def own_supported(self) -> tuple[int, ...]:
        """Compatibility/readability alias for ``own_supported_years``."""

        return self.own_supported_years


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
    # See ``SegmentRevenueForecast`` for the distinction between unavailable
    # validation history (``None``) and zero validated coverage.
    driver_coverage: Decimal | None = Field(default=None, allow_inf_nan=True)
    modeled_revenue_share: Decimal | None = Field(default=None, allow_inf_nan=True)
    genuine_coverage: Decimal | None = Field(default=None, allow_inf_nan=True)
    reconstruction_error: Decimal | None = Field(default=None, allow_inf_nan=True)
    reconstruction_error_by_year: dict[int, Decimal] = Field(default_factory=dict)
    derived_reconstruction_years: tuple[int, ...] = ()
    supported_years: tuple[int, ...] = ()
    # ``supported_years`` describes historical driver reconstruction.  These
    # fields describe the forward reconciliation and are populated by the
    # reconciliation seam when consensus is available.
    own_supported_years: tuple[int, ...] = ()
    consensus_years: tuple[int, ...] = ()
    divergence_by_year: dict[int, Decimal] = Field(default_factory=dict)
    divergence: Decimal | None = Field(default=None, allow_inf_nan=True)
    confidence: Literal["high", "medium", "low"] = "low"
    selected_revenue_by_year: dict[int, Decimal] = Field(default_factory=dict)
    selected_source_by_year: dict[int, str] = Field(default_factory=dict)
    selected_confidence_by_year: dict[int, str] = Field(default_factory=dict)
    independent_revenue_by_year: dict[int, Decimal] = Field(default_factory=dict)
    consensus_revenue_by_year: dict[int, Decimal] = Field(default_factory=dict)
    management_revenue_by_year: dict[int, Decimal] = Field(default_factory=dict)
    # Optional sibling output.  See the serializer below for the compatibility
    # guarantee when no gross-economics evidence is supplied.
    operating_economics: "CompanyOperatingEconomicsForecast | None" = None

    @model_serializer(mode="wrap")
    def serialize_company_forecast(self, handler):
        data = handler(self)
        if self.operating_economics is None:
            data.pop("operating_economics", None)
        return data

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

    @field_validator(
        "driver_coverage",
        "modeled_revenue_share",
        "genuine_coverage",
        "reconstruction_error",
    )
    @classmethod
    def validate_reconstruction_metric(cls, value: Decimal | None) -> Decimal | None:
        return _finite_decimal(value, "Company reconstruction audit")

    @field_validator("reconstruction_error_by_year")
    @classmethod
    def validate_reconstruction_error_by_year(
        cls, value: dict[int, Decimal]
    ) -> dict[int, Decimal]:
        normalized: dict[int, Decimal] = {}
        for year, error in value.items():
            normalized_year = int(year)
            normalized_error = _non_negative_decimal(
                error, "Company reconstruction errors"
            )
            normalized[normalized_year] = normalized_error
        return normalized

    @field_validator("supported_years")
    @classmethod
    def validate_supported_year_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value:
            _validate_year_sequence(value, "Company supported years")
        return value

    @field_validator("derived_reconstruction_years")
    @classmethod
    def validate_derived_reconstruction_years(
        cls, value: tuple[int, ...]
    ) -> tuple[int, ...]:
        if value:
            _validate_year_sequence(value, "Company derived reconstruction years")
        return value

    @field_validator("own_supported_years", "consensus_years")
    @classmethod
    def validate_reconciliation_year_values(
        cls, value: tuple[int, ...]
    ) -> tuple[int, ...]:
        if value:
            _validate_year_sequence(value, "Company reconciliation years")
        return value

    @field_validator("divergence_by_year")
    @classmethod
    def validate_divergence_by_year(
        cls, value: dict[int, Decimal]
    ) -> dict[int, Decimal]:
        normalized: dict[int, Decimal] = {}
        for year, divergence in value.items():
            normalized[int(year)] = _finite_decimal(
                divergence, "Company revenue divergence"
            )
        return normalized

    @field_validator("divergence")
    @classmethod
    def validate_divergence(cls, value: Decimal | None) -> Decimal | None:
        return _finite_decimal(value, "Company revenue divergence")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_reconstruction_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @field_validator(
        "selected_revenue_by_year",
        "independent_revenue_by_year",
        "consensus_revenue_by_year",
        "management_revenue_by_year",
    )
    @classmethod
    def validate_revenue_audit_maps(
        cls, value: dict[int, Decimal]
    ) -> dict[int, Decimal]:
        return {
            int(year): _non_negative_decimal(amount, "Company revenue audit")
            for year, amount in value.items()
        }

    @field_validator("selected_source_by_year")
    @classmethod
    def normalize_selected_sources(cls, value: dict[int, str]) -> dict[int, str]:
        return {
            int(year): _normalize_required_text(source, "Company selected source")
            for year, source in value.items()
        }

    @field_validator("selected_confidence_by_year")
    @classmethod
    def normalize_selected_confidences(cls, value: dict[int, str]) -> dict[int, str]:
        normalized = {
            int(year): str(getattr(confidence, "value", confidence)).strip().casefold()
            for year, confidence in value.items()
        }
        if set(normalized.values()) - _CONFIDENCE_LEVELS:
            raise ValueError("Selected revenue confidence must be high, medium, or low")
        return normalized

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
        if self.driver_coverage is not None and not (
            Decimal(0) <= self.driver_coverage <= Decimal(1)
        ):
            raise ValueError("Company driver coverage must be between 0 and 1")
        for metric, label in (
            (self.modeled_revenue_share, "Company modeled revenue share"),
            (self.genuine_coverage, "Company genuine coverage"),
        ):
            if metric is not None and not Decimal(0) <= metric <= Decimal(1):
                raise ValueError(f"{label} must be between 0 and 1")
        _validate_subset(
            self.own_supported_years,
            self.fiscal_years,
            "own-supported years",
        )
        _validate_subset(
            self.consensus_years,
            self.fiscal_years,
            "consensus years",
        )
        _validate_year_map(
            self.divergence_by_year,
            self.fiscal_years,
            "divergence_by_year",
        )
        for values, label in (
            (self.selected_revenue_by_year, "selected_revenue_by_year"),
            (self.independent_revenue_by_year, "independent_revenue_by_year"),
            (self.consensus_revenue_by_year, "consensus_revenue_by_year"),
            (self.management_revenue_by_year, "management_revenue_by_year"),
        ):
            if not set(values).issubset(self.fiscal_years):
                raise ValueError(f"{label} contains a year outside fiscal_years")
        _validate_year_map(
            self.selected_source_by_year,
            self.fiscal_years,
            "selected_source_by_year",
        )
        _validate_year_map(
            self.selected_confidence_by_year,
            self.fiscal_years,
            "selected_confidence_by_year",
        )
        if set(self.reconstruction_error_by_year) - set(self.supported_years):
            raise ValueError(
                "Company reconstruction errors must be reported for supported years"
            )
        segment_ids = [item.segment.segment_id for item in self.segment_forecasts]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("Company segment forecasts must have unique segment IDs")
        for item in self.segment_forecasts:
            if item.fiscal_years != self.fiscal_years:
                raise ValueError("Segment and company forecast years must match")
            if (
                item.operating_economics is not None
                and item.operating_economics.fiscal_years != self.fiscal_years
            ):
                raise ValueError(
                    "Segment operating-economics and company forecast years must match"
                )
        if (
            self.operating_economics is not None
            and self.operating_economics.fiscal_years != self.fiscal_years
        ):
            raise ValueError(
                "Company operating-economics and forecast years must match"
            )
        if (
            self.transition_start_year is not None
            and self.explicit_years
            and self.transition_start_year <= max(self.explicit_years)
        ):
            raise ValueError(
                "Transition must start after the last explicit forecast year"
            )
        return self

    @property
    def driver_confidence(self) -> str:
        """Compatibility alias for the reconstruction confidence field."""

        return self.confidence

    @property
    def coverage_ratio(self) -> Decimal | None:
        """Compatibility/readability alias for ``driver_coverage``."""

        return self.driver_coverage

    @property
    def own_supported(self) -> tuple[int, ...]:
        """Compatibility/readability alias for ``own_supported_years``."""

        return self.own_supported_years

    @property
    def audit_diagnostics(self) -> dict[str, object]:
        """Expose reconciliation and reconstruction facts as one audit view."""

        return {
            "driver_coverage": self.driver_coverage,
            "modeled_revenue_share": self.modeled_revenue_share,
            "genuine_coverage": self.genuine_coverage,
            "reconstruction_error": self.reconstruction_error,
            "reconstruction_error_by_year": dict(self.reconstruction_error_by_year),
            "supported_years": self.supported_years,
            "own_supported_years": self.own_supported_years,
            "consensus_years": self.consensus_years,
            "divergence_by_year": dict(self.divergence_by_year),
            "divergence": self.divergence,
            "selected_revenue_by_year": dict(self.selected_revenue_by_year),
            "selected_source_by_year": dict(self.selected_source_by_year),
            "selected_confidence_by_year": dict(self.selected_confidence_by_year),
            "independent_revenue_by_year": dict(self.independent_revenue_by_year),
            "consensus_revenue_by_year": dict(self.consensus_revenue_by_year),
            "management_revenue_by_year": dict(self.management_revenue_by_year),
            "confidence": self.confidence,
            "derived_reconstruction_years": self.derived_reconstruction_years,
            "warnings": self.warnings,
        }

    @property
    def diagnostics(self) -> dict[str, object]:
        """Short alias for :attr:`audit_diagnostics`."""

        return self.audit_diagnostics

    def materialize_revenue_anchors(self, parameters):
        """Materialize this selected absolute revenue path into FCFF inputs."""

        return _materialize_company_revenue_anchors(parameters, self)


class CompanyOperatingEconomicsForecast(BaseModel):
    """Consolidated gross economics using the revenue consolidation set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    company_id: str
    fiscal_years: tuple[int, ...]
    segment_economics: tuple[SegmentOperatingEconomicsForecast, ...] = Field(
        default=(), validation_alias=AliasChoices(
            "segment_economics", "segments", "segment_forecasts"
        )
    )
    consolidated_revenue: tuple[Decimal | None, ...]
    consolidated_gross_profit: tuple[Decimal | None, ...] = Field(
        validation_alias=AliasChoices("consolidated_gross_profit", "gross_profit")
    )
    consolidated_gross_margin: tuple[Decimal | None, ...] = Field(
        validation_alias=AliasChoices("consolidated_gross_margin", "gross_margin")
    )
    years: tuple[OperatingEconomicsYear, ...] = ()
    source_by_year: dict[int, str] = Field(default_factory=dict)
    confidence_by_year: dict[int, str] = Field(default_factory=dict)
    provenance_by_year: dict[int, AssumptionProvenance | EvidenceReference | ForecastProvenance | str] = Field(default_factory=dict)
    provenance_chain_by_year: dict[int, tuple[AssumptionProvenance | EvidenceReference | ForecastProvenance | str, ...]] = Field(default_factory=dict)
    source_provenance_by_year: dict[int, tuple[EvidenceReference, ...]] = Field(default_factory=dict)
    gross_margin_provenance_by_year: dict[int, AssumptionProvenance | EvidenceReference | ForecastProvenance | str] = Field(default_factory=dict)
    gross_margin_provenance_chain_by_year: dict[int, tuple[AssumptionProvenance | EvidenceReference | ForecastProvenance | str, ...]] = Field(default_factory=dict)
    gross_margin_source_provenance_by_year: dict[int, tuple[EvidenceReference, ...]] = Field(default_factory=dict)
    gross_profit_provenance_by_year: dict[int, AssumptionProvenance | EvidenceReference | ForecastProvenance | str] = Field(default_factory=dict)
    gross_profit_provenance_chain_by_year: dict[int, tuple[AssumptionProvenance | EvidenceReference | ForecastProvenance | str, ...]] = Field(default_factory=dict)
    gross_profit_source_provenance_by_year: dict[int, tuple[EvidenceReference, ...]] = Field(default_factory=dict)
    method_by_year: dict[int, str] = Field(default_factory=dict)
    audit_by_year: dict[int, tuple[str, ...]] = Field(default_factory=dict)
    diagnostics: OperatingEconomicsDiagnostics = Field(
        default_factory=OperatingEconomicsDiagnostics
    )
    warnings: tuple[str, ...] = ()
    unit: str = "currency"

    @field_validator("company_id")
    @classmethod
    def normalize_company_id(cls, value: str) -> str:
        return _normalize_required_text(value, "Operating economics company identifier")

    @field_validator("fiscal_years")
    @classmethod
    def validate_year_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        _validate_year_sequence(value, "Company economics years")
        return value

    @field_validator("consolidated_revenue")
    @classmethod
    def validate_company_values(cls, value: tuple[Decimal | None, ...]) -> tuple[Decimal | None, ...]:
        return tuple(
            None
            if item is None
            else _non_negative_decimal(item, "Company economics values")
            for item in value
        )

    @field_validator("consolidated_gross_profit", "consolidated_gross_margin")
    @classmethod
    def validate_company_economics(cls, value: tuple[Decimal | None, ...]) -> tuple[Decimal | None, ...]:
        return tuple(_finite_decimal(item, "Company economics values") for item in value)

    @field_validator("unit")
    @classmethod
    def normalize_unit(cls, value: str) -> str:
        return _normalize_required_text(value, "Company economics unit")

    @field_validator("source_by_year", "confidence_by_year", "method_by_year", mode="before")
    @classmethod
    def normalize_year_maps(cls, value: dict[int, str] | None) -> dict[int, str]:
        return {int(year): str(getattr(item, "value", item)).strip() for year, item in (value or {}).items()}

    @field_validator("confidence_by_year")
    @classmethod
    def validate_confidence_map(cls, value: dict[int, str]) -> dict[int, str]:
        normalized = {year: confidence.casefold() for year, confidence in value.items()}
        if set(normalized.values()) - _CONFIDENCE_LEVELS:
            raise ValueError("Operating economics confidence must be high, medium, or low")
        return normalized

    @field_validator("warnings")
    @classmethod
    def normalize_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_normalize_required_text(item, "Company economics warning") for item in value)

    @model_validator(mode="after")
    def validate_forecast(self) -> "CompanyOperatingEconomicsForecast":
        length = len(self.fiscal_years)
        if any(
            len(path) != length
            for path in (
                self.consolidated_revenue,
                self.consolidated_gross_profit,
                self.consolidated_gross_margin,
            )
        ):
            raise ValueError("Company economics paths must match fiscal_years")
        if self.years and tuple(item.fiscal_year for item in self.years) != self.fiscal_years:
            raise ValueError("Company economics year records must match fiscal_years")
        for segment in self.segment_economics:
            if segment.fiscal_years != self.fiscal_years:
                raise ValueError("Segment and company economics years must match")
        for mapping, label in (
            (self.source_by_year, "source_by_year"),
            (self.confidence_by_year, "confidence_by_year"),
            (self.provenance_by_year, "provenance_by_year"),
            (self.provenance_chain_by_year, "provenance_chain_by_year"),
            (self.source_provenance_by_year, "source_provenance_by_year"),
            (self.gross_margin_provenance_by_year, "gross_margin_provenance_by_year"),
            (
                self.gross_margin_provenance_chain_by_year,
                "gross_margin_provenance_chain_by_year",
            ),
            (
                self.gross_margin_source_provenance_by_year,
                "gross_margin_source_provenance_by_year",
            ),
            (self.gross_profit_provenance_by_year, "gross_profit_provenance_by_year"),
            (
                self.gross_profit_provenance_chain_by_year,
                "gross_profit_provenance_chain_by_year",
            ),
            (
                self.gross_profit_source_provenance_by_year,
                "gross_profit_source_provenance_by_year",
            ),
            (self.method_by_year, "method_by_year"),
            (self.audit_by_year, "audit_by_year"),
        ):
            if not set(mapping).issubset(self.fiscal_years):
                raise ValueError(f"Company economics {label} contains an unknown year")
        if not self.years:
            object.__setattr__(
                self,
                "years",
                tuple(
                    OperatingEconomicsYear(
                        fiscal_year=year,
                        revenue=revenue,
                        gross_profit=profit,
                        gross_margin=margin,
                        source=self.source_by_year.get(year, "unavailable"),
                        confidence=self.confidence_by_year.get(year, "low"),
                        provenance=self.provenance_by_year.get(year),
                        provenance_chain=self.provenance_chain_by_year.get(year, ()),
                        source_provenance=self.source_provenance_by_year.get(year, ()),
                        gross_margin_provenance=self.gross_margin_provenance_by_year.get(year),
                        gross_margin_provenance_chain=self.gross_margin_provenance_chain_by_year.get(year, ()),
                        gross_margin_source_provenance=self.gross_margin_source_provenance_by_year.get(year, ()),
                        gross_profit_provenance=self.gross_profit_provenance_by_year.get(year),
                        gross_profit_provenance_chain=self.gross_profit_provenance_chain_by_year.get(year, ()),
                        gross_profit_source_provenance=self.gross_profit_source_provenance_by_year.get(year, ()),
                        method=self.method_by_year.get(year, "unavailable"),
                        audit=self.audit_by_year.get(year, ()),
                    )
                    for year, revenue, profit, margin in zip(
                        self.fiscal_years,
                        self.consolidated_revenue,
                        self.consolidated_gross_profit,
                        self.consolidated_gross_margin,
                        strict=True,
                    )
                ),
            )
        return self

    @property
    def segments(self) -> tuple[SegmentOperatingEconomicsForecast, ...]:
        """Compatibility alias for the selected segment economics."""

        return self.segment_economics

    @property
    def gross_profit(self) -> tuple[Decimal | None, ...]:
        return self.consolidated_gross_profit

    @property
    def gross_margin(self) -> tuple[Decimal | None, ...]:
        return self.consolidated_gross_margin

    @property
    def gross_profit_by_year(self) -> dict[int, Decimal | None]:
        return dict(zip(self.fiscal_years, self.consolidated_gross_profit, strict=True))

    @property
    def gross_margin_by_year(self) -> dict[int, Decimal | None]:
        return dict(zip(self.fiscal_years, self.consolidated_gross_margin, strict=True))

    @property
    def margin_provenance(self):
        return self.gross_margin_provenance_by_year

    @property
    def profit_provenance(self):
        return self.gross_profit_provenance_by_year

    @property
    def margin_source_provenance(self):
        return self.gross_margin_source_provenance_by_year

    @property
    def profit_source_provenance(self):
        return self.gross_profit_source_provenance_by_year

    @property
    def margin_diagnostics(self) -> OperatingEconomicsMetricDiagnostics:
        return self.diagnostics.gross_margin

    @property
    def profit_diagnostics(self) -> OperatingEconomicsMetricDiagnostics:
        return self.diagnostics.gross_profit

    @property
    def gross_margin_coverage(self) -> Decimal | None:
        return self.diagnostics.gross_margin.coverage

    @property
    def gross_profit_coverage(self) -> Decimal | None:
        return self.diagnostics.gross_profit.coverage

    @property
    def gross_margin_supported_years(self) -> tuple[int, ...]:
        return self.diagnostics.gross_margin.supported_years

    @property
    def gross_profit_supported_years(self) -> tuple[int, ...]:
        return self.diagnostics.gross_profit.supported_years


# The names below make the additive contract discoverable for both the
# "forecast" and "economics" terminology used by existing clients.
OperatingEconomicsForecast = CompanyOperatingEconomicsForecast
OperatingEconomicsForecastResult = CompanyOperatingEconomicsForecast
GrossEconomicsForecast = CompanyOperatingEconomicsForecast
SegmentOperatingEconomics = SegmentOperatingEconomicsForecast
SegmentGrossEconomicsForecast = SegmentOperatingEconomicsForecast
CompanyOperatingEconomics = CompanyOperatingEconomicsForecast
CompanyGrossEconomicsForecast = CompanyOperatingEconomicsForecast
OperatingMetricDiagnostics = OperatingEconomicsMetricDiagnostics
GrossMarginDiagnostics = OperatingEconomicsMetricDiagnostics
GrossProfitDiagnostics = OperatingEconomicsMetricDiagnostics


# ``CompanyOperatingForecast`` is declared before its optional sibling contract
# to keep the revenue schema in its historical location.
CompanyOperatingForecast.model_rebuild()


def _materialize_company_revenue_anchors(parameters, selected: CompanyOperatingForecast):
    """Materialize a company forecast without importing service code.

    The broader reconciliation service handles generic mappings and detailed
    reconciliation wrappers.  This schema-level path is the narrow operation
    needed by :meth:`CompanyOperatingForecast.materialize_revenue_anchors`.
    """

    if parameters.revenue_growth is not None:
        return parameters

    anchors = dict(parameters.revenue_anchors)
    sources = dict(parameters.revenue_anchor_sources)
    for year, value in zip(
        selected.fiscal_years, selected.consolidated_revenue, strict=True
    ):
        source = selected.source_by_year.get(
            year,
            "explicit" if year in selected.explicit_years else "independent_operating",
        )
        normalized_source = str(getattr(source, "value", source)).strip().casefold()
        if normalized_source == "unavailable":
            continue
        if value <= 0:
            raise ValueError(f"Selected revenue FY{year} must be positive for FCFF anchors")

        if normalized_source == "explicit":
            incoming_source = ForecastAssumptionSource.EXPLICIT
            incoming_rank = 4
        elif normalized_source == "management_guidance":
            incoming_source = ForecastAssumptionSource.MANAGEMENT_GUIDANCE
            incoming_rank = 3
        elif normalized_source == "normalized_historical":
            incoming_source = ForecastAssumptionSource.NORMALIZED_HISTORICAL
            incoming_rank = 0
        elif normalized_source == "current_run_rate":
            incoming_source = ForecastAssumptionSource.CURRENT_RUN_RATE
            incoming_rank = -1
        else:
            incoming_source = ForecastAssumptionSource.FORWARD_EVIDENCE
            incoming_rank = 2 if "independent" in normalized_source else 1

        existing_source = sources.get(year, ForecastAssumptionSource.EXPLICIT)
        existing_normalized = str(
            getattr(existing_source, "value", existing_source)
        ).strip().casefold()
        existing_rank = {
            "explicit": 4,
            "management_guidance": 3,
            "forward_evidence": 1,
            "normalized_historical": 0,
            "current_run_rate": -1,
        }.get(existing_normalized, 0)
        if year in anchors and existing_rank >= incoming_rank:
            continue
        anchors[year] = value
        sources[year] = incoming_source

    if anchors == parameters.revenue_anchors and sources == parameters.revenue_anchor_sources:
        return parameters
    return parameters.model_copy(
        update={
            "revenue_anchors": anchors,
            "revenue_anchor_sources": sources,
        }
    )


__all__ = [
    "CompanyOperatingEconomics",
    "CompanyOperatingEconomicsForecast",
    "CompanyGrossEconomicsForecast",
    "CompanyOperatingForecast",
    "EvidenceReference",
    "ExtractedOperatingDriverDefinition",
    "ExtractedOperatingDriverObservation",
    "ExtractedOperatingEvidenceResponse",
    "ExtractedOperatingResponse",
    "ExtractedInvestmentProgram",
    "ExtractedOperatingInvestmentProgram",
    "ExtractedOperatingObservation",
    "ExtractedOperatingSegment",
    "OperatingArchetype",
    "OperatingDriverDefinition",
    "OperatingDriverForecast",
    "OperatingDriverObservation",
    "OperatingEconomicsDiagnostics",
    "OperatingEconomicsForecast",
    "OperatingEconomicsForecastConfig",
    "OperatingEconomicsForecastResult",
    "OperatingEconomicsMetricDiagnostics",
    "OperatingEconomicsYear",
    "OperatingMetricDiagnostics",
    "GrossMarginDiagnostics",
    "GrossProfitDiagnostics",
    "GrossEconomicsForecast",
    "OperatingEvidenceGap",
    "OperatingEvidenceAudit",
    "OperatingEvidenceExtractionResponse",
    "OperatingEvidenceExtractionResult",
    "OperatingEvidenceResult",
    "OperatingDriverExtractionResult",
    "OperatingEvidenceRejection",
    "OperatingExtractionRejection",
    "OperatingInvestmentProgram",
    "OperatingDocumentAudit",
    "OperatingEvidenceAuditRecord",
    "OperatingExtractionCacheEntry",
    "OperatingExtractionResult",
    "OperatingSegment",
    "ResolvedRevenueYear",
    "SegmentOperatingEconomics",
    "SegmentOperatingEconomicsForecast",
    "SegmentGrossEconomicsForecast",
    "SegmentRevenueForecast",
    "canonical_operating_segment_id",
    "canonical_operating_segment_identity",
    "normalize_operating_fiscal_period",
    "normalize_operating_unit",
    "operating_units_compatible",
    "operating_periods_compatible",
]
