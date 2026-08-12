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
    model_validator,
)

from edgarito.schemas.valuation.assumptions import AssumptionProvenance

_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")
_MIN_FISCAL_YEAR = 1900
_MAX_FISCAL_YEAR = 2200
_CONFIDENCE_LEVELS = {"high", "medium", "low"}

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

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return _normalize_required_text(value, "Segment source")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()


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

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return _normalize_required_text(value, "Driver source")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

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
    origin: Literal[
        "reported",
        "first_party_observation",
        "management_guidance",
        "derived",
        "extracted_evidence",
    ]
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

    @field_validator(
        "program_id", "name", "segment_id", "fiscal_period", "unit", "purpose"
    )
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        normalized = _normalize_optional_text(value, "Investment program text")
        return normalized or ("FY" if value == "" else normalized)

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
        if not set(required).issubset(input_metrics):
            raise ValueError(
                "Extracted required inputs must be listed in input_metrics"
            )
        if not set(self.optional_inputs).issubset(input_metrics):
            raise ValueError(
                "Extracted optional inputs must be listed in input_metrics"
            )
        if set(required) & set(self.optional_inputs):
            raise ValueError("Extracted required and optional inputs must be disjoint")
        if self.units and not set(self.input_metrics).issubset(self.units):
            raise ValueError(
                "Extracted operating definition requires units for all inputs"
            )
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
    value: float | None = None
    low: float | None = None
    high: float | None = None
    unit: str
    currency: str | None = None
    basis: str | None = None
    origin: Literal["reported", "first_party_observation", "management_guidance"] = (
        "reported"
    )
    supporting_text: str
    confidence: Literal["high", "medium", "low"] = "medium"

    @field_validator(
        "segment_id", "driver_id", "fiscal_period", "unit", "basis", "supporting_text"
    )
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Extracted operating observation text")

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

    @field_validator("value", "low", "high")
    @classmethod
    def reject_negative_values(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("Extracted operating observations cannot be negative")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return str(getattr(value, "value", value)).strip().casefold()

    @model_validator(mode="after")
    def require_value(self) -> "ExtractedOperatingObservation":
        if self.value is None and self.low is None and self.high is None:
            raise ValueError(
                "Extracted operating observation requires a value or range"
            )
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
    value: float | None = None
    low: float | None = None
    high: float | None = None
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
        return "FY" if value is None or not str(value).strip() else str(value).strip()

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
    values: dict[str, Decimal] = Field(default_factory=dict)
    unit: str | None = None
    source: str
    confidence: Literal["high", "medium", "low"]

    @field_validator(
        "segment_id", "segment_name", "driver_id", "fiscal_period", "unit", "source"
    )
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "Operating evidence audit text")

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
    warnings: tuple[str, ...] = ()

    @field_validator("unsupported_evidence", "missing_evidence", "warnings")
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
    reconstruction_error: Decimal | None = Field(default=None, allow_inf_nan=True)
    reconstruction_error_by_year: dict[int, Decimal] = Field(default_factory=dict)
    supported_years: tuple[int, ...] = ()
    # Forward years for which this segment supplied its own usable operating
    # path.  ``supported_years`` is reserved for the historical reconstruction
    # audit above; keeping the two sets separate makes reconciliation audits
    # unambiguous.
    own_supported_years: tuple[int, ...] = ()
    confidence: Literal["high", "medium", "low"] = "low"

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

    @field_validator("driver_coverage", "reconstruction_error")
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
    reconstruction_error: Decimal | None = Field(default=None, allow_inf_nan=True)
    reconstruction_error_by_year: dict[int, Decimal] = Field(default_factory=dict)
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

    @field_validator("driver_coverage", "reconstruction_error")
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
            "warnings": self.warnings,
        }

    @property
    def diagnostics(self) -> dict[str, object]:
        """Short alias for :attr:`audit_diagnostics`."""

        return self.audit_diagnostics

    def materialize_revenue_anchors(self, parameters):
        """Materialize this selected absolute revenue path into FCFF inputs."""

        from edgarito.services.operating.reconciliation import (
            materialize_revenue_anchors,
        )

        return materialize_revenue_anchors(parameters, self)


def _normalize_required_text(value: str, label: str) -> str:
    normalized = str(getattr(value, "value", value)).strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized


def _coerce_extracted_text_map(value: Any, label: str) -> dict[str, str]:
    """Normalize API-safe key/value pairs and legacy mapping fixtures.

    The model-facing schema uses a list of objects because arbitrary JSON
    object properties are not accepted by OpenAI Structured Outputs.  The
    domain model still exposes the more useful mapping shape to deterministic
    consumers, and legacy cached/fixture responses remain readable.
    """

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {
            _normalize_required_text(
                key, f"Extracted {label} key"
            ): _normalize_required_text(item, f"Extracted {label} value")
            for key, item in value.items()
        }
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Extracted {label} map must be an object or key/value list")
    result: dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"Extracted {label} entries must be objects")
        if set(item) != {"key", "value"}:
            raise ValueError(
                f"Extracted {label} entries must contain only key and value"
            )
        key = _normalize_required_text(item["key"], f"Extracted {label} key")
        if key in result:
            raise ValueError(f"Extracted {label} keys must be unique")
        result[key] = _normalize_required_text(
            item["value"], f"Extracted {label} value"
        )
    return result


def _coerce_operating_archetype(value: OperatingArchetype | str) -> OperatingArchetype:
    """Normalize the small vocabulary accepted at the extraction boundary."""

    if isinstance(value, OperatingArchetype):
        return value
    normalized = (
        str(getattr(value, "value", value))
        .strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("×", "_times_")
    )
    aliases = {
        "volume_times_price": OperatingArchetype.VOLUME_PRICE,
        "volume_and_price": OperatingArchetype.VOLUME_PRICE,
        "subscriber_arpu": OperatingArchetype.SUBSCRIBERS_ARPU,
        "subscribers_and_arpu": OperatingArchetype.SUBSCRIBERS_ARPU,
        "capacity_utilization_and_price": OperatingArchetype.CAPACITY_UTILIZATION_PRICE,
        "capacity_times_utilization_times_price": OperatingArchetype.CAPACITY_UTILIZATION_PRICE,
        "transactions_and_take_rate": OperatingArchetype.TRANSACTIONS_TAKE_RATE,
        "backlog_and_conversion": OperatingArchetype.BACKLOG_CONVERSION,
        "store_count_and_sales_per_store": OperatingArchetype.STORE_COUNT_SALES_PER_STORE,
        "segment_growth": OperatingArchetype.GENERIC_SEGMENT_GROWTH,
    }
    aliased = aliases.get(normalized)
    if aliased is not None:
        return aliased
    try:
        return OperatingArchetype(normalized)
    except ValueError as error:
        raise ValueError(f"Unsupported operating archetype: {value}") from error


def _archetype_metrics(archetype: OperatingArchetype) -> tuple[str, ...]:
    """Return canonical input names for an extracted archetype mapping."""

    return {
        OperatingArchetype.VOLUME_PRICE: ("volume", "price"),
        OperatingArchetype.SUBSCRIBERS_ARPU: ("subscribers", "arpu"),
        OperatingArchetype.CAPACITY_UTILIZATION_PRICE: (
            "capacity",
            "utilization",
            "price",
        ),
        OperatingArchetype.TRANSACTIONS_TAKE_RATE: ("transactions", "take_rate"),
        OperatingArchetype.BACKLOG_CONVERSION: ("backlog", "conversion_rate"),
        OperatingArchetype.STORE_COUNT_SALES_PER_STORE: (
            "store_count",
            "sales_per_store",
        ),
        OperatingArchetype.GENERIC_SEGMENT_GROWTH: ("growth",),
    }[archetype]


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
    "SegmentRevenueForecast",
]
