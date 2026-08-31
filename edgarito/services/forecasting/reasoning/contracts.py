"""Strict, provider-neutral contracts for ForecastReasoner v1.

The response models in this module are deliberately not executable forecast
models.  They describe a proposal which must pass the deterministic validator
and compiler before it can reach the existing driver-based executor.
"""

from __future__ import annotations

import datetime
import re
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal, Mapping

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from edgarito.schemas.forecasting import (
    ForecastOverride,
    ForecastPlan,
    ForecastScope,
)
from edgarito.schemas.guidance.management import ManagementGuidance
from edgarito.schemas.normalization.financials import NormalizedCompanyFinancials
from edgarito.schemas.operating import (
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingInvestmentProgram,
    OperatingSegment,
)
from edgarito.services.research.consensus import EvidenceConsensus
from edgarito.services.research.contracts import ResearchEvidence

# Pydantic's normal Decimal schema is a string-pattern schema.  OpenAI
# Structured Outputs accepts numeric JSON values, so override only the emitted
# schema while retaining Decimal arithmetic locally.
ForecastDecimal = Annotated[Decimal, WithJsonSchema({"type": "number"})]

_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9_.:-]*$")


class ForecastReasoningValueBasis(str, Enum):
    ABSOLUTE = "absolute"
    PERCENT_OF_REVENUE = "percent_of_revenue"
    PERCENTAGE_POINTS = "percentage_points"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
            aliases = {
                "amount": cls.ABSOLUTE,
                "currency": cls.ABSOLUTE,
                "pp": cls.PERCENTAGE_POINTS,
                "percentage_point": cls.PERCENTAGE_POINTS,
            }
            return aliases.get(normalized)
        return None


# This short name is convenient for callers importing the focused package.
ForecastValueBasis = ForecastReasoningValueBasis


def _text(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized


def _tuple_records(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if hasattr(value, "applications"):
        return tuple(item.guidance for item in value.applications)
    if hasattr(value, "records"):
        return tuple(value.records)
    if hasattr(value, "eligible_records"):
        return tuple(value.eligible_records)
    if isinstance(value, (str, bytes, Mapping, BaseModel)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:  # Decimal raises several implementation exceptions.
        raise ValueError(f"{label} must be numeric") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def _normalise_model_records(value: Any, model: type[BaseModel]) -> tuple[Any, ...]:
    return tuple(
        item if isinstance(item, model) else model.model_validate(item)
        for item in _tuple_records(value)
    )


def _normalise_override_records(value: Any) -> tuple[ForecastOverride, ...]:
    if isinstance(value, Mapping) and not {
        "scope",
        "metric",
        "strategy",
    }.issubset(value):
        records: list[Any] = []
        for key, item in value.items():
            payload = dict(item) if isinstance(item, Mapping) else {}
            if isinstance(key, tuple):
                if len(key) == 2:
                    payload.setdefault("scope", key[0])
                    payload.setdefault("metric", key[1])
                elif len(key) == 3:
                    payload.setdefault("scope", key[0])
                    payload.setdefault("scope_id", key[1])
                    payload.setdefault("metric", key[2])
            elif isinstance(key, str) and ":" in key:
                parts = key.split(":")
                if len(parts) == 2:
                    payload.setdefault("scope", parts[0])
                    payload.setdefault("metric", parts[1])
                elif len(parts) == 3:
                    payload.setdefault("scope", parts[0])
                    payload.setdefault("scope_id", parts[1])
                    payload.setdefault("metric", parts[2])
            records.append(payload)
        value = records
    return _normalise_model_records(value, ForecastOverride)


class HistoricalFactSummary(BaseModel):
    """Small normalized historical fact; never a filing or source-text blob."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fiscal_year: int
    metric: str
    value: ForecastDecimal
    unit: str
    fiscal_period: str = "FY"
    scope: ForecastScope = ForecastScope.COMPANY
    scope_id: str = "company"
    currency: str | None = None
    method: str = "normalized_historical"
    source: str | None = None
    source_id: str | None = None
    source_date: datetime.date | None = None

    @field_validator(
        "metric",
        "unit",
        "fiscal_period",
        "scope_id",
        "method",
        "source",
        "source_id",
    )
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return _text(value, "Historical fact text")

    @field_validator("value", mode="before")
    @classmethod
    def normalize_value(cls, value: Any) -> Decimal:
        return _decimal(value, "Historical fact value")

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        return value.strip().upper() if value else None

    @model_validator(mode="after")
    def validate_scope(self) -> "HistoricalFactSummary":
        if self.scope == ForecastScope.COMPANY and self.scope_id != "company":
            raise ValueError("Company historical facts require scope_id='company'")
        if self.scope == ForecastScope.SEGMENT and self.scope_id == "company":
            raise ValueError("Segment historical facts require a segment scope_id")
        return self


class ForecastReasoningInput(BaseModel):
    """Immutable, compact context supplied to ForecastReasoner."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    company_id: str
    company_name: str | None = None
    ticker: str | None = None
    unit: str
    as_of: datetime.date
    forecast_years: tuple[int, ...] = Field(
        validation_alias=AliasChoices("forecast_years", "years", "fiscal_years")
    )
    segments: tuple[OperatingSegment, ...] = ()
    definitions: tuple[OperatingDriverDefinition, ...] = Field(
        default=(), validation_alias=AliasChoices("definitions", "accepted_definitions")
    )
    observations: tuple[OperatingDriverObservation, ...] = Field(
        default=(),
        validation_alias=AliasChoices("observations", "accepted_observations"),
    )
    management_guidance: tuple[ManagementGuidance, ...] = Field(
        default=(), validation_alias=AliasChoices("management_guidance", "guidance")
    )
    management_constraints: tuple[Any, ...] = Field(
        default=(),
        validation_alias=AliasChoices("management_constraints", "constraints"),
    )
    investment_programs: tuple[OperatingInvestmentProgram, ...] = Field(
        default=(), validation_alias=AliasChoices("investment_programs", "programs")
    )
    historical_facts: tuple[HistoricalFactSummary, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "historical_facts", "normalized_historical_facts", "history"
        ),
    )
    research_evidence: tuple[ResearchEvidence, ...] = Field(
        default=(), validation_alias=AliasChoices("research_evidence", "research")
    )
    evidence_consensus: tuple[EvidenceConsensus, ...] = Field(
        default=(), validation_alias=AliasChoices("evidence_consensus", "consensus")
    )
    manual_overrides: tuple[ForecastOverride, ...] = Field(
        default=(), validation_alias=AliasChoices("manual_overrides", "overrides")
    )
    manual_forward_driver_observations: tuple[OperatingDriverObservation, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "manual_forward_driver_observations",
            "forward_driver_observations",
            "manual_driver_observations",
            "manual_forward_observations",
        ),
    )

    @field_validator("company_id", "company_name", "ticker", "unit")
    @classmethod
    def normalize_identity_text(cls, value: str | None) -> str | None:
        return (
            _text(value, "Forecast reasoning identity") if value is not None else None
        )

    @field_validator("forecast_years", mode="before")
    @classmethod
    def normalize_years(cls, value: Any) -> tuple[int, ...]:
        if isinstance(value, int):
            value = (value,)
        years = tuple(int(item) for item in (value or ()))
        if not years:
            raise ValueError("Forecast reasoning requires an exact forecast horizon")
        if tuple(sorted(years)) != years or len(years) != len(set(years)):
            raise ValueError("Forecast reasoning years must be sorted and unique")
        return years

    @field_validator("segments", mode="before")
    @classmethod
    def normalize_segments(cls, value: Any) -> tuple[OperatingSegment, ...]:
        value = getattr(value, "segments", value)
        return _normalise_model_records(value, OperatingSegment)

    @field_validator("definitions", mode="before")
    @classmethod
    def normalize_definitions(cls, value: Any) -> tuple[OperatingDriverDefinition, ...]:
        value = getattr(value, "definitions", value)
        return _normalise_model_records(value, OperatingDriverDefinition)

    @field_validator(
        "observations", "manual_forward_driver_observations", mode="before"
    )
    @classmethod
    def normalize_observations(
        cls, value: Any
    ) -> tuple[OperatingDriverObservation, ...]:
        value = getattr(
            value, "observations", getattr(value, "eligible_records", value)
        )
        return _normalise_model_records(value, OperatingDriverObservation)

    @field_validator("management_guidance", mode="before")
    @classmethod
    def normalize_guidance(cls, value: Any) -> tuple[ManagementGuidance, ...]:
        return _normalise_model_records(value, ManagementGuidance)

    @field_validator("investment_programs", mode="before")
    @classmethod
    def normalize_programs(cls, value: Any) -> tuple[OperatingInvestmentProgram, ...]:
        value = getattr(value, "investment_programs", getattr(value, "programs", value))
        return _normalise_model_records(value, OperatingInvestmentProgram)

    @field_validator("historical_facts", mode="before")
    @classmethod
    def normalize_history(cls, value: Any) -> tuple[HistoricalFactSummary, ...]:
        return _normalise_model_records(value, HistoricalFactSummary)

    @field_validator("research_evidence", mode="before")
    @classmethod
    def normalize_research(cls, value: Any) -> tuple[ResearchEvidence, ...]:
        from pydantic import TypeAdapter

        from edgarito.services.research.contracts import EvidenceItem

        values = _tuple_records(value)
        adapter = TypeAdapter(EvidenceItem)
        return tuple(
            item
            if isinstance(item, ResearchEvidence)
            else adapter.validate_python(item)
            for item in values
        )

    @field_validator("evidence_consensus", mode="before")
    @classmethod
    def normalize_consensus(cls, value: Any) -> tuple[EvidenceConsensus, ...]:
        return _normalise_model_records(value, EvidenceConsensus)

    @field_validator("management_constraints", mode="before")
    @classmethod
    def normalize_constraints(cls, value: Any) -> tuple[Any, ...]:
        value = getattr(value, "management_constraints", value)
        return _tuple_records(value)

    @field_validator("manual_overrides", mode="before")
    @classmethod
    def normalize_overrides(cls, value: Any) -> tuple[ForecastOverride, ...]:
        return _normalise_override_records(value)

    @model_validator(mode="after")
    def validate_identity(self) -> "ForecastReasoningInput":
        if not self.company_id:
            raise ValueError("Forecast reasoning company_id cannot be blank")
        if not self.unit:
            raise ValueError("Forecast reasoning unit cannot be blank")
        return self

    @property
    def years(self) -> tuple[int, ...]:
        return self.forecast_years

    @property
    def overrides(self) -> tuple[ForecastOverride, ...]:
        return self.manual_overrides

    @property
    def forward_driver_observations(self) -> tuple[OperatingDriverObservation, ...]:
        return self.manual_forward_driver_observations

    @classmethod
    def from_artifacts(
        cls,
        financials: NormalizedCompanyFinancials,
        *,
        as_of: datetime.date,
        forecast_years: tuple[int, ...] | list[int],
        segments: Any = (),
        definitions: Any = (),
        observations: Any = (),
        management_guidance: Any = (),
        management_constraints: Any = (),
        investment_programs: Any = (),
        historical_facts: Any = None,
        **kwargs: Any,
    ) -> "ForecastReasoningInput":
        """Build compact reasoning input from existing typed artifacts.

        Only scalar normalized facts are copied.  Filing text and provider
        retrieval payloads never cross this boundary.
        """

        if not isinstance(financials, NormalizedCompanyFinancials):
            financials = NormalizedCompanyFinancials.model_validate(financials)
        if historical_facts is None:
            historical_facts = tuple(
                HistoricalFactSummary(
                    fiscal_year=item.fiscal_year,
                    metric=item.concept.value,
                    value=item.value,
                    unit=item.unit,
                    fiscal_period=item.fiscal_period.value,
                    method=(
                        item.derivation_kind.value
                        if item.derivation_kind is not None
                        else "normalized_historical"
                    ),
                    source=item.provider,
                    source_id=item.accession_number,
                    source_date=item.filed,
                )
                for item in financials.observations
                if item.filed is None or item.filed <= as_of
            )
        return cls(
            company_id=financials.company_id,
            company_name=financials.company_name,
            ticker=financials.ticker,
            unit=next(
                (
                    item.unit
                    for item in financials.observations
                    if item.concept.value == "revenue"
                ),
                "currency",
            ),
            as_of=as_of,
            forecast_years=tuple(forecast_years),
            segments=segments,
            definitions=definitions,
            observations=observations,
            management_guidance=management_guidance,
            management_constraints=management_constraints,
            investment_programs=investment_programs,
            historical_facts=historical_facts,
            **kwargs,
        )


class ReasonedForecastAssumption(BaseModel):
    """Strict non-executable model proposal for one target path."""

    model_config = ConfigDict(extra="forbid")

    assumption_id: str
    scope: ForecastScope
    scope_id: str = Field(
        default="company", validation_alias=AliasChoices("scope_id", "segment_id")
    )
    target_type: Literal["operating_driver", "forecast_metric"]
    metric: str | None = None
    driver_id: str | None = None
    unit: str
    basis: ForecastReasoningValueBasis = Field(
        validation_alias=AliasChoices("basis", "value_basis")
    )
    fiscal_years: tuple[int, ...]
    low: tuple[ForecastDecimal, ...]
    base: tuple[ForecastDecimal, ...]
    high: tuple[ForecastDecimal, ...]
    method: str = Field(
        default="evidence",
        validation_alias=AliasChoices("method", "supported_method"),
    )
    evidence_ids: tuple[str, ...] = ()
    rationale: str
    confidence: str
    evidence_based: bool = False
    model_assumption: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_source_label(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        source = data.pop(
            "assumption_type",
            data.pop(
                "source_type",
                data.pop(
                    "assumption_origin",
                    data.pop(
                        "origin", data.pop("source", data.pop("classification", None))
                    ),
                ),
            ),
        )
        if source is not None:
            normalized = str(getattr(source, "value", source)).strip().casefold()
            if normalized in {"evidence_based", "evidence", "reasoned_assumption"}:
                data.setdefault("evidence_based", True)
                data.setdefault("model_assumption", False)
            elif normalized in {"model_assumption", "model", "inference"}:
                data.setdefault("evidence_based", False)
                data.setdefault("model_assumption", True)
        return data

    @field_validator("assumption_id")
    @classmethod
    def normalize_assumption_id(cls, value: str) -> str:
        return _text(value, "Assumption ID")

    @field_validator("scope_id", "unit", "method", "rationale")
    @classmethod
    def normalize_assumption_text(cls, value: str) -> str:
        return _text(value, "Forecast assumption text")

    @field_validator("metric", "driver_id")
    @classmethod
    def normalize_optional_target(cls, value: str | None) -> str | None:
        return _text(value, "Forecast assumption target") if value is not None else None

    @field_validator("fiscal_years", mode="before")
    @classmethod
    def normalize_assumption_years(cls, value: Any) -> tuple[int, ...]:
        years = tuple(int(item) for item in (value or ()))
        if not years or tuple(sorted(years)) != years or len(years) != len(set(years)):
            raise ValueError(
                "Forecast assumption fiscal_years must be sorted and unique"
            )
        return years

    @field_validator("low", "base", "high", mode="before")
    @classmethod
    def normalize_paths(cls, value: Any) -> tuple[Decimal, ...]:
        values = (
            (value,)
            if isinstance(value, (str, int, float, Decimal))
            else tuple(value or ())
        )
        return tuple(_decimal(item, "Forecast assumption path") for item in values)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value: Any) -> tuple[str, ...]:
        values = (value,) if isinstance(value, str) else tuple(value or ())
        normalized = tuple(_text(item, "Evidence ID") for item in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("Forecast assumption evidence IDs must be unique")
        if any(
            item.casefold().startswith(("http://", "https://")) for item in normalized
        ):
            raise ValueError("URLs are not evidence citations; use catalog IDs")
        return normalized

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        normalized = str(value).strip().casefold()
        if normalized not in {"high", "medium", "low"}:
            raise ValueError(
                "Forecast assumption confidence must be high, medium, or low"
            )
        return normalized

    @model_validator(mode="after")
    def validate_shape(self) -> "ReasonedForecastAssumption":
        if self.target_type == "operating_driver":
            if not self.driver_id or self.metric is not None:
                raise ValueError("Operating-driver assumptions require driver_id only")
        elif not self.metric or self.driver_id is not None:
            raise ValueError("Forecast-metric assumptions require metric only")
        if not self.low or len(self.low) != len(self.fiscal_years):
            raise ValueError("Forecast assumption low path must match fiscal horizon")
        if len(self.base) != len(self.fiscal_years) or len(self.high) != len(
            self.fiscal_years
        ):
            raise ValueError("Forecast assumption paths must match fiscal horizon")
        for low, base, high in zip(self.low, self.base, self.high, strict=True):
            if not low <= base <= high:
                raise ValueError(
                    "Forecast assumption paths require low <= base <= high"
                )
        if self.evidence_based == self.model_assumption:
            raise ValueError(
                "Forecast assumptions must be exactly evidence_based or model_assumption"
            )
        return self

    @property
    def target_key(self) -> tuple[str, str, str]:
        return (
            self.scope.value,
            self.scope_id,
            _canonical_assumption_metric(self.metric)
            if self.target_type == "forecast_metric"
            else f"driver:{canonical_driver_id(self.driver_id)}",
        )

    @property
    def assumption_type(self) -> str:
        return "evidence_based" if self.evidence_based else "model_assumption"

    @property
    def supported_method(self) -> str:
        return self.method


class ProposedModelingDecision(BaseModel):
    """A constrained planning proposal, separate from ForecastPlan."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    scope: ForecastScope
    scope_id: str = "company"
    metric: str | None = None
    driver_id: str | None = None
    strategy: str
    unit: str | None = None
    basis: ForecastReasoningValueBasis | None = None
    fiscal_years: tuple[int, ...] = ()
    rationale: str

    @field_validator("decision_id", "scope_id", "strategy", "rationale")
    @classmethod
    def normalize_decision_text(cls, value: str) -> str:
        return _text(value, "Modeling decision text")

    @field_validator("metric", "driver_id", "unit")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return _text(value, "Modeling decision target") if value is not None else None

    @field_validator("fiscal_years", mode="before")
    @classmethod
    def normalize_decision_years(cls, value: Any) -> tuple[int, ...]:
        return tuple(int(item) for item in (value or ()))

    @model_validator(mode="after")
    def validate_decision_target(self) -> "ProposedModelingDecision":
        if (self.metric is None) == (self.driver_id is None):
            raise ValueError(
                "Modeling decisions require exactly one metric or driver_id"
            )
        return self


class ForecastUnresolvedItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    category: str
    reason: str
    status: str = "unresolved"
    scope: ForecastScope | None = None
    scope_id: str | None = None
    metric: str | None = None
    driver_id: str | None = None
    fiscal_years: tuple[int, ...] = ()

    @field_validator("item_id", "category", "reason", "status")
    @classmethod
    def normalize_unresolved_text(cls, value: str) -> str:
        return _text(value, "Unresolved item text")


class ForecastReasoningResponse(BaseModel):
    """Strict structured output accepted from OpenAI, before post-validation."""

    model_config = ConfigDict(extra="forbid")

    assumptions: tuple[ReasonedForecastAssumption, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "assumptions", "reasoned_assumptions", "reasoned_forecast_assumptions"
        ),
    )
    modeling_decisions: tuple[ProposedModelingDecision, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "modeling_decisions",
            "proposed_modeling_decisions",
            "proposed_decisions",
            "decisions",
        ),
    )
    unresolved_items: tuple[ForecastUnresolvedItem, ...] = Field(
        default=(), validation_alias=AliasChoices("unresolved_items", "unresolved")
    )
    warnings: tuple[str, ...] = ()
    overall_confidence: str = "medium"

    @field_validator(
        "assumptions", "modeling_decisions", "unresolved_items", mode="before"
    )
    @classmethod
    def normalize_response_records(cls, value: Any) -> tuple[Any, ...]:
        return _tuple_records(value)

    @field_validator("warnings", mode="before")
    @classmethod
    def normalize_warnings(cls, value: Any) -> tuple[str, ...]:
        values = (value,) if isinstance(value, str) else tuple(value or ())
        return tuple(str(item).strip() for item in values if str(item).strip())

    @field_validator("overall_confidence", mode="before")
    @classmethod
    def normalize_overall_confidence(cls, value: str) -> str:
        normalized = str(value).strip().casefold()
        if normalized not in {"high", "medium", "low"}:
            raise ValueError(
                "Overall forecast reasoning confidence must be high, medium, or low"
            )
        return normalized


class ForecastReasoningMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    reasoning_effort: str
    prompt_version: str
    schema_version: str
    validator_version: str
    context_version: str
    prompt_hash: str
    schema_hash: str
    validator_hash: str
    context_hash: str
    evidence_bundle_hash: str
    research_hash: str
    manual_inputs_hash: str

    @property
    def prompt_identity(self) -> str:
        return self.prompt_hash

    @property
    def schema_identity(self) -> str:
        return self.schema_hash

    @property
    def validator_identity(self) -> str:
        return self.validator_hash


class ForecastReasoningCacheIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    company_id: str
    as_of: datetime.date
    forecast_years: tuple[int, ...]
    evidence_bundle_hash: str
    research_hash: str
    manual_inputs_hash: str
    model: str
    reasoning_effort: str
    prompt_version: str
    schema_version: str
    validator_version: str
    context_version: str
    prompt_hash: str
    schema_hash: str
    validator_hash: str
    context_hash: str

    @property
    def digest(self) -> str:
        from edgarito.services.forecasting.reasoning.evidence import content_hash

        return content_hash(self.model_dump(mode="json"))


class ForecastReasoningValidationIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    assumption_id: str | None = None
    decision_id: str | None = None
    code: str
    reason: str


class ForecastReasoningInputValidationError(ValueError):
    """Execution-gate failure for invalid supplied forecast artifacts."""

    code = "INVALID_FORECAST_REASONING_INPUT"

    def __init__(
        self,
        issues: tuple[ForecastReasoningValidationIssue, ...]
        | list[ForecastReasoningValidationIssue],
    ):
        self.issues = tuple(issues)
        details = "; ".join(f"{issue.code}: {issue.reason}" for issue in self.issues)
        super().__init__(f"{self.code}: {details}")


class ForecastReasoningCompilation(BaseModel):
    """Compiler output, still immutable and suitable for execution."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    plan: ForecastPlan
    observations: tuple[OperatingDriverObservation, ...] = ()
    overrides: tuple[ForecastOverride, ...] = ()
    retained_ranges: dict[str, tuple[tuple[Decimal, Decimal, Decimal], ...]] = {}
    collisions: tuple[ForecastReasoningValidationIssue, ...] = ()
    warnings: tuple[str, ...] = ()


class ForecastReasoningResult(BaseModel):
    """Rich immutable audit/result surface produced by the opt-in service."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    proposal_identity: str
    audit_identity: tuple[str, ...] = ()
    proposal: Any
    accepted_assumptions: tuple[ReasonedForecastAssumption, ...] = ()
    accepted_decisions: tuple[ProposedModelingDecision, ...] = ()
    rejected_assumptions: tuple[ForecastReasoningValidationIssue, ...] = ()
    rejected_decisions: tuple[ForecastReasoningValidationIssue, ...] = ()
    unresolved_items: tuple[Any, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence_catalog: Any
    metadata: ForecastReasoningMetadata
    compiled_plan: ForecastPlan
    compiled_observations: tuple[OperatingDriverObservation, ...] = ()
    compiled_overrides: tuple[ForecastOverride, ...] = ()
    retained_ranges: dict[str, tuple[tuple[Decimal, Decimal, Decimal], ...]] = {}
    collisions: tuple[ForecastReasoningValidationIssue, ...] = ()
    cache_hit: bool = False
    driver_result: Any | None = None

    @property
    def forecast(self):
        return getattr(self.driver_result, "forecast", None)

    @property
    def validation(self):
        return getattr(self.driver_result, "validation", None)

    @property
    def canonical_forecast(self):
        return self.forecast

    @property
    def canonical_validation(self):
        return self.validation

    @property
    def compiled_plan_result(self) -> ForecastPlan:
        return self.compiled_plan


class ReasonedDriverBasedForecastResult(BaseModel):
    """Canonical driver result paired with the reasoner audit."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    reasoning: ForecastReasoningResult
    driver_result: Any

    @property
    def reasoning_result(self) -> ForecastReasoningResult:
        return self.reasoning

    @property
    def forecast(self):
        return getattr(self.driver_result, "forecast", None)

    @property
    def validation(self):
        return getattr(self.driver_result, "validation", None)

    @property
    def canonical_driver_result(self):
        return self.driver_result

    @property
    def canonical_forecast(self):
        return self.forecast

    @property
    def canonical_validation(self):
        return self.validation

    @property
    def driver_forecast(self):
        return self.forecast

    @property
    def driver_validation(self):
        return self.validation


ForecastReasoningAssumption = ReasonedForecastAssumption
ForecastReasoningDecision = ProposedModelingDecision
ForecastReasonedDriverResult = ReasonedDriverBasedForecastResult
build_forecast_reasoning_input = ForecastReasoningInput.from_artifacts
build_reasoning_input = ForecastReasoningInput.from_artifacts


def _canonical_assumption_metric(value: str | None) -> str:
    raw = str(value or "").strip().casefold()
    normalized = raw.replace("-", "_").replace(" ", "_")
    return {
        "research_and_development": "r_and_d",
        "research_and_development_expense": "r_and_d",
        "selling_general_and_administrative": "sg_and_a",
        "selling_general_and_administrative_expense": "sg_and_a",
        "depreciation": "depreciation_and_amortization",
        "capital_expenditures": "capex",
        "capital_expenditure": "capex",
        "owc": "operating_working_capital",
        "effective_tax_rate": "tax_rate",
    }.get(normalized, normalized)


def canonical_driver_id(value: str | None) -> str:
    """Normalize configured executor vocabulary for collision identity."""

    raw = str(value or "").strip().casefold()
    normalized = raw.replace("-", "_").replace(" ", "_")
    try:
        from edgarito.config.operating import OPERATING_VOCABULARY

        configured = OPERATING_VOCABULARY.metric_aliases.get(normalized)
        if configured:
            return configured
    except (ImportError, AttributeError):
        pass
    return {"growth_rate": "growth", "conversion": "conversion_rate"}.get(
        normalized, raw
    )


__all__ = [
    "ForecastDecimal",
    "ForecastReasoningValueBasis",
    "ForecastValueBasis",
    "ForecastReasoningInput",
    "HistoricalFactSummary",
    "ReasonedForecastAssumption",
    "ProposedModelingDecision",
    "ForecastUnresolvedItem",
    "ForecastReasoningResponse",
    "ForecastReasoningMetadata",
    "ForecastReasoningCacheIdentity",
    "ForecastReasoningValidationIssue",
    "ForecastReasoningInputValidationError",
    "ForecastReasoningCompilation",
    "ForecastReasoningResult",
    "ReasonedDriverBasedForecastResult",
    "ForecastReasoningAssumption",
    "ForecastReasoningDecision",
    "ForecastReasonedDriverResult",
    "build_forecast_reasoning_input",
    "build_reasoning_input",
    "canonical_driver_id",
]
