"""Immutable contracts for isolated, point-in-time forecast evaluation.

The evaluation package deliberately has its own boundary around the existing
forecasting services.  In particular, realized outcomes are represented by
different models from the information set supplied to a forecaster.  This
makes it difficult to accidentally put realized data in a prompt, evidence
catalog, or cache identity.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from edgarito.schemas.forecasting import FcffForecastParameters, ForecastScope
from edgarito.schemas.normalization.financials import (
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.forecasting.reasoning.contracts import ForecastReasoningInput


def _decimal(value: Any, label: str) -> Decimal:
    if value is None:
        raise ValueError(f"{label} cannot be null")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def _text(value: str, label: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label} cannot be blank")
    return result


def _unit_key(unit: str, currency: str | None = None) -> str:
    raw = str(unit).strip().casefold()
    parts = raw.split()
    inferred_currency = (
        parts[0]
        if parts and len(parts[0]) == 3 and parts[0].isalpha()
        else None
    )
    if currency is not None and inferred_currency is not None and currency.casefold() != inferred_currency:
        return f"{currency.casefold()}::incompatible:{raw.replace(' ', '')}"
    if currency is None and inferred_currency is not None:
        currency = inferred_currency
    if inferred_currency is not None:
        raw = " ".join(parts[1:]) or "base"
    normalized = raw.replace(" ", "")
    normalized = {
        "%": "percent",
        "percentage": "percent",
        "percentagepoints": "percent",
        "pp": "percent",
    }.get(
        normalized, normalized
    )
    return f"{currency.casefold()}::{normalized}" if currency else normalized


def canonical_financial_metric(value: Any) -> str:
    """Return the canonical name used by financial scoring and matching."""

    raw = (
        str(getattr(value, "value", value))
        .strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
    )
    return {
        "operating_income": "ebit",
        "operating_income_loss": "ebit",
        "effective_tax_rate": "tax_rate",
        "tax_rate_percent": "tax_rate",
        "depreciation": "depreciation_and_amortization",
        "da": "depreciation_and_amortization",
        "capital_expenditures": "capex",
        "capital_expenditure": "capex",
        "owc": "operating_working_capital",
        "operating_working_capital_to_revenue": "operating_working_capital",
        "change_in_operating_working_capital": "delta_nwc",
        "change_in_working_capital": "delta_nwc",
    }.get(raw, raw)


def _decimal_key(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _target_key_text(value: Any) -> str:
    return canonical_financial_metric(value)


def _basis_key(value: Any) -> str:
    raw = str(getattr(value, "value", value)).strip().casefold().replace("-", "_").replace(" ", "_")
    return {
        "amount": "absolute",
        "currency": "absolute",
        "ratio": "percent_of_revenue",
        "percent": "percent_of_revenue",
        "percentage": "percentage_points",
        "pp": "percentage_points",
    }.get(raw, raw)


def _contains_actual(value: Any, seen: set[int] | None = None) -> bool:
    """Return whether a value contains one of the outcome-only contracts."""

    seen = seen or set()
    if value is None or id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, (ActualOutcomeData, ActualFinancialObservation, ActualAssumptionOutcome)):
        return True
    if isinstance(value, BaseModel):
        return any(_contains_actual(item, seen) for item in value.__dict__.values())
    if isinstance(value, Mapping):
        return any(
            str(key).casefold()
            in {
                "actual",
                "actual_value",
                "actuals",
                "actual_outcomes",
                "actual_financials",
                "actual_observations",
                "assumption_outcomes",
                "financial_observations",
            }
            or _contains_actual(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Sequence | set | frozenset):
        return any(_contains_actual(item, seen) for item in value)
    return False


def _stable_value(value: Any, seen: set[int] | None = None) -> Any:
    """Convert typed and small fake objects to a deterministic identity form."""

    seen = seen or set()
    if value is None or isinstance(value, (str, int, float, bool, Decimal)):
        return value
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if id(value) in seen:
        return "<cycle>"
    seen.add(id(value))
    if isinstance(value, NormalizedCompanyFinancials):
        return _stable_value(
            {
                "provider": value.provider,
                "company_id": value.company_id,
                "company_name": value.company_name,
                "ticker": value.ticker,
                "identifiers": value.identifiers,
                "retrieved_at": value.retrieved_at,
                "observations": value.observations,
            },
            seen,
        )
    if isinstance(value, BaseModel):
        return _stable_value(value.model_dump(mode="python"), seen)
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item, seen)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key).casefold() != "generated_at"
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        values = [_stable_value(item, seen) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(
                values,
                key=lambda item: json.dumps(
                    item, ensure_ascii=True, sort_keys=True, default=str
                ),
            )
        return values
    values = getattr(value, "__dict__", None)
    return _stable_value(values, seen) if isinstance(values, dict) else str(value)


class ImmutableNormalizedCompanyFinancials(NormalizedCompanyFinancials):
    """Frozen view of the existing normalized information-set contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    observations: tuple[FinancialObservation, ...] = ()


class InformationAvailabilityRecord(BaseModel):
    """Explicit availability link for an information-set item."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity: str = Field(validation_alias=AliasChoices("identity", "item_identity", "evidence_id"))
    category: str
    available_on: datetime.date = Field(
        validation_alias=AliasChoices("available_on", "availability_date", "source_date")
    )
    content_hash: str = Field(
        validation_alias=AliasChoices(
            "content_hash", "payload_hash", "canonical_content_hash"
        )
    )
    source: str
    source_id: str
    provenance: str
    manual: bool = False

    @field_validator(
        "identity", "category", "content_hash", "source", "source_id", "provenance"
    )
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return _text(value, "Information availability record")

    @property
    def payload_hash(self) -> str:
        return self.content_hash

    @property
    def canonical_content_hash(self) -> str:
        return self.content_hash

    @model_validator(mode="after")
    def validate_manual_source(self) -> "InformationAvailabilityRecord":
        if self.manual and self.source.casefold() not in {"manual", "case"}:
            raise ValueError("Manual availability records must identify a manual source")
        return self

class ActualFinancialObservation(BaseModel):
    """One realized financial outcome, kept outside the information set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fiscal_year: int
    fiscal_period: str = Field(default="FY", validation_alias=AliasChoices("fiscal_period", "period"))
    period_start: datetime.date | None = None
    period_end: datetime.date
    metric: str = Field(validation_alias=AliasChoices("metric", "concept", "name"))
    value: Decimal | None = Field(default=None, validation_alias=AliasChoices("value", "actual", "actual_value"))
    unit: str = "currency"
    scale: Decimal = Decimal(1)
    currency: str | None = None
    source: str = "evaluation_fixture"
    source_id: str | None = None
    source_concept: str | None = None
    source_date: datetime.date | None = None
    filing_date: datetime.date | None = None
    value_kind: Literal["reported", "derived"] = "reported"
    reconstruction_provenance: str | None = None
    sign_normalization: str | None = None
    provenance: str | None = None

    @field_validator("metric", "fiscal_period", "unit", "source")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return _text(value, "Actual financial observation text")

    @field_validator("value", mode="before")
    @classmethod
    def normalize_value(cls, value: Any) -> Decimal | None:
        return None if value is None else _decimal(value, "Actual financial value")

    @field_validator("scale", mode="before")
    @classmethod
    def normalize_scale(cls, value: Any) -> Decimal:
        result = _decimal(value, "Actual financial scale")
        if result <= 0:
            raise ValueError("Actual financial scale must be positive")
        return result

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        return _text(value, "Actual financial currency") if value is not None else None

    @field_validator("source_id", "source_concept", "reconstruction_provenance", "sign_normalization")
    @classmethod
    def normalize_optional_metadata(cls, value: str | None) -> str | None:
        return _text(value, "Actual financial source metadata") if value is not None else None

    @model_validator(mode="after")
    def require_derived_provenance(self) -> "ActualFinancialObservation":
        if self.value_kind == "derived" and not self.reconstruction_provenance:
            raise ValueError("Derived actual observations require reconstruction provenance")
        return self

    @model_validator(mode="before")
    @classmethod
    def normalize_filing_date_alias(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "filing_date" in value and "source_date" not in value:
            payload = dict(value)
            payload["source_date"] = payload["filing_date"]
            return payload
        return value

    @field_validator("provenance")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return _text(value, "Actual financial provenance") if value is not None else None

    @property
    def concept(self) -> str:
        return self.metric

    @property
    def actual(self) -> Decimal | None:
        return self.value

    @property
    def unit_key(self) -> str:
        return _unit_key(self.unit, self.currency)


class ActualAssumptionOutcome(BaseModel):
    """One realized target for matching a reasoned assumption."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_type: Literal["operating_driver", "forecast_metric"]
    scope: ForecastScope | str = ForecastScope.COMPANY
    scope_id: str = "company"
    metric: str | None = None
    driver_id: str | None = None
    fiscal_year: int
    basis: str = "absolute"
    actual: Decimal | None = Field(
        default=None, validation_alias=AliasChoices("actual", "actual_value", "value")
    )
    unit: str = "currency"
    scale: Decimal = Decimal(1)
    currency: str | None = None
    source: str = "evaluation_fixture"
    source_id: str | None = None
    source_concept: str | None = None
    source_date: datetime.date | None = None
    provenance: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_target_alias(cls, value: Any) -> Any:
        if not isinstance(value, Mapping) or "target" not in value:
            return value
        payload = dict(value)
        target = payload.pop("target")
        target_type = str(getattr(payload.get("target_type"), "value", payload.get("target_type")))
        if target_type == "operating_driver":
            payload.setdefault("driver_id", target)
        else:
            payload.setdefault("metric", target)
        return payload

    @field_validator("scope_id", "basis", "unit", "source")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return _text(value, "Actual assumption outcome text")

    @field_validator("metric", "driver_id", "provenance")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return _text(value, "Actual assumption outcome target") if value is not None else None

    @field_validator("actual", mode="before")
    @classmethod
    def normalize_actual(cls, value: Any) -> Decimal | None:
        return None if value is None else _decimal(value, "Actual assumption value")

    @field_validator("scale", mode="before")
    @classmethod
    def normalize_scale(cls, value: Any) -> Decimal:
        result = _decimal(value, "Actual assumption scale")
        if result <= 0:
            raise ValueError("Actual assumption scale must be positive")
        return result

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        return _text(value, "Actual assumption currency") if value is not None else None

    @field_validator("source_id", "source_concept")
    @classmethod
    def normalize_source_metadata(cls, value: str | None) -> str | None:
        return _text(value, "Actual assumption source metadata") if value is not None else None

    @model_validator(mode="after")
    def validate_target(self) -> "ActualAssumptionOutcome":
        if self.target_type == "operating_driver" and (not self.driver_id or self.metric is not None):
            raise ValueError("Operating-driver outcomes require driver_id only")
        if self.target_type == "forecast_metric" and (not self.metric or self.driver_id is not None):
            raise ValueError("Forecast-metric outcomes require metric only")
        if self.scope == ForecastScope.COMPANY and self.scope_id != "company":
            raise ValueError("Company outcomes require scope_id='company'")
        return self

    @property
    def target(self) -> str:
        return self.driver_id or self.metric or ""

    @property
    def actual_value(self) -> Decimal | None:
        return self.actual

    @property
    def target_key(self) -> tuple[str, str, str, str, int, str, str, str]:
        scope = getattr(self.scope, "value", self.scope)
        return (
            self.target_type,
            str(scope),
            self.scope_id,
            _target_key_text(self.target),
            self.fiscal_year,
            _basis_key(self.basis),
            self.unit_key,
            _decimal_key(self.scale),
        )

    @property
    def unit_key(self) -> str:
        return _unit_key(self.unit, self.currency)


class ActualOutcomeData(BaseModel):
    """Outcome-only data attached after a forecast has completed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    company: str | None = None
    ticker: str | None = None
    observations: tuple[ActualFinancialObservation, ...] = Field(
        default=(), validation_alias=AliasChoices("observations", "financial_observations")
    )
    assumption_outcomes: tuple[ActualAssumptionOutcome, ...] = Field(
        default=(), validation_alias=AliasChoices("assumption_outcomes", "assumptions")
    )
    outcome_dates: tuple[datetime.date, ...] = ()

    @field_validator("company", "ticker")
    @classmethod
    def normalize_identity(cls, value: str | None) -> str | None:
        return _text(value, "Actual outcome identity") if value is not None else None

    @field_validator("observations", "assumption_outcomes", "outcome_dates", mode="before")
    @classmethod
    def normalize_collections(cls, value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, Mapping)):
            return (value,)
        return tuple(value)

    @model_validator(mode="after")
    def require_unique_observations(self) -> "ActualOutcomeData":
        keys = [
            (item.fiscal_year, item.fiscal_period, canonical_financial_metric(item.metric))
            for item in self.observations
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Actual financial observations must be unique by period and metric")
        assumption_keys = [item.target_key for item in self.assumption_outcomes]
        if len(assumption_keys) != len(set(assumption_keys)):
            raise ValueError(
                "Actual assumption outcomes must be unique by normalized target and unit"
            )
        return self

    def for_year(self, year: int) -> tuple[ActualFinancialObservation, ...]:
        return tuple(item for item in self.observations if item.fiscal_year == year)

    @property
    def financial_observations(self) -> tuple[ActualFinancialObservation, ...]:
        return self.observations

    @property
    def assumptions(self) -> tuple[ActualAssumptionOutcome, ...]:
        return self.assumption_outcomes

    def assumption(self, key: tuple[str, ...]) -> ActualAssumptionOutcome | None:
        return next((item for item in self.assumption_outcomes if item.target_key == key), None)


class EvidenceSnapshot(BaseModel):
    """Optional typed wrapper for frozen operating/hybrid evidence payloads.

    The case accepts arbitrary existing operating result objects for backwards
    compatibility, while this wrapper provides a strict, serializable shape for
    fixture authors and new callers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    operating: Any = None
    hybrid: Any = None
    metadata: tuple[tuple[str, str], ...] = ()

    @field_validator("metadata", mode="before")
    @classmethod
    def normalize_metadata(cls, value: Any) -> tuple[tuple[str, str], ...]:
        if isinstance(value, Mapping):
            value = value.items()
        return tuple(sorted((str(key), str(item)) for key, item in (value or ())))

    @model_validator(mode="after")
    def reject_outcomes(self) -> "EvidenceSnapshot":
        if _contains_actual((self.operating, self.hybrid)):
            raise ValueError("Actual outcome data cannot be part of evidence")
        return self


class ForecastBacktestCase(BaseModel):
    """A frozen point-in-time information set and exact forecast horizon."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    ticker: str
    company: str = Field(validation_alias=AliasChoices("company", "company_name"))
    as_of: datetime.date
    fiscal_years: tuple[int, ...] = Field(
        validation_alias=AliasChoices(
            "fiscal_years", "exact_fiscal_years", "target_years", "forecast_years"
        )
    )
    point_in_time_financials: NormalizedCompanyFinancials = Field(
        validation_alias=AliasChoices(
            "point_in_time_financials", "financials", "information_set"
        )
    )
    reasoning_input: ForecastReasoningInput | None = None
    evidence_snapshot: Any = Field(
        default=None, validation_alias=AliasChoices("evidence_snapshot", "evidence")
    )
    operating_evidence: Any = Field(
        default=None, validation_alias=AliasChoices("operating_evidence", "operating")
    )
    hybrid_evidence: Any = Field(
        default=None, validation_alias=AliasChoices("hybrid_evidence", "hybrid")
    )
    availability_manifest: tuple[InformationAvailabilityRecord, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "availability_manifest", "information_availability", "availability_records"
        ),
    )
    parameters: FcffForecastParameters | None = None
    expected_archetypes: tuple[Any, ...] | None = None

    @field_validator("ticker", "company")
    @classmethod
    def normalize_identity(cls, value: str) -> str:
        return _text(value, "Backtest identity")

    @field_validator("point_in_time_financials", mode="before")
    @classmethod
    def freeze_financials(cls, value: Any) -> ImmutableNormalizedCompanyFinancials:
        if isinstance(value, ImmutableNormalizedCompanyFinancials):
            return value
        normalized = (
            value
            if isinstance(value, NormalizedCompanyFinancials)
            else NormalizedCompanyFinancials.model_validate(value)
        )
        return ImmutableNormalizedCompanyFinancials.model_validate(
            normalized.model_dump(mode="python")
        )

    @field_validator("fiscal_years", mode="before")
    @classmethod
    def normalize_years(cls, value: Any) -> tuple[int, ...]:
        if isinstance(value, int):
            value = (value,)
        years = tuple(int(item) for item in (value or ()))
        if not years or tuple(sorted(years)) != years or len(years) != len(set(years)):
            raise ValueError("Backtest fiscal_years must be sorted and unique")
        return years

    @field_validator(
        "reasoning_input",
        "evidence_snapshot",
        "operating_evidence",
        "hybrid_evidence",
        mode="before",
    )
    @classmethod
    def reject_outcome_payloads(cls, value: Any) -> Any:
        if _contains_actual(value):
            raise ValueError("Actual outcome data cannot be part of a backtest information set")
        return value

    @field_validator("expected_archetypes", mode="before")
    @classmethod
    def normalize_archetypes(cls, value: Any) -> tuple[Any, ...] | None:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return (value,)
        return tuple(value)

    @field_validator("availability_manifest", mode="before")
    @classmethod
    def normalize_availability_manifest(
        cls, value: Any
    ) -> tuple[InformationAvailabilityRecord, ...]:
        if value is None:
            return ()
        if isinstance(value, Mapping):
            value = tuple(
                {
                    **(
                        item.model_dump(mode="python")
                        if isinstance(item, InformationAvailabilityRecord)
                        else item
                        if isinstance(item, Mapping)
                        else {}
                    ),
                    "identity": key,
                }
                for key, item in value.items()
            )
        return tuple(
            item
            if isinstance(item, InformationAvailabilityRecord)
            else InformationAvailabilityRecord.model_validate(item)
            for item in value
        )

    @model_validator(mode="after")
    def validate_case_identity(self) -> "ForecastBacktestCase":
        financials = self.point_in_time_financials
        if financials.company_name.casefold() != self.company.casefold():
            raise ValueError("Backtest company must match point-in-time financials")
        if financials.ticker and financials.ticker.casefold() != self.ticker.casefold():
            raise ValueError("Backtest ticker must match point-in-time financials")
        if self.reasoning_input is not None:
            if self.reasoning_input.company_id != financials.company_id:
                raise ValueError("Reasoning input company does not match the case")
            if self.reasoning_input.as_of != self.as_of:
                raise ValueError("Reasoning input as_of does not match the case")
            if self.reasoning_input.forecast_years != self.fiscal_years:
                raise ValueError("Reasoning input horizon does not match the case")
        if self.parameters is not None and self.parameters.forecast_years != len(self.fiscal_years):
            raise ValueError("FCFF parameters horizon does not match the case")
        if self.parameters is None:
            object.__setattr__(
                self,
                "parameters",
                FcffForecastParameters(forecast_years=len(self.fiscal_years)),
            )
        identities = tuple(item.identity for item in self.availability_manifest)
        if len(identities) != len(set(identities)):
            raise ValueError("Availability manifest identities must be unique")
        return self

    @property
    def exact_fiscal_years(self) -> tuple[int, ...]:
        return self.fiscal_years

    @property
    def target_years(self) -> tuple[int, ...]:
        return self.fiscal_years

    @property
    def information_set(self) -> NormalizedCompanyFinancials:
        return self.point_in_time_financials

    @property
    def point_in_time(self) -> NormalizedCompanyFinancials:
        return self.point_in_time_financials

    @property
    def financials(self) -> NormalizedCompanyFinancials:
        return self.point_in_time_financials

    @property
    def evidence(self) -> Any:
        return self.evidence_snapshot

    @property
    def reasoning_input_snapshot(self) -> ForecastReasoningInput | None:
        return self.reasoning_input

    @property
    def information_availability(self) -> tuple[InformationAvailabilityRecord, ...]:
        return self.availability_manifest

    @property
    def case_id(self) -> str:
        from edgarito.services.forecasting.reasoning.evidence import content_hash

        return content_hash(
            _stable_value(
                {
                    "ticker": self.ticker,
                    "company": self.company,
                    "as_of": self.as_of,
                    "fiscal_years": self.fiscal_years,
                    "financials": self.point_in_time_financials,
                    "reasoning_input": self.reasoning_input,
                    "evidence": self.evidence_snapshot,
                    "operating_evidence": self.operating_evidence,
                    "hybrid_evidence": self.hybrid_evidence,
                    "parameters": self.parameters,
                    "availability_manifest": self.availability_manifest,
                }
            )
        )


class LeakageStatus(str, Enum):
    INCLUDED = "included"
    EXCLUDED = "excluded"
    UNDATED = "undated"


class LeakageEvidenceRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    category: str
    status: LeakageStatus
    date: datetime.date | None = None
    dates: tuple[datetime.date, ...] = ()
    reason: str | None = None


class LeakageAudit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    as_of: datetime.date
    availability_mode: str = "point_in_time"
    included: tuple[LeakageEvidenceRecord, ...] = ()
    excluded: tuple[LeakageEvidenceRecord, ...] = ()
    undated: tuple[LeakageEvidenceRecord, ...] = ()
    information_dates: tuple[tuple[str, datetime.date], ...] = ()
    issues: tuple[str, ...] = ()
    valid: bool = False

    @property
    def passed(self) -> bool:
        return self.valid and not self.issues


class AssumptionScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    assumption_id: str
    target_type: str
    scope: str
    scope_id: str
    target: str
    fiscal_year: int
    basis: str
    reasoned_unit: str | None = None
    actual_unit: str | None = None
    low: Decimal
    base: Decimal
    high: Decimal
    actual: Decimal | None = None
    absolute_error: Decimal | None = None
    percentage_error: Decimal | None = None
    interval_hit: bool | None = None
    normalized_interval_position: Decimal | None = None
    confidence: str
    kind: str
    evidence_ids: tuple[str, ...] = ()
    scored: bool = False
    unmatched_reason: str | None = None

    @property
    def interval_position(self) -> Decimal | None:
        return self.normalized_interval_position


class AssumptionScoreReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[AssumptionScore, ...] = ()
    scored_count: int = 0
    unscored_count: int = 0


class FinancialMetricScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str
    metric: str
    fiscal_year: int
    forecast: Decimal | None = None
    actual: Decimal | None = None
    forecast_unit: str | None = None
    forecast_currency: str | None = None
    forecast_scale: Decimal | None = None
    actual_unit: str | None = None
    actual_currency: str | None = None
    actual_scale: Decimal | None = None
    absolute_error: Decimal | None = None
    percentage_error: Decimal | None = None
    sign_error: bool | None = None
    yoy_direction_error: bool | None = None
    scored: bool = False
    unmatched_reason: str | None = None

    @property
    def error(self) -> Decimal | None:
        return self.absolute_error

    @property
    def safe_percentage_error(self) -> Decimal | None:
        return self.percentage_error

    @property
    def direction_error(self) -> bool | None:
        return self.yoy_direction_error


class FinancialScoreReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str
    required_years: tuple[int, ...] = ()
    scores: tuple[FinancialMetricScore, ...] = ()
    per_method: dict[str, tuple[FinancialMetricScore, ...]] = Field(default_factory=dict)

    @property
    def scored_count(self) -> int:
        return sum(item.scored for item in self.scores)

    @property
    def unscored_count(self) -> int:
        return sum(not item.scored for item in self.scores)


class CalibrationStratum(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    confidence: str
    sample_size: int
    hit_count: int
    interval_coverage: Decimal | None = None
    warning: str | None = None


class CalibrationSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_size: int = 0
    hit_count: int = 0
    interval_coverage: Decimal | None = None
    strata: tuple[CalibrationStratum, ...] = ()
    warning: str = "No statistical significance is claimed by these descriptive aggregates."

    @property
    def coverage(self) -> Decimal | None:
        return self.interval_coverage


class ComplexityDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    segments: int = 0
    driver_assumptions: int = 0
    financial_assumptions: int = 0
    model_assumptions: int = 0
    evidence_based: int = 0
    unresolved: int = 0
    rejected: int = 0
    manual: int = 0
    validation_findings: int = 0
    over_modeling_indicators: tuple[str, ...] = ()


class StabilityObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_identity: str
    target: str
    fiscal_year: int
    base: Decimal | None = None
    low: Decimal | None = None
    high: Decimal | None = None


class StabilityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_count: int
    observations: tuple[StabilityObservation, ...] = ()
    minimum: dict[str, Decimal | None] = Field(default_factory=dict)
    maximum: dict[str, Decimal | None] = Field(default_factory=dict)
    mean: dict[str, Decimal | None] = Field(default_factory=dict)
    variance: dict[str, Decimal | None] = Field(default_factory=dict)
    dispersion: dict[str, Decimal | None] = Field(default_factory=dict)


class RouteIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    route: str
    collaborator: str | None = None
    available: bool = True
    reason: str | None = None
    evidence_identity: str | None = None


class BaselineComparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    route: RouteIdentity
    method: str
    financial_scores: FinancialScoreReport | None = None
    metric_deltas: dict[str, Any] = Field(default_factory=dict)
    unavailable_reason: str | None = None


class ForecastBacktestResult(BaseModel):
    """Complete immutable audit surface for one isolated backtest case."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    case_id: str
    ticker: str
    company: str
    as_of: datetime.date
    fiscal_years: tuple[int, ...]
    leakage_audit: LeakageAudit
    reasoned_result: Any = None
    canonical_forecast: Any = None
    actual_outcomes: ActualOutcomeData
    assumption_scores: AssumptionScoreReport = Field(default_factory=AssumptionScoreReport)
    financial_scores: FinancialScoreReport = Field(
        default_factory=lambda: FinancialScoreReport(method="reasoned")
    )
    calibration: CalibrationSummary = Field(default_factory=CalibrationSummary)
    complexity: ComplexityDiagnostics = Field(default_factory=ComplexityDiagnostics)
    validation: Any = None
    normalized_comparison: BaselineComparison | None = None
    hybrid_comparison: BaselineComparison | None = None
    actual_outcome_audit: Any = None
    diagnostics: tuple[str, ...] = ()
    routes: tuple[RouteIdentity, ...] = ()
    report_identity: str

    @property
    def actuals(self) -> ActualOutcomeData:
        return self.actual_outcomes

    @property
    def reasoning_result(self) -> Any:
        return self.reasoned_result

    @property
    def reasoned_forecast(self) -> Any:
        return self.canonical_forecast

    @property
    def baseline_comparisons(self) -> tuple[BaselineComparison, ...]:
        return tuple(item for item in (self.normalized_comparison, self.hybrid_comparison) if item is not None)


BacktestCase = ForecastBacktestCase
EvaluationResult = ForecastBacktestResult


__all__ = [
    "ActualFinancialObservation",
    "ActualAssumptionOutcome",
    "ActualOutcomeData",
    "EvidenceSnapshot",
    "ForecastBacktestCase",
    "ImmutableNormalizedCompanyFinancials",
    "InformationAvailabilityRecord",
    "LeakageStatus",
    "LeakageEvidenceRecord",
    "LeakageAudit",
    "AssumptionScore",
    "AssumptionScoreReport",
    "FinancialMetricScore",
    "FinancialScoreReport",
    "CalibrationStratum",
    "CalibrationSummary",
    "ComplexityDiagnostics",
    "StabilityObservation",
    "StabilityReport",
    "RouteIdentity",
    "BaselineComparison",
    "ForecastBacktestResult",
    "BacktestCase",
    "EvaluationResult",
    "canonical_financial_metric",
    "_contains_actual",
]
