import datetime
import re
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")


class ValuationAssumptionKind(str, Enum):
    RISK_FREE_RATE = "risk_free_rate"
    EQUITY_RISK_PREMIUM = "equity_risk_premium"
    COUNTRY_RISK_PREMIUM = "country_risk_premium"
    UNLEVERED_BETA = "unlevered_beta"
    LEVERED_BETA = "levered_beta"
    PRETAX_COST_OF_DEBT = "pretax_cost_of_debt"
    COST_OF_EQUITY = "cost_of_equity"
    WACC = "wacc"
    NORMALIZED_TAX_RATE = "normalized_tax_rate"
    REVENUE_GROWTH = "revenue_growth"
    OPERATING_MARGIN = "operating_margin"
    DEPRECIATION_TO_REVENUE = "depreciation_to_revenue"
    CAPEX_TO_REVENUE = "capex_to_revenue"
    OPERATING_WORKING_CAPITAL_TO_REVENUE = "operating_working_capital_to_revenue"
    ROIC = "roic"
    TERMINAL_GROWTH = "terminal_growth"
    TERMINAL_ROIC = "terminal_roic"


class AssumptionUnit(str, Enum):
    PERCENTAGE_POINTS = "percentage_points"
    MULTIPLE = "multiple"


class AssumptionOrigin(str, Enum):
    EXPLICIT = "explicit"
    MARKET_OBSERVATION = "market_observation"
    REFERENCE_DATASET = "reference_dataset"
    HISTORICAL_METRIC = "historical_metric"
    DERIVED = "derived"
    REASONED_ASSUMPTION = "reasoned_assumption"
    MODEL_ASSUMPTION = "model_assumption"


class ValuationScenario(str, Enum):
    BASE = "base"
    DOWNSIDE = "downside"
    UPSIDE = "upside"
    CUSTOM = "custom"


_MULTIPLE_ASSUMPTIONS = {
    ValuationAssumptionKind.UNLEVERED_BETA,
    ValuationAssumptionKind.LEVERED_BETA,
}


class AssumptionProvenance(BaseModel):
    """Auditable origin of one selected assumption value."""

    model_config = ConfigDict(frozen=True)

    origin: AssumptionOrigin
    provider: Optional[str] = None
    dataset: Optional[str] = None
    series_id: Optional[str] = None
    version: Optional[str] = None
    observed_on: Optional[datetime.date] = None
    retrieved_at: Optional[datetime.datetime] = None
    methodology: Optional[str] = None
    assumption_id: Optional[str] = None
    evidence_ids: tuple[str, ...] = ()
    model: Optional[str] = None
    prompt_hash: Optional[str] = None
    prompt_version: Optional[str] = None
    schema_version: Optional[str] = None
    validator_version: Optional[str] = None

    @field_validator(
        "provider",
        "dataset",
        "series_id",
        "version",
        "methodology",
        "assumption_id",
        "model",
        "prompt_hash",
        "prompt_version",
        "schema_version",
        "validator_version",
    )
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Provenance text fields cannot be blank")
        return normalized

    @field_validator("evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(str(item).strip() for item in value if str(item).strip())
        if len(normalized) != len(set(normalized)):
            raise ValueError("Assumption provenance evidence IDs must be unique")
        return normalized

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(
        cls, value: Optional[datetime.datetime]
    ) -> Optional[datetime.datetime]:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("retrieved_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_external_source(self) -> "AssumptionProvenance":
        if self.origin == AssumptionOrigin.MARKET_OBSERVATION and not (
            self.provider and self.observed_on
        ):
            raise ValueError("Market assumptions require provider and observed_on")
        if self.origin == AssumptionOrigin.REFERENCE_DATASET and not (
            self.provider and self.dataset and self.version and self.observed_on
        ):
            raise ValueError(
                "Reference assumptions require provider, dataset, version, and "
                "observed_on"
            )
        return self


class ValuationAssumption(BaseModel):
    """One selected scalar assumption, optionally scoped to a forecast year."""

    model_config = ConfigDict(frozen=True)

    kind: ValuationAssumptionKind
    value: Decimal
    unit: AssumptionUnit
    selected_on: datetime.date
    provenance: AssumptionProvenance
    forecast_year: Optional[int] = Field(default=None, ge=1, le=100)
    currency: Optional[str] = None
    country: Optional[str] = None
    industry: Optional[str] = None
    company_id: Optional[str] = None
    rationale: Optional[str] = None

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Assumption values must be finite")
        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not _CURRENCY_PATTERN.fullmatch(normalized):
            raise ValueError("Currency must be a three-letter ISO code")
        return normalized

    @field_validator("country")
    @classmethod
    def normalize_country(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("Country cannot be blank")
        return normalized

    @field_validator("industry", "company_id", "rationale")
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Assumption text fields cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_selected_date(self) -> "ValuationAssumption":
        observed_on = self.provenance.observed_on
        if observed_on is not None and observed_on > self.selected_on:
            raise ValueError("An assumption cannot select a future observation")
        expected_unit = (
            AssumptionUnit.MULTIPLE
            if self.kind in _MULTIPLE_ASSUMPTIONS
            else AssumptionUnit.PERCENTAGE_POINTS
        )
        if self.unit != expected_unit:
            raise ValueError(
                f"{self.kind.value} must use the {expected_unit.value} unit"
            )
        return self

    @property
    def key(self) -> tuple:
        return (
            self.kind,
            self.forecast_year,
            self.currency,
            self.country,
            self.industry,
            self.company_id,
        )


class ValuationAssumptionSet(BaseModel):
    """Scenario-specific assumptions selected as of one valuation date."""

    model_config = ConfigDict(frozen=True)

    valuation_date: datetime.date
    currency: str
    scenario: ValuationScenario = ValuationScenario.BASE
    assumptions: tuple[ValuationAssumption, ...]
    name: Optional[str] = None

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not _CURRENCY_PATTERN.fullmatch(normalized):
            raise ValueError("Currency must be a three-letter ISO code")
        return normalized

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Assumption-set name cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_assumptions(self) -> "ValuationAssumptionSet":
        if not self.assumptions:
            raise ValueError("An assumption set cannot be empty")
        keys = [assumption.key for assumption in self.assumptions]
        if len(keys) != len(set(keys)):
            raise ValueError("Assumption keys must be unique within a scenario")
        for assumption in self.assumptions:
            if assumption.selected_on > self.valuation_date:
                raise ValueError("Assumptions cannot be selected after valuation_date")
            if assumption.currency not in (None, self.currency):
                raise ValueError(
                    "Assumption currency must match the valuation currency"
                )
        return self

    def find(
        self,
        kind: ValuationAssumptionKind,
        *,
        forecast_year: Optional[int] = None,
        currency: Optional[str] = None,
        country: Optional[str] = None,
        industry: Optional[str] = None,
        company_id: Optional[str] = None,
    ) -> Optional[ValuationAssumption]:
        normalized_currency = currency.strip().upper() if currency else None
        normalized_country = country.strip().upper() if country else None
        matches = [
            assumption
            for assumption in self.assumptions
            if assumption.kind == kind
            and assumption.forecast_year == forecast_year
            and (
                normalized_currency is None
                or assumption.currency == normalized_currency
            )
            and (normalized_country is None or assumption.country == normalized_country)
            and (industry is None or assumption.industry == industry.strip())
            and (company_id is None or assumption.company_id == company_id.strip())
        ]
        if len(matches) > 1:
            raise ValueError(
                "Assumption lookup is ambiguous; specify a more precise scope"
            )
        return matches[0] if matches else None

    def require(
        self,
        kind: ValuationAssumptionKind,
        *,
        forecast_year: Optional[int] = None,
        currency: Optional[str] = None,
        country: Optional[str] = None,
        industry: Optional[str] = None,
        company_id: Optional[str] = None,
    ) -> ValuationAssumption:
        assumption = self.find(
            kind,
            forecast_year=forecast_year,
            currency=currency,
            country=country,
            industry=industry,
            company_id=company_id,
        )
        if assumption is None:
            qualifier = f" for forecast year {forecast_year}" if forecast_year else ""
            raise ValueError(f"Missing {kind.value} assumption{qualifier}")
        return assumption


__all__ = [
    "AssumptionOrigin",
    "AssumptionProvenance",
    "AssumptionUnit",
    "ValuationAssumption",
    "ValuationAssumptionKind",
    "ValuationAssumptionSet",
    "ValuationScenario",
]
