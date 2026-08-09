"""Versioned, provider-neutral objects intended for durable analysis exports.

The models in this module are snapshots, not calculation models.  Conversion
services copy the existing domain objects into these immutable shapes so that a
future renderer can consume one stable contract without changing analysis code.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.enums.provider import ProviderName
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialStatement,
    ObservationDerivationKind,
)
from edgarito.schemas.red_flags import RedFlagCategory, RedFlagSeverity
from edgarito.schemas.valuation.intrinsic import ValuationConfidence, WarningSeverity
from edgarito.services.forecasting.models import (
    FcffForecastDriver,
    ForecastAssumptionSource,
    ForecastSeedType,
)
from edgarito.services.metrics.models import FinancialMetric
from edgarito.services.valuation.models import (
    BusinessArchetype,
    CompanyLifecycle,
    Cyclicality,
    DataReadiness,
    EconomicTrait,
    FinancialInstitutionKind,
    ForecastProfile,
    ModelRole,
    RelativeValuationBasis,
    ValuationInput,
    ValuationModel,
)

from ._utils import snapshot


class ExportSection(str, Enum):
    FINANCIAL_DATA = "financial_data"
    METRICS = "metrics"
    FORECAST = "forecast"
    VALUATION = "valuation"
    RED_FLAGS = "red_flags"
    COMPANY_ANALYSIS = "company_analysis"


class CanonicalSecurityIdentifiers(BaseModel):
    """Detached identifiers with immutable provider and exchange mappings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str | None = None
    isin: str | None = None
    cik: int | None = None
    exchange: str | None = None
    exchange_symbols: tuple[tuple[str, str], ...] = ()
    provider_symbols: tuple[tuple[ProviderName, str], ...] = ()

    @field_validator("exchange_symbols", mode="before")
    @classmethod
    def normalize_exchange_symbols(cls, value) -> tuple[tuple[str, str], ...]:
        items = value.items() if isinstance(value, Mapping) else (value or ())
        normalized = tuple((str(key).strip().upper(), symbol) for key, symbol in items)
        return tuple(sorted(normalized, key=lambda item: item[0]))

    @field_validator("provider_symbols", mode="before")
    @classmethod
    def normalize_provider_symbols(cls, value) -> tuple[tuple[ProviderName, str], ...]:
        items = value.items() if isinstance(value, Mapping) else (value or ())
        normalized = tuple((ProviderName(key), symbol) for key, symbol in items)
        return tuple(sorted(normalized, key=lambda item: item[0].value))

    @model_validator(mode="after")
    def require_identifier(self) -> CanonicalSecurityIdentifiers:
        if not any(
            (
                self.ticker,
                self.isin,
                self.cik,
                self.exchange_symbols,
                self.provider_symbols,
            )
        ):
            raise ValueError("Provide at least one security identifier")
        return self

    @field_serializer("exchange_symbols")
    def serialize_exchange_symbols(
        self, value: tuple[tuple[str, str], ...]
    ) -> dict[str, str]:
        return dict(value)

    @field_serializer("provider_symbols")
    def serialize_provider_symbols(
        self, value: tuple[tuple[ProviderName, str], ...]
    ) -> dict[str, str]:
        return {provider.value: symbol for provider, symbol in value}


class CanonicalExportMetadata(BaseModel):
    """Fields shared by every canonical section snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    section: ExportSection
    schema_version: int = Field(default=1, ge=1)
    company_id: str
    company_name: str
    ticker: str | None = None
    identifiers: CanonicalSecurityIdentifiers | None = None
    provider: str
    generated_at: datetime.datetime | None = None
    as_of: datetime.date | None = None

    @field_validator("generated_at")
    @classmethod
    def require_timezone(
        cls, value: datetime.datetime | None
    ) -> datetime.datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("generated_at must include a timezone")
        return value


class CanonicalFinancialObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    concept: FinancialConcept
    statement: FinancialStatement
    value: Decimal
    unit: str
    granularity: Granularity
    fiscal_year: int
    fiscal_period: FiscalPeriod
    period_start: datetime.date | None = None
    period_end: datetime.date
    provider: str
    taxonomy: str
    source_concept: str
    accession_number: str | None = None
    form: str | None = None
    filed: datetime.date | None = None
    derivation_kind: ObservationDerivationKind | None = None
    derivation: str | None = None


class FinancialDataExport(CanonicalExportMetadata):
    section: Literal[ExportSection.FINANCIAL_DATA] = ExportSection.FINANCIAL_DATA

    retrieved_at: datetime.datetime | None = None
    observations: tuple[CanonicalFinancialObservation, ...] = ()


class CanonicalMetricObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: FinancialMetric
    value: Decimal
    unit: str
    granularity: Granularity
    fiscal_year: int
    fiscal_period: FiscalPeriod
    period_end: datetime.date
    provider: str
    formula: str
    input_concepts: tuple[FinancialConcept, ...] = ()


class MetricsExport(CanonicalExportMetadata):
    section: Literal[ExportSection.METRICS] = ExportSection.METRICS

    observations: tuple[CanonicalMetricObservation, ...] = ()


class SimplifiedFcfForecastParametersExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_years: int
    revenue_growth: tuple[Decimal, ...] | None = None
    free_cash_flow_margin: tuple[Decimal, ...] | None = None
    historical_window: int


class SimplifiedFcfForecastObservationExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_year: int
    fiscal_year: int
    period_end: datetime.date
    revenue_growth: Decimal
    revenue: Decimal
    free_cash_flow_margin: Decimal
    free_cash_flow: Decimal
    unit: str
    formula: str


class SimplifiedFcfForecastExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    company_id: str
    company_name: str
    ticker: str | None = None
    identifiers: CanonicalSecurityIdentifiers | None = None
    method: str
    base_fiscal_year: int
    base_period_end: datetime.date
    base_revenue: Decimal
    base_free_cash_flow: Decimal
    unit: str
    parameters: SimplifiedFcfForecastParametersExport
    historical_fiscal_years: tuple[int, ...]
    revenue_growth_source: ForecastAssumptionSource
    free_cash_flow_margin_source: ForecastAssumptionSource
    observations: tuple[SimplifiedFcfForecastObservationExport, ...] = ()


class FcffForecastParametersExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_years: int
    revenue_growth: tuple[Decimal, ...] | None = None
    operating_margin: tuple[Decimal, ...] | None = None
    tax_rate: tuple[Decimal, ...] | None = None
    depreciation_to_revenue: tuple[Decimal, ...] | None = None
    capex_to_revenue: tuple[Decimal, ...] | None = None
    operating_working_capital_to_revenue: tuple[Decimal, ...] | None = None
    revenue_anchors: tuple[tuple[int, Decimal], ...] = ()
    assumption_source_overrides: tuple[
        tuple[FcffForecastDriver, ForecastAssumptionSource], ...
    ] = ()
    historical_window: int


class FcffForecastObservationExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_year: int
    fiscal_year: int
    period_end: datetime.date
    revenue_growth: Decimal
    revenue: Decimal
    operating_margin: Decimal
    operating_income: Decimal
    tax_rate: Decimal
    nopat: Decimal
    depreciation_to_revenue: Decimal
    depreciation_and_amortization: Decimal
    capex_to_revenue: Decimal
    capital_expenditures: Decimal
    operating_working_capital_to_revenue: Decimal
    operating_working_capital: Decimal
    change_in_operating_working_capital: Decimal
    fcff: Decimal
    unit: str
    formula: str


class FcffForecastExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    company_id: str
    company_name: str
    ticker: str | None = None
    identifiers: CanonicalSecurityIdentifiers | None = None
    method: str
    seed_type: ForecastSeedType
    seed_methodology: str
    seed_period_end: datetime.date | None = None
    current_fiscal_year: int | None = None
    actual_quarters: int
    financial_snapshot_retrieved_at: datetime.datetime | None = None
    availability_mode: str | None = None
    base_fiscal_year: int
    base_period_end: datetime.date
    base_revenue: Decimal
    base_operating_income: Decimal
    base_tax_rate: Decimal
    base_nopat: Decimal
    base_depreciation_and_amortization: Decimal
    base_capital_expenditures: Decimal
    base_operating_working_capital: Decimal
    base_fcff: Decimal | None = None
    unit: str
    parameters: FcffForecastParametersExport
    historical_fiscal_years: tuple[int, ...]
    assumption_sources: tuple[
        tuple[FcffForecastDriver, ForecastAssumptionSource], ...
    ] = ()
    observations: tuple[FcffForecastObservationExport, ...] = ()
    warnings: tuple[str, ...] = ()


class AdaptiveMultistagePlanExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    requested_years: int
    effective_years: int
    high_growth_years: int
    transition_years: int
    stable_years: int
    initial_growth_rate: Decimal
    terminal_growth_rate: Decimal
    max_annual_growth_fade: Decimal
    extended_to_stable: bool
    explicit_growth_prefix_years: int
    terminal_return_on_invested_capital: Decimal | None = None
    terminal_roic_source: str | None = None
    terminal_roic_methodology: str | None = None
    terminal_roic_confidence: str | None = None
    terminal_roic_warnings: tuple[str, ...] = ()
    terminal_reinvestment_rate: Decimal | None = None
    terminal_capex_to_revenue: Decimal | None = None
    depreciable_asset_life_years: int | None = None
    forward_evidence_score: Decimal
    forward_evidence_summary: tuple[str, ...] = ()


class ForecastKind(str, Enum):
    SIMPLIFIED_FCF = "simplified_fcf"
    FCFF = "fcff"


class ForecastExport(CanonicalExportMetadata):
    section: Literal[ExportSection.FORECAST] = ExportSection.FORECAST

    forecast_type: ForecastKind
    forecast: SimplifiedFcfForecastExport | FcffForecastExport
    adaptive_plan: AdaptiveMultistagePlanExport | None = None

    @model_validator(mode="after")
    def validate_forecast_type(self) -> ForecastExport:
        expected_type = {
            ForecastKind.SIMPLIFIED_FCF: SimplifiedFcfForecastExport,
            ForecastKind.FCFF: FcffForecastExport,
        }[self.forecast_type]
        if not isinstance(self.forecast, expected_type):
            raise ValueError(
                f"forecast_type {self.forecast_type.value} requires "
                f"{expected_type.__name__}, got {type(self.forecast).__name__}"
            )
        return self

    @property
    def observations(self) -> tuple[Any, ...]:
        return self.forecast.observations

    @property
    def parameters(self) -> Any:
        return self.forecast.parameters


class ValuationInputProvenanceExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str
    source: str
    methodology: str | None = None
    observed_on: datetime.date | None = None


class ValuationAssumptionExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: Decimal | str
    unit: str | None = None
    source: str


class ForecastSummaryPointExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    period: Decimal
    amount: Decimal
    present_value: Decimal | None = None
    unit: str


class ValuationWarningExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    severity: WarningSeverity
    summary: str
    detail: str | None = None


class ValuationModelResultExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ValuationModel
    adapter: str
    company_id: str
    company_name: str
    ticker: str | None = None
    valuation_date: datetime.date
    currency: str
    equity_value: Decimal
    diluted_shares: Decimal
    value_per_share: Decimal
    assumptions: tuple[ValuationAssumptionExport, ...] = ()
    forecast_summary: tuple[ForecastSummaryPointExport, ...] = ()
    confidence: ValuationConfidence
    warnings: tuple[ValuationWarningExport, ...] = ()
    provenance: tuple[ValuationInputProvenanceExport, ...] = ()
    details: Any = None

    @field_validator("details", mode="before")
    @classmethod
    def freeze_details(cls, value: Any) -> Any:
        return snapshot(value)


class ValuationProfileExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    company_id: str
    company_name: str
    ticker: str | None = None
    identifiers: CanonicalSecurityIdentifiers | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    exchange: str | None = None
    reporting_currency: str | None = None
    latest_revenue: Decimal | None = None
    business_archetype: BusinessArchetype
    financial_institution_kind: FinancialInstitutionKind
    actuarial_detail_supplied: bool
    regulatory_capital_constraints_supplied: bool
    lifecycle: CompanyLifecycle
    cyclicality: Cyclicality
    economic_traits: tuple[EconomicTrait, ...] = ()
    annual_fiscal_years: tuple[int, ...] = ()
    revenue_growth_rates: tuple[Decimal, ...] = ()
    positive_fcf_periods: int
    positive_earnings_periods: int
    latest_book_equity: Decimal | None = None
    available_inputs: tuple[ValuationInput, ...] = ()
    peer_count: int | None = None
    inference_notes: tuple[str, ...] = ()


class ModelSuitabilityExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ValuationModel
    role: ModelRole
    suitability_score: int
    data_readiness: DataReadiness
    forecast_profile: ForecastProfile | None = None
    reasons: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    hard_rejections: tuple[str, ...] = ()
    missing_inputs: tuple[ValuationInput, ...] = ()
    relative_bases: tuple[RelativeValuationBasis, ...] = ()


class ValuationSelectionExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    profile: ValuationProfileExport
    models: tuple[ModelSuitabilityExport, ...] = ()


class ExecutedValuationExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: ModelRole
    suitability: ModelSuitabilityExport
    result: ValuationModelResultExport


class SkippedValuationExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ValuationModel
    role: ModelRole
    readiness: DataReadiness
    missing_inputs: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


class ValuationDispersionExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum_value_per_share: Decimal
    maximum_value_per_share: Decimal
    median_value_per_share: Decimal
    range_as_percent_of_median: Decimal


class ValuationExport(CanonicalExportMetadata):
    section: Literal[ExportSection.VALUATION] = ExportSection.VALUATION

    economic_profile: ValuationProfileExport
    selection: ValuationSelectionExport
    executed_models: tuple[ExecutedValuationExport, ...] = ()
    skipped_models: tuple[SkippedValuationExport, ...] = ()
    relative_cross_checks: tuple[Any, ...] = ()
    dispersion: ValuationDispersionExport | None = None

    @field_validator("relative_cross_checks", mode="before")
    @classmethod
    def freeze_relative_cross_checks(cls, value: Any) -> Any:
        return tuple(snapshot(item) for item in value)


class RedFlagSourceObservationExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    concept: FinancialConcept
    value: Decimal
    unit: str
    granularity: Granularity
    fiscal_year: int
    fiscal_period: FiscalPeriod
    period_end: datetime.date
    provider: str
    source_concept: str


class RedFlagEvidenceExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    value: Decimal
    unit: str
    threshold: Decimal | None = None
    threshold_unit: str | None = None
    comparison: str
    formula: str
    fiscal_year: int
    fiscal_period: FiscalPeriod
    period_end: datetime.date
    granularity: Granularity
    input_concepts: tuple[FinancialConcept, ...] = ()
    source_observations: tuple[RedFlagSourceObservationExport, ...] = ()


class RedFlagExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    category: RedFlagCategory
    severity: RedFlagSeverity
    message: str
    evidence: tuple[RedFlagEvidenceExport, ...] = ()


class RedFlagWarningExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    category: RedFlagCategory | None = None
    message: str
    period: tuple[int, FiscalPeriod] | None = None
    required_concepts: tuple[FinancialConcept, ...] = ()


class RedFlagsExport(CanonicalExportMetadata):
    section: Literal[ExportSection.RED_FLAGS] = ExportSection.RED_FLAGS

    source_schema_version: int = Field(default=1, ge=1)
    granularity: Granularity
    configuration_name: str
    evaluated_periods: tuple[tuple[int, FiscalPeriod], ...] = ()
    flags: tuple[RedFlagExport, ...] = ()
    warnings: tuple[RedFlagWarningExport, ...] = ()


class CompanyAnalysisReport(CanonicalExportMetadata):
    """Optional composition of the independent canonical analysis sections."""

    section: Literal[ExportSection.COMPANY_ANALYSIS] = ExportSection.COMPANY_ANALYSIS

    financial_data: FinancialDataExport | None = None
    metrics: MetricsExport | None = None
    forecast: ForecastExport | None = None
    valuation: ValuationExport | None = None
    red_flags: RedFlagsExport | None = None


# Short aliases keep the public package pleasant to use while retaining the
# explicit ``Canonical`` names for callers that want to distinguish export types.
FinancialObservationExport = CanonicalFinancialObservation
MetricObservationExport = CanonicalMetricObservation
NormalizedFinancialDataExport = FinancialDataExport
CanonicalFinancialDataExport = FinancialDataExport
CanonicalMetrics = MetricsExport
CanonicalForecast = ForecastExport
CanonicalValuation = ValuationExport
CanonicalRedFlags = RedFlagsExport
CanonicalMetricsExport = MetricsExport
CanonicalForecastExport = ForecastExport
CanonicalValuationExport = ValuationExport
CanonicalRedFlagsExport = RedFlagsExport
