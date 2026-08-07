import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgarito.schemas.normalization.classification import Sector
from edgarito.schemas.valuation.assumptions import ValuationAssumptionSet
from edgarito.services.forecasting.models import AdaptiveMultistagePlan


def _decimal_close(left: Decimal, right: Decimal) -> bool:
    scale = max(abs(left), abs(right), Decimal(1))
    return abs(left - right) <= scale * Decimal("1e-24")


class ValuationModel(str, Enum):
    FCFF_DCF = "fcff_dcf"
    EQUITY_DCF = "equity_dcf"
    RESIDUAL_INCOME = "residual_income"
    NAV_SOTP = "nav_sotp"
    COMPARABLE_MULTIPLES = "comparable_multiples"

    @property
    def label(self) -> str:
        return {
            ValuationModel.FCFF_DCF: "FCFF DCF",
            ValuationModel.EQUITY_DCF: "Equity DCF / DDM",
            ValuationModel.RESIDUAL_INCOME: "Residual Income",
            ValuationModel.NAV_SOTP: "NAV / Sum-of-the-Parts",
            ValuationModel.COMPARABLE_MULTIPLES: "Comparable Multiples",
        }[self]


class BusinessArchetype(str, Enum):
    UNRESOLVED = "unresolved"
    GENERAL_OPERATING = "general_operating"
    FINANCIAL_INTERMEDIARY = "financial_intermediary"
    ASSET_MANAGER = "asset_manager"
    REIT_PROPERTY = "reit_property"
    RESOURCE_PRODUCER = "resource_producer"
    PROJECT_PIPELINE = "project_pipeline"
    HOLDING_COMPANY = "holding_company"
    CONGLOMERATE = "conglomerate"


class CompanyLifecycle(str, Enum):
    PRE_REVENUE = "pre_revenue"
    UNPROFITABLE_GROWTH = "unprofitable_growth"
    GROWTH = "growth"
    MATURE = "mature"
    DECLINING = "declining"
    DISTRESSED = "distressed"
    UNKNOWN = "unknown"


class Cyclicality(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


class EconomicTrait(str, Enum):
    REGULATED_CAPITAL = "regulated_capital"
    LEASE_INTENSIVE = "lease_intensive"
    BACKLOG_DRIVEN = "backlog_driven"
    DIVIDEND_PAYER = "dividend_payer"
    STABLE_PAYOUT = "stable_payout"
    BOOK_VALUE_UNRELIABLE = "book_value_unreliable"
    FINANCING_SUBSIDIARY = "financing_subsidiary"
    MULTI_SEGMENT = "multi_segment"
    PRICING_POWER = "pricing_power"


class ForecastProfile(str, Enum):
    STANDARD = "standard"
    REVENUE_TO_MARGIN = "revenue_to_margin"
    NORMALIZED_CYCLE = "normalized_cycle"
    BACKLOG_DRIVEN = "backlog_driven"
    DIVIDEND_OR_FCFE = "dividend_or_fcfe"
    EXCESS_RETURN = "excess_return"
    ASSET_LEVEL = "asset_level"
    PRODUCT_PIPELINE = "product_pipeline"
    SEGMENT_LEVEL = "segment_level"


class RelativeValuationBasis(str, Enum):
    PE = "price_to_earnings"
    PRICE_TO_BOOK = "price_to_book"
    PRICE_TO_TANGIBLE_BOOK = "price_to_tangible_book"
    PRICE_TO_AFFO = "price_to_affo"
    PRICE_TO_NAV = "price_to_nav"
    EV_TO_REVENUE = "ev_to_revenue"
    EV_TO_EBIT = "ev_to_ebit"
    EV_TO_EBITDA = "ev_to_ebitda"
    EV_TO_FCF = "ev_to_fcf"
    DIVIDEND_YIELD = "dividend_yield"


class ValuationInput(str, Enum):
    CLASSIFICATION = "classification"
    REVENUE_HISTORY = "revenue_history"
    FCF_HISTORY = "fcf_history"
    EARNINGS_HISTORY = "earnings_history"
    BOOK_EQUITY = "book_equity"
    BALANCE_SHEET = "balance_sheet"

    FCFF_FORECAST = "fcff_forecast"
    EQUITY_CASH_FLOW_FORECAST = "equity_cash_flow_forecast"
    DIVIDEND_FORECAST = "dividend_forecast"
    FORECAST_ROE = "forecast_roe"
    WACC = "wacc"
    COST_OF_EQUITY = "cost_of_equity"
    TERMINAL_GROWTH = "terminal_growth"
    NET_DEBT = "net_debt"
    DILUTED_SHARES = "diluted_shares"
    TANGIBLE_BOOK_EQUITY = "tangible_book_equity"

    ASSET_LEVEL_VALUES = "asset_level_values"
    SEGMENT_VALUES = "segment_values"
    AFFO = "affo"
    RESERVE_DATA = "reserve_data"
    PIPELINE_DATA = "pipeline_data"

    PEER_SET = "peer_set"
    PEER_VALUATION_DATA = "peer_valuation_data"
    TARGET_MULTIPLE_METRICS = "target_multiple_metrics"
    MARKET_PRICE = "market_price"


class ModelRole(str, Enum):
    PRIMARY = "primary"
    CONDITIONAL = "conditional"
    CROSSCHECK = "crosscheck"
    NOT_RECOMMENDED = "not_recommended"


class DataReadiness(str, Enum):
    READY = "ready"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class ValuationProfileOverrides(BaseModel):
    """Explicit facts that cannot be inferred reliably from provider labels."""

    model_config = ConfigDict(frozen=True)

    sector: Optional[Sector] = None
    industry: Optional[str] = None
    business_archetype: Optional[BusinessArchetype] = None
    lifecycle: Optional[CompanyLifecycle] = None
    cyclicality: Optional[Cyclicality] = None
    economic_traits: set[EconomicTrait] = Field(default_factory=set)
    available_inputs: set[ValuationInput] = Field(default_factory=set)
    peer_count: Optional[int] = Field(default=None, ge=0)


class ValuationProfile(BaseModel):
    provider: str
    company_id: str
    company_name: str
    ticker: Optional[str] = None
    sector: Optional[Sector] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    exchange: Optional[str] = None
    reporting_currency: Optional[str] = None
    latest_revenue: Optional[Decimal] = None

    business_archetype: BusinessArchetype
    lifecycle: CompanyLifecycle
    cyclicality: Cyclicality
    economic_traits: set[EconomicTrait] = Field(default_factory=set)

    annual_fiscal_years: tuple[int, ...] = ()
    revenue_growth_rates: tuple[Decimal, ...] = ()
    positive_fcf_periods: int = 0
    positive_earnings_periods: int = 0
    latest_book_equity: Optional[Decimal] = None
    available_inputs: set[ValuationInput] = Field(default_factory=set)
    peer_count: Optional[int] = None
    inference_notes: list[str] = Field(default_factory=list)


class ModelSuitability(BaseModel):
    model: ValuationModel
    role: ModelRole
    suitability_score: int = Field(ge=0, le=100)
    data_readiness: DataReadiness
    forecast_profile: Optional[ForecastProfile] = None
    reasons: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    hard_rejections: list[str] = Field(default_factory=list)
    missing_inputs: set[ValuationInput] = Field(default_factory=set)
    relative_bases: tuple[RelativeValuationBasis, ...] = ()


class ValuationSelection(BaseModel):
    profile: ValuationProfile
    models: list[ModelSuitability] = Field(default_factory=list)

    @property
    def primary(self) -> Optional[ModelSuitability]:
        return next(
            (model for model in self.models if model.role == ModelRole.PRIMARY),
            None,
        )


class PeerSelectionParameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_peers: int = Field(default=8, ge=1, le=50)
    preferred_minimum: int = Field(default=5, ge=1, le=50)
    minimum_score: int = Field(default=50, ge=0, le=100)
    require_same_sector: bool = True


class PeerCandidateAssessment(BaseModel):
    ticker: str
    company_id: str
    company_name: str
    score: int = Field(ge=0, le=100)
    selected: bool = False
    reasons: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)


class PeerUniverse(BaseModel):
    target_ticker: str
    target_company_id: str
    parameters: PeerSelectionParameters
    candidates: list[PeerCandidateAssessment] = Field(default_factory=list)
    selected_tickers: tuple[str, ...] = ()
    warnings: list[str] = Field(default_factory=list)


class MultipleStatus(str, Enum):
    COMPUTED = "computed"
    UNAVAILABLE = "unavailable"
    NOT_MEANINGFUL = "not_meaningful"


class TradingMultiple(BaseModel):
    basis: RelativeValuationBasis
    status: MultipleStatus
    value: Optional[Decimal] = None
    unit: str = "multiple"
    numerator: Optional[Decimal] = None
    denominator: Optional[Decimal] = None
    reason: Optional[str] = None


class LtmFundamentals(BaseModel):
    period_start: datetime.date
    period_end: datetime.date
    currency: str
    revenue: Optional[Decimal] = None
    revenue_growth: Optional[Decimal] = None
    operating_income: Optional[Decimal] = None
    depreciation_and_amortization: Optional[Decimal] = None
    ebitda: Optional[Decimal] = None
    net_income: Optional[Decimal] = None
    free_cash_flow: Optional[Decimal] = None
    capital_expenditures: Optional[Decimal] = None
    dividends_paid: Optional[Decimal] = None
    book_equity: Optional[Decimal] = None
    tangible_book_equity: Optional[Decimal] = None
    cash_and_equivalents: Optional[Decimal] = None
    gross_debt: Optional[Decimal] = None
    shares: Optional[Decimal] = None
    share_basis: Optional[str] = None


class CompanyTradingMultiples(BaseModel):
    provider: str
    market_provider: str
    company_id: str
    company_name: str
    ticker: str
    price_date: datetime.date
    price: Decimal
    currency: str
    market_capitalization: Optional[Decimal] = None
    enterprise_value: Optional[Decimal] = None
    fundamentals: LtmFundamentals
    multiples: list[TradingMultiple] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PeerMultipleSummary(BaseModel):
    basis: RelativeValuationBasis
    median: Decimal
    minimum: Decimal
    maximum: Decimal
    sample_size: int = Field(ge=1)


class ComparableMultiplesReport(BaseModel):
    universe: PeerUniverse
    target: CompanyTradingMultiples
    peers: list[CompanyTradingMultiples] = Field(default_factory=list)
    summaries: list[PeerMultipleSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HistoricalMultipleObservation(BaseModel):
    observed_on: datetime.date
    value: Decimal = Field(gt=0)
    fundamentals_period_end: Optional[datetime.date] = None
    price_date: Optional[datetime.date] = None


class HistoricalMultipleSummary(BaseModel):
    basis: RelativeValuationBasis
    observations: tuple[HistoricalMultipleObservation, ...] = ()
    median: Optional[Decimal] = None
    percentile_25: Optional[Decimal] = None
    percentile_75: Optional[Decimal] = None
    minimum: Optional[Decimal] = None
    maximum: Optional[Decimal] = None
    current: Optional[Decimal] = None
    volatility: Optional[Decimal] = None
    trend: Optional[Decimal] = None
    warnings: tuple[str, ...] = ()


class MultipleConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResolvedMultiple(BaseModel):
    """Auditable market multiple with separate fundamental and premium legs."""

    basis: RelativeValuationBasis
    point_estimate: Decimal = Field(gt=0)
    lower_bound: Decimal = Field(gt=0)
    upper_bound: Decimal = Field(gt=0)
    fundamental_anchor: Decimal = Field(gt=0)
    peer_anchor: Optional[Decimal] = None
    historical_anchor: Optional[Decimal] = None
    historical_percentile_25: Optional[Decimal] = None
    historical_percentile_75: Optional[Decimal] = None
    historical_volatility: Optional[Decimal] = None
    historical_trend: Optional[Decimal] = None
    historical_sample_size: int = Field(default=0, ge=0)
    current_target_anchor: Optional[Decimal] = None
    market_anchor: Optional[Decimal] = None
    observed_premium: Optional[Decimal] = None
    resolved_premium: Optional[Decimal] = None
    historical_peer_premium: Optional[Decimal] = None
    premium_history_sample_size: int = Field(default=0, ge=0)
    premium_mean_reversion_beta: Optional[Decimal] = Field(default=None, ge=0, le=1)
    historical_persistence: Decimal = Field(ge=0, le=1)
    fundamental_support: Decimal = Field(ge=0, le=1)
    horizon_retention: Decimal = Field(ge=0, le=1)
    persistence_factor: Decimal = Field(ge=0, le=1)
    sample_size: int = Field(ge=0)
    peer_confidence: MultipleConfidence = MultipleConfidence.LOW
    target_history_confidence: MultipleConfidence = MultipleConfidence.LOW
    premium_persistence_confidence: MultipleConfidence = MultipleConfidence.LOW
    confidence: MultipleConfidence
    methodology: str
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_range(self) -> "ResolvedMultiple":
        if not self.lower_bound <= self.point_estimate <= self.upper_bound:
            raise ValueError(
                "Resolved multiple point estimate must be inside its range"
            )
        return self


class ComparableImpliedValuationCase(BaseModel):
    label: str
    multiple: Decimal = Field(gt=0)
    implied_enterprise_value: Decimal
    implied_equity_value: Decimal
    implied_value_per_share: Decimal
    present_value_per_share: Decimal


class ComparableImpliedValuation(BaseModel):
    provider: str
    company_id: str
    company_name: str
    ticker: Optional[str] = None
    valuation_date: datetime.date
    target_date: datetime.date
    horizon_years: Decimal = Field(gt=0)
    currency: str
    basis: RelativeValuationBasis
    forecast_metric: Decimal
    forecast_metric_label: str
    projected_net_debt: Decimal
    projected_diluted_shares: Decimal = Field(gt=0)
    discount_rate: Decimal
    resolved_multiple: ResolvedMultiple
    lower_case: ComparableImpliedValuationCase
    point_case: ComparableImpliedValuationCase
    upper_case: ComparableImpliedValuationCase
    current_price: Optional[Decimal] = None
    current_price_implied_multiple: Optional[Decimal] = None
    analyst_target_price: Optional[Decimal] = None
    analyst_target_implied_multiple: Optional[Decimal] = None
    intrinsic_value_per_share: Optional[Decimal] = None
    warnings: tuple[str, ...] = ()


class CostOfEquityMethod(str, Enum):
    CAPM = "capm"


class CostOfEquityResult(BaseModel):
    """Auditable CAPM result; rates use percentage points."""

    model_config = ConfigDict(frozen=True)

    method: CostOfEquityMethod = CostOfEquityMethod.CAPM
    risk_free_rate: Decimal
    levered_beta: Decimal
    equity_risk_premium: Decimal
    country_risk_premium: Decimal = Decimal(0)
    cost_of_equity: Decimal
    formula: str = "risk-free rate + beta × equity risk premium + country risk premium"

    @field_validator(
        "risk_free_rate",
        "levered_beta",
        "equity_risk_premium",
        "country_risk_premium",
        "cost_of_equity",
    )
    @classmethod
    def require_finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Cost-of-equity values must be finite")
        return value

    @model_validator(mode="after")
    def validate_calculation(self) -> "CostOfEquityResult":
        expected = (
            self.risk_free_rate
            + self.levered_beta * self.equity_risk_premium
            + self.country_risk_premium
        )
        if not _decimal_close(self.cost_of_equity, expected):
            raise ValueError("cost_of_equity does not match its CAPM components")
        return self


class WaccResult(BaseModel):
    """Market-value weighted cost of capital; rates use percentage points."""

    model_config = ConfigDict(frozen=True)

    cost_of_equity: Decimal
    pretax_cost_of_debt: Decimal
    normalized_tax_rate: Decimal
    after_tax_cost_of_debt: Decimal
    market_value_equity: Decimal
    market_value_debt: Decimal
    equity_weight: Decimal
    debt_weight: Decimal
    wacc: Decimal
    formula: str = (
        "equity weight × cost of equity + debt weight × after-tax cost of debt"
    )

    @field_validator(
        "cost_of_equity",
        "pretax_cost_of_debt",
        "normalized_tax_rate",
        "after_tax_cost_of_debt",
        "market_value_equity",
        "market_value_debt",
        "equity_weight",
        "debt_weight",
        "wacc",
    )
    @classmethod
    def require_finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("WACC values must be finite")
        return value

    @model_validator(mode="after")
    def validate_calculation(self) -> "WaccResult":
        if self.market_value_equity < 0 or self.market_value_debt < 0:
            raise ValueError("Market values cannot be negative")
        if self.market_value_equity + self.market_value_debt <= 0:
            raise ValueError("WACC requires positive total capital")
        if not Decimal(0) <= self.normalized_tax_rate <= Decimal(100):
            raise ValueError("Tax rate must be between 0% and 100%")
        if self.equity_weight < 0 or self.debt_weight < 0:
            raise ValueError("Capital weights cannot be negative")
        if self.equity_weight + self.debt_weight != Decimal(1):
            raise ValueError("Capital weights must sum to one")
        expected_after_tax_debt = self.pretax_cost_of_debt * (
            Decimal(1) - self.normalized_tax_rate / Decimal(100)
        )
        if not _decimal_close(self.after_tax_cost_of_debt, expected_after_tax_debt):
            raise ValueError("after_tax_cost_of_debt does not match its inputs")
        expected_wacc = (
            self.equity_weight * self.cost_of_equity
            + self.debt_weight * self.after_tax_cost_of_debt
        )
        if not _decimal_close(self.wacc, expected_wacc):
            raise ValueError("wacc does not match its capital components")
        return self


class CashFlow(BaseModel):
    """A cash flow occurring at a fractional number of periods from valuation."""

    model_config = ConfigDict(frozen=True)

    amount: Decimal
    period: Decimal = Field(ge=0)
    label: Optional[str] = None

    @field_validator("amount", "period")
    @classmethod
    def require_finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Cash-flow values must be finite")
        return value

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Cash-flow labels cannot be blank")
        return normalized


class DiscountedCashFlow(BaseModel):
    """One discounted cash flow with its complete calculation bridge."""

    model_config = ConfigDict(frozen=True)

    amount: Decimal
    period: Decimal = Field(ge=0)
    discount_rate: Decimal
    discount_factor: Decimal = Field(gt=0)
    present_value: Decimal
    label: Optional[str] = None

    @field_validator(
        "amount", "period", "discount_rate", "discount_factor", "present_value"
    )
    @classmethod
    def require_finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Discounted cash-flow values must be finite")
        return value

    @model_validator(mode="after")
    def validate_present_value(self) -> "DiscountedCashFlow":
        if self.discount_rate <= Decimal("-100"):
            raise ValueError("Discount rate must be greater than -100%")
        if not _decimal_close(self.present_value, self.amount * self.discount_factor):
            raise ValueError("Present value does not match amount × discount factor")
        return self


class PresentValueResult(BaseModel):
    """A collection of consistently discounted cash flows."""

    model_config = ConfigDict(frozen=True)

    discount_rate: Decimal
    unit: str
    cash_flows: tuple[DiscountedCashFlow, ...]
    total_present_value: Decimal

    @field_validator("discount_rate", "total_present_value")
    @classmethod
    def require_finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Present-value values must be finite")
        return value

    @field_validator("unit")
    @classmethod
    def normalize_unit(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Present-value unit cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_cash_flows(self) -> "PresentValueResult":
        if not self.cash_flows:
            raise ValueError("At least one cash flow is required")
        if any(item.discount_rate != self.discount_rate for item in self.cash_flows):
            raise ValueError("All cash flows must use the result discount rate")
        expected = sum((item.present_value for item in self.cash_flows), Decimal(0))
        if not _decimal_close(self.total_present_value, expected):
            raise ValueError("total_present_value does not match its cash flows")
        return self


class TerminalValueMethod(str, Enum):
    PERPETUITY_GROWTH = "perpetuity_growth"
    EXIT_MULTIPLE = "exit_multiple"


class TerminalValueResult(BaseModel):
    """Undiscounted terminal value at the end of an explicit forecast period."""

    model_config = ConfigDict(frozen=True)

    method: TerminalValueMethod
    terminal_value: Decimal
    final_cash_flow: Optional[Decimal] = None
    discount_rate: Optional[Decimal] = None
    perpetual_growth_rate: Optional[Decimal] = None
    terminal_metric: Optional[Decimal] = None
    exit_multiple: Optional[Decimal] = None
    formula: str

    @field_validator(
        "terminal_value",
        "final_cash_flow",
        "discount_rate",
        "perpetual_growth_rate",
        "terminal_metric",
        "exit_multiple",
    )
    @classmethod
    def require_finite(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("Terminal-value inputs must be finite")
        return value

    @model_validator(mode="after")
    def validate_method_inputs(self) -> "TerminalValueResult":
        if self.terminal_value < 0:
            raise ValueError("Terminal value cannot be negative")
        if self.method == TerminalValueMethod.PERPETUITY_GROWTH:
            if None in (
                self.final_cash_flow,
                self.discount_rate,
                self.perpetual_growth_rate,
            ):
                raise ValueError(
                    "Perpetuity growth requires cash flow, rate, and growth"
                )
            if self.terminal_metric is not None or self.exit_multiple is not None:
                raise ValueError(
                    "Perpetuity growth cannot include exit-multiple inputs"
                )
            assert self.final_cash_flow is not None
            assert self.discount_rate is not None
            assert self.perpetual_growth_rate is not None
            if self.final_cash_flow < 0:
                raise ValueError("Final cash flow cannot be negative")
            if self.discount_rate <= self.perpetual_growth_rate:
                raise ValueError("Discount rate must exceed perpetual growth")
            expected = (
                self.final_cash_flow
                * (Decimal(1) + self.perpetual_growth_rate / Decimal(100))
                / ((self.discount_rate - self.perpetual_growth_rate) / Decimal(100))
            )
        else:
            if self.terminal_metric is None or self.exit_multiple is None:
                raise ValueError("Exit multiple requires a metric and multiple")
            if any(
                value is not None
                for value in (
                    self.final_cash_flow,
                    self.discount_rate,
                    self.perpetual_growth_rate,
                )
            ):
                raise ValueError("Exit multiple cannot include perpetuity inputs")
            if self.terminal_metric < 0 or self.exit_multiple < 0:
                raise ValueError("Exit-multiple inputs cannot be negative")
            expected = self.terminal_metric * self.exit_multiple
        if not _decimal_close(self.terminal_value, expected):
            raise ValueError("terminal_value does not match its method inputs")
        return self


class CashFlowTiming(str, Enum):
    END_OF_PERIOD = "end_of_period"
    MID_YEAR = "mid_year"


class TerminalMetric(str, Enum):
    EBITDA = "ebitda"
    EBIT = "ebit"
    FCFF = "fcff"
    REVENUE = "revenue"


class FcffDcfCapitalBridge(BaseModel):
    """Normalized enterprise-to-equity inputs with explicit source labels."""

    model_config = ConfigDict(frozen=True)

    fiscal_year: int
    period_end: datetime.date
    unit: str
    net_debt: Decimal
    diluted_shares: Decimal = Field(gt=0)
    net_debt_source: str
    shares_source: str
    gross_debt: Optional[Decimal] = None
    cash_and_equivalents: Optional[Decimal] = None
    non_operating_assets: Decimal = Decimal(0)
    non_operating_assets_source: str = "none reported"

    @field_validator(
        "net_debt",
        "diluted_shares",
        "gross_debt",
        "cash_and_equivalents",
        "non_operating_assets",
    )
    @classmethod
    def require_finite(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("Capital-bridge values must be finite")
        return value

    @field_validator(
        "unit", "net_debt_source", "shares_source", "non_operating_assets_source"
    )
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Capital-bridge text fields cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_debt_bridge(self) -> "FcffDcfCapitalBridge":
        if self.non_operating_assets < 0:
            raise ValueError("Non-operating assets cannot be negative")
        components = (self.gross_debt, self.cash_and_equivalents)
        if (components[0] is None) != (components[1] is None):
            raise ValueError("Gross debt and cash must be provided together")
        if self.gross_debt is not None and self.cash_and_equivalents is not None:
            if self.gross_debt < 0 or self.cash_and_equivalents < 0:
                raise ValueError("Gross debt and cash cannot be negative")
            expected = self.gross_debt - self.cash_and_equivalents
            if not _decimal_close(self.net_debt, expected):
                raise ValueError("Net debt does not match gross debt minus cash")
        return self


class FcffDcfParameters(BaseModel):
    """FCFF DCF assumptions; rates use percentage points."""

    model_config = ConfigDict(frozen=True)

    wacc: Decimal
    wacc_source: str = "explicit"
    cash_flow_timing: CashFlowTiming = CashFlowTiming.END_OF_PERIOD
    terminal_method: TerminalValueMethod = TerminalValueMethod.PERPETUITY_GROWTH
    perpetual_growth_rate: Optional[Decimal] = None
    perpetual_growth_source: Optional[str] = None
    exit_multiple: Optional[Decimal] = None
    exit_metric: TerminalMetric = TerminalMetric.EBITDA

    @field_validator("wacc", "perpetual_growth_rate", "exit_multiple")
    @classmethod
    def require_finite(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("FCFF DCF assumptions must be finite")
        return value

    @field_validator("wacc_source", "perpetual_growth_source")
    @classmethod
    def normalize_source(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Assumption sources cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_terminal_assumptions(self) -> "FcffDcfParameters":
        if self.wacc <= Decimal("-100"):
            raise ValueError("WACC must be greater than -100%")
        if self.terminal_method == TerminalValueMethod.PERPETUITY_GROWTH:
            if self.perpetual_growth_rate is None:
                raise ValueError("Perpetuity growth requires perpetual_growth_rate")
            if self.exit_multiple is not None:
                raise ValueError("Perpetuity growth cannot include exit_multiple")
            if self.wacc <= self.perpetual_growth_rate:
                raise ValueError("WACC must exceed perpetual growth")
        else:
            if self.exit_multiple is None:
                raise ValueError("Exit-multiple valuation requires exit_multiple")
            if self.exit_multiple < 0:
                raise ValueError("exit_multiple cannot be negative")
            if self.perpetual_growth_rate is not None:
                raise ValueError("Exit multiple cannot include perpetual growth")
        return self


class ShareRepurchaseParameters(BaseModel):
    """Future repurchase cash and execution assumptions."""

    model_config = ConfigDict(frozen=True)

    annual_cash_amounts: tuple[Decimal, ...]
    initial_purchase_price: Optional[Decimal] = None
    price_growth_rate: Optional[Decimal] = None
    discount_rate: Optional[Decimal] = None
    source: str = "explicit profile or CLI assumptions"

    @field_validator("annual_cash_amounts")
    @classmethod
    def validate_cash_amounts(cls, values: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
        if not values:
            raise ValueError("Share repurchases require at least one cash amount")
        if any(not value.is_finite() or value <= 0 for value in values):
            raise ValueError(
                "Share-repurchase cash amounts must be finite and positive"
            )
        return values

    @field_validator("initial_purchase_price", "price_growth_rate", "discount_rate")
    @classmethod
    def validate_optional_number(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("Share-repurchase assumptions must be finite")
        return value

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Share-repurchase source cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_ranges(self) -> "ShareRepurchaseParameters":
        if self.initial_purchase_price is not None and self.initial_purchase_price <= 0:
            raise ValueError("initial_purchase_price must be positive")
        for name in ("price_growth_rate", "discount_rate"):
            value = getattr(self, name)
            if value is not None and value <= Decimal("-100"):
                raise ValueError(f"{name} must be greater than -100%")
        return self


class ShareRepurchasePeriod(BaseModel):
    """One modeled future repurchase distribution."""

    model_config = ConfigDict(frozen=True)

    forecast_year: int = Field(ge=1)
    fiscal_year: int
    period_end: datetime.date
    discount_period: Decimal = Field(ge=0)
    cash_spent: Decimal = Field(gt=0)
    present_value_cash_spent: Decimal = Field(gt=0)
    purchase_price: Decimal = Field(gt=0)
    shares_repurchased: Decimal = Field(gt=0)

    @field_validator(
        "discount_period",
        "cash_spent",
        "present_value_cash_spent",
        "purchase_price",
        "shares_repurchased",
    )
    @classmethod
    def require_finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Share-repurchase period values must be finite")
        return value


class ShareRepurchaseResult(BaseModel):
    """PV-consistent split between tendering and remaining shareholders."""

    model_config = ConfigDict(frozen=True)

    source: str
    discount_rate: Decimal
    discount_rate_source: str
    price_growth_rate: Decimal
    initial_purchase_price: Decimal = Field(gt=0)
    purchase_price_source: str
    starting_shares: Decimal = Field(gt=0)
    ending_shares: Decimal = Field(gt=0)
    shares_repurchased: Decimal = Field(gt=0)
    total_cash_spent: Decimal = Field(gt=0)
    present_value_cash_spent: Decimal = Field(gt=0)
    pre_repurchase_equity_value: Decimal
    residual_equity_value: Decimal
    pre_repurchase_value_per_share: Decimal
    value_per_remaining_share: Decimal
    accretion_percentage: Decimal
    periods: tuple[ShareRepurchasePeriod, ...]

    @field_validator(
        "discount_rate",
        "price_growth_rate",
        "starting_shares",
        "ending_shares",
        "shares_repurchased",
        "total_cash_spent",
        "present_value_cash_spent",
        "pre_repurchase_equity_value",
        "residual_equity_value",
        "pre_repurchase_value_per_share",
        "value_per_remaining_share",
        "accretion_percentage",
    )
    @classmethod
    def require_finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Share-repurchase result values must be finite")
        return value

    @field_validator("source", "discount_rate_source", "purchase_price_source")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Share-repurchase result sources cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_bridge(self) -> "ShareRepurchaseResult":
        if not self.periods:
            raise ValueError("Share-repurchase result requires at least one period")
        if self.discount_rate <= Decimal("-100"):
            raise ValueError("Share-repurchase discount rate must exceed -100%")
        if self.price_growth_rate <= Decimal("-100"):
            raise ValueError("Share-price growth rate must exceed -100%")
        if not _decimal_close(
            self.ending_shares, self.starting_shares - self.shares_repurchased
        ):
            raise ValueError("Ending shares do not match modeled repurchases")
        if not _decimal_close(
            self.residual_equity_value,
            self.pre_repurchase_equity_value - self.present_value_cash_spent,
        ):
            raise ValueError("Residual equity does not reflect buyback cash spent")
        if not _decimal_close(
            self.value_per_remaining_share,
            self.residual_equity_value / self.ending_shares,
        ):
            raise ValueError("Remaining-holder value does not match equity and shares")
        expected_accretion = (
            self.value_per_remaining_share / self.pre_repurchase_value_per_share
            - Decimal(1)
        ) * Decimal(100)
        if not _decimal_close(self.accretion_percentage, expected_accretion):
            raise ValueError("Buyback accretion does not match per-share values")
        return self


class FcffDcfResult(BaseModel):
    """Auditable enterprise-to-equity FCFF DCF result."""

    model_config = ConfigDict(frozen=True)

    provider: str
    company_id: str
    company_name: str
    ticker: Optional[str] = None
    valuation_date: datetime.date
    unit: str
    parameters: FcffDcfParameters
    assumptions: Optional[ValuationAssumptionSet] = None
    multistage_plan: Optional[AdaptiveMultistagePlan] = None
    capital_bridge: FcffDcfCapitalBridge
    explicit_forecast_present_value: PresentValueResult
    terminal_value: TerminalValueResult
    terminal_present_value: DiscountedCashFlow
    enterprise_value: Decimal
    equity_value: Decimal
    value_per_share: Decimal
    share_repurchases: Optional[ShareRepurchaseResult] = None
    terminal_value_percentage: Optional[Decimal] = None
    warnings: tuple[str, ...] = ()

    @field_validator(
        "enterprise_value",
        "equity_value",
        "value_per_share",
        "terminal_value_percentage",
    )
    @classmethod
    def require_finite(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("FCFF DCF result values must be finite")
        return value

    @model_validator(mode="after")
    def validate_value_bridge(self) -> "FcffDcfResult":
        if self.capital_bridge.unit != self.unit:
            raise ValueError("Capital bridge and DCF must use one currency")
        expected_enterprise = (
            self.explicit_forecast_present_value.total_present_value
            + self.terminal_present_value.present_value
        )
        if not _decimal_close(self.enterprise_value, expected_enterprise):
            raise ValueError("Enterprise value does not match discounted cash flows")
        expected_equity = (
            self.enterprise_value
            - self.capital_bridge.net_debt
            + self.capital_bridge.non_operating_assets
        )
        if not _decimal_close(self.equity_value, expected_equity):
            raise ValueError(
                "Equity value does not match enterprise value minus net debt plus "
                "non-operating assets"
            )
        expected_per_share = self.equity_value / self.capital_bridge.diluted_shares
        if not _decimal_close(self.value_per_share, expected_per_share):
            raise ValueError(
                "Per-share value does not match equity value divided by shares"
            )
        expected_percentage = (
            self.terminal_present_value.present_value
            / self.enterprise_value
            * Decimal(100)
            if self.enterprise_value != 0
            else None
        )
        if self.terminal_value_percentage is None:
            if expected_percentage is not None:
                raise ValueError("terminal_value_percentage is required")
        elif expected_percentage is None or not _decimal_close(
            self.terminal_value_percentage, expected_percentage
        ):
            raise ValueError(
                "terminal_value_percentage does not match enterprise value"
            )
        return self
