from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from edgarito.schemas.normalization.classification import Sector


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
