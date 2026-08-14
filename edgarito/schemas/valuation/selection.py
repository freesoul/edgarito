"""Provider-neutral valuation classification and model-selection contracts."""

from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.normalization.classification import Sector


class ValuationModel(str, Enum):
    FCFF_DCF = "fcff_dcf"
    EQUITY_DCF = "equity_dcf"
    DIVIDEND_DISCOUNT = "dividend_discount"
    RESIDUAL_INCOME = "residual_income"
    NAV_SOTP = "nav_sotp"
    COMPARABLE_MULTIPLES = "comparable_multiples"

    @property
    def label(self) -> str:
        return {
            ValuationModel.FCFF_DCF: "FCFF DCF",
            ValuationModel.EQUITY_DCF: "FCFE / Equity DCF",
            ValuationModel.DIVIDEND_DISCOUNT: "Dividend Discount Model",
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


class FinancialInstitutionKind(str, Enum):
    BANK = "bank"
    INSURER = "insurer"
    OTHER = "other"


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


class PeerEvidenceGroup(str, Enum):
    """Deterministic product-economics buckets used for peer evidence."""

    AUTO_OEM = "auto_oem"
    EV_GROWTH = "ev_growth"
    ENERGY_STORAGE = "energy_storage"
    TECHNOLOGY_PLATFORM = "technology_platform"
    GENERAL_OPERATING = "general_operating"


def _normalize_peer_evidence_group(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = (
        (value.value if isinstance(value, PeerEvidenceGroup) else str(value))
        .strip()
        .casefold()
    )
    if not normalized:
        raise ValueError("evidence_group cannot be blank")
    try:
        return PeerEvidenceGroup(normalized).value
    except ValueError as exc:
        supported = ", ".join(item.value for item in PeerEvidenceGroup)
        raise ValueError(f"evidence_group must be one of: {supported}") from exc


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
    PER = "price_to_earnings"
    P_E = "price_to_earnings"
    PRICE_TO_BOOK = "price_to_book"
    PRICE_TO_TANGIBLE_BOOK = "price_to_tangible_book"
    PRICE_TO_AFFO = "price_to_affo"
    PRICE_TO_NAV = "price_to_nav"
    EV_TO_REVENUE = "ev_to_revenue"
    EV_TO_EBIT = "ev_to_ebit"
    EV_TO_EBITDA = "ev_to_ebitda"
    EV_TO_FCF = "ev_to_fcf"
    DIVIDEND_YIELD = "dividend_yield"

    @property
    def label(self) -> str:
        return {
            RelativeValuationBasis.PE: "P/E (PER)",
            RelativeValuationBasis.PRICE_TO_BOOK: "P/B",
            RelativeValuationBasis.PRICE_TO_TANGIBLE_BOOK: "P/TBV",
            RelativeValuationBasis.PRICE_TO_AFFO: "P/AFFO",
            RelativeValuationBasis.PRICE_TO_NAV: "P/NAV",
            RelativeValuationBasis.EV_TO_REVENUE: "EV/Revenue",
            RelativeValuationBasis.EV_TO_EBIT: "EV/EBIT",
            RelativeValuationBasis.EV_TO_EBITDA: "EV/EBITDA",
            RelativeValuationBasis.EV_TO_FCF: "EV/FCF",
            RelativeValuationBasis.DIVIDEND_YIELD: "Dividend Yield",
        }[self]

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None
        normalized = value.strip().casefold()
        aliases = {
            "pe": cls.PE,
            "per": cls.PE,
            "p/e": cls.PE,
            "p-e": cls.PE,
            "p_e": cls.PE,
            "price/earnings": cls.PE,
            "price-earnings": cls.PE,
            "ev/ebitda": cls.EV_TO_EBITDA,
            "ev-ebitda": cls.EV_TO_EBITDA,
            "ev_ebitda": cls.EV_TO_EBITDA,
            "ev/ebit": cls.EV_TO_EBIT,
            "ev-ebit": cls.EV_TO_EBIT,
            "ev_ebit": cls.EV_TO_EBIT,
            "ev/fcf": cls.EV_TO_FCF,
            "ev-fcf": cls.EV_TO_FCF,
            "ev_fcf": cls.EV_TO_FCF,
            "ev/revenue": cls.EV_TO_REVENUE,
            "ev-revenue": cls.EV_TO_REVENUE,
            "ev_revenue": cls.EV_TO_REVENUE,
        }
        return aliases.get(normalized)


class ValuationInput(str, Enum):
    CLASSIFICATION = "classification"
    REVENUE_HISTORY = "revenue_history"
    FCF_HISTORY = "fcf_history"
    EARNINGS_HISTORY = "earnings_history"
    DIVIDEND_HISTORY = "dividend_history"
    BOOK_EQUITY = "book_equity"
    COMMON_EQUITY = "common_equity"
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
    PAYOUT_POLICY = "payout_policy"

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
    financial_institution_kind: Optional[FinancialInstitutionKind] = None
    actuarial_detail_supplied: bool = False
    regulatory_capital_constraints_supplied: bool = False
    lifecycle: Optional[CompanyLifecycle] = None
    cyclicality: Optional[Cyclicality] = None
    economic_traits: set[EconomicTrait] = Field(default_factory=set)
    evidence_group: Optional[str] = None
    available_inputs: set[ValuationInput] = Field(default_factory=set)
    peer_count: Optional[int] = Field(default=None, ge=0)

    @field_validator("evidence_group", mode="before")
    @classmethod
    def normalize_evidence_group(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_peer_evidence_group(value)


class ValuationProfile(BaseModel):
    provider: str
    company_id: str
    company_name: str
    ticker: Optional[str] = None
    identifiers: Optional[SecurityIdentifiers] = None
    sector: Optional[Sector] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    exchange: Optional[str] = None
    reporting_currency: Optional[str] = None
    latest_revenue: Optional[Decimal] = None

    business_archetype: BusinessArchetype
    financial_institution_kind: FinancialInstitutionKind = (
        FinancialInstitutionKind.OTHER
    )
    actuarial_detail_supplied: bool = False
    regulatory_capital_constraints_supplied: bool = False
    lifecycle: CompanyLifecycle
    cyclicality: Cyclicality
    economic_traits: set[EconomicTrait] = Field(default_factory=set)
    evidence_group: Optional[str] = None

    annual_fiscal_years: tuple[int, ...] = ()
    revenue_growth_rates: tuple[Decimal, ...] = ()
    positive_fcf_periods: int = 0
    positive_earnings_periods: int = 0
    latest_book_equity: Optional[Decimal] = None
    available_inputs: set[ValuationInput] = Field(default_factory=set)
    peer_count: Optional[int] = None
    inference_notes: list[str] = Field(default_factory=list)

    @field_validator("evidence_group", mode="before")
    @classmethod
    def normalize_evidence_group(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_peer_evidence_group(value)


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


class MultipleConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


__all__ = [
    "BusinessArchetype",
    "CompanyLifecycle",
    "Cyclicality",
    "DataReadiness",
    "EconomicTrait",
    "FinancialInstitutionKind",
    "ForecastProfile",
    "ModelRole",
    "ModelSuitability",
    "MultipleConfidence",
    "PeerEvidenceGroup",
    "RelativeValuationBasis",
    "ValuationInput",
    "ValuationModel",
    "ValuationProfile",
    "ValuationProfileOverrides",
    "ValuationSelection",
]
