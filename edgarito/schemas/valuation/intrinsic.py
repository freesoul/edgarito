"""Provider-neutral inputs and results for intrinsic valuation models."""

import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgarito.schemas.valuation.selection import (
    DataReadiness,
    ModelRole,
    ModelSuitability,
    ValuationModel,
    ValuationProfile,
    ValuationSelection,
)


class ValuationConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WarningSeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ModelWarning(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    severity: WarningSeverity
    summary: str
    detail: str | None = None


class InputProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    source: str
    methodology: str | None = None
    observed_on: datetime.date | None = None


class ResolvedModelAssumption(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    value: Decimal | str
    unit: str | None = None
    source: str


class ForecastSummaryPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    period: Decimal = Field(ge=0)
    amount: Decimal
    present_value: Decimal | None = None
    unit: str


class IntrinsicValuationContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    company_id: str
    company_name: str
    ticker: str | None = None
    valuation_date: datetime.date
    currency: str
    diluted_shares: Decimal = Field(gt=0)
    confidence: ValuationConfidence = ValuationConfidence.MEDIUM
    provenance: tuple[InputProvenance, ...] = ()


class FcfeForecastPeriod(BaseModel):
    """One FCFE period, reconciled for either a corporate or regulated firm."""

    model_config = ConfigDict(frozen=True)

    label: str
    period: Decimal = Field(gt=0)
    net_income: Decimal | None = None
    fcfe: Decimal
    depreciation_and_amortization: Decimal | None = None
    capital_expenditures: Decimal | None = None
    working_capital_change: Decimal | None = None
    net_borrowing: Decimal | None = None
    required_common_equity_change: Decimal | None = None
    explicit_fcfe: bool = False

    @model_validator(mode="after")
    def validate_reconciliation(self) -> "FcfeForecastPeriod":
        if self.explicit_fcfe:
            if any(
                item is not None
                for item in (
                    self.depreciation_and_amortization,
                    self.capital_expenditures,
                    self.working_capital_change,
                    self.net_borrowing,
                    self.required_common_equity_change,
                )
            ):
                raise ValueError("Explicit FCFE cannot include reconciliation fields")
            return self
        if self.net_income is None:
            raise ValueError("Reconciled FCFE requires net income")
        if self.required_common_equity_change is not None:
            if any(
                item is not None
                for item in (
                    self.depreciation_and_amortization,
                    self.capital_expenditures,
                    self.working_capital_change,
                    self.net_borrowing,
                )
            ):
                raise ValueError(
                    "Regulated FCFE cannot include corporate reinvestment or debt fields"
                )
            expected = self.net_income - self.required_common_equity_change
        else:
            items = (
                self.depreciation_and_amortization,
                self.capital_expenditures,
                self.working_capital_change,
                self.net_borrowing,
            )
            if any(item is None for item in items):
                raise ValueError(
                    "Corporate FCFE requires every reconciliation component"
                )
            expected = (
                self.net_income
                + self.depreciation_and_amortization  # type: ignore[operator]
                - self.capital_expenditures  # type: ignore[operator]
                - self.working_capital_change  # type: ignore[operator]
                + self.net_borrowing  # type: ignore[operator]
            )
        if self.fcfe != expected:
            raise ValueError("FCFE does not match its period reconciliation")
        return self


class FcfeDcfDetails(BaseModel):
    model_config = ConfigDict(frozen=True)

    periods: tuple[FcfeForecastPeriod, ...]
    explicit_fcfe_present_value: Decimal
    terminal_fcfe: Decimal
    terminal_value: Decimal
    terminal_present_value: Decimal
    cost_of_equity: Decimal
    terminal_growth_rate: Decimal
    terminal_return_on_equity: Decimal
    terminal_retention_ratio: Decimal
    regulated_financial: bool = False


class FcfeDcfInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: IntrinsicValuationContext
    periods: tuple[FcfeForecastPeriod, ...]
    cost_of_equity: Decimal
    terminal_growth_rate: Decimal
    terminal_return_on_equity: Decimal
    terminal_net_income: Decimal | None = None
    regulated_financial: bool = False

    @model_validator(mode="after")
    def validate_periods(self) -> "FcfeDcfInput":
        if not self.periods:
            raise ValueError("FCFE DCF requires at least one forecast period")
        expected = tuple(Decimal(index) for index in range(1, len(self.periods) + 1))
        if tuple(period.period for period in self.periods) != expected:
            raise ValueError("FCFE periods must be consecutive annual periods")
        regulated = all(
            period.required_common_equity_change is not None for period in self.periods
        )
        explicit = all(period.explicit_fcfe for period in self.periods)
        if not explicit and regulated != self.regulated_financial:
            raise ValueError("FCFE period type must match regulated_financial")
        if explicit and self.terminal_net_income is None:
            raise ValueError("Explicit FCFE paths require terminal_net_income")
        return self


class DividendForecastPeriod(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    period: Decimal = Field(gt=0)
    earnings: Decimal | None = None
    dividends: Decimal
    payout_ratio: Decimal | None = None

    @model_validator(mode="after")
    def validate_payout(self) -> "DividendForecastPeriod":
        if self.payout_ratio is not None:
            if self.earnings is None:
                raise ValueError("A payout ratio requires earnings")
            if self.dividends != self.earnings * self.payout_ratio:
                raise ValueError("Dividends must equal earnings multiplied by payout")
        return self


class DividendDiscountDetails(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str
    periods: tuple[DividendForecastPeriod, ...] = ()
    explicit_dividend_present_value: Decimal
    terminal_dividend: Decimal
    terminal_value: Decimal
    terminal_present_value: Decimal
    cost_of_equity: Decimal
    terminal_growth_rate: Decimal
    terminal_return_on_equity: Decimal
    terminal_payout_ratio: Decimal
    terminal_retention_ratio: Decimal


class DividendDiscountInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: IntrinsicValuationContext
    mode: str
    cost_of_equity: Decimal
    terminal_growth_rate: Decimal | None = None
    terminal_return_on_equity: Decimal | None = None
    terminal_payout_ratio: Decimal | None = None
    periods: tuple[DividendForecastPeriod, ...] = ()
    next_dividend: Decimal | None = None
    terminal_earnings: Decimal | None = None
    distributable_fcfe: Decimal | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> "DividendDiscountInput":
        if self.mode not in {"gordon", "multistage"}:
            raise ValueError("DDM mode must be gordon or multistage")
        if self.mode == "gordon" and self.next_dividend is None:
            raise ValueError("Gordon DDM requires next_dividend")
        if self.mode == "multistage" and not self.periods:
            raise ValueError("Multistage DDM requires explicit periods")
        supplied = sum(
            value is not None
            for value in (
                self.terminal_growth_rate,
                self.terminal_return_on_equity,
                self.terminal_payout_ratio,
            )
        )
        if supplied < 2:
            raise ValueError(
                "DDM requires any two terminal ROE, payout, and growth inputs"
            )
        return self


class ResidualIncomePeriod(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    period: int = Field(gt=0)
    opening_book_value: Decimal
    return_on_equity: Decimal
    net_income: Decimal
    payout_ratio: Decimal
    dividends: Decimal
    equity_charge: Decimal
    residual_income: Decimal
    ending_book_value: Decimal
    transition: bool = False

    @model_validator(mode="after")
    def validate_roll_forward(self) -> "ResidualIncomePeriod":
        if self.net_income != (
            self.opening_book_value * self.return_on_equity / Decimal(100)
        ):
            raise ValueError("Net income does not match opening book value times ROE")
        if self.dividends != self.net_income * self.payout_ratio:
            raise ValueError("Dividends do not match earnings times payout")
        if self.residual_income != self.net_income - self.equity_charge:
            raise ValueError(
                "Residual income does not match earnings less equity charge"
            )
        if (
            self.ending_book_value
            != self.opening_book_value + self.net_income - self.dividends
        ):
            raise ValueError("Ending book value does not roll forward")
        return self


class ResidualIncomeDetails(BaseModel):
    model_config = ConfigDict(frozen=True)

    starting_book_value: Decimal
    book_value_basis: str
    periods: tuple[ResidualIncomePeriod, ...]
    residual_income_present_value: Decimal
    transition_years: int = Field(ge=0, le=20)
    excess_return_persistence: Decimal = Field(ge=0, le=1)
    cost_of_equity: Decimal


class ResidualIncomeInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: IntrinsicValuationContext
    starting_book_value: Decimal = Field(gt=0)
    book_value_basis: str
    return_on_equity_path: tuple[Decimal, ...]
    payout_ratio_path: tuple[Decimal, ...]
    cost_of_equity: Decimal
    excess_return_persistence: Decimal = Field(default=Decimal("0.75"), ge=0, le=1)

    @model_validator(mode="after")
    def validate_paths(self) -> "ResidualIncomeInput":
        if not self.return_on_equity_path:
            raise ValueError("Residual income requires an explicit ROE path")
        if len(self.return_on_equity_path) != len(self.payout_ratio_path):
            raise ValueError(
                "Residual-income ROE and payout paths must have equal length"
            )
        if any(value < 0 or value > 1 for value in self.payout_ratio_path):
            raise ValueError("Payout ratios must be between zero and one")
        return self


class ComponentValuationMethod(str, Enum):
    DCF = "dcf"
    MULTIPLE = "multiple"
    ASSET_NAV = "asset_nav"
    RNPV = "rnpv"
    MARKET_VALUE = "market_value"
    SPECIALIZED_ADAPTER = "specialized_adapter"


class ComponentValueBasis(str, Enum):
    ENTERPRISE = "enterprise"
    EQUITY = "equity"


class SotpComponent(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    method: ComponentValuationMethod
    value_basis: ComponentValueBasis
    value: Decimal
    currency: str
    ownership: Decimal = Field(default=Decimal(1), ge=0, le=1)
    component_net_debt: Decimal = Decimal(0)
    fx_rate_to_reporting_currency: Decimal = Field(default=Decimal(1), gt=0)
    fx_rate_date: datetime.date
    fx_rate_source: str
    included_balance_sheet_items: frozenset[str] = frozenset()
    provenance: tuple[InputProvenance, ...] = ()


class SotpAdjustmentKind(str, Enum):
    NON_OPERATING_ASSET = "non_operating_asset"
    CORPORATE_DEBT = "corporate_debt"
    OTHER_LIABILITY = "other_liability"
    MINORITY_INTEREST = "minority_interest"


class SotpAdjustment(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: SotpAdjustmentKind
    amount: Decimal = Field(ge=0)
    currency: str
    fx_rate_to_reporting_currency: Decimal = Field(default=Decimal(1), gt=0)
    fx_rate_date: datetime.date
    fx_rate_source: str
    balance_sheet_item: str | None = None
    provenance: tuple[InputProvenance, ...] = ()


class ValuedSotpComponent(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    method: ComponentValuationMethod
    reported_value: Decimal
    component_net_debt: Decimal
    equity_value_before_ownership: Decimal
    ownership: Decimal
    owned_equity_value: Decimal
    reporting_currency: str


class SotpDetails(BaseModel):
    model_config = ConfigDict(frozen=True)

    components: tuple[ValuedSotpComponent, ...]
    owned_gross_asset_value: Decimal
    non_operating_assets: Decimal
    corporate_debt: Decimal
    other_liabilities: Decimal
    minority_interests: Decimal
    pre_discount_equity_value: Decimal
    holding_company_discount: Decimal = Field(ge=0, lt=1)


class SotpValuationInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: IntrinsicValuationContext
    components: tuple[SotpComponent, ...]
    adjustments: tuple[SotpAdjustment, ...] = ()
    holding_company_discount: Decimal = Field(default=Decimal(0), ge=0, lt=1)
    adapter: str = "generic SOTP"

    @model_validator(mode="after")
    def validate_components(self) -> "SotpValuationInput":
        if not self.components:
            raise ValueError("SOTP requires at least one component")
        return self


class PropertyAsset(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    noi: Decimal
    cap_rate: Decimal = Field(gt=0)
    ownership: Decimal = Field(default=Decimal(1), ge=0, le=1)
    currency: str
    provenance: tuple[InputProvenance, ...]


class ResourceProjectYear(BaseModel):
    model_config = ConfigDict(frozen=True)

    year: int = Field(gt=0)
    production: Decimal = Field(ge=0)
    commodity_price: Decimal
    operating_costs: Decimal = Field(ge=0)
    sustaining_capex: Decimal = Field(ge=0)
    development_capex: Decimal = Field(ge=0)
    taxes_and_royalties: Decimal = Field(ge=0)
    closure_costs: Decimal = Field(ge=0)

    @property
    def cash_flow(self) -> Decimal:
        return (
            self.production * self.commodity_price
            - self.operating_costs
            - self.sustaining_capex
            - self.development_capex
            - self.taxes_and_royalties
            - self.closure_costs
        )


class ResourceProject(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    scenario: str
    reserves: Decimal = Field(ge=0)
    discount_rate: Decimal
    currency: str
    years: tuple[ResourceProjectYear, ...]
    ownership: Decimal = Field(default=Decimal(1), ge=0, le=1)
    provenance: tuple[InputProvenance, ...]

    @model_validator(mode="after")
    def validate_finite_schedule(self) -> "ResourceProject":
        if not self.years:
            raise ValueError("A resource project requires a finite annual schedule")
        if tuple(year.year for year in self.years) != tuple(
            range(1, len(self.years) + 1)
        ):
            raise ValueError(
                "Resource project years must be consecutive and start at one"
            )
        production = sum((year.production for year in self.years), Decimal(0))
        if production > self.reserves:
            raise ValueError("Cumulative production cannot exceed project reserves")
        return self


class PipelineProjectYear(BaseModel):
    model_config = ConfigDict(frozen=True)

    year: int = Field(gt=0)
    development_cost: Decimal = Field(ge=0)
    success_cash_flow: Decimal = Field(ge=0)


class PipelineProject(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    success_probability: Decimal = Field(ge=0, le=1)
    success_probability_provenance: InputProvenance
    discount_rate: Decimal
    currency: str
    years: tuple[PipelineProjectYear, ...]
    ownership: Decimal = Field(default=Decimal(1), ge=0, le=1)
    provenance: tuple[InputProvenance, ...] = ()

    @model_validator(mode="after")
    def validate_schedule(self) -> "PipelineProject":
        if not self.years:
            raise ValueError("A pipeline project requires an explicit annual schedule")
        if tuple(year.year for year in self.years) != tuple(
            range(1, len(self.years) + 1)
        ):
            raise ValueError(
                "Pipeline project years must be consecutive and start at one"
            )
        if self.success_probability_provenance.field != "success_probability":
            raise ValueError("Success probability requires field-specific provenance")
        return self


DetailsT = TypeVar("DetailsT")


class IntrinsicValuationResult(BaseModel, Generic[DetailsT]):
    model_config = ConfigDict(frozen=True)

    model: ValuationModel
    adapter: str
    company_id: str
    company_name: str
    ticker: str | None = None
    valuation_date: datetime.date
    currency: str
    equity_value: Decimal
    diluted_shares: Decimal = Field(gt=0)
    value_per_share: Decimal
    assumptions: tuple[ResolvedModelAssumption, ...] = ()
    forecast_summary: tuple[ForecastSummaryPoint, ...] = ()
    confidence: ValuationConfidence
    warnings: tuple[ModelWarning, ...] = ()
    provenance: tuple[InputProvenance, ...] = ()
    details: DetailsT

    @model_validator(mode="after")
    def validate_per_share(self) -> "IntrinsicValuationResult[DetailsT]":
        if self.value_per_share != self.equity_value / self.diluted_shares:
            raise ValueError(
                "Value per share must equal equity value divided by shares"
            )
        return self


class ExecutedValuation(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: ModelRole
    suitability: ModelSuitability
    result: IntrinsicValuationResult[Any]


class SkippedValuation(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: ValuationModel
    role: ModelRole
    readiness: DataReadiness
    missing_inputs: frozenset[str] = frozenset()
    reasons: tuple[str, ...] = ()


class ValuationDispersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    minimum_value_per_share: Decimal
    maximum_value_per_share: Decimal
    median_value_per_share: Decimal
    range_as_percent_of_median: Decimal


class ValuationRunResult(BaseModel):
    """Independent model outputs. Deliberately has no blended-value field."""

    model_config = ConfigDict(frozen=True)

    economic_profile: ValuationProfile
    selection: ValuationSelection
    executed_models: tuple[ExecutedValuation, ...] = ()
    skipped_models: tuple[SkippedValuation, ...] = ()
    relative_cross_checks: tuple[Any, ...] = ()
    dispersion: ValuationDispersion | None = None
