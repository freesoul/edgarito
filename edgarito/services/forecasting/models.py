import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgarito.schemas.identifiers import SecurityIdentifiers


class ForecastAssumptionSource(str, Enum):
    EXPLICIT = "explicit"
    TRAILING_AVERAGE = "trailing_average"
    ADAPTIVE_MULTISTAGE = "adaptive_multistage"


class ForecastSeedType(str, Enum):
    FISCAL_YEAR = "FY"
    TTM = "TTM"
    YTD_PLUS_FORECAST = "YTD+forecast"


class SimplifiedFcfForecastParameters(BaseModel):
    """Inputs for a revenue-times-FCF-margin forecast.

    Rates and margins use percentage points: ``5`` means 5%, not 0.05. A
    one-value path is repeated for every projected year; otherwise it must
    contain exactly one value per year.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_years: int = Field(default=5, ge=1, le=30)
    revenue_growth: Optional[tuple[Decimal, ...]] = None
    free_cash_flow_margin: Optional[tuple[Decimal, ...]] = None
    historical_window: int = Field(default=3, ge=1, le=10)

    @field_validator("revenue_growth", "free_cash_flow_margin", mode="before")
    @classmethod
    def normalize_path(cls, value):
        if value is None:
            return None
        if isinstance(value, (str, int, float, Decimal)):
            return (value,)
        return tuple(value)

    @field_validator("revenue_growth")
    @classmethod
    def validate_growth_path(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(not rate.is_finite() for rate in value):
            raise ValueError("Revenue growth must contain finite values")
        if value is not None and any(
            rate <= Decimal("-100") or rate > Decimal("1000") for rate in value
        ):
            raise ValueError(
                "Revenue growth must be greater than -100% and at most 1000%"
            )
        return value

    @field_validator("free_cash_flow_margin")
    @classmethod
    def validate_margin_path(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(not margin.is_finite() for margin in value):
            raise ValueError("Free cash flow margin must contain finite values")
        if value is not None and any(abs(margin) > Decimal("500") for margin in value):
            raise ValueError("Free cash flow margin must be between -500% and 500%")
        return value

    @model_validator(mode="after")
    def validate_path_lengths(self) -> "SimplifiedFcfForecastParameters":
        for name, path in (
            ("revenue_growth", self.revenue_growth),
            ("free_cash_flow_margin", self.free_cash_flow_margin),
        ):
            if path is not None and len(path) not in (1, self.forecast_years):
                raise ValueError(
                    f"{name} must contain one value or {self.forecast_years} values"
                )
        return self


class SimplifiedFcfForecastObservation(BaseModel):
    forecast_year: int
    fiscal_year: int
    period_end: datetime.date
    revenue_growth: Decimal
    revenue: Decimal
    free_cash_flow_margin: Decimal
    free_cash_flow: Decimal
    unit: str
    formula: str = "revenue × free cash flow margin"


class SimplifiedFcfForecast(BaseModel):
    provider: str
    company_id: str
    company_name: str
    ticker: Optional[str] = None
    identifiers: Optional[SecurityIdentifiers] = None
    method: str = "revenue_margin"

    base_fiscal_year: int
    base_period_end: datetime.date
    base_revenue: Decimal
    base_free_cash_flow: Decimal
    unit: str

    parameters: SimplifiedFcfForecastParameters
    historical_fiscal_years: tuple[int, ...]
    revenue_growth_source: ForecastAssumptionSource
    free_cash_flow_margin_source: ForecastAssumptionSource
    observations: list[SimplifiedFcfForecastObservation] = Field(default_factory=list)


class FcffForecastDriver(str, Enum):
    REVENUE_GROWTH = "revenue_growth"
    OPERATING_MARGIN = "operating_margin"
    TAX_RATE = "tax_rate"
    DEPRECIATION_TO_REVENUE = "depreciation_to_revenue"
    CAPEX_TO_REVENUE = "capex_to_revenue"
    OPERATING_WORKING_CAPITAL_TO_REVENUE = "operating_working_capital_to_revenue"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


class FcffForecastParameters(BaseModel):
    """Year-specific operating drivers for an unlevered FCFF forecast.

    Every value uses percentage points. A one-value path is repeated for each
    forecast year. A shorter explicit path is extended with its final value;
    omitted paths are inferred from complete annual historical periods.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_years: int = Field(default=5, ge=1, le=30)
    revenue_growth: Optional[tuple[Decimal, ...]] = None
    operating_margin: Optional[tuple[Decimal, ...]] = None
    tax_rate: Optional[tuple[Decimal, ...]] = None
    depreciation_to_revenue: Optional[tuple[Decimal, ...]] = None
    capex_to_revenue: Optional[tuple[Decimal, ...]] = None
    operating_working_capital_to_revenue: Optional[tuple[Decimal, ...]] = None
    historical_window: int = Field(default=3, ge=1, le=10)

    @field_validator(
        "revenue_growth",
        "operating_margin",
        "tax_rate",
        "depreciation_to_revenue",
        "capex_to_revenue",
        "operating_working_capital_to_revenue",
        mode="before",
    )
    @classmethod
    def normalize_path(cls, value):
        if value is None:
            return None
        if isinstance(value, (str, int, float, Decimal)):
            return (value,)
        return tuple(value)

    @field_validator(
        "revenue_growth",
        "operating_margin",
        "tax_rate",
        "depreciation_to_revenue",
        "capex_to_revenue",
        "operating_working_capital_to_revenue",
    )
    @classmethod
    def validate_finite_path(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(not item.is_finite() for item in value):
            raise ValueError("FCFF driver paths must contain finite values")
        return value

    @field_validator("revenue_growth")
    @classmethod
    def validate_growth(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(
            item <= Decimal("-100") or item > Decimal("1000") for item in value
        ):
            raise ValueError(
                "Revenue growth must be greater than -100% and at most 1000%"
            )
        return value

    @field_validator("tax_rate")
    @classmethod
    def validate_tax_rate(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(
            item < Decimal(0) or item > Decimal(100) for item in value
        ):
            raise ValueError("Tax rate must be between 0% and 100%")
        return value

    @field_validator("depreciation_to_revenue", "capex_to_revenue")
    @classmethod
    def validate_nonnegative_ratio(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(
            item < Decimal(0) or item > Decimal(500) for item in value
        ):
            raise ValueError("D&A and capex ratios must be between 0% and 500%")
        return value

    @field_validator("operating_margin", "operating_working_capital_to_revenue")
    @classmethod
    def validate_signed_ratio(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(abs(item) > Decimal(500) for item in value):
            raise ValueError(
                "Operating margin and working-capital ratio must be "
                "between -500% and 500%"
            )
        return value

    @model_validator(mode="after")
    def validate_path_lengths(self) -> "FcffForecastParameters":
        for driver in FcffForecastDriver:
            path = getattr(self, driver.value)
            if path is not None and not 1 <= len(path) <= self.forecast_years:
                raise ValueError(
                    f"{driver.value} must contain between one and "
                    f"{self.forecast_years} values"
                )
        return self


class FcffForecastObservation(BaseModel):
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
    formula: str = (
        "NOPAT + depreciation and amortization - capital expenditures - "
        "change in operating working capital"
    )


class FcffForecast(BaseModel):
    provider: str
    company_id: str
    company_name: str
    ticker: Optional[str] = None
    identifiers: Optional[SecurityIdentifiers] = None
    method: str = "driver_based_fcff"
    seed_type: ForecastSeedType = ForecastSeedType.FISCAL_YEAR
    seed_methodology: str = "Latest complete fiscal year"
    seed_period_end: Optional[datetime.date] = None
    current_fiscal_year: Optional[int] = None
    actual_quarters: int = Field(default=0, ge=0, le=4)

    base_fiscal_year: int
    base_period_end: datetime.date
    base_revenue: Decimal
    base_operating_income: Decimal
    base_tax_rate: Decimal
    base_nopat: Decimal
    base_depreciation_and_amortization: Decimal
    base_capital_expenditures: Decimal
    base_operating_working_capital: Decimal
    base_fcff: Optional[Decimal] = None
    unit: str

    parameters: FcffForecastParameters
    historical_fiscal_years: tuple[int, ...]
    assumption_sources: dict[FcffForecastDriver, ForecastAssumptionSource]
    observations: list[FcffForecastObservation] = Field(default_factory=list)


class AdaptiveMultistagePlan(BaseModel):
    """Stages selected to converge an FCFF projection to perpetual growth."""

    model_config = ConfigDict(frozen=True)

    requested_years: int = Field(ge=1, le=30)
    effective_years: int = Field(ge=1, le=30)
    high_growth_years: int = Field(ge=0, le=30)
    transition_years: int = Field(ge=0, le=30)
    stable_years: int = Field(ge=0, le=30)
    initial_growth_rate: Decimal
    terminal_growth_rate: Decimal
    max_annual_growth_fade: Decimal = Field(gt=0)
    extended_to_stable: bool = False
    explicit_growth_prefix_years: int = Field(default=0, ge=0, le=30)
    terminal_return_on_invested_capital: Optional[Decimal] = None
    terminal_roic_source: Optional[str] = None
    terminal_roic_methodology: Optional[str] = None
    terminal_roic_confidence: Optional[str] = None
    terminal_roic_warnings: tuple[str, ...] = ()
    terminal_reinvestment_rate: Optional[Decimal] = None
    terminal_capex_to_revenue: Optional[Decimal] = None
    depreciable_asset_life_years: Optional[int] = None

    @model_validator(mode="after")
    def validate_stages(self) -> "AdaptiveMultistagePlan":
        if self.effective_years < self.requested_years:
            raise ValueError("effective_years cannot be below requested_years")
        if (
            self.high_growth_years
            + self.explicit_growth_prefix_years
            + self.transition_years
            + self.stable_years
            != self.effective_years
        ):
            raise ValueError("Adaptive stages must span the effective forecast")
        return self


# The generic historical public names now point to the valuation-grade default.
FreeCashFlowForecastParameters = FcffForecastParameters
FreeCashFlowForecastObservation = FcffForecastObservation
FreeCashFlowForecast = FcffForecast
