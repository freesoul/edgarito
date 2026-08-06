import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgarito.schemas.identifiers import SecurityIdentifiers


class ForecastAssumptionSource(str, Enum):
    EXPLICIT = "explicit"
    TRAILING_AVERAGE = "trailing_average"


class FreeCashFlowForecastParameters(BaseModel):
    """Inputs for a revenue-times-FCF-margin forecast.

    Rates and margins use percentage points: ``5`` means 5%, not 0.05. A
    one-value path is repeated for every projected year; otherwise it must
    contain exactly one value per year.
    """

    model_config = ConfigDict(frozen=True)

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
    def validate_path_lengths(self) -> "FreeCashFlowForecastParameters":
        for name, path in (
            ("revenue_growth", self.revenue_growth),
            ("free_cash_flow_margin", self.free_cash_flow_margin),
        ):
            if path is not None and len(path) not in (1, self.forecast_years):
                raise ValueError(
                    f"{name} must contain one value or {self.forecast_years} values"
                )
        return self


class FreeCashFlowForecastObservation(BaseModel):
    forecast_year: int
    fiscal_year: int
    period_end: datetime.date
    revenue_growth: Decimal
    revenue: Decimal
    free_cash_flow_margin: Decimal
    free_cash_flow: Decimal
    unit: str
    formula: str = "revenue × free cash flow margin"


class FreeCashFlowForecast(BaseModel):
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

    parameters: FreeCashFlowForecastParameters
    historical_fiscal_years: tuple[int, ...]
    revenue_growth_source: ForecastAssumptionSource
    free_cash_flow_margin_source: ForecastAssumptionSource
    observations: list[FreeCashFlowForecastObservation] = Field(default_factory=list)
