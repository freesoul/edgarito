import datetime
from decimal import Decimal
from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from edgarito.schemas.identifiers import SecurityIdentifiers


class YahooFinancialReport(BaseModel):
    """One Yahoo statement column with its original line-item names."""

    model_config = ConfigDict(frozen=True)

    period_end: datetime.date
    values: dict[str, Decimal] = Field(default_factory=dict)


class YahooCompanyFinancials(BaseModel):
    """Serializable snapshot of the yfinance financial-statement tables."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    company_name: str
    identifiers: Optional[SecurityIdentifiers] = None
    currency: str
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    beta: Optional[Decimal] = None
    market_capitalization: Optional[Decimal] = None
    # Optional while pre-provenance cache entries remain readable. Every newly
    # retrieved Yahoo snapshot supplies this timestamp.
    retrieved_at: Optional[datetime.datetime] = None
    annual_income_statements: tuple[YahooFinancialReport, ...] = ()
    quarterly_income_statements: tuple[YahooFinancialReport, ...] = ()
    annual_balance_sheets: tuple[YahooFinancialReport, ...] = ()
    quarterly_balance_sheets: tuple[YahooFinancialReport, ...] = ()
    annual_cash_flow_statements: tuple[YahooFinancialReport, ...] = ()
    quarterly_cash_flow_statements: tuple[YahooFinancialReport, ...] = ()

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(
        cls, value: Optional[datetime.datetime]
    ) -> Optional[datetime.datetime]:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("retrieved_at must include a timezone")
        return value


class YahooRevenueEstimateRow(BaseModel):
    """Raw annual/current-period row returned by yfinance analysis data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    period: str
    average: Optional[Decimal] = Field(
        default=None,
        validation_alias=AliasChoices("average", "avg", "value"),
    )
    low: Optional[Decimal] = None
    high: Optional[Decimal] = None
    analyst_count: Optional[int] = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "analyst_count", "numberOfAnalysts", "number_of_analysts"
        ),
    )
    currency: Optional[str] = None

    @field_validator("average", "low", "high")
    @classmethod
    def require_finite_amount(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and (not value.is_finite() or value < 0):
            raise ValueError("Yahoo revenue estimates must be finite and non-negative")
        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: Optional[str]) -> Optional[str]:
        return value.strip().upper() if value else None


class YahooRevenueEstimateResponse(BaseModel):
    """Serializable cache value for Yahoo annual revenue estimates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    rows: tuple[YahooRevenueEstimateRow, ...] = ()
    retrieved_at: Optional[datetime.datetime] = None
    source_version: Optional[str] = None

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(
        cls, value: Optional[datetime.datetime]
    ) -> Optional[datetime.datetime]:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("retrieved_at must include a timezone")
        return value
