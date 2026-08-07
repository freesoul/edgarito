import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
