import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


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
    annual_income_statements: tuple[YahooFinancialReport, ...] = ()
    quarterly_income_statements: tuple[YahooFinancialReport, ...] = ()
    annual_balance_sheets: tuple[YahooFinancialReport, ...] = ()
    quarterly_balance_sheets: tuple[YahooFinancialReport, ...] = ()
    annual_cash_flow_statements: tuple[YahooFinancialReport, ...] = ()
    quarterly_cash_flow_statements: tuple[YahooFinancialReport, ...] = ()
