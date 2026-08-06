import datetime
from decimal import Decimal
from typing import Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
)

from edgarito.enums.edgar.period import FiscalPeriod


class FmpModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @field_validator("*", mode="before")
    @classmethod
    def normalize_missing_value(cls, value):
        if value in ("", "-", "None", "null"):
            return None
        return value


class CompanyProfile(FmpModel):
    symbol: str
    company_name: str = Field(alias="companyName")
    cik: Optional[str] = None
    currency: Optional[str] = None
    country: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None


class CompanyProfileResponse(RootModel[list[CompanyProfile]]):
    pass


class SecuritySearchResult(FmpModel):
    """The common subset returned by FMP's identifier search endpoints."""

    symbol: str
    name: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("name", "companyName")
    )
    cik: Optional[str] = None
    isin: Optional[str] = None
    currency: Optional[str] = None
    stock_exchange: Optional[str] = Field(default=None, alias="stockExchange")
    exchange_short_name: Optional[str] = Field(default=None, alias="exchangeShortName")


class FinancialStatement(FmpModel):
    symbol: str
    date: datetime.date
    fiscal_year: int = Field(alias="fiscalYear")
    period: FiscalPeriod
    reported_currency: str = Field(alias="reportedCurrency")
    cik: Optional[str] = None
    filing_date: Optional[datetime.date] = Field(default=None, alias="filingDate")
    accepted_date: Optional[datetime.datetime] = Field(
        default=None, alias="acceptedDate"
    )


class IncomeStatement(FinancialStatement):
    revenue: Optional[Decimal] = None
    operating_income: Optional[Decimal] = Field(default=None, alias="operatingIncome")
    net_income: Optional[Decimal] = Field(default=None, alias="netIncome")


class BalanceSheet(FinancialStatement):
    total_assets: Optional[Decimal] = Field(default=None, alias="totalAssets")
    total_liabilities: Optional[Decimal] = Field(default=None, alias="totalLiabilities")
    total_stockholders_equity: Optional[Decimal] = Field(
        default=None, alias="totalStockholdersEquity"
    )
    cash_and_cash_equivalents: Optional[Decimal] = Field(
        default=None, alias="cashAndCashEquivalents"
    )


class CashFlowStatement(FinancialStatement):
    operating_cash_flow: Optional[Decimal] = Field(
        default=None, alias="operatingCashFlow"
    )
    capital_expenditure: Optional[Decimal] = Field(
        default=None, alias="capitalExpenditure"
    )


class IncomeStatementResponse(RootModel[list[IncomeStatement]]):
    pass


class BalanceSheetResponse(RootModel[list[BalanceSheet]]):
    pass


class CashFlowStatementResponse(RootModel[list[CashFlowStatement]]):
    pass


class FmpCompanyFinancials(BaseModel):
    profile: CompanyProfile
    annual_income_statements: list[IncomeStatement]
    quarterly_income_statements: list[IncomeStatement]
    annual_balance_sheets: list[BalanceSheet]
    quarterly_balance_sheets: list[BalanceSheet]
    annual_cash_flow_statements: list[CashFlowStatement]
    quarterly_cash_flow_statements: list[CashFlowStatement]
