import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AlphaVantageModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @field_validator("*", mode="before")
    @classmethod
    def normalize_missing_value(cls, value):
        if value in ("", "-", "None", "null"):
            return None
        return value


class CompanyOverview(AlphaVantageModel):
    symbol: str = Field(alias="Symbol")
    name: str = Field(alias="Name")
    cik: Optional[str] = Field(default=None, alias="CIK")
    currency: Optional[str] = Field(default=None, alias="Currency")
    fiscal_year_end: Optional[str] = Field(default=None, alias="FiscalYearEnd")


class FinancialReport(AlphaVantageModel):
    fiscal_date_ending: datetime.date = Field(alias="fiscalDateEnding")
    reported_currency: str = Field(alias="reportedCurrency")


class IncomeStatementReport(FinancialReport):
    total_revenue: Optional[Decimal] = Field(default=None, alias="totalRevenue")
    operating_income: Optional[Decimal] = Field(default=None, alias="operatingIncome")
    net_income: Optional[Decimal] = Field(default=None, alias="netIncome")


class BalanceSheetReport(FinancialReport):
    total_assets: Optional[Decimal] = Field(default=None, alias="totalAssets")
    total_liabilities: Optional[Decimal] = Field(default=None, alias="totalLiabilities")
    total_shareholder_equity: Optional[Decimal] = Field(
        default=None, alias="totalShareholderEquity"
    )
    cash_and_cash_equivalents_at_carrying_value: Optional[Decimal] = Field(
        default=None, alias="cashAndCashEquivalentsAtCarryingValue"
    )


class CashFlowReport(FinancialReport):
    operating_cashflow: Optional[Decimal] = Field(
        default=None, alias="operatingCashflow"
    )
    capital_expenditures: Optional[Decimal] = Field(
        default=None, alias="capitalExpenditures"
    )


class IncomeStatementResponse(AlphaVantageModel):
    symbol: str = Field(alias="symbol")
    annual_reports: list[IncomeStatementReport] = Field(alias="annualReports")
    quarterly_reports: list[IncomeStatementReport] = Field(alias="quarterlyReports")


class BalanceSheetResponse(AlphaVantageModel):
    symbol: str = Field(alias="symbol")
    annual_reports: list[BalanceSheetReport] = Field(alias="annualReports")
    quarterly_reports: list[BalanceSheetReport] = Field(alias="quarterlyReports")


class CashFlowResponse(AlphaVantageModel):
    symbol: str = Field(alias="symbol")
    annual_reports: list[CashFlowReport] = Field(alias="annualReports")
    quarterly_reports: list[CashFlowReport] = Field(alias="quarterlyReports")


class AlphaVantageCompanyFinancials(BaseModel):
    overview: CompanyOverview
    income_statement: IncomeStatementResponse
    balance_sheet: BalanceSheetResponse
    cash_flow: CashFlowResponse
