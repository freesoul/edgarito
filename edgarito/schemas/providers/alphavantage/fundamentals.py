import datetime
from decimal import Decimal
from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


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
    sector: Optional[str] = Field(default=None, alias="Sector")
    industry: Optional[str] = Field(default=None, alias="Industry")
    country: Optional[str] = Field(default=None, alias="Country")
    exchange: Optional[str] = Field(default=None, alias="Exchange")


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


class EarningsEstimateReport(AlphaVantageModel):
    """One annual or quarterly Alpha Vantage earnings-estimate row.

    Alpha Vantage has used a few names for the revenue fields over time.  The
    aliases below keep the provider schema tolerant while the normalizer
    exposes one provider-neutral representation.
    """

    fiscal_date_ending: Optional[datetime.date] = Field(
        default=None,
        validation_alias=AliasChoices(
            "fiscalDateEnding", "fiscal_date_ending", "date"
        ),
    )
    horizon: Optional[str] = None
    fiscal_year: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("fiscalYear", "fiscal_year", "year"),
    )
    estimated_revenue: Optional[Decimal] = Field(
        default=None,
        validation_alias=AliasChoices(
            "estimatedRevenue",
            "revenueEstimate",
            "averageRevenueEstimate",
            "revenue_estimate_average",
            "estimated_revenue",
            "revenue",
        ),
    )
    revenue_estimate_low: Optional[Decimal] = Field(
        default=None,
        validation_alias=AliasChoices(
            "revenueEstimateLow",
            "estimatedRevenueLow",
            "lowRevenueEstimate",
            "revenue_estimate_low",
            "low",
        ),
    )
    revenue_estimate_high: Optional[Decimal] = Field(
        default=None,
        validation_alias=AliasChoices(
            "revenueEstimateHigh",
            "estimatedRevenueHigh",
            "highRevenueEstimate",
            "revenue_estimate_high",
            "high",
        ),
    )
    analyst_count: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices(
            "numberOfAnalysts",
            "numberOfRevenueAnalysts",
            "revenue_estimate_analyst_count",
            "analystCount",
            "analyst_count",
        ),
    )
    reported_currency: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "reportedCurrency",
            "revenueCurrency",
            "revenue_currency",
            "currency",
        ),
    )

    @field_validator(
        "estimated_revenue",
        "revenue_estimate_low",
        "revenue_estimate_high",
        mode="before",
    )
    @classmethod
    def unwrap_numeric_value(cls, value):
        if isinstance(value, dict):
            return value.get("raw", value.get("value", value.get("val")))
        return value

    @field_validator("analyst_count", mode="before")
    @classmethod
    def normalize_analyst_count(cls, value):
        if value in (None, "", "-", "None", "null"):
            return None
        try:
            return int(Decimal(str(value)))
        except (ArithmeticError, TypeError, ValueError):
            return None


class AlphaVantageCompanyFinancials(BaseModel):
    overview: CompanyOverview
    income_statement: IncomeStatementResponse
    balance_sheet: BalanceSheetResponse
    cash_flow: CashFlowResponse


class EarningsEstimatesResponse(AlphaVantageModel):
    """Alpha Vantage ``EARNINGS_ESTIMATES`` response."""

    symbol: str = Field(default="", alias="symbol")
    annual_earnings: list[EarningsEstimateReport] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "annualEarnings",
            "annualEstimates",
            "annualReports",
            "estimates",
            "annual_earnings",
        ),
    )
    quarterly_earnings: list[EarningsEstimateReport] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "quarterlyEarnings",
            "quarterlyEstimates",
            "quarterlyReports",
            "quarterly_earnings",
        ),
    )
