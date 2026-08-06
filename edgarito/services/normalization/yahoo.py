import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from edgarito.enums.edgar.period import FISCAL_PERIOD_PRIORITY, FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.providers.yahoo.fundamentals import (
    YahooCompanyFinancials,
    YahooFinancialReport,
)


@dataclass(frozen=True)
class YahooConceptDefinition:
    concept: FinancialConcept
    statement_name: str
    source_concepts: tuple[str, ...]
    unit: str = "currency"
    absolute_value: bool = False


CONCEPT_DEFINITIONS = (
    YahooConceptDefinition(
        FinancialConcept.REVENUE, "income_statements", ("TotalRevenue",)
    ),
    YahooConceptDefinition(
        FinancialConcept.OPERATING_INCOME,
        "income_statements",
        ("OperatingIncome", "TotalOperatingIncomeAsReported"),
    ),
    YahooConceptDefinition(
        FinancialConcept.PRETAX_INCOME, "income_statements", ("PretaxIncome",)
    ),
    YahooConceptDefinition(
        FinancialConcept.INCOME_TAX_EXPENSE,
        "income_statements",
        ("TaxProvision",),
    ),
    YahooConceptDefinition(
        FinancialConcept.NET_INCOME, "income_statements", ("NetIncome",)
    ),
    YahooConceptDefinition(
        FinancialConcept.INTEREST_EXPENSE,
        "income_statements",
        ("InterestExpense", "InterestExpenseNonOperating"),
    ),
    YahooConceptDefinition(
        FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES,
        "income_statements",
        ("BasicAverageShares",),
        unit="shares",
    ),
    YahooConceptDefinition(
        FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
        "income_statements",
        ("DilutedAverageShares",),
        unit="shares",
    ),
    YahooConceptDefinition(
        FinancialConcept.TOTAL_ASSETS, "balance_sheets", ("TotalAssets",)
    ),
    YahooConceptDefinition(
        FinancialConcept.CURRENT_ASSETS, "balance_sheets", ("CurrentAssets",)
    ),
    YahooConceptDefinition(
        FinancialConcept.ACCOUNTS_RECEIVABLE,
        "balance_sheets",
        ("AccountsReceivable", "Receivables"),
    ),
    YahooConceptDefinition(
        FinancialConcept.INVENTORY, "balance_sheets", ("Inventory",)
    ),
    YahooConceptDefinition(
        FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS,
        "balance_sheets",
        ("PrepaidAssets", "OtherCurrentAssets"),
    ),
    YahooConceptDefinition(
        FinancialConcept.TOTAL_LIABILITIES,
        "balance_sheets",
        ("TotalLiabilitiesNetMinorityInterest", "TotalLiabilities"),
    ),
    YahooConceptDefinition(
        FinancialConcept.CURRENT_LIABILITIES,
        "balance_sheets",
        ("CurrentLiabilities",),
    ),
    YahooConceptDefinition(
        FinancialConcept.ACCOUNTS_PAYABLE,
        "balance_sheets",
        ("AccountsPayable", "Payables"),
    ),
    YahooConceptDefinition(
        FinancialConcept.ACCRUED_LIABILITIES,
        "balance_sheets",
        ("CurrentAccruedExpenses",),
    ),
    YahooConceptDefinition(
        FinancialConcept.DEFERRED_REVENUE_CURRENT,
        "balance_sheets",
        ("CurrentDeferredRevenue",),
    ),
    YahooConceptDefinition(
        FinancialConcept.SHORT_TERM_DEBT,
        "balance_sheets",
        ("CurrentDebt",),
    ),
    YahooConceptDefinition(
        FinancialConcept.LONG_TERM_DEBT_CURRENT,
        "balance_sheets",
        ("CurrentPortionOfLongTermDebt",),
    ),
    YahooConceptDefinition(
        FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
        "balance_sheets",
        ("LongTermDebt",),
    ),
    YahooConceptDefinition(
        FinancialConcept.STOCKHOLDERS_EQUITY,
        "balance_sheets",
        ("StockholdersEquity", "CommonStockEquity"),
    ),
    YahooConceptDefinition(
        FinancialConcept.CASH_AND_EQUIVALENTS,
        "balance_sheets",
        ("CashAndCashEquivalents", "CashFinancial"),
    ),
    YahooConceptDefinition(
        FinancialConcept.SHORT_TERM_INVESTMENTS,
        "balance_sheets",
        ("OtherShortTermInvestments", "ShortTermInvestments"),
    ),
    YahooConceptDefinition(
        FinancialConcept.NONCURRENT_INVESTMENTS,
        "balance_sheets",
        ("InvestmentinFinancialAssets", "LongTermEquityInvestment"),
    ),
    YahooConceptDefinition(FinancialConcept.GOODWILL, "balance_sheets", ("Goodwill",)),
    YahooConceptDefinition(
        FinancialConcept.INTANGIBLE_ASSETS_NET,
        "balance_sheets",
        ("OtherIntangibleAssets",),
    ),
    YahooConceptDefinition(
        FinancialConcept.SHARES_OUTSTANDING,
        "balance_sheets",
        ("OrdinarySharesNumber",),
        unit="shares",
    ),
    YahooConceptDefinition(
        FinancialConcept.OPERATING_CASH_FLOW,
        "cash_flow_statements",
        ("OperatingCashFlow",),
    ),
    YahooConceptDefinition(
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
        "cash_flow_statements",
        ("DepreciationAndAmortization", "DepreciationAmortizationDepletion"),
    ),
    YahooConceptDefinition(
        FinancialConcept.CAPITAL_EXPENDITURES,
        "cash_flow_statements",
        ("CapitalExpenditure",),
        absolute_value=True,
    ),
    YahooConceptDefinition(
        FinancialConcept.DIVIDENDS_PAID,
        "cash_flow_statements",
        ("CashDividendsPaid", "CommonStockDividendPaid"),
        absolute_value=True,
    ),
)


class YahooFinancialsNormalizer:
    """Normalize Yahoo's standardized statement rows without inventing splits."""

    _TAXONOMY = "yahoo-standardized"

    def normalize(
        self,
        company_financials: YahooCompanyFinancials,
        granularity: Optional[Granularity] = None,
        concepts: Optional[set[FinancialConcept]] = None,
    ) -> NormalizedCompanyFinancials:
        fiscal_end_month = self._fiscal_end_month(company_financials)
        observations = {}
        for definition in CONCEPT_DEFINITIONS:
            if concepts and definition.concept not in concepts:
                continue
            for selected_granularity, prefix in (
                (Granularity.ANNUAL, "annual"),
                (Granularity.QUARTERLY, "quarterly"),
            ):
                if granularity not in (None, selected_granularity):
                    continue
                reports = getattr(
                    company_financials, f"{prefix}_{definition.statement_name}"
                )
                self._add_reports(
                    observations,
                    definition,
                    reports,
                    selected_granularity,
                    fiscal_end_month,
                    company_financials.currency,
                )

        return NormalizedCompanyFinancials(
            provider="yahoo",
            company_id=company_financials.symbol,
            company_name=company_financials.company_name,
            ticker=company_financials.symbol,
            observations=sorted(observations.values(), key=self._sort_key),
        )

    def _add_reports(
        self,
        observations: dict,
        definition: YahooConceptDefinition,
        reports: tuple[YahooFinancialReport, ...],
        granularity: Granularity,
        fiscal_end_month: int,
        currency: str,
    ) -> None:
        for report in reports:
            source_concept, value = self._first_value(
                report, definition.source_concepts
            )
            if value is None or source_concept is None:
                continue
            if definition.absolute_value:
                value = abs(value)
            fiscal_year, fiscal_period = self._fiscal_period(
                report.period_end, granularity, fiscal_end_month
            )
            key = (definition.concept, granularity, report.period_end)
            observations.setdefault(
                key,
                FinancialObservation(
                    concept=definition.concept,
                    statement=definition.concept.statement,
                    value=value,
                    unit=(
                        currency if definition.unit == "currency" else definition.unit
                    ),
                    granularity=granularity,
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period,
                    period_end=report.period_end,
                    provider="yahoo",
                    taxonomy=self._TAXONOMY,
                    source_concept=source_concept,
                ),
            )

    @staticmethod
    def _first_value(
        report: YahooFinancialReport, source_concepts: tuple[str, ...]
    ) -> tuple[Optional[str], Optional[Decimal]]:
        for source_concept in source_concepts:
            if source_concept in report.values:
                return source_concept, report.values[source_concept]
        return None, None

    @staticmethod
    def _fiscal_end_month(company_financials: YahooCompanyFinancials) -> int:
        annual_dates = [
            report.period_end
            for reports in (
                company_financials.annual_income_statements,
                company_financials.annual_balance_sheets,
                company_financials.annual_cash_flow_statements,
            )
            for report in reports
        ]
        if not annual_dates:
            raise ValueError("Yahoo data does not identify the fiscal year end")
        return max(annual_dates).month

    @staticmethod
    def _fiscal_period(
        period_end: datetime.date,
        granularity: Granularity,
        fiscal_end_month: int,
    ) -> tuple[int, FiscalPeriod]:
        if granularity == Granularity.ANNUAL:
            return period_end.year, FiscalPeriod.FY
        months_after_year_end = (period_end.month - fiscal_end_month) % 12
        candidates = {
            0: FiscalPeriod.Q4,
            3: FiscalPeriod.Q1,
            6: FiscalPeriod.Q2,
            9: FiscalPeriod.Q3,
        }
        expected = min(
            candidates,
            key=lambda month: min(
                abs(months_after_year_end - month),
                12 - abs(months_after_year_end - month),
            ),
        )
        fiscal_year = period_end.year + (
            1 if period_end.month > fiscal_end_month else 0
        )
        return fiscal_year, candidates[expected]

    @staticmethod
    def _sort_key(observation: FinancialObservation):
        granularity_order = 0 if observation.granularity == Granularity.ANNUAL else 1
        return (
            granularity_order,
            observation.fiscal_year,
            FISCAL_PERIOD_PRIORITY[observation.fiscal_period],
            observation.concept.value,
        )


__all__ = ["YahooFinancialsNormalizer"]
