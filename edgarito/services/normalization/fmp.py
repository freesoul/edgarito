from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from edgarito.enums.edgar.period import FISCAL_PERIOD_PRIORITY, FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    FinancialStatement,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.providers.fmp.fundamentals import (
    FinancialStatement as FmpFinancialStatement,
)
from edgarito.schemas.providers.fmp.fundamentals import FmpCompanyFinancials


@dataclass(frozen=True)
class FmpConceptDefinition:
    concept: FinancialConcept
    statement: FinancialStatement
    annual_reports_name: str
    quarterly_reports_name: str
    source_concept: str
    value_attribute: str
    absolute_value: bool = False


CONCEPT_DEFINITIONS = (
    FmpConceptDefinition(
        FinancialConcept.REVENUE,
        FinancialStatement.INCOME_STATEMENT,
        "annual_income_statements",
        "quarterly_income_statements",
        "revenue",
        "revenue",
    ),
    FmpConceptDefinition(
        FinancialConcept.OPERATING_INCOME,
        FinancialStatement.INCOME_STATEMENT,
        "annual_income_statements",
        "quarterly_income_statements",
        "operatingIncome",
        "operating_income",
    ),
    FmpConceptDefinition(
        FinancialConcept.NET_INCOME,
        FinancialStatement.INCOME_STATEMENT,
        "annual_income_statements",
        "quarterly_income_statements",
        "netIncome",
        "net_income",
    ),
    FmpConceptDefinition(
        FinancialConcept.TOTAL_ASSETS,
        FinancialStatement.BALANCE_SHEET,
        "annual_balance_sheets",
        "quarterly_balance_sheets",
        "totalAssets",
        "total_assets",
    ),
    FmpConceptDefinition(
        FinancialConcept.TOTAL_LIABILITIES,
        FinancialStatement.BALANCE_SHEET,
        "annual_balance_sheets",
        "quarterly_balance_sheets",
        "totalLiabilities",
        "total_liabilities",
    ),
    FmpConceptDefinition(
        FinancialConcept.STOCKHOLDERS_EQUITY,
        FinancialStatement.BALANCE_SHEET,
        "annual_balance_sheets",
        "quarterly_balance_sheets",
        "totalStockholdersEquity",
        "total_stockholders_equity",
    ),
    FmpConceptDefinition(
        FinancialConcept.CASH_AND_EQUIVALENTS,
        FinancialStatement.BALANCE_SHEET,
        "annual_balance_sheets",
        "quarterly_balance_sheets",
        "cashAndCashEquivalents",
        "cash_and_cash_equivalents",
    ),
    FmpConceptDefinition(
        FinancialConcept.OPERATING_CASH_FLOW,
        FinancialStatement.CASH_FLOW,
        "annual_cash_flow_statements",
        "quarterly_cash_flow_statements",
        "operatingCashFlow",
        "operating_cash_flow",
    ),
    FmpConceptDefinition(
        FinancialConcept.CAPITAL_EXPENDITURES,
        FinancialStatement.CASH_FLOW,
        "annual_cash_flow_statements",
        "quarterly_cash_flow_statements",
        "capitalExpenditure",
        "capital_expenditure",
        absolute_value=True,
    ),
)


class FmpNormalizer:
    """Normalize FMP statements into common financial observations."""

    _TAXONOMY = "fmp-standardized"

    def normalize(
        self,
        company_financials: FmpCompanyFinancials,
        granularity: Optional[Granularity] = None,
        concepts: Optional[set[FinancialConcept]] = None,
    ) -> NormalizedCompanyFinancials:
        self._validate_symbols(company_financials)
        observations = {}

        for definition in CONCEPT_DEFINITIONS:
            if concepts and definition.concept not in concepts:
                continue
            if granularity in (None, Granularity.ANNUAL):
                self._add_reports(
                    observations,
                    definition,
                    getattr(company_financials, definition.annual_reports_name),
                    Granularity.ANNUAL,
                )
            if granularity in (None, Granularity.QUARTERLY):
                self._add_reports(
                    observations,
                    definition,
                    getattr(company_financials, definition.quarterly_reports_name),
                    Granularity.QUARTERLY,
                )

        profile = company_financials.profile
        normalized = sorted(observations.values(), key=self._observation_sort_key)
        return NormalizedCompanyFinancials(
            provider="fmp",
            company_id=self._company_id(profile.cik, profile.symbol),
            company_name=profile.company_name,
            ticker=profile.symbol.upper(),
            observations=normalized,
        )

    def _add_reports(
        self,
        observations: dict,
        definition: FmpConceptDefinition,
        reports: list[FmpFinancialStatement],
        granularity: Granularity,
    ) -> None:
        for report in reports:
            if granularity == Granularity.ANNUAL and report.period != FiscalPeriod.FY:
                continue
            if granularity == Granularity.QUARTERLY and report.period == FiscalPeriod.FY:
                continue
            value: Optional[Decimal] = getattr(report, definition.value_attribute)
            if value is None:
                continue
            if definition.absolute_value:
                value = abs(value)

            key = (definition.concept, granularity, report.date)
            observations.setdefault(
                key,
                FinancialObservation(
                    concept=definition.concept,
                    statement=definition.statement,
                    value=value,
                    unit=report.reported_currency,
                    granularity=granularity,
                    fiscal_year=report.fiscal_year,
                    fiscal_period=report.period,
                    period_end=report.date,
                    provider="fmp",
                    taxonomy=self._TAXONOMY,
                    source_concept=definition.source_concept,
                    filed=report.filing_date,
                ),
            )

    @staticmethod
    def _validate_symbols(company_financials: FmpCompanyFinancials) -> None:
        expected = company_financials.profile.symbol.upper()
        reports = (
            company_financials.annual_income_statements
            + company_financials.quarterly_income_statements
            + company_financials.annual_balance_sheets
            + company_financials.quarterly_balance_sheets
            + company_financials.annual_cash_flow_statements
            + company_financials.quarterly_cash_flow_statements
        )
        if any(report.symbol.upper() != expected for report in reports):
            raise ValueError("FMP responses contain inconsistent symbols")

    @staticmethod
    def _company_id(cik: Optional[str], symbol: str) -> str:
        if cik and cik.isdigit():
            return cik.zfill(10)
        return cik or symbol.upper()

    @staticmethod
    def _observation_sort_key(observation: FinancialObservation):
        granularity_order = (
            0 if observation.granularity == Granularity.ANNUAL else 1
        )
        return (
            granularity_order,
            observation.fiscal_year,
            FISCAL_PERIOD_PRIORITY[observation.fiscal_period],
            observation.concept.value,
        )
