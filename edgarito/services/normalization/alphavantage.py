import calendar
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
from edgarito.schemas.providers.alphavantage.fundamentals import (
    AlphaVantageCompanyFinancials,
    FinancialReport,
)


@dataclass(frozen=True)
class AlphaVantageConceptDefinition:
    concept: FinancialConcept
    response_name: str
    source_concept: str


CONCEPT_DEFINITIONS = (
    AlphaVantageConceptDefinition(
        FinancialConcept.REVENUE,
        "income_statement",
        "totalRevenue",
    ),
    AlphaVantageConceptDefinition(
        FinancialConcept.OPERATING_INCOME,
        "income_statement",
        "operatingIncome",
    ),
    AlphaVantageConceptDefinition(
        FinancialConcept.NET_INCOME,
        "income_statement",
        "netIncome",
    ),
    AlphaVantageConceptDefinition(
        FinancialConcept.TOTAL_ASSETS,
        "balance_sheet",
        "totalAssets",
    ),
    AlphaVantageConceptDefinition(
        FinancialConcept.TOTAL_LIABILITIES,
        "balance_sheet",
        "totalLiabilities",
    ),
    AlphaVantageConceptDefinition(
        FinancialConcept.STOCKHOLDERS_EQUITY,
        "balance_sheet",
        "totalShareholderEquity",
    ),
    AlphaVantageConceptDefinition(
        FinancialConcept.CASH_AND_EQUIVALENTS,
        "balance_sheet",
        "cashAndCashEquivalentsAtCarryingValue",
    ),
    AlphaVantageConceptDefinition(
        FinancialConcept.OPERATING_CASH_FLOW,
        "cash_flow",
        "operatingCashflow",
    ),
    AlphaVantageConceptDefinition(
        FinancialConcept.CAPITAL_EXPENDITURES,
        "cash_flow",
        "capitalExpenditures",
    ),
)


class AlphaVantageNormalizer:
    """Normalize Alpha Vantage fundamentals into common financial observations."""

    _TAXONOMY = "gaap-ifrs"

    def normalize(
        self,
        company_financials: AlphaVantageCompanyFinancials,
        granularity: Optional[Granularity] = None,
        concepts: Optional[set[FinancialConcept]] = None,
    ) -> NormalizedCompanyFinancials:
        self._validate_symbols(company_financials)
        fiscal_end_month = self._fiscal_end_month(company_financials)
        observations: dict[
            tuple[FinancialConcept, Granularity, datetime.date], FinancialObservation
        ] = {}

        for definition in CONCEPT_DEFINITIONS:
            if concepts and definition.concept not in concepts:
                continue

            response = getattr(company_financials, definition.response_name)
            if granularity in (None, Granularity.ANNUAL):
                self._add_reports(
                    observations,
                    definition,
                    response.annual_reports,
                    Granularity.ANNUAL,
                    fiscal_end_month,
                )
            if granularity in (None, Granularity.QUARTERLY):
                self._add_reports(
                    observations,
                    definition,
                    response.quarterly_reports,
                    Granularity.QUARTERLY,
                    fiscal_end_month,
                )

        normalized = sorted(observations.values(), key=self._observation_sort_key)
        overview = company_financials.overview
        return NormalizedCompanyFinancials(
            provider="alphavantage",
            company_id=self._company_id(overview.cik, overview.symbol),
            company_name=overview.name,
            ticker=overview.symbol.upper(),
            observations=normalized,
        )

    def _add_reports(
        self,
        observations: dict[
            tuple[FinancialConcept, Granularity, datetime.date], FinancialObservation
        ],
        definition: AlphaVantageConceptDefinition,
        reports: list[FinancialReport],
        granularity: Granularity,
        fiscal_end_month: int,
    ) -> None:
        attribute_name = self._snake_case(definition.source_concept)
        for report in reports:
            value: Optional[Decimal] = getattr(report, attribute_name)
            if value is None:
                continue

            if granularity == Granularity.ANNUAL:
                fiscal_year = report.fiscal_date_ending.year
                fiscal_period = FiscalPeriod.FY
            else:
                fiscal_period = self._quarter_for_date(
                    report.fiscal_date_ending, fiscal_end_month
                )
                fiscal_year = report.fiscal_date_ending.year + (
                    1 if report.fiscal_date_ending.month > fiscal_end_month else 0
                )

            key = (definition.concept, granularity, report.fiscal_date_ending)
            observations.setdefault(
                key,
                FinancialObservation(
                    concept=definition.concept,
                    statement=definition.concept.statement,
                    value=value,
                    unit=report.reported_currency,
                    granularity=granularity,
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period,
                    period_end=report.fiscal_date_ending,
                    provider="alphavantage",
                    taxonomy=self._TAXONOMY,
                    source_concept=definition.source_concept,
                ),
            )

    @staticmethod
    def _snake_case(value: str) -> str:
        result = []
        for character in value:
            if character.isupper():
                result.extend(("_", character.lower()))
            else:
                result.append(character)
        return "".join(result)

    @staticmethod
    def _quarter_for_date(
        period_end: datetime.date, fiscal_end_month: int
    ) -> FiscalPeriod:
        months_after_year_end = (period_end.month - fiscal_end_month) % 12
        candidates = {
            0: FiscalPeriod.Q4,
            3: FiscalPeriod.Q1,
            6: FiscalPeriod.Q2,
            9: FiscalPeriod.Q3,
        }

        def circular_distance(expected_month: int) -> int:
            difference = abs(months_after_year_end - expected_month)
            return min(difference, 12 - difference)

        expected_month = min(candidates, key=circular_distance)
        return candidates[expected_month]

    @staticmethod
    def _fiscal_end_month(company_financials: AlphaVantageCompanyFinancials) -> int:
        annual_dates = [
            report.fiscal_date_ending
            for response in (
                company_financials.income_statement,
                company_financials.balance_sheet,
                company_financials.cash_flow,
            )
            for report in response.annual_reports
        ]
        if annual_dates:
            return max(annual_dates).month

        fiscal_year_end = company_financials.overview.fiscal_year_end
        if fiscal_year_end:
            month_lookup = {
                month.lower(): index
                for index, month in enumerate(calendar.month_name)
                if month
            }
            month = month_lookup.get(fiscal_year_end.strip().lower())
            if month is not None:
                return month
        raise ValueError("Alpha Vantage data does not identify the fiscal year end")

    @staticmethod
    def _validate_symbols(company_financials: AlphaVantageCompanyFinancials) -> None:
        expected = company_financials.overview.symbol.upper()
        symbols = {
            company_financials.income_statement.symbol.upper(),
            company_financials.balance_sheet.symbol.upper(),
            company_financials.cash_flow.symbol.upper(),
        }
        if symbols != {expected}:
            raise ValueError("Alpha Vantage responses contain inconsistent symbols")

    @staticmethod
    def _company_id(cik: Optional[str], symbol: str) -> str:
        if cik and cik.isdigit():
            return cik.zfill(10)
        return cik or symbol.upper()

    @staticmethod
    def _observation_sort_key(observation: FinancialObservation):
        granularity_order = 0 if observation.granularity == Granularity.ANNUAL else 1
        return (
            granularity_order,
            observation.fiscal_year,
            FISCAL_PERIOD_PRIORITY[observation.fiscal_period],
            observation.concept.value,
        )
