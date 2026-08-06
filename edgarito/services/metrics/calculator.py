import datetime
from decimal import Decimal
from typing import Optional

from edgarito.enums.edgar.period import FISCAL_PERIOD_PRIORITY, FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.metrics.models import (
    CompanyMetrics,
    FinancialMetric,
    MetricObservation,
)

PeriodKey = tuple[int, FiscalPeriod]

OPERATING_WORKING_CAPITAL_REQUIRED_ASSET_CONCEPTS = frozenset(
    {
        FinancialConcept.ACCOUNTS_RECEIVABLE,
        FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS,
    }
)
OPERATING_WORKING_CAPITAL_ASSET_CONCEPTS = (
    OPERATING_WORKING_CAPITAL_REQUIRED_ASSET_CONCEPTS
    | {FinancialConcept.INVENTORY}
)
OPERATING_WORKING_CAPITAL_DETAIL_LIABILITY_CONCEPTS = frozenset(
    {
        FinancialConcept.ACCOUNTS_PAYABLE,
        FinancialConcept.ACCRUED_LIABILITIES,
        FinancialConcept.DEFERRED_REVENUE_CURRENT,
    }
)
OPERATING_WORKING_CAPITAL_AGGREGATE_LIABILITY_CONCEPTS = frozenset(
    {
        FinancialConcept.CURRENT_LIABILITIES,
        FinancialConcept.SHORT_TERM_DEBT,
        FinancialConcept.LONG_TERM_DEBT_CURRENT,
    }
)
OPERATING_WORKING_CAPITAL_CONCEPTS = (
    OPERATING_WORKING_CAPITAL_ASSET_CONCEPTS
    | OPERATING_WORKING_CAPITAL_DETAIL_LIABILITY_CONCEPTS
    | OPERATING_WORKING_CAPITAL_AGGREGATE_LIABILITY_CONCEPTS
)
DEBT_CONCEPTS = frozenset(
    {
        FinancialConcept.SHORT_TERM_DEBT,
        FinancialConcept.LONG_TERM_DEBT_CURRENT,
        FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
    }
)
NOPAT_CONCEPTS = frozenset(
    {
        FinancialConcept.OPERATING_INCOME,
        FinancialConcept.PRETAX_INCOME,
        FinancialConcept.INCOME_TAX_EXPENSE,
    }
)

METRIC_CONCEPTS: dict[FinancialMetric, frozenset[FinancialConcept]] = {
    FinancialMetric.REVENUE_GROWTH: frozenset({FinancialConcept.REVENUE}),
    FinancialMetric.OPERATING_MARGIN: frozenset(
        {FinancialConcept.OPERATING_INCOME, FinancialConcept.REVENUE}
    ),
    FinancialMetric.NET_MARGIN: frozenset(
        {FinancialConcept.NET_INCOME, FinancialConcept.REVENUE}
    ),
    FinancialMetric.EFFECTIVE_TAX_RATE: frozenset(
        {FinancialConcept.INCOME_TAX_EXPENSE, FinancialConcept.PRETAX_INCOME}
    ),
    FinancialMetric.NOPAT: NOPAT_CONCEPTS,
    FinancialMetric.EBITDA: frozenset(
        {
            FinancialConcept.OPERATING_INCOME,
            FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
        }
    ),
    FinancialMetric.FREE_CASH_FLOW: frozenset(
        {
            FinancialConcept.OPERATING_CASH_FLOW,
            FinancialConcept.CAPITAL_EXPENDITURES,
        }
    ),
    FinancialMetric.FREE_CASH_FLOW_MARGIN: frozenset(
        {
            FinancialConcept.OPERATING_CASH_FLOW,
            FinancialConcept.CAPITAL_EXPENDITURES,
            FinancialConcept.REVENUE,
        }
    ),
    FinancialMetric.OPERATING_WORKING_CAPITAL: OPERATING_WORKING_CAPITAL_CONCEPTS,
    FinancialMetric.CHANGE_IN_OPERATING_WORKING_CAPITAL: (
        OPERATING_WORKING_CAPITAL_CONCEPTS
    ),
    FinancialMetric.GROSS_DEBT: DEBT_CONCEPTS,
    FinancialMetric.NET_DEBT: DEBT_CONCEPTS
    | frozenset({FinancialConcept.CASH_AND_EQUIVALENTS}),
    FinancialMetric.TANGIBLE_BOOK_EQUITY: frozenset(
        {
            FinancialConcept.STOCKHOLDERS_EQUITY,
            FinancialConcept.GOODWILL,
            FinancialConcept.INTANGIBLE_ASSETS_NET,
        }
    ),
    FinancialMetric.FCFF: NOPAT_CONCEPTS
    | OPERATING_WORKING_CAPITAL_CONCEPTS
    | frozenset(
        {
            FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
            FinancialConcept.CAPITAL_EXPENDITURES,
        }
    ),
    FinancialMetric.RETURN_ON_ASSETS: frozenset(
        {FinancialConcept.NET_INCOME, FinancialConcept.TOTAL_ASSETS}
    ),
    FinancialMetric.RETURN_ON_EQUITY: frozenset(
        {FinancialConcept.NET_INCOME, FinancialConcept.STOCKHOLDERS_EQUITY}
    ),
    FinancialMetric.LIABILITIES_TO_ASSETS: frozenset(
        {FinancialConcept.TOTAL_LIABILITIES, FinancialConcept.TOTAL_ASSETS}
    ),
    FinancialMetric.CASH_TO_LIABILITIES: frozenset(
        {
            FinancialConcept.CASH_AND_EQUIVALENTS,
            FinancialConcept.TOTAL_LIABILITIES,
        }
    ),
    FinancialMetric.OPERATING_CASH_FLOW_TO_NET_INCOME: frozenset(
        {FinancialConcept.OPERATING_CASH_FLOW, FinancialConcept.NET_INCOME}
    ),
}


def operating_working_capital_input_concepts(
    observations: dict[FinancialConcept, FinancialObservation],
) -> tuple[FinancialConcept, ...]:
    """Return the concepts used by the best available OWC calculation."""
    if OPERATING_WORKING_CAPITAL_DETAIL_LIABILITY_CONCEPTS <= observations.keys():
        concepts = (
            OPERATING_WORKING_CAPITAL_REQUIRED_ASSET_CONCEPTS
            | OPERATING_WORKING_CAPITAL_DETAIL_LIABILITY_CONCEPTS
        )
    elif FinancialConcept.CURRENT_LIABILITIES in observations:
        concepts = OPERATING_WORKING_CAPITAL_REQUIRED_ASSET_CONCEPTS | {
            FinancialConcept.CURRENT_LIABILITIES,
            *(
                concept
                for concept in (
                    FinancialConcept.SHORT_TERM_DEBT,
                    FinancialConcept.LONG_TERM_DEBT_CURRENT,
                )
                if concept in observations
            ),
        }
    else:
        return ()
    if FinancialConcept.INVENTORY in observations:
        concepts = concepts | {FinancialConcept.INVENTORY}
    return tuple(sorted(concepts, key=lambda concept: concept.value))


def operating_working_capital_formula(
    observations: dict[FinancialConcept, FinancialObservation],
) -> str:
    if OPERATING_WORKING_CAPITAL_DETAIL_LIABILITY_CONCEPTS <= observations.keys():
        return (
            "receivables + separately reported inventory + prepaid/other current "
            "assets - payables - accrued liabilities - current deferred revenue"
        )
    return (
        "receivables + separately reported inventory + prepaid/other current "
        "assets - current liabilities + reported current debt"
    )


def operating_working_capital_value(
    observations: Optional[dict[FinancialConcept, FinancialObservation]],
) -> Optional[FinancialObservation]:
    """Calculate OWC from detailed liabilities or a current-liability residual."""
    if observations is None or not (
        OPERATING_WORKING_CAPITAL_REQUIRED_ASSET_CONCEPTS <= observations.keys()
    ):
        return None

    asset_terms = (
        (observations.get(FinancialConcept.ACCOUNTS_RECEIVABLE), 1),
        (observations.get(FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS), 1),
        *(
            ((observations[FinancialConcept.INVENTORY], 1),)
            if FinancialConcept.INVENTORY in observations
            else ()
        ),
    )

    if OPERATING_WORKING_CAPITAL_DETAIL_LIABILITY_CONCEPTS <= observations.keys():
        terms = asset_terms + (
            (observations.get(FinancialConcept.ACCOUNTS_PAYABLE), -1),
            (observations.get(FinancialConcept.ACCRUED_LIABILITIES), -1),
            (observations.get(FinancialConcept.DEFERRED_REVENUE_CURRENT), -1),
        )
    elif FinancialConcept.CURRENT_LIABILITIES in observations:
        # A missing current-debt tag means the filing reported no value under the
        # supported debt concepts; only explicitly reported debt is added back.
        terms = asset_terms + (
            (observations.get(FinancialConcept.CURRENT_LIABILITIES), -1),
            *(
                (observations[concept], 1)
                for concept in (
                    FinancialConcept.SHORT_TERM_DEBT,
                    FinancialConcept.LONG_TERM_DEBT_CURRENT,
                )
                if concept in observations
            ),
        )
    else:
        return None
    return FinancialMetricsService._combine_amounts(terms)


class FinancialMetricsService:
    """Calculate deterministic metrics from one provider's normalized data."""

    def calculate(
        self,
        financials: NormalizedCompanyFinancials,
        granularity: Optional[Granularity] = None,
        metrics: Optional[set[FinancialMetric]] = None,
    ) -> CompanyMetrics:
        selected_metrics = set(FinancialMetric) if metrics is None else metrics
        observations: list[MetricObservation] = []

        for selected_granularity in Granularity:
            if granularity is not None and selected_granularity != granularity:
                continue
            observations.extend(
                self._calculate_granularity(
                    financials,
                    selected_granularity,
                    selected_metrics,
                )
            )

        observations.sort(key=self._observation_sort_key)
        return CompanyMetrics(
            provider=financials.provider,
            company_id=financials.company_id,
            company_name=financials.company_name,
            ticker=financials.ticker,
            observations=observations,
        )

    @staticmethod
    def required_concepts(
        metrics: Optional[set[FinancialMetric]] = None,
    ) -> set[FinancialConcept]:
        selected_metrics = set(FinancialMetric) if metrics is None else metrics
        return {
            concept
            for metric in selected_metrics
            for concept in METRIC_CONCEPTS[metric]
        }

    def _calculate_granularity(
        self,
        financials: NormalizedCompanyFinancials,
        granularity: Granularity,
        selected_metrics: set[FinancialMetric],
    ) -> list[MetricObservation]:
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]] = {}
        for observation in financials.observations:
            if observation.granularity != granularity:
                continue
            by_period.setdefault(observation.period_key, {}).setdefault(
                observation.concept, observation
            )

        periods = sorted(by_period, key=self._period_sort_key)
        results: list[MetricObservation] = []
        for index, period in enumerate(periods):
            current = by_period[period]
            previous = None
            if index > 0 and self._consecutive(periods[index - 1], period, granularity):
                previous = by_period[periods[index - 1]]
            self._calculate_period(
                results,
                financials.provider,
                granularity,
                period,
                current,
                previous,
                selected_metrics,
            )
        return results

    def _calculate_period(
        self,
        results: list[MetricObservation],
        provider: str,
        granularity: Granularity,
        period: PeriodKey,
        current: dict[FinancialConcept, FinancialObservation],
        previous: Optional[dict[FinancialConcept, FinancialObservation]],
        selected_metrics: set[FinancialMetric],
    ) -> None:
        revenue = current.get(FinancialConcept.REVENUE)
        operating_income = current.get(FinancialConcept.OPERATING_INCOME)
        pretax_income = current.get(FinancialConcept.PRETAX_INCOME)
        income_tax_expense = current.get(FinancialConcept.INCOME_TAX_EXPENSE)
        net_income = current.get(FinancialConcept.NET_INCOME)
        assets = current.get(FinancialConcept.TOTAL_ASSETS)
        liabilities = current.get(FinancialConcept.TOTAL_LIABILITIES)
        equity = current.get(FinancialConcept.STOCKHOLDERS_EQUITY)
        cash = current.get(FinancialConcept.CASH_AND_EQUIVALENTS)
        operating_cash_flow = current.get(FinancialConcept.OPERATING_CASH_FLOW)
        depreciation_and_amortization = current.get(
            FinancialConcept.DEPRECIATION_AND_AMORTIZATION
        )
        capital_expenditures = current.get(FinancialConcept.CAPITAL_EXPENDITURES)

        if FinancialMetric.REVENUE_GROWTH in selected_metrics and previous is not None:
            previous_revenue = previous.get(FinancialConcept.REVENUE)
            self._add_ratio(
                results,
                FinancialMetric.REVENUE_GROWTH,
                revenue,
                previous_revenue,
                provider,
                granularity,
                period,
                "100 × (revenue - prior period revenue) / prior period revenue",
                (FinancialConcept.REVENUE,),
                subtract_denominator=True,
            )

        if FinancialMetric.OPERATING_MARGIN in selected_metrics:
            self._add_ratio(
                results,
                FinancialMetric.OPERATING_MARGIN,
                operating_income,
                revenue,
                provider,
                granularity,
                period,
                "100 × operating income / revenue",
                (FinancialConcept.OPERATING_INCOME, FinancialConcept.REVENUE),
            )

        if FinancialMetric.NET_MARGIN in selected_metrics:
            self._add_ratio(
                results,
                FinancialMetric.NET_MARGIN,
                net_income,
                revenue,
                provider,
                granularity,
                period,
                "100 × net income / revenue",
                (FinancialConcept.NET_INCOME, FinancialConcept.REVENUE),
            )

        if FinancialMetric.EFFECTIVE_TAX_RATE in selected_metrics:
            self._add_ratio(
                results,
                FinancialMetric.EFFECTIVE_TAX_RATE,
                income_tax_expense,
                pretax_income,
                provider,
                granularity,
                period,
                "100 × income tax expense / pretax income",
                (
                    FinancialConcept.INCOME_TAX_EXPENSE,
                    FinancialConcept.PRETAX_INCOME,
                ),
            )

        nopat = self._nopat(operating_income, pretax_income, income_tax_expense)
        if FinancialMetric.NOPAT in selected_metrics and nopat is not None:
            self._add_amount(
                results,
                FinancialMetric.NOPAT,
                nopat,
                provider,
                granularity,
                period,
                "operating income × (1 - income tax expense / pretax income)",
                (
                    FinancialConcept.OPERATING_INCOME,
                    FinancialConcept.INCOME_TAX_EXPENSE,
                    FinancialConcept.PRETAX_INCOME,
                ),
            )

        ebitda = self._combine_amounts(
            (
                (operating_income, 1),
                (depreciation_and_amortization, 1),
            )
        )
        if FinancialMetric.EBITDA in selected_metrics and ebitda is not None:
            self._add_amount(
                results,
                FinancialMetric.EBITDA,
                ebitda,
                provider,
                granularity,
                period,
                "operating income + depreciation and amortization",
                (
                    FinancialConcept.OPERATING_INCOME,
                    FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
                ),
            )

        free_cash_flow = self._free_cash_flow(operating_cash_flow, capital_expenditures)
        if (
            FinancialMetric.FREE_CASH_FLOW in selected_metrics
            and free_cash_flow is not None
        ):
            self._add_amount(
                results,
                FinancialMetric.FREE_CASH_FLOW,
                free_cash_flow,
                provider,
                granularity,
                period,
                "operating cash flow - capital expenditures",
                (
                    FinancialConcept.OPERATING_CASH_FLOW,
                    FinancialConcept.CAPITAL_EXPENDITURES,
                ),
            )

        if FinancialMetric.FREE_CASH_FLOW_MARGIN in selected_metrics:
            self._add_ratio(
                results,
                FinancialMetric.FREE_CASH_FLOW_MARGIN,
                free_cash_flow,
                revenue,
                provider,
                granularity,
                period,
                "100 × (operating cash flow - capital expenditures) / revenue",
                (
                    FinancialConcept.OPERATING_CASH_FLOW,
                    FinancialConcept.CAPITAL_EXPENDITURES,
                    FinancialConcept.REVENUE,
                ),
            )

        operating_working_capital = operating_working_capital_value(current)
        if (
            FinancialMetric.OPERATING_WORKING_CAPITAL in selected_metrics
            and operating_working_capital is not None
        ):
            self._add_amount(
                results,
                FinancialMetric.OPERATING_WORKING_CAPITAL,
                operating_working_capital,
                provider,
                granularity,
                period,
                operating_working_capital_formula(current),
                operating_working_capital_input_concepts(current),
            )

        previous_operating_working_capital = (
            operating_working_capital_value(previous)
            if previous is not None
            else None
        )
        change_in_operating_working_capital = self._combine_amounts(
            (
                (operating_working_capital, 1),
                (previous_operating_working_capital, -1),
            )
        )
        if (
            FinancialMetric.CHANGE_IN_OPERATING_WORKING_CAPITAL in selected_metrics
            and change_in_operating_working_capital is not None
        ):
            self._add_amount(
                results,
                FinancialMetric.CHANGE_IN_OPERATING_WORKING_CAPITAL,
                change_in_operating_working_capital,
                provider,
                granularity,
                period,
                "current operating working capital - prior operating working capital",
                tuple(
                    sorted(OPERATING_WORKING_CAPITAL_CONCEPTS, key=lambda c: c.value)
                ),
            )

        gross_debt = self._gross_debt(current)
        if FinancialMetric.GROSS_DEBT in selected_metrics and gross_debt is not None:
            self._add_amount(
                results,
                FinancialMetric.GROSS_DEBT,
                gross_debt,
                provider,
                granularity,
                period,
                "short-term debt + current long-term debt + noncurrent long-term debt",
                tuple(sorted(DEBT_CONCEPTS, key=lambda c: c.value)),
            )

        net_debt = self._combine_amounts(((gross_debt, 1), (cash, -1)))
        if FinancialMetric.NET_DEBT in selected_metrics and net_debt is not None:
            self._add_amount(
                results,
                FinancialMetric.NET_DEBT,
                net_debt,
                provider,
                granularity,
                period,
                "gross debt - cash and equivalents",
                (
                    *tuple(sorted(DEBT_CONCEPTS, key=lambda c: c.value)),
                    FinancialConcept.CASH_AND_EQUIVALENTS,
                ),
            )

        tangible_book_equity = self._combine_amounts(
            (
                (equity, 1),
                (current.get(FinancialConcept.GOODWILL), -1),
                (current.get(FinancialConcept.INTANGIBLE_ASSETS_NET), -1),
            )
        )
        if (
            FinancialMetric.TANGIBLE_BOOK_EQUITY in selected_metrics
            and tangible_book_equity is not None
        ):
            self._add_amount(
                results,
                FinancialMetric.TANGIBLE_BOOK_EQUITY,
                tangible_book_equity,
                provider,
                granularity,
                period,
                "stockholders' equity - goodwill - net intangible assets",
                (
                    FinancialConcept.STOCKHOLDERS_EQUITY,
                    FinancialConcept.GOODWILL,
                    FinancialConcept.INTANGIBLE_ASSETS_NET,
                ),
            )

        fcff = self._combine_amounts(
            (
                (nopat, 1),
                (depreciation_and_amortization, 1),
                (capital_expenditures, -1),
                (change_in_operating_working_capital, -1),
            )
        )
        if FinancialMetric.FCFF in selected_metrics and fcff is not None:
            self._add_amount(
                results,
                FinancialMetric.FCFF,
                fcff,
                provider,
                granularity,
                period,
                "NOPAT + depreciation and amortization - capital expenditures - "
                "change in operating working capital",
                tuple(
                    sorted(METRIC_CONCEPTS[FinancialMetric.FCFF], key=lambda c: c.value)
                ),
            )

        if previous is not None:
            if FinancialMetric.RETURN_ON_ASSETS in selected_metrics:
                self._add_return_on_balance(
                    results,
                    FinancialMetric.RETURN_ON_ASSETS,
                    net_income,
                    assets,
                    previous.get(FinancialConcept.TOTAL_ASSETS),
                    provider,
                    granularity,
                    period,
                    "100 × net income / average total assets",
                    (FinancialConcept.NET_INCOME, FinancialConcept.TOTAL_ASSETS),
                )
            if FinancialMetric.RETURN_ON_EQUITY in selected_metrics:
                self._add_return_on_balance(
                    results,
                    FinancialMetric.RETURN_ON_EQUITY,
                    net_income,
                    equity,
                    previous.get(FinancialConcept.STOCKHOLDERS_EQUITY),
                    provider,
                    granularity,
                    period,
                    "100 × net income / average stockholders' equity",
                    (
                        FinancialConcept.NET_INCOME,
                        FinancialConcept.STOCKHOLDERS_EQUITY,
                    ),
                )

        if FinancialMetric.LIABILITIES_TO_ASSETS in selected_metrics:
            self._add_ratio(
                results,
                FinancialMetric.LIABILITIES_TO_ASSETS,
                liabilities,
                assets,
                provider,
                granularity,
                period,
                "100 × total liabilities / total assets",
                (FinancialConcept.TOTAL_LIABILITIES, FinancialConcept.TOTAL_ASSETS),
            )

        if FinancialMetric.CASH_TO_LIABILITIES in selected_metrics:
            self._add_ratio(
                results,
                FinancialMetric.CASH_TO_LIABILITIES,
                cash,
                liabilities,
                provider,
                granularity,
                period,
                "100 × cash and equivalents / total liabilities",
                (
                    FinancialConcept.CASH_AND_EQUIVALENTS,
                    FinancialConcept.TOTAL_LIABILITIES,
                ),
            )

        if FinancialMetric.OPERATING_CASH_FLOW_TO_NET_INCOME in selected_metrics:
            self._add_ratio(
                results,
                FinancialMetric.OPERATING_CASH_FLOW_TO_NET_INCOME,
                operating_cash_flow,
                net_income,
                provider,
                granularity,
                period,
                "100 × operating cash flow / net income",
                (FinancialConcept.OPERATING_CASH_FLOW, FinancialConcept.NET_INCOME),
            )

    @staticmethod
    def _nopat(
        operating_income: Optional[FinancialObservation],
        pretax_income: Optional[FinancialObservation],
        income_tax_expense: Optional[FinancialObservation],
    ) -> Optional[FinancialObservation]:
        if (
            operating_income is None
            or pretax_income is None
            or income_tax_expense is None
            or len(
                {
                    operating_income.unit,
                    pretax_income.unit,
                    income_tax_expense.unit,
                }
            )
            != 1
            or pretax_income.value == 0
        ):
            return None
        tax_rate = income_tax_expense.value / pretax_income.value
        return operating_income.model_copy(
            update={"value": operating_income.value * (Decimal(1) - tax_rate)}
        )

    @staticmethod
    def _gross_debt(
        observations: dict[FinancialConcept, FinancialObservation],
    ) -> Optional[FinancialObservation]:
        return FinancialMetricsService._combine_amounts(
            (
                (observations.get(FinancialConcept.SHORT_TERM_DEBT), 1),
                (observations.get(FinancialConcept.LONG_TERM_DEBT_CURRENT), 1),
                (observations.get(FinancialConcept.LONG_TERM_DEBT_NONCURRENT), 1),
            )
        )

    @staticmethod
    def _combine_amounts(
        terms: tuple[tuple[Optional[FinancialObservation], int], ...],
    ) -> Optional[FinancialObservation]:
        observations = [observation for observation, _ in terms]
        if any(observation is None for observation in observations):
            return None
        present = [
            observation for observation in observations if observation is not None
        ]
        if len({observation.unit for observation in present}) != 1:
            return None
        value = sum(
            (
                observation.value * coefficient
                for observation, coefficient in terms
                if observation is not None
            ),
            Decimal(0),
        )
        return present[0].model_copy(
            update={
                "value": value,
                "period_end": max(observation.period_end for observation in present),
            }
        )

    @staticmethod
    def _free_cash_flow(
        operating_cash_flow: Optional[FinancialObservation],
        capital_expenditures: Optional[FinancialObservation],
    ) -> Optional[FinancialObservation]:
        if (
            operating_cash_flow is None
            or capital_expenditures is None
            or operating_cash_flow.unit != capital_expenditures.unit
        ):
            return None
        return operating_cash_flow.model_copy(
            update={"value": operating_cash_flow.value - capital_expenditures.value}
        )

    def _add_return_on_balance(
        self,
        results: list[MetricObservation],
        metric: FinancialMetric,
        income: Optional[FinancialObservation],
        current_balance: Optional[FinancialObservation],
        previous_balance: Optional[FinancialObservation],
        provider: str,
        granularity: Granularity,
        period: PeriodKey,
        formula: str,
        input_concepts: tuple[FinancialConcept, ...],
    ) -> None:
        if (
            income is None
            or current_balance is None
            or previous_balance is None
            or len({income.unit, current_balance.unit, previous_balance.unit}) != 1
        ):
            return
        average_balance = (current_balance.value + previous_balance.value) / Decimal(2)
        if average_balance == 0:
            return
        self._add_metric(
            results,
            metric,
            income.value / average_balance * Decimal(100),
            "%",
            provider,
            granularity,
            period,
            income.period_end,
            formula,
            input_concepts,
        )

    def _add_ratio(
        self,
        results: list[MetricObservation],
        metric: FinancialMetric,
        numerator: Optional[FinancialObservation],
        denominator: Optional[FinancialObservation],
        provider: str,
        granularity: Granularity,
        period: PeriodKey,
        formula: str,
        input_concepts: tuple[FinancialConcept, ...],
        subtract_denominator: bool = False,
    ) -> None:
        if (
            numerator is None
            or denominator is None
            or numerator.unit != denominator.unit
            or denominator.value == 0
        ):
            return
        numerator_value = numerator.value
        if subtract_denominator:
            numerator_value -= denominator.value
        self._add_metric(
            results,
            metric,
            numerator_value / denominator.value * Decimal(100),
            "%",
            provider,
            granularity,
            period,
            numerator.period_end,
            formula,
            input_concepts,
        )

    def _add_amount(
        self,
        results: list[MetricObservation],
        metric: FinancialMetric,
        observation: FinancialObservation,
        provider: str,
        granularity: Granularity,
        period: PeriodKey,
        formula: str,
        input_concepts: tuple[FinancialConcept, ...],
    ) -> None:
        self._add_metric(
            results,
            metric,
            observation.value,
            observation.unit,
            provider,
            granularity,
            period,
            observation.period_end,
            formula,
            input_concepts,
        )

    @staticmethod
    def _add_metric(
        results: list[MetricObservation],
        metric: FinancialMetric,
        value: Decimal,
        unit: str,
        provider: str,
        granularity: Granularity,
        period: PeriodKey,
        period_end: datetime.date,
        formula: str,
        input_concepts: tuple[FinancialConcept, ...],
    ) -> None:
        results.append(
            MetricObservation(
                metric=metric,
                value=value,
                unit=unit,
                granularity=granularity,
                fiscal_year=period[0],
                fiscal_period=period[1],
                period_end=period_end,
                provider=provider,
                formula=formula,
                input_concepts=input_concepts,
            )
        )

    @staticmethod
    def _consecutive(
        previous: PeriodKey, current: PeriodKey, granularity: Granularity
    ) -> bool:
        if granularity == Granularity.ANNUAL:
            return (
                previous[1] == FiscalPeriod.FY
                and current[1] == FiscalPeriod.FY
                and current[0] == previous[0] + 1
            )
        quarter_number = {
            FiscalPeriod.Q1: 1,
            FiscalPeriod.Q2: 2,
            FiscalPeriod.Q3: 3,
            FiscalPeriod.Q4: 4,
        }
        if previous[1] not in quarter_number or current[1] not in quarter_number:
            return False
        previous_index = previous[0] * 4 + quarter_number[previous[1]]
        current_index = current[0] * 4 + quarter_number[current[1]]
        return current_index == previous_index + 1

    @staticmethod
    def _period_sort_key(period: PeriodKey):
        return period[0], FISCAL_PERIOD_PRIORITY[period[1]]

    @staticmethod
    def _observation_sort_key(observation: MetricObservation):
        granularity_order = 0 if observation.granularity == Granularity.ANNUAL else 1
        metric_order = list(FinancialMetric).index(observation.metric)
        return (
            granularity_order,
            observation.fiscal_year,
            FISCAL_PERIOD_PRIORITY[observation.fiscal_period],
            metric_order,
        )
