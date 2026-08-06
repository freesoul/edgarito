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


METRIC_CONCEPTS: dict[FinancialMetric, frozenset[FinancialConcept]] = {
    FinancialMetric.REVENUE_GROWTH: frozenset({FinancialConcept.REVENUE}),
    FinancialMetric.OPERATING_MARGIN: frozenset(
        {FinancialConcept.OPERATING_INCOME, FinancialConcept.REVENUE}
    ),
    FinancialMetric.NET_MARGIN: frozenset(
        {FinancialConcept.NET_INCOME, FinancialConcept.REVENUE}
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
        net_income = current.get(FinancialConcept.NET_INCOME)
        assets = current.get(FinancialConcept.TOTAL_ASSETS)
        liabilities = current.get(FinancialConcept.TOTAL_LIABILITIES)
        equity = current.get(FinancialConcept.STOCKHOLDERS_EQUITY)
        cash = current.get(FinancialConcept.CASH_AND_EQUIVALENTS)
        operating_cash_flow = current.get(FinancialConcept.OPERATING_CASH_FLOW)
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
