from decimal import Decimal

from edgarito.enums.edgar.period import FISCAL_PERIOD_PRIORITY, FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    FinancialStatement,
    NormalizedCompanyFinancials,
)
from edgarito.services.metrics.models import (
    CompanyMetrics,
    FinancialMetric,
    MetricObservation,
)

CONCEPT_ORDER = {concept: index for index, concept in enumerate(FinancialConcept)}
STATEMENT_LABELS = {
    FinancialStatement.INCOME_STATEMENT: "Income statement",
    FinancialStatement.BALANCE_SHEET: "Balance sheet",
    FinancialStatement.CASH_FLOW: "Cash flow statement",
}
METRIC_ORDER = {metric: index for index, metric in enumerate(FinancialMetric)}


class FinancialsConsolePresenter:
    def render(self, financials: NormalizedCompanyFinancials, limit: int = 5) -> str:
        identifier = financials.ticker or f"CIK {financials.company_id}"
        lines = [
            f"{identifier} - {financials.company_name}",
            f"Provider: {financials.provider.upper()} | CIK: {financials.company_id}",
        ]

        granularities = []
        if any(o.granularity == Granularity.ANNUAL for o in financials.observations):
            granularities.append(Granularity.ANNUAL)
        if any(o.granularity == Granularity.QUARTERLY for o in financials.observations):
            granularities.append(Granularity.QUARTERLY)

        for granularity in granularities:
            lines.extend(["", granularity.value.upper()])
            observations = [
                observation
                for observation in financials.observations
                if observation.granularity == granularity
            ]
            periods = sorted(
                {o.period_key for o in observations},
                key=lambda item: (item[0], FISCAL_PERIOD_PRIORITY[item[1]]),
            )[-limit:]
            period_set = set(periods)
            observations = [o for o in observations if o.period_key in period_set]

            for statement in FinancialStatement:
                statement_observations = [
                    observation
                    for observation in observations
                    if observation.statement == statement
                ]
                if not statement_observations:
                    continue
                lines.extend(["", STATEMENT_LABELS[statement]])
                lines.extend(
                    self._render_table(statement_observations, periods, granularity)
                )

        if any(observation.is_derived for observation in financials.observations):
            lines.extend(["", "* Derived from reported SEC values"])
        if not financials.observations:
            lines.extend(["", "No matching financial observations were found."])
        return "\n".join(lines)

    def _render_table(
        self,
        observations: list[FinancialObservation],
        periods: list[tuple[int, FiscalPeriod]],
        granularity: Granularity,
    ) -> list[str]:
        by_concept: dict[FinancialConcept, list[FinancialObservation]] = {}
        for observation in observations:
            by_concept.setdefault(observation.concept, []).append(observation)

        period_labels = [self._period_label(period, granularity) for period in periods]
        concept_width = max(24, max(len(concept.label) for concept in by_concept) + 9)
        value_width = max(
            13,
            max((len(label) for label in period_labels), default=0) + 2,
        )
        header = f"{'Metric':<{concept_width}}" + "".join(
            f"{label:>{value_width}}" for label in period_labels
        )
        lines = [header, "-" * len(header)]

        for concept in sorted(by_concept, key=lambda item: CONCEPT_ORDER[item]):
            concept_observations = by_concept[concept]
            scale, suffix = self._scale(concept_observations)
            values = {o.period_key: o for o in concept_observations}
            row_label = f"{concept.label} ({concept_observations[0].unit} {suffix})"
            row = f"{row_label:<{concept_width}}"
            for period in periods:
                observation = values.get(period)
                formatted = (
                    "-"
                    if observation is None
                    else self._format_value(observation, scale)
                )
                row += f"{formatted:>{value_width}}"
            lines.append(row)
        return lines

    @staticmethod
    def _period_label(
        period: tuple[int, FiscalPeriod], granularity: Granularity
    ) -> str:
        fiscal_year, fiscal_period = period
        if granularity == Granularity.ANNUAL:
            return f"FY{fiscal_year}"
        return f"FY{fiscal_year} {fiscal_period.value}"

    @staticmethod
    def _scale(observations: list[FinancialObservation]) -> tuple[Decimal, str]:
        largest = max((abs(o.value) for o in observations), default=Decimal(0))
        if largest >= Decimal("1000000000"):
            return Decimal("1000000000"), "B"
        if largest >= Decimal("1000000"):
            return Decimal("1000000"), "M"
        if largest >= Decimal("1000"):
            return Decimal("1000"), "K"
        return Decimal(1), ""

    @staticmethod
    def _format_value(observation: FinancialObservation, scale: Decimal) -> str:
        value = observation.value / scale
        marker = "*" if observation.is_derived else ""
        return f"{value:,.1f}{marker}"


class MetricsConsolePresenter:
    def render(self, metrics: CompanyMetrics, limit: int = 5) -> str:
        identifier = metrics.ticker or f"CIK {metrics.company_id}"
        lines = [
            f"{identifier} - {metrics.company_name}",
            f"Provider: {metrics.provider.upper()} | CIK: {metrics.company_id}",
        ]

        for granularity in Granularity:
            observations = [
                observation
                for observation in metrics.observations
                if observation.granularity == granularity
            ]
            if not observations:
                continue
            periods = sorted(
                {observation.period_key for observation in observations},
                key=lambda item: (item[0], FISCAL_PERIOD_PRIORITY[item[1]]),
            )[-limit:]
            period_set = set(periods)
            observations = [
                observation
                for observation in observations
                if observation.period_key in period_set
            ]
            lines.extend(["", granularity.value.upper(), ""])
            lines.extend(self._render_table(observations, periods, granularity))

        if not metrics.observations:
            lines.extend(
                ["", "No metrics could be calculated from the available data."]
            )
        return "\n".join(lines)

    def _render_table(
        self,
        observations: list[MetricObservation],
        periods: list[tuple[int, FiscalPeriod]],
        granularity: Granularity,
    ) -> list[str]:
        by_metric: dict[FinancialMetric, list[MetricObservation]] = {}
        for observation in observations:
            by_metric.setdefault(observation.metric, []).append(observation)

        period_labels = [
            FinancialsConsolePresenter._period_label(period, granularity)
            for period in periods
        ]
        metric_width = max(
            32,
            max(
                len(self._row_label(metric, metric_observations))
                for metric, metric_observations in by_metric.items()
            )
            + 2,
        )
        value_width = max(
            13,
            max((len(label) for label in period_labels), default=0) + 2,
        )
        header = f"{'Metric':<{metric_width}}" + "".join(
            f"{label:>{value_width}}" for label in period_labels
        )
        lines = [header, "-" * len(header)]

        for metric in sorted(by_metric, key=lambda item: METRIC_ORDER[item]):
            metric_observations = by_metric[metric]
            values = {
                observation.period_key: observation
                for observation in metric_observations
            }
            scale, suffix = self._scale(metric_observations)
            row_label = self._row_label(metric, metric_observations, suffix)
            row = f"{row_label:<{metric_width}}"
            for period in periods:
                observation = values.get(period)
                formatted = (
                    "-"
                    if observation is None
                    else self._format_value(observation, scale)
                )
                row += f"{formatted:>{value_width}}"
            lines.append(row)
        return lines

    @staticmethod
    def _row_label(
        metric: FinancialMetric,
        observations: list[MetricObservation],
        suffix: str = "",
    ) -> str:
        unit = observations[0].unit
        if unit == "%":
            return f"{metric.label} (%)"
        rendered_unit = f"{unit} {suffix}".rstrip()
        return f"{metric.label} ({rendered_unit})"

    @staticmethod
    def _scale(observations: list[MetricObservation]) -> tuple[Decimal, str]:
        if observations[0].unit == "%":
            return Decimal(1), ""
        largest = max(
            (abs(observation.value) for observation in observations),
            default=Decimal(0),
        )
        if largest >= Decimal("1000000000"):
            return Decimal("1000000000"), "B"
        if largest >= Decimal("1000000"):
            return Decimal("1000000"), "M"
        if largest >= Decimal("1000"):
            return Decimal("1000"), "K"
        return Decimal(1), ""

    @staticmethod
    def _format_value(observation: MetricObservation, scale: Decimal) -> str:
        value = observation.value / scale
        suffix = "%" if observation.unit == "%" else ""
        return f"{value:,.1f}{suffix}"
