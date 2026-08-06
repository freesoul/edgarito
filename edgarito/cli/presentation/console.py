from decimal import Decimal

from edgarito.enums.edgar.period import FISCAL_PERIOD_PRIORITY, FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.classification import (
    NormalizedCompanyClassification,
)
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    FinancialStatement,
    NormalizedCompanyFinancials,
)
from edgarito.services.forecasting import (
    FcffForecast,
    ForecastAssumptionSource,
    SimplifiedFcfForecast,
)
from edgarito.services.metrics.models import (
    CompanyMetrics,
    FinancialMetric,
    MetricObservation,
)
from edgarito.services.valuation import (
    ModelRole,
    ModelSuitability,
    ValuationSelection,
)

CONCEPT_ORDER = {concept: index for index, concept in enumerate(FinancialConcept)}
STATEMENT_LABELS = {
    FinancialStatement.INCOME_STATEMENT: "Income statement",
    FinancialStatement.BALANCE_SHEET: "Balance sheet",
    FinancialStatement.CASH_FLOW: "Cash flow statement",
}
METRIC_ORDER = {metric: index for index, metric in enumerate(FinancialMetric)}


class ClassificationConsolePresenter:
    def render(self, classification: NormalizedCompanyClassification) -> str:
        lines = [
            f"{classification.ticker} - {classification.company_name}",
            f"Provider: {classification.provider.upper()} | "
            f"Company ID: {classification.company_id}",
            "",
            f"Sector: {classification.sector.value if classification.sector else '-'}",
            f"Industry: {classification.industry or '-'}",
            f"Country: {classification.country or '-'}",
            f"Exchange: {classification.exchange or '-'}",
            "",
            f"Source sector: {classification.source_sector or '-'}",
            f"Source industry: {classification.source_industry or '-'}",
            f"Industry taxonomy: {classification.industry_taxonomy}",
        ]
        return "\n".join(lines)


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


class ForecastConsolePresenter:
    def render(self, forecast: FcffForecast | SimplifiedFcfForecast) -> str:
        if isinstance(forecast, SimplifiedFcfForecast):
            return self._render_simplified(forecast)
        return self._render_fcff(forecast)

    def _render_fcff(self, forecast: FcffForecast) -> str:
        identifier = forecast.ticker or f"CIK {forecast.company_id}"
        scale, suffix = self._scale_values(self._fcff_amounts(forecast))
        amount_unit = f"{forecast.unit} {suffix}".rstrip()
        periods = [f"FY{o.fiscal_year}E" for o in forecast.observations]
        label_width = 39
        value_width = max(13, max((len(period) for period in periods), default=0) + 2)
        header = f"{'Metric':<{label_width}}" + "".join(
            f"{period:>{value_width}}" for period in periods
        )
        base_fcff = (
            "-"
            if forecast.base_fcff is None
            else f"{forecast.base_fcff / scale:,.1f} {amount_unit}"
        )
        lines = [
            f"{identifier} - {forecast.company_name}",
            f"Provider: {forecast.provider.upper()} | CIK: {forecast.company_id}",
            "Method: driver-based FCFF",
            f"Base FY{forecast.base_fiscal_year}: "
            f"Revenue {forecast.base_revenue / scale:,.1f} {amount_unit} | "
            f"EBIT {forecast.base_operating_income / scale:,.1f} {amount_unit} | "
            f"FCFF {base_fcff}",
            f"Base operating NWC: "
            f"{forecast.base_operating_working_capital / scale:,.1f} {amount_unit}",
            "Assumption sources:",
        ]
        lines.extend(
            f"  {driver.label}: "
            f"{self._source_label(source, forecast.historical_fiscal_years)}"
            for driver, source in forecast.assumption_sources.items()
        )
        lines.extend(
            [
                "",
                header,
                "-" * len(header),
                self._row(
                    "Revenue Growth (%)",
                    [item.revenue_growth for item in forecast.observations],
                    value_width,
                    label_width,
                    percent=True,
                ),
                self._row(
                    f"Revenue ({amount_unit})",
                    [item.revenue / scale for item in forecast.observations],
                    value_width,
                    label_width,
                ),
                self._row(
                    "Operating Margin (%)",
                    [item.operating_margin for item in forecast.observations],
                    value_width,
                    label_width,
                    percent=True,
                ),
                self._row(
                    f"EBIT ({amount_unit})",
                    [item.operating_income / scale for item in forecast.observations],
                    value_width,
                    label_width,
                ),
                self._row(
                    "Tax Rate (%)",
                    [item.tax_rate for item in forecast.observations],
                    value_width,
                    label_width,
                    percent=True,
                ),
                self._row(
                    f"NOPAT ({amount_unit})",
                    [item.nopat / scale for item in forecast.observations],
                    value_width,
                    label_width,
                ),
                self._row(
                    "D&A / Revenue (%)",
                    [item.depreciation_to_revenue for item in forecast.observations],
                    value_width,
                    label_width,
                    percent=True,
                ),
                self._row(
                    f"D&A ({amount_unit})",
                    [
                        item.depreciation_and_amortization / scale
                        for item in forecast.observations
                    ],
                    value_width,
                    label_width,
                ),
                self._row(
                    "Capex / Revenue (%)",
                    [item.capex_to_revenue for item in forecast.observations],
                    value_width,
                    label_width,
                    percent=True,
                ),
                self._row(
                    f"Capital Expenditures ({amount_unit})",
                    [
                        item.capital_expenditures / scale
                        for item in forecast.observations
                    ],
                    value_width,
                    label_width,
                ),
                self._row(
                    "Operating NWC / Revenue (%)",
                    [
                        item.operating_working_capital_to_revenue
                        for item in forecast.observations
                    ],
                    value_width,
                    label_width,
                    percent=True,
                ),
                self._row(
                    f"Operating NWC ({amount_unit})",
                    [
                        item.operating_working_capital / scale
                        for item in forecast.observations
                    ],
                    value_width,
                    label_width,
                ),
                self._row(
                    f"Change in Operating NWC ({amount_unit})",
                    [
                        item.change_in_operating_working_capital / scale
                        for item in forecast.observations
                    ],
                    value_width,
                    label_width,
                ),
                self._row(
                    f"FCFF ({amount_unit})",
                    [item.fcff / scale for item in forecast.observations],
                    value_width,
                    label_width,
                ),
            ]
        )
        return "\n".join(lines)

    def _render_simplified(self, forecast: SimplifiedFcfForecast) -> str:
        identifier = forecast.ticker or f"CIK {forecast.company_id}"
        scale, suffix = self._scale_values(self._simplified_amounts(forecast))
        amount_unit = f"{forecast.unit} {suffix}".rstrip()
        periods = [f"FY{o.fiscal_year}E" for o in forecast.observations]
        label_width = 30
        value_width = max(13, max((len(period) for period in periods), default=0) + 2)
        header = f"{'Metric':<{label_width}}" + "".join(
            f"{period:>{value_width}}" for period in periods
        )

        lines = [
            f"{identifier} - {forecast.company_name}",
            f"Provider: {forecast.provider.upper()} | CIK: {forecast.company_id}",
            "Method: simplified projected revenue × free cash flow margin",
            f"Base FY{forecast.base_fiscal_year}: "
            f"Revenue {forecast.base_revenue / scale:,.1f} {amount_unit} | "
            f"Free Cash Flow {forecast.base_free_cash_flow / scale:,.1f} {amount_unit}",
            "Revenue growth assumptions: "
            f"{self._source_label(forecast.revenue_growth_source, forecast.historical_fiscal_years)}",
            "FCF margin assumptions: "
            f"{self._source_label(forecast.free_cash_flow_margin_source, forecast.historical_fiscal_years)}",
            "",
            header,
            "-" * len(header),
            self._row(
                "Revenue Growth (%)",
                [o.revenue_growth for o in forecast.observations],
                value_width,
                label_width,
                percent=True,
            ),
            self._row(
                f"Revenue ({amount_unit})",
                [o.revenue / scale for o in forecast.observations],
                value_width,
                label_width,
            ),
            self._row(
                "FCF Margin (%)",
                [o.free_cash_flow_margin for o in forecast.observations],
                value_width,
                label_width,
                percent=True,
            ),
            self._row(
                f"Free Cash Flow ({amount_unit})",
                [o.free_cash_flow / scale for o in forecast.observations],
                value_width,
                label_width,
            ),
        ]
        return "\n".join(lines)

    @staticmethod
    def _source_label(
        source: ForecastAssumptionSource, historical_fiscal_years: tuple[int, ...]
    ) -> str:
        if source == ForecastAssumptionSource.EXPLICIT:
            return "explicit"
        years = historical_fiscal_years
        period = f"FY{years[0]}" if len(years) == 1 else f"FY{years[0]}–FY{years[-1]}"
        return f"trailing average from {period}"

    @staticmethod
    def _row(
        label: str,
        values: list[Decimal],
        value_width: int,
        label_width: int,
        percent: bool = False,
    ) -> str:
        row = f"{label:<{label_width}}"
        for value in values:
            rendered = f"{value:,.1f}{'%' if percent else ''}"
            row += f"{rendered:>{value_width}}"
        return row

    @staticmethod
    def _scale_values(values: list[Decimal]) -> tuple[Decimal, str]:
        largest = max((abs(value) for value in values), default=Decimal(0))
        if largest >= Decimal("1000000000"):
            return Decimal("1000000000"), "B"
        if largest >= Decimal("1000000"):
            return Decimal("1000000"), "M"
        if largest >= Decimal("1000"):
            return Decimal("1000"), "K"
        return Decimal(1), ""

    @staticmethod
    def _simplified_amounts(forecast: SimplifiedFcfForecast) -> list[Decimal]:
        values = [forecast.base_revenue, forecast.base_free_cash_flow]
        values.extend(
            value
            for observation in forecast.observations
            for value in (observation.revenue, observation.free_cash_flow)
        )
        return values

    @staticmethod
    def _fcff_amounts(forecast: FcffForecast) -> list[Decimal]:
        values = [
            forecast.base_revenue,
            forecast.base_operating_income,
            forecast.base_nopat,
            forecast.base_depreciation_and_amortization,
            forecast.base_capital_expenditures,
            forecast.base_operating_working_capital,
        ]
        if forecast.base_fcff is not None:
            values.append(forecast.base_fcff)
        values.extend(
            value
            for observation in forecast.observations
            for value in (
                observation.revenue,
                observation.operating_income,
                observation.nopat,
                observation.depreciation_and_amortization,
                observation.capital_expenditures,
                observation.operating_working_capital,
                observation.change_in_operating_working_capital,
                observation.fcff,
            )
        )
        return values


class ValuationSelectionConsolePresenter:
    def render(self, selection: ValuationSelection) -> str:
        profile = selection.profile
        identifier = profile.ticker or f"CIK {profile.company_id}"
        lines = [
            f"{identifier} - {profile.company_name}",
            f"Economic profile: {self._label(profile.business_archetype.value)}",
            f"Sector: {profile.sector.value if profile.sector else '-'} | "
            f"Industry: {profile.industry or '-'}",
            f"Lifecycle: {self._label(profile.lifecycle.value)} | "
            f"Cyclicality: {self._label(profile.cyclicality.value)}",
        ]
        if profile.economic_traits:
            traits = ", ".join(
                self._label(trait.value)
                for trait in sorted(
                    profile.economic_traits, key=lambda item: item.value
                )
            )
            lines.append(f"Economic traits: {traits}")
        if profile.annual_fiscal_years:
            lines.append(
                f"Annual history: FY{profile.annual_fiscal_years[0]}–"
                f"FY{profile.annual_fiscal_years[-1]}"
            )

        for role in (
            ModelRole.PRIMARY,
            ModelRole.CONDITIONAL,
            ModelRole.CROSSCHECK,
            ModelRole.NOT_RECOMMENDED,
        ):
            models = [model for model in selection.models if model.role == role]
            if not models:
                continue
            lines.extend(["", self._label(role.value).upper()])
            for model in models:
                lines.extend(self._render_model(model))
        return "\n".join(lines)

    def _render_model(self, model: ModelSuitability) -> list[str]:
        lines = [
            f"{model.model.label} — suitability {model.suitability_score}/100; "
            f"data {self._label(model.data_readiness.value)}"
        ]
        if model.forecast_profile:
            lines.append(
                f"  Forecast profile: {self._label(model.forecast_profile.value)}"
            )
        if model.relative_bases:
            bases = ", ".join(
                self._label(basis.value) for basis in model.relative_bases
            )
            lines.append(f"  Suggested bases: {bases}")
        for reason in model.reasons:
            lines.append(f"  + {reason}")
        for rejection in model.hard_rejections:
            lines.append(f"  ! {rejection}")
        for limitation in model.limitations:
            lines.append(f"  ~ {limitation}")
        if model.missing_inputs:
            missing = ", ".join(
                self._label(item.value)
                for item in sorted(model.missing_inputs, key=lambda item: item.value)
            )
            lines.append(f"  Missing: {missing}")
        return lines

    @staticmethod
    def _label(value: str) -> str:
        acronyms = {
            "affo": "AFFO",
            "dcf": "DCF",
            "ddm": "DDM",
            "ebit": "EBIT",
            "ebitda": "EBITDA",
            "ev": "EV",
            "fcf": "FCF",
            "fcfe": "FCFE",
            "fcff": "FCFF",
            "nav": "NAV",
            "roe": "ROE",
            "sotp": "SOTP",
            "wacc": "WACC",
        }
        return " ".join(acronyms.get(part, part.title()) for part in value.split("_"))
