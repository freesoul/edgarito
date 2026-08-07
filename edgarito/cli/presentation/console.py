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
from edgarito.schemas.valuation.specialized import SpecializedValuationExtraction
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
    ComparableImpliedValuation,
    ComparableMultiplesReport,
    FcffDcfResult,
    ModelRole,
    ModelSuitability,
    MultipleStatus,
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
            f"Forecast seed: {forecast.seed_type.value} through "
            f"{(forecast.seed_period_end or forecast.base_period_end).isoformat()}",
            f"Seed methodology: {forecast.seed_methodology}",
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


class FcffDcfConsolePresenter:
    def render(self, result: FcffDcfResult, *, profile_name: str | None = None) -> str:
        identifier = result.ticker or result.company_id
        values = [
            result.enterprise_value,
            result.equity_value,
            result.capital_bridge.net_debt,
            result.capital_bridge.non_operating_assets,
            result.terminal_value.terminal_value,
            *(
                item.amount
                for item in result.explicit_forecast_present_value.cash_flows
            ),
        ]
        if result.share_repurchases is not None:
            values.extend(
                [
                    result.share_repurchases.total_cash_spent,
                    result.share_repurchases.present_value_cash_spent,
                    result.share_repurchases.residual_equity_value,
                ]
            )
        scale, suffix = self._scale_values(values)
        share_scale, share_suffix = self._scale_values(
            [result.capital_bridge.diluted_shares]
        )
        amount_unit = f"{result.unit} {suffix}".rstrip()
        timing = result.parameters.cash_flow_timing.value.replace("_", " ")
        terminal_method = result.parameters.terminal_method.value.replace("_", " ")
        lines = [
            f"{identifier} - {result.company_name}",
            f"Provider: {result.provider.upper()} | Valuation date: "
            f"{result.valuation_date.isoformat()}",
            f"Valuation profile: {profile_name or 'unspecified'}",
            f"Model: FCFF DCF | Timing: {timing}",
            "Forecast seed: "
            f"{result.forecast_seed_type} through "
            f"{result.forecast_seed_period_end.isoformat() if result.forecast_seed_period_end else '-'}",
            f"Forecast seed method: {result.forecast_seed_methodology}",
            f"WACC: {result.parameters.wacc:,.2f}% ({result.parameters.wacc_source})",
            f"Terminal method: {terminal_method}",
        ]
        if result.multistage_plan is not None:
            plan = result.multistage_plan
            stages = []
            if plan.explicit_growth_prefix_years:
                stages.append(f"{plan.explicit_growth_prefix_years} explicit")
            if plan.high_growth_years:
                stages.append(f"{plan.high_growth_years} high-growth")
            if plan.transition_years:
                stages.append(f"{plan.transition_years} transition")
            if plan.stable_years:
                stages.append(f"{plan.stable_years} stable")
            extension = (
                f"; extended from {plan.requested_years} requested years"
                if plan.extended_to_stable
                else ""
            )
            lines.append(
                "Projection: adaptive multistage | "
                f"{' + '.join(stages)} years{extension} | stable growth anchor "
                f"{plan.terminal_growth_rate:,.2f}%"
            )
            if plan.terminal_return_on_invested_capital is not None:
                details = (
                    "Stable reinvestment: "
                    f"{plan.terminal_reinvestment_rate:,.2f}% of NOPAT at "
                    f"{plan.terminal_return_on_invested_capital:,.2f}% terminal ROIC"
                )
                if plan.terminal_capex_to_revenue is not None:
                    details += (
                        f" | terminal capex/revenue "
                        f"{plan.terminal_capex_to_revenue:,.2f}%"
                    )
                if plan.depreciable_asset_life_years is not None:
                    details += (
                        f" | {plan.depreciable_asset_life_years}-year depreciable "
                        "asset life"
                    )
                lines.append(details)
                lines.append(
                    "Terminal ROIC resolution: "
                    f"{plan.terminal_roic_source or 'unspecified'} | confidence "
                    f"{plan.terminal_roic_confidence or 'unspecified'}"
                )
                if plan.terminal_roic_methodology:
                    lines.append(
                        f"Terminal ROIC method: {plan.terminal_roic_methodology}"
                    )
        if result.parameters.perpetual_growth_rate is not None:
            source = result.parameters.perpetual_growth_source or "explicit"
            lines.append(
                "Terminal growth: "
                f"{result.parameters.perpetual_growth_rate:,.2f}% ({source})"
            )
        if result.parameters.exit_multiple is not None:
            metric = result.parameters.exit_metric.value.upper()
            lines.append(
                f"Terminal multiple: {result.parameters.exit_multiple:,.2f}x {metric}"
            )
        if result.assumptions is not None:
            lines.extend(["", "Resolved assumptions:"])
            for assumption in result.assumptions.assumptions:
                provider = (
                    assumption.provenance.provider or assumption.provenance.origin.value
                )
                lines.append(
                    f"  {assumption.kind.value}: {assumption.value:,.3f} [{provider}]"
                )
        lines.extend(
            [
                "",
                f"{'Cash flow':<26}{'Period':>10}{'FCFF':>18}{'Factor':>12}{'PV':>18}",
                "-" * 84,
            ]
        )
        for item in result.explicit_forecast_present_value.cash_flows:
            lines.append(
                f"{(item.label or 'FCFF'):<26}{item.period:>10,.1f}"
                f"{item.amount / scale:>18,.1f}{item.discount_factor:>12,.4f}"
                f"{item.present_value / scale:>18,.1f}"
            )
        terminal = result.terminal_present_value
        lines.append(
            f"{'Terminal value':<26}{terminal.period:>10,.1f}"
            f"{terminal.amount / scale:>18,.1f}{terminal.discount_factor:>12,.4f}"
            f"{terminal.present_value / scale:>18,.1f}"
        )
        lines.extend(
            [
                "",
                f"Explicit FCFF PV ({amount_unit}): "
                f"{result.explicit_forecast_present_value.total_present_value / scale:,.1f}",
                f"Terminal value PV ({amount_unit}): "
                f"{result.terminal_present_value.present_value / scale:,.1f}",
                f"Enterprise value ({amount_unit}): "
                f"{result.enterprise_value / scale:,.1f}",
                f"Less net debt ({amount_unit}): "
                f"{result.capital_bridge.net_debt / scale:,.1f}",
                f"Add non-operating investments ({amount_unit}): "
                f"{result.capital_bridge.non_operating_assets / scale:,.1f}",
                f"Equity value ({amount_unit}): {result.equity_value / scale:,.1f}",
                f"Diluted shares ({share_suffix or 'units'}): "
                f"{result.capital_bridge.diluted_shares / share_scale:,.1f}",
            ]
        )
        if result.share_repurchases is not None:
            repurchases = result.share_repurchases
            lines.extend(
                [
                    "",
                    "Share repurchase analysis",
                    f"{'Period':<12}{'Cash spent':>18}{'Purchase price':>18}"
                    f"{'Shares retired':>18}{'PV cash':>18}",
                    "-" * 84,
                ]
            )
            for period in repurchases.periods:
                lines.append(
                    f"{f'FY{period.fiscal_year}E':<12}"
                    f"{period.cash_spent / scale:>18,.1f}"
                    f"{period.purchase_price:>18,.2f}"
                    f"{period.shares_repurchased / share_scale:>18,.2f}"
                    f"{period.present_value_cash_spent / scale:>18,.1f}"
                )
            lines.extend(
                [
                    f"Total buyback cash ({amount_unit}): "
                    f"{repurchases.total_cash_spent / scale:,.1f}",
                    f"PV of buyback cash ({amount_unit}): "
                    f"{repurchases.present_value_cash_spent / scale:,.1f}",
                    f"Projected shares retired ({share_suffix or 'units'}): "
                    f"{repurchases.shares_repurchased / share_scale:,.2f}",
                    f"Remaining diluted shares ({share_suffix or 'units'}): "
                    f"{repurchases.ending_shares / share_scale:,.2f}",
                    f"Residual equity value ({amount_unit}): "
                    f"{repurchases.residual_equity_value / scale:,.1f}",
                    f"Buyback accretion / (dilution): "
                    f"{repurchases.accretion_percentage:+,.2f}%",
                    f"Repurchase discount rate: {repurchases.discount_rate:,.2f}% "
                    f"({repurchases.discount_rate_source})",
                    f"Repurchase-price growth: {repurchases.price_growth_rate:,.2f}%",
                    f"Purchase-price basis: {repurchases.purchase_price_source}",
                    f"Buyback source: {repurchases.source}",
                ]
            )
        if result.terminal_value_percentage is not None:
            lines.append(
                "Terminal PV / enterprise value: "
                f"{result.terminal_value_percentage:,.1f}%"
            )
        lines.extend(
            [
                "",
                f"Net debt source: {result.capital_bridge.net_debt_source}",
                "Non-operating investments source: "
                f"{result.capital_bridge.non_operating_assets_source}",
                f"Shares source: {result.capital_bridge.shares_source}",
                "Capital bridge dates: "
                f"debt={result.capital_bridge.debt_date or 'explicit/unknown'}, "
                f"cash={result.capital_bridge.cash_date or 'explicit/unknown'}, "
                f"shares={result.capital_bridge.shares_date or 'explicit/unknown'}, "
                "non-operating assets="
                f"{result.capital_bridge.non_operating_assets_date or 'none/explicit'}",
                f"Debt scope: {result.capital_bridge.debt_scope}",
            ]
        )
        if result.warnings:
            lines.extend(["", "WARNINGS"])
            lines.extend(f"- {warning}" for warning in result.warnings)
        lines.extend(["", "VALUATION CONCLUSION"])
        if result.share_repurchases is not None:
            lines.extend(
                [
                    f"Value per share without buybacks ({result.unit}): "
                    f"{result.value_per_share:,.2f}",
                    f"Final value per share after buybacks ({result.unit}): "
                    f"{result.share_repurchases.value_per_remaining_share:,.2f}",
                ]
            )
        else:
            lines.append(
                f"Final value per share ({result.unit}): {result.value_per_share:,.2f}"
            )
        return "\n".join(lines)

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


class ComparableMultiplesConsolePresenter:
    def render(self, report: ComparableMultiplesReport) -> str:
        target = report.target
        selected = set(report.universe.selected_tickers)
        lines = [
            f"{target.ticker} - {target.company_name}",
            f"LTM period: {target.fundamentals.period_start.isoformat()} to "
            f"{target.fundamentals.period_end.isoformat()} | Price: "
            f"{target.price:,.2f} {target.currency} on {target.price_date.isoformat()}",
            f"Selected peers ({len(selected)}): "
            f"{', '.join(report.universe.selected_tickers) or '-'}",
            f"Candidate source: {report.universe.discovery_source} | "
            f"confidence {report.universe.discovery_confidence}",
            f"Discovery method: {report.universe.discovery_methodology}",
            "",
            "PEER SELECTION",
            f"{'Ticker':<12} {'Score':>7}  Decision / evidence",
            "-" * 78,
        ]
        for candidate in report.universe.candidates:
            decision = "selected" if candidate.selected else "excluded"
            detail = candidate.exclusions or candidate.reasons
            lines.append(
                f"{candidate.ticker:<12} {candidate.score:>6}/100  "
                f"{decision}: {'; '.join(detail) or '-'}"
            )

        target_multiples = {item.basis: item for item in target.multiples}
        summaries = {item.basis: item for item in report.summaries}
        bases = list(dict.fromkeys([*target_multiples, *summaries]))
        lines.extend(
            [
                "",
                "LTM MULTIPLES",
                f"{'Basis':<28} {'Target':>12} {'Peer median':>14} "
                f"{'Peer range':>21} {'N':>4}",
                "-" * 83,
            ]
        )
        for basis in bases:
            target_multiple = target_multiples.get(basis)
            summary = summaries.get(basis)
            target_value = (
                self._format_multiple(target_multiple.value, target_multiple.unit)
                if target_multiple
                and target_multiple.status == MultipleStatus.COMPUTED
                and target_multiple.value is not None
                else "-"
            )
            median_value = (
                self._format_multiple(summary.median, target_multiple.unit)
                if summary and target_multiple
                else "-"
            )
            peer_range = (
                f"{self._format_multiple(summary.minimum, target_multiple.unit)}–"
                f"{self._format_multiple(summary.maximum, target_multiple.unit)}"
                if summary and target_multiple
                else "-"
            )
            lines.append(
                f"{ValuationSelectionConsolePresenter._label(basis.value):<28} "
                f"{target_value:>12} {median_value:>14} {peer_range:>21} "
                f"{summary.sample_size if summary else 0:>4}"
            )

        warnings = [
            *target.warnings,
            *(warning for peer in report.peers for warning in peer.warnings),
            *report.warnings,
        ]
        if warnings:
            lines.extend(["", "WARNINGS"])
            lines.extend(f"- {warning}" for warning in dict.fromkeys(warnings))
        return "\n".join(lines)

    @staticmethod
    def _format_multiple(value: Decimal, unit: str) -> str:
        return f"{value:,.2f}%" if unit == "percent" else f"{value:,.2f}x"


class ComparableImpliedValuationConsolePresenter:
    def render(self, result: ComparableImpliedValuation) -> str:
        multiple = result.resolved_multiple

        def anchor(value):
            return f"{value:,.2f}x" if value is not None else "unavailable"

        peer_label = {
            "forward": "Peer forward baseline",
            "current_ltm_fallback": "Peer baseline (current LTM)",
            "dcf_fallback": "Base multiple (DCF fallback)",
        }.get(multiple.peer_anchor_source, "Peer/base multiple")

        lines = [
            "MARKET-RELATIVE IMPLIED VALUATION",
            f"{result.ticker or result.company_id} - {result.company_name}",
            f"Valuation date: {result.valuation_date.isoformat()} | Target date: "
            f"{result.target_date.isoformat()} ({result.horizon_years:,.2f} years)",
            f"Basis: {ValuationSelectionConsolePresenter._label(result.basis.value)} | "
            f"Metric: {result.forecast_metric_label}",
            "",
            "MULTIPLE RESOLUTION",
            f"{peer_label + ':':<36}{anchor(multiple.market_anchor)}",
            f"DCF-implied forward multiple:   {anchor(multiple.fundamental_anchor)}",
            "DCF-implied premium vs peers:   "
            + (
                f"{multiple.fundamental_premium:+,.1%}"
                if multiple.fundamental_premium is not None
                else "unavailable"
            ),
            f"Target historical median:       {anchor(multiple.historical_anchor)}",
            "Target historical IQR:          "
            + (
                f"{multiple.historical_percentile_25:,.2f}x-"
                f"{multiple.historical_percentile_75:,.2f}x"
                if multiple.historical_percentile_25 is not None
                and multiple.historical_percentile_75 is not None
                else "unavailable"
            ),
            f"Historical observations:         {multiple.historical_sample_size}",
            "Historical multiple volatility: "
            + (
                f"{multiple.historical_volatility:,.1%}"
                if multiple.historical_volatility is not None
                else "unavailable"
            ),
            "Historical multiple trend:      "
            + (
                f"{multiple.historical_trend:+,.1%}"
                if multiple.historical_trend is not None
                else "unavailable"
            ),
            f"Current target comparative multiple: "
            f"{anchor(multiple.current_target_anchor)}",
            "Current target premium vs base: "
            + (
                f"{multiple.observed_premium:+,.1%}"
                if multiple.observed_premium is not None
                else "unavailable"
            ),
            "Historical long-run premium:    "
            + (
                f"{multiple.historical_peer_premium:+,.1%}"
                if multiple.historical_peer_premium is not None
                else " unavailable"
            ),
            f"Synchronized premium observations: "
            f"{multiple.premium_history_sample_size}",
            "Median premium observation interval: "
            + (
                f"{multiple.premium_observation_interval_years:,.2f} years"
                if multiple.premium_observation_interval_years is not None
                else "unavailable"
            ),
            "Raw AR(1) phi (deviation persistence): "
            + (
                f"{multiple.premium_mean_reversion_beta:,.2f}"
                if multiple.premium_mean_reversion_beta is not None
                else "unavailable"
            ),
            f"Shrunk AR(1) phi:               "
            f"{multiple.shrunk_premium_persistence:,.2f}",
            "Statistical premium at horizon: "
            + (
                f"{multiple.statistical_premium:+,.1%}"
                if multiple.statistical_premium is not None
                else "unavailable"
            ),
            f"Premium-history weight:         {multiple.premium_history_weight:,.1%}",
            f"Fundamental quality support:    {multiple.fundamental_support:,.1%}",
            f"Horizon evidence retention:     {multiple.horizon_retention:,.1%}",
            f"Statistical-anchor evidence weight: {multiple.persistence_factor:,.1%}",
            "Resolved target premium:        "
            + (
                f"{multiple.resolved_premium:+,.1%}"
                if multiple.resolved_premium is not None
                else "unavailable"
            ),
            f"Resolved forward multiple:      {multiple.point_estimate:,.2f}x",
            f"Reasonable range:                {multiple.lower_bound:,.2f}x-"
            f"{multiple.upper_bound:,.2f}x",
            "Range evidence: DCF anchor, peer IQR, and synchronized premium IQR",
            "Confidence:",
            f"  peer baseline:                {multiple.peer_confidence.value}",
            f"  target history:               "
            f"{multiple.target_history_confidence.value}",
            f"  premium persistence:          "
            f"{multiple.premium_persistence_confidence.value}",
            f"  overall relative valuation:   {multiple.confidence.value}",
            f"Peer sample: {multiple.sample_size}",
            "",
            f"{'Case':<12}{'Multiple':>12}{'Target-date price':>22}{'Present value':>20}",
            "-" * 66,
        ]
        for case in (result.lower_case, result.point_case, result.upper_case):
            lines.append(
                f"{case.label:<12}{case.multiple:>11,.2f}x"
                f"{case.implied_value_per_share:>18,.2f} {result.currency}"
                f"{case.present_value_per_share:>16,.2f} {result.currency}"
            )
        if result.intrinsic_value_per_share is not None:
            difference = (
                result.point_case.present_value_per_share
                - result.intrinsic_value_per_share
            )
            lines.extend(
                [
                    "",
                    "MODEL COMPARISON",
                    f"Intrinsic FCFF DCF:             "
                    f"{result.intrinsic_value_per_share:,.2f} {result.currency}",
                    f"Relative target-date price:     "
                    f"{result.point_case.implied_value_per_share:,.2f} "
                    f"{result.currency}",
                    f"Relative present-value equivalent today: "
                    f"{result.point_case.present_value_per_share:,.2f} "
                    f"{result.currency}",
                    f"Market-premium difference:      {difference:+,.2f} "
                    f"{result.currency}",
                    "The DCF values forecast cash flows; the relative estimate "
                    "also retains an evidence-constrained market premium.",
                ]
            )
        if result.current_price is not None:
            lines.extend(
                [
                    f"Current price:                   {result.current_price:,.2f} "
                    f"{result.currency}",
                    "Current-price implied forward multiple: "
                    + (
                        f"{result.current_price_implied_multiple:,.2f}x "
                        f"{ValuationSelectionConsolePresenter._label(result.basis.value)}"
                        if result.current_price_implied_multiple is not None
                        else "unavailable"
                    ),
                ]
            )
        if (
            result.analyst_target_price is not None
            and result.analyst_target_implied_multiple is not None
        ):
            lines.extend(
                [
                    f"Analyst target price:            "
                    f"{result.analyst_target_price:,.2f} {result.currency}",
                    "Analyst target vs resolved target-date price: "
                    f"{result.analyst_target_price - result.point_case.implied_value_per_share:+,.2f} "
                    f"{result.currency}",
                    f"Analyst-target implied multiple: "
                    f"{result.analyst_target_implied_multiple:,.2f}x "
                    f"{ValuationSelectionConsolePresenter._label(result.basis.value)}",
                ]
            )
        if result.warnings:
            lines.extend(["", "WARNINGS"])
            lines.extend(f"- {warning}" for warning in result.warnings)
        return "\n".join(lines)


class SpecializedExtractionConsolePresenter:
    def render(self, extraction: SpecializedValuationExtraction) -> str:
        identifier = extraction.ticker or f"CIK {extraction.company_id}"
        lines = [
            f"{identifier} - {extraction.company_name}",
            f"Extractor: {extraction.input_type.label} | "
            f"Readiness: {extraction.readiness.value}",
            f"Source scope: {extraction.source_scope}",
        ]
        if extraction.fields:
            lines.extend(
                [
                    "",
                    "EXTRACTED FIELDS",
                    f"{'Period':<17} {'Field':<43} {'Value':>18} {'Unit':<10} Origin",
                    "-" * 100,
                ]
            )
            for field in extraction.fields:
                period_kind = {
                    "annual": "",
                    "quarterly": "quarter",
                    "year_to_date": "YTD",
                    "instant": "instant",
                }[field.period_kind.value]
                period_label = (
                    f"FY{field.fiscal_year}"
                    if field.fiscal_period == "FY"
                    else f"FY{field.fiscal_year} {field.fiscal_period} {period_kind}"
                )
                lines.append(
                    f"{period_label:<17} {field.name:<43} "
                    f"{field.value:>18,.2f} {field.unit:<10} {field.origin.value}"
                )
                lines.append(f"        Source: {', '.join(field.source_concepts)}")
                if field.derivation:
                    lines.append(f"        Formula: {field.derivation}")
        else:
            lines.extend(["", "No supported standard facts were extracted."])

        if extraction.missing_inputs:
            lines.extend(["", "STILL REQUIRED"])
            lines.extend(f"- {item}" for item in extraction.missing_inputs)
        if extraction.limitations:
            lines.extend(["", "LIMITATIONS"])
            lines.extend(f"- {item}" for item in extraction.limitations)
        return "\n".join(lines)
