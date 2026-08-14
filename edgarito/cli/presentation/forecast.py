from decimal import Decimal

from edgarito.schemas.forecasting import (
    FcffForecast,
    ForecastAssumptionSource,
    SimplifiedFcfForecast,
)


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
