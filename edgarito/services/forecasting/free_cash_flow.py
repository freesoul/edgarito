import datetime
from decimal import Decimal

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.forecasting.models import (
    ForecastAssumptionSource,
    FreeCashFlowForecast,
    FreeCashFlowForecastObservation,
    FreeCashFlowForecastParameters,
)


class FreeCashFlowForecastService:
    """Build deterministic annual FCF forecasts from normalized financials."""

    _REQUIRED_CONCEPTS = {
        FinancialConcept.REVENUE,
        FinancialConcept.OPERATING_CASH_FLOW,
        FinancialConcept.CAPITAL_EXPENDITURES,
    }

    @classmethod
    def required_concepts(cls) -> set[FinancialConcept]:
        return set(cls._REQUIRED_CONCEPTS)

    def forecast(
        self,
        financials: NormalizedCompanyFinancials,
        parameters: FreeCashFlowForecastParameters | None = None,
    ) -> FreeCashFlowForecast:
        parameters = parameters or FreeCashFlowForecastParameters()
        periods = self._complete_annual_periods(financials)
        if not periods:
            raise ValueError(
                "FCF forecasting requires annual revenue, operating cash flow, "
                "and capital expenditures in the same currency"
            )

        historical_periods = periods[-parameters.historical_window :]
        base_year, base_values = periods[-1]
        base_revenue = base_values[FinancialConcept.REVENUE]
        base_fcf = self._free_cash_flow(base_values)

        growth_path, growth_source = self._growth_path(parameters, historical_periods)
        margin_path, margin_source = self._margin_path(parameters, historical_periods)

        projected_revenue = base_revenue.value
        observations = []
        for index, (growth, margin) in enumerate(
            zip(growth_path, margin_path, strict=True), start=1
        ):
            projected_revenue *= Decimal(1) + growth / Decimal(100)
            projected_fcf = projected_revenue * margin / Decimal(100)
            observations.append(
                FreeCashFlowForecastObservation(
                    forecast_year=index,
                    fiscal_year=base_year + index,
                    period_end=self._future_date(base_revenue.period_end, index),
                    revenue_growth=growth,
                    revenue=projected_revenue,
                    free_cash_flow_margin=margin,
                    free_cash_flow=projected_fcf,
                    unit=base_revenue.unit,
                )
            )

        return FreeCashFlowForecast(
            provider=financials.provider,
            company_id=financials.company_id,
            company_name=financials.company_name,
            ticker=financials.ticker,
            identifiers=financials.identifiers,
            base_fiscal_year=base_year,
            base_period_end=base_revenue.period_end,
            base_revenue=base_revenue.value,
            base_free_cash_flow=base_fcf,
            unit=base_revenue.unit,
            parameters=parameters,
            historical_fiscal_years=tuple(year for year, _ in historical_periods),
            revenue_growth_source=growth_source,
            free_cash_flow_margin_source=margin_source,
            observations=observations,
        )

    def _growth_path(
        self,
        parameters: FreeCashFlowForecastParameters,
        historical_periods: list[
            tuple[int, dict[FinancialConcept, FinancialObservation]]
        ],
    ) -> tuple[tuple[Decimal, ...], ForecastAssumptionSource]:
        if parameters.revenue_growth is not None:
            return (
                self._expand_path(parameters.revenue_growth, parameters.forecast_years),
                ForecastAssumptionSource.EXPLICIT,
            )

        growth_rates = []
        for (previous_year, previous), (current_year, current) in zip(
            historical_periods, historical_periods[1:], strict=False
        ):
            previous_revenue = previous[FinancialConcept.REVENUE]
            current_revenue = current[FinancialConcept.REVENUE]
            if (
                current_year != previous_year + 1
                or previous_revenue.unit != current_revenue.unit
                or previous_revenue.value == 0
            ):
                continue
            growth_rates.append(
                (current_revenue.value - previous_revenue.value)
                / previous_revenue.value
                * Decimal(100)
            )
        if not growth_rates:
            raise ValueError(
                "Revenue growth could not be inferred from consecutive annual "
                "periods; provide --revenue-growth"
            )
        average_growth = sum(growth_rates, Decimal(0)) / len(growth_rates)
        return (
            (average_growth,) * parameters.forecast_years,
            ForecastAssumptionSource.TRAILING_AVERAGE,
        )

    def _margin_path(
        self,
        parameters: FreeCashFlowForecastParameters,
        historical_periods: list[
            tuple[int, dict[FinancialConcept, FinancialObservation]]
        ],
    ) -> tuple[tuple[Decimal, ...], ForecastAssumptionSource]:
        if parameters.free_cash_flow_margin is not None:
            return (
                self._expand_path(
                    parameters.free_cash_flow_margin, parameters.forecast_years
                ),
                ForecastAssumptionSource.EXPLICIT,
            )

        margins = []
        for _, values in historical_periods:
            revenue = values[FinancialConcept.REVENUE]
            if revenue.value == 0:
                continue
            margins.append(self._free_cash_flow(values) / revenue.value * Decimal(100))
        if not margins:
            raise ValueError(
                "Free cash flow margin could not be inferred; provide --fcf-margin"
            )
        average_margin = sum(margins, Decimal(0)) / len(margins)
        return (
            (average_margin,) * parameters.forecast_years,
            ForecastAssumptionSource.TRAILING_AVERAGE,
        )

    @staticmethod
    def _complete_annual_periods(
        financials: NormalizedCompanyFinancials,
    ) -> list[tuple[int, dict[FinancialConcept, FinancialObservation]]]:
        by_year: dict[int, dict[FinancialConcept, FinancialObservation]] = {}
        for observation in financials.observations:
            if (
                observation.granularity != Granularity.ANNUAL
                or observation.fiscal_period != FiscalPeriod.FY
            ):
                continue
            by_year.setdefault(observation.fiscal_year, {}).setdefault(
                observation.concept, observation
            )

        complete = []
        for fiscal_year, values in sorted(by_year.items()):
            if not FreeCashFlowForecastService._REQUIRED_CONCEPTS <= values.keys():
                continue
            units = {
                values[concept].unit
                for concept in FreeCashFlowForecastService._REQUIRED_CONCEPTS
            }
            if len(units) != 1:
                continue
            complete.append((fiscal_year, values))
        return complete

    @staticmethod
    def _free_cash_flow(
        values: dict[FinancialConcept, FinancialObservation],
    ) -> Decimal:
        return (
            values[FinancialConcept.OPERATING_CASH_FLOW].value
            - values[FinancialConcept.CAPITAL_EXPENDITURES].value
        )

    @staticmethod
    def _expand_path(path: tuple[Decimal, ...], years: int) -> tuple[Decimal, ...]:
        return path * years if len(path) == 1 else path

    @staticmethod
    def _future_date(base_date: datetime.date, years: int) -> datetime.date:
        try:
            return base_date.replace(year=base_date.year + years)
        except ValueError:
            return base_date.replace(year=base_date.year + years, day=28)
