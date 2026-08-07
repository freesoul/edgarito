import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.forecasting.models import (
    FcffForecast,
    FcffForecastDriver,
    FcffForecastObservation,
    FcffForecastParameters,
    ForecastAssumptionSource,
)
from edgarito.services.metrics.calculator import (
    OPERATING_WORKING_CAPITAL_CONCEPTS,
    operating_working_capital_value,
)

PERCENT = Decimal(100)


@dataclass(frozen=True)
class _HistoricalDrivers:
    fiscal_year: int
    period_end: datetime.date
    unit: str
    revenue: Decimal
    operating_income: Decimal
    pretax_income: Decimal
    income_tax_expense: Decimal
    depreciation_and_amortization: Decimal
    capital_expenditures: Decimal
    operating_working_capital: Decimal


class FcffForecastService:
    """Forecast annual unlevered FCFF from explicit operating drivers."""

    _CORE_REQUIRED_CONCEPTS = frozenset(
        {
            FinancialConcept.REVENUE,
            FinancialConcept.OPERATING_INCOME,
            FinancialConcept.PRETAX_INCOME,
            FinancialConcept.INCOME_TAX_EXPENSE,
            FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
            FinancialConcept.CAPITAL_EXPENDITURES,
        }
    )
    _REQUIRED_CONCEPTS = _CORE_REQUIRED_CONCEPTS | OPERATING_WORKING_CAPITAL_CONCEPTS

    @classmethod
    def required_concepts(cls) -> set[FinancialConcept]:
        return set(cls._REQUIRED_CONCEPTS)

    def forecast(
        self,
        financials: NormalizedCompanyFinancials,
        parameters: Optional[FcffForecastParameters] = None,
    ) -> FcffForecast:
        parameters = parameters or FcffForecastParameters()
        periods = self._complete_annual_periods(financials)
        if not periods:
            self._raise_missing_inputs(financials)

        historical_periods = periods[-parameters.historical_window :]
        paths, sources = self._driver_paths(parameters, historical_periods)
        base = periods[-1]
        previous = periods[-2] if len(periods) > 1 else None
        base_tax_rate = self._effective_tax_rate(base)
        if base_tax_rate is None:
            base_tax_rate = paths[FcffForecastDriver.TAX_RATE][0]
        base_nopat = base.operating_income * (Decimal(1) - base_tax_rate / PERCENT)
        base_fcff = self._historical_fcff(base, previous, base_nopat)

        projected_revenue = base.revenue
        previous_working_capital = base.operating_working_capital
        observations = []
        for index in range(parameters.forecast_years):
            growth = paths[FcffForecastDriver.REVENUE_GROWTH][index]
            operating_margin = paths[FcffForecastDriver.OPERATING_MARGIN][index]
            tax_rate = paths[FcffForecastDriver.TAX_RATE][index]
            depreciation_ratio = paths[FcffForecastDriver.DEPRECIATION_TO_REVENUE][
                index
            ]
            capex_ratio = paths[FcffForecastDriver.CAPEX_TO_REVENUE][index]
            working_capital_ratio = paths[
                FcffForecastDriver.OPERATING_WORKING_CAPITAL_TO_REVENUE
            ][index]

            projected_revenue *= Decimal(1) + growth / PERCENT
            operating_income = projected_revenue * operating_margin / PERCENT
            nopat = operating_income * (Decimal(1) - tax_rate / PERCENT)
            depreciation = projected_revenue * depreciation_ratio / PERCENT
            capital_expenditures = projected_revenue * capex_ratio / PERCENT
            operating_working_capital = (
                projected_revenue * working_capital_ratio / PERCENT
            )
            change_in_working_capital = (
                operating_working_capital - previous_working_capital
            )
            fcff = (
                nopat + depreciation - capital_expenditures - change_in_working_capital
            )
            forecast_year = index + 1
            observations.append(
                FcffForecastObservation(
                    forecast_year=forecast_year,
                    fiscal_year=base.fiscal_year + forecast_year,
                    period_end=self._future_date(base.period_end, forecast_year),
                    revenue_growth=growth,
                    revenue=projected_revenue,
                    operating_margin=operating_margin,
                    operating_income=operating_income,
                    tax_rate=tax_rate,
                    nopat=nopat,
                    depreciation_to_revenue=depreciation_ratio,
                    depreciation_and_amortization=depreciation,
                    capex_to_revenue=capex_ratio,
                    capital_expenditures=capital_expenditures,
                    operating_working_capital_to_revenue=working_capital_ratio,
                    operating_working_capital=operating_working_capital,
                    change_in_operating_working_capital=change_in_working_capital,
                    fcff=fcff,
                    unit=base.unit,
                )
            )
            previous_working_capital = operating_working_capital

        return FcffForecast(
            provider=financials.provider,
            company_id=financials.company_id,
            company_name=financials.company_name,
            ticker=financials.ticker,
            identifiers=financials.identifiers,
            base_fiscal_year=base.fiscal_year,
            base_period_end=base.period_end,
            base_revenue=base.revenue,
            base_operating_income=base.operating_income,
            base_tax_rate=base_tax_rate,
            base_nopat=base_nopat,
            base_depreciation_and_amortization=base.depreciation_and_amortization,
            base_capital_expenditures=base.capital_expenditures,
            base_operating_working_capital=base.operating_working_capital,
            base_fcff=base_fcff,
            unit=base.unit,
            parameters=parameters,
            historical_fiscal_years=tuple(
                period.fiscal_year for period in historical_periods
            ),
            assumption_sources=sources,
            observations=observations,
        )

    def _driver_paths(
        self,
        parameters: FcffForecastParameters,
        historical_periods: list[_HistoricalDrivers],
    ) -> tuple[
        dict[FcffForecastDriver, tuple[Decimal, ...]],
        dict[FcffForecastDriver, ForecastAssumptionSource],
    ]:
        paths = {}
        sources = {}
        for driver in FcffForecastDriver:
            explicit_path = getattr(parameters, driver.value)
            if explicit_path is not None:
                paths[driver] = self._expand_path(
                    explicit_path, parameters.forecast_years
                )
                sources[driver] = ForecastAssumptionSource.EXPLICIT
                continue

            historical_values = self._historical_values(driver, historical_periods)
            if not historical_values:
                option = driver.value.replace("_", "-")
                required_history = (
                    "complete, consecutive annual periods"
                    if driver == FcffForecastDriver.REVENUE_GROWTH
                    else "complete annual periods"
                )
                raise ValueError(
                    f"{driver.label} could not be inferred from {required_history}; "
                    f"provide --{option}"
                )
            average = sum(historical_values, Decimal(0)) / len(historical_values)
            paths[driver] = (average,) * parameters.forecast_years
            sources[driver] = ForecastAssumptionSource.TRAILING_AVERAGE
        return paths, sources

    def _historical_values(
        self,
        driver: FcffForecastDriver,
        periods: list[_HistoricalDrivers],
    ) -> list[Decimal]:
        if driver == FcffForecastDriver.REVENUE_GROWTH:
            return [
                (current.revenue - previous.revenue) / previous.revenue * PERCENT
                for previous, current in zip(periods, periods[1:], strict=False)
                if current.fiscal_year == previous.fiscal_year + 1
                and previous.revenue != 0
            ]

        values = []
        for period in periods:
            if period.revenue == 0:
                continue
            if driver == FcffForecastDriver.OPERATING_MARGIN:
                numerator = period.operating_income
            elif driver == FcffForecastDriver.TAX_RATE:
                tax_rate = self._effective_tax_rate(period)
                if tax_rate is not None:
                    values.append(tax_rate)
                continue
            elif driver == FcffForecastDriver.DEPRECIATION_TO_REVENUE:
                numerator = period.depreciation_and_amortization
            elif driver == FcffForecastDriver.CAPEX_TO_REVENUE:
                numerator = period.capital_expenditures
            else:
                numerator = period.operating_working_capital
            values.append(numerator / period.revenue * PERCENT)
        return values

    @classmethod
    def _complete_annual_periods(
        cls, financials: NormalizedCompanyFinancials
    ) -> list[_HistoricalDrivers]:
        by_year: dict[int, dict[FinancialConcept, FinancialObservation]] = {}
        for observation in financials.observations:
            if (
                observation.granularity == Granularity.ANNUAL
                and observation.fiscal_period == FiscalPeriod.FY
            ):
                by_year.setdefault(observation.fiscal_year, {}).setdefault(
                    observation.concept, observation
                )

        periods = []
        for fiscal_year, values in sorted(by_year.items()):
            if not cls._CORE_REQUIRED_CONCEPTS <= values.keys():
                continue
            operating_working_capital = operating_working_capital_value(values)
            if operating_working_capital is None:
                continue
            units = {
                values[concept].unit for concept in cls._CORE_REQUIRED_CONCEPTS
            } | {operating_working_capital.unit}
            if len(units) != 1:
                continue
            revenue = values[FinancialConcept.REVENUE]
            if revenue.value <= 0:
                continue
            periods.append(
                _HistoricalDrivers(
                    fiscal_year=fiscal_year,
                    period_end=revenue.period_end,
                    unit=revenue.unit,
                    revenue=revenue.value,
                    operating_income=values[FinancialConcept.OPERATING_INCOME].value,
                    pretax_income=values[FinancialConcept.PRETAX_INCOME].value,
                    income_tax_expense=values[
                        FinancialConcept.INCOME_TAX_EXPENSE
                    ].value,
                    depreciation_and_amortization=values[
                        FinancialConcept.DEPRECIATION_AND_AMORTIZATION
                    ].value,
                    capital_expenditures=abs(
                        values[FinancialConcept.CAPITAL_EXPENDITURES].value
                    ),
                    operating_working_capital=operating_working_capital.value,
                )
            )
        return periods

    @staticmethod
    def _effective_tax_rate(period: _HistoricalDrivers) -> Optional[Decimal]:
        if period.pretax_income <= 0 or period.income_tax_expense < 0:
            return None
        rate = period.income_tax_expense / period.pretax_income * PERCENT
        return rate if Decimal(0) <= rate <= PERCENT else None

    @staticmethod
    def _historical_fcff(
        current: _HistoricalDrivers,
        previous: Optional[_HistoricalDrivers],
        nopat: Decimal,
    ) -> Optional[Decimal]:
        if previous is None or current.fiscal_year != previous.fiscal_year + 1:
            return None
        change_in_working_capital = (
            current.operating_working_capital - previous.operating_working_capital
        )
        return (
            nopat
            + current.depreciation_and_amortization
            - current.capital_expenditures
            - change_in_working_capital
        )

    @classmethod
    def _raise_missing_inputs(cls, financials: NormalizedCompanyFinancials) -> None:
        annual = [
            observation
            for observation in financials.observations
            if observation.granularity == Granularity.ANNUAL
            and observation.fiscal_period == FiscalPeriod.FY
        ]
        latest_year = max(
            (observation.fiscal_year for observation in annual), default=None
        )
        present = {
            observation.concept
            for observation in annual
            if observation.fiscal_year == latest_year
        }
        missing = sorted(
            cls._CORE_REQUIRED_CONCEPTS - present, key=lambda item: item.value
        )
        details = [concept.value for concept in missing]
        latest_values = {
            observation.concept: observation
            for observation in annual
            if observation.fiscal_year == latest_year
        }
        if operating_working_capital_value(latest_values) is None:
            has_detailed_assets = {
                FinancialConcept.ACCOUNTS_RECEIVABLE,
                FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS,
            } <= present
            has_aggregate_assets = {
                FinancialConcept.CURRENT_ASSETS,
                FinancialConcept.CASH_AND_EQUIVALENTS,
            } <= present
            has_detailed_liabilities = {
                FinancialConcept.ACCOUNTS_PAYABLE,
                FinancialConcept.ACCRUED_LIABILITIES,
                FinancialConcept.DEFERRED_REVENUE_CURRENT,
            } <= present
            has_current_liabilities = FinancialConcept.CURRENT_LIABILITIES in present
            if not has_detailed_assets and not has_aggregate_assets:
                details.append(
                    "working-capital assets (receivables and prepaid/other current "
                    "assets, or total current assets and cash)"
                )
            if not has_current_liabilities and not (
                has_detailed_assets and has_detailed_liabilities
            ):
                details.append(
                    "working-capital liabilities (accrued/payables/deferred revenue "
                    "with detailed assets, or total current liabilities)"
                )
        detail = ", ".join(details)
        suffix = f" Missing for FY{latest_year}: {detail}." if details else ""
        raise ValueError(
            "Driver-based FCFF forecasting requires complete annual revenue, "
            "operating income, pretax income, tax, D&A, capex, and operating "
            f"working-capital components in one currency.{suffix}"
        )

    @staticmethod
    def _expand_path(path: tuple[Decimal, ...], years: int) -> tuple[Decimal, ...]:
        if len(path) == 1:
            return path * years
        return (*path, *((path[-1],) * (years - len(path))))

    @staticmethod
    def _future_date(base_date: datetime.date, years: int) -> datetime.date:
        try:
            return base_date.replace(year=base_date.year + years)
        except ValueError:
            return base_date.replace(year=base_date.year + years, day=28)


# Preserve the old generic service import while changing its semantics to FCFF.
FreeCashFlowForecastService = FcffForecastService


__all__ = ["FcffForecastService", "FreeCashFlowForecastService"]
