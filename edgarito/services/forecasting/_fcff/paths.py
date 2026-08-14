"""Build FCFF driver paths from explicit or historical assumptions."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from edgarito.services.forecasting._fcff.contracts import (
    PERCENT,
    _ForecastContext,
    _HistoricalDrivers,
)
from edgarito.services.forecasting.models import (
    FcffForecastDriver,
    FcffForecastParameters,
    ForecastAssumptionSource,
    ForecastSeedType,
)


def driver_paths(
    service: Any,
    parameters: FcffForecastParameters,
    historical_periods: list[_HistoricalDrivers],
    *,
    fallback_revenue_growth: Decimal | None = None,
) -> tuple[
    dict[FcffForecastDriver, tuple[Decimal, ...]],
    dict[FcffForecastDriver, ForecastAssumptionSource],
]:
    paths = {}
    sources = {}
    for driver in FcffForecastDriver:
        explicit_path = getattr(parameters, driver.value)
        if explicit_path is not None:
            paths[driver] = expand_path(explicit_path, parameters.forecast_years)
            sources[driver] = ForecastAssumptionSource.EXPLICIT
            continue

        historical_values = service._historical_values(driver, historical_periods)
        if not historical_values:
            if (
                driver == FcffForecastDriver.REVENUE_GROWTH
                and fallback_revenue_growth is not None
            ):
                paths[driver] = (
                    fallback_revenue_growth,
                ) * parameters.forecast_years
                sources[driver] = ForecastAssumptionSource.CURRENT_RUN_RATE
                continue
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
    sources.update(parameters.assumption_source_overrides)
    return paths, sources


def current_run_rate_growth(context: _ForecastContext) -> Decimal | None:
    latest_annual_revenue = context.latest_annual.revenue
    if latest_annual_revenue <= 0:
        return None
    if context.actual_ytd is not None and context.actual_quarters:
        annualized_revenue = (
            context.actual_ytd.revenue
            * Decimal(4)
            / Decimal(context.actual_quarters)
        )
        return (annualized_revenue / latest_annual_revenue - Decimal(1)) * PERCENT
    if context.seed_type in {ForecastSeedType.TTM, ForecastSeedType.YTD_RUN_RATE}:
        return (context.base.revenue / latest_annual_revenue - Decimal(1)) * PERCENT
    return None


def historical_values(
    service: Any,
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
            tax_rate = service._effective_tax_rate(period)
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


def effective_tax_rate(period: _HistoricalDrivers) -> Decimal | None:
    if period.pretax_income <= 0 or period.income_tax_expense < 0:
        return None
    rate = period.income_tax_expense / period.pretax_income * PERCENT
    return rate if Decimal(0) <= rate <= PERCENT else None


def historical_fcff(
    current: _HistoricalDrivers,
    previous: _HistoricalDrivers | None,
    nopat: Decimal,
) -> Decimal | None:
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


def expand_path(path: tuple[Decimal, ...], years: int) -> tuple[Decimal, ...]:
    if len(path) == 1:
        return path * years
    return (*path, *((path[-1],) * (years - len(path))))


__all__ = [name for name in globals() if not name.startswith("__")]
