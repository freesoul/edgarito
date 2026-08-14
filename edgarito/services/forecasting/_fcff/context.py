"""Historical period discovery and forecast seed context."""

from __future__ import annotations

import datetime
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.forecasting._fcff.contracts import (
    _ForecastContext,
    _HistoricalDrivers,
)
from edgarito.services.metrics.calculator import operating_working_capital_value


def forecast_context(
    service: Any,
    financials: NormalizedCompanyFinancials,
    annual_periods: list[_HistoricalDrivers],
    parameters: Any,
    as_of: datetime.date | None,
) -> _ForecastContext:
    latest_annual = annual_periods[-1]
    quarterly = service._complete_quarterly_periods(financials)
    newer = [
        item for item in quarterly if item.period_end > latest_annual.period_end
    ]
    if not newer:
        selected = tuple(annual_periods[-parameters.historical_window :])
        return _ForecastContext(
            base=latest_annual,
            latest_annual=latest_annual,
            path_periods=selected,
            seed_type=service._forecast_seed_type("fiscal_year"),
            seed_methodology=(
                f"Latest complete FY{latest_annual.fiscal_year}; no newer complete "
                "quarterly operating context was available"
            ),
            seed_period_end=latest_annual.period_end,
        )

    latest = newer[-1]
    current = [item for item in newer if item.fiscal_year == latest.fiscal_year]
    current.sort(key=lambda item: item.period_end)
    latest_four = quarterly[-4:]
    has_ltm = len(latest_four) == 4 and all(
        service._quarter_index(right) == service._quarter_index(left) + 1
        for left, right in zip(latest_four, latest_four[1:], strict=False)
    )
    ltm = service._aggregate_quarters(latest_four) if has_ltm else None
    path_periods = list(annual_periods[-parameters.historical_window :])
    if ltm is not None:
        path_periods.append(
            _HistoricalDrivers(
                **{
                    **ltm.__dict__,
                    "fiscal_year": latest_annual.fiscal_year + 1,
                }
            )
        )

    if latest.fiscal_period != FiscalPeriod.Q4:
        actual_ytd = service._aggregate_quarters(current)
        fiscal_end = service._fiscal_year_end(
            latest.fiscal_year, latest_annual.period_end
        )
        if as_of is not None and fiscal_end <= as_of:
            next_fiscal_end = service._future_date(fiscal_end, 1)
            if ltm is not None:
                return _ForecastContext(
                    base=_HistoricalDrivers(
                        **{**ltm.__dict__, "fiscal_year": latest.fiscal_year}
                    ),
                    latest_annual=latest_annual,
                    path_periods=tuple(path_periods),
                    seed_type=service._forecast_seed_type("ttm"),
                    seed_methodology=(
                        f"The FY{latest.fiscal_year} year ended on "
                        f"{fiscal_end.isoformat()} but final-period reporting was "
                        "not yet available; four consecutive reported quarters "
                        "form the current run-rate and the first forecast is the "
                        "next unelapsed fiscal year"
                    ),
                    seed_period_end=latest.period_end,
                    current_fiscal_year=latest.fiscal_year + 1,
                    actual_quarters=4,
                    fiscal_year_end=next_fiscal_end,
                )

            run_rate = service._annualize_ytd(actual_ytd, len(current), fiscal_end)
            path_periods.append(run_rate)
            return _ForecastContext(
                base=run_rate,
                latest_annual=latest_annual,
                path_periods=tuple(path_periods),
                seed_type=service._forecast_seed_type("ytd_run_rate"),
                seed_methodology=(
                    f"The FY{latest.fiscal_year} year ended on "
                    f"{fiscal_end.isoformat()} before final-period reporting was "
                    f"available; {len(current)} reported quarter(s) were annualized "
                    "as a low-confidence run-rate and the first forecast is the "
                    "next unelapsed fiscal year"
                ),
                seed_period_end=latest.period_end,
                current_fiscal_year=latest.fiscal_year + 1,
                actual_quarters=len(current),
                fiscal_year_end=next_fiscal_end,
            )
        base = (
            _HistoricalDrivers(
                **{**ltm.__dict__, "fiscal_year": latest.fiscal_year}
            )
            if ltm is not None
            else latest_annual
        )
        return _ForecastContext(
            base=base,
            latest_annual=latest_annual,
            path_periods=tuple(path_periods),
            seed_type=service._forecast_seed_type("ytd_plus_forecast"),
            seed_methodology=(
                f"FY{latest.fiscal_year} estimate uses {len(current)} actual fiscal "
                f"quarter(s) through {latest.period_end.isoformat()} plus a driver-"
                "based forecast of the remaining period"
                + ("; latest-four-quarter metrics seed normalization" if has_ltm else "")
            ),
            seed_period_end=latest.period_end,
            current_fiscal_year=latest.fiscal_year,
            actual_ytd=actual_ytd,
            actual_quarters=len(current),
            fiscal_year_end=fiscal_end,
        )

    assert ltm is not None
    return _ForecastContext(
        base=ltm,
        latest_annual=latest_annual,
        path_periods=tuple(path_periods),
        seed_type=service._forecast_seed_type("ttm"),
        seed_methodology=(
            "Four consecutive fiscal quarters form a current run-rate; the TTM "
            "is not inserted into completed annual history"
        ),
        seed_period_end=latest.period_end,
        current_fiscal_year=latest.fiscal_year + 1,
        actual_quarters=4,
        fiscal_year_end=service._fiscal_year_end(
            latest.fiscal_year + 1, latest_annual.period_end
        ),
    )


def annualize_ytd(
    actual: _HistoricalDrivers,
    quarter_count: int,
    fiscal_end: datetime.date,
) -> _HistoricalDrivers:
    scale = Decimal(4) / Decimal(quarter_count)
    return _HistoricalDrivers(
        fiscal_year=actual.fiscal_year,
        period_end=fiscal_end,
        unit=actual.unit,
        revenue=actual.revenue * scale,
        operating_income=actual.operating_income * scale,
        pretax_income=actual.pretax_income * scale,
        income_tax_expense=actual.income_tax_expense * scale,
        depreciation_and_amortization=actual.depreciation_and_amortization * scale,
        capital_expenditures=actual.capital_expenditures * scale,
        # Working capital is a point-in-time balance, not an additive flow.
        operating_working_capital=actual.operating_working_capital,
    )


def complete_quarterly_periods(
    financials: NormalizedCompanyFinancials,
    core_required_concepts: Iterable[FinancialConcept],
) -> list[_HistoricalDrivers]:
    core_required_concepts = frozenset(core_required_concepts)
    by_period: dict[tuple[int, FiscalPeriod], dict] = {}
    for item in financials.observations:
        if item.granularity == Granularity.QUARTERLY and item.fiscal_period in {
            FiscalPeriod.Q1,
            FiscalPeriod.Q2,
            FiscalPeriod.Q3,
            FiscalPeriod.Q4,
        }:
            by_period.setdefault(item.period_key, {}).setdefault(item.concept, item)
    periods = []
    for (fiscal_year, _fiscal_period), values in by_period.items():
        if not core_required_concepts <= values.keys():
            continue
        owc = operating_working_capital_value(values)
        if owc is None:
            continue
        units = {values[concept].unit for concept in core_required_concepts} | {owc.unit}
        revenue = values[FinancialConcept.REVENUE]
        if len(units) != 1 or revenue.value <= 0:
            continue
        periods.append(
            _HistoricalDrivers(
                fiscal_year=fiscal_year,
                period_end=revenue.period_end,
                unit=revenue.unit,
                revenue=revenue.value,
                operating_income=values[FinancialConcept.OPERATING_INCOME].value,
                pretax_income=values[FinancialConcept.PRETAX_INCOME].value,
                income_tax_expense=values[FinancialConcept.INCOME_TAX_EXPENSE].value,
                depreciation_and_amortization=values[
                    FinancialConcept.DEPRECIATION_AND_AMORTIZATION
                ].value,
                capital_expenditures=abs(
                    values[FinancialConcept.CAPITAL_EXPENDITURES].value
                ),
                operating_working_capital=owc.value,
                fiscal_period=_fiscal_period,
            )
        )
    return sorted(periods, key=lambda item: item.period_end)


def aggregate_quarters(periods: list[_HistoricalDrivers]) -> _HistoricalDrivers:
    latest = periods[-1]
    return _HistoricalDrivers(
        fiscal_year=latest.fiscal_year,
        period_end=latest.period_end,
        unit=latest.unit,
        revenue=sum((item.revenue for item in periods), Decimal(0)),
        operating_income=sum((item.operating_income for item in periods), Decimal(0)),
        pretax_income=sum((item.pretax_income for item in periods), Decimal(0)),
        income_tax_expense=sum(
            (item.income_tax_expense for item in periods), Decimal(0)
        ),
        depreciation_and_amortization=sum(
            (item.depreciation_and_amortization for item in periods), Decimal(0)
        ),
        capital_expenditures=sum(
            (item.capital_expenditures for item in periods), Decimal(0)
        ),
        operating_working_capital=latest.operating_working_capital,
        fiscal_period=latest.fiscal_period,
    )


def quarter_index(period: _HistoricalDrivers) -> int:
    quarter = {
        FiscalPeriod.Q1: 0,
        FiscalPeriod.Q2: 1,
        FiscalPeriod.Q3: 2,
        FiscalPeriod.Q4: 3,
    }[period.fiscal_period]
    return period.fiscal_year * 4 + quarter


def fiscal_year_end(fiscal_year: int, annual_end: datetime.date) -> datetime.date:
    try:
        return annual_end.replace(year=fiscal_year)
    except ValueError:
        return annual_end.replace(year=fiscal_year, day=28)


def future_date(base_date: datetime.date, years: int) -> datetime.date:
    try:
        return base_date.replace(year=base_date.year + years)
    except ValueError:
        return base_date.replace(year=base_date.year + years, day=28)


def complete_annual_periods(
    financials: NormalizedCompanyFinancials,
    core_required_concepts: Iterable[FinancialConcept],
) -> list[_HistoricalDrivers]:
    core_required_concepts = frozenset(core_required_concepts)
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
        if not core_required_concepts <= values.keys():
            continue
        operating_working_capital = operating_working_capital_value(values)
        if operating_working_capital is None:
            continue
        units = {values[concept].unit for concept in core_required_concepts} | {
            operating_working_capital.unit
        }
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
                income_tax_expense=values[FinancialConcept.INCOME_TAX_EXPENSE].value,
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


def raise_missing_inputs(
    financials: NormalizedCompanyFinancials,
    core_required_concepts: Iterable[FinancialConcept],
) -> None:
    core_required_concepts = frozenset(core_required_concepts)
    annual = [
        observation
        for observation in financials.observations
        if observation.granularity == Granularity.ANNUAL
        and observation.fiscal_period == FiscalPeriod.FY
    ]
    latest_year = max((observation.fiscal_year for observation in annual), default=None)
    present = {
        observation.concept
        for observation in annual
        if observation.fiscal_year == latest_year
    }
    missing = sorted(core_required_concepts - present, key=lambda item: item.value)
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


__all__ = [name for name in globals() if not name.startswith("__")]
