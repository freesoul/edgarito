"""Calculate annual FCFF observations from a prepared forecast context."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from edgarito.schemas.normalization.financials import NormalizedCompanyFinancials
from edgarito.services.financial_observation_availability import (
    ObservationAvailabilityMode,
)
from edgarito.services.forecasting._fcff.contracts import (
    PERCENT,
    _ForecastContext,
)
from edgarito.services.forecasting.models import (
    FcffForecast,
    FcffForecastDcfStub,
    FcffForecastDriver,
    FcffForecastObservation,
    FcffForecastParameters,
    FcffForecastYtdAnchor,
    ForecastAssumptionSource,
    ForecastSeedType,
)


def build_forecast(
    service: Any,
    financials: NormalizedCompanyFinancials,
    parameters: FcffForecastParameters,
    context: _ForecastContext,
    annual_periods,
    availability_mode: ObservationAvailabilityMode,
) -> FcffForecast:
    historical_periods = annual_periods[-parameters.historical_window :]
    normalized_historical_periods = annual_periods[
        -(parameters.historical_window + 1) :
    ]
    normalized_historical_growth_path = tuple(
        service._historical_values(
            FcffForecastDriver.REVENUE_GROWTH,
            list(normalized_historical_periods),
        )
    )
    normalized_historical_growth = (
        sum(normalized_historical_growth_path, Decimal(0))
        / Decimal(len(normalized_historical_growth_path))
        if normalized_historical_growth_path
        else None
    )
    paths, sources = service._driver_paths(
        parameters,
        list(context.path_periods),
        fallback_revenue_growth=service._current_run_rate_growth(context),
    )
    base = context.base
    previous = (
        annual_periods[-2]
        if len(annual_periods) > 1 and context.seed_type == ForecastSeedType.FISCAL_YEAR
        else None
    )
    base_tax_rate = service._effective_tax_rate(base)
    if base_tax_rate is None:
        base_tax_rate = paths[FcffForecastDriver.TAX_RATE][0]
    base_nopat = base.operating_income * (Decimal(1) - base_tax_rate / PERCENT)
    base_fcff = service._historical_fcff(base, previous, base_nopat)

    projected_revenue = base.revenue
    previous_working_capital = base.operating_working_capital
    observations = []
    capex_constraints_applied: list[int] = []
    ytd_capex_ratio = None
    first_fiscal_year = (
        context.current_fiscal_year
        if context.current_fiscal_year is not None
        else base.fiscal_year + 1
    )
    for index in range(parameters.forecast_years):
        fiscal_year = first_fiscal_year + index
        growth = paths[FcffForecastDriver.REVENUE_GROWTH][index]
        operating_margin = paths[FcffForecastDriver.OPERATING_MARGIN][index]
        tax_rate = paths[FcffForecastDriver.TAX_RATE][index]
        depreciation_ratio = paths[FcffForecastDriver.DEPRECIATION_TO_REVENUE][index]
        capex_ratio = paths[FcffForecastDriver.CAPEX_TO_REVENUE][index]
        working_capital_ratio = paths[
            FcffForecastDriver.OPERATING_WORKING_CAPITAL_TO_REVENUE
        ][index]

        if index == 0 and context.actual_ytd is not None:
            actual = context.actual_ytd
            revenue_anchor = parameters.revenue_anchors.get(fiscal_year)
            if revenue_anchor is not None and revenue_anchor < actual.revenue:
                raise ValueError(
                    f"FY{fiscal_year} revenue anchor is below reported YTD revenue"
                )
            projected_revenue = revenue_anchor or max(
                actual.revenue,
                context.latest_annual.revenue * (Decimal(1) + growth / PERCENT),
            )
            remaining_revenue = projected_revenue - actual.revenue
            operating_income = (
                actual.operating_income
                + remaining_revenue * operating_margin / PERCENT
            )
            actual_tax_rate = service._effective_tax_rate(actual)
            actual_nopat = actual.operating_income * (
                Decimal(1)
                - (actual_tax_rate if actual_tax_rate is not None else tax_rate)
                / PERCENT
            )
            projected_nopat = (
                remaining_revenue
                * operating_margin
                / PERCENT
                * (Decimal(1) - tax_rate / PERCENT)
            )
            nopat = actual_nopat + projected_nopat
            depreciation = (
                actual.depreciation_and_amortization
                + remaining_revenue * depreciation_ratio / PERCENT
            )
            provisional_capex = (
                actual.capital_expenditures
                + remaining_revenue * capex_ratio / PERCENT
            )
            capital_expenditures = provisional_capex
            ytd_capex_ratio = capex_ratio
            operating_margin = operating_income / projected_revenue * PERCENT
            tax_rate = (
                (Decimal(1) - nopat / operating_income) * PERCENT
                if operating_income > 0
                else tax_rate
            )
            depreciation_ratio = depreciation / projected_revenue * PERCENT
            capex_ratio = capital_expenditures / projected_revenue * PERCENT
            growth = (
                projected_revenue / context.latest_annual.revenue - Decimal(1)
            ) * PERCENT
        else:
            previous_revenue = projected_revenue
            projected_revenue = parameters.revenue_anchors.get(
                fiscal_year
            ) or projected_revenue * (Decimal(1) + growth / PERCENT)
            if fiscal_year in parameters.revenue_anchors:
                growth = (
                    projected_revenue / previous_revenue - Decimal(1)
                ) * PERCENT
            operating_income = projected_revenue * operating_margin / PERCENT
            nopat = operating_income * (Decimal(1) - tax_rate / PERCENT)
            depreciation = projected_revenue * depreciation_ratio / PERCENT
            provisional_capex = projected_revenue * capex_ratio / PERCENT
            capital_expenditures = provisional_capex

        capex_constraint = service._capex_constraint_for(parameters, fiscal_year)
        if capex_constraint is not None:
            if (
                index == 0
                and context.actual_ytd is not None
                and (
                    (
                        capex_constraint.point is not None
                        and capex_constraint.point
                        < context.actual_ytd.capital_expenditures
                    )
                    or (
                        capex_constraint.maximum is not None
                        and capex_constraint.maximum
                        < context.actual_ytd.capital_expenditures
                    )
                )
            ):
                raise ValueError(
                    f"FY{fiscal_year} CAPEX constraint is below reported YTD CAPEX"
                )
            constrained_capex = service._apply_capex_constraint(
                provisional_capex, capex_constraint
            )
            if (
                constrained_capex != provisional_capex
                or capex_constraint.point is not None
            ):
                capex_constraints_applied.append(fiscal_year)
            capital_expenditures = constrained_capex
            capex_ratio = capital_expenditures / projected_revenue * PERCENT
            if index == 0 and context.actual_ytd is not None:
                if remaining_revenue != 0:
                    ytd_capex_ratio = (
                        (capital_expenditures - context.actual_ytd.capital_expenditures)
                        / remaining_revenue
                        * PERCENT
                    )
                else:
                    ytd_capex_ratio = capex_ratio
        operating_working_capital = (
            projected_revenue * working_capital_ratio / PERCENT
        )
        change_in_working_capital = (
            operating_working_capital - previous_working_capital
        )
        fcff = (
            nopat
            + depreciation
            - capital_expenditures
            - change_in_working_capital
        )
        forecast_year = index + 1
        first_period_end = context.fiscal_year_end or service._future_date(
            context.latest_annual.period_end, 1
        )
        observations.append(
            FcffForecastObservation(
                forecast_year=forecast_year,
                fiscal_year=fiscal_year,
                period_end=service._future_date(first_period_end, index),
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

    if context.actual_ytd is not None and ytd_capex_ratio is not None:
        ytd_anchor_capex_ratio = ytd_capex_ratio
    else:
        ytd_anchor_capex_ratio = paths[FcffForecastDriver.CAPEX_TO_REVENUE][0]

    ytd_anchor = None
    dcf_stub = None
    if (
        context.seed_type == ForecastSeedType.YTD_PLUS_FORECAST
        and context.actual_ytd is not None
    ):
        actual_ytd = context.actual_ytd
        ytd_anchor = FcffForecastYtdAnchor(
            fiscal_year=first_fiscal_year,
            ytd_period_end=actual_ytd.period_end,
            fiscal_year_end=context.fiscal_year_end or actual_ytd.period_end,
            actual_quarters=context.actual_quarters,
            actual_revenue=actual_ytd.revenue,
            actual_operating_income=actual_ytd.operating_income,
            actual_pretax_income=actual_ytd.pretax_income,
            actual_income_tax_expense=actual_ytd.income_tax_expense,
            actual_tax_rate=service._effective_tax_rate(actual_ytd),
            actual_depreciation_and_amortization=(
                actual_ytd.depreciation_and_amortization
            ),
            actual_capital_expenditures=actual_ytd.capital_expenditures,
            actual_operating_working_capital=actual_ytd.operating_working_capital,
            latest_annual_revenue=context.latest_annual.revenue,
            revenue_anchor=parameters.revenue_anchors.get(first_fiscal_year),
            revenue_growth=paths[FcffForecastDriver.REVENUE_GROWTH][0],
            operating_margin=paths[FcffForecastDriver.OPERATING_MARGIN][0],
            tax_rate=paths[FcffForecastDriver.TAX_RATE][0],
            depreciation_to_revenue=paths[FcffForecastDriver.DEPRECIATION_TO_REVENUE][0],
            capex_to_revenue=ytd_anchor_capex_ratio,
            operating_working_capital_to_revenue=paths[
                FcffForecastDriver.OPERATING_WORKING_CAPITAL_TO_REVENUE
            ][0],
        )
        first_observation = observations[0]
        actual_tax_rate = service._effective_tax_rate(actual_ytd)
        if actual_tax_rate is None:
            actual_tax_rate = paths[FcffForecastDriver.TAX_RATE][0]
        actual_ytd_nopat = actual_ytd.operating_income * (
            Decimal(1) - actual_tax_rate / PERCENT
        )
        dcf_stub = FcffForecastDcfStub(
            forecast_year=first_observation.forecast_year,
            fiscal_year=first_observation.fiscal_year,
            period_start=actual_ytd.period_end,
            period_end=first_observation.period_end,
            unit=first_observation.unit,
            annual_nopat=first_observation.nopat,
            actual_ytd_nopat=actual_ytd_nopat,
            annual_depreciation_and_amortization=(
                first_observation.depreciation_and_amortization
            ),
            actual_ytd_depreciation_and_amortization=(
                actual_ytd.depreciation_and_amortization
            ),
            annual_capital_expenditures=first_observation.capital_expenditures,
            actual_ytd_capital_expenditures=actual_ytd.capital_expenditures,
            fiscal_year_end_operating_working_capital=(
                first_observation.operating_working_capital
            ),
            actual_ytd_operating_working_capital=actual_ytd.operating_working_capital,
            fcff=(
                first_observation.nopat
                - actual_ytd_nopat
                + first_observation.depreciation_and_amortization
                - actual_ytd.depreciation_and_amortization
                - first_observation.capital_expenditures
                + actual_ytd.capital_expenditures
                - first_observation.operating_working_capital
                + actual_ytd.operating_working_capital
            ),
        )

    if capex_constraints_applied:
        sources[FcffForecastDriver.CAPEX_TO_REVENUE] = (
            ForecastAssumptionSource.MANAGEMENT_GUIDANCE
        )
    capex_source_path = tuple(
        ForecastAssumptionSource.MANAGEMENT_GUIDANCE
        if observation.fiscal_year in capex_constraints_applied
        else sources[FcffForecastDriver.CAPEX_TO_REVENUE]
        for observation in observations
    )

    return FcffForecast(
        provider=financials.provider,
        company_id=financials.company_id,
        company_name=financials.company_name,
        ticker=financials.ticker,
        identifiers=financials.identifiers,
        seed_type=context.seed_type,
        seed_methodology=context.seed_methodology,
        seed_period_end=context.seed_period_end,
        current_fiscal_year=context.current_fiscal_year,
        actual_quarters=context.actual_quarters,
        financial_snapshot_retrieved_at=financials.retrieved_at,
        availability_mode=availability_mode.value,
        base_fiscal_year=base.fiscal_year,
        base_period_end=context.seed_period_end,
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
        assumption_source_paths={
            FcffForecastDriver.CAPEX_TO_REVENUE: capex_source_path
        },
        observations=observations,
        warnings=service._incomplete_quarter_warnings(
            financials, context.seed_period_end
        ),
        ytd_anchor=ytd_anchor,
        dcf_stub=dcf_stub,
        capex_constraints_applied=tuple(capex_constraints_applied),
        current_growth_rate=(observations[0].revenue_growth if observations else None),
        normalized_historical_growth=normalized_historical_growth,
        normalized_historical_growth_path=normalized_historical_growth_path,
    )


__all__ = ["build_forecast"]
