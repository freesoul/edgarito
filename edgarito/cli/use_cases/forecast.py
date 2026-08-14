"""CLI use cases for explicit and profile-backed forecasts."""

from __future__ import annotations

import argparse

from edgarito.cli.presentation.console import ForecastConsolePresenter
from edgarito.cli.use_cases.context import call_with_context, dependency
from edgarito.cli.use_cases.financial_retrieval import retrieve_financials
from edgarito.config.valuation import (
    ForecastMethod,
    ValuationProfileLoader,
)
from edgarito.enums.granularity import Granularity
from edgarito.enums.market import Market
from edgarito.services.forecasting import (
    FcffForecastParameters,
    FcffForecastService,
    SimplifiedFcfForecastParameters,
    SimplifiedFcfForecastService,
)
from edgarito.services.valuation import DepreciableAssetLifeResolver


def load_selected_valuation_profile(args, *, context=None):
    ticker = getattr(args, "ticker", None)
    loader = dependency(context, "ValuationProfileLoader", ValuationProfileLoader)
    if ticker:
        profile, _, _ = loader.load_for_ticker(ticker, args.profile)
        return profile
    return loader.load(args.profile)


def resolve_depreciable_asset_life_configuration(
    financials,
    profile_context,
    configuration,
    *,
    context=None,
):
    if configuration.depreciable_asset_life_years is not None:
        return configuration, None
    resolver_type = dependency(
        context,
        "DepreciableAssetLifeResolver",
        DepreciableAssetLifeResolver,
    )
    resolution = resolver_type().resolve(
        financials,
        industry=profile_context.industry,
        business_archetype=profile_context.business_archetype,
        sector=profile_context.sector,
    )
    if resolution.value is None:
        return configuration, resolution
    return (
        configuration.model_copy(
            update={"depreciable_asset_life_years": resolution.value}
        ),
        resolution,
    )


async def run_forecast(args: argparse.Namespace, *, context=None) -> int:
    profile_loader = dependency(
        context,
        "_load_selected_valuation_profile",
        load_selected_valuation_profile,
    )
    simplified_parameters_type = dependency(
        context,
        "SimplifiedFcfForecastParameters",
        SimplifiedFcfForecastParameters,
    )
    simplified_service_type = dependency(
        context,
        "SimplifiedFcfForecastService",
        SimplifiedFcfForecastService,
    )
    fcff_service_type = dependency(
        context,
        "FcffForecastService",
        FcffForecastService,
    )
    presenter_type = dependency(
        context, "ForecastConsolePresenter", ForecastConsolePresenter
    )
    profile = call_with_context(profile_loader, args, context=context)
    forecast_method = (
        ForecastMethod(args.forecast_method)
        if args.forecast_method is not None
        else profile.forecast.default_method
    )
    fcff_driver_arguments = (
        args.operating_margin,
        args.tax_rate,
        args.depreciation_to_revenue,
        args.capex_to_revenue,
        args.operating_working_capital_to_revenue,
    )
    if forecast_method == ForecastMethod.SIMPLIFIED:
        if any(value is not None for value in fcff_driver_arguments):
            raise ValueError(
                "FCFF driver options cannot be used with --forecast-method simplified"
            )
        configured = profile.forecast.simplified
        parameters = simplified_parameters_type(
            forecast_years=(
                args.years if args.years is not None else configured.forecast_years
            ),
            revenue_growth=(
                args.revenue_growth
                if args.revenue_growth is not None
                else configured.revenue_growth
            ),
            free_cash_flow_margin=(
                args.fcf_margin
                if args.fcf_margin is not None
                else configured.free_cash_flow_margin
            ),
            historical_window=(
                args.historical_window
                if args.historical_window is not None
                else configured.historical_window
            ),
        )
        service = simplified_service_type()
    else:
        if args.fcf_margin is not None:
            raise ValueError(
                "--fcf-margin requires --forecast-method simplified; use the "
                "FCFF operating, tax, D&A, capex, and working-capital drivers"
            )
        configured = profile.forecast.fcff
        parameters_builder = dependency(context, "_fcff_parameters", fcff_parameters)
        parameters = call_with_context(
            parameters_builder,
            args,
            configured,
            context=context,
        )
        service = fcff_service_type()
    retrieve = dependency(context, "_retrieve_financials", retrieve_financials)
    financials = await call_with_context(
        retrieve,
        args,
        Granularity.ANNUAL,
        service.required_concepts(),
        context=context,
    )
    forecast = service.forecast(financials, parameters)
    print(presenter_type().render(forecast))
    return 0


def fcff_parameters(
    args: argparse.Namespace,
    configured,
    *,
    context=None,
) -> FcffForecastParameters:
    return FcffForecastParameters(
        forecast_years=(
            args.years if args.years is not None else configured.forecast_years
        ),
        revenue_growth=(
            args.revenue_growth
            if args.revenue_growth is not None
            else configured.revenue_growth
        ),
        operating_margin=(
            args.operating_margin
            if args.operating_margin is not None
            else configured.operating_margin
        ),
        tax_rate=args.tax_rate if args.tax_rate is not None else configured.tax_rate,
        depreciation_to_revenue=(
            args.depreciation_to_revenue
            if args.depreciation_to_revenue is not None
            else configured.depreciation_to_revenue
        ),
        capex_to_revenue=(
            args.capex_to_revenue
            if args.capex_to_revenue is not None
            else configured.capex_to_revenue
        ),
        operating_working_capital_to_revenue=(
            args.operating_working_capital_to_revenue
            if args.operating_working_capital_to_revenue is not None
            else configured.operating_working_capital_to_revenue
        ),
        revenue_anchors=(
            {} if args.revenue_growth is not None else configured.revenue_anchors
        ),
        assumption_source_overrides={
            driver: source
            for driver, source in configured.assumption_source_overrides.items()
            if not (
                args.revenue_growth is not None and driver.value == "revenue_growth"
            )
        },
        historical_window=(
            args.historical_window
            if args.historical_window is not None
            else configured.historical_window
        ),
    )


def market_for_args(args: argparse.Namespace, *, context=None) -> Market:
    del context
    return Market(getattr(args, "market", Market.US.value))


# Keep the historical private names available inside this focused module.
_load_selected_valuation_profile = load_selected_valuation_profile
_resolve_depreciable_asset_life_configuration = resolve_depreciable_asset_life_configuration
_run_forecast = run_forecast
_fcff_parameters = fcff_parameters
_market_for_args = market_for_args


__all__ = [
    "_fcff_parameters",
    "_load_selected_valuation_profile",
    "_market_for_args",
    "_resolve_depreciable_asset_life_configuration",
    "_run_forecast",
    "fcff_parameters",
    "load_selected_valuation_profile",
    "market_for_args",
    "resolve_depreciable_asset_life_configuration",
    "run_forecast",
]
