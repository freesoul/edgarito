"""Historical and evidence-backed FCFF forecast construction.

This stage stops before capital-bridge resolution and valuation.  It builds the
initial forecast, applies optional management guidance, and returns the
objects needed by the later operating-evidence and DCF stages.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from edgarito.cli.use_cases.context import (
    ValuationDependencyContext,
    call_with_context,
)
from edgarito.cli.use_cases.financial_retrieval import retrieve_financials
from edgarito.cli.use_cases.forecast import fcff_parameters
from edgarito.cli.use_cases.forward_assumptions import (
    financial_snapshot_warnings as _default_financial_snapshot_warnings,
)
from edgarito.cli.use_cases.forward_assumptions import (
    management_guidance_overlay as _default_management_guidance_overlay,
)
from edgarito.config.valuation import ForecastValuationProfile
from edgarito.enums.market import Market
from edgarito.schemas.forecasting import FcffForecastParameters
from edgarito.schemas.guidance.management import GuidanceOverlayResult
from edgarito.schemas.normalization.financials import FinancialConcept
from edgarito.services.financials.availability import (
    ObservationAvailabilityMode,
)
from edgarito.services.forecasting._fcff.service import FcffForecastService
from edgarito.services.valuation import (
    FcffDcfCapitalBridgeResolver,
    ValuationProfileBuilder,
)
from edgarito.services.valuation.models import CashFlowTiming, TerminalValueMethod


@dataclass(frozen=True)
class ForecastConstructionResult:
    """Inputs shared by operating evidence, assumptions, and DCF stages."""

    forecast_parameters: FcffForecastParameters
    terminal_configuration: object
    terminal_method: TerminalValueMethod
    cash_flow_timing: CashFlowTiming
    forecast_service: FcffForecastService
    bridge_resolver: FcffDcfCapitalBridgeResolver
    financials: object
    valuation_date: datetime.date
    forecast: object
    seed_forecast: object
    guidance_overlay: GuidanceOverlayResult | None
    warnings: tuple[str, ...]


def _resolve(dependencies, name: str, default):
    return ValuationDependencyContext(dependencies).resolve(name, default)


async def construct_fcff_forecast(
    *,
    args,
    profile: ForecastValuationProfile,
    market: Market,
    dependencies=None,
) -> ForecastConstructionResult:
    """Build the historical FCFF forecast and optional guidance overlay."""

    forecast_parameters_builder = _resolve(
        dependencies, "_fcff_parameters", fcff_parameters
    )
    retrieve = _resolve(dependencies, "_retrieve_financials", retrieve_financials)
    financial_snapshot_warnings = _resolve(
        dependencies,
        "_financial_snapshot_warnings",
        _default_financial_snapshot_warnings,
    )
    valuation_step = _resolve(dependencies, "_valuation_step", _null_step)
    management_guidance_overlay = _resolve(
        dependencies,
        "_management_guidance_overlay",
        _default_management_guidance_overlay,
    )
    forecast_service_type = _resolve(
        dependencies, "FcffForecastService", FcffForecastService
    )
    bridge_resolver_type = _resolve(
        dependencies,
        "FcffDcfCapitalBridgeResolver",
        FcffDcfCapitalBridgeResolver,
    )
    profile_builder_type = _resolve(
        dependencies, "ValuationProfileBuilder", ValuationProfileBuilder
    )
    openai_api_key = _resolve(dependencies, "OPENAI_API_KEY", "")

    sec_backed_evidence_allowed = market == Market.US
    warnings: list[str] = []
    forecast_parameters = call_with_context(
        forecast_parameters_builder,
        args,
        profile.forecast.fcff,
        context=dependencies,
    )
    terminal_configuration = profile.valuation.terminal_value
    terminal_method = (
        TerminalValueMethod(args.terminal_method)
        if args.terminal_method is not None
        else terminal_configuration.method
    )
    cash_flow_timing = (
        CashFlowTiming(args.cash_flow_timing)
        if args.cash_flow_timing is not None
        else profile.valuation.cash_flow_timing
    )
    forecast_service = forecast_service_type()
    bridge_resolver = bridge_resolver_type()
    required_concepts = (
        forecast_service.required_concepts()
        | bridge_resolver.required_concepts()
        | profile_builder_type.required_concepts()
        | {
            FinancialConcept.INTEREST_EXPENSE,
            FinancialConcept.STOCKHOLDERS_EQUITY,
        }
    )
    with call_with_context(
        valuation_step,
        "retrieving financial data",
        context=dependencies,
    ):
        financials = await call_with_context(
            retrieve,
            args,
            None,
            required_concepts,
            context=dependencies,
        )
    valuation_date = datetime.date.today()
    warnings.extend(
        call_with_context(
            financial_snapshot_warnings,
            financials,
            args,
            context=dependencies,
        )
    )
    with call_with_context(
        valuation_step,
        "building historical forecast",
        context=dependencies,
    ):
        forecast = forecast_service.forecast(
            financials,
            forecast_parameters,
            as_of=valuation_date,
            availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
        )
    guidance_overlay: GuidanceOverlayResult | None = None
    if openai_api_key and sec_backed_evidence_allowed:
        original_forecast_parameters = forecast_parameters
        try:
            with call_with_context(
                valuation_step,
                "retrieving management guidance",
                context=dependencies,
            ):
                candidate_parameters, candidate_overlay = await call_with_context(
                    management_guidance_overlay,
                    args,
                    financials,
                    original_forecast_parameters,
                    forecast,
                    valuation_date,
                    market=market,
                    context=dependencies,
                )
            warnings.extend(candidate_overlay.warnings)
            candidate_forecast = forecast
            if candidate_overlay.applications:
                candidate_forecast = forecast_service.forecast(
                    financials,
                    candidate_parameters,
                    as_of=valuation_date,
                    availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
                )
            forecast_parameters = candidate_parameters
            guidance_overlay = candidate_overlay
            forecast = candidate_forecast
        except Exception as exc:
            forecast_parameters = original_forecast_parameters
            warnings.append(
                "Management-guidance extraction unavailable; historical forecast "
                f"retained ({exc})"
            )
            guidance_overlay = None
    elif args.verbose or args.audit:
        warnings.append(
            (
                "SEC/EDGAR management-guidance extraction skipped for the "
                f"{market.value} market"
            )
            if not sec_backed_evidence_allowed
            else "AI management-guidance extraction skipped because OpenAI is not configured"
        )
    return ForecastConstructionResult(
        forecast_parameters=forecast_parameters,
        terminal_configuration=terminal_configuration,
        terminal_method=terminal_method,
        cash_flow_timing=cash_flow_timing,
        forecast_service=forecast_service,
        bridge_resolver=bridge_resolver,
        financials=financials,
        valuation_date=valuation_date,
        forecast=forecast,
        seed_forecast=forecast,
        guidance_overlay=guidance_overlay,
        warnings=tuple(warnings),
    )


def _null_step(_name):
    class _Step:
        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

    return _Step()
__all__ = ["ForecastConstructionResult", "construct_fcff_forecast"]
