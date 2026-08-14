"""Valuation result, export, and presentation use case."""

from __future__ import annotations

from edgarito.cli.presentation.console import ValuationReportConsolePresenter
from edgarito.cli.use_cases.context import (
    ValuationDependencyContext,
    call_with_context,
)
from edgarito.cli.use_cases.forward_assumptions import (
    forward_growth_evidence as _default_forward_growth_evidence,
)
from edgarito.services.export import ValuationExcelRenderer
from edgarito.services.financials.availability import (
    ObservationAvailabilityMode,
)
from edgarito.services.valuation import (
    DecisionScenarioPolicy,
    DecisionValuationService,
    IntrinsicDecisionContext,
)


def _resolve(dependencies, name: str, default):
    return ValuationDependencyContext(dependencies).resolve(name, default)


def present_valuation_results(
    *,
    args,
    selected_model,
    profile,
    financials,
    forecast,
    seed_forecast,
    forecast_parameters,
    result,
    capital_bridge,
    terminal_roic,
    multistage_configuration,
    use_multistage,
    valuation_date,
    tax_assumption,
    share_repurchase_parameters,
    forward_evidence,
    guidance_overlay,
    operating_audit,
    profile_context,
    configured_terminal_roic,
    discount_configuration,
    terminal_configuration,
    automatic_inputs,
    relative_result,
    provider_relative_result,
    peer_report,
    additional_warnings,
    dependencies=None,
) -> None:
    """Export and render a completed valuation without recalculating it."""

    excel_renderer = _resolve(
        dependencies, "ValuationExcelRenderer", ValuationExcelRenderer
    )
    decision_context_type = _resolve(
        dependencies, "IntrinsicDecisionContext", IntrinsicDecisionContext
    )
    decision_policy_type = _resolve(
        dependencies, "DecisionScenarioPolicy", DecisionScenarioPolicy
    )
    decision_service_type = _resolve(
        dependencies, "DecisionValuationService", DecisionValuationService
    )
    report_presenter = _resolve(
        dependencies,
        "ValuationReportConsolePresenter",
        ValuationReportConsolePresenter,
    )
    valuation_step = _resolve(dependencies, "_valuation_step", _null_step)
    forward_growth_evidence = _resolve(
        dependencies, "_forward_growth_evidence", _default_forward_growth_evidence
    )

    if getattr(args, "excel_output", None) is not None:
        output = excel_renderer().render(
            forecast,
            result,
            args.excel_output,
            relative=relative_result or provider_relative_result,
            peer_report=peer_report,
            discount_timing_basis="calendar",
        )
        print(f"Exported valuation Excel workbook to {output}")
    current_price = (
        relative_result.current_price if relative_result is not None else None
    )
    if current_price is None and provider_relative_result is not None:
        current_price = provider_relative_result.current_price
    if current_price is None and peer_report is not None:
        current_price = peer_report.target.price
    market_data = automatic_inputs.get("market_data")
    if current_price is None and market_data is not None:
        latest_price = market_data.latest_price
        current_price = latest_price.close if latest_price is not None else None
    decision_configuration = profile.valuation.decision_analysis
    decision_result = None
    if current_price is not None and decision_configuration.enabled:
        decision_context = decision_context_type(
            financials=financials,
            requested_parameters=forecast_parameters,
            seed_forecast=seed_forecast,
            base_forecast=forecast,
            base_result=result,
            capital_bridge=capital_bridge,
            terminal_roic=terminal_roic.value,
            multistage_configuration=multistage_configuration,
            use_multistage=use_multistage,
            valuation_date=valuation_date,
            availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
            normalized_tax_rate=(
                tax_assumption.value if tax_assumption is not None else None
            ),
            share_repurchase_parameters=share_repurchase_parameters,
            forward_evidence=forward_evidence
            or call_with_context(
                forward_growth_evidence,
                profile_context.lifecycle,
                profile_context.economic_traits,
                guidance_overlay,
                tuple(item.fiscal_year for item in forecast.observations),
                context=dependencies,
            ),
            flexible_revenue_growth=(
                args.revenue_growth is None
                and profile.forecast.fcff.revenue_growth is None
                and not (
                    guidance_overlay
                    and any(
                        item.driver == "revenue_growth"
                        for item in guidance_overlay.applications
                    )
                )
            ),
            flexible_operating_margin=(
                args.operating_margin is None
                and profile.forecast.fcff.operating_margin is None
                and not (
                    guidance_overlay
                    and any(
                        item.driver == "operating_margin"
                        for item in guidance_overlay.applications
                    )
                )
            ),
            flexible_terminal_roic=configured_terminal_roic is None,
            flexible_wacc=(args.wacc is None and discount_configuration.wacc is None),
            flexible_terminal_growth=(
                args.terminal_growth is None
                and terminal_configuration.perpetual_growth_rate is None
            ),
        )
        try:
            decision_policy = decision_policy_type(
                revenue_growth_delta=decision_configuration.revenue_growth_delta,
                operating_margin_delta=decision_configuration.operating_margin_delta,
                bear_wacc_delta=decision_configuration.bear_wacc_delta,
                bull_wacc_delta=decision_configuration.bull_wacc_delta,
                terminal_growth_delta=decision_configuration.terminal_growth_delta,
                terminal_roic_spread_change=(
                    decision_configuration.terminal_roic_spread_change
                ),
                fair_value_band=decision_configuration.fair_value_band,
                sensitivity_size=decision_configuration.sensitivity_size,
            )
            with call_with_context(
                valuation_step,
                "running decision and sensitivity analysis",
                context=dependencies,
            ):
                decision_result = decision_service_type(decision_policy).build(
                    decision_context,
                    current_price,
                    relative_result or provider_relative_result,
                )
        except ValueError as exc:
            additional_warnings.append(f"Decision analysis unavailable: {exc}")
    elif current_price is None and decision_configuration.enabled:
        additional_warnings.append(
            "Decision analysis skipped: no current market price was available"
        )
    report_output = report_presenter().render(
        intrinsic=result if selected_model in {"fcff-dcf", "both"} else None,
        peer_report=peer_report,
        relative=relative_result,
        provider_relative=provider_relative_result,
        decision=decision_result,
        profile_name=profile.name,
        show_scenarios=args.scenarios,
        show_sensitivity=args.sensitivity,
        show_reverse_dcf=args.reverse_dcf,
        verbose=args.verbose or args.audit,
        additional_warnings=tuple(additional_warnings),
        management_guidance=guidance_overlay,
        operating_audit=operating_audit,
    )
    if report_output:
        print(report_output)


def _null_step(_name):
    class _Step:
        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

    return _Step()


__all__ = ["present_valuation_results"]
