"""FCFF DCF execution use case.

All forecast and assumption construction happens before this boundary.  This
module only turns the resolved forecast inputs into the DCF service request and
keeps the existing buyback and warning handling in one focused stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from edgarito.cli.use_cases.context import ValuationDependencyContext
from edgarito.services.valuation.fcff_dcf import FcffDcfService
from edgarito.services.valuation.models import (
    FcffDcfParameters,
    ShareRepurchaseParameters,
    TerminalMetric,
)


@dataclass(frozen=True)
class DcfExecutionResult:
    """Result and reusable inputs produced by the DCF stage."""

    result: Any
    parameters: FcffDcfParameters
    share_repurchase_parameters: ShareRepurchaseParameters | None


def _resolve(dependencies, name: str, default):
    return ValuationDependencyContext(dependencies).resolve(name, default)


def execute_fcff_dcf(
    *,
    args,
    profile,
    forecast,
    resolved,
    capital_bridge,
    multistage_plan,
    valuation_date,
    terminal_method,
    cash_flow_timing,
    terminal_configuration,
    terminal_roic,
    asset_life_resolution=None,
    dependencies=None,
) -> DcfExecutionResult:
    """Execute the unchanged FCFF DCF algorithm behind a narrow use-case API."""

    parameters_type = _resolve(dependencies, "FcffDcfParameters", FcffDcfParameters)
    dcf_service_type = _resolve(dependencies, "FcffDcfService", FcffDcfService)
    repurchase_parameters_type = _resolve(
        dependencies, "ShareRepurchaseParameters", ShareRepurchaseParameters
    )
    terminal_metric_type = _resolve(dependencies, "TerminalMetric", TerminalMetric)
    parameters = parameters_type(
        wacc=resolved.wacc,
        wacc_source=resolved.wacc_source,
        cash_flow_timing=cash_flow_timing,
        terminal_method=terminal_method,
        perpetual_growth_rate=resolved.perpetual_growth_rate,
        perpetual_growth_source=resolved.perpetual_growth_source,
        exit_multiple=(
            (
                args.exit_multiple
                if args.exit_multiple is not None
                else terminal_configuration.exit_multiple
            )
            if terminal_method.value == "exit_multiple"
            else None
        ),
        exit_metric=(
            terminal_metric_type(args.exit_metric)
            if args.exit_metric is not None
            else terminal_configuration.exit_metric
        ),
    )
    repurchase_configuration = profile.valuation.share_repurchases
    repurchase_cash = (
        tuple(args.buyback_cash)
        if args.buyback_cash is not None
        else repurchase_configuration.annual_cash_amounts
    )
    share_repurchase_parameters = None
    if not args.no_buybacks and repurchase_cash:
        share_repurchase_parameters = repurchase_parameters_type(
            annual_cash_amounts=repurchase_cash,
            initial_purchase_price=(
                args.buyback_price
                if args.buyback_price is not None
                else repurchase_configuration.initial_purchase_price
            ),
            price_growth_rate=(
                args.buyback_price_growth
                if args.buyback_price_growth is not None
                else repurchase_configuration.price_growth_rate
            ),
            discount_rate=(
                args.buyback_discount_rate
                if args.buyback_discount_rate is not None
                else repurchase_configuration.discount_rate
            ),
            source=(
                "CLI override"
                if args.buyback_cash is not None
                else repurchase_configuration.source or "valuation profile"
            ),
        )
    elif not args.no_buybacks and any(
        value is not None
        for value in (
            args.buyback_price,
            args.buyback_price_growth,
            args.buyback_discount_rate,
        )
    ):
        raise ValueError(
            "Buyback price or rate assumptions require --buyback-cash or a "
            "profile repurchase schedule"
        )

    result = dcf_service_type().value(
        forecast,
        parameters,
        capital_bridge,
        resolved.assumption_set,
        multistage_plan,
        valuation_date,
        share_repurchase_parameters,
    )
    asset_life_warnings = (
        asset_life_resolution.warnings if asset_life_resolution is not None else ()
    )
    if terminal_roic.warnings or asset_life_warnings:
        result = result.model_copy(
            update={
                "warnings": tuple(
                    dict.fromkeys(
                        [
                            *result.warnings,
                            *terminal_roic.warnings,
                            *asset_life_warnings,
                        ]
                    )
                )
            }
        )
    return DcfExecutionResult(
        result=result,
        parameters=parameters,
        share_repurchase_parameters=share_repurchase_parameters,
    )


__all__ = ["DcfExecutionResult", "execute_fcff_dcf"]
