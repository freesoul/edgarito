"""CLI application composition root.

The command-specific workflows live in the focused use-case modules.  This
module supplies the historical private adapters and the parsed-command
dispatcher; ``cli.__main__`` adds the final facade-level monkeypatch bridge.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import contextmanager
from typing import Optional

from pydantic import ValidationError

import edgarito.cli.comparables as _comparables
from edgarito.cli.parser import build_parser
from edgarito.cli.use_cases import (
    financial_retrieval,
    forward_assumptions,
    operating_evidence,
)
from edgarito.cli.use_cases import forecast as forecast_use_case
from edgarito.cli.use_cases import valuation as valuation_use_case
from edgarito.cli.use_cases.context import dependency
from edgarito.logger import configure_logger

logger = logging.getLogger(__name__)

_build_comparable_report = _comparables._build_comparable_report
_resolve_comparable_peer_symbols = _comparables._resolve_comparable_peer_symbols
_default_run_comparables = _comparables._run_comparables


# These imports are deliberately re-exported through the application boundary.
# They form the provider/presenter surface historically patched on
# ``cli.__main__``.  The use cases use them as defaults and read overrides from
# this module only through their explicit context argument.
for _module in (
    financial_retrieval,
    forecast_use_case,
    forward_assumptions,
    operating_evidence,
    valuation_use_case,
):
    for _name, _value in vars(_module).items():
        if not _name.startswith("__"):
            globals().setdefault(_name, _value)


@contextmanager
def _valuation_step(name: str, *, context=None):
    del context
    with valuation_use_case.valuation_step(name):
        yield


async def _run_financials(args, *, context=None):
    return await financial_retrieval.run_financials(
        args, context=context
    )


async def _run_sec_inventory(args, *, context=None):
    return await financial_retrieval.run_sec_inventory(
        args, context=context
    )


async def _run_metrics(args, *, context=None):
    return await financial_retrieval.run_metrics(args, context=context)


async def _run_export(args, *, context=None):
    return await financial_retrieval.run_export(args, context=context)


async def _run_red_flags(args, *, context=None):
    return await financial_retrieval.run_red_flags(
        args, context=context
    )


async def _run_classification(args, *, context=None):
    return await financial_retrieval.run_classification(
        args, context=context
    )


async def _run_forecast(args, *, context=None):
    return await forecast_use_case.run_forecast(
        args, context=context
    )


def _load_selected_valuation_profile(args, *, context=None):
    return forecast_use_case.load_selected_valuation_profile(
        args, context=context
    )


def _resolve_depreciable_asset_life_configuration(
    financials, profile_context, configuration, *, context=None
):
    return forecast_use_case.resolve_depreciable_asset_life_configuration(
        financials,
        profile_context,
        configuration,
        context=context,
    )


def _fcff_parameters(args, configured, *, context=None):
    return forecast_use_case.fcff_parameters(args, configured, context=context)


def _market_for_args(args, *, context=None):
    return forecast_use_case.market_for_args(args, context=context)


async def _retrieve_classification(
    args, *, provider, crosscheck, context=None
):
    return await financial_retrieval.retrieve_classification(
        args,
        provider=provider,
        crosscheck=crosscheck,
        context=context,
    )


async def _retrieve_financials(args, granularity, concepts, *, context=None):
    return await financial_retrieval.retrieve_financials(
        args, granularity, concepts, context=context
    )


async def _retrieve_forward_estimates(
    args, financials, forecast, *, use_cache, make_cache, context=None
):
    return await forward_assumptions.retrieve_forward_estimates(
        args,
        financials,
        forecast,
        use_cache=use_cache,
        make_cache=make_cache,
        context=context,
    )


async def _retrieve_operating_evidence(
    financials,
    forecast,
    as_of,
    *,
    provider=None,
    args=None,
    metadata=None,
    fiscal_years=None,
    availability_mode=None,
    context=None,
):
    return await operating_evidence.retrieve_operating_evidence(
        financials,
        forecast,
        as_of,
        provider=provider,
        args=args,
        metadata=metadata,
        fiscal_years=fiscal_years,
        availability_mode=availability_mode,
        context=context,
    )


def _operating_quality_audit(
    evidence, *, discovery_warnings=(), context=None
):
    return operating_evidence.operating_quality_audit(
        evidence,
        discovery_warnings=discovery_warnings,
        context=context,
    )


def _retain_operating_audit_metadata(current, discovered, *, context=None):
    del context
    return operating_evidence.retain_operating_audit_metadata(current, discovered)


def _default_operating_vocabulary_audit(profile, *, context=None):
    return operating_evidence.default_operating_vocabulary_audit(
        profile, context=context
    )


def _operating_evidence_provider(
    args, financials, *, market=None, context=None
):
    return operating_evidence.operating_evidence_provider(
        args,
        financials,
        market=market,
        context=context,
    )


def _call_with_supported_kwargs(resolver, kwargs, *, context=None):
    del context
    return operating_evidence.call_with_supported_kwargs(resolver, kwargs)


def _merge_forward_growth_evidence(
    management,
    consensus,
    *,
    management_has_revenue_guidance,
    forecast_years=(),
    context=None,
):
    del context
    return forward_assumptions.merge_forward_growth_evidence(
        management,
        consensus,
        management_has_revenue_guidance=management_has_revenue_guidance,
        forecast_years=forecast_years,
    )


def _materialize_forward_revenue_anchors(
    parameters, estimates, forecast, warnings, *, context=None
):
    del context
    return forward_assumptions.materialize_forward_revenue_anchors(
        parameters, estimates, forecast, warnings
    )


def _financial_snapshot_warnings(financials, args, *, context=None):
    del context
    return forward_assumptions.financial_snapshot_warnings(financials, args)


async def _management_guidance_overlay(
    args,
    financials,
    parameters,
    baseline,
    valuation_date,
    *,
    market=None,
    context=None,
):
    return await forward_assumptions.management_guidance_overlay(
        args,
        financials,
        parameters,
        baseline,
        valuation_date,
        market=market,
        context=context,
    )


def _forward_growth_evidence(
    lifecycle, economic_traits, guidance_overlay, forecast_years=(), *, context=None
):
    del context
    return forward_assumptions.forward_growth_evidence(
        lifecycle, economic_traits, guidance_overlay, forecast_years
    )


async def _retrieve_automatic_assumption_inputs(
    args,
    financials,
    currency,
    *,
    needs_wacc,
    needs_terminal,
    sector_override=None,
    industry_override=None,
    context=None,
):
    return await forward_assumptions.retrieve_automatic_assumption_inputs(
        args,
        financials,
        currency,
        needs_wacc=needs_wacc,
        needs_terminal=needs_terminal,
        sector_override=sector_override,
        industry_override=industry_override,
        context=context,
    )


def _apply_classification_overrides(
    classification, *, sector=None, industry=None, context=None
):
    del context
    return forward_assumptions.apply_classification_overrides(
        classification,
        sector=sector,
        industry=industry,
    )


def _resolve_wacc(override, configuration, *, context=None):
    return forward_assumptions.resolve_wacc(
        override, configuration, context=context
    )


def _granularity(period, *, context=None):
    del context
    return financial_retrieval.granularity(period)


async def _run_valuation(args, *, context=None):
    return await valuation_use_case.run_valuation(
        args, context=context
    )


async def _run_profile_intrinsic_valuation(
    args, profile, selected_model, *, context=None
):
    return await valuation_use_case.run_profile_intrinsic_valuation(
        args,
        profile,
        selected_model,
        context=context,
    )


def _profile_model_runner(*, dependencies=None, **kwargs):
    """Keep the former nested runner seam facade-aware."""

    return valuation_use_case.profile_model_runner(
        **kwargs,
        dependencies=dependencies,
    )


async def _run_valuation_models(args, *, context=None):
    return await valuation_use_case.run_valuation_models(
        args, context=context
    )


async def _run_specialized_inputs(args, *, context=None):
    return await valuation_use_case.run_specialized_inputs(
        args, context=context
    )


def _validate_limit(limit, *, context=None):
    del context
    return financial_retrieval.validate_limit(limit)


def _parse_mappings(values, option, *, context=None):
    del context
    return financial_retrieval.parse_mappings(values, option)


def _parse_provider_symbols(values, *, context=None):
    return financial_retrieval.parse_provider_symbols(
        values, context=context
    )


async def _run_comparables(args, *, context=None):
    del context
    return await _default_run_comparables(args)


def _call_handler(handler, args, context):
    """Call a command handler while tolerating legacy one-argument fakes."""

    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return handler(args, context=context)
    if "context" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return handler(args, context=context)
    return handler(args)


def main(
    argv: Optional[list[str]] = None,
    *,
    context=None,
    dependencies=None,
) -> int:
    if context is None:
        context = dependencies
    parser_factory = dependency(context, "build_parser", build_parser)
    parser = parser_factory()
    args = parser.parse_args(argv)
    configure = dependency(context, "configure_logger", configure_logger)
    configure(
        logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING
    )
    handlers = {
        "financials": ("_run_financials", financial_retrieval.run_financials),
        "sec-inventory": ("_run_sec_inventory", financial_retrieval.run_sec_inventory),
        "metrics": ("_run_metrics", financial_retrieval.run_metrics),
        "export": ("_run_export", financial_retrieval.run_export),
        "red-flags": ("_run_red_flags", financial_retrieval.run_red_flags),
        "forecast": ("_run_forecast", forecast_use_case.run_forecast),
        "valuation": ("_run_valuation", valuation_use_case.run_valuation),
        "valuation-models": (
            "_run_valuation_models",
            valuation_use_case.run_valuation_models,
        ),
        "classification": (
            "_run_classification",
            financial_retrieval.run_classification,
        ),
        "comparables": ("_run_comparables", _default_run_comparables),
        "specialized-inputs": (
            "_run_specialized_inputs",
            valuation_use_case.run_specialized_inputs,
        ),
    }
    try:
        handler_name, default_handler = handlers.get(args.command, (None, None))
        if handler_name is None:
            return 1
        handler = dependency(context, handler_name, default_handler)
        return asyncio.run(_call_handler(handler, args, context))
    except (ValueError, RuntimeError, FileNotFoundError, ValidationError) as exc:
        parser.error(str(exc))
    return 1
