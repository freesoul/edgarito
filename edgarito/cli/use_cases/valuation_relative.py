"""Relative and comparable valuation use cases.

The command coordinator supplies the comparable report and intrinsic anchor;
this module owns the equity-basis translation and provider-neutral relative
valuation calculation.  It deliberately contains no CLI parsing or printing.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from decimal import Decimal

from edgarito.cli.use_cases.context import (
    ValuationDependencyContext,
    call_with_context,
)
from edgarito.cli.use_cases.financial_retrieval import parse_provider_symbols
from edgarito.enums.provider import ProviderName
from edgarito.schemas.valuation.assumptions import ValuationAssumptionKind
from edgarito.schemas.valuation.intrinsic import IntrinsicValuationContext
from edgarito.schemas.valuation.relative import (
    ForwardValuationMetric,
    RelativeNumeratorBasis,
)
from edgarito.schemas.valuation.selection import (
    MultipleConfidence,
    RelativeValuationBasis,
)
from edgarito.services.financials.availability import (
    ObservationAvailabilityMode,
)
from edgarito.services.valuation import (
    ComparableImpliedValuationService,
    ForwardPeerMultiplesService,
    HistoricalMultiplesService,
    MultipleResolver,
)
from edgarito.services.valuation.intrinsic import (
    PropertyNavAdapter,
    ReitAffoAdapter,
    SotpValuationInput,
    SotpValuationService,
)
from edgarito.services.valuation.models import ResolvedMultiple
from edgarito.services.valuation.relative import (
    EQUITY_RELATIVE_BASES,
    ProviderNeutralRelativeValuationService,
)


@dataclass(frozen=True)
class ComparableRetrievalResult:
    """Comparable evidence and a provider failure diagnostic."""

    bundle: object | None = None
    error: str | None = None


@dataclass(frozen=True)
class RelativeValuationResult:
    """Peer report and relative valuation outputs for presentation."""

    peer_report: object | None = None
    relative_result: object | None = None
    provider_relative_result: object | None = None
    warnings: tuple[str, ...] = ()


def _resolve(dependencies, name: str, default):
    return ValuationDependencyContext(dependencies).resolve(name, default)


async def retrieve_comparable_bundle(
    *,
    args,
    profile,
    financials,
    selected_model,
    valuation_date: datetime.date,
    dependencies=None,
) -> ComparableRetrievalResult:
    """Discover and retrieve comparable-company evidence when requested."""

    if selected_model not in {"comparables", "both"}:
        return ComparableRetrievalResult()
    parse_symbols = _resolve(
        dependencies, "_parse_provider_symbols", parse_provider_symbols
    )
    resolve_peers = _resolve(
        dependencies,
        "_resolve_comparable_peer_symbols",
        lambda _args, _profile, _target: ((), "none"),
    )
    build_report = _resolve(
        dependencies,
        "_build_comparable_report",
        _missing_comparable_report,
    )
    valuation_step = _resolve(dependencies, "_valuation_step", _null_step)
    provider_symbols = call_with_context(
        parse_symbols,
        args.provider_symbol,
        context=dependencies,
    )
    fallback_symbol = args.ticker or financials.ticker
    if fallback_symbol is None:
        return ComparableRetrievalResult(
            error="Automatic peer discovery requires a ticker or a provider symbol"
        )
    target_symbol = (
        provider_symbols.get(ProviderName.YAHOO, fallback_symbol).strip().upper()
    )
    peer_symbols, peer_source = call_with_context(
        resolve_peers,
        args,
        profile,
        target_symbol,
        context=dependencies,
    )
    try:
        with call_with_context(
            valuation_step,
            "retrieving comparable-company data",
            context=dependencies,
        ):
            bundle = await call_with_context(
                build_report,
                args,
                profile,
                target_symbol,
                peer_symbols,
                peer_source=peer_source,
                as_of=valuation_date,
                availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
                context=dependencies,
            )
    except (RuntimeError, ValueError) as exc:
        return ComparableRetrievalResult(error=str(exc))
    return ComparableRetrievalResult(bundle=bundle)


def calculate_relative_valuation(
    *,
    args,
    profile,
    comparable_bundle,
    comparable_error,
    selected_model,
    financials,
    forecast,
    result,
    capital_bridge,
    resolved,
    valuation_date: datetime.date,
    terminal_method,
    dependencies=None,
) -> RelativeValuationResult:
    """Resolve peer multiples and calculate the selected relative output."""

    warnings: list[str] = []
    peer_report = None
    relative_result = None
    provider_relative_result = None
    if selected_model not in {"comparables", "both"}:
        return RelativeValuationResult()
    if terminal_method.value != "perpetuity_growth":
        raise ValueError(
            "Relative multiple resolution requires a perpetuity-growth DCF "
            "for its independent fundamental anchor"
        )
    relative_configuration = profile.relative_valuation
    basis = RelativeValuationBasis(
        args.relative_basis or relative_configuration.basis
    )
    horizon_years = (
        args.horizon_years
        if args.horizon_years is not None
        else relative_configuration.horizon_years
    )
    if horizon_years <= 0:
        raise ValueError("--horizon-years must be positive")
    if comparable_bundle is None:
        warnings.append(
            "Relative valuation skipped: automatic peer evidence could not be "
            f"prepared ({comparable_error or 'unknown provider failure'})"
        )
        return RelativeValuationResult(warnings=tuple(warnings))

    peer_report = comparable_bundle.report
    comparable_financials = comparable_bundle.target_financials
    comparable_market = comparable_bundle.target_market
    comparable_peer_sources = comparable_bundle.peer_sources
    forward_peer_service = _resolve(
        dependencies, "ForwardPeerMultiplesService", ForwardPeerMultiplesService
    )
    historical_service = _resolve(
        dependencies, "HistoricalMultiplesService", HistoricalMultiplesService
    )
    multiple_resolver = _resolve(dependencies, "MultipleResolver", MultipleResolver)
    comparable_service = _resolve(
        dependencies,
        "ComparableImpliedValuationService",
        ComparableImpliedValuationService,
    )
    equity_valuator = _resolve(
        dependencies, "_equity_relative_valuation", equity_relative_valuation
    )
    if basis not in EQUITY_RELATIVE_BASES:
        peer_report = forward_peer_service().build(
            peer_report,
            {
                symbol: peer_financials
                for symbol, (peer_financials, _market) in comparable_peer_sources.items()
            },
            basis,
            valuation_date,
            horizon_years,
        )
    relative_ready = True
    if peer_report.universe.discovery_confidence == "low":
        warnings.append(
            "Relative valuation skipped: selected peer evidence has low "
            "economic-comparability confidence"
        )
        relative_ready = False
    elif len(peer_report.universe.selected_tickers) < (
        relative_configuration.multiple_resolution.minimum_peer_sample
    ):
        warnings.append(
            "Relative valuation skipped: peer evidence is below the "
            f"configured minimum sample of "
            f"{relative_configuration.multiple_resolution.minimum_peer_sample}"
        )
        relative_ready = False
    if relative_ready and basis in EQUITY_RELATIVE_BASES:
        try:
            provider_relative_result = equity_valuator(
                basis=basis,
                report=peer_report,
                profile=profile,
                intrinsic=result,
                valuation_date=valuation_date,
                horizon_years=horizon_years,
                discount_rate=(
                    resolved.assumption_set.find(
                        ValuationAssumptionKind.COST_OF_EQUITY
                    ).value
                    if resolved.assumption_set.find(
                        ValuationAssumptionKind.COST_OF_EQUITY
                    )
                    is not None
                    else resolved.wacc
                ),
                dependencies=dependencies,
            )
        except ValueError as exc:
            warnings.append(f"Relative valuation skipped: {exc}")
    if relative_ready and basis not in EQUITY_RELATIVE_BASES:
        target_history = historical_service().compute(
            comparable_financials,
            comparable_market,
            basis,
        )
        peer_histories = tuple(
            historical_service().compute(peer_financials, market, basis)
            for peer_financials, market in comparable_peer_sources.values()
        )
        resolved_multiple = multiple_resolver().resolve(
            basis=basis,
            target=peer_report.target,
            target_history=target_history,
            peer_histories=peer_histories,
            peer_report=peer_report,
            target_forecast=forecast,
            intrinsic_valuation=result,
            horizon_years=horizon_years,
            policy=relative_configuration.multiple_resolution,
        )
        if peer_report.warnings:
            resolved_multiple = resolved_multiple.model_copy(
                update={
                    "warnings": tuple(
                        dict.fromkeys(
                            [*resolved_multiple.warnings, *peer_report.warnings]
                        )
                    )
                }
            )
        relative_result = comparable_service().value(
            target_forecast=forecast,
            capital_bridge=capital_bridge,
            projected_shares=capital_bridge.diluted_shares,
            resolved_multiple=resolved_multiple,
            valuation_date=valuation_date,
            horizon_years=horizon_years,
            discount_rate=resolved.wacc,
            current_price=peer_report.target.price,
            analyst_target_price=args.analyst_target_price,
            intrinsic_value_per_share=result.value_per_share,
        )
    return RelativeValuationResult(
        peer_report=peer_report,
        relative_result=relative_result,
        provider_relative_result=provider_relative_result,
        warnings=tuple(warnings),
    )


def _missing_comparable_report(*_args, **_kwargs):
    raise RuntimeError("comparable report use case is unavailable")


def _null_step(_name):
    class _Step:
        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

    return _Step()


def equity_relative_valuation(
    *,
    basis,
    report,
    profile,
    intrinsic,
    valuation_date: datetime.date,
    horizon_years: Decimal,
    discount_rate: Decimal,
    dependencies=None,
):
    """Value an equity relative basis using independent peer evidence."""

    property_nav_adapter = _resolve(
        dependencies, "PropertyNavAdapter", PropertyNavAdapter
    )
    reit_affo_adapter = _resolve(dependencies, "ReitAffoAdapter", ReitAffoAdapter)
    sotp_service = _resolve(
        dependencies, "SotpValuationService", SotpValuationService
    )
    sotp_input = _resolve(dependencies, "SotpValuationInput", SotpValuationInput)
    intrinsic_context = _resolve(
        dependencies, "IntrinsicValuationContext", IntrinsicValuationContext
    )
    resolved_multiple_type = _resolve(
        dependencies, "ResolvedMultiple", ResolvedMultiple
    )
    forward_metric_type = _resolve(
        dependencies, "ForwardValuationMetric", ForwardValuationMetric
    )
    numerator_basis_type = _resolve(
        dependencies, "RelativeNumeratorBasis", RelativeNumeratorBasis
    )
    multiple_confidence_type = _resolve(
        dependencies, "MultipleConfidence", MultipleConfidence
    )
    relative_service = _resolve(
        dependencies,
        "ProviderNeutralRelativeValuationService",
        ProviderNeutralRelativeValuationService,
    )

    fundamentals = report.target.fundamentals
    metric = None
    label = ""
    if basis == RelativeValuationBasis.PE:
        earnings = profile.valuation.dividend_discount.earnings
        metric = (
            earnings[0]
            if earnings
            else (
                fundamentals.net_income_common
                if fundamentals.net_income_common is not None
                else fundamentals.net_income
            )
        )
        label = (
            "forward common earnings" if earnings else "LTM common earnings fallback"
        )
    elif basis == RelativeValuationBasis.PRICE_TO_BOOK:
        metric = (
            profile.valuation.residual_income.starting_book_value
            or fundamentals.book_equity
        )
        label = "common book equity"
    elif basis == RelativeValuationBasis.PRICE_TO_TANGIBLE_BOOK:
        metric = fundamentals.tangible_book_equity
        label = "tangible common equity"
    elif basis == RelativeValuationBasis.PRICE_TO_AFFO:
        affo = profile.valuation.reit.affo_forecast
        if affo:
            metric = affo[0]
            label = "forward AFFO"
        else:
            metric = reit_affo_adapter.derive_affo(
                ffo=profile.valuation.reit.ffo,
                recurring_adjustments=profile.valuation.reit.recurring_affo_adjustments,
            )
            label = "current AFFO"
    elif basis == RelativeValuationBasis.PRICE_TO_NAV:
        components = profile.valuation.sotp.components
        if not components and profile.valuation.reit.properties:
            components = property_nav_adapter().to_components(
                profile.valuation.reit.properties,
                valuation_date=valuation_date,
            )
        nav = sotp_service().value(
            sotp_input(
                context=intrinsic_context(
                    company_id=report.target.company_id,
                    company_name=report.target.company_name,
                    ticker=report.target.ticker,
                    valuation_date=valuation_date,
                    currency=report.target.currency,
                    diluted_shares=report.target.fundamentals.shares,
                ),
                components=components,
                adjustments=profile.valuation.sotp.adjustments,
                holding_company_discount=profile.valuation.sotp.holding_company_discount,
            )
        )
        metric = nav.equity_value
        label = "profile NAV"
    if metric is None or metric <= 0:
        raise ValueError(f"positive target denominator unavailable for {basis.value}")
    summary = next((item for item in report.summaries if item.basis == basis), None)
    if summary is None:
        raise ValueError(f"peer denominator unavailable for {basis.value}")
    policy = profile.relative_valuation.multiple_resolution
    if summary.sample_size < policy.minimum_peer_sample:
        raise ValueError(
            f"only {summary.sample_size} usable peer denominators; "
            f"policy requires {policy.minimum_peer_sample}"
        )
    fundamental = intrinsic.equity_value / metric
    point = summary.median
    lower = summary.percentile_25 or summary.minimum
    upper = summary.percentile_75 or summary.maximum
    confidence = (
        multiple_confidence_type.HIGH
        if summary.sample_size >= 8
        else multiple_confidence_type.MEDIUM
        if summary.sample_size >= policy.minimum_peer_sample
        else multiple_confidence_type.LOW
    )
    market_cap = report.target.market_capitalization
    current_anchor = market_cap / metric if market_cap is not None else None
    resolved_multiple = resolved_multiple_type(
        basis=basis,
        point_estimate=point,
        lower_bound=max(Decimal("0.01"), lower),
        upper_bound=upper,
        fundamental_anchor=fundamental,
        peer_anchor=summary.median,
        peer_anchor_source="current peer denominator evidence",
        peer_anchor_percentile_25=summary.percentile_25,
        peer_anchor_percentile_75=summary.percentile_75,
        current_target_anchor=current_anchor,
        market_anchor=summary.median,
        observed_premium=(
            current_anchor / summary.median - Decimal(1)
            if current_anchor is not None
            else None
        ),
        resolved_premium=Decimal(0),
        historical_persistence=Decimal(0),
        fundamental_support=Decimal(1),
        horizon_retention=Decimal(1),
        persistence_factor=Decimal(0),
        sample_size=summary.sample_size,
        peer_confidence=confidence,
        confidence=confidence,
        methodology=(
            "Independent peer median with peer percentile bounds; the DCF-implied "
            "equity multiple is retained as a diagnostic only"
        ),
        warnings=(
            ("Target denominator uses current/LTM fallback rather than a forecast",)
            if "fallback" in label
            else ()
        ),
    )
    target_date = valuation_date + datetime.timedelta(
        days=int(horizon_years * Decimal(365))
    )
    return relative_service().value(
        valuation_date=valuation_date,
        horizon_years=horizon_years,
        metric=forward_metric_type(
            basis=basis,
            amount=metric,
            label=label,
            target_date=target_date,
            currency=report.target.currency,
            numerator_basis=numerator_basis_type.EQUITY_VALUE,
        ),
        diluted_shares=report.target.fundamentals.shares,
        discount_rate=discount_rate,
        resolved_multiple=resolved_multiple,
        current_price=report.target.price,
    )


__all__ = ["equity_relative_valuation"]
