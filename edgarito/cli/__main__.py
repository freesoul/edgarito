import argparse
import asyncio
import datetime
import logging
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from edgarito.cli.comparables import (
    _build_comparable_report,
    _resolve_comparable_peer_symbols,
    _run_comparables,
)
from edgarito.cli.parser import build_parser
from edgarito.cli.presentation.console import (
    ClassificationConsolePresenter,
    ComparableImpliedValuationConsolePresenter,
    ComparableMultiplesConsolePresenter,
    DecisionValuationConsolePresenter,
    FcffDcfConsolePresenter,
    FinancialsConsolePresenter,
    ForecastConsolePresenter,
    MetricsConsolePresenter,
    SpecializedExtractionConsolePresenter,
    ValuationSelectionConsolePresenter,
)
from edgarito.config.valuation import ForecastMethod, ValuationProfileLoader
from edgarito.enums.granularity import Granularity
from edgarito.enums.market import Market
from edgarito.enums.provider import ProviderName
from edgarito.logger import configure_logger
from edgarito.schemas.market import ReferenceSeriesKind, ReferenceValueUnit
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.valuation.assumptions import ValuationAssumptionKind
from edgarito.schemas.valuation.specialized import SpecializedInputType
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.forecasting import (
    AdaptiveMultistageFcffForecastService,
    FcffForecastParameters,
    FcffForecastService,
    SimplifiedFcfForecastParameters,
    SimplifiedFcfForecastService,
)
from edgarito.services.metrics import FinancialMetric, FinancialMetricsService
from edgarito.services.normalization.classification import (
    CompanyClassificationNormalizer,
)
from edgarito.services.normalization.yahoo_market import YahooMarketNormalizer
from edgarito.services.providers.damodaran import DamodaranClient
from edgarito.services.providers.ecb import EcbClient
from edgarito.services.providers.edgar import EdgarClient
from edgarito.services.providers.fred import FredClient
from edgarito.services.providers.treasury import TreasuryClient
from edgarito.services.providers.yahoo import YahooFinanceClient
from edgarito.services.reconciliation.classification import (
    CompanyClassificationService,
)
from edgarito.services.reconciliation.financials import FinancialDataService
from edgarito.services.valuation import (
    BusinessArchetype,
    CashFlowTiming,
    CompanyLifecycle,
    ComparableImpliedValuationService,
    Cyclicality,
    DecisionScenarioPolicy,
    DecisionValuationService,
    DiscountRateService,
    EcbMarketDataCurrencyConverter,
    EconomicTrait,
    FcffDcfCapitalBridgeResolver,
    FcffDcfParameters,
    FcffDcfService,
    ForwardPeerMultiplesService,
    HistoricalMultiplesService,
    IntrinsicDecisionContext,
    MultipleResolver,
    RelativeValuationBasis,
    ShareRepurchaseParameters,
    SpecializedValuationExtractor,
    TerminalMetric,
    TerminalRoicResolver,
    TerminalValueMethod,
    ValuationAssumptionResolver,
    ValuationInput,
    ValuationModelSelector,
    ValuationProfileBuilder,
    ValuationProfileOverrides,
)
from edgarito.settings import (
    ALPHAVANTAGE_API_KEY,
    CLASSIFICATION_PROVIDER_CONFIGURATION,
    FMP_API_KEY,
    FRED_API_KEY,
    OPENFIGI_API_KEY,
    PROVIDER_CONFIGURATION,
)


async def _run_financials(args: argparse.Namespace) -> int:
    _validate_limit(args.limit)
    granularity = _granularity(args.period)

    concepts = (
        {FinancialConcept(value) for value in args.concept} if args.concept else None
    )
    financials = await _retrieve_financials(args, granularity, concepts)

    print(FinancialsConsolePresenter().render(financials, limit=args.limit))
    return 0


async def _run_metrics(args: argparse.Namespace) -> int:
    _validate_limit(args.limit)
    granularity = _granularity(args.period)
    selected_metrics = (
        {FinancialMetric(value) for value in args.metric} if args.metric else None
    )
    concepts = FinancialMetricsService.required_concepts(selected_metrics)
    financials = await _retrieve_financials(args, granularity, concepts)
    metrics = FinancialMetricsService().calculate(
        financials,
        granularity=granularity,
        metrics=selected_metrics,
    )

    print(MetricsConsolePresenter().render(metrics, limit=args.limit))
    return 0


def _load_selected_valuation_profile(args):
    ticker = getattr(args, "ticker", None)
    if ticker:
        profile, _, _ = ValuationProfileLoader.load_for_ticker(ticker, args.profile)
        return profile
    return ValuationProfileLoader.load(args.profile)


async def _run_forecast(args: argparse.Namespace) -> int:
    profile = _load_selected_valuation_profile(args)
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
        parameters = SimplifiedFcfForecastParameters(
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
        service = SimplifiedFcfForecastService()
    else:
        if args.fcf_margin is not None:
            raise ValueError(
                "--fcf-margin requires --forecast-method simplified; use the "
                "FCFF operating, tax, D&A, capex, and working-capital drivers"
            )
        configured = profile.forecast.fcff
        parameters = _fcff_parameters(args, configured)
        service = FcffForecastService()
    financials = await _retrieve_financials(
        args,
        Granularity.ANNUAL,
        service.required_concepts(),
    )
    forecast = service.forecast(financials, parameters)
    print(ForecastConsolePresenter().render(forecast))
    return 0


async def _run_valuation(args: argparse.Namespace) -> int:
    generated_profile_path = None
    should_generate_profile = False
    if args.ticker:
        profile, generated_profile_path, should_generate_profile = (
            ValuationProfileLoader.load_for_ticker(args.ticker, args.profile)
        )
    else:
        profile = ValuationProfileLoader.load(args.profile)
    selected_model = args.model or (
        "both" if profile.relative_valuation.enabled else "fcff-dcf"
    )
    forecast_parameters = _fcff_parameters(args, profile.forecast.fcff)
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
    forecast_service = FcffForecastService()
    bridge_resolver = FcffDcfCapitalBridgeResolver()
    required_concepts = (
        forecast_service.required_concepts()
        | bridge_resolver.required_concepts()
        | {
            FinancialConcept.INTEREST_EXPENSE,
            FinancialConcept.STOCKHOLDERS_EQUITY,
        }
    )
    financials = await _retrieve_financials(
        args,
        None,
        required_concepts,
    )
    valuation_date = datetime.date.today()
    forecast = forecast_service.forecast(
        financials, forecast_parameters, as_of=valuation_date
    )
    seed_forecast = forecast
    bridge_configuration = profile.valuation.capital_bridge
    has_cli_debt_bridge = any(
        value is not None for value in (args.net_debt, args.gross_debt, args.cash)
    )
    capital_bridge = bridge_resolver.resolve(
        financials,
        fiscal_year=forecast.base_fiscal_year,
        period_end=forecast.base_period_end,
        unit=forecast.unit,
        net_debt=(
            args.net_debt if has_cli_debt_bridge else bridge_configuration.net_debt
        ),
        gross_debt=(
            args.gross_debt if has_cli_debt_bridge else bridge_configuration.gross_debt
        ),
        cash_and_equivalents=(
            args.cash
            if has_cli_debt_bridge
            else bridge_configuration.cash_and_equivalents
        ),
        diluted_shares=(
            args.shares
            if args.shares is not None
            else bridge_configuration.diluted_shares
        ),
        non_operating_assets=(
            args.non_operating_assets
            if args.non_operating_assets is not None
            else bridge_configuration.non_operating_assets
        ),
        valuation_date=valuation_date,
    )
    discount_configuration = profile.valuation.discount_rates
    needs_automatic_wacc = (
        args.wacc is None
        and discount_configuration.wacc is None
        and not discount_configuration.can_calculate_wacc
    )
    needs_automatic_terminal = (
        terminal_method == TerminalValueMethod.PERPETUITY_GROWTH
        and args.terminal_growth is None
        and terminal_configuration.perpetual_growth_rate is None
    )
    automatic_inputs = await _retrieve_automatic_assumption_inputs(
        args,
        financials,
        forecast.unit,
        needs_wacc=needs_automatic_wacc,
        needs_terminal=needs_automatic_terminal,
        sector_override=profile.model_selection.sector,
        industry_override=profile.model_selection.industry,
    )
    resolved = ValuationAssumptionResolver().resolve(
        financials=financials,
        capital_bridge=capital_bridge,
        discount_configuration=discount_configuration,
        terminal_configuration=terminal_configuration,
        terminal_is_perpetuity=(
            terminal_method == TerminalValueMethod.PERPETUITY_GROWTH
        ),
        valuation_date=valuation_date,
        wacc_override=args.wacc,
        terminal_growth_override=args.terminal_growth,
        **automatic_inputs,
    )
    profile_context = ValuationProfileBuilder().build(
        financials,
        automatic_inputs.get("classification"),
        ValuationProfileOverrides(
            sector=profile.model_selection.sector,
            industry=profile.model_selection.industry,
            business_archetype=profile.model_selection.business_archetype,
            lifecycle=profile.model_selection.lifecycle,
            cyclicality=profile.model_selection.cyclicality,
            economic_traits=set(profile.model_selection.economic_traits),
        ),
    )
    comparable_bundle = None
    comparable_error = None
    relative_result = None
    if selected_model in {"comparables", "both"}:
        provider_symbols = _parse_provider_symbols(args.provider_symbol)
        fallback_symbol = args.ticker or financials.ticker
        if fallback_symbol is None:
            comparable_error = (
                "Automatic peer discovery requires a ticker or a provider symbol"
            )
        else:
            target_symbol = (
                provider_symbols.get(ProviderName.YAHOO, fallback_symbol)
                .strip()
                .upper()
            )
            peer_symbols, peer_source = _resolve_comparable_peer_symbols(
                args,
                profile,
                target_symbol,
            )
            try:
                comparable_bundle = await _build_comparable_report(
                    args,
                    profile,
                    target_symbol,
                    peer_symbols,
                    peer_source=peer_source,
                    as_of=valuation_date,
                )
            except (RuntimeError, ValueError) as exc:
                comparable_error = str(exc)

    peer_roics = ()
    if comparable_bundle is not None:
        peer_roics = comparable_bundle.reliable_peer_roics
    configured_terminal_roic = (
        args.terminal_roic
        if args.terminal_roic is not None
        else profile.valuation.multistage.terminal_return_on_invested_capital
    )
    terminal_roic = TerminalRoicResolver().resolve(
        financials,
        wacc=resolved.wacc,
        terminal_growth=(
            resolved.perpetual_growth_rate
            if resolved.perpetual_growth_rate is not None
            else profile.valuation.multistage.stable_growth_rate or Decimal(0)
        ),
        valuation_date=valuation_date,
        currency=forecast.unit,
        explicit_roic=configured_terminal_roic,
        explicit_source=(
            "explicit CLI override"
            if args.terminal_roic is not None
            else "explicit valuation profile"
        ),
        lifecycle=profile_context.lifecycle,
        cyclicality=profile_context.cyclicality,
        peer_roics=peer_roics,
    )
    if should_generate_profile and args.ticker and generated_profile_path is not None:
        discovered_peers = (
            comparable_bundle.report.universe.selected_tickers
            if comparable_bundle is not None
            else ()
        )
        profile, generated_profile_path, created = (
            ValuationProfileLoader.create_generated(
                ticker=args.ticker,
                base_profile=profile,
                inferred_profile=profile_context,
                terminal_roic=terminal_roic.value,
                terminal_roic_confidence=terminal_roic.confidence,
                generated_on=valuation_date,
                peers=discovered_peers,
                path=generated_profile_path,
            )
        )
        should_generate_profile = False
        if created:
            print(f"Generated valuation profile: {generated_profile_path.resolve()}")
    resolved = replace(
        resolved,
        assumption_set=resolved.assumption_set.model_copy(
            update={
                "assumptions": (
                    *resolved.assumption_set.assumptions,
                    terminal_roic.assumption,
                )
            }
        ),
    )
    multistage_plan = None
    multistage_configuration = profile.valuation.multistage.model_copy(
        update={
            "terminal_return_on_invested_capital": terminal_roic.value,
        }
    )
    use_multistage = args.projection_method == "adaptive" or (
        args.projection_method is None
        and multistage_configuration.enabled
        and (
            terminal_method == TerminalValueMethod.PERPETUITY_GROWTH
            or multistage_configuration.stable_growth_rate is not None
        )
    )
    tax_assumption = resolved.assumption_set.find(
        ValuationAssumptionKind.NORMALIZED_TAX_RATE
    )
    if use_multistage:
        stable_growth_rate = (
            resolved.perpetual_growth_rate
            if terminal_method == TerminalValueMethod.PERPETUITY_GROWTH
            else multistage_configuration.stable_growth_rate
        )
        if stable_growth_rate is None:
            raise ValueError(
                "Adaptive multistage projection with an exit multiple requires "
                "valuation.multistage.stable_growth_rate in the profile"
            )
        forecast, multistage_plan = AdaptiveMultistageFcffForecastService(
            forecast_service
        ).forecast(
            financials,
            forecast,
            forecast_parameters,
            stable_growth_rate,
            multistage_configuration,
            normalized_tax_rate=(
                tax_assumption.value if tax_assumption is not None else None
            ),
            as_of=valuation_date,
        )
        multistage_plan = multistage_plan.model_copy(
            update={
                "terminal_roic_source": terminal_roic.source,
                "terminal_roic_methodology": terminal_roic.methodology,
                "terminal_roic_confidence": terminal_roic.confidence,
                "terminal_roic_warnings": terminal_roic.warnings,
            }
        )
    parameters = FcffDcfParameters(
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
            if terminal_method == TerminalValueMethod.EXIT_MULTIPLE
            else None
        ),
        exit_metric=(
            TerminalMetric(args.exit_metric)
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
        share_repurchase_parameters = ShareRepurchaseParameters(
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
    result = FcffDcfService().value(
        forecast,
        parameters,
        capital_bridge,
        resolved.assumption_set,
        multistage_plan,
        valuation_date,
        share_repurchase_parameters,
    )
    if terminal_roic.warnings:
        result = result.model_copy(
            update={
                "warnings": tuple(
                    dict.fromkeys([*result.warnings, *terminal_roic.warnings])
                )
            }
        )
    if selected_model in {"fcff-dcf", "both"}:
        print(FcffDcfConsolePresenter().render(result, profile_name=profile.name))
    if selected_model in {"comparables", "both"}:
        if terminal_method != TerminalValueMethod.PERPETUITY_GROWTH:
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
            print(
                "\nRelative valuation skipped: automatic peer evidence could not be "
                f"prepared ({comparable_error or 'unknown provider failure'})."
            )
        else:
            report = comparable_bundle.report
            comparable_financials = comparable_bundle.target_financials
            comparable_market = comparable_bundle.target_market
            comparable_peer_sources = comparable_bundle.peer_sources
            report = ForwardPeerMultiplesService().build(
                report,
                {
                    symbol: financials
                    for symbol, (
                        financials,
                        _market,
                    ) in comparable_peer_sources.items()
                },
                basis,
                valuation_date,
                horizon_years,
            )
            if selected_model == "both":
                print("\n" + "=" * 84 + "\n")
            print(ComparableMultiplesConsolePresenter().render(report))
            relative_ready = True
            if report.universe.discovery_confidence == "low":
                print(
                    "\nRelative valuation skipped: selected peer evidence has low "
                    "economic-comparability confidence."
                )
                relative_ready = False
            elif len(report.universe.selected_tickers) < (
                relative_configuration.multiple_resolution.minimum_peer_sample
            ):
                print(
                    "\nRelative valuation skipped: peer evidence is below the "
                    f"configured minimum sample of "
                    f"{relative_configuration.multiple_resolution.minimum_peer_sample}."
                )
                relative_ready = False
            if relative_ready:
                target_history = HistoricalMultiplesService().compute(
                    comparable_financials,
                    comparable_market,
                    basis,
                )
                peer_histories = tuple(
                    HistoricalMultiplesService().compute(financials, market, basis)
                    for financials, market in comparable_peer_sources.values()
                )
                resolved_multiple = MultipleResolver().resolve(
                    basis=basis,
                    target=report.target,
                    target_history=target_history,
                    peer_histories=peer_histories,
                    peer_report=report,
                    target_forecast=forecast,
                    intrinsic_valuation=result,
                    horizon_years=horizon_years,
                    policy=relative_configuration.multiple_resolution,
                )
                if report.warnings:
                    resolved_multiple = resolved_multiple.model_copy(
                        update={
                            "warnings": tuple(
                                dict.fromkeys(
                                    [*resolved_multiple.warnings, *report.warnings]
                                )
                            )
                        }
                    )
                relative_result = ComparableImpliedValuationService().value(
                    target_forecast=forecast,
                    capital_bridge=capital_bridge,
                    projected_shares=capital_bridge.diluted_shares,
                    resolved_multiple=resolved_multiple,
                    valuation_date=valuation_date,
                    horizon_years=horizon_years,
                    discount_rate=resolved.wacc,
                    current_price=report.target.price,
                    analyst_target_price=args.analyst_target_price,
                    intrinsic_value_per_share=result.value_per_share,
                )
                print("\n" + "=" * 84 + "\n")
                print(
                    ComparableImpliedValuationConsolePresenter().render(relative_result)
                )
    current_price = (
        relative_result.current_price if relative_result is not None else None
    )
    if current_price is None and comparable_bundle is not None:
        current_price = comparable_bundle.report.target.price
    market_data = automatic_inputs.get("market_data")
    if current_price is None and market_data is not None:
        latest_price = market_data.latest_price
        current_price = latest_price.close if latest_price is not None else None
    decision_configuration = profile.valuation.decision_analysis
    if current_price is not None and decision_configuration.enabled:
        decision_context = IntrinsicDecisionContext(
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
            normalized_tax_rate=(
                tax_assumption.value if tax_assumption is not None else None
            ),
            share_repurchase_parameters=share_repurchase_parameters,
            flexible_revenue_growth=(
                args.revenue_growth is None
                and profile.forecast.fcff.revenue_growth is None
            ),
            flexible_operating_margin=(
                args.operating_margin is None
                and profile.forecast.fcff.operating_margin is None
            ),
            flexible_terminal_roic=configured_terminal_roic is None,
            flexible_wacc=(args.wacc is None and discount_configuration.wacc is None),
            flexible_terminal_growth=(
                args.terminal_growth is None
                and terminal_configuration.perpetual_growth_rate is None
            ),
        )
        try:
            decision_policy = DecisionScenarioPolicy(
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
            decision_result = DecisionValuationService(decision_policy).build(
                decision_context,
                current_price,
                relative_result,
            )
        except ValueError as exc:
            print(f"\nDecision analysis unavailable: {exc}")
        else:
            print("\n" + "=" * 84 + "\n")
            print(
                DecisionValuationConsolePresenter().render(
                    decision_result,
                    show_scenarios=args.scenarios,
                    show_sensitivity=args.sensitivity,
                    show_reverse_dcf=args.reverse_dcf,
                )
            )
    elif current_price is None and decision_configuration.enabled:
        print("\nDecision analysis skipped: no current market price was available.")
    return 0


async def _run_valuation_models(args: argparse.Namespace) -> int:
    valuation_profile = _load_selected_valuation_profile(args)
    configuration = valuation_profile.model_selection
    financials = await _retrieve_financials(
        args,
        Granularity.ANNUAL,
        ValuationProfileBuilder.required_concepts(),
    )
    classification = await _retrieve_classification(
        args,
        provider=(
            ProviderName(args.classification_provider)
            if args.classification_provider
            else None
        ),
        crosscheck=False,
    )
    overrides = ValuationProfileOverrides(
        sector=configuration.sector,
        industry=configuration.industry,
        business_archetype=(
            BusinessArchetype(args.business_type)
            if args.business_type
            else configuration.business_archetype
        ),
        lifecycle=(
            CompanyLifecycle(args.lifecycle)
            if args.lifecycle
            else configuration.lifecycle
        ),
        cyclicality=(
            Cyclicality(args.cyclicality)
            if args.cyclicality
            else configuration.cyclicality
        ),
        economic_traits=(
            {EconomicTrait(value) for value in args.trait}
            if args.trait is not None
            else set(configuration.economic_traits)
        ),
        available_inputs=(
            {
                *valuation_profile.configured_valuation_inputs,
                *(ValuationInput(value) for value in args.available_input),
            }
            if args.available_input is not None
            else set(valuation_profile.configured_valuation_inputs)
        ),
        peer_count=(
            args.peer_count if args.peer_count is not None else configuration.peer_count
        ),
    )
    profile = ValuationProfileBuilder().build(financials, classification, overrides)
    selection = ValuationModelSelector().select(profile)
    print(ValuationSelectionConsolePresenter().render(selection))
    return 0


async def _run_classification(args: argparse.Namespace) -> int:
    classification = await _retrieve_classification(
        args,
        provider=ProviderName(args.provider) if args.provider else None,
        crosscheck=args.crosscheck,
    )
    print(ClassificationConsolePresenter().render(classification))
    return 0


async def _run_specialized_inputs(args: argparse.Namespace) -> int:
    configuration = _load_selected_valuation_profile(args).specialized_inputs
    history = args.history if args.history is not None else configuration.history
    if history < 1:
        raise ValueError("--history must be at least 1")
    if not args.user_agent:
        raise ValueError("SEC retrieval requires EDGARITO_USER_AGENT / user_agent")
    async with EdgarClient(
        FileSystemCache(Path(args.cache_dir)), args.user_agent
    ) as client:
        cik = args.cik
        if cik is None:
            cik = await client.get_cik(
                args.ticker,
                use_cache=not args.refresh,
                make_cache=True,
            )
        facts = await client.get_company_facts(
            cik,
            use_cache=not args.refresh,
            make_cache=True,
        )
    extraction = SpecializedValuationExtractor().extract(
        facts,
        SpecializedInputType(args.type),
        ticker=args.ticker,
        historical_periods=history,
    )
    print(SpecializedExtractionConsolePresenter().render(extraction))
    return 0


async def _retrieve_classification(
    args: argparse.Namespace,
    *,
    provider: Optional[ProviderName],
    crosscheck: bool,
):
    async with CompanyClassificationService(
        cache=FileSystemCache(Path(args.cache_dir)),
        provider_configuration=CLASSIFICATION_PROVIDER_CONFIGURATION,
        alphavantage_api_key=ALPHAVANTAGE_API_KEY,
        fmp_api_key=FMP_API_KEY,
        openfigi_api_key=OPENFIGI_API_KEY,
    ) as service:
        classification = await service.retrieve(
            ticker=args.ticker,
            cik=args.cik,
            isin=args.isin,
            exchange=args.exchange,
            exchange_symbols=_parse_mappings(args.exchange_symbol, "--exchange-symbol"),
            provider_symbols=_parse_provider_symbols(args.provider_symbol),
            provider=provider,
            use_cache=not args.refresh,
            make_cache=True,
            crosscheck=crosscheck,
        )
    return classification


async def _retrieve_financials(
    args: argparse.Namespace,
    granularity: Optional[Granularity],
    concepts: Optional[set[FinancialConcept]],
) -> NormalizedCompanyFinancials:
    cache = FileSystemCache(Path(args.cache_dir))
    async with FinancialDataService(
        cache=cache,
        provider_configuration=PROVIDER_CONFIGURATION,
        user_agent=args.user_agent,
        alphavantage_api_key=ALPHAVANTAGE_API_KEY,
        fmp_api_key=FMP_API_KEY,
        openfigi_api_key=OPENFIGI_API_KEY,
    ) as service:
        return await service.retrieve(
            ticker=args.ticker,
            cik=args.cik,
            isin=args.isin,
            exchange=args.exchange,
            exchange_symbols=_parse_mappings(args.exchange_symbol, "--exchange-symbol"),
            provider_symbols=_parse_provider_symbols(args.provider_symbol),
            market=Market(args.market),
            provider=ProviderName(args.provider) if args.provider else None,
            granularity=granularity,
            concepts=concepts,
            use_cache=not args.refresh,
            make_cache=True,
            crosscheck=args.crosscheck,
        )


def _granularity(period: str) -> Optional[Granularity]:
    return None if period == "all" else Granularity(period)


def _fcff_parameters(args: argparse.Namespace, configured) -> FcffForecastParameters:
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
        tax_rate=(args.tax_rate if args.tax_rate is not None else configured.tax_rate),
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
        historical_window=(
            args.historical_window
            if args.historical_window is not None
            else configured.historical_window
        ),
    )


async def _retrieve_automatic_assumption_inputs(
    args: argparse.Namespace,
    financials: NormalizedCompanyFinancials,
    currency: str,
    *,
    needs_wacc: bool,
    needs_terminal: bool,
    sector_override=None,
    industry_override: Optional[str] = None,
) -> dict:
    inputs = {
        "classification": None,
        "market_data": None,
        "risk_free_series": None,
        "inflation_series": None,
        "country_snapshot": None,
        "industry_snapshot": None,
        "company_beta": None,
    }
    if not (needs_wacc or needs_terminal):
        return inputs

    cache = FileSystemCache(Path(args.cache_dir))
    use_cache = not args.refresh
    symbol = financials.ticker or args.ticker
    if needs_wacc:
        if not symbol:
            raise ValueError(
                "Automatic WACC requires a ticker to retrieve Yahoo classification "
                "and market capitalization; provide --ticker or explicit WACC inputs"
            )
        try:
            async with YahooFinanceClient(cache) as yahoo:
                source, history = await asyncio.gather(
                    yahoo.get_company_financials(
                        symbol, use_cache=use_cache, make_cache=True
                    ),
                    yahoo.get_price_history(
                        symbol,
                        period="1mo",
                        use_cache=use_cache,
                        make_cache=True,
                    ),
                )
                classification = CompanyClassificationNormalizer().normalize_yahoo(
                    source
                )
                if (
                    classification.industry is None
                    or classification.country is None
                    or source.beta is None
                ):
                    source = await yahoo.get_company_financials(
                        symbol, use_cache=False, make_cache=True
                    )
                    classification = CompanyClassificationNormalizer().normalize_yahoo(
                        source
                    )
                classification = _apply_classification_overrides(
                    classification,
                    sector=sector_override,
                    industry=industry_override,
                )
            inputs["classification"] = classification
            inputs["company_beta"] = source.beta
            market_data = YahooMarketNormalizer().normalize(history)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "Automatic WACC could not retrieve Yahoo classification/price data; "
                "set WACC or the missing CAPM/capital-weight inputs in the profile. "
                f"Cause: {exc}"
            ) from exc
        if market_data.currency != currency.strip().upper():
            try:
                async with EcbClient(cache) as ecb:
                    market_data = await EcbMarketDataCurrencyConverter(ecb).convert(
                        market_data,
                        currency,
                        use_cache=use_cache,
                        make_cache=True,
                    )
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"Automatic WACC could not align the Yahoo quote currency "
                    f"({market_data.currency}) with the financial-statement currency "
                    f"({currency}); provide an explicit WACC or market-value equity. "
                    f"Cause: {exc}"
                ) from exc
        inputs["market_data"] = market_data

        try:
            async with DamodaranClient(cache) as damodaran:
                country_snapshot, industry_snapshot = await asyncio.gather(
                    damodaran.get_country_risk_premiums(
                        use_cache=use_cache, make_cache=True
                    ),
                    damodaran.get_industry_betas(use_cache=use_cache, make_cache=True),
                )
            inputs["country_snapshot"] = country_snapshot
            inputs["industry_snapshot"] = industry_snapshot
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "Automatic WACC could not retrieve the versioned Damodaran country "
                "and industry references; set beta, ERP, country premium, and tax "
                f"inputs in the profile. Cause: {exc}"
            ) from exc

    normalized_currency = currency.strip().upper()
    try:
        if normalized_currency == "EUR":
            start = datetime.date.today() - datetime.timedelta(days=365 * 6)
            async with EcbClient(cache) as ecb:
                risk_free_task = ecb.get_series(
                    "YC",
                    "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
                    kind=ReferenceSeriesKind.GOVERNMENT_YIELD,
                    unit=ReferenceValueUnit.PERCENTAGE_POINTS,
                    start_period=datetime.date.today() - datetime.timedelta(days=45),
                    end_period=datetime.date.today(),
                    use_cache=use_cache,
                    make_cache=True,
                )
                if needs_terminal:
                    risk_free, inflation = await asyncio.gather(
                        risk_free_task,
                        ecb.get_series(
                            "HICP",
                            "M.U2.N.000000.4D0.ANR",
                            kind=ReferenceSeriesKind.INFLATION_RATE,
                            unit=ReferenceValueUnit.PERCENT_CHANGE,
                            start_period=start,
                            end_period=datetime.date.today(),
                            use_cache=use_cache,
                            make_cache=True,
                        ),
                    )
                    inputs["inflation_series"] = inflation
                else:
                    risk_free = await risk_free_task
            inputs["risk_free_series"] = risk_free
        elif normalized_currency == "DKK":
            today = datetime.date.today()
            inflation_start = today - datetime.timedelta(days=365 * 6)
            async with EcbClient(cache) as ecb:
                risk_free_task = ecb.get_series(
                    "IRS",
                    "M.DK.L.L40.CI.0000.DKK.N.Z",
                    kind=ReferenceSeriesKind.GOVERNMENT_YIELD,
                    unit=ReferenceValueUnit.PERCENTAGE_POINTS,
                    start_period=today - datetime.timedelta(days=120),
                    end_period=today,
                    use_cache=use_cache,
                    make_cache=True,
                )
                if needs_terminal:
                    risk_free, inflation = await asyncio.gather(
                        risk_free_task,
                        ecb.get_series(
                            "HICP",
                            "M.DK.N.000000.4D0.ANR",
                            kind=ReferenceSeriesKind.INFLATION_RATE,
                            unit=ReferenceValueUnit.PERCENT_CHANGE,
                            start_period=inflation_start,
                            end_period=today,
                            use_cache=use_cache,
                            make_cache=True,
                        ),
                    )
                    inputs["inflation_series"] = inflation
                else:
                    risk_free = await risk_free_task
            inputs["risk_free_series"] = risk_free
        elif normalized_currency == "USD":
            async with TreasuryClient(cache) as treasury:
                inputs["risk_free_series"] = await treasury.get_par_yield(
                    120,
                    use_cache=use_cache,
                    make_cache=True,
                )
            if needs_terminal and FRED_API_KEY:
                async with FredClient(cache, FRED_API_KEY) as fred:
                    inputs["inflation_series"] = await fred.get_series(
                        "FPCPITOTLZGUSA",
                        kind=ReferenceSeriesKind.INFLATION_RATE,
                        unit=ReferenceValueUnit.PERCENT_CHANGE,
                        observation_start=datetime.date.today()
                        - datetime.timedelta(days=365 * 15),
                        observation_end=datetime.date.today(),
                        country="US",
                        use_cache=use_cache,
                        make_cache=True,
                    )
        else:
            raise ValueError(
                f"automatic macro assumptions currently support DKK, EUR, and USD, not "
                f"{normalized_currency}; set risk_free_rate/WACC and terminal growth "
                "in the profile"
            )
    except RuntimeError as exc:
        raise ValueError(
            "Automatic valuation assumptions could not retrieve the sovereign-yield "
            "or inflation series; provide risk_free_rate/WACC and terminal growth in "
            f"the profile. Cause: {exc}"
        ) from exc
    return inputs


def _apply_classification_overrides(
    classification,
    *,
    sector=None,
    industry: Optional[str] = None,
):
    """Apply explicit valuation-profile economics without erasing raw labels."""
    updates = {}
    if sector is not None:
        updates["sector"] = sector
        updates["sector_taxonomy"] = "valuation-profile"
    if industry is not None:
        updates["industry"] = industry
        updates["industry_taxonomy"] = "valuation-profile"
    return classification.model_copy(update=updates) if updates else classification


def _resolve_wacc(override: Optional[Decimal], configuration) -> tuple[Decimal, str]:
    if override is not None:
        return override, "explicit CLI override"
    if configuration.wacc is not None:
        return configuration.wacc, "explicit valuation profile"

    beta = configuration.levered_beta
    if beta is None and configuration.unlevered_beta is not None:
        required = {
            "market_value_debt": configuration.market_value_debt,
            "market_value_equity": configuration.market_value_equity,
            "normalized_tax_rate": configuration.normalized_tax_rate,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Levering the profile beta requires: " + ", ".join(missing)
            )
        assert configuration.market_value_debt is not None
        assert configuration.market_value_equity is not None
        assert configuration.normalized_tax_rate is not None
        beta = DiscountRateService.lever_beta(
            configuration.unlevered_beta,
            configuration.market_value_debt,
            configuration.market_value_equity,
            configuration.normalized_tax_rate,
        )

    cost_of_equity = configuration.cost_of_equity
    if cost_of_equity is None:
        capm_inputs = {
            "risk_free_rate": configuration.risk_free_rate,
            "levered_beta": beta,
            "equity_risk_premium": configuration.equity_risk_premium,
        }
        missing = [name for name, value in capm_inputs.items() if value is None]
        if missing:
            raise ValueError(
                "FCFF DCF requires WACC. Provide --wacc, set valuation.discount_rates.wacc, "
                "or complete the profile CAPM/WACC inputs. Missing: "
                + ", ".join(missing)
            )
        assert configuration.risk_free_rate is not None
        assert beta is not None
        assert configuration.equity_risk_premium is not None
        cost_of_equity = DiscountRateService.cost_of_equity(
            configuration.risk_free_rate,
            beta,
            configuration.equity_risk_premium,
            configuration.country_risk_premium or Decimal(0),
        ).cost_of_equity

    wacc_inputs = {
        "pretax_cost_of_debt": configuration.pretax_cost_of_debt,
        "normalized_tax_rate": configuration.normalized_tax_rate,
        "market_value_equity": configuration.market_value_equity,
        "market_value_debt": configuration.market_value_debt,
    }
    missing = [name for name, value in wacc_inputs.items() if value is None]
    if missing:
        raise ValueError(
            "FCFF DCF WACC calculation is missing profile inputs: " + ", ".join(missing)
        )
    assert configuration.pretax_cost_of_debt is not None
    assert configuration.normalized_tax_rate is not None
    assert configuration.market_value_equity is not None
    assert configuration.market_value_debt is not None
    result = DiscountRateService.wacc(
        cost_of_equity,
        configuration.pretax_cost_of_debt,
        configuration.normalized_tax_rate,
        configuration.market_value_equity,
        configuration.market_value_debt,
    )
    return result.wacc, "derived from valuation profile CAPM and capital weights"


def _validate_limit(limit: int) -> None:
    if limit < 1:
        raise ValueError("--limit must be at least 1")


def _parse_mappings(values: Optional[list[str]], option: str) -> dict[str, str]:
    mappings = {}
    for value in values or []:
        key, separator, mapped_value = value.partition("=")
        key = key.strip()
        mapped_value = mapped_value.strip()
        if not separator or not key or not mapped_value:
            raise ValueError(f"{option} must use NAME=SYMBOL syntax")
        normalized_key = key.lower()
        if normalized_key in mappings:
            raise ValueError(f"Duplicate {option} mapping for {key}")
        mappings[normalized_key] = mapped_value
    return mappings


def _parse_provider_symbols(values: Optional[list[str]]) -> dict[ProviderName, str]:
    raw_mappings = _parse_mappings(values, "--provider-symbol")
    try:
        return {
            ProviderName(provider): symbol for provider, symbol in raw_mappings.items()
        }
    except ValueError as exc:
        choices = ", ".join(provider.value for provider in ProviderName)
        raise ValueError(
            f"Unknown provider in --provider-symbol; choose one of: {choices}"
        ) from exc


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logger(
        logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING
    )
    try:
        if args.command == "financials":
            return asyncio.run(_run_financials(args))
        if args.command == "metrics":
            return asyncio.run(_run_metrics(args))
        if args.command == "forecast":
            return asyncio.run(_run_forecast(args))
        if args.command == "valuation":
            return asyncio.run(_run_valuation(args))
        if args.command == "valuation-models":
            return asyncio.run(_run_valuation_models(args))
        if args.command == "classification":
            return asyncio.run(_run_classification(args))
        if args.command == "comparables":
            return asyncio.run(_run_comparables(args))
        if args.command == "specialized-inputs":
            return asyncio.run(_run_specialized_inputs(args))
    except (ValueError, RuntimeError, FileNotFoundError, ValidationError) as exc:
        parser.error(str(exc))
    return 1
