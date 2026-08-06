import argparse
import asyncio
import datetime
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from edgarito.cli.presentation.console import (
    ClassificationConsolePresenter,
    ComparableMultiplesConsolePresenter,
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
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.valuation.specialized import SpecializedInputType
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.forecasting import (
    FcffForecastParameters,
    FcffForecastService,
    SimplifiedFcfForecastParameters,
    SimplifiedFcfForecastService,
)
from edgarito.services.metrics import FinancialMetric, FinancialMetricsService
from edgarito.services.normalization.classification import (
    CompanyClassificationNormalizer,
)
from edgarito.services.normalization.yahoo import YahooFinancialsNormalizer
from edgarito.services.normalization.yahoo_market import YahooMarketNormalizer
from edgarito.services.providers.edgar import EdgarClient
from edgarito.services.providers.yahoo import YahooFinanceClient
from edgarito.services.reconciliation.classification import (
    CompanyClassificationService,
)
from edgarito.services.reconciliation.financials import FinancialDataService
from edgarito.services.valuation import (
    BusinessArchetype,
    CompanyLifecycle,
    ComparableMultiplesService,
    Cyclicality,
    EconomicTrait,
    LtmMultiplesService,
    PeerSelectionParameters,
    PeerUniverseSelector,
    SpecializedValuationExtractor,
    ValuationInput,
    ValuationModelSelector,
    ValuationProfileBuilder,
    ValuationProfileOverrides,
)
from edgarito.settings import (
    ALPHAVANTAGE_API_KEY,
    CLASSIFICATION_PROVIDER_CONFIGURATION,
    EDGARITO_CACHE_DIR,
    EDGARITO_USER_AGENT,
    FMP_API_KEY,
    OPENFIGI_API_KEY,
    PROVIDER_CONFIGURATION,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edgarito", description="Retrieve and analyze normalized financials"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    financials = subparsers.add_parser(
        "financials", help="Display normalized historical financials"
    )
    metrics = subparsers.add_parser(
        "metrics", help="Calculate metrics from normalized financials"
    )
    forecast = subparsers.add_parser(
        "forecast", help="Project annual driver-based FCFF"
    )
    valuation_models = subparsers.add_parser(
        "valuation-models",
        help="Rank suitable valuation models and report missing inputs",
    )
    classification = subparsers.add_parser(
        "classification", help="Retrieve normalized company sector and industry"
    )
    comparables = subparsers.add_parser(
        "comparables",
        help="Select peers and compute keyless Yahoo-backed LTM multiples",
    )
    specialized_inputs = subparsers.add_parser(
        "specialized-inputs",
        help="Extract REIT, resource, biotech, or SOTP valuation inputs",
    )

    for command_parser in (financials, metrics):
        _add_retrieval_arguments(command_parser)
    _add_retrieval_arguments(forecast, include_period=False)
    _add_retrieval_arguments(valuation_models, include_period=False)
    for command_parser in (
        forecast,
        valuation_models,
        comparables,
        specialized_inputs,
    ):
        _add_valuation_profile_argument(command_parser)

    financials.add_argument(
        "--concept",
        action="append",
        choices=[concept.value for concept in FinancialConcept],
        help="Limit output to a concept; repeat this option for multiple concepts",
    )
    metrics.add_argument(
        "--metric",
        action="append",
        choices=[metric.value for metric in FinancialMetric],
        help="Limit output to a metric; repeat this option for multiple metrics",
    )
    forecast.add_argument(
        "--forecast-method",
        "--method",
        choices=("fcff", "simplified"),
        help="Forecast method; overrides the selected profile",
    )
    forecast.add_argument(
        "--years",
        type=int,
        help="Number of annual forecast periods; overrides the selected profile",
    )
    forecast.add_argument(
        "--revenue-growth",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help=(
            "Revenue growth in percentage points; provide once for a constant "
            "rate or once per forecast year"
        ),
    )
    forecast.add_argument(
        "--operating-margin",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="EBIT margin; provide once or once per forecast year",
    )
    forecast.add_argument(
        "--tax-rate",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="Normalized operating tax rate; provide once or once per forecast year",
    )
    forecast.add_argument(
        "--depreciation-to-revenue",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="D&A as a percentage of revenue; provide once or per forecast year",
    )
    forecast.add_argument(
        "--capex-to-revenue",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="Capex as a percentage of revenue; provide once or per forecast year",
    )
    forecast.add_argument(
        "--operating-working-capital-to-revenue",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help=(
            "Operating working capital as a percentage of revenue; provide once "
            "or per forecast year"
        ),
    )
    forecast.add_argument(
        "--fcf-margin",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help=(
            "FCF margin for --forecast-method simplified; provide once or once "
            "per forecast year"
        ),
    )
    forecast.add_argument(
        "--historical-window",
        type=int,
        help="Annual periods used to infer omitted assumptions; overrides the profile",
    )
    valuation_models.add_argument(
        "--classification-provider",
        choices=[
            provider.value
            for provider in CLASSIFICATION_PROVIDER_CONFIGURATION.available_providers
        ],
        help="Override the configured classification provider",
    )
    valuation_models.add_argument(
        "--business-type",
        choices=[item.value for item in BusinessArchetype],
        help="Override the inferred economic business type",
    )
    valuation_models.add_argument(
        "--lifecycle",
        choices=[item.value for item in CompanyLifecycle],
        help="Override the inferred company lifecycle",
    )
    valuation_models.add_argument(
        "--cyclicality",
        choices=[item.value for item in Cyclicality],
        help="Override inferred cyclicality",
    )
    valuation_models.add_argument(
        "--trait",
        action="append",
        choices=[item.value for item in EconomicTrait],
        help="Add a known economic trait; repeat for multiple traits",
    )
    valuation_models.add_argument(
        "--available-input",
        action="append",
        choices=[item.value for item in ValuationInput],
        help="Mark external valuation data as available; repeat as needed",
    )
    valuation_models.add_argument(
        "--peer-count",
        type=int,
        help="Number of genuinely comparable companies available",
    )
    _add_identifier_arguments(classification)
    classification.add_argument(
        "--provider",
        choices=[
            provider.value
            for provider in CLASSIFICATION_PROVIDER_CONFIGURATION.available_providers
        ],
        help="Override the configured classification provider",
    )
    classification.add_argument("--refresh", action="store_true")
    classification.add_argument("--crosscheck", action="store_true")
    classification.add_argument("--cache-dir", default=EDGARITO_CACHE_DIR)
    classification.add_argument("--verbose", action="store_true")
    comparables.add_argument("--ticker", required=True, help="Target Yahoo symbol")
    comparables.add_argument(
        "--peer",
        action="append",
        required=True,
        help="Candidate Yahoo symbol; repeat to supply the candidate universe",
    )
    comparables.add_argument("--max-peers", type=int, help="Maximum selected peers")
    comparables.add_argument(
        "--preferred-minimum",
        type=int,
        help="Preferred minimum selected peers",
    )
    comparables.add_argument(
        "--minimum-score",
        type=int,
        help="Minimum comparability score from 0 to 100",
    )
    sector_requirement = comparables.add_mutually_exclusive_group()
    sector_requirement.add_argument(
        "--allow-cross-sector",
        dest="require_same_sector",
        action="store_false",
        help="Do not hard-exclude candidates from a different sector",
    )
    sector_requirement.add_argument(
        "--require-same-sector",
        dest="require_same_sector",
        action="store_true",
        help="Hard-exclude candidates from a different sector",
    )
    comparables.set_defaults(require_same_sector=None)
    comparables.add_argument(
        "--as-of",
        type=datetime.date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Use the latest price on or before this date",
    )
    comparables.add_argument("--refresh", action="store_true")
    comparables.add_argument("--cache-dir", default=EDGARITO_CACHE_DIR)
    comparables.add_argument("--verbose", action="store_true")
    specialized_identifier = specialized_inputs.add_mutually_exclusive_group(
        required=True
    )
    specialized_identifier.add_argument("--ticker", help="US-listed SEC ticker")
    specialized_identifier.add_argument("--cik", type=int, help="SEC Central Index Key")
    specialized_inputs.add_argument(
        "--type",
        required=True,
        choices=[item.value for item in SpecializedInputType],
        help="Specialized valuation input profile",
    )
    specialized_inputs.add_argument(
        "--history",
        type=int,
        help="Number of latest reporting period ends; overrides the profile",
    )
    specialized_inputs.add_argument("--refresh", action="store_true")
    specialized_inputs.add_argument("--cache-dir", default=EDGARITO_CACHE_DIR)
    specialized_inputs.add_argument(
        "--user-agent",
        default=EDGARITO_USER_AGENT,
        help="SEC user agent in 'Name (email@example.com)' form",
    )
    specialized_inputs.add_argument("--verbose", action="store_true")
    return parser


def _add_retrieval_arguments(
    command_parser: argparse.ArgumentParser, *, include_period: bool = True
) -> None:
    _add_identifier_arguments(command_parser)

    command_parser.add_argument(
        "--market",
        choices=[market.value for market in Market],
        default=Market.US.value,
        help="Stock market configuration to use (default: us)",
    )
    command_parser.add_argument(
        "--provider",
        choices=[provider.value for provider in ProviderName],
        help="Override the configured default provider",
    )

    if include_period:
        command_parser.add_argument(
            "--period",
            choices=("annual", "quarterly", "all"),
            default="annual",
            help="Period granularity to display (default: annual)",
        )
        command_parser.add_argument(
            "--limit", type=int, default=5, help="Number of latest periods to display"
        )
    command_parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached provider snapshots"
    )
    command_parser.add_argument(
        "--crosscheck",
        action="store_true",
        help="Compare with the other configured providers and emit warnings",
    )
    command_parser.add_argument(
        "--cache-dir",
        default=EDGARITO_CACHE_DIR,
        help="Snapshot cache directory (default: cache)",
    )
    command_parser.add_argument(
        "--user-agent",
        default=EDGARITO_USER_AGENT,
        help=(
            "SEC user agent in 'Name (email@example.com)' form; "
            "or configure user_agent in .env"
        ),
    )
    command_parser.add_argument("--verbose", action="store_true")


def _add_valuation_profile_argument(
    command_parser: argparse.ArgumentParser,
) -> None:
    command_parser.add_argument(
        "--profile",
        type=Path,
        metavar="PATH",
        help=(
            "Forecast/valuation JSON profile; defaults to "
            "configs/valuation/default.json"
        ),
    )


def _add_identifier_arguments(command_parser: argparse.ArgumentParser) -> None:
    identifier = command_parser.add_mutually_exclusive_group(required=True)
    identifier.add_argument("--ticker", help="Stock ticker, for example AAPL")
    identifier.add_argument("--cik", type=int, help="SEC Central Index Key")
    identifier.add_argument(
        "--isin", help="12-character ISIN, for example US0378331005"
    )
    command_parser.add_argument(
        "--exchange",
        help="Exchange used to disambiguate a ticker or identifier, for example XETRA",
    )
    command_parser.add_argument(
        "--exchange-symbol",
        action="append",
        metavar="EXCHANGE=SYMBOL",
        help="Map an exchange to its symbol; repeat for multiple exchanges",
    )
    command_parser.add_argument(
        "--provider-symbol",
        action="append",
        metavar="PROVIDER=SYMBOL",
        help="Map a provider to its symbol; repeat for multiple providers",
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


async def _run_forecast(args: argparse.Namespace) -> int:
    profile = ValuationProfileLoader.load(args.profile)
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
        parameters = FcffForecastParameters(
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
            tax_rate=(
                args.tax_rate if args.tax_rate is not None else configured.tax_rate
            ),
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
        service = FcffForecastService()
    financials = await _retrieve_financials(
        args,
        Granularity.ANNUAL,
        service.required_concepts(),
    )
    forecast = service.forecast(financials, parameters)
    print(ForecastConsolePresenter().render(forecast))
    return 0


async def _run_valuation_models(args: argparse.Namespace) -> int:
    valuation_profile = ValuationProfileLoader.load(args.profile)
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


async def _run_comparables(args: argparse.Namespace) -> int:
    configuration = ValuationProfileLoader.load(args.profile).comparables
    target_symbol = args.ticker.strip().upper()
    peer_symbols = list(dict.fromkeys(symbol.strip().upper() for symbol in args.peer))
    parameters = PeerSelectionParameters(
        max_peers=(
            args.max_peers if args.max_peers is not None else configuration.max_peers
        ),
        preferred_minimum=(
            args.preferred_minimum
            if args.preferred_minimum is not None
            else configuration.preferred_minimum
        ),
        minimum_score=(
            args.minimum_score
            if args.minimum_score is not None
            else configuration.minimum_score
        ),
        require_same_sector=(
            args.require_same_sector
            if args.require_same_sector is not None
            else configuration.require_same_sector
        ),
    )
    symbols = [target_symbol, *peer_symbols]
    cache = FileSystemCache(Path(args.cache_dir))
    async with YahooFinanceClient(cache) as client:
        results = await asyncio.gather(
            *(
                _retrieve_yahoo_comparable_source(
                    client,
                    symbol,
                    args.as_of,
                    use_cache=not args.refresh,
                )
                for symbol in symbols
            ),
            return_exceptions=True,
        )

    if isinstance(results[0], BaseException):
        raise RuntimeError(f"Target retrieval failed for {target_symbol}: {results[0]}")

    profile_builder = ValuationProfileBuilder()
    classification_normalizer = CompanyClassificationNormalizer()
    financials_normalizer = YahooFinancialsNormalizer()
    market_normalizer = YahooMarketNormalizer()
    multiples_service = LtmMultiplesService()
    bundles = {}
    retrieval_warnings = []
    for symbol, result in zip(symbols, results, strict=True):
        if isinstance(result, BaseException):
            retrieval_warnings.append(f"{symbol} retrieval failed: {result}")
            continue
        source, history = result
        financials = financials_normalizer.normalize(source)
        classification = classification_normalizer.normalize_yahoo(source)
        profile = profile_builder.build(financials, classification)
        market_data = market_normalizer.normalize(history)
        try:
            multiples = multiples_service.compute(financials, market_data, args.as_of)
        except ValueError as exc:
            multiples = None
            retrieval_warnings.append(f"{symbol} multiples unavailable: {exc}")
        bundles[symbol] = (profile, multiples)

    target_profile, target_multiples = bundles[target_symbol]
    if target_multiples is None:
        raise ValueError(f"LTM multiples could not be computed for {target_symbol}")
    candidate_profiles = [
        bundles[symbol][0] for symbol in peer_symbols if symbol in bundles
    ]
    universe = PeerUniverseSelector().select(
        target_profile, candidate_profiles, parameters
    )
    peer_multiples = [
        bundles[symbol][1]
        for symbol in peer_symbols
        if symbol in bundles and bundles[symbol][1] is not None
    ]
    report = ComparableMultiplesService().build(
        universe,
        target_multiples,
        peer_multiples,
    )
    if retrieval_warnings:
        report = report.model_copy(
            update={"warnings": [*report.warnings, *retrieval_warnings]}
        )
    print(ComparableMultiplesConsolePresenter().render(report))
    return 0


async def _run_specialized_inputs(args: argparse.Namespace) -> int:
    configuration = ValuationProfileLoader.load(args.profile).specialized_inputs
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


async def _retrieve_yahoo_comparable_source(
    client: YahooFinanceClient,
    symbol: str,
    as_of: Optional[datetime.date],
    *,
    use_cache: bool,
):
    history_arguments = {"period": "1mo"}
    if as_of is not None:
        history_arguments = {
            "start": as_of - datetime.timedelta(days=14),
            "end": as_of + datetime.timedelta(days=1),
        }
    return await asyncio.gather(
        client.get_company_financials(symbol, use_cache=use_cache, make_cache=True),
        client.get_price_history(
            symbol,
            **history_arguments,
            use_cache=use_cache,
            make_cache=True,
        ),
    )


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


def _percentage(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid percentage: {value!r}") from exc


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
