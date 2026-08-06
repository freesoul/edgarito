import argparse
import asyncio
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from edgarito.cli.presentation.console import (
    ClassificationConsolePresenter,
    FinancialsConsolePresenter,
    ForecastConsolePresenter,
    MetricsConsolePresenter,
)
from edgarito.enums.granularity import Granularity
from edgarito.enums.market import Market
from edgarito.enums.provider import ProviderName
from edgarito.logger import configure_logger
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.forecasting import (
    FreeCashFlowForecastParameters,
    FreeCashFlowForecastService,
)
from edgarito.services.metrics import FinancialMetric, FinancialMetricsService
from edgarito.services.reconciliation.classification import (
    CompanyClassificationService,
)
from edgarito.services.reconciliation.financials import FinancialDataService
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
        "forecast", help="Project annual free cash flow from explicit assumptions"
    )
    classification = subparsers.add_parser(
        "classification", help="Retrieve normalized company sector and industry"
    )

    for command_parser in (financials, metrics):
        _add_retrieval_arguments(command_parser)
    _add_retrieval_arguments(forecast, include_period=False)

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
        "--years",
        type=int,
        default=5,
        help="Number of annual forecast periods (default: 5)",
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
        "--fcf-margin",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help=(
            "FCF margin in percentage points; provide once for a constant margin "
            "or once per forecast year"
        ),
    )
    forecast.add_argument(
        "--historical-window",
        type=int,
        default=3,
        help="Annual periods used to infer omitted assumptions (default: 3)",
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
    parameters = FreeCashFlowForecastParameters(
        forecast_years=args.years,
        revenue_growth=args.revenue_growth,
        free_cash_flow_margin=args.fcf_margin,
        historical_window=args.historical_window,
    )
    financials = await _retrieve_financials(
        args,
        Granularity.ANNUAL,
        FreeCashFlowForecastService.required_concepts(),
    )
    forecast = FreeCashFlowForecastService().forecast(financials, parameters)
    print(ForecastConsolePresenter().render(forecast))
    return 0


async def _run_classification(args: argparse.Namespace) -> int:
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
            provider=ProviderName(args.provider) if args.provider else None,
            use_cache=not args.refresh,
            make_cache=True,
            crosscheck=args.crosscheck,
        )
    print(ClassificationConsolePresenter().render(classification))
    return 0


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
        if args.command == "classification":
            return asyncio.run(_run_classification(args))
    except (ValueError, RuntimeError, FileNotFoundError, ValidationError) as exc:
        parser.error(str(exc))
    return 1
