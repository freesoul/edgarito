import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from edgarito.cli.presentation.console import (
    FinancialsConsolePresenter,
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
from edgarito.services.metrics import FinancialMetric, FinancialMetricsService
from edgarito.services.reconciliation.financials import FinancialDataService
from edgarito.settings import (
    ALPHAVANTAGE_API_KEY,
    EDGARITO_CACHE_DIR,
    EDGARITO_USER_AGENT,
    FMP_API_KEY,
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

    for command_parser in (financials, metrics):
        _add_retrieval_arguments(command_parser)

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
    return parser


def _add_retrieval_arguments(command_parser: argparse.ArgumentParser) -> None:
    identifier = command_parser.add_mutually_exclusive_group(required=True)
    identifier.add_argument("--ticker", help="Stock ticker, for example AAPL")
    identifier.add_argument("--cik", type=int, help="SEC Central Index Key")

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
    ) as service:
        return await service.retrieve(
            ticker=args.ticker,
            cik=args.cik,
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
    except (ValueError, RuntimeError, FileNotFoundError, ValidationError) as exc:
        parser.error(str(exc))
    return 1
