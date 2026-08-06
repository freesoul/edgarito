import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from edgarito.cli.presentation.console import FinancialsConsolePresenter
from edgarito.enums.granularity import Granularity
from edgarito.enums.market import Market
from edgarito.enums.provider import ProviderName
from edgarito.logger import configure_logger
from edgarito.schemas.normalization.financials import FinancialConcept
from edgarito.services.cache.filesystem_cache import FileSystemCache
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
        prog="edgarito", description="Retrieve normalized company financials"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    financials = subparsers.add_parser(
        "financials", help="Display normalized historical financials"
    )

    identifier = financials.add_mutually_exclusive_group(required=True)
    identifier.add_argument("--ticker", help="Stock ticker, for example AAPL")
    identifier.add_argument("--cik", type=int, help="SEC Central Index Key")

    financials.add_argument(
        "--market",
        choices=[market.value for market in Market],
        default=Market.US.value,
        help="Stock market configuration to use (default: us)",
    )
    financials.add_argument(
        "--provider",
        choices=[provider.value for provider in ProviderName],
        help="Override the configured default provider",
    )

    financials.add_argument(
        "--period",
        choices=("annual", "quarterly", "all"),
        default="annual",
        help="Period granularity to display (default: annual)",
    )
    financials.add_argument(
        "--concept",
        action="append",
        choices=[concept.value for concept in FinancialConcept],
        help="Limit output to a concept; repeat this option for multiple concepts",
    )
    financials.add_argument(
        "--limit", type=int, default=5, help="Number of latest periods to display"
    )
    financials.add_argument(
        "--refresh", action="store_true", help="Ignore cached provider snapshots"
    )
    financials.add_argument(
        "--crosscheck",
        action="store_true",
        help="Compare with the other configured providers and emit warnings",
    )
    financials.add_argument(
        "--cache-dir",
        default=EDGARITO_CACHE_DIR,
        help="Snapshot cache directory (default: cache)",
    )
    financials.add_argument(
        "--user-agent",
        default=EDGARITO_USER_AGENT,
        help="SEC user agent in 'Name email@example.com' form; or configure it in the environment/dotenv",
    )
    financials.add_argument("--verbose", action="store_true")
    return parser


async def _run_financials(args: argparse.Namespace) -> int:
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    granularity: Optional[Granularity]
    if args.period == "all":
        granularity = None
    else:
        granularity = Granularity(args.period)

    concepts = (
        {FinancialConcept(value) for value in args.concept} if args.concept else None
    )
    cache = FileSystemCache(Path(args.cache_dir))
    async with FinancialDataService(
        cache=cache,
        provider_configuration=PROVIDER_CONFIGURATION,
        user_agent=args.user_agent,
        alphavantage_api_key=ALPHAVANTAGE_API_KEY,
        fmp_api_key=FMP_API_KEY,
    ) as service:
        financials = await service.retrieve(
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

    print(FinancialsConsolePresenter().render(financials, limit=args.limit))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logger(
        logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING
    )
    try:
        if args.command == "financials":
            return asyncio.run(_run_financials(args))
    except (ValueError, RuntimeError, FileNotFoundError, ValidationError) as exc:
        parser.error(str(exc))
    return 1
