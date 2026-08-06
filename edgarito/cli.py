import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from edgarito.enums.granularity import Granularity
from edgarito.logger import configure_logger
from edgarito.schemas.normalization.financials import FinancialConcept
from edgarito.schemas.use_cases.retrieve_financials import RetrieveFinancialsRequest
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization.sec_us_gaap import SecUsGaapNormalizer
from edgarito.services.presentation.console import FinancialsConsolePresenter
from edgarito.services.providers.edgar import EdgarClient
from edgarito.services.use_cases.retrieve_financials import RetrieveFinancials


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgarito", description="Retrieve normalized company financials")
    subparsers = parser.add_subparsers(dest="command", required=True)
    financials = subparsers.add_parser("financials", help="Display normalized SEC historicals")

    identifier = financials.add_mutually_exclusive_group(required=True)
    identifier.add_argument("--ticker", help="US stock ticker, for example AAPL")
    identifier.add_argument("--cik", type=int, help="SEC Central Index Key")

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
    financials.add_argument("--limit", type=int, default=5, help="Number of latest periods to display")
    financials.add_argument("--refresh", action="store_true", help="Ignore cached provider snapshots")
    financials.add_argument(
        "--cache-dir",
        default=os.getenv("EDGARITO_CACHE_DIR", "cache"),
        help="Snapshot cache directory (default: cache)",
    )
    financials.add_argument(
        "--user-agent",
        default=os.getenv("EDGARITO_USER_AGENT"),
        help="SEC user agent in 'Name email@example.com' form; or set EDGARITO_USER_AGENT",
    )
    financials.add_argument("--verbose", action="store_true")
    return parser


async def _run_financials(args: argparse.Namespace) -> int:
    if not args.user_agent:
        raise ValueError("Provide --user-agent or set EDGARITO_USER_AGENT as required by the SEC")
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    granularity: Optional[Granularity]
    if args.period == "all":
        granularity = None
    else:
        granularity = Granularity(args.period)

    concepts = {FinancialConcept(value) for value in args.concept} if args.concept else None
    request = RetrieveFinancialsRequest(
        ticker=args.ticker,
        cik=args.cik,
        granularity=granularity,
        concepts=concepts,
        use_cache=not args.refresh,
        make_cache=True,
    )

    cache = FileSystemCache(Path(args.cache_dir))
    async with EdgarClient(cache=cache, user_agent=args.user_agent) as edgar:
        use_case = RetrieveFinancials(edgar, SecUsGaapNormalizer())
        financials = await use_case.execute(request)

    print(FinancialsConsolePresenter().render(financials, limit=args.limit))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logger(logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING)
    try:
        if args.command == "financials":
            return asyncio.run(_run_financials(args))
    except (ValueError, RuntimeError, FileNotFoundError, ValidationError) as exc:
        parser.error(str(exc))
    return 1
