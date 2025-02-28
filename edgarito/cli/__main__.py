import logging
import aiohttp
import os
from typing import List, Optional
import asyncio

import typer

from edgarito.cli.logger import configure_logger
from edgarito.cli.settings import settings

from edgarito.services.edgar_rest_client.low_level_client import EDGARLowLevelClient
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.edgar_rest_client.submissions_client import SubmissionsClient
from edgarito.services.downloader.download_service import DownloadService

from edgarito.schemas.edgar_responses.company_ticker import CompanyTickerResponse
from edgarito.schemas.edgar_responses.submission import TransposedFiling

from edgarito.enums.edgar.filing_type import FilingType


class Cli:

    def __init__(self):
        self._logger = logging.getLogger(__class__.__name__)
        self._cache = FileSystemCache(root_directory=settings.cache_path)

    async def cik_from_ticker(self, ticker: str, use_cache: bool = True, make_cache: bool = False) -> int:
        async with EDGARLowLevelClient(cache=self._cache, user_agent=settings.user_agent) as client:
            tickers = await client.get_tickers(use_cache=use_cache, make_cache=make_cache)
            if ticker:
                for ticker_it in tickers:
                    if ticker.lower() == ticker_it.ticker.lower():
                        resolved_cik = ticker_it.cik_str  # that's the name from the response, but it's an int
                        break
                else:
                    raise ValueError(f"Ticker {ticker} not found")

        self._logger.info(f"CIK: {resolved_cik}")
        return resolved_cik

    async def find_all_ciks(self, use_cache: bool = True, make_cache: bool = False) -> List[CompanyTickerResponse]:
        async with EDGARLowLevelClient(cache=self._cache, user_agent=settings.user_agent) as client:
            tickers = await client.get_tickers(use_cache=use_cache, make_cache=make_cache)
            tickers = sorted(tickers, key=lambda x: x.title)
            for ticker in tickers:
                self._logger.info(f"{ticker.cik_str}\t{ticker.ticker}\t\t{ticker.title}")
            return tickers

    async def find_ticker_from_cik(self, cik: int, use_cache: bool = True, make_cache: bool = False) -> str:
        async with EDGARLowLevelClient(cache=self._cache, user_agent=settings.user_agent) as client:
            tickers = await client.get_tickers(use_cache=use_cache, make_cache=make_cache)
            for ticker in tickers:
                if ticker.cik_str == cik:
                    # self._logger.info(f"{ticker.ticker}")
                    self._logger.info(f"{ticker.ticker} has cik {ticker.cik_str} and title {ticker.title}")
                    return ticker.ticker
            raise ValueError(f"CIK {cik} not found")

    async def find_submissions_from_ticker(
        self, ticker: str, type: Optional[FilingType] = None, use_cache: bool = True, make_cache: bool = False
    ) -> List[TransposedFiling]:
        resolved_cik = await self.cik_from_ticker(ticker, use_cache=use_cache, make_cache=make_cache)
        return await self.find_submissions_from_cik(resolved_cik, type=type, use_cache=use_cache, make_cache=make_cache)

    async def find_submissions_from_cik(
        self, cik: int, type: Optional[FilingType] = None, use_cache: bool = True, make_cache: bool = False, limit: Optional[int] = None
    ) -> List[TransposedFiling]:
        async with EDGARLowLevelClient(cache=self._cache, user_agent=settings.user_agent) as client:
            submissions_client = SubmissionsClient(client)
            submissions = await submissions_client.get_all_submission_filings_transposed(cik, filing_type=type, use_cache=use_cache, make_cache=make_cache)

            if limit:
                submissions = submissions[-limit:]  # Return the last N submissions
            for submission in submissions:
                self._logger.info(f"{submission.filingDate}\t\t{submission.accessionNumber}\t{submission.form} ({submission.core_type})")
            return submissions


if __name__ == "__main__":

    configure_logger(settings.log_level)

    logging.debug(f"Using log level {settings.log_level}")

    app = typer.Typer()

    context_settings = {
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }

    @app.command(context_settings=context_settings)
    def cik(
        ticker: str = typer.Option(..., help="Ticker to resolve CIK"),
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
    ):
        cli = Cli()
        asyncio.run(cli.cik_from_ticker(ticker, use_cache, make_cache))

    @app.command(context_settings=context_settings)
    def tickers(
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
    ):
        cli = Cli()
        asyncio.run(cli.find_all_ciks(use_cache, make_cache))

    @app.command(context_settings=context_settings)
    def ticker(
        cik: int = typer.Option(..., help="CIK to resolve ticker"),
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
    ):
        cli = Cli()
        asyncio.run(cli.find_ticker_from_cik(cik, use_cache, make_cache))

    @app.command(context_settings=context_settings)
    def submissions(
        ticker: str = typer.Option(None, help="Ticker to resolve CIK"),
        cik: int = typer.Option(None, help="CIK to resolve ticker"),
        type: FilingType = typer.Option(None, help="Type of filing"),
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
        limit: int = typer.Option(None, help="Limit the number of submissions"),
    ):
        cli = Cli()
        if not ticker and not cik:
            raise typer.Abort("Provide a valid ticker with --ticker or a CIK with --cik")
        if ticker:
            asyncio.run(cli.find_submissions_from_ticker(ticker, type, use_cache, make_cache))
        elif cik:
            asyncio.run(cli.find_submissions_from_cik(cik, type, use_cache, make_cache, limit))

    app()
