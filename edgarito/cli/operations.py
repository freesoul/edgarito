"""Core CLI operations and business logic."""

import pathlib
import logging
import aiohttp
from typing import List, Optional

from edgarito.cli.settings import settings
from edgarito.cli.formatters import FinancialFormatter

from edgarito.services.retrieval.edgar_rest_client.low_level_client import EDGARLowLevelClient
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.retrieval.edgar_rest_client.submissions_client import SubmissionsClient
from edgarito.services.retrieval.downloader.download_service import DownloadService

from edgarito.schemas.edgar_responses.company_ticker import CompanyTickerResponse
from edgarito.schemas.edgar_responses.submission import TransposedFiling
from edgarito.schemas.edgar_responses.company_facts import CompanyFacts

from edgarito.enums.edgar.core_filing_type import CoreFilingType
from edgarito.enums.granularity import Granularity

from edgarito.services.retrieval.company_data_loader import CompanyDataLoader


class CliOperations:
    """Handles core CLI operations for EDGAR data retrieval and analysis."""

    def __init__(self):
        self._logger = logging.getLogger(__class__.__name__)
        self._cache = FileSystemCache(root_directory=settings.cache_path)

    async def cik_from_ticker(self, ticker: str, use_cache: bool = True, make_cache: bool = False) -> int:
        """Resolve CIK from ticker symbol."""
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
        """Find and list all company tickers and CIKs."""
        async with EDGARLowLevelClient(cache=self._cache, user_agent=settings.user_agent) as client:
            tickers = await client.get_tickers(use_cache=use_cache, make_cache=make_cache)
            tickers = sorted(tickers, key=lambda x: x.title)
            for ticker in tickers:
                self._logger.info(f"{ticker.cik_str}\t{ticker.ticker}\t\t{ticker.title}")
            return tickers

    async def find_ticker_from_cik(self, cik: int, use_cache: bool = True, make_cache: bool = False) -> str:
        """Resolve ticker from CIK."""
        async with EDGARLowLevelClient(cache=self._cache, user_agent=settings.user_agent) as client:
            tickers = await client.get_tickers(use_cache=use_cache, make_cache=make_cache)
            for ticker in tickers:
                if ticker.cik_str == cik:
                    # self._logger.info(f"{ticker.ticker}")
                    self._logger.info(f"{ticker.ticker} has cik {ticker.cik_str} and title {ticker.title}")
                    return ticker.ticker
            raise ValueError(f"CIK {cik} not found")

    async def find_submissions_from_ticker(
        self, ticker: str, type: Optional[CoreFilingType] = None, use_cache: bool = True, make_cache: bool = False
    ) -> List[TransposedFiling]:
        """Find submissions by ticker symbol."""
        resolved_cik = await self.cik_from_ticker(ticker, use_cache=use_cache, make_cache=make_cache)
        return await self.find_submissions_from_cik(resolved_cik, type=type, use_cache=use_cache, make_cache=make_cache)

    async def find_submissions_from_cik(
        self, cik: int, type: Optional[CoreFilingType] = None, use_cache: bool = True, make_cache: bool = False, limit: Optional[int] = None
    ) -> List[TransposedFiling]:
        """Find submissions by CIK."""
        async with EDGARLowLevelClient(cache=self._cache, user_agent=settings.user_agent) as client:
            submissions_client = SubmissionsClient(client)
            submissions = await submissions_client.get_all_submission_filings_transposed(cik, filing_type=type, use_cache=use_cache, make_cache=make_cache)

            if limit:
                submissions = submissions[-limit:]  # Return the last N submissions
            for submission in submissions:
                self._logger.info(f"{submission.filingDate}\t\t{submission.accessionNumber}\t{submission.form} ({submission.core_type})")
            return submissions

    async def download_from_cik(
        self, cik: int, type: Optional[CoreFilingType] = None, use_cache: bool = True, make_cache: bool = False, limit: Optional[int] = 5
    ) -> List[pathlib.Path]:
        """Download filings by CIK."""
        filings = await self.find_submissions_from_cik(cik, type=type, use_cache=use_cache, make_cache=make_cache, limit=limit)
        async with aiohttp.ClientSession(headers={"User-Agent": settings.user_agent, "Accept-Encoding": "gzip, deflate"}) as session:
            download_service = DownloadService(session, download_root_dir=f"{settings.cache_path}/downloads")
            return await download_service.download_multiple(cik=cik, filings=filings)

    async def download_from_ticker(
        self, ticker: str, type: Optional[CoreFilingType] = None, use_cache: bool = True, make_cache: bool = False, limit: Optional[int] = 5
    ) -> List[pathlib.Path]:
        """Download filings by ticker symbol."""
        resolved_cik = await self.cik_from_ticker(ticker, use_cache=use_cache, make_cache=make_cache)
        return await self.download_from_cik(resolved_cik, type=type, use_cache=use_cache, make_cache=make_cache, limit=limit)

    async def facts_from_cik(self, cik: int, use_cache: bool = True, make_cache: bool = True) -> CompanyFacts:
        """Load company facts using CompanyDataLoader (includes 6-K data integration)."""
        async with aiohttp.ClientSession(headers={"User-Agent": settings.user_agent, "Accept-Encoding": "gzip, deflate"}) as session:
            loader = CompanyDataLoader(
                session,
                cache_dir=settings.cache_path,
                download_dir=f"{settings.cache_path}/downloads",
                user_agent=settings.user_agent,
                use_cache=use_cache,
                make_cache=make_cache
            )
            result = await loader.load_from_cik(cik)
            return result.facts

    async def facts_from_ticker(self, ticker: str, use_cache: bool = True, make_cache: bool = True) -> CompanyFacts:
        """Load company facts by ticker symbol."""
        cik = await self.cik_from_ticker(ticker, use_cache=use_cache, make_cache=make_cache)
        return await self.facts_from_cik(cik, use_cache=use_cache, make_cache=make_cache)

    async def display_financials_from_ticker(self, ticker: str, use_cache: bool = True, make_cache: bool = True):
        """Display formatted financial statements for all available periods by ticker."""
        async with aiohttp.ClientSession(headers={"User-Agent": settings.user_agent, "Accept-Encoding": "gzip, deflate"}) as session:
            loader = CompanyDataLoader(
                session, 
                cache_dir=settings.cache_path,
                download_dir=f"{settings.cache_path}/downloads",
                user_agent=settings.user_agent,
                use_cache=use_cache,
                make_cache=make_cache
            )
            result = await loader.load_from_ticker(ticker)
        
        FinancialFormatter.display_financials(result.facts, ticker)

    async def display_financials_from_cik(self, cik: int, use_cache: bool = True, make_cache: bool = True):
        """Display formatted financial statements for all available periods by CIK."""
        async with aiohttp.ClientSession(headers={"User-Agent": settings.user_agent, "Accept-Encoding": "gzip, deflate"}) as session:
            loader = CompanyDataLoader(
                session,
                cache_dir=settings.cache_path,
                download_dir=f"{settings.cache_path}/downloads",
                user_agent=settings.user_agent,
                use_cache=use_cache,
                make_cache=make_cache
            )
            result = await loader.load_from_cik(cik)
        
        ticker = await self.find_ticker_from_cik(cik, use_cache=use_cache, make_cache=make_cache)
        FinancialFormatter.display_financials(result.facts, ticker)

    async def analyze_red_flags_from_ticker(
        self, 
        ticker: str, 
        granularity: Granularity,
        use_cache: bool = True, 
        make_cache: bool = True
    ):
        """Analyze company for financial red flags from ticker."""
        facts = await self.facts_from_ticker(ticker, use_cache=use_cache, make_cache=make_cache)
        
        from edgarito.services.analysis.red_flags_service import RedFlagsService
        from edgarito.services.market.yahoo_finance_service import YahooFinanceService
        
        # Try to get market data from Yahoo Finance
        yahoo_service = YahooFinanceService()
        market_data = await yahoo_service.get_market_data(ticker)
        market_cap = market_data.market_cap if market_data else None
        
        if market_cap:
            self._logger.info(f"Fetched market cap for {ticker}: ${market_cap:,.0f}")
        else:
            self._logger.warning(f"Could not fetch market cap for {ticker}, proceeding without it")
        
        service = RedFlagsService(facts, market_cap=market_cap)
        report = service.analyze(granularity)
        
        print(report)

    async def analyze_red_flags_from_cik(
        self, 
        cik: int, 
        granularity: Granularity,
        use_cache: bool = True, 
        make_cache: bool = True
    ):
        """Analyze company for financial red flags from CIK."""
        facts = await self.facts_from_cik(cik, use_cache=use_cache, make_cache=make_cache)
        
        from edgarito.services.analysis.red_flags_service import RedFlagsService
        from edgarito.services.market.yahoo_finance_service import YahooFinanceService
        
        # Resolve ticker from CIK
        ticker = await self.find_ticker_from_cik(cik, use_cache=use_cache, make_cache=make_cache)
        
        # Try to get market data from Yahoo Finance
        yahoo_service = YahooFinanceService()
        market_data = await yahoo_service.get_market_data(ticker)
        market_cap = market_data.market_cap if market_data else None
        
        if market_cap:
            self._logger.info(f"Fetched market cap for {ticker}: ${market_cap:,.0f}")
        else:
            self._logger.warning(f"Could not fetch market cap for {ticker}, proceeding without it")
        
        service = RedFlagsService(facts, market_cap=market_cap)
        report = service.analyze(granularity)
        
        print(report)
