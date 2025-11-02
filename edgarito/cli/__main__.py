import pathlib
import logging
import aiohttp
from typing import List, Optional
import asyncio

import typer

from edgarito.cli.logger import configure_logger
from edgarito.cli.settings import settings

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
from edgarito.services.financial.statements import FinancialStatements

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
        self, ticker: str, type: Optional[CoreFilingType] = None, use_cache: bool = True, make_cache: bool = False
    ) -> List[TransposedFiling]:
        resolved_cik = await self.cik_from_ticker(ticker, use_cache=use_cache, make_cache=make_cache)
        return await self.find_submissions_from_cik(resolved_cik, type=type, use_cache=use_cache, make_cache=make_cache)

    async def find_submissions_from_cik(
        self, cik: int, type: Optional[CoreFilingType] = None, use_cache: bool = True, make_cache: bool = False, limit: Optional[int] = None
    ) -> List[TransposedFiling]:
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
        filings = await self.find_submissions_from_cik(cik, type=type, use_cache=use_cache, make_cache=make_cache, limit=limit)
        async with aiohttp.ClientSession(headers={"User-Agent": settings.user_agent, "Accept-Encoding": "gzip, deflate"}) as session:
            download_service = DownloadService(session, download_root_dir=f"{settings.cache_path}/downloads")
            return await download_service.download_multiple(cik=cik, filings=filings)

    async def download_from_ticker(
        self, ticker: str, type: Optional[CoreFilingType] = None, use_cache: bool = True, make_cache: bool = False, limit: Optional[int] = 5
    ) -> List[pathlib.Path]:
        resolved_cik = await self.cik_from_ticker(ticker, use_cache=use_cache, make_cache=make_cache)
        return await self.download_from_cik(resolved_cik, type=type, use_cache=use_cache, make_cache=make_cache, limit=limit)

    async def facts_from_cik(self, cik: int, use_cache: bool = True, make_cache: bool = True) -> CompanyFacts:
        """Load company facts using CompanyDataLoader (includes 6-K data integration)"""
        
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
        cik = await self.cik_from_ticker(ticker, use_cache=use_cache, make_cache=make_cache)
        return await self.facts_from_cik(cik, use_cache=use_cache, make_cache=make_cache)

    async def display_financials_from_ticker(self, ticker: str, use_cache: bool = True, make_cache: bool = True):
        """Display formatted financial statements for all available periods"""
        
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
        
        self._display_financials(result.facts, ticker)

    async def display_financials_from_cik(self, cik: int, use_cache: bool = True, make_cache: bool = True):
        """Display formatted financial statements for all available periods"""
        
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
        self._display_financials(result.facts, ticker)

    async def analyze_red_flags_from_ticker(
        self, 
        ticker: str, 
        granularity: Granularity,
        use_cache: bool = True, 
        make_cache: bool = True
    ):
        """Analyze company for financial red flags from ticker"""
        facts = await self.facts_from_ticker(ticker, use_cache=use_cache, make_cache=make_cache)
        
        from edgarito.services.analysis.red_flags_service import RedFlagsService
        
        service = RedFlagsService(facts)
        report = service.analyze(granularity)
        
        print(report)

    async def analyze_red_flags_from_cik(
        self, 
        cik: int, 
        granularity: Granularity,
        use_cache: bool = True, 
        make_cache: bool = True
    ):
        """Analyze company for financial red flags from CIK"""
        facts = await self.facts_from_cik(cik, use_cache=use_cache, make_cache=make_cache)
        
        from edgarito.services.analysis.red_flags_service import RedFlagsService
        
        service = RedFlagsService(facts)
        report = service.analyze(granularity)
        
        print(report)

    def _display_financials(self, facts: CompanyFacts, identifier: str):
        """Internal method to format and display financial statements"""
        
        statements = FinancialStatements(facts)
        
        print(f"\n{'='*100}")
        print(f"FINANCIAL STATEMENTS: {identifier.upper()} - {facts.entityName}")
        print(f"{'='*100}")
        
        # Display Annual Data
        self._display_period_data(statements, Granularity.ANNUAL)
        
        # Display Quarterly Data
        self._display_period_data(statements, Granularity.QUARTERLY)

    def _display_period_data(self, statements, granularity: Granularity):
        """Display financial data for a specific granularity (ANNUAL or QUARTERLY)"""
        period_label = "ANNUAL" if granularity.name == "ANNUAL" else "QUARTERLY"
        
        print(f"\n{'-'*100}")
        print(f"{period_label} DATA")
        print(f"{'-'*100}\n")
        
        # Get all statement data
        try:
            assets = statements.balance_sheet.get_total_assets(granularity)
            revenue = statements.income_statement.get_revenue(granularity)
            net_income = statements.income_statement.get_net_income(granularity)
            ocf = statements.cash_flow.get_operating_cash_flow(granularity)
            
            # Get all unique periods, but prioritize periods with income statement data
            # We want to show periods where we have meaningful financial data, not just balance sheet
            all_periods = set()
            
            # Primary data sources (income statement is most important)
            if revenue: all_periods.update(revenue.periods)
            if net_income: all_periods.update(net_income.periods)
            
            # Add balance sheet periods that also have some income/cash flow data
            if assets:
                income_periods = set()
                if revenue: income_periods.update(revenue.periods)
                if net_income: income_periods.update(net_income.periods)
                if ocf: income_periods.update(ocf.periods)
                
                # Only add balance sheet periods if they have ANY income/cash flow data
                for period in assets.periods:
                    if period in income_periods:
                        all_periods.add(period)
            
            # Add OCF periods
            if ocf: all_periods.update(ocf.periods)
            
            if not all_periods:
                print(f"No {period_label.lower()} data available.\n")
                return
            
            # Sort periods chronologically
            sorted_periods = sorted(all_periods, key=lambda p: (p.year, p.fp.value if p.fp else 0))
            
            # Print header
            print(f"{'Period':<15} {'Assets':<18} {'Revenue':<18} {'Net Income':<18} {'Op. Cash Flow':<18}")
            print(f"{'-'*95}")
            
            # Print each period
            for period in sorted_periods:
                period_str = f"{period.year}"
                if period.fp and granularity.name == "QUARTERLY":
                    period_str += f" {period.fp.value}"
                
                assets_val = self._format_value(assets, period) if assets else "N/A"
                revenue_val = self._format_value(revenue, period) if revenue else "N/A"
                net_income_val = self._format_value(net_income, period) if net_income else "N/A"
                ocf_val = self._format_value(ocf, period) if ocf else "N/A"
                
                print(f"{period_str:<15} {assets_val:<18} {revenue_val:<18} {net_income_val:<18} {ocf_val:<18}")
            
            # Try to compute and display key metrics
            self._display_metrics(statements, granularity, sorted_periods)
            
        except Exception as e:
            print(f"Error displaying {period_label.lower()} data: {e}\n")

    def _format_value(self, measurements, period) -> str:
        """Format a value in millions with proper unit"""
        try:
            value_dict = {p: v for p, v in zip(measurements.periods, measurements.values)}
            if period in value_dict:
                value = value_dict[period]
                # Convert to millions
                value_millions = value / 1_000_000
                return f"${value_millions:,.1f}M"
            
            # For balance sheet, check if there's an FY period when looking for Q4
            # Balance sheets are point-in-time, so FY period = Q4 period
            from edgarito.enums.edgar.period import FiscalPeriod
            if period.fp == FiscalPeriod.Q4:
                fy_period = [p for p in measurements.periods if p.year == period.year and p.fp == FiscalPeriod.Year]
                if fy_period:
                    value = value_dict[fy_period[0]]
                    value_millions = value / 1_000_000
                    return f"${value_millions:,.1f}M"
            
            return "N/A"
        except:
            return "N/A"

    def _display_metrics(self, statements, granularity, sorted_periods):
        """Display computed financial metrics"""
        try:
            # Try to compute some key metrics
            gross_margin = statements.metrics.gross_margin(granularity)
            net_margin = statements.metrics.net_margin(granularity)
            roe = statements.metrics.return_on_equity(granularity)
            current_ratio = statements.metrics.current_ratio(granularity)
            
            if any([gross_margin, net_margin, roe, current_ratio]):
                print(f"\n{'Metrics':<15} {'Gross Margin':<15} {'Net Margin':<15} {'ROE':<15} {'Current Ratio':<15}")
                print(f"{'-'*75}")
                
                for period in sorted_periods:
                    period_str = f"{period.year}"
                    if period.fp and granularity.name == "QUARTERLY":
                        period_str += f" {period.fp.value}"
                    
                    gm = self._format_ratio(gross_margin, period) if gross_margin else "N/A"
                    nm = self._format_ratio(net_margin, period) if net_margin else "N/A"
                    roe_val = self._format_ratio(roe, period) if roe else "N/A"
                    cr = self._format_ratio(current_ratio, period, is_percentage=False) if current_ratio else "N/A"
                    
                    print(f"{period_str:<15} {gm:<15} {nm:<15} {roe_val:<15} {cr:<15}")
                
        except Exception as e:
            # Silently skip metrics if computation fails
            pass
        
        print()  # Add spacing after section

    def _format_ratio(self, measurements, period, is_percentage: bool = True) -> str:
        """Format a ratio value"""
        try:
            value_dict = {p: v for p, v in zip(measurements.periods, measurements.values)}
            if period in value_dict:
                value = value_dict[period]
                if is_percentage:
                    return f"{value*100:.1f}%"
                else:
                    return f"{value:.2f}x"
            return "N/A"
        except:
            return "N/A"


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
        type: str = typer.Option(None, help="Type of filing (e.g., '10-K', '10-Q')"),
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
        limit: int = typer.Option(None, help="Limit the number of submissions"),
    ):
        cli = Cli()
        if not ticker and not cik:
            raise typer.Abort("Provide a valid ticker with --ticker or a CIK with --cik")
        
        # Convert string to CoreFilingType if provided
        filing_type = CoreFilingType.try_from_string(type) if type else None
        
        if ticker:
            asyncio.run(cli.find_submissions_from_ticker(ticker, filing_type, use_cache, make_cache))
        elif cik:
            asyncio.run(cli.find_submissions_from_cik(cik, filing_type, use_cache, make_cache, limit))

    @app.command(context_settings=context_settings)
    def download(
        ticker: str = typer.Option(None, help="Ticker to resolve CIK"),
        cik: int = typer.Option(None, help="CIK to resolve ticker"),
        type: str = typer.Option(None, help="Type of filing (e.g., '10-K', '10-Q')"),
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
        limit: int = typer.Option(5, help="Limit the number of submissions"),
    ):
        cli = Cli()
        if not ticker and not cik:
            raise typer.Abort("Provide a valid ticker with --ticker or a CIK with --cik")
        
        # Convert string to CoreFilingType if provided
        filing_type = CoreFilingType.try_from_string(type) if type else None
        
        if ticker:
            asyncio.run(cli.download_from_ticker(ticker, filing_type, use_cache, make_cache, limit))
        elif cik:
            asyncio.run(cli.download_from_cik(cik, filing_type, use_cache, make_cache, limit))

    @app.command(context_settings=context_settings)
    def financials(
        ticker: str = typer.Option(None, help="Ticker to display financials"),
        cik: int = typer.Option(None, help="CIK to display financials"),
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
    ):
        """Display financial statements (annual and quarterly) for a company"""
        cli = Cli()
        if not ticker and not cik:
            raise typer.Abort("Provide a valid ticker with --ticker or a CIK with --cik")
        
        if ticker:
            asyncio.run(cli.display_financials_from_ticker(ticker, use_cache, make_cache))
        elif cik:
            asyncio.run(cli.display_financials_from_cik(cik, use_cache, make_cache))

    @app.command(context_settings=context_settings)
    def redflags(
        ticker: str = typer.Option(None, help="Ticker to analyze"),
        cik: int = typer.Option(None, help="CIK to analyze"),
        granularity: str = typer.Option("annual", help="Analysis granularity: 'annual' or 'quarterly'"),
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
    ):
        """Analyze company for financial red flags"""
        cli = Cli()
        if not ticker and not cik:
            raise typer.Abort("Provide a valid ticker with --ticker or a CIK with --cik")
        
        # Parse granularity
        gran = Granularity.ANNUAL if granularity.lower() == "annual" else Granularity.QUARTERLY
        
        if ticker:
            asyncio.run(cli.analyze_red_flags_from_ticker(ticker, gran, use_cache, make_cache))
        elif cik:
            asyncio.run(cli.analyze_red_flags_from_cik(cik, gran, use_cache, make_cache))

    app()
