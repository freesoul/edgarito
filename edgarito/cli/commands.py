"""Typer command definitions for the CLI."""

import asyncio
import typer

from edgarito.cli.operations import CliOperations
from edgarito.enums.edgar.core_filing_type import CoreFilingType
from edgarito.enums.granularity import Granularity


def create_app() -> typer.Typer:
    """Create and configure the Typer application with all commands."""
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
        """Resolve CIK from ticker symbol."""
        operations = CliOperations()
        asyncio.run(operations.cik_from_ticker(ticker, use_cache, make_cache))

    @app.command(context_settings=context_settings)
    def tickers(
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
    ):
        """List all company tickers and CIKs."""
        operations = CliOperations()
        asyncio.run(operations.find_all_ciks(use_cache, make_cache))

    @app.command(context_settings=context_settings)
    def ticker(
        cik: int = typer.Option(..., help="CIK to resolve ticker"),
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
    ):
        """Resolve ticker from CIK."""
        operations = CliOperations()
        asyncio.run(operations.find_ticker_from_cik(cik, use_cache, make_cache))

    @app.command(context_settings=context_settings)
    def submissions(
        ticker: str = typer.Option(None, help="Ticker to resolve CIK"),
        cik: int = typer.Option(None, help="CIK to resolve ticker"),
        type: str = typer.Option(None, help="Type of filing (e.g., '10-K', '10-Q')"),
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
        limit: int = typer.Option(None, help="Limit the number of submissions"),
    ):
        """Find and list company submissions/filings."""
        operations = CliOperations()
        if not ticker and not cik:
            raise typer.Abort("Provide a valid ticker with --ticker or a CIK with --cik")
        
        # Convert string to CoreFilingType if provided
        filing_type = CoreFilingType.try_from_string(type) if type else None
        
        if ticker:
            asyncio.run(operations.find_submissions_from_ticker(ticker, filing_type, use_cache, make_cache))
        elif cik:
            asyncio.run(operations.find_submissions_from_cik(cik, filing_type, use_cache, make_cache, limit))

    @app.command(context_settings=context_settings)
    def download(
        ticker: str = typer.Option(None, help="Ticker to resolve CIK"),
        cik: int = typer.Option(None, help="CIK to resolve ticker"),
        type: str = typer.Option(None, help="Type of filing (e.g., '10-K', '10-Q')"),
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
        limit: int = typer.Option(5, help="Limit the number of submissions"),
    ):
        """Download company filings."""
        operations = CliOperations()
        if not ticker and not cik:
            raise typer.Abort("Provide a valid ticker with --ticker or a CIK with --cik")
        
        # Convert string to CoreFilingType if provided
        filing_type = CoreFilingType.try_from_string(type) if type else None
        
        if ticker:
            asyncio.run(operations.download_from_ticker(ticker, filing_type, use_cache, make_cache, limit))
        elif cik:
            asyncio.run(operations.download_from_cik(cik, filing_type, use_cache, make_cache, limit))

    @app.command(context_settings=context_settings)
    def financials(
        ticker: str = typer.Option(None, help="Ticker to display financials"),
        cik: int = typer.Option(None, help="CIK to display financials"),
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
    ):
        """Display financial statements (annual and quarterly) for a company."""
        operations = CliOperations()
        if not ticker and not cik:
            raise typer.Abort("Provide a valid ticker with --ticker or a CIK with --cik")
        
        if ticker:
            asyncio.run(operations.display_financials_from_ticker(ticker, use_cache, make_cache))
        elif cik:
            asyncio.run(operations.display_financials_from_cik(cik, use_cache, make_cache))

    @app.command(context_settings=context_settings)
    def redflags(
        ticker: str = typer.Option(None, help="Ticker to analyze"),
        cik: int = typer.Option(None, help="CIK to analyze"),
        granularity: str = typer.Option("annual", help="Analysis granularity: 'annual' or 'quarterly'"),
        use_cache: bool = typer.Option(True, help="Use cache"),
        make_cache: bool = typer.Option(True, help="Make cache"),
    ):
        """Analyze company for financial red flags."""
        operations = CliOperations()
        if not ticker and not cik:
            raise typer.Abort("Provide a valid ticker with --ticker or a CIK with --cik")
        
        # Parse granularity
        gran = Granularity.ANNUAL if granularity.lower() == "annual" else Granularity.QUARTERLY
        
        if ticker:
            asyncio.run(operations.analyze_red_flags_from_ticker(ticker, gran, use_cache, make_cache))
        elif cik:
            asyncio.run(operations.analyze_red_flags_from_cik(cik, gran, use_cache, make_cache))

    return app
