import os
from typing import Optional
import asyncio

from edgarito.cli.logger import configure_logger
from edgarito.cli.settings import settings
from edgarito.enums.cli.actions import Action

from edgarito.services.edgar_rest_client.low_level_client import EDGARLowLevelClient
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.edgar_rest_client.submissions_client import SubmissionsClient

if __name__ == "__main__":
    configure_logger()

    # fix for Windows
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def main():
        cache = FileSystemCache(root_directory=settings.cache_path)

        # Resolve ticker CIK if provided.
        resolved_cik: Optional[int] = None

        if settings.cik:
            resolved_cik = settings.cik

        elif settings.ticker:
            async with EDGARLowLevelClient(cache=cache, user_agent=settings.user_agent) as client:
                tickers = await client.get_tickers(use_cache=settings.use_cache, make_cache=settings.make_cache)
                if settings.ticker:
                    for ticker in tickers:
                        if ticker.ticker.lower() == settings.ticker.lower():
                            resolved_cik = ticker.cik_str  # that's the name from the response, but it's an int
                            break
                    else:
                        raise ValueError(f"Ticker {settings.ticker} not found")

        # Do actions.
        if settings.action == Action.FIND_ALL_CICKS:
            async with EDGARLowLevelClient(cache=cache, user_agent=settings.user_agent) as client:
                tickers = await client.get_tickers(use_cache=settings.use_cache, make_cache=settings.make_cache)
                for ticker in tickers:
                    print(f"{ticker.ticker}: {ticker.cik_str}")

        elif settings.action == Action.FIND_CIK:
            if resolved_cik is None:
                raise ValueError("Provide a valid ticker with --ticker")
            print(f"CIK: {resolved_cik}")

        elif settings.action == Action.FIND_SUBMISSIONS:
            if resolved_cik is None:
                raise ValueError("Provide a valid ticker with --ticker or a CIK with --cik")
            async with EDGARLowLevelClient(cache=cache, user_agent=settings.user_agent) as client:
                submissions_client = SubmissionsClient(client)
                submissions = await submissions_client.get_all_submission_filings_transposed(
                    resolved_cik, use_cache=settings.use_cache, make_cache=settings.make_cache
                )
                for submission in submissions:
                    print(submission)

    asyncio.run(main())
