import asyncio

from edgarito.cli.logger import configure_logger
from edgarito.cli.settings import settings
from edgarito.enums.cli.actions import Action
from edgarito.services.edgar_client.low_level_client import EDGARLowLevelClient
from edgarito.services.cache.filesystem_cache import FileSystemCache

if __name__ == "__main__":
    configure_logger()

    async def main():
        if settings.use_cache:
            cache = FileSystemCache(root_directory=settings.cache_path)

        if settings.action == Action.FIND_TICKER_CIK:
            client = EDGARLowLevelClient(cache=cache, user_agent=settings.user_agent)
            tickers = await client.get_tickers(use_cache=settings.use_cache, make_cache=settings.make_cache)

            if settings.ticker:
                for ticker in tickers:
                    if ticker.ticker.lower() == settings.ticker.lower():
                        print(f"{ticker.ticker}: {ticker.cik_str}")
                        break
            else:
                for ticker in tickers:
                    print(f"{ticker.ticker}: {ticker.cik_str}")

    asyncio.run(main())
