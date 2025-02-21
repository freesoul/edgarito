from typing import Optional, List
import logging
import asyncio
import json

import aiohttp
import urllib.parse

from edgarito.services.cache.filesystem_cache import FileSystemCache

from edgarito.schemas.edgar.company_ticker import CompanyTicker


class EDGARLowLevelClient:

    def __init__(self, cache: FileSystemCache, user_agent: str, session: Optional[aiohttp.ClientSession] = None):
        """
        user_agent should be 'Name (email)'
        """
        self._logger = logging.getLogger(__class__.__name__)
        self._cache = cache

        if session is None:
            # https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
            self._session = aiohttp.ClientSession(headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        else:
            self._session = session

    async def get_tickers(self, use_cache: bool = True, make_cache: bool = True) -> List[CompanyTicker]:
        raw_json = await self._fetch_json_with_retry_and_cache("https://www.sec.gov/files/company_tickers.json", use_cache=use_cache, make_cache=make_cache)
        return [CompanyTicker(**d) for d in raw_json.values()]

    async def _fetch_json_with_retry_and_cache(
        self,
        url: str,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> dict:
        if use_cache or make_cache:
            local_filesystem_cached_file_path = f"edgar_rest/{urllib.parse.urlparse(url).path.lstrip("/")}"

        if use_cache:
            cached_data = self._cache.read(local_filesystem_cached_file_path)
            if cached_data is not None:
                return json.loads(cached_data)

        data = await self._fetch_json_with_retry(url)

        if make_cache:
            self._cache.save(local_filesystem_cached_file_path, json.dumps(data))

        return data

    async def _fetch_json_with_retry(
        self,
        url: str,
        timeout: int = 10,
        threshold_exceeded_delay: int = 10,
    ) -> dict:
        # The host vary and has to match: data.sec.gov for most routes, but www.sec.gov for the ticker route.
        host = urllib.parse.urlparse(url).netloc
        while True:
            async with self._session.get(url, timeout=timeout, headers={"Host": host}) as resp:
                if resp.status == 403 and "Request Rate Threshold Exceeded" in await resp.text():
                    await asyncio.sleep(threshold_exceeded_delay)
                    continue
                return await resp.json()


if __name__ == "__main__":
    import asyncio
    from edgarito.services.cache.filesystem_cache import FileSystemCache
    from edgarito.cli.logger import configure_logger

    configure_logger()

    async def main():
        cache = FileSystemCache("./cache")
        client = EDGARLowLevelClient(cache, user_agent="Jean Francois Kener (betterask.jf@gmail.com)")
        data = await client.get_tickers()
        print(data)

    asyncio.run(main())
