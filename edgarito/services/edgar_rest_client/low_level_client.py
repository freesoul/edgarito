from typing import Optional, List
import logging
import asyncio
import json

import aiohttp
import urllib.parse

from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.taxonomy_client.xbrl_taxonomy_client import TaxonomyClient

from edgarito.schemas.edgar_responses.company_ticker import CompanyTickerResponse
from edgarito.schemas.edgar_responses.submission import CompanySubmissionsResponse, FilingRecent
from edgarito.schemas.edgar_responses.company_facts import CompanyFacts, Fact


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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._session.close()

    async def get_tickers(self, use_cache: bool = True, make_cache: bool = True) -> List[CompanyTickerResponse]:
        raw_json = await self._fetch_json_with_retry_and_cache("https://www.sec.gov/files/company_tickers.json", use_cache=use_cache, make_cache=make_cache)
        return [CompanyTickerResponse(**d) for d in raw_json.values()]

    async def get_submissions(self, cik: int, use_cache: bool = True, make_cache: bool = True) -> CompanySubmissionsResponse:
        cik_str = str(cik).zfill(10)
        raw_json = await self._fetch_json_with_retry_and_cache(
            f"https://data.sec.gov/submissions/CIK{cik_str}.json", use_cache=use_cache, make_cache=make_cache
        )
        return CompanySubmissionsResponse(**raw_json)

    async def get_submission_additional_filings(self, remote_file_name: str, use_cache: bool = True, make_cache: bool = True) -> Optional[FilingRecent]:
        raw_json = await self._fetch_json_with_retry_and_cache(
            f"https://data.sec.gov/submissions/{remote_file_name}", use_cache=use_cache, make_cache=make_cache
        )
        return FilingRecent(**raw_json)

    async def get_company_fact(self, cik: int, fact_name: str, use_cache: bool = True, make_cache: bool = True) -> Optional[Fact]:
        cik_str = str(cik).zfill(10)
        try:
            raw_json = await self._fetch_json_with_retry_and_cache(
                f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik_str}/us-gaap/{fact_name}.json", use_cache=use_cache, make_cache=make_cache
            )
        except FileNotFoundError as e:
            self._logger.warning(f"Fact not found: {fact_name} for CIK {cik_str}")
            return
        return Fact(**raw_json)

    async def get_company_facts(self, cik: int, use_cache: bool = True, make_cache: bool = True, taxonomy_url: Optional[str] = None) -> CompanyFacts:
        """
        If taxonomy_url is provided, it will move deprecated facts to the us_gaap_deprecated field.
        """
        cik_str = str(cik).zfill(10)
        raw_json = await self._fetch_json_with_retry_and_cache(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json", use_cache=use_cache, make_cache=make_cache
        )
        if taxonomy_url:
            await self._move_deprecated_facts(raw_json, taxonomy_url=taxonomy_url)
        schema = CompanyFacts(**raw_json)
        return schema

    async def _move_deprecated_facts(self, raw_json: dict, taxonomy_url: str) -> None:
        raw_json["facts"]["us_gaap_deprecated"] = {}

        taxonomy_client = TaxonomyClient(self._cache)
        await taxonomy_client.load(taxonomy_url)
        valid_facts = taxonomy_client.get_gaap_keys()
        for fact_name in list(raw_json["facts"]["us-gaap"].keys()):
            if fact_name not in valid_facts:
                self._logger.warning(f"Deprecated fact: {fact_name}")
                raw_json["facts"]["us_gaap_deprecated"][fact_name] = raw_json["facts"]["us-gaap"].pop(fact_name)

    async def _fetch_json_with_retry_and_cache(
        self,
        url: str,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> dict:
        if use_cache or make_cache:
            local_filesystem_cached_file_path = f"edgar_rest/{FileSystemCache.path_from_url(url)}"

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
                elif resp.status == 404:
                    raise FileNotFoundError(f"404 Not Found: {url}")
                return await resp.json()


if __name__ == "__main__":
    import asyncio
    from edgarito.services.cache.filesystem_cache import FileSystemCache
    from edgarito.cli.logger import configure_logger

    configure_logger()

    async def main():
        cache = FileSystemCache("./cache")
        client = EDGARLowLevelClient(cache, user_agent="Jean Francois Kener (betterask.jf@gmail.com)")
        # data = await client.get_tickers()
        data = await client.get_company_fact(cik=1018724, fact_name="CommonStockSharesOutstanding")
        print(data)

    asyncio.run(main())
