import asyncio
import json
import logging
import urllib.parse
from typing import List, Optional

import aiohttp

from edgarito.schemas.providers.edgar.company_facts import CompanyFacts, Fact
from edgarito.schemas.providers.edgar.company_ticker import CompanyTickerResponse
from edgarito.schemas.providers.edgar.submission import (
    CompanySubmissionsResponse,
    FilingRecent,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache


class EdgarClient:
    def __init__(
        self,
        cache: FileSystemCache,
        user_agent: str,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """
        user_agent should be 'Name (email)'
        """
        self._logger = logging.getLogger(__class__.__name__)
        self._cache = cache
        self._user_agent = user_agent
        self._owns_session = session is None

        if session is None:
            # https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
            )
        else:
            self._session = session

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._owns_session:
            await self._session.close()

    async def get_cik(
        self, ticker: str, use_cache: bool = True, make_cache: bool = True
    ) -> int:
        """Resolve a ticker to its SEC Central Index Key."""
        normalized_ticker = ticker.strip().upper()
        for company in await self.get_tickers(
            use_cache=use_cache, make_cache=make_cache
        ):
            if company.ticker.upper() == normalized_ticker:
                return company.cik
        raise ValueError(
            f"Ticker '{ticker}' was not found in the SEC company ticker list"
        )

    async def get_tickers(
        self, use_cache: bool = True, make_cache: bool = True
    ) -> List[CompanyTickerResponse]:
        raw_json = await self._fetch_json_with_retry_and_cache(
            "https://www.sec.gov/files/company_tickers.json",
            use_cache=use_cache,
            make_cache=make_cache,
        )
        return [CompanyTickerResponse(**d) for d in raw_json.values()]

    async def get_all_submissions(
        self, cik: int, use_cache: bool = True, make_cache: bool = True
    ) -> CompanySubmissionsResponse:

        first = await self.get_submissions(
            cik, use_cache=use_cache, make_cache=make_cache
        )
        self._logger.info(
            f"Got {len(first.filings.recent.accessionNumber)} filings for CIK {cik}"
        )

        for next_filings_info in first.filings.files:
            self._logger.info(f"Getting additional filings for {next_filings_info}")
            additional_filings = await self.get_submission_additional_filings(
                next_filings_info.name, use_cache=use_cache, make_cache=make_cache
            )
            self._logger.info(
                f"Got {len(additional_filings.accessionNumber)} additional filings for {next_filings_info}"
            )
            first.filings.recent.extend_in_place(additional_filings)

        return first

    async def get_submissions(
        self, cik: int, use_cache: bool = True, make_cache: bool = True
    ) -> CompanySubmissionsResponse:
        cik_str = str(cik).zfill(10)
        raw_json = await self._fetch_json_with_retry_and_cache(
            f"https://data.sec.gov/submissions/CIK{cik_str}.json",
            use_cache=use_cache,
            make_cache=make_cache,
        )
        return CompanySubmissionsResponse(**raw_json)

    async def get_submission_additional_filings(
        self, remote_file_name: str, use_cache: bool = True, make_cache: bool = True
    ) -> Optional[FilingRecent]:
        raw_json = await self._fetch_json_with_retry_and_cache(
            f"https://data.sec.gov/submissions/{remote_file_name}",
            use_cache=use_cache,
            make_cache=make_cache,
        )
        return FilingRecent(**raw_json)

    async def get_company_fact(
        self, cik: int, fact_name: str, use_cache: bool = True, make_cache: bool = True
    ) -> Optional[Fact]:
        cik_str = str(cik).zfill(10)
        try:
            raw_json = await self._fetch_json_with_retry_and_cache(
                f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik_str}/us-gaap/{fact_name}.json",
                use_cache=use_cache,
                make_cache=make_cache,
            )
        except FileNotFoundError:
            self._logger.warning(f"Fact not found: {fact_name} for CIK {cik_str}")
            return
        return Fact(**raw_json)

    async def get_company_facts(
        self, cik: int, use_cache: bool = True, make_cache: bool = True
    ) -> CompanyFacts:
        """
        Fetch company facts (XBRL financial data) from SEC EDGAR.
        """
        cik_str = str(cik).zfill(10)
        raw_json = await self._fetch_json_with_retry_and_cache(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json",
            use_cache=use_cache,
            make_cache=make_cache,
        )
        schema = CompanyFacts(**raw_json)
        return schema

    async def _fetch_json_with_retry_and_cache(
        self,
        url: str,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> dict:
        if use_cache or make_cache:
            local_filesystem_cached_file_path = (
                f"providers/edgar/{FileSystemCache.path_from_url(url)}"
            )

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
        max_attempts: int = 5,
    ) -> dict:
        # The host vary and has to match: data.sec.gov for most routes, but www.sec.gov for the ticker route.
        host = urllib.parse.urlparse(url).netloc
        headers = {
            "Host": host,
            "User-Agent": self._user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        for attempt in range(1, max_attempts + 1):
            async with self._session.get(url, timeout=timeout, headers=headers) as resp:
                response_text = await resp.text()
                if resp.status in (403, 429) and (
                    "Request Rate Threshold Exceeded" in response_text
                    or resp.status == 429
                ):
                    if attempt == max_attempts:
                        raise RuntimeError(
                            f"SEC rate limit persisted after {max_attempts} attempts: {url}"
                        )
                    await asyncio.sleep(threshold_exceeded_delay)
                    continue
                if resp.status == 404:
                    raise FileNotFoundError(f"404 Not Found: {url}")
                if resp.status >= 400:
                    raise RuntimeError(
                        f"SEC request failed with HTTP {resp.status}: {url}"
                    )
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"SEC returned an invalid JSON response for {url}"
                    ) from exc

        raise RuntimeError(f"SEC request failed after {max_attempts} attempts: {url}")
