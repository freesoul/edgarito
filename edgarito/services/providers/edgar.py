import asyncio
import datetime
import json
import logging
import re
import urllib.parse
from typing import List, Optional

import aiohttp

from edgarito.schemas.providers.edgar.company_facts import CompanyFacts, Fact
from edgarito.schemas.providers.edgar.company_ticker import CompanyTickerResponse
from edgarito.schemas.providers.edgar.filing import SecFiling, SecFilingDocument
from edgarito.schemas.providers.edgar.submission import (
    CompanySubmissionsResponse,
    FilingRecent,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache


class EdgarClient:
    GUIDANCE_FORMS = frozenset(
        {
            "8-K",
            "8-K/A",
            "6-K",
            "6-K/A",
            "10-Q",
            "10-Q/A",
            "10-K",
            "10-K/A",
        }
    )

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

    async def get_filings(
        self,
        cik: int,
        *,
        forms: set[str] | frozenset[str] | None = None,
        as_of: datetime.date | None = None,
        since: datetime.date | None = None,
        use_cache: bool = True,
        make_cache: bool = True,
        include_older_submissions: bool = False,
    ) -> list[SecFiling]:
        """Return filing metadata with a strict filing-date look-ahead cutoff."""
        submissions = (
            await self.get_all_submissions(
                cik, use_cache=use_cache, make_cache=make_cache
            )
            if include_older_submissions
            else await self.get_submissions(
                cik, use_cache=use_cache, make_cache=make_cache
            )
        )
        selected_forms = {item.upper() for item in forms} if forms else None
        filings: list[SecFiling] = []
        for item in submissions.filings.recent.transpose():
            if selected_forms is not None and item.form.upper() not in selected_forms:
                continue
            if as_of is not None and item.filingDate > as_of:
                continue
            if since is not None and item.filingDate < since:
                continue
            filings.append(self._sec_filing(cik, item))
        return sorted(
            filings,
            key=lambda filing: (
                filing.filing_date,
                filing.acceptance_datetime or datetime.datetime.min,
            ),
            reverse=True,
        )

    async def get_guidance_filings(
        self,
        cik: int,
        *,
        as_of: datetime.date,
        lookback_days: int = 180,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> list[SecFiling]:
        return await self.get_filings(
            cik,
            forms=self.GUIDANCE_FORMS,
            as_of=as_of,
            since=as_of - datetime.timedelta(days=lookback_days),
            use_cache=use_cache,
            make_cache=make_cache,
        )

    async def get_operating_filings(
        self,
        cik: int,
        *,
        as_of: datetime.date,
        lookback_days: int = 1825,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> list[SecFiling]:
        """Return a historical operating-evidence filing window.

        Guidance retrieval intentionally stays short-window and current-report
        oriented.  Operating discovery needs prior 10-K/10-Q tables as well as
        8-K evidence, so it uses the older-submissions files when available and
        applies the as-of cutoff before selection.
        """

        return await self.get_filings(
            cik,
            forms=self.GUIDANCE_FORMS,
            as_of=as_of,
            since=as_of - datetime.timedelta(days=max(0, lookback_days)),
            use_cache=use_cache,
            make_cache=make_cache,
            include_older_submissions=True,
        )

    async def get_raw_operating_filings(
        self,
        cik: int,
        *,
        as_of: datetime.date,
        lookback_days: int = 1825,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> list[SecFiling]:
        """Return the complete operating filing inventory before selection."""
        return await self.get_operating_filings(
            cik,
            as_of=as_of,
            lookback_days=lookback_days,
            use_cache=use_cache,
            make_cache=make_cache,
        )

    async def get_filing_documents(
        self,
        filing: SecFiling,
        *,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> SecFiling:
        """Fetch and parse the full-submission SGML document blocks.

        Raw full-submission text and parsed document JSON use distinct SEC cache
        entries keyed by immutable CIK/accession identity.
        """
        base = f"providers/edgar/filings/{filing.cik}/{filing.accession_number}"
        parsed_path = f"{base}/documents.json"
        if use_cache:
            cached = self._cache.read(parsed_path)
            if cached is not None:
                documents = tuple(
                    SecFilingDocument.model_validate(item)
                    for item in json.loads(cached)
                )
                return filing.model_copy(
                    update={
                        "documents": self._mark_primary_documents(filing, documents)
                    }
                )

        raw_path = f"{base}/full-submission.txt"
        submission_text = self._cache.read(raw_path) if use_cache else None
        if submission_text is None:
            submission_text = await self._fetch_text_with_retry(filing.archive_url)
            if make_cache:
                self._cache.save(raw_path, submission_text)

        documents = tuple(self.parse_submission_documents(submission_text))
        documents = self._mark_primary_documents(filing, documents)
        if make_cache:
            self._cache.save(
                parsed_path,
                json.dumps(
                    [item.model_dump(mode="json") for item in documents],
                    ensure_ascii=False,
                ),
            )
        return filing.model_copy(update={"documents": documents})

    async def get_filing_exhibits(
        self,
        filing: SecFiling,
        *,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> tuple[SecFilingDocument, ...]:
        """Retrieve the parsed filing and expose its attachment inventory.

        The full submission remains the cache boundary.  Exhibit ranking is a
        selector concern, so this method intentionally does not duplicate SEC
        requests or introduce issuer-specific URLs.
        """

        populated = await self.get_filing_documents(
            filing, use_cache=use_cache, make_cache=make_cache
        )
        return tuple(
            document
            for document in populated.documents
            if re.fullmatch(r"EX-99(?:\.\d+)?", document.document_type.strip(), re.I)
        )

    @staticmethod
    def _mark_primary_documents(
        filing: SecFiling, documents: tuple[SecFilingDocument, ...]
    ) -> tuple[SecFilingDocument, ...]:
        primary_filename = filing.primary_document.casefold()
        return tuple(
            document.model_copy(
                update={"is_primary": document.filename.casefold() == primary_filename}
            )
            for document in documents
        )

    @staticmethod
    def parse_submission_documents(text: str) -> list[SecFilingDocument]:
        """Split SEC full-submission SGML without interpreting document HTML."""
        documents: list[SecFilingDocument] = []
        blocks = re.findall(
            r"<DOCUMENT>(.*?)(?:</DOCUMENT>|\Z)", text, flags=re.I | re.S
        )
        for block in blocks:

            def value(tag: str, source: str = block) -> str:
                match = re.search(rf"<{tag}>\s*([^\r\n<]*)", source, flags=re.I)
                return match.group(1).strip() if match else ""

            text_match = re.search(r"<TEXT>(.*?)(?:</TEXT>|\Z)", block, re.I | re.S)
            content = text_match.group(1).strip() if text_match else ""
            filename = value("FILENAME")
            document_type = value("TYPE")
            if not filename and not content:
                continue
            documents.append(
                SecFilingDocument(
                    filename=filename or f"document-{len(documents) + 1}.txt",
                    document_type=document_type or "UNKNOWN",
                    description=value("DESCRIPTION"),
                    sequence=value("SEQUENCE") or None,
                    content=content,
                )
            )
        return documents

    @staticmethod
    def _sec_filing(cik: int, item: FilingRecent | object) -> SecFiling:
        raw_items = getattr(item, "items", "") or ""
        return SecFiling(
            cik=cik,
            accession_number=item.accessionNumber,
            form=item.form,
            filing_date=item.filingDate,
            acceptance_datetime=item.acceptanceDateTime,
            report_date=item.reportDate,
            items=tuple(part.strip() for part in raw_items.split(",") if part.strip()),
            primary_document=item.primaryDocument,
            primary_document_description=item.primaryDocDescription or "",
        )

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

    async def _fetch_text_with_retry(
        self,
        url: str,
        timeout: int = 20,
        threshold_exceeded_delay: int = 2,
        max_attempts: int = 5,
    ) -> str:
        host = urllib.parse.urlparse(url).netloc
        headers = {
            "Host": host,
            "User-Agent": self._user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        for attempt in range(1, max_attempts + 1):
            async with self._session.get(url, timeout=timeout, headers=headers) as resp:
                response_text = await resp.text(errors="replace")
                if resp.status in (403, 429):
                    if attempt == max_attempts:
                        raise RuntimeError(
                            f"SEC rate limit persisted after {max_attempts} attempts: {url}"
                        )
                    await asyncio.sleep(threshold_exceeded_delay * attempt)
                    continue
                if resp.status == 404:
                    raise FileNotFoundError(f"404 Not Found: {url}")
                if resp.status >= 500 and attempt < max_attempts:
                    await asyncio.sleep(threshold_exceeded_delay * attempt)
                    continue
                if resp.status >= 400:
                    raise RuntimeError(
                        f"SEC request failed with HTTP {resp.status}: {url}"
                    )
                return response_text
        raise RuntimeError(f"SEC request failed after {max_attempts} attempts: {url}")
