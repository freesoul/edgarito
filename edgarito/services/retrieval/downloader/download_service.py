import enum
import asyncio
import os
from typing import Union, Optional
import logging
import aiohttp

import pathlib

from edgarito.schemas.edgar_responses.submission import TransposedFiling
from edgarito.services.cache.filesystem_cache import FileSystemCache


class DownloadOptions(enum.Enum):
    PRIMARY_DOCUMENT = "primary_document"
    TXT_SUBMISSION = "txt_submission"


class DownloadService:

    def __init__(
        self, 
        session: aiohttp.ClientSession, 
        download_root_dir: Union[str, pathlib.Path],
        cache: Optional[FileSystemCache] = None
    ):
        self._logger = logging.getLogger(__class__.__name__)
        self._session = session
        self.download_root_dir = pathlib.Path(download_root_dir)
        self._cache = cache
        os.makedirs(self.download_root_dir, exist_ok=True)

    async def download_multiple(
        self,
        cik: int,
        filings: list[TransposedFiling],
        what: Union[DownloadOptions, str] = DownloadOptions.TXT_SUBMISSION,
        custom_save_path: Optional[Union[str, pathlib.Path]] = None,
        parallelize: int = 1,
    ) -> list[pathlib.Path]:
        if isinstance(what, str):
            raise ValueError("It does not make sense to spell the exact file to download when downloading multiple filings")
        semaphore = asyncio.Semaphore(parallelize)
        async with semaphore:
            return await asyncio.gather(*[self.download_by_filing(cik=cik, filing=filing, custom_save_path=custom_save_path, what=what) for filing in filings])

    async def download_by_filing(
        self,
        cik: int,
        filing: TransposedFiling,
        custom_save_path: Optional[Union[str, pathlib.Path]],
        what: Union[DownloadOptions, str] = DownloadOptions.TXT_SUBMISSION,
    ) -> pathlib.Path:
        file_to_download = self._get_file_name(what, accession_number=filing.accessionNumber, primary_document=filing.primaryDocument)
        return await self.download(cik=cik, accession_number=filing.accessionNumber, custom_save_path=custom_save_path, file_to_download=file_to_download)

    async def download(
        self, cik: int, accession_number: str, file_to_download: str, custom_save_path: Optional[Union[str, pathlib.Path]] = None
    ) -> pathlib.Path:
        download_dir = custom_save_path if custom_save_path else (self.download_root_dir / str(cik).zfill(10) / accession_number)
        download_file_path = download_dir / file_to_download
        
        # Check if file already exists (cache)
        if download_file_path.exists():
            self._logger.debug(f"Using cached file: {download_file_path}")
            return download_file_path
        
        # Download from SEC
        content = await self.fetch(cik=cik, accession_number=accession_number, document_name=file_to_download)
        os.makedirs(download_dir, exist_ok=True)
        with open(download_file_path, "wb") as file:
            file.write(content)
        return download_file_path

    async def fetch(self, cik: int, accession_number: str, document_name: str) -> bytes:
        accession_number_no_dashes = accession_number.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_no_dashes}/{document_name}"
        self._logger.info(f"Fetching {url}")
        async with self._session.get(url) as response:
            return await response.read()

    def _get_file_name(self, what: Union[DownloadOptions, str], accession_number: Optional[str] = None, primary_document: Optional[str] = None) -> str:
        if isinstance(what, str):
            return what

        if what == DownloadOptions.PRIMARY_DOCUMENT:
            if primary_document is None:
                raise ValueError("Primary document is required when downloading primary document")
            return primary_document

        elif what == DownloadOptions.TXT_SUBMISSION:
            if accession_number is None:
                raise ValueError("Accession number is required when downloading txt submission")
            return f"{accession_number}.txt"

        raise ValueError(f"Unknown download option: {what}")
