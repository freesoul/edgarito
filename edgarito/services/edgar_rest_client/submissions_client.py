import logging
from typing import List

from edgarito.schemas.edgar_responses.submission import CompanySubmissionsResponse, TransposedFiling

from edgarito.services.edgar_rest_client.low_level_client import EDGARLowLevelClient


class SubmissionsClient:

    def __init__(self, client: EDGARLowLevelClient):
        self._logger = logging.getLogger(__name__)
        self._client = client

    async def get_all_submissions(self, cik: int, use_cache: bool = True, make_cache: bool = True) -> CompanySubmissionsResponse:
        first = await self._client.get_submissions(cik)
        self._logger.info(f"Got {len(first.filings.recent.accessionNumber)} filings for CIK {cik}")

        for next_filings_info in first.filings.files:
            self._logger.info(f"Getting additional filings for {next_filings_info}")
            additional_filings = await self._client.get_submission_additional_filings(next_filings_info.name)
            self._logger.info(f"Got {len(additional_filings.accessionNumber)} additional filings for {next_filings_info}")
            first.filings.recent.extend_in_place(additional_filings)

        return first

    async def get_all_submission_filings_transposed(self, cik: int, use_cache: bool = True, make_cache: bool = True) -> List[TransposedFiling]:
        response = await self.get_all_submissions(cik, use_cache=use_cache, make_cache=make_cache)
        return response.filings.recent.transpose()
