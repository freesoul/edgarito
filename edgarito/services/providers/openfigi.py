import json
from typing import Optional

import aiohttp

from edgarito.schemas.providers.fmp.fundamentals import SecuritySearchResult
from edgarito.services.cache.filesystem_cache import FileSystemCache


class OpenFigiClient:
    """Map ISINs to listed symbols through OpenFIGI's public mapping API."""

    _MAPPING_URL = "https://api.openfigi.com/v3/mapping"

    def __init__(
        self,
        cache: FileSystemCache,
        api_key: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self._cache = cache
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._owns_session and self._session is not None:
            await self._session.close()

    async def search_isin(
        self, isin: str, use_cache: bool = True, make_cache: bool = True
    ) -> list[SecuritySearchResult]:
        normalized_isin = isin.strip().upper()
        cache_path = f"providers/openfigi/isin/{normalized_isin}.json"
        if use_cache:
            cached_data = self._cache.read(cache_path)
            if cached_data is not None:
                return self._parse_response(json.loads(cached_data), normalized_isin)

        if self._session is None:
            self._session = aiohttp.ClientSession()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["X-OPENFIGI-APIKEY"] = self._api_key
        payload = [{"idType": "ID_ISIN", "idValue": normalized_isin}]
        try:
            async with self._session.post(
                self._MAPPING_URL,
                json=payload,
                headers=headers,
                timeout=20,
            ) as response:
                response_text = await response.text()
                if response.status == 429:
                    raise RuntimeError(
                        "OpenFIGI rate limit reached; retry later or configure "
                        "OPENFIGI_API_KEY"
                    )
                if response.status >= 400:
                    raise RuntimeError(
                        "OpenFIGI mapping failed with HTTP "
                        f"{response.status} for ISIN {normalized_isin}"
                    )
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise RuntimeError(
                f"OpenFIGI mapping failed for ISIN {normalized_isin}"
            ) from exc

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenFIGI returned invalid JSON") from exc
        if make_cache:
            self._cache.save(cache_path, json.dumps(data))
        return self._parse_response(data, normalized_isin)

    @staticmethod
    def _parse_response(data, isin: str) -> list[SecuritySearchResult]:
        if (
            not isinstance(data, list)
            or len(data) != 1
            or not isinstance(data[0], dict)
        ):
            raise RuntimeError("OpenFIGI returned an unexpected mapping response")
        mapping = data[0]
        if mapping.get("error"):
            raise RuntimeError(f"OpenFIGI mapping failed: {mapping['error']}")
        if mapping.get("warning"):
            return []

        results = []
        seen = set()
        for item in mapping.get("data", []):
            if not isinstance(item, dict) or not item.get("ticker"):
                continue
            if item.get("marketSector") not in (None, "Equity"):
                continue
            key = (item["ticker"], item.get("exchCode"))
            if key in seen:
                continue
            seen.add(key)
            results.append(
                SecuritySearchResult(
                    symbol=item["ticker"],
                    name=item.get("name"),
                    isin=isin,
                    exchangeShortName=item.get("exchCode"),
                )
            )
        return results
