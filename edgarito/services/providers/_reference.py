import calendar
import datetime
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping, Optional

import aiohttp

from edgarito.schemas.market import MarketDataFrequency
from edgarito.services.cache.filesystem_cache import FileSystemCache


@dataclass(frozen=True)
class RetrievedText:
    content: str
    retrieved_at: datetime.datetime
    sha256: str
    source_version: Optional[str] = None


class CachedTextProvider:
    """Shared HTTP lifecycle and integrity-preserving text cache."""

    def __init__(
        self,
        cache: FileSystemCache,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self._cache = cache
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._owns_session and self._session is not None:
            await self._session.close()

    async def _retrieve_text(
        self,
        *,
        provider: str,
        url: str,
        cache_path: str,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        expected_sha256: Optional[str] = None,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> RetrievedText:
        if use_cache:
            cached = self._cache.read(cache_path)
            if cached is not None:
                payload = self._parse_cached_payload(cached, provider)
                self._verify_checksum(payload, expected_sha256, provider)
                return payload

        if self._session is None:
            self._session = aiohttp.ClientSession()
        try:
            async with self._session.get(
                url,
                params=dict(params or {}),
                headers=dict(headers or {}),
                timeout=20,
            ) as response:
                content = await response.text()
                if response.status >= 400:
                    raise RuntimeError(
                        f"{provider} request failed with HTTP {response.status}"
                    )
                response_headers = getattr(response, "headers", {})
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise RuntimeError(f"{provider} request failed") from exc

        source_version = response_headers.get("ETag") or response_headers.get(
            "Last-Modified"
        )
        payload = RetrievedText(
            content=content,
            retrieved_at=datetime.datetime.now(datetime.timezone.utc),
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            source_version=source_version,
        )
        self._verify_checksum(payload, expected_sha256, provider)
        if make_cache:
            self._cache.save(
                cache_path,
                json.dumps(
                    {
                        "content": payload.content,
                        "retrieved_at": payload.retrieved_at.isoformat(),
                        "sha256": payload.sha256,
                        "source_version": payload.source_version,
                    }
                ),
            )
        return payload

    @staticmethod
    def _parse_cached_payload(content: str, provider: str) -> RetrievedText:
        try:
            data = json.loads(content)
            payload = RetrievedText(
                content=data["content"],
                retrieved_at=datetime.datetime.fromisoformat(data["retrieved_at"]),
                sha256=data["sha256"],
                source_version=data.get("source_version"),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid cached {provider} response") from exc
        actual = hashlib.sha256(payload.content.encode("utf-8")).hexdigest()
        if actual != payload.sha256:
            raise RuntimeError(f"Cached {provider} response failed its checksum")
        if (
            payload.retrieved_at.tzinfo is None
            or payload.retrieved_at.utcoffset() is None
        ):
            raise RuntimeError(f"Cached {provider} response has no retrieval timezone")
        return payload

    @staticmethod
    def _verify_checksum(
        payload: RetrievedText, expected_sha256: Optional[str], provider: str
    ) -> None:
        if expected_sha256 is not None and payload.sha256 != expected_sha256:
            raise RuntimeError(
                f"{provider} response checksum does not match the declared release"
            )


def frequency_from_code(code: str) -> MarketDataFrequency:
    normalized = code.strip().upper()
    mappings = {
        "D": MarketDataFrequency.DAILY,
        "B": MarketDataFrequency.DAILY,
        "W": MarketDataFrequency.WEEKLY,
        "BW": MarketDataFrequency.BIWEEKLY,
        "M": MarketDataFrequency.MONTHLY,
        "Q": MarketDataFrequency.QUARTERLY,
        "H": MarketDataFrequency.SEMIANNUAL,
        "SA": MarketDataFrequency.SEMIANNUAL,
        "A": MarketDataFrequency.ANNUAL,
        "I": MarketDataFrequency.IRREGULAR,
    }
    try:
        return mappings[normalized]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported reference-data frequency: {code!r}") from exc


def period_end_from_value(value: str, frequency: MarketDataFrequency) -> datetime.date:
    normalized = value.strip().upper()
    try:
        if frequency == MarketDataFrequency.ANNUAL:
            return datetime.date(int(normalized[:4]), 12, 31)
        if frequency == MarketDataFrequency.QUARTERLY:
            if "Q" in normalized:
                year_text, quarter_text = normalized.replace("-", "").split("Q", 1)
                year, quarter = int(year_text), int(quarter_text)
            else:
                observed = datetime.date.fromisoformat(normalized)
                year, quarter = observed.year, ((observed.month - 1) // 3) + 1
            month = quarter * 3
            return datetime.date(year, month, calendar.monthrange(year, month)[1])
        if frequency == MarketDataFrequency.SEMIANNUAL:
            match = re.fullmatch(r"(\d{4})-?[SH](\d)", normalized)
            if match:
                year, half = int(match.group(1)), int(match.group(2))
            else:
                observed = datetime.date.fromisoformat(normalized)
                year, half = observed.year, 1 if observed.month <= 6 else 2
            month = 6 if half == 1 else 12
            return datetime.date(year, month, calendar.monthrange(year, month)[1])
        if frequency == MarketDataFrequency.MONTHLY:
            if re.fullmatch(r"\d{4}-\d{2}", normalized):
                year, month = (int(part) for part in normalized.split("-"))
            else:
                observed = datetime.date.fromisoformat(normalized)
                year, month = observed.year, observed.month
            return datetime.date(year, month, calendar.monthrange(year, month)[1])
        return datetime.date.fromisoformat(normalized)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid {frequency.value} reference period: {value!r}"
        ) from exc


__all__ = [
    "CachedTextProvider",
    "RetrievedText",
    "frequency_from_code",
    "period_end_from_value",
]
