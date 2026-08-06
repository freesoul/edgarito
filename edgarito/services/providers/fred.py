import datetime
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Optional

import aiohttp

from edgarito.schemas.market import (
    MarketDataFrequency,
    ReferenceMarketSeries,
    ReferenceObservation,
    ReferenceSeriesKind,
    ReferenceValueUnit,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.providers._reference import (
    CachedTextProvider,
    frequency_from_code,
    period_end_from_value,
)


class FredClient(CachedTextProvider):
    """Retrieve FRED series through the supported API using a free API key."""

    _BASE_URL = "https://api.stlouisfed.org/fred"
    _SERIES_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

    def __init__(
        self,
        cache: FileSystemCache,
        api_key: Optional[str],
        session: Optional[aiohttp.ClientSession] = None,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("A FRED API key is required")
        super().__init__(cache, session)
        self._api_key = api_key.strip()

    async def get_series(
        self,
        series_id: str,
        *,
        kind: ReferenceSeriesKind = ReferenceSeriesKind.OTHER,
        unit: Optional[ReferenceValueUnit] = None,
        observation_start: Optional[datetime.date] = None,
        observation_end: Optional[datetime.date] = None,
        vintage_date: Optional[datetime.date] = None,
        currency: Optional[str] = None,
        country: Optional[str] = None,
        region: Optional[str] = None,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> ReferenceMarketSeries:
        normalized_id = series_id.strip().upper()
        if not self._SERIES_PATTERN.fullmatch(normalized_id):
            raise ValueError(f"Invalid FRED series id: {series_id!r}")
        if (
            observation_start
            and observation_end
            and observation_start > observation_end
        ):
            raise ValueError("observation_start cannot be after observation_end")

        common_params = {
            "api_key": self._api_key,
            "file_type": "json",
            "series_id": normalized_id,
        }
        if vintage_date is not None:
            common_params["realtime_start"] = vintage_date.isoformat()
            common_params["realtime_end"] = vintage_date.isoformat()

        metadata_payload = await self._retrieve_text(
            provider="FRED",
            url=f"{self._BASE_URL}/series",
            cache_path=self._cache_path(normalized_id, "metadata", vintage_date),
            params=common_params,
            use_cache=use_cache,
            make_cache=make_cache,
        )
        observation_params = dict(common_params)
        if observation_start is not None:
            observation_params["observation_start"] = observation_start.isoformat()
        if observation_end is not None:
            observation_params["observation_end"] = observation_end.isoformat()
        range_key = (
            f"{observation_start.isoformat() if observation_start else 'first'}_"
            f"{observation_end.isoformat() if observation_end else 'latest'}"
        )
        observations_payload = await self._retrieve_text(
            provider="FRED",
            url=f"{self._BASE_URL}/series/observations",
            cache_path=self._cache_path(
                normalized_id, f"observations_{range_key}", vintage_date
            ),
            params=observation_params,
            use_cache=use_cache,
            make_cache=make_cache,
        )

        metadata = self._parse_metadata(metadata_payload.content, normalized_id)
        frequency = frequency_from_code(metadata["frequency_short"])
        observations = self._parse_observations(
            observations_payload.content, normalized_id, frequency
        )
        source_version = metadata.get("last_updated")
        if vintage_date is not None:
            source_version = f"vintage:{vintage_date.isoformat()}"
        return ReferenceMarketSeries(
            provider="fred",
            series_id=normalized_id,
            name=metadata["title"],
            kind=kind,
            unit=unit or self._unit_from_metadata(metadata.get("units", "")),
            frequency=frequency,
            retrieved_at=max(
                metadata_payload.retrieved_at, observations_payload.retrieved_at
            ),
            observations=observations,
            currency=currency,
            country=country,
            region=region,
            source_version=source_version,
        )

    @staticmethod
    def _cache_path(
        series_id: str, resource: str, vintage_date: Optional[datetime.date]
    ) -> str:
        vintage = vintage_date.isoformat() if vintage_date else "current"
        return f"providers/fred/{series_id}/{vintage}/{resource}.json"

    @staticmethod
    def _parse_json(content: str, resource: str) -> dict:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"FRED returned invalid JSON for {resource}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"FRED returned an invalid {resource} response")
        if data.get("error_message"):
            raise RuntimeError(f"FRED: {data['error_message']}")
        return data

    @classmethod
    def _parse_metadata(cls, content: str, series_id: str) -> dict:
        data = cls._parse_json(content, f"series {series_id}")
        series = data.get("seriess")
        if not isinstance(series, list) or len(series) != 1:
            raise RuntimeError(f"FRED returned no unique metadata for {series_id}")
        metadata = series[0]
        if not isinstance(metadata, dict) or not all(
            metadata.get(key) for key in ("title", "frequency_short")
        ):
            raise RuntimeError(f"FRED returned incomplete metadata for {series_id}")
        return metadata

    @classmethod
    def _parse_observations(
        cls,
        content: str,
        series_id: str,
        frequency: MarketDataFrequency,
    ) -> tuple[ReferenceObservation, ...]:
        data = cls._parse_json(content, f"observations {series_id}")
        raw_observations = data.get("observations")
        if not isinstance(raw_observations, list):
            raise RuntimeError(f"FRED returned invalid observations for {series_id}")
        observations = []
        for item in raw_observations:
            if not isinstance(item, dict) or item.get("value") in (None, "."):
                continue
            try:
                observations.append(
                    ReferenceObservation(
                        period_end=period_end_from_value(item["date"], frequency),
                        value=Decimal(item["value"]),
                    )
                )
            except (KeyError, ValueError, InvalidOperation) as exc:
                raise RuntimeError(
                    f"FRED returned an invalid observation for {series_id}"
                ) from exc
        if not observations:
            raise RuntimeError(f"FRED returned no observations for {series_id}")
        return tuple(observations)

    @staticmethod
    def _unit_from_metadata(units: str) -> ReferenceValueUnit:
        normalized = units.strip().casefold()
        if "percent change" in normalized:
            return ReferenceValueUnit.PERCENT_CHANGE
        if normalized in {"percent", "percentage points"}:
            return ReferenceValueUnit.PERCENTAGE_POINTS
        if "index" in normalized:
            return ReferenceValueUnit.INDEX_POINTS
        return ReferenceValueUnit.DECIMAL


__all__ = ["FredClient"]
