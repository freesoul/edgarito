import csv
import datetime
import io
import re
from decimal import Decimal, InvalidOperation
from typing import Optional

import aiohttp

from edgarito.schemas.market import (
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


class EcbClient(CachedTextProvider):
    """Retrieve provider-neutral series from the free ECB SDMX data API."""

    _BASE_URL = "https://data-api.ecb.europa.eu/service/data"
    _PATH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.,_+@-]*$")

    def __init__(
        self,
        cache: FileSystemCache,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(cache, session)

    async def get_series(
        self,
        flow_ref: str,
        key: str,
        *,
        kind: ReferenceSeriesKind = ReferenceSeriesKind.OTHER,
        unit: Optional[ReferenceValueUnit] = None,
        start_period: Optional[datetime.date] = None,
        end_period: Optional[datetime.date] = None,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> ReferenceMarketSeries:
        normalized_flow = self._normalize_path_part(flow_ref, "flow reference")
        normalized_key = self._normalize_path_part(key, "series key")
        if start_period and end_period and start_period > end_period:
            raise ValueError("start_period cannot be after end_period")
        params = {"format": "csvdata"}
        if start_period is not None:
            params["startPeriod"] = start_period.isoformat()
        if end_period is not None:
            params["endPeriod"] = end_period.isoformat()
        range_key = (
            f"{start_period.isoformat() if start_period else 'first'}_"
            f"{end_period.isoformat() if end_period else 'latest'}"
        )
        payload = await self._retrieve_text(
            provider="ECB",
            url=f"{self._BASE_URL}/{normalized_flow}/{normalized_key}",
            cache_path=(
                f"providers/ecb/{normalized_flow}/{normalized_key}/{range_key}.json"
            ),
            params=params,
            headers={"Accept": "text/csv"},
            use_cache=use_cache,
            make_cache=make_cache,
        )
        metadata, observations = self._parse(payload.content)
        currency = metadata.get("CURRENCY") or None
        region = metadata.get("REF_AREA") or None
        return ReferenceMarketSeries(
            provider="ecb",
            series_id=f"{normalized_flow}/{normalized_key}",
            name=metadata.get("TITLE") or metadata.get("TITLE_COMPL") or normalized_key,
            kind=kind,
            unit=unit or self._unit_from_code(metadata.get("UNIT", "")),
            frequency=frequency_from_code(metadata["FREQ"]),
            retrieved_at=payload.retrieved_at,
            observations=observations,
            currency=currency,
            region=region,
            source_version=payload.source_version,
        )

    @classmethod
    def _normalize_path_part(cls, value: str, name: str) -> str:
        normalized = value.strip()
        if not cls._PATH_PATTERN.fullmatch(normalized):
            raise ValueError(f"Invalid ECB {name}: {value!r}")
        return normalized

    @staticmethod
    def _parse(
        content: str,
    ) -> tuple[dict[str, str], tuple[ReferenceObservation, ...]]:
        try:
            rows = list(csv.DictReader(io.StringIO(content)))
        except csv.Error as exc:
            raise RuntimeError("ECB returned invalid CSV") from exc
        if not rows or not rows[0].get("KEY"):
            raise RuntimeError("ECB returned no series observations")
        series_keys = {row.get("KEY") for row in rows}
        if len(series_keys) != 1:
            raise RuntimeError("ECB query returned more than one series")
        metadata = rows[0]
        if not metadata.get("FREQ"):
            raise RuntimeError("ECB response is missing series frequency")
        frequency = frequency_from_code(metadata["FREQ"])
        observations = []
        for row in rows:
            value = row.get("OBS_VALUE")
            if not value:
                continue
            try:
                observations.append(
                    ReferenceObservation(
                        period_end=period_end_from_value(row["TIME_PERIOD"], frequency),
                        value=Decimal(value),
                    )
                )
            except (KeyError, ValueError, InvalidOperation) as exc:
                raise RuntimeError("ECB returned an invalid observation") from exc
        if not observations:
            raise RuntimeError("ECB returned no usable observations")
        return metadata, tuple(observations)

    @staticmethod
    def _unit_from_code(code: str) -> ReferenceValueUnit:
        normalized = code.strip().upper()
        if normalized in {"PC", "PCPA", "PCT"}:
            return ReferenceValueUnit.PERCENTAGE_POINTS
        if "INDEX" in normalized or normalized in {"I15", "I21"}:
            return ReferenceValueUnit.INDEX_POINTS
        return ReferenceValueUnit.DECIMAL


__all__ = ["EcbClient"]
