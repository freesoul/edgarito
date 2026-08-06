import datetime
import xml.etree.ElementTree as ElementTree
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
from edgarito.services.providers._reference import CachedTextProvider

_TENOR_FIELDS = {
    1: ("BC_1MONTH", "1-Month"),
    2: ("BC_2MONTH", "2-Month"),
    3: ("BC_3MONTH", "3-Month"),
    4: ("BC_4MONTH", "4-Month"),
    6: ("BC_6MONTH", "6-Month"),
    12: ("BC_1YEAR", "1-Year"),
    24: ("BC_2YEAR", "2-Year"),
    36: ("BC_3YEAR", "3-Year"),
    60: ("BC_5YEAR", "5-Year"),
    84: ("BC_7YEAR", "7-Year"),
    120: ("BC_10YEAR", "10-Year"),
    240: ("BC_20YEAR", "20-Year"),
    360: ("BC_30YEAR", "30-Year"),
}


class TreasuryClient(CachedTextProvider):
    """Retrieve free U.S. Treasury daily par yield-curve observations."""

    _URL = (
        "https://home.treasury.gov/resource-center/data-chart-center/"
        "interest-rates/pages/xml"
    )
    _ATOM = "http://www.w3.org/2005/Atom"
    _DATA = "http://schemas.microsoft.com/ado/2007/08/dataservices"
    _META = "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"

    def __init__(
        self,
        cache: FileSystemCache,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(cache, session)

    async def get_par_yield(
        self,
        tenor_months: int,
        year: Optional[int] = None,
        *,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> ReferenceMarketSeries:
        try:
            field, label = _TENOR_FIELDS[tenor_months]
        except KeyError as exc:
            supported = ", ".join(str(value) for value in _TENOR_FIELDS)
            raise ValueError(
                f"Unsupported Treasury tenor {tenor_months}; use months: {supported}"
            ) from exc
        selected_year = year or datetime.date.today().year
        if selected_year < 1990 or selected_year > 9999:
            raise ValueError("Treasury par yields are available from 1990")

        payload = await self._retrieve_text(
            provider="Treasury",
            url=self._URL,
            cache_path=(
                f"providers/treasury/daily_treasury_yield_curve/{selected_year}.json"
            ),
            params={
                "data": "daily_treasury_yield_curve",
                "field_tdr_date_value": str(selected_year),
            },
            use_cache=use_cache,
            make_cache=make_cache,
        )
        observations, feed_version = self._parse(payload.content, field)
        return ReferenceMarketSeries(
            provider="treasury",
            series_id=f"daily_treasury_yield_curve:{field}",
            name=f"{label} Treasury Par Yield",
            kind=ReferenceSeriesKind.GOVERNMENT_YIELD,
            unit=ReferenceValueUnit.PERCENTAGE_POINTS,
            frequency=MarketDataFrequency.DAILY,
            retrieved_at=payload.retrieved_at,
            observations=observations,
            currency="USD",
            country="US",
            tenor_months=tenor_months,
            source_version=feed_version or payload.source_version,
        )

    @classmethod
    def _parse(
        cls, content: str, field: str
    ) -> tuple[tuple[ReferenceObservation, ...], Optional[str]]:
        namespaces = {"atom": cls._ATOM, "d": cls._DATA, "m": cls._META}
        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError as exc:
            raise RuntimeError("Treasury returned invalid XML") from exc
        observations = []
        for entry in root.findall("atom:entry", namespaces):
            properties = entry.find("atom:content/m:properties", namespaces)
            if properties is None:
                continue
            date_element = properties.find("d:NEW_DATE", namespaces)
            value_element = properties.find(f"d:{field}", namespaces)
            if (
                date_element is None
                or value_element is None
                or not date_element.text
                or not value_element.text
            ):
                continue
            try:
                period_end = datetime.date.fromisoformat(
                    date_element.text.split("T", 1)[0]
                )
                value = Decimal(value_element.text)
            except (ValueError, InvalidOperation) as exc:
                raise RuntimeError(
                    "Treasury returned an invalid yield observation"
                ) from exc
            observations.append(
                ReferenceObservation(
                    period_end=period_end,
                    available_on=period_end,
                    value=value,
                )
            )
        if not observations:
            raise RuntimeError(f"Treasury returned no observations for {field}")
        updated = root.find("atom:updated", namespaces)
        return tuple(observations), updated.text if updated is not None else None


__all__ = ["TreasuryClient"]
