"""Adapt raw provider clients to the shared normalized financials contract."""

from dataclasses import dataclass
from typing import Optional, Protocol

from edgarito.enums.granularity import Granularity
from edgarito.enums.provider import ProviderName
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.services.normalization.alphavantage import AlphaVantageNormalizer
from edgarito.services.normalization.fmp import FmpNormalizer
from edgarito.services.normalization.sec_us_gaap import SecUsGaapNormalizer
from edgarito.services.providers.alphavantage import AlphaVantageClient
from edgarito.services.providers.edgar import EdgarClient
from edgarito.services.providers.fmp import FmpClient


@dataclass(frozen=True)
class FinancialsQuery:
    ticker: Optional[str] = None
    cik: Optional[int] = None
    granularity: Optional[Granularity] = Granularity.ANNUAL
    concepts: Optional[set[FinancialConcept]] = None
    use_cache: bool = True
    make_cache: bool = True


class NormalizedFinancialsProvider(Protocol):
    name: ProviderName

    async def retrieve(
        self, query: FinancialsQuery
    ) -> NormalizedCompanyFinancials: ...


class SecFinancialsProvider:
    name = ProviderName.SEC

    def __init__(
        self,
        client: EdgarClient,
        normalizer: Optional[SecUsGaapNormalizer] = None,
    ):
        self._client = client
        self._normalizer = normalizer or SecUsGaapNormalizer()

    async def retrieve(self, query: FinancialsQuery) -> NormalizedCompanyFinancials:
        ticker = query.ticker.upper() if query.ticker else None
        cik = query.cik
        if ticker is not None:
            cik = await self._client.get_cik(
                ticker,
                use_cache=query.use_cache,
                make_cache=query.make_cache,
            )
        if cik is None:
            raise ValueError("SEC retrieval requires a ticker or CIK")

        facts = await self._client.get_company_facts(
            cik,
            use_cache=query.use_cache,
            make_cache=query.make_cache,
        )
        return self._normalizer.normalize(
            facts,
            ticker=ticker,
            granularity=query.granularity,
            concepts=query.concepts,
        )


class AlphaVantageFinancialsProvider:
    name = ProviderName.ALPHAVANTAGE

    def __init__(
        self,
        client: AlphaVantageClient,
        normalizer: Optional[AlphaVantageNormalizer] = None,
    ):
        self._client = client
        self._normalizer = normalizer or AlphaVantageNormalizer()

    async def retrieve(self, query: FinancialsQuery) -> NormalizedCompanyFinancials:
        if not query.ticker:
            raise ValueError("Alpha Vantage retrieval requires a ticker")
        source = await self._client.get_company_financials(
            query.ticker,
            use_cache=query.use_cache,
            make_cache=query.make_cache,
        )
        return self._normalizer.normalize(
            source,
            granularity=query.granularity,
            concepts=query.concepts,
        )


class FmpFinancialsProvider:
    name = ProviderName.FMP

    def __init__(
        self,
        client: FmpClient,
        normalizer: Optional[FmpNormalizer] = None,
    ):
        self._client = client
        self._normalizer = normalizer or FmpNormalizer()

    async def retrieve(self, query: FinancialsQuery) -> NormalizedCompanyFinancials:
        if not query.ticker:
            raise ValueError("FMP retrieval requires a ticker")
        source = await self._client.get_company_financials(
            query.ticker,
            use_cache=query.use_cache,
            make_cache=query.make_cache,
        )
        return self._normalizer.normalize(
            source,
            granularity=query.granularity,
            concepts=query.concepts,
        )
