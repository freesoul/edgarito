"""Adapt raw provider clients to the shared normalized financials contract."""

from dataclasses import dataclass
from typing import Optional, Protocol

from edgarito.enums.granularity import Granularity
from edgarito.enums.provider import ProviderName
from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.services.normalization.alphavantage import AlphaVantageNormalizer
from edgarito.services.normalization.fmp import FmpNormalizer
from edgarito.services.normalization.sec_us_gaap import SecUsGaapNormalizer
from edgarito.services.normalization.yahoo import YahooFinancialsNormalizer
from edgarito.services.providers.alphavantage import AlphaVantageClient
from edgarito.services.providers.edgar import EdgarClient
from edgarito.services.providers.fmp import FmpClient
from edgarito.services.providers.yahoo import YahooFinanceClient


@dataclass(frozen=True)
class FinancialsQuery:
    ticker: Optional[str] = None
    cik: Optional[int] = None
    identifiers: Optional[SecurityIdentifiers] = None
    granularity: Optional[Granularity] = Granularity.ANNUAL
    concepts: Optional[set[FinancialConcept]] = None
    use_cache: bool = True
    make_cache: bool = True

    def __post_init__(self) -> None:
        if self.identifiers is None:
            if self.ticker is None and self.cik is None:
                raise ValueError("FinancialsQuery requires security identifiers")
            object.__setattr__(
                self,
                "identifiers",
                SecurityIdentifiers(ticker=self.ticker, cik=self.cik),
            )
        elif self.ticker is not None or self.cik is not None:
            raise ValueError("Use identifiers or ticker/cik, not both")

        object.__setattr__(self, "ticker", self.identifiers.ticker)
        object.__setattr__(self, "cik", self.identifiers.cik)

    def symbol_for(self, provider: ProviderName) -> Optional[str]:
        return self.identifiers.symbol_for(provider) if self.identifiers else None


class NormalizedFinancialsProvider(Protocol):
    name: ProviderName

    async def retrieve(self, query: FinancialsQuery) -> NormalizedCompanyFinancials: ...


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
        ticker = query.symbol_for(self.name)
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
        symbol = query.symbol_for(self.name)
        if not symbol:
            raise ValueError("Alpha Vantage retrieval requires a symbol")
        source = await self._client.get_company_financials(
            symbol,
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
        symbol = query.symbol_for(self.name)
        if not symbol:
            raise ValueError("FMP retrieval requires a symbol")
        source = await self._client.get_company_financials(
            symbol,
            use_cache=query.use_cache,
            make_cache=query.make_cache,
        )
        return self._normalizer.normalize(
            source,
            granularity=query.granularity,
            concepts=query.concepts,
        )


class YahooFinancialsProvider:
    name = ProviderName.YAHOO

    def __init__(
        self,
        client: YahooFinanceClient,
        normalizer: Optional[YahooFinancialsNormalizer] = None,
    ):
        self._client = client
        self._normalizer = normalizer or YahooFinancialsNormalizer()

    async def retrieve(self, query: FinancialsQuery) -> NormalizedCompanyFinancials:
        symbol = query.symbol_for(self.name)
        if not symbol:
            raise ValueError("Yahoo retrieval requires a symbol")
        source = await self._client.get_company_financials(
            symbol,
            use_cache=query.use_cache,
            make_cache=query.make_cache,
        )
        return self._normalizer.normalize(
            source,
            granularity=query.granularity,
            concepts=query.concepts,
        )
