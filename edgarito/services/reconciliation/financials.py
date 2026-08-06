import warnings
from typing import Mapping, Optional

from edgarito.config.providers import ProviderConfiguration
from edgarito.enums.granularity import Granularity
from edgarito.enums.market import Market
from edgarito.enums.provider import ProviderName
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.providers.alphavantage import AlphaVantageClient
from edgarito.services.providers.edgar import EdgarClient
from edgarito.services.providers.fmp import FmpClient
from edgarito.services.reconciliation.crosscheck import (
    CrosscheckReport,
    FinancialDataCrosscheckWarning,
    FinancialsCrosschecker,
)
from edgarito.services.reconciliation.providers import (
    AlphaVantageFinancialsProvider,
    FinancialsQuery,
    FmpFinancialsProvider,
    NormalizedFinancialsProvider,
    SecFinancialsProvider,
)


class FinancialDataService:
    """Retrieve normalized financials and optionally crosscheck other providers."""

    def __init__(
        self,
        cache: FileSystemCache,
        provider_configuration: ProviderConfiguration,
        user_agent: Optional[str] = None,
        alphavantage_api_key: Optional[str] = None,
        fmp_api_key: Optional[str] = None,
        providers: Optional[Mapping[ProviderName, NormalizedFinancialsProvider]] = None,
        crosschecker: Optional[FinancialsCrosschecker] = None,
    ):
        self._cache = cache
        self._configuration = provider_configuration
        self._user_agent = user_agent
        self._alphavantage_api_key = alphavantage_api_key
        self._fmp_api_key = fmp_api_key
        self._providers = dict(providers or {})
        self._crosschecker = crosschecker or FinancialsCrosschecker()
        self._owned_clients = []
        self.last_crosschecks: list[CrosscheckReport] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        for client in reversed(self._owned_clients):
            await client.__aexit__(exc_type, exc, tb)
        self._owned_clients.clear()

    async def retrieve(
        self,
        *,
        ticker: Optional[str] = None,
        cik: Optional[int] = None,
        market: Market = Market.US,
        provider: Optional[ProviderName] = None,
        granularity: Optional[Granularity] = Granularity.ANNUAL,
        concepts: Optional[set[FinancialConcept]] = None,
        use_cache: bool = True,
        make_cache: bool = True,
        crosscheck: bool = True,
    ) -> NormalizedCompanyFinancials:
        market = Market(market)
        provider = ProviderName(provider) if provider is not None else None
        if (ticker is None) == (cik is None):
            raise ValueError("Provide exactly one of ticker or cik")
        if cik is not None and market != Market.US:
            raise ValueError("CIK identifiers are only supported for US stocks")

        market_configuration = self._configuration.for_market(market)
        selected_provider = provider or market_configuration.default_provider
        if selected_provider not in market_configuration.available_providers:
            raise ValueError(
                f"Provider '{selected_provider.value}' is not available for {market.value}"
            )
        if cik is not None and selected_provider != ProviderName.SEC:
            raise ValueError(
                f"Provider '{selected_provider.value}' requires a ticker, not a CIK"
            )

        query = FinancialsQuery(
            ticker=ticker,
            cik=cik,
            granularity=granularity,
            concepts=concepts,
            use_cache=use_cache,
            make_cache=make_cache,
        )
        self.last_crosschecks = []
        primary_provider = self._provider(selected_provider)
        primary = await primary_provider.retrieve(query)

        if crosscheck:
            await self._crosscheck_available_providers(
                primary,
                query,
                selected_provider,
                market_configuration.available_providers,
            )
        return primary

    async def _crosscheck_available_providers(
        self,
        primary: NormalizedCompanyFinancials,
        query: FinancialsQuery,
        selected_provider: ProviderName,
        available_providers: tuple[ProviderName, ...],
    ) -> None:
        ticker = query.ticker or primary.ticker
        for provider_name in available_providers:
            if provider_name == selected_provider:
                continue
            if not ticker:
                warnings.warn(
                    f"Crosscheck with {provider_name.value} skipped: no ticker is available",
                    FinancialDataCrosscheckWarning,
                    stacklevel=3,
                )
                continue

            crosscheck_query = FinancialsQuery(
                ticker=ticker,
                granularity=query.granularity,
                concepts=query.concepts,
                use_cache=query.use_cache,
                make_cache=query.make_cache,
            )
            try:
                secondary = await self._provider(provider_name).retrieve(
                    crosscheck_query
                )
                report = self._crosschecker.compare(primary, secondary)
                self.last_crosschecks.append(report)
                if report.has_issues:
                    warnings.warn(
                        report.summary(),
                        FinancialDataCrosscheckWarning,
                        stacklevel=3,
                    )
            except Exception as exc:
                warnings.warn(
                    f"Crosscheck with {provider_name.value} failed: {exc}",
                    FinancialDataCrosscheckWarning,
                    stacklevel=3,
                )

    def _provider(self, name: ProviderName) -> NormalizedFinancialsProvider:
        existing = self._providers.get(name)
        if existing is not None:
            return existing

        if name == ProviderName.SEC:
            if not self._user_agent:
                raise ValueError(
                    "The SEC provider requires EDGARITO_USER_AGENT / user_agent"
                )
            client = EdgarClient(self._cache, self._user_agent)
            provider = SecFinancialsProvider(client)
        elif name == ProviderName.ALPHAVANTAGE:
            if not self._alphavantage_api_key:
                raise ValueError(
                    "The Alpha Vantage provider requires ALPHAVANTAGE_API_KEY / "
                    "alphavantage_api_key"
                )
            client = AlphaVantageClient(self._cache, self._alphavantage_api_key)
            provider = AlphaVantageFinancialsProvider(client)
        elif name == ProviderName.FMP:
            if not self._fmp_api_key:
                raise ValueError("The FMP provider requires FMP_API_KEY / fmp_api_key")
            client = FmpClient(self._cache, self._fmp_api_key)
            provider = FmpFinancialsProvider(client)
        else:
            raise ValueError(f"Unsupported provider: {name.value}")

        self._owned_clients.append(client)
        self._providers[name] = provider
        return provider
