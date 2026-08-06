import warnings
from typing import Mapping, Optional

from edgarito.config.providers import ProviderConfiguration
from edgarito.enums.granularity import Granularity
from edgarito.enums.market import Market
from edgarito.enums.provider import ProviderName
from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.identifiers import SecurityIdentifierResolver
from edgarito.services.providers.alphavantage import AlphaVantageClient
from edgarito.services.providers.edgar import EdgarClient
from edgarito.services.providers.fmp import FmpClient
from edgarito.services.providers.openfigi import OpenFigiClient
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
        identifier_resolver: Optional[SecurityIdentifierResolver] = None,
        openfigi_api_key: Optional[str] = None,
    ):
        self._cache = cache
        self._configuration = provider_configuration
        self._user_agent = user_agent
        self._alphavantage_api_key = alphavantage_api_key
        self._fmp_api_key = fmp_api_key
        self._openfigi_api_key = openfigi_api_key
        self._providers = dict(providers or {})
        self._crosschecker = crosschecker or FinancialsCrosschecker()
        self._identifier_resolver = identifier_resolver
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
        isin: Optional[str] = None,
        exchange: Optional[str] = None,
        exchange_symbols: Optional[Mapping[str, str]] = None,
        provider_symbols: Optional[Mapping[ProviderName | str, str]] = None,
        identifiers: Optional[SecurityIdentifiers] = None,
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
        supplied_fields = any(
            (ticker, cik, isin, exchange, exchange_symbols, provider_symbols)
        )
        if identifiers is not None and supplied_fields:
            raise ValueError(
                "Use identifiers or individual identifier arguments, not both"
            )
        if identifiers is None:
            identifiers = SecurityIdentifiers(
                ticker=ticker,
                cik=cik,
                isin=isin,
                exchange=exchange,
                exchange_symbols=dict(exchange_symbols or {}),
                provider_symbols=dict(provider_symbols or {}),
            )

        market_configuration = self._configuration.for_market(market)
        selected_provider = provider or market_configuration.default_provider
        if selected_provider not in market_configuration.available_providers:
            raise ValueError(
                f"Provider '{selected_provider.value}' is not available for {market.value}"
            )
        identifiers = await self._resolve_identifiers(
            identifiers,
            selected_provider,
            use_cache=use_cache,
            make_cache=make_cache,
        )

        query = FinancialsQuery(
            identifiers=identifiers,
            granularity=granularity,
            concepts=concepts,
            use_cache=use_cache,
            make_cache=make_cache,
        )
        self.last_crosschecks = []
        primary_provider = self._provider(selected_provider)
        primary = await primary_provider.retrieve(query)
        primary.identifiers = self._identifiers_from_result(
            identifiers, selected_provider, primary
        )

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
        for provider_name in available_providers:
            if provider_name == selected_provider:
                continue
            identifiers = primary.identifiers or query.identifiers
            try:
                identifiers = await self._resolve_identifiers(
                    identifiers,
                    provider_name,
                    use_cache=query.use_cache,
                    make_cache=query.make_cache,
                )
            except Exception as exc:
                warnings.warn(
                    f"Crosscheck with {provider_name.value} skipped: {exc}",
                    FinancialDataCrosscheckWarning,
                    stacklevel=3,
                )
                continue

            crosscheck_query = FinancialsQuery(
                identifiers=identifiers,
                granularity=query.granularity,
                concepts=query.concepts,
                use_cache=query.use_cache,
                make_cache=query.make_cache,
            )
            try:
                secondary = await self._provider(provider_name).retrieve(
                    crosscheck_query
                )
                secondary.identifiers = self._identifiers_from_result(
                    identifiers, provider_name, secondary
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

    async def _resolve_identifiers(
        self,
        identifiers: SecurityIdentifiers,
        provider: ProviderName,
        *,
        use_cache: bool,
        make_cache: bool,
    ) -> SecurityIdentifiers:
        if self._identifier_resolver is None:
            search_client = None
            if self._fmp_api_key:
                search_client = FmpClient(self._cache, self._fmp_api_key)
                self._owned_clients.append(search_client)
            isin_search_client = OpenFigiClient(
                self._cache, api_key=self._openfigi_api_key
            )
            self._owned_clients.append(isin_search_client)
            self._identifier_resolver = SecurityIdentifierResolver(
                search_client, isin_search_client
            )
        return await self._identifier_resolver.resolve(
            identifiers,
            provider,
            use_cache=use_cache,
            make_cache=make_cache,
        )

    @staticmethod
    def _identifiers_from_result(
        identifiers: SecurityIdentifiers,
        provider: ProviderName,
        result: NormalizedCompanyFinancials,
    ) -> SecurityIdentifiers:
        updates = {}
        if result.ticker:
            if provider not in identifiers.provider_symbols:
                updates["provider_symbols"] = {provider: result.ticker}
            if identifiers.ticker is None:
                updates["ticker"] = result.ticker
        if identifiers.cik is None and result.company_id.isdigit():
            updates["cik"] = int(result.company_id)
        return identifiers.with_updates(**updates) if updates else identifiers

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
