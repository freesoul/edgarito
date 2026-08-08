import datetime
import warnings
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from edgarito.config.providers import ProviderConfiguration
from edgarito.enums.granularity import Granularity
from edgarito.enums.market import Market
from edgarito.enums.provider import ProviderName
from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.identifiers import SecurityIdentifierResolver
from edgarito.services.providers.alphavantage import AlphaVantageClient
from edgarito.services.providers.edgar import EdgarClient
from edgarito.services.providers.fmp import FmpClient
from edgarito.services.providers.openfigi import OpenFigiClient
from edgarito.services.providers.yahoo import YahooFinanceClient
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
    YahooFinancialsProvider,
)


@dataclass(frozen=True)
class FinancialDataScore:
    """Quality dimensions used to choose one provider result.

    Completeness is the proportion of the observations available from any
    candidate that this candidate supplies.  It is intentionally calculated
    over observation keys rather than values: automatic selection must not
    reconcile, average, or otherwise combine provider data.
    """

    completeness: float
    latest_period_end: Optional[datetime.date]
    latest_filed: Optional[datetime.date]
    retrieved_at: Optional[datetime.datetime]
    observation_count: int

    @property
    def ranking_key(self) -> tuple[float, int, int, float, int]:
        """Return a deterministic key with completeness ahead of freshness."""
        return (
            self.completeness,
            self.latest_period_end.toordinal()
            if self.latest_period_end is not None
            else datetime.date.min.toordinal(),
            self.latest_filed.toordinal()
            if self.latest_filed is not None
            else datetime.date.min.toordinal(),
            self._retrieved_timestamp(),
            self.observation_count,
        )

    def _retrieved_timestamp(self) -> float:
        if self.retrieved_at is None:
            return float("-inf")
        return self.retrieved_at.astimezone(datetime.timezone.utc).timestamp()


class FinancialDataSelector:
    """Rank complete provider datasets without reconciling their observations."""

    @classmethod
    def rank(
        cls,
        candidates: Sequence[NormalizedCompanyFinancials],
        *,
        concepts: Optional[set[FinancialConcept]] = None,
    ) -> list[tuple[NormalizedCompanyFinancials, FinancialDataScore]]:
        candidates = list(candidates)
        if not candidates:
            raise ValueError("At least one financial data candidate is required")

        candidate_keys = [
            cls._observation_keys(candidate, concepts) for candidate in candidates
        ]
        expected_keys = set().union(*candidate_keys)
        ranked = [
            (
                candidate,
                cls.score(
                    candidate,
                    expected_keys=expected_keys,
                    concepts=concepts,
                ),
            )
            for candidate in candidates
        ]
        return sorted(
            ranked,
            key=lambda item: item[1].ranking_key,
            reverse=True,
        )

    @classmethod
    def select(
        cls,
        candidates: Sequence[NormalizedCompanyFinancials],
        *,
        concepts: Optional[set[FinancialConcept]] = None,
    ) -> NormalizedCompanyFinancials:
        """Return the best complete dataset, leaving every dataset untouched."""
        return cls.rank(candidates, concepts=concepts)[0][0]

    @classmethod
    def score(
        cls,
        financials: NormalizedCompanyFinancials,
        *,
        expected_keys: Optional[set[tuple]] = None,
        concepts: Optional[set[FinancialConcept]] = None,
    ) -> FinancialDataScore:
        keys = cls._observation_keys(financials, concepts)
        expected_count = len(expected_keys) if expected_keys is not None else len(keys)
        completeness = (
            len(keys & (expected_keys or keys)) / expected_count
            if expected_count
            else 0.0
        )
        observations = cls._observations(financials, concepts)
        latest_period_end = max(
            (observation.period_end for observation in observations),
            default=None,
        )
        latest_filed = max(
            (
                observation.filed
                for observation in observations
                if observation.filed is not None
            ),
            default=None,
        )
        return FinancialDataScore(
            completeness=completeness,
            latest_period_end=latest_period_end,
            latest_filed=latest_filed,
            retrieved_at=financials.retrieved_at,
            observation_count=len(keys),
        )

    @staticmethod
    def _observations(
        financials: NormalizedCompanyFinancials,
        concepts: Optional[set[FinancialConcept]],
    ) -> list[FinancialObservation]:
        return [
            observation
            for observation in financials.observations
            if not concepts or observation.concept in concepts
        ]

    @classmethod
    def _observation_keys(
        cls,
        financials: NormalizedCompanyFinancials,
        concepts: Optional[set[FinancialConcept]],
    ) -> set[tuple]:
        return {
            (
                observation.concept,
                observation.granularity,
                observation.fiscal_year,
                observation.fiscal_period,
            )
            for observation in cls._observations(financials, concepts)
        }


class FinancialDataService:
    """Retrieve normalized financials and optionally crosscheck other providers.

    An omitted provider uses all configured providers to select one dataset.  An
    explicit provider retains the original primary-plus-crosscheck behavior.
    """

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
        self.last_selection_failures: list[tuple[ProviderName, str]] = []

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
        self.last_crosschecks = []
        self.last_selection_failures = []

        if provider is None:
            return await self._retrieve_best_available(
                identifiers=identifiers,
                granularity=granularity,
                concepts=concepts,
                use_cache=use_cache,
                make_cache=make_cache,
                available_providers=market_configuration.available_providers,
            )

        selected_provider = provider
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

    async def _retrieve_best_available(
        self,
        *,
        identifiers: SecurityIdentifiers,
        granularity: Optional[Granularity],
        concepts: Optional[set[FinancialConcept]],
        use_cache: bool,
        make_cache: bool,
        available_providers: tuple[ProviderName, ...],
    ) -> NormalizedCompanyFinancials:
        """Retrieve every configured provider and return one unmodified result."""
        candidates: list[NormalizedCompanyFinancials] = []
        failures: list[tuple[ProviderName, str]] = []

        for provider_name in available_providers:
            try:
                provider_identifiers = await self._resolve_identifiers(
                    identifiers,
                    provider_name,
                    use_cache=use_cache,
                    make_cache=make_cache,
                )
                query = FinancialsQuery(
                    identifiers=provider_identifiers,
                    granularity=granularity,
                    concepts=concepts,
                    use_cache=use_cache,
                    make_cache=make_cache,
                )
                result = await self._provider(provider_name).retrieve(query)
                result.identifiers = self._identifiers_from_result(
                    provider_identifiers, provider_name, result
                )
                if not FinancialDataSelector._observation_keys(result, concepts):
                    raise ValueError("provider returned no usable observations")
                candidates.append(result)
            except Exception as exc:
                failures.append((provider_name, str(exc)))

        self.last_selection_failures = failures
        if not candidates:
            details = "; ".join(
                f"{provider_name.value}: {message}"
                for provider_name, message in failures
            )
            raise ValueError(
                "No configured financial data provider returned usable data"
                + (f" ({details})" if details else "")
            )

        return FinancialDataSelector.select(candidates, concepts=concepts)

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
        elif name == ProviderName.YAHOO:
            client = YahooFinanceClient(self._cache)
            provider = YahooFinancialsProvider(client)
        else:
            raise ValueError(f"Unsupported provider: {name.value}")

        self._owned_clients.append(client)
        self._providers[name] = provider
        return provider
