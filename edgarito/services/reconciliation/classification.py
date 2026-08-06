import re
import warnings
from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol

from edgarito.config.providers import ClassificationProviderConfiguration
from edgarito.enums.provider import ProviderName
from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.normalization.classification import (
    NormalizedCompanyClassification,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.identifiers import SecurityIdentifierResolver
from edgarito.services.normalization.classification import (
    CompanyClassificationNormalizer,
)
from edgarito.services.providers.alphavantage import AlphaVantageClient
from edgarito.services.providers.fmp import FmpClient
from edgarito.services.providers.openfigi import OpenFigiClient


class ClassificationProvider(Protocol):
    name: ProviderName

    async def retrieve(
        self, ticker: str, use_cache: bool, make_cache: bool
    ) -> NormalizedCompanyClassification: ...


class AlphaVantageClassificationProvider:
    name = ProviderName.ALPHAVANTAGE

    def __init__(self, client: AlphaVantageClient):
        self._client = client
        self._normalizer = CompanyClassificationNormalizer()

    async def retrieve(
        self, ticker: str, use_cache: bool = True, make_cache: bool = True
    ) -> NormalizedCompanyClassification:
        overview = await self._client.get_overview(ticker, use_cache, make_cache)
        return self._normalizer.normalize_alphavantage(overview)


class FmpClassificationProvider:
    name = ProviderName.FMP

    def __init__(self, client: FmpClient):
        self._client = client
        self._normalizer = CompanyClassificationNormalizer()

    async def retrieve(
        self, ticker: str, use_cache: bool = True, make_cache: bool = True
    ) -> NormalizedCompanyClassification:
        profile = await self._client.get_profile(ticker, use_cache, make_cache)
        return self._normalizer.normalize_fmp(profile)


@dataclass
class ClassificationCrosscheckReport:
    primary_provider: str
    secondary_provider: str
    differences: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.differences)

    def summary(self) -> str:
        details = "; ".join(self.differences)
        return (
            f"Classification crosscheck {self.primary_provider} vs "
            f"{self.secondary_provider}: {details}"
        )


class ClassificationCrosscheckWarning(UserWarning):
    pass


class CompanyClassificationService:
    def __init__(
        self,
        cache: FileSystemCache,
        provider_configuration: ClassificationProviderConfiguration,
        alphavantage_api_key: Optional[str] = None,
        fmp_api_key: Optional[str] = None,
        providers: Optional[Mapping[ProviderName, ClassificationProvider]] = None,
        identifier_resolver: Optional[SecurityIdentifierResolver] = None,
        openfigi_api_key: Optional[str] = None,
    ):
        self._cache = cache
        self._configuration = provider_configuration
        self._alphavantage_api_key = alphavantage_api_key
        self._fmp_api_key = fmp_api_key
        self._openfigi_api_key = openfigi_api_key
        self._providers = dict(providers or {})
        self._identifier_resolver = identifier_resolver
        self._owned_clients = []
        self.last_crosschecks: list[ClassificationCrosscheckReport] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        for client in reversed(self._owned_clients):
            await client.__aexit__(exc_type, exc, tb)
        self._owned_clients.clear()

    async def retrieve(
        self,
        ticker: Optional[str] = None,
        *,
        cik: Optional[int] = None,
        isin: Optional[str] = None,
        exchange: Optional[str] = None,
        exchange_symbols: Optional[Mapping[str, str]] = None,
        provider_symbols: Optional[Mapping[ProviderName | str, str]] = None,
        identifiers: Optional[SecurityIdentifiers] = None,
        provider: Optional[ProviderName] = None,
        use_cache: bool = True,
        make_cache: bool = True,
        crosscheck: bool = True,
    ) -> NormalizedCompanyClassification:
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
        selected = provider or self._configuration.default_provider
        if selected not in self._configuration.available_providers:
            raise ValueError(
                f"Classification provider '{selected.value}' is not available"
            )
        self.last_crosschecks = []
        identifiers = await self._resolve_identifiers(
            identifiers, selected, use_cache=use_cache, make_cache=make_cache
        )
        symbol = identifiers.symbol_for(selected)
        if not symbol:
            raise ValueError(
                f"No symbol is available for classification provider {selected.value}"
            )
        primary = await self._provider(selected).retrieve(symbol, use_cache, make_cache)
        identifiers = self._identifiers_from_result(identifiers, selected, primary)
        primary.identifiers = identifiers
        if crosscheck:
            for secondary_name in self._configuration.available_providers:
                if secondary_name == selected:
                    continue
                try:
                    secondary_identifiers = await self._resolve_identifiers(
                        identifiers,
                        secondary_name,
                        use_cache=use_cache,
                        make_cache=make_cache,
                    )
                    secondary_symbol = secondary_identifiers.symbol_for(secondary_name)
                    if not secondary_symbol:
                        raise ValueError(
                            f"No symbol is available for {secondary_name.value}"
                        )
                    secondary = await self._provider(secondary_name).retrieve(
                        secondary_symbol, use_cache, make_cache
                    )
                    secondary.identifiers = self._identifiers_from_result(
                        secondary_identifiers, secondary_name, secondary
                    )
                    report = self._compare(primary, secondary)
                    self.last_crosschecks.append(report)
                    if report.has_issues:
                        warnings.warn(
                            report.summary(),
                            ClassificationCrosscheckWarning,
                            stacklevel=2,
                        )
                except Exception as exc:
                    warnings.warn(
                        f"Classification crosscheck with {secondary_name.value} "
                        f"failed: {exc}",
                        ClassificationCrosscheckWarning,
                        stacklevel=2,
                    )
        return primary

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
        result: NormalizedCompanyClassification,
    ) -> SecurityIdentifiers:
        updates = {}
        if provider not in identifiers.provider_symbols:
            updates["provider_symbols"] = {provider: result.ticker}
        if identifiers.ticker is None:
            updates["ticker"] = result.ticker
        if identifiers.cik is None and result.company_id.isdigit():
            updates["cik"] = int(result.company_id)
        if identifiers.exchange is None and result.exchange:
            updates["exchange"] = result.exchange
            updates["exchange_symbols"] = {result.exchange: result.ticker}
        return identifiers.with_updates(**updates)

    def _provider(self, name: ProviderName) -> ClassificationProvider:
        existing = self._providers.get(name)
        if existing is not None:
            return existing
        if name == ProviderName.FMP:
            if not self._fmp_api_key:
                raise ValueError("The FMP provider requires FMP_API_KEY / fmp_key")
            client = FmpClient(self._cache, self._fmp_api_key)
            provider = FmpClassificationProvider(client)
        elif name == ProviderName.ALPHAVANTAGE:
            if not self._alphavantage_api_key:
                raise ValueError(
                    "The Alpha Vantage provider requires ALPHAVANTAGE_API_KEY / "
                    "alphavantage_api_key"
                )
            client = AlphaVantageClient(self._cache, self._alphavantage_api_key)
            provider = AlphaVantageClassificationProvider(client)
        else:
            raise ValueError(f"Unsupported classification provider: {name.value}")
        self._owned_clients.append(client)
        self._providers[name] = provider
        return provider

    @classmethod
    def _compare(
        cls,
        primary: NormalizedCompanyClassification,
        secondary: NormalizedCompanyClassification,
    ) -> ClassificationCrosscheckReport:
        report = ClassificationCrosscheckReport(primary.provider, secondary.provider)
        if primary.sector != secondary.sector:
            primary_sector = primary.sector.value if primary.sector else "-"
            secondary_sector = secondary.sector.value if secondary.sector else "-"
            report.differences.append(
                f"sector differs ({primary_sector} vs {secondary_sector})"
            )
        if cls._industry_key(primary.industry) != cls._industry_key(secondary.industry):
            report.differences.append(
                f"industry differs ({primary.industry or '-'} vs "
                f"{secondary.industry or '-'})"
            )
        return report

    @staticmethod
    def _industry_key(value: Optional[str]) -> Optional[str]:
        return re.sub(r"[^a-z0-9]", "", value.casefold()) if value else None
