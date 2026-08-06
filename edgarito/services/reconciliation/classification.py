import re
import warnings
from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol

from edgarito.config.providers import ClassificationProviderConfiguration
from edgarito.enums.provider import ProviderName
from edgarito.schemas.normalization.classification import (
    NormalizedCompanyClassification,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization.classification import (
    CompanyClassificationNormalizer,
)
from edgarito.services.providers.alphavantage import AlphaVantageClient
from edgarito.services.providers.fmp import FmpClient


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
    ):
        self._cache = cache
        self._configuration = provider_configuration
        self._alphavantage_api_key = alphavantage_api_key
        self._fmp_api_key = fmp_api_key
        self._providers = dict(providers or {})
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
        ticker: str,
        provider: Optional[ProviderName] = None,
        use_cache: bool = True,
        make_cache: bool = True,
        crosscheck: bool = True,
    ) -> NormalizedCompanyClassification:
        selected = provider or self._configuration.default_provider
        if selected not in self._configuration.available_providers:
            raise ValueError(
                f"Classification provider '{selected.value}' is not available"
            )
        self.last_crosschecks = []
        primary = await self._provider(selected).retrieve(ticker, use_cache, make_cache)
        if crosscheck:
            for secondary_name in self._configuration.available_providers:
                if secondary_name == selected:
                    continue
                try:
                    secondary = await self._provider(secondary_name).retrieve(
                        ticker, use_cache, make_cache
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
