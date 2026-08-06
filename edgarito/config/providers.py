from dataclasses import dataclass
from typing import Mapping, Optional

from edgarito.enums.market import Market
from edgarito.enums.provider import ProviderName


PROVIDER_MARKETS = {
    ProviderName.SEC: frozenset({Market.US}),
    ProviderName.ALPHAVANTAGE: frozenset({Market.US, Market.EU}),
    ProviderName.FMP: frozenset({Market.US, Market.EU}),
}


@dataclass(frozen=True)
class MarketProviderConfiguration:
    default_provider: ProviderName
    available_providers: tuple[ProviderName, ...]

    def __post_init__(self):
        if not self.available_providers:
            raise ValueError("At least one provider must be available")
        if self.default_provider not in self.available_providers:
            raise ValueError("The default provider must be listed as available")
        if len(set(self.available_providers)) != len(self.available_providers):
            raise ValueError("Available providers cannot contain duplicates")


@dataclass(frozen=True)
class ProviderConfiguration:
    us: MarketProviderConfiguration
    eu: MarketProviderConfiguration

    def __post_init__(self):
        for market in Market:
            market_configuration = self.for_market(market)
            unsupported = [
                provider
                for provider in market_configuration.available_providers
                if market not in PROVIDER_MARKETS[provider]
            ]
            if unsupported:
                names = ", ".join(provider.value for provider in unsupported)
                raise ValueError(f"Providers do not support {market.value}: {names}")

    def for_market(self, market: Market) -> MarketProviderConfiguration:
        return self.us if market == Market.US else self.eu

    @classmethod
    def from_environment(
        cls, environment: Optional[Mapping[str, str]] = None
    ) -> "ProviderConfiguration":
        values = environment or {}
        return cls(
            us=cls._market_from_environment(
                Market.US,
                values,
                default_provider=ProviderName.SEC,
                default_available=(
                    ProviderName.SEC,
                    ProviderName.ALPHAVANTAGE,
                    ProviderName.FMP,
                ),
            ),
            eu=cls._market_from_environment(
                Market.EU,
                values,
                default_provider=ProviderName.ALPHAVANTAGE,
                default_available=(ProviderName.ALPHAVANTAGE, ProviderName.FMP),
            ),
        )

    @staticmethod
    def _market_from_environment(
        market: Market,
        environment: Mapping[str, str],
        default_provider: ProviderName,
        default_available: tuple[ProviderName, ...],
    ) -> MarketProviderConfiguration:
        prefix = market.value.upper()
        default_value = ProviderConfiguration._environment_value(
            environment,
            f"EDGARITO_{prefix}_DEFAULT_PROVIDER",
            f"{market.value}_default_provider",
        )
        available_value = ProviderConfiguration._environment_value(
            environment,
            f"EDGARITO_{prefix}_AVAILABLE_PROVIDERS",
            f"{market.value}_available_providers",
        )
        selected_default = (
            ProviderName(default_value.strip().lower())
            if default_value
            else default_provider
        )
        available = (
            tuple(
                ProviderName(item.strip().lower())
                for item in available_value.split(",")
                if item.strip()
            )
            if available_value is not None
            else default_available
        )
        return MarketProviderConfiguration(selected_default, available)

    @staticmethod
    def _environment_value(
        environment: Mapping[str, str], canonical_name: str, legacy_name: str
    ) -> Optional[str]:
        return environment.get(canonical_name) or environment.get(legacy_name)
