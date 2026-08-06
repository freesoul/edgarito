import re
from typing import Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgarito.enums.provider import ProviderName

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._^-]*$")
_ISIN_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


class SecurityIdentifiers(BaseModel):
    """Provider-neutral identifiers and symbols for one listed security.

    ``ticker`` is the preferred display symbol. Provider and exchange mappings are
    intentionally explicit because the same security can use different symbol
    syntax in different data sources and on different exchanges.
    """

    model_config = ConfigDict(frozen=True)

    ticker: Optional[str] = None
    isin: Optional[str] = None
    cik: Optional[int] = None
    exchange: Optional[str] = None
    exchange_symbols: dict[str, str] = Field(default_factory=dict)
    provider_symbols: dict[ProviderName, str] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: Optional[str]) -> Optional[str]:
        return cls._normalize_symbol(value, "ticker") if value is not None else None

    @field_validator("isin")
    @classmethod
    def normalize_isin(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not _ISIN_PATTERN.fullmatch(normalized) or not cls._valid_isin_checksum(
            normalized
        ):
            raise ValueError(f"Invalid ISIN: {value!r}")
        return normalized

    @field_validator("cik")
    @classmethod
    def validate_cik(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and not 0 < value <= 9_999_999_999:
            raise ValueError("CIK must contain between 1 and 10 digits")
        return value

    @field_validator("exchange")
    @classmethod
    def normalize_exchange(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("Exchange cannot be empty")
        return normalized

    @field_validator("exchange_symbols", mode="before")
    @classmethod
    def normalize_exchange_symbols(
        cls, value: Optional[Mapping[str, str]]
    ) -> dict[str, str]:
        return {
            cls._normalize_mapping_key(exchange, "exchange"): cls._normalize_symbol(
                symbol, f"symbol for exchange {exchange!r}"
            )
            for exchange, symbol in (value or {}).items()
        }

    @field_validator("provider_symbols", mode="before")
    @classmethod
    def normalize_provider_symbols(
        cls, value: Optional[Mapping[ProviderName | str, str]]
    ) -> dict[ProviderName, str]:
        return {
            ProviderName(provider): cls._normalize_symbol(
                symbol, f"symbol for provider {provider!r}"
            )
            for provider, symbol in (value or {}).items()
        }

    @model_validator(mode="after")
    def require_identifier(self) -> "SecurityIdentifiers":
        if not any(
            (
                self.ticker,
                self.isin,
                self.cik,
                self.exchange_symbols,
                self.provider_symbols,
            )
        ):
            raise ValueError("Provide at least one security identifier")
        if (
            self.exchange
            and self.exchange_symbols
            and self.exchange not in self.exchange_symbols
        ):
            raise ValueError(
                f"Exchange {self.exchange!r} has no entry in exchange_symbols"
            )
        return self

    def symbol_for(self, provider: ProviderName | str) -> Optional[str]:
        """Return the best known symbol for a provider without doing I/O."""
        provider_name = ProviderName(provider)
        if provider_name in self.provider_symbols:
            return self.provider_symbols[provider_name]
        if self.exchange and self.exchange in self.exchange_symbols:
            return self.exchange_symbols[self.exchange]
        if not self.exchange and len(self.exchange_symbols) == 1:
            return next(iter(self.exchange_symbols.values()))
        return self.ticker

    def with_updates(self, **updates) -> "SecurityIdentifiers":
        """Return a copy enriched with identifiers learned during resolution."""
        values = self.model_dump()
        provider_symbols = dict(self.provider_symbols)
        provider_symbols.update(updates.pop("provider_symbols", {}))
        exchange_symbols = dict(self.exchange_symbols)
        exchange_symbols.update(updates.pop("exchange_symbols", {}))
        values.update(updates)
        values["provider_symbols"] = provider_symbols
        values["exchange_symbols"] = exchange_symbols
        return SecurityIdentifiers.model_validate(values)

    @staticmethod
    def _normalize_mapping_key(value: str, label: str) -> str:
        normalized = str(value).strip().upper()
        if not normalized:
            raise ValueError(f"{label.title()} cannot be empty")
        return normalized

    @staticmethod
    def _normalize_symbol(value: str, label: str) -> str:
        normalized = str(value).strip().upper()
        if not _SYMBOL_PATTERN.fullmatch(normalized):
            raise ValueError(f"Invalid {label}: {value!r}")
        return normalized

    @staticmethod
    def _valid_isin_checksum(isin: str) -> bool:
        digits = "".join(
            str(int(character, 36)) if character.isalpha() else character
            for character in isin
        )
        total = 0
        for index, character in enumerate(reversed(digits)):
            digit = int(character)
            if index % 2 == 1:
                digit *= 2
            total += digit // 10 + digit % 10
        return total % 10 == 0
