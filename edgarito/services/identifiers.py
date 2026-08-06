import re
from typing import Optional, Protocol

from edgarito.enums.provider import ProviderName
from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.providers.fmp.fundamentals import SecuritySearchResult


class IdentifierSearchClient(Protocol):
    async def search_isin(
        self, isin: str, use_cache: bool = True, make_cache: bool = True
    ) -> list[SecuritySearchResult]: ...

    async def search_cik(
        self, cik: int, use_cache: bool = True, make_cache: bool = True
    ) -> list[SecuritySearchResult]: ...

    async def search_exchange_variants(
        self, symbol: str, use_cache: bool = True, make_cache: bool = True
    ) -> list[SecuritySearchResult]: ...


class SecurityIdentifierResolver:
    """Enrich identifier mappings with FMP's security search directory."""

    def __init__(
        self,
        search_client: Optional[IdentifierSearchClient] = None,
        isin_search_client: Optional[IdentifierSearchClient] = None,
    ):
        self._search_client = search_client
        self._isin_search_client = isin_search_client or search_client

    async def resolve(
        self,
        identifiers: SecurityIdentifiers,
        provider: ProviderName,
        *,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> SecurityIdentifiers:
        provider = ProviderName(provider)
        explicit_symbol = identifiers.provider_symbols.get(provider)
        exchange_symbol = (
            identifiers.exchange_symbols.get(identifiers.exchange)
            if identifiers.exchange
            else None
        )
        sole_exchange_symbol = (
            next(iter(identifiers.exchange_symbols.values()))
            if not identifiers.exchange and len(identifiers.exchange_symbols) == 1
            else None
        )
        if explicit_symbol or exchange_symbol or sole_exchange_symbol:
            return identifiers

        if provider == ProviderName.SEC and (
            identifiers.cik is not None or identifiers.ticker is not None
        ):
            return identifiers

        if identifiers.ticker and not identifiers.exchange:
            return identifiers

        search_client = (
            self._isin_search_client if identifiers.isin else self._search_client
        )
        if search_client is None:
            if identifiers.ticker:
                return identifiers
            if identifiers.isin:
                raise ValueError(
                    "ISIN resolution requires OpenFIGI access or an explicit "
                    "--provider-symbol mapping"
                )
            raise ValueError(
                "CIK resolution for a symbol-based provider requires an FMP API "
                "key or an explicit --provider-symbol mapping"
            )

        if identifiers.isin:
            matches = await search_client.search_isin(
                identifiers.isin, use_cache=use_cache, make_cache=make_cache
            )
        elif identifiers.cik is not None:
            matches = await search_client.search_cik(
                identifiers.cik, use_cache=use_cache, make_cache=make_cache
            )
        elif identifiers.ticker and identifiers.exchange:
            matches = await search_client.search_exchange_variants(
                identifiers.ticker, use_cache=use_cache, make_cache=make_cache
            )
        else:
            return identifiers

        match = self._select_match(matches, identifiers, provider)
        if match is None:
            description = (
                f"ISIN {identifiers.isin}"
                if identifiers.isin
                else f"CIK {identifiers.cik}"
                if identifiers.cik is not None
                else f"{identifiers.ticker} on {identifiers.exchange}"
            )
            raise ValueError(f"No provider symbol was found for {description}")

        exchange = identifiers.exchange or match.exchange_short_name
        exchange_symbols = {}
        if exchange:
            exchange_symbols[exchange] = match.symbol
        provider_symbols = {provider: match.symbol}
        cik = identifiers.cik
        if cik is None and match.cik and match.cik.isdigit():
            cik = int(match.cik)
        return identifiers.with_updates(
            ticker=identifiers.ticker or match.symbol,
            cik=cik,
            exchange=exchange,
            exchange_symbols=exchange_symbols,
            provider_symbols=provider_symbols,
        )

    @classmethod
    def _select_match(
        cls,
        matches: list[SecuritySearchResult],
        identifiers: SecurityIdentifiers,
        provider: ProviderName,
    ) -> Optional[SecuritySearchResult]:
        if identifiers.exchange:
            exchange_key = cls._exchange_key(identifiers.exchange)
            matches = [
                match
                for match in matches
                if exchange_key
                in {
                    cls._exchange_key(match.exchange_short_name),
                    cls._exchange_key(match.stock_exchange),
                }
            ]
        elif provider == ProviderName.SEC:
            us_matches = [
                match
                for match in matches
                if cls._exchange_key(match.exchange_short_name) == "us"
            ]
            if us_matches:
                matches = us_matches
        if not matches:
            return None
        unique_symbols = {match.symbol for match in matches}
        if len(unique_symbols) == 1:
            return matches[0]
        if len(matches) > 1:
            choices = ", ".join(
                f"{match.symbol} ({match.exchange_short_name or match.stock_exchange or '?'})"
                for match in matches[:5]
            )
            raise ValueError(
                f"Identifier is ambiguous: {choices}. Specify --exchange or an "
                "explicit --provider-symbol mapping"
            )
        return matches[0]

    @staticmethod
    def _exchange_key(value: Optional[str]) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold()) if value else ""
