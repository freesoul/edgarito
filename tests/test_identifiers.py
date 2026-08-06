import asyncio

import pytest
from test_provider_routing_and_crosscheck import _financials

from edgarito.config.providers import MarketProviderConfiguration, ProviderConfiguration
from edgarito.enums.market import Market
from edgarito.enums.provider import ProviderName
from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.providers.fmp.fundamentals import SecuritySearchResult
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.identifiers import SecurityIdentifierResolver
from edgarito.services.reconciliation.financials import FinancialDataService


class _SearchClient:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def search_isin(self, isin, use_cache=True, make_cache=True):
        self.calls.append(("isin", isin))
        return self.results

    async def search_cik(self, cik, use_cache=True, make_cache=True):
        self.calls.append(("cik", cik))
        return self.results

    async def search_exchange_variants(self, symbol, use_cache=True, make_cache=True):
        self.calls.append(("exchange", symbol))
        return self.results


class _Provider:
    def __init__(self, name):
        self.name = name
        self.symbols = []

    async def retrieve(self, query):
        self.symbols.append(query.symbol_for(self.name))
        return _financials(self.name.value)


def test_identifiers_normalize_and_prefer_provider_then_exchange_symbols():
    identifiers = SecurityIdentifiers(
        ticker=" race ",
        isin="nl0011585146",
        cik=1648416,
        exchange="nyse",
        exchange_symbols={"nyse": "race", "mil": "race.mi"},
        provider_symbols={"alphavantage": "race", "fmp": "race.us"},
    )

    assert identifiers.ticker == "RACE"
    assert identifiers.isin == "NL0011585146"
    assert identifiers.exchange == "NYSE"
    assert identifiers.symbol_for(ProviderName.FMP) == "RACE.US"
    assert identifiers.symbol_for(ProviderName.SEC) == "RACE"


def test_identifiers_reject_an_invalid_isin_checksum():
    with pytest.raises(ValueError, match="Invalid ISIN"):
        SecurityIdentifiers(isin="US0378331004")


def test_resolver_uses_isin_and_exchange_to_select_a_listing():
    search = _SearchClient(
        [
            SecuritySearchResult(
                symbol="RACE",
                name="Ferrari N.V.",
                exchangeShortName="NYSE",
                cik="0001648416",
            ),
            SecuritySearchResult(
                symbol="RACE.MI",
                name="Ferrari N.V.",
                exchangeShortName="MIL",
                cik="0001648416",
            ),
        ]
    )
    resolver = SecurityIdentifierResolver(search)

    resolved = asyncio.run(
        resolver.resolve(
            SecurityIdentifiers(isin="NL0011585146", exchange="MIL"),
            ProviderName.FMP,
        )
    )

    assert search.calls == [("isin", "NL0011585146")]
    assert resolved.ticker == "RACE.MI"
    assert resolved.cik == 1648416
    assert resolved.symbol_for(ProviderName.FMP) == "RACE.MI"
    assert resolved.exchange_symbols == {"MIL": "RACE.MI"}


def test_resolver_prefers_openfigi_us_composite_for_sec():
    isin_search = _SearchClient(
        [
            SecuritySearchResult(
                symbol="GOOGL", name="Alphabet Inc.", exchangeShortName="US"
            ),
            SecuritySearchResult(
                symbol="ABEA", name="Alphabet Inc.", exchangeShortName="GR"
            ),
        ]
    )
    resolver = SecurityIdentifierResolver(isin_search_client=isin_search)

    resolved = asyncio.run(
        resolver.resolve(
            SecurityIdentifiers(isin="US02079K3059"),
            ProviderName.SEC,
        )
    )

    assert isin_search.calls == [("isin", "US02079K3059")]
    assert resolved.ticker == "GOOGL"
    assert resolved.symbol_for(ProviderName.SEC) == "GOOGL"
    assert resolved.exchange == "US"


def test_financial_service_routes_each_provider_with_its_own_symbol(tmp_path):
    sec = _Provider(ProviderName.SEC)
    alpha = _Provider(ProviderName.ALPHAVANTAGE)
    configuration = ProviderConfiguration(
        us=MarketProviderConfiguration(
            ProviderName.SEC,
            (ProviderName.SEC, ProviderName.ALPHAVANTAGE),
        ),
        eu=MarketProviderConfiguration(
            ProviderName.ALPHAVANTAGE,
            (ProviderName.ALPHAVANTAGE,),
        ),
    )
    service = FinancialDataService(
        FileSystemCache(tmp_path),
        configuration,
        providers={ProviderName.SEC: sec, ProviderName.ALPHAVANTAGE: alpha},
    )

    result = asyncio.run(
        service.retrieve(
            identifiers=SecurityIdentifiers(
                ticker="BRK.B",
                provider_symbols={
                    ProviderName.SEC: "BRK-B",
                    ProviderName.ALPHAVANTAGE: "BRK.B",
                },
            ),
            market=Market.US,
        )
    )

    assert sec.symbols == ["BRK-B"]
    assert alpha.symbols == ["BRK.B"]
    assert result.identifiers.provider_symbols[ProviderName.SEC] == "BRK-B"
    assert result.identifiers.provider_symbols[ProviderName.ALPHAVANTAGE] == "BRK.B"
