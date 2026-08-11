import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest

from edgarito.schemas.providers.yahoo.fundamentals import YahooCompanyFinancials
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.valuation import (
    MarketAwarePeerDiscoveryProvider,
    MassiveRelatedCompaniesPeerDiscoveryProvider,
    OpenAIPeerDiscoveryProvider,
    PeerDiscoveryResult,
    YahooScreenerPeerDiscoveryProvider,
)

FIXTURES = Path(__file__).parent / "fixtures" / "peer_discovery"


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def text(self):
        return json.dumps(self._payload)


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return _Response(self.payload)


class _StaticProvider:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def discover(self, target, *, max_candidates=30):
        self.calls += 1
        return self.result.model_copy(
            update={"candidate_tickers": self.result.candidate_tickers[:max_candidates]}
        )


class _FailingProvider:
    async def discover(self, target, *, max_candidates=30):
        raise RuntimeError("fixture outage")


class _OpenAIProvider:
    def __init__(self, tickers):
        self.tickers = tickers
        self.calls = []

    async def extract_structured(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["response_model"](tickers=self.tickers)


class _FailingOpenAIProvider:
    async def extract_structured(self, **kwargs):
        raise RuntimeError("OpenAI fixture outage")


def _result(provider, target, candidates, confidence="high"):
    return PeerDiscoveryResult(
        provider=provider,
        target_ticker=target,
        candidate_tickers=tuple(candidates),
        methodology=f"{provider} fixture",
        confidence=confidence,
    )


def _target(symbol="AAPL", company_name="Apple Inc."):
    return YahooCompanyFinancials(
        symbol=symbol,
        company_name=company_name,
        currency="USD",
        exchange="NasdaqGS",
        sector="Technology",
        industry="Consumer Electronics",
        country="United States",
        market_capitalization=Decimal("3000000000000"),
    )


def test_openai_discovery_uses_injected_structured_client():
    openai = _OpenAIProvider([" msft ", "GOOGL", "AMZN", "META"])

    result = asyncio.run(
        OpenAIPeerDiscoveryProvider(openai).discover(_target(), max_candidates=3)
    )

    assert result.provider == "openai"
    assert result.target_ticker == "AAPL"
    assert result.candidate_tickers == ("MSFT", "GOOGL", "AMZN")
    assert len(openai.calls) == 1
    call = openai.calls[0]
    assert call["response_model"].__name__ == "OpenAIPeerDiscoveryResponse"
    assert "Target ticker: AAPL" in call["content"]
    assert "Return at most 3" in call["content"]
    assert "economically comparable" in call["instructions"]


def test_openai_discovery_normalizes_and_excludes_issuer_listings():
    openai = _OpenAIProvider(
        [
            " aapl ",
            "AAPL.L",
            "msft",
            "MSFT.US",
            "MSFT",
            "BRK.B",
            "brk-b",
            "bad ticker",
            "$META",
            "META",
        ]
    )

    result = asyncio.run(OpenAIPeerDiscoveryProvider(openai).discover(_target()))

    assert result.candidate_tickers == ("MSFT", "BRK.B", "META")
    assert "AAPL" not in result.candidate_tickers
    assert "AAPL.L" not in result.candidate_tickers
    assert any("invalid ticker" in warning for warning in result.warnings)
    assert any("cross-listing" in warning for warning in result.warnings)
    assert any("duplicate issuer" in warning for warning in result.warnings)


def test_openai_discovery_propagates_failures_for_orchestration_fallback():
    with pytest.raises(RuntimeError, match="OpenAI fixture outage"):
        asyncio.run(
            OpenAIPeerDiscoveryProvider(_FailingOpenAIProvider()).discover(_target())
        )


def test_us_discovery_uses_massive_first_and_yahoo_when_massive_is_sparse(tmp_path):
    payload = json.loads(
        (FIXTURES / "massive_aapl_related.json").read_text(encoding="utf-8")
    )
    session = _Session(payload)
    massive = MassiveRelatedCompaniesPeerDiscoveryProvider(
        FileSystemCache(tmp_path),
        "test-key",
        session=session,
        use_cache=False,
        make_cache=False,
    )
    yahoo = _StaticProvider(_result("yahoo-screener", "AAPL", ("ORCL",)))
    target = YahooCompanyFinancials(
        symbol="AAPL",
        company_name="Apple Inc.",
        currency="USD",
        exchange="NasdaqGS",
        sector="Technology",
        industry="Consumer Electronics",
        country="United States",
        market_capitalization=Decimal("3000000000000"),
    )

    primary = asyncio.run(
        MarketAwarePeerDiscoveryProvider(yahoo, massive).discover(target)
    )

    assert primary.provider == "massive-related"
    assert primary.candidate_tickers == (
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "ADBE",
    )
    assert "AAPL" not in primary.candidate_tickers
    assert yahoo.calls == 0
    assert session.requests[0][0].endswith("/v1/related-companies/AAPL")
    assert session.requests[0][1]["params"] == {"apiKey": "test-key"}

    sparse_massive = _StaticProvider(
        _result("massive-related", "AAPL", ("MSFT", "GOOGL"), "low")
    )
    fallback_yahoo = _StaticProvider(
        _result(
            "yahoo-screener",
            "AAPL",
            ("GOOGL", "AMZN", "META", "NVDA", "ADBE"),
        )
    )
    fallback = asyncio.run(
        MarketAwarePeerDiscoveryProvider(
            fallback_yahoo, sparse_massive, minimum_candidates=5
        ).discover(target)
    )

    assert fallback.provider == "massive-related+yahoo-screener"
    assert fallback.candidate_tickers == (
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "ADBE",
    )
    assert any("too few" in warning for warning in fallback.warnings)
    assert fallback_yahoo.calls == 1

    outage_yahoo = _StaticProvider(
        _result("yahoo-screener", "AAPL", ("MSFT", "AMZN", "META", "NVDA", "ADBE"))
    )
    outage = asyncio.run(
        MarketAwarePeerDiscoveryProvider(outage_yahoo, _FailingProvider()).discover(
            target
        )
    )
    assert outage.provider == "yahoo-screener"
    assert outage.candidate_tickers == ("MSFT", "AMZN", "META", "NVDA", "ADBE")
    assert any("Massive discovery failed" in warning for warning in outage.warnings)


def test_european_discovery_skips_massive_and_prefers_regional_yahoo_peers(tmp_path):
    response = json.loads(
        (FIXTURES / "yahoo_race_europe.json").read_text(encoding="utf-8")
    )
    screen_calls = []

    def screen(query, **kwargs):
        screen_calls.append((query, kwargs))
        return response

    yahoo = YahooScreenerPeerDiscoveryProvider(
        FileSystemCache(tmp_path),
        screen=screen,
        use_cache=False,
        make_cache=False,
    )
    massive = _StaticProvider(_result("massive-related", "RACE.MI", ("F",)))
    target = YahooCompanyFinancials(
        symbol="RACE.MI",
        company_name="Ferrari N.V.",
        currency="EUR",
        exchange="Milan",
        sector="Consumer Cyclical",
        industry="Auto Manufacturers",
        country="Italy",
        market_capitalization=Decimal("60000000000"),
    )

    result = asyncio.run(
        MarketAwarePeerDiscoveryProvider(yahoo, massive).discover(target)
    )

    assert result.provider == "yahoo-screener"
    assert result.candidate_tickers == (
        "STLAM.MI",
        "MONC.MI",
        "P911.DE",
        "CFR.SW",
        "RMS.PA",
    )
    assert "RACE.MI" not in result.candidate_tickers
    assert "TINY.MI" not in result.candidate_tickers
    assert massive.calls == 0
    assert len(screen_calls) == 1
    assert "Non-U.S. issuer bypassed" in result.methodology


def test_yahoo_discovery_excludes_target_cross_listing_by_issuer_identity(tmp_path):
    response = {
        "quotes": [
            {
                "symbol": "ASML.AS",
                "quoteType": "EQUITY",
                "longName": "ASML Holding NV",
                "marketCap": 1000,
                "exchange": "AMS",
                "region": "nl",
            },
            {
                "symbol": "BESI.AS",
                "quoteType": "EQUITY",
                "longName": "BE Semiconductor Industries NV",
                "marketCap": 1000,
                "exchange": "AMS",
                "region": "nl",
            },
            {
                "symbol": "ASM.AS",
                "quoteType": "EQUITY",
                "longName": "ASM International NV",
                "marketCap": 1100,
                "exchange": "AMS",
                "region": "nl",
            },
        ]
    }

    provider = YahooScreenerPeerDiscoveryProvider(
        FileSystemCache(tmp_path),
        screen=lambda *_args, **_kwargs: response,
        use_cache=False,
        make_cache=False,
    )
    target = YahooCompanyFinancials(
        symbol="ASML",
        company_name="ASML Holding N.V.",
        currency="EUR",
        industry="Semiconductors",
        country="Netherlands",
        market_capitalization=Decimal("1000"),
    )

    result = asyncio.run(provider.discover(target, max_candidates=2))

    assert "ASML.AS" not in result.candidate_tickers
    assert result.candidate_tickers == ("BESI.AS", "ASM.AS")
