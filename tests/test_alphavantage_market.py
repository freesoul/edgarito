import asyncio
import datetime
import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.providers.alphavantage.market import (
    DailyTimeSeriesResponse,
    DividendResponse,
    GlobalQuoteResponse,
    SplitResponse,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization.alphavantage_market import (
    AlphaVantageMarketNormalizer,
)
from edgarito.services.providers.alphavantage import (
    AlphaVantageClient,
    AlphaVantageOutputSize,
)

UTC = datetime.timezone.utc


def _market_responses():
    return {
        "TIME_SERIES_DAILY": {
            "Meta Data": {
                "1. Information": "Daily Prices",
                "2. Symbol": "AAPL",
                "3. Last Refreshed": "2026-08-05",
                "4. Output Size": "Compact",
                "5. Time Zone": "US/Eastern",
            },
            "Time Series (Daily)": {
                "2026-08-05": {
                    "1. open": "210.00",
                    "2. high": "214.00",
                    "3. low": "209.00",
                    "4. close": "212.00",
                    "5. volume": "1000000",
                },
                "2026-08-04": {
                    "1. open": "208.00",
                    "2. high": "211.00",
                    "3. low": "207.00",
                    "4. close": "210.00",
                    "5. volume": "900000",
                },
            },
        },
        "GLOBAL_QUOTE": {
            "Global Quote": {
                "01. symbol": "AAPL",
                "02. open": "210.00",
                "03. high": "214.00",
                "04. low": "209.00",
                "05. price": "212.00",
                "06. volume": "1000000",
                "07. latest trading day": "2026-08-05",
                "08. previous close": "210.00",
                "09. change": "2.00",
                "10. change percent": "0.9524%",
            }
        },
        "DIVIDENDS": {
            "symbol": "AAPL",
            "data": [
                {
                    "ex_dividend_date": "2026-08-10",
                    "declaration_date": "2026-07-31",
                    "record_date": "2026-08-10",
                    "payment_date": "2026-08-13",
                    "amount": "0.26",
                },
                {
                    "ex_dividend_date": "2026-05-11",
                    "declaration_date": "None",
                    "record_date": "2026-05-11",
                    "payment_date": "2026-05-14",
                    "amount": "0.25",
                },
            ],
        },
        "SPLITS": {
            "symbol": "AAPL",
            "data": [{"effective_date": "2020-08-31", "split_factor": "4.0"}],
        },
    }


class _FakeResponse:
    def __init__(self, data: dict, status: int = 200):
        self._data = data
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self) -> str:
        return json.dumps(self._data)


class _FakeSession:
    def __init__(self, responses: dict[str, dict]):
        self.responses = responses
        self.calls: list[dict] = []

    def get(self, url: str, params: dict, timeout: int):
        self.calls.append(dict(params))
        response = self.responses[params["function"]]
        if isinstance(response, list):
            response = response.pop(0)
        return _FakeResponse(response)


def test_client_retrieves_and_caches_only_scoped_market_endpoints(tmp_path):
    session = _FakeSession(_market_responses())
    client = AlphaVantageClient(
        FileSystemCache(tmp_path),
        "secret-api-key",
        session=session,
        min_request_interval=0,
    )

    daily = asyncio.run(client.get_daily_prices("aapl"))
    assert asyncio.run(client.get_daily_prices("AAPL")) == daily
    assert asyncio.run(client.get_latest_close("AAPL")) == Decimal("212.00")
    assert asyncio.run(client.get_latest_close("AAPL")) == Decimal("212.00")
    dividends = asyncio.run(client.get_dividends("AAPL"))
    splits = asyncio.run(client.get_splits("AAPL"))

    assert daily.metadata.symbol == "AAPL"
    assert daily.time_series[datetime.date(2026, 8, 5)].close == Decimal("212.00")
    assert dividends.data[1].declaration_date is None
    assert splits.data[0].split_factor == Decimal("4.0")
    assert [call["function"] for call in session.calls] == [
        "TIME_SERIES_DAILY",
        "GLOBAL_QUOTE",
        "DIVIDENDS",
        "SPLITS",
    ]
    assert "outputsize" not in session.calls[0]
    cached_files = sorted(
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.json")
    )
    assert cached_files == [
        "providers/alphavantage/AAPL/dividends.json",
        "providers/alphavantage/AAPL/global_quote.json",
        "providers/alphavantage/AAPL/splits.json",
        "providers/alphavantage/AAPL/time_series_daily_compact.json",
    ]
    assert all("secret-api-key" not in path for path in cached_files)


def test_full_daily_history_is_explicit_and_uses_a_separate_cache(tmp_path):
    responses = _market_responses()
    session = _FakeSession({"TIME_SERIES_DAILY": responses["TIME_SERIES_DAILY"]})
    client = AlphaVantageClient(
        FileSystemCache(tmp_path),
        "secret-api-key",
        session=session,
        min_request_interval=0,
    )

    asyncio.run(client.get_daily_prices("AAPL", AlphaVantageOutputSize.FULL))

    assert session.calls[0]["outputsize"] == "full"
    assert (
        tmp_path / "providers/alphavantage/AAPL/time_series_daily_full.json"
    ).is_file()


def test_market_normalizer_combines_raw_prices_dividends_and_splits():
    responses = _market_responses()
    daily = DailyTimeSeriesResponse.model_validate(responses["TIME_SERIES_DAILY"])

    normalized = AlphaVantageMarketNormalizer().normalize(
        symbol="aapl",
        currency="usd",
        daily_prices=daily,
        dividends=DividendResponse.model_validate(responses["DIVIDENDS"]),
        splits=SplitResponse.model_validate(responses["SPLITS"]),
        identifiers=SecurityIdentifiers(ticker="AAPL", isin="US0378331005"),
        exchange="NASDAQ",
        retrieved_at=datetime.datetime(2026, 8, 6, 12, tzinfo=UTC),
    )

    assert normalized.provider == "alphavantage"
    assert normalized.provider_symbol == "AAPL"
    assert normalized.currency == "USD"
    assert normalized.source_version == "2026-08-05"
    assert [price.observed_on for price in normalized.prices] == [
        datetime.date(2026, 8, 4),
        datetime.date(2026, 8, 5),
    ]
    assert normalized.latest_price.close == Decimal("212.00")
    assert normalized.latest_price.adjusted_close is None
    assert normalized.dividends[0].amount == Decimal("0.25")
    assert normalized.dividends[0].currency == "USD"
    assert normalized.splits[0].factor == Decimal("4.0")


def test_market_normalizer_maps_global_quote_to_a_snapshot():
    response = GlobalQuoteResponse.model_validate(_market_responses()["GLOBAL_QUOTE"])
    normalized = AlphaVantageMarketNormalizer().normalize_quote(
        response,
        currency="USD",
        retrieved_at=datetime.datetime(2026, 8, 6, 12, tzinfo=UTC),
    )

    assert normalized.frequency.value == "snapshot"
    assert normalized.latest_price.observed_on == datetime.date(2026, 8, 5)
    assert normalized.latest_price.close == Decimal("212.00")


def test_market_normalizer_rejects_mixed_symbols():
    responses = _market_responses()
    responses["DIVIDENDS"]["symbol"] = "MSFT"

    with pytest.raises(ValueError, match="same symbol"):
        AlphaVantageMarketNormalizer().normalize(
            symbol="AAPL",
            currency="USD",
            daily_prices=DailyTimeSeriesResponse.model_validate(
                responses["TIME_SERIES_DAILY"]
            ),
            dividends=DividendResponse.model_validate(responses["DIVIDENDS"]),
        )


def test_daily_response_rejects_an_empty_price_history():
    response = _market_responses()["TIME_SERIES_DAILY"]
    response["Time Series (Daily)"] = {}

    with pytest.raises(ValidationError, match="cannot be empty"):
        DailyTimeSeriesResponse.model_validate(response)
