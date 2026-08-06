import asyncio
import json
from decimal import Decimal

import pytest

import edgarito.cli.__main__ as cli_module
from edgarito.cli import main
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import FinancialConcept
from edgarito.schemas.providers.alphavantage.fundamentals import (
    AlphaVantageCompanyFinancials,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization.alphavantage import AlphaVantageNormalizer
from edgarito.services.providers.alphavantage import AlphaVantageClient


def _report(date: str) -> dict:
    return {
        "fiscalDateEnding": date,
        "reportedCurrency": "USD",
        "totalRevenue": "100000000",
        "operatingIncome": "20000000",
        "netIncome": "15000000",
        "totalAssets": "500000000",
        "totalLiabilities": "300000000",
        "totalShareholderEquity": "200000000",
        "cashAndCashEquivalentsAtCarryingValue": "50000000",
        "operatingCashflow": "25000000",
        "capitalExpenditures": "5000000",
    }


def _api_responses() -> dict[str, dict]:
    annual = _report("2024-09-28")
    quarterly = [_report("2024-12-28"), _report("2023-12-30")]
    return {
        "OVERVIEW": {
            "Symbol": "AAPL",
            "Name": "Apple Inc.",
            "CIK": "320193",
            "Currency": "USD",
            "FiscalYearEnd": "September",
        },
        "INCOME_STATEMENT": {
            "symbol": "AAPL",
            "annualReports": [annual],
            "quarterlyReports": quarterly,
        },
        "BALANCE_SHEET": {
            "symbol": "AAPL",
            "annualReports": [annual],
            "quarterlyReports": quarterly,
        },
        "CASH_FLOW": {
            "symbol": "AAPL",
            "annualReports": [annual],
            "quarterlyReports": quarterly,
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
        self.calls: list[tuple[str, str]] = []

    def get(self, url: str, params: dict, timeout: int):
        self.calls.append((params["function"], params["symbol"]))
        response = self.responses[params["function"]]
        if isinstance(response, list):
            response = response.pop(0)
        return _FakeResponse(response)


def test_provider_retrieves_and_caches_all_fundamental_responses(tmp_path):
    session = _FakeSession(_api_responses())
    cache = FileSystemCache(tmp_path)
    client = AlphaVantageClient(
        cache, "secret-api-key", session=session, min_request_interval=0
    )

    first = asyncio.run(client.get_company_financials("aapl"))
    second = asyncio.run(client.get_company_financials("AAPL"))

    assert first == second
    assert first.overview.name == "Apple Inc."
    assert len(session.calls) == 4
    cached_files = sorted(
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.json")
    )
    assert cached_files == [
        "providers/alphavantage/AAPL/balance_sheet.json",
        "providers/alphavantage/AAPL/cash_flow.json",
        "providers/alphavantage/AAPL/income_statement.json",
        "providers/alphavantage/AAPL/overview.json",
    ]
    assert all("secret-api-key" not in path for path in cached_files)


def test_provider_surfaces_api_information_without_caching_it(tmp_path):
    session = _FakeSession({"INCOME_STATEMENT": {"Information": "API limit"}})
    client = AlphaVantageClient(
        FileSystemCache(tmp_path),
        "secret-api-key",
        session=session,
        min_request_interval=0,
    )

    with pytest.raises(RuntimeError, match="Alpha Vantage: API limit"):
        asyncio.run(client.get_income_statement("AAPL"))

    assert not list(tmp_path.rglob("*.json"))


def test_provider_redacts_api_key_from_information_message(tmp_path):
    api_key = "private-api-key"
    session = _FakeSession(
        {"OVERVIEW": {"Information": f"The API key {api_key} has reached its limit"}}
    )
    client = AlphaVantageClient(
        FileSystemCache(tmp_path),
        api_key,
        session=session,
        min_request_interval=0,
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(client.get_overview("AAPL"))

    assert api_key not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_provider_retries_a_per_second_burst_limit_once(tmp_path):
    responses = _api_responses()
    session = _FakeSession(
        {
            "INCOME_STATEMENT": [
                {"Information": "Please use 1 request per second"},
                responses["INCOME_STATEMENT"],
            ]
        }
    )
    client = AlphaVantageClient(
        FileSystemCache(tmp_path),
        "secret-api-key",
        session=session,
        min_request_interval=0,
    )

    response = asyncio.run(client.get_income_statement("AAPL"))

    assert len(response.annual_reports) == 1
    assert session.calls == [
        ("INCOME_STATEMENT", "AAPL"),
        ("INCOME_STATEMENT", "AAPL"),
    ]


def test_normalizer_maps_reports_to_common_financials_and_fiscal_periods():
    responses = _api_responses()
    financials = AlphaVantageCompanyFinancials(
        overview=responses["OVERVIEW"],
        income_statement=responses["INCOME_STATEMENT"],
        balance_sheet=responses["BALANCE_SHEET"],
        cash_flow=responses["CASH_FLOW"],
    )

    normalized = AlphaVantageNormalizer().normalize(financials)

    assert normalized.provider == "alphavantage"
    assert normalized.company_id == "0000320193"
    assert normalized.company_name == "Apple Inc."
    assert normalized.ticker == "AAPL"
    assert len(normalized.observations) == 27

    annual_revenue = next(
        observation
        for observation in normalized.observations
        if observation.concept == FinancialConcept.REVENUE
        and observation.granularity == Granularity.ANNUAL
    )
    assert annual_revenue.value == Decimal("100000000")
    assert annual_revenue.fiscal_year == 2024
    assert annual_revenue.fiscal_period == FiscalPeriod.FY
    assert annual_revenue.source_concept == "totalRevenue"
    assert annual_revenue.accession_number is None
    assert annual_revenue.filed is None

    quarterly_revenues = [
        observation
        for observation in normalized.observations
        if observation.concept == FinancialConcept.REVENUE
        and observation.granularity == Granularity.QUARTERLY
    ]
    assert [
        (observation.fiscal_year, observation.fiscal_period)
        for observation in quarterly_revenues
    ] == [(2024, FiscalPeriod.Q1), (2025, FiscalPeriod.Q1)]


def test_normalizer_filters_granularity_and_concepts():
    responses = _api_responses()
    financials = AlphaVantageCompanyFinancials(
        overview=responses["OVERVIEW"],
        income_statement=responses["INCOME_STATEMENT"],
        balance_sheet=responses["BALANCE_SHEET"],
        cash_flow=responses["CASH_FLOW"],
    )

    normalized = AlphaVantageNormalizer().normalize(
        financials,
        granularity=Granularity.ANNUAL,
        concepts={FinancialConcept.NET_INCOME},
    )

    assert len(normalized.observations) == 1
    assert normalized.observations[0].concept == FinancialConcept.NET_INCOME
    assert normalized.observations[0].granularity == Granularity.ANNUAL


def test_cli_can_override_the_default_with_alphavantage(tmp_path, capsys, monkeypatch):
    responses = _api_responses()
    cache = FileSystemCache(tmp_path)
    for function, response in responses.items():
        cache.save(
            f"providers/alphavantage/AAPL/{function.lower()}.json",
            json.dumps(response),
        )
    monkeypatch.setattr(cli_module, "ALPHAVANTAGE_API_KEY", "test-api-key")

    exit_code = main(
        [
            "financials",
            "--ticker",
            "AAPL",
            "--provider",
            "alphavantage",
            "--cache-dir",
            str(tmp_path),
            "--limit",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Provider: ALPHAVANTAGE" in output
