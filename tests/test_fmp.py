import asyncio
import json
from decimal import Decimal

import pytest

import edgarito.cli.__main__ as cli_module
from edgarito.cli import main
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import FinancialConcept
from edgarito.schemas.providers.fmp.fundamentals import FmpCompanyFinancials
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization.fmp import FmpNormalizer
from edgarito.services.providers.fmp import FmpClient


def _statement(date: str, fiscal_year: int, period: str) -> dict:
    return {
        "symbol": "AAPL",
        "date": date,
        "fiscalYear": fiscal_year,
        "period": period,
        "reportedCurrency": "USD",
        "cik": "320193",
        "filingDate": "2025-01-31",
        "acceptedDate": "2025-01-31 18:01:30",
        "revenue": 100_000_000,
        "operatingIncome": 20_000_000,
        "netIncome": 15_000_000,
        "totalAssets": 500_000_000,
        "totalLiabilities": 300_000_000,
        "totalStockholdersEquity": 200_000_000,
        "cashAndCashEquivalents": 50_000_000,
        "operatingCashFlow": 25_000_000,
        "capitalExpenditure": -5_000_000,
    }


def _api_responses() -> dict[tuple[str, str | None], list[dict]]:
    annual = [_statement("2024-09-28", 2024, "FY")]
    quarterly = [
        _statement("2025-03-29", 2025, "Q2"),
        _statement("2024-12-28", 2025, "Q1"),
    ]
    return {
        ("profile", None): [
            {
                "symbol": "AAPL",
                "companyName": "Apple Inc.",
                "cik": "320193",
                "currency": "USD",
                "country": "US",
                "exchange": "NASDAQ Global Select",
            }
        ],
        ("income-statement", "annual"): annual,
        ("income-statement", "quarter"): quarterly,
        ("balance-sheet-statement", "annual"): annual,
        ("balance-sheet-statement", "quarter"): quarterly,
        ("cash-flow-statement", "annual"): annual,
        ("cash-flow-statement", "quarter"): quarterly,
    }


class _FakeResponse:
    def __init__(self, data, status: int = 200):
        self._data = data
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self) -> str:
        return json.dumps(self._data)


class _FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.calls: list[tuple[str, str | None, str]] = []

    def get(self, url: str, params: dict, timeout: int):
        endpoint = url.rsplit("/", 1)[-1]
        period = params.get("period")
        self.calls.append((endpoint, period, params["symbol"]))
        return _FakeResponse(self.responses[(endpoint, period)])


def _company_financials() -> FmpCompanyFinancials:
    responses = _api_responses()
    return FmpCompanyFinancials(
        profile=responses[("profile", None)][0],
        annual_income_statements=responses[("income-statement", "annual")],
        quarterly_income_statements=responses[("income-statement", "quarter")],
        annual_balance_sheets=responses[("balance-sheet-statement", "annual")],
        quarterly_balance_sheets=responses[("balance-sheet-statement", "quarter")],
        annual_cash_flow_statements=responses[("cash-flow-statement", "annual")],
        quarterly_cash_flow_statements=responses[("cash-flow-statement", "quarter")],
    )


def test_provider_retrieves_and_caches_all_fmp_responses(tmp_path):
    session = _FakeSession(_api_responses())
    cache = FileSystemCache(tmp_path)
    client = FmpClient(cache, "secret-api-key", session=session)

    first = asyncio.run(client.get_company_financials("aapl"))
    second = asyncio.run(client.get_company_financials("AAPL"))

    assert first == second
    assert first.profile.company_name == "Apple Inc."
    assert len(session.calls) == 7
    cached_files = sorted(
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.json")
    )
    assert cached_files == [
        "providers/fmp/AAPL/balance-sheet-statement_annual.json",
        "providers/fmp/AAPL/balance-sheet-statement_quarter.json",
        "providers/fmp/AAPL/cash-flow-statement_annual.json",
        "providers/fmp/AAPL/cash-flow-statement_quarter.json",
        "providers/fmp/AAPL/income-statement_annual.json",
        "providers/fmp/AAPL/income-statement_quarter.json",
        "providers/fmp/AAPL/profile.json",
    ]
    assert all("secret-api-key" not in path for path in cached_files)


def test_provider_surfaces_an_api_error_without_caching_it(tmp_path):
    session = _FakeSession({("profile", None): {"Error Message": "Invalid API KEY"}})
    client = FmpClient(FileSystemCache(tmp_path), "secret-api-key", session=session)

    with pytest.raises(RuntimeError, match="FMP: Invalid API KEY"):
        asyncio.run(client.get_profile("AAPL"))

    assert not list(tmp_path.rglob("*.json"))


def test_normalizer_maps_fmp_statements_to_common_financials():
    normalized = FmpNormalizer().normalize(_company_financials())

    assert normalized.provider == "fmp"
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
    assert annual_revenue.source_concept == "revenue"
    assert annual_revenue.filed.isoformat() == "2025-01-31"

    capital_expenditure = next(
        observation
        for observation in normalized.observations
        if observation.concept == FinancialConcept.CAPITAL_EXPENDITURES
    )
    assert capital_expenditure.value == Decimal("5000000")


def test_normalizer_filters_granularity_and_concepts():
    normalized = FmpNormalizer().normalize(
        _company_financials(),
        granularity=Granularity.QUARTERLY,
        concepts={FinancialConcept.NET_INCOME},
    )

    assert len(normalized.observations) == 2
    assert all(
        observation.concept == FinancialConcept.NET_INCOME
        and observation.granularity == Granularity.QUARTERLY
        for observation in normalized.observations
    )


def test_cli_can_override_the_default_with_fmp(tmp_path, capsys, monkeypatch):
    cache = FileSystemCache(tmp_path)
    for (endpoint, period), response in _api_responses().items():
        suffix = f"_{period}" if period else ""
        cache.save(
            f"providers/fmp/AAPL/{endpoint}{suffix}.json",
            json.dumps(response),
        )
    monkeypatch.setattr(cli_module, "FMP_API_KEY", "test-api-key")

    exit_code = main(
        [
            "financials",
            "--ticker",
            "AAPL",
            "--provider",
            "fmp",
            "--cache-dir",
            str(tmp_path),
            "--limit",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Provider: FMP" in output
