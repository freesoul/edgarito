import asyncio
from decimal import Decimal

import pandas as pd

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.enums.provider import ProviderName
from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.normalization.financials import FinancialConcept
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization.yahoo import YahooFinancialsNormalizer
from edgarito.services.normalization.yahoo_market import YahooMarketNormalizer
from edgarito.services.providers.yahoo import YahooFinanceClient
from edgarito.services.reconciliation.providers import (
    FinancialsQuery,
    YahooFinancialsProvider,
)


def _statement(period: str, values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({pd.Timestamp(period): values})


class _FakeTicker:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.calls = []

    def get_income_stmt(self, freq: str):
        self.calls.append(("income", freq))
        period = "2025-12-31" if freq == "yearly" else "2025-03-31"
        return _statement(
            period,
            {
                "TotalRevenue": 1000,
                "OperatingIncome": 250,
                "PretaxIncome": 220,
                "TaxProvision": 44,
                "NetIncome": 176,
                "InterestExpense": 12,
                "BasicAverageShares": 100,
                "DilutedAverageShares": 105,
            },
        )

    def get_balance_sheet(self, freq: str):
        self.calls.append(("balance", freq))
        period = "2025-12-31" if freq == "yearly" else "2025-03-31"
        return _statement(
            period,
            {
                "TotalAssets": 2000,
                "CurrentAssets": 700,
                "AccountsReceivable": 150,
                "OtherCurrentAssets": 40,
                "TotalLiabilitiesNetMinorityInterest": 1200,
                "CurrentLiabilities": 500,
                "AccountsPayable": 130,
                "CurrentDebt": 60,
                "LongTermDebt": 300,
                "StockholdersEquity": 800,
                "CashAndCashEquivalents": 200,
                "OtherShortTermInvestments": 80,
                "InvestmentinFinancialAssets": 30,
                "Goodwill": 90,
                "OtherIntangibleAssets": 70,
                "OrdinarySharesNumber": 98,
            },
        )

    def get_cash_flow(self, freq: str):
        self.calls.append(("cash", freq))
        period = "2025-12-31" if freq == "yearly" else "2025-03-31"
        return _statement(
            period,
            {
                "OperatingCashFlow": 240,
                "DepreciationAndAmortization": 35,
                "CapitalExpenditure": -60,
                "CashDividendsPaid": -20,
            },
        )

    def get_info(self):
        self.calls.append(("info", None))
        return {
            "symbol": self.symbol,
            "longName": "SAP SE",
            "financialCurrency": "EUR",
            "fullExchangeName": "XETRA",
            "sector": "Technology",
            "industry": "Software Infrastructure",
            "country": "Germany",
            "beta": 1.15,
        }

    def history(self, **kwargs):
        self.calls.append(("history", kwargs))
        return pd.DataFrame(
            {
                "Open": [240.0, 250.0],
                "High": [255.0, 265.0],
                "Low": [235.0, 245.0],
                "Close": [250.0, 260.0],
                "Adj Close": [248.0, 260.0],
                "Volume": [1000, 1200],
                "Dividends": [5.0, 0.0],
                "Stock Splits": [0.0, 2.0],
            },
            index=pd.DatetimeIndex(["2026-08-04", "2026-08-05"], tz="Europe/London"),
        )

    def get_history_metadata(self, repair: bool):
        self.calls.append(("metadata", repair))
        return {"currency": "GBp", "fullExchangeName": "LSE"}


class _TickerFactory:
    def __init__(self):
        self.instances = []

    def __call__(self, symbol):
        ticker = _FakeTicker(symbol)
        self.instances.append(ticker)
        return ticker


def test_yahoo_financial_client_caches_serializable_statement_snapshots(tmp_path):
    factory = _TickerFactory()
    client = YahooFinanceClient(FileSystemCache(tmp_path), ticker_factory=factory)

    first = asyncio.run(client.get_company_financials("sap.de"))
    second = asyncio.run(client.get_company_financials("SAP.DE"))

    assert first == second
    assert first.company_name == "SAP SE"
    assert first.currency == "EUR"
    assert first.sector == "Technology"
    assert first.industry == "Software Infrastructure"
    assert first.beta == Decimal("1.15")
    assert first.annual_income_statements[0].values["TotalRevenue"] == Decimal("1000.0")
    assert len(factory.instances) == 1
    assert (tmp_path / "providers/yahoo/SAP.DE/financials.json").is_file()


def test_yahoo_normalizer_maps_extended_financial_concepts_and_periods(tmp_path):
    factory = _TickerFactory()
    client = YahooFinanceClient(FileSystemCache(tmp_path), ticker_factory=factory)
    source = asyncio.run(client.get_company_financials("SAP.DE"))

    normalized = YahooFinancialsNormalizer().normalize(source)
    annual = {
        item.concept: item
        for item in normalized.observations
        if item.granularity == Granularity.ANNUAL
    }
    quarterly = {
        item.concept: item
        for item in normalized.observations
        if item.granularity == Granularity.QUARTERLY
    }

    assert normalized.provider == "yahoo"
    assert annual[FinancialConcept.REVENUE].value == Decimal("1000.0")
    assert annual[FinancialConcept.REVENUE].source_concept == "TotalRevenue"
    assert annual[FinancialConcept.CAPITAL_EXPENDITURES].value == Decimal("60.0")
    assert annual[FinancialConcept.DIVIDENDS_PAID].value == Decimal("20.0")
    assert annual[FinancialConcept.SHORT_TERM_INVESTMENTS].value == Decimal("80.0")
    assert annual[FinancialConcept.NONCURRENT_INVESTMENTS].value == Decimal("30.0")
    assert annual[FinancialConcept.SHARES_OUTSTANDING].unit == "shares"
    assert quarterly[FinancialConcept.REVENUE].fiscal_period == FiscalPeriod.Q1
    assert quarterly[FinancialConcept.REVENUE].fiscal_year == 2025


def test_yahoo_provider_uses_its_explicit_symbol_mapping(tmp_path):
    factory = _TickerFactory()
    provider = YahooFinancialsProvider(
        YahooFinanceClient(FileSystemCache(tmp_path), ticker_factory=factory)
    )
    query = FinancialsQuery(
        identifiers=SecurityIdentifiers(
            ticker="SAP",
            provider_symbols={ProviderName.YAHOO: "SAP.DE"},
        ),
        granularity=Granularity.ANNUAL,
    )

    result = asyncio.run(provider.retrieve(query))

    assert factory.instances[0].symbol == "SAP.DE"
    assert result.ticker == "SAP.DE"


def test_yahoo_market_history_includes_adjustments_actions_and_pence_scaling(
    tmp_path,
):
    factory = _TickerFactory()
    client = YahooFinanceClient(FileSystemCache(tmp_path), ticker_factory=factory)

    source = asyncio.run(client.get_price_history("VOD.L", period="1mo"))
    cached = asyncio.run(client.get_price_history("VOD.L", period="1mo"))
    normalized = YahooMarketNormalizer().normalize(source)

    assert source == cached
    assert len(factory.instances) == 1
    history_kwargs = factory.instances[0].calls[0][1]
    assert history_kwargs["auto_adjust"] is False
    assert history_kwargs["actions"] is True
    assert history_kwargs["repair"] is True
    assert normalized.currency == "GBP"
    assert normalized.latest_price.close == Decimal("2.600")
    assert normalized.latest_price.adjusted_close == Decimal("2.600")
    assert normalized.dividends[0].amount == Decimal("0.050")
    assert normalized.splits[0].factor == Decimal("2.0")
    assert normalized.source_version.startswith("yfinance/")
