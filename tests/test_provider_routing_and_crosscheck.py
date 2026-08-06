import asyncio
import datetime
from decimal import Decimal

import pytest

from edgarito.config.providers import (
    MarketProviderConfiguration,
    ProviderConfiguration,
)
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.enums.market import Market
from edgarito.enums.provider import ProviderName
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.reconciliation.crosscheck import (
    CrosscheckIssueKind,
    FinancialDataCrosscheckWarning,
    FinancialsCrosschecker,
)
from edgarito.services.reconciliation.financials import FinancialDataService


def _configuration() -> ProviderConfiguration:
    return ProviderConfiguration(
        us=MarketProviderConfiguration(
            ProviderName.SEC,
            (ProviderName.SEC, ProviderName.ALPHAVANTAGE),
        ),
        eu=MarketProviderConfiguration(
            ProviderName.ALPHAVANTAGE,
            (ProviderName.ALPHAVANTAGE,),
        ),
    )


def _financials(provider: str, value: str = "100") -> NormalizedCompanyFinancials:
    return NormalizedCompanyFinancials(
        provider=provider,
        company_id="0000320193",
        company_name="Apple Inc.",
        ticker="AAPL",
        observations=[
            FinancialObservation(
                concept=FinancialConcept.REVENUE,
                statement=FinancialConcept.REVENUE.statement,
                value=Decimal(value),
                unit="USD",
                granularity=Granularity.ANNUAL,
                fiscal_year=2025,
                fiscal_period=FiscalPeriod.FY,
                period_end=datetime.date(2025, 9, 27),
                provider=provider,
                taxonomy="test",
                source_concept="revenue",
            )
        ],
    )


class _FakeProvider:
    def __init__(self, name: ProviderName, financials: NormalizedCompanyFinancials):
        self.name = name
        self.financials = financials
        self.queries = []

    async def retrieve(self, query):
        self.queries.append(query)
        return self.financials


def test_provider_configuration_defaults_and_environment_overrides():
    defaults = ProviderConfiguration.from_environment({})
    assert defaults.us.default_provider == ProviderName.SEC
    assert defaults.us.available_providers == (
        ProviderName.SEC,
        ProviderName.ALPHAVANTAGE,
        ProviderName.FMP,
        ProviderName.YAHOO,
    )
    assert defaults.eu.default_provider == ProviderName.YAHOO
    assert defaults.eu.available_providers == (
        ProviderName.YAHOO,
        ProviderName.ALPHAVANTAGE,
        ProviderName.FMP,
    )

    configured = ProviderConfiguration.from_environment(
        {
            "EDGARITO_US_DEFAULT_PROVIDER": "alphavantage",
            "EDGARITO_US_AVAILABLE_PROVIDERS": "alphavantage, sec",
            "eu_default_provider": "alphavantage",
            "eu_available_providers": "alphavantage",
        }
    )
    assert configured.us.default_provider == ProviderName.ALPHAVANTAGE
    assert configured.us.available_providers == (
        ProviderName.ALPHAVANTAGE,
        ProviderName.SEC,
    )


def test_provider_configuration_rejects_a_provider_for_an_unsupported_market():
    with pytest.raises(ValueError, match="do not support eu: sec"):
        ProviderConfiguration(
            us=MarketProviderConfiguration(ProviderName.SEC, (ProviderName.SEC,)),
            eu=MarketProviderConfiguration(ProviderName.SEC, (ProviderName.SEC,)),
        )


def test_crosschecker_reports_values_and_missing_observations_without_merging():
    primary = _financials("sec", "100")
    secondary = _financials("alphavantage", "120")
    secondary.observations.append(
        FinancialObservation(
            concept=FinancialConcept.NET_INCOME,
            statement=FinancialConcept.NET_INCOME.statement,
            value=Decimal("20"),
            unit="USD",
            granularity=Granularity.ANNUAL,
            fiscal_year=2025,
            fiscal_period=FiscalPeriod.FY,
            period_end=datetime.date(2025, 9, 27),
            provider="alphavantage",
            taxonomy="test",
            source_concept="netIncome",
        )
    )

    report = FinancialsCrosschecker().compare(primary, secondary)

    assert [issue.kind for issue in report.issues] == [
        CrosscheckIssueKind.MISSING_FROM_PRIMARY,
        CrosscheckIssueKind.VALUE_MISMATCH,
    ]
    assert len(primary.observations) == 1
    assert primary.observations[0].value == Decimal("100")


def test_service_crosschecks_by_default_and_only_warns(tmp_path):
    sec = _FakeProvider(ProviderName.SEC, _financials("sec", "100"))
    alpha = _FakeProvider(ProviderName.ALPHAVANTAGE, _financials("alphavantage", "120"))
    service = FinancialDataService(
        FileSystemCache(tmp_path),
        _configuration(),
        providers={ProviderName.SEC: sec, ProviderName.ALPHAVANTAGE: alpha},
    )

    with pytest.warns(FinancialDataCrosscheckWarning, match="value mismatch"):
        result = asyncio.run(service.retrieve(ticker="AAPL"))

    assert result.provider == "sec"
    assert result.observations[0].value == Decimal("100")
    assert len(sec.queries) == 1
    assert len(alpha.queries) == 1
    assert len(service.last_crosschecks) == 1


def test_service_allows_provider_override_and_disabling_crosscheck(tmp_path):
    sec = _FakeProvider(ProviderName.SEC, _financials("sec"))
    alpha = _FakeProvider(ProviderName.ALPHAVANTAGE, _financials("alphavantage"))
    service = FinancialDataService(
        FileSystemCache(tmp_path),
        _configuration(),
        providers={ProviderName.SEC: sec, ProviderName.ALPHAVANTAGE: alpha},
    )

    result = asyncio.run(
        service.retrieve(
            ticker="AAPL",
            provider=ProviderName.ALPHAVANTAGE,
            crosscheck=False,
        )
    )

    assert result.provider == "alphavantage"
    assert not sec.queries
    assert len(alpha.queries) == 1
    assert service.last_crosschecks == []


def test_service_rejects_an_unavailable_market_provider(tmp_path):
    service = FinancialDataService(
        FileSystemCache(tmp_path),
        _configuration(),
        providers={
            ProviderName.SEC: _FakeProvider(ProviderName.SEC, _financials("sec"))
        },
    )

    with pytest.raises(ValueError, match="not available for eu"):
        asyncio.run(
            service.retrieve(
                ticker="SAP.DEX",
                market=Market.EU,
                provider=ProviderName.SEC,
                crosscheck=False,
            )
        )


def test_service_uses_yahoo_as_the_keyless_eu_default(tmp_path):
    yahoo = _FakeProvider(ProviderName.YAHOO, _financials("yahoo"))
    service = FinancialDataService(
        FileSystemCache(tmp_path),
        ProviderConfiguration.from_environment({}),
        providers={ProviderName.YAHOO: yahoo},
    )

    result = asyncio.run(
        service.retrieve(ticker="SAP.DE", market=Market.EU, crosscheck=False)
    )

    assert result.provider == "yahoo"
    assert yahoo.queries[0].symbol_for(ProviderName.YAHOO) == "SAP.DE"
