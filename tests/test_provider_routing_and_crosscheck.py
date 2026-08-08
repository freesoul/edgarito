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
from edgarito.services.reconciliation.financials import (
    FinancialDataSelector,
    FinancialDataService,
)


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


def test_service_explicit_provider_crosschecks_by_default_and_only_warns(tmp_path):
    sec = _FakeProvider(ProviderName.SEC, _financials("sec", "100"))
    alpha = _FakeProvider(ProviderName.ALPHAVANTAGE, _financials("alphavantage", "120"))
    service = FinancialDataService(
        FileSystemCache(tmp_path),
        _configuration(),
        providers={ProviderName.SEC: sec, ProviderName.ALPHAVANTAGE: alpha},
    )

    with pytest.warns(FinancialDataCrosscheckWarning, match="value mismatch"):
        result = asyncio.run(
            service.retrieve(ticker="AAPL", provider=ProviderName.SEC)
        )

    assert result.provider == "sec"
    assert result.observations[0].value == Decimal("100")
    assert len(sec.queries) == 1
    assert len(alpha.queries) == 1
    assert len(service.last_crosschecks) == 1


def test_service_without_provider_retrieves_all_and_returns_one_best_dataset(tmp_path):
    sec = _FakeProvider(ProviderName.SEC, _financials("sec", "100"))
    alpha_financials = _financials("alphavantage", "120")
    alpha_financials.observations.append(
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
    alpha = _FakeProvider(ProviderName.ALPHAVANTAGE, alpha_financials)
    service = FinancialDataService(
        FileSystemCache(tmp_path),
        _configuration(),
        providers={ProviderName.SEC: sec, ProviderName.ALPHAVANTAGE: alpha},
    )

    result = asyncio.run(
        service.retrieve(
            ticker="AAPL",
            concepts={FinancialConcept.REVENUE, FinancialConcept.NET_INCOME},
        )
    )

    assert result.provider == "alphavantage"
    assert result.observations is alpha_financials.observations
    assert len(sec.queries) == 1
    assert len(alpha.queries) == 1
    assert service.last_crosschecks == []


def test_selector_prefers_newer_dataset_when_completeness_is_equal():
    older = _financials("sec")
    newer = _financials("alphavantage")
    newer.observations[0] = newer.observations[0].model_copy(
        update={"period_end": datetime.date(2026, 9, 27)}
    )

    ranked = FinancialDataSelector.rank([older, newer])

    assert ranked[0][0] is newer
    assert ranked[0][1].latest_period_end == datetime.date(2026, 9, 27)


def test_selector_prefers_more_complete_dataset_when_freshness_is_equal():
    incomplete = _financials("sec")
    complete = _financials("alphavantage")
    complete.observations.append(
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

    ranked = FinancialDataSelector.rank(
        [incomplete, complete],
        concepts={FinancialConcept.REVENUE, FinancialConcept.NET_INCOME},
    )

    assert ranked[0][0] is complete
    assert ranked[0][1].completeness == 1
    assert ranked[1][1].completeness == 0.5


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


def test_service_skips_failed_provider_and_reports_all_provider_failure(tmp_path):
    class _FailingProvider(_FakeProvider):
        async def retrieve(self, query):
            self.queries.append(query)
            raise RuntimeError("upstream unavailable")

    failing = _FailingProvider(ProviderName.SEC, _financials("sec"))
    alpha = _FakeProvider(ProviderName.ALPHAVANTAGE, _financials("alphavantage"))
    service = FinancialDataService(
        FileSystemCache(tmp_path),
        _configuration(),
        providers={ProviderName.SEC: failing, ProviderName.ALPHAVANTAGE: alpha},
    )

    result = asyncio.run(service.retrieve(ticker="AAPL"))

    assert result.provider == "alphavantage"
    assert service.last_selection_failures == [
        (ProviderName.SEC, "upstream unavailable")
    ]

    failing_alpha = _FailingProvider(
        ProviderName.ALPHAVANTAGE, _financials("alphavantage")
    )
    failed_service = FinancialDataService(
        FileSystemCache(tmp_path),
        _configuration(),
        providers={ProviderName.SEC: failing, ProviderName.ALPHAVANTAGE: failing_alpha},
    )
    with pytest.raises(
        ValueError,
        match="No configured financial data provider returned usable data.*sec.*alphavantage",
    ):
        asyncio.run(failed_service.retrieve(ticker="AAPL"))


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
