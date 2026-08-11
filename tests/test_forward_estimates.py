import asyncio
import datetime
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd

import edgarito.cli.__main__ as cli_module
from edgarito.config.valuation import MultistageValuationConfiguration
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.forward import ForwardRevenueEstimateResult
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.providers.alphavantage.fundamentals import (
    EarningsEstimatesResponse,
)
from edgarito.schemas.providers.yahoo.fundamentals import YahooRevenueEstimateResponse
from edgarito.services.forecasting import (
    AdaptiveMultistageFcffForecastService,
    FcffForecastParameters,
    FcffForecastService,
    ForwardGrowthEvidence,
)
from edgarito.services.forward_estimates import ForwardRevenueEstimateService


def _observation(concept, value, year):
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(value),
        unit="USD",
        granularity=Granularity.ANNUAL,
        fiscal_year=year,
        fiscal_period=FiscalPeriod.FY,
        period_end=datetime.date(year, 12, 31),
        provider="test",
        taxonomy="test",
        source_concept=concept.value,
    )


def _financials():
    values = {
        2023: {
            FinancialConcept.REVENUE: "100",
            FinancialConcept.OPERATING_INCOME: "20",
            FinancialConcept.PRETAX_INCOME: "18",
            FinancialConcept.INCOME_TAX_EXPENSE: "3.6",
            FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "4",
            FinancialConcept.CAPITAL_EXPENDITURES: "5",
            FinancialConcept.ACCOUNTS_RECEIVABLE: "15",
            FinancialConcept.INVENTORY: "10",
            FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: "5",
            FinancialConcept.ACCOUNTS_PAYABLE: "8",
            FinancialConcept.ACCRUED_LIABILITIES: "4",
            FinancialConcept.DEFERRED_REVENUE_CURRENT: "2",
        },
        2024: {
            FinancialConcept.REVENUE: "120",
            FinancialConcept.OPERATING_INCOME: "30",
            FinancialConcept.PRETAX_INCOME: "24",
            FinancialConcept.INCOME_TAX_EXPENSE: "4.8",
            FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "5",
            FinancialConcept.CAPITAL_EXPENDITURES: "6",
            FinancialConcept.ACCOUNTS_RECEIVABLE: "18",
            FinancialConcept.INVENTORY: "12",
            FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: "6",
            FinancialConcept.ACCOUNTS_PAYABLE: "9",
            FinancialConcept.ACCRUED_LIABILITIES: "5",
            FinancialConcept.DEFERRED_REVENUE_CURRENT: "2",
        },
    }
    return NormalizedCompanyFinancials(
        provider="test",
        company_id="1",
        company_name="Example",
        ticker="EX",
        observations=[
            _observation(concept, value, year)
            for year, year_values in values.items()
            for concept, value in year_values.items()
        ],
    )


class _Alpha:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def get_earnings_estimates(self, *_args, **_kwargs):
        if self.error:
            raise RuntimeError(self.error)
        return self.response


class _Yahoo:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def get_revenue_estimate(self, *_args, **_kwargs):
        if self.error:
            raise RuntimeError(self.error)
        return self.response


def _alpha_response():
    return EarningsEstimatesResponse.model_validate(
        {
            "symbol": "EX",
            "annualEarnings": [
                {
                    "fiscalDateEnding": "2025-12-31",
                    "estimatedRevenue": "144",
                    "numberOfAnalysts": "12",
                },
                {
                    "fiscalDateEnding": "2026-12-31",
                    "estimatedRevenue": "158.4",
                    "numberOfAnalysts": "12",
                },
            ],
        }
    )


def _yahoo_table():
    return pd.DataFrame(
        {
            "avg": [Decimal("144"), Decimal("158.4")],
            "low": [Decimal("140"), Decimal("154")],
            "high": [Decimal("148"), Decimal("163")],
            "numberOfAnalysts": [8, 8],
        },
        index=["0y", "+1y"],
    )


def test_alpha_annual_estimates_normalize_and_derive_growth():
    service = ForwardRevenueEstimateService(
        alpha_client=_Alpha(_alpha_response()),
        yahoo_client=_Yahoo(error="must not be called"),
    )
    result = asyncio.run(
        service.resolve(
            "EX",
            forecast_years=(2025, 2026),
            base_revenue=Decimal("120"),
        )
    )

    assert result.selected_provider == "alphavantage"
    assert result.years == (2025, 2026)
    evidence = service.to_growth_evidence(
        result,
        forecast_years=(2025, 2026),
        base_revenue=Decimal("120"),
    )
    assert evidence.growth_path == (Decimal("20"), Decimal("10"))
    assert evidence.guidance is False
    assert evidence.source == "analyst_consensus / alphavantage"


def test_alpha_real_estimates_payload_is_normalized():
    response = {
        "symbol": "EX",
        "estimates": [
            {
                "date": "2025-12-31",
                "horizon": "fiscal year",
                "revenue_estimate_average": "144",
                "revenue_estimate_low": "140",
                "revenue_estimate_high": "148",
                "revenue_estimate_analyst_count": "12",
            },
            {
                "date": "2025-09-30",
                "horizon": "fiscal quarter",
                "revenue_estimate_average": "35",
            },
        ],
    }
    service = ForwardRevenueEstimateService(
        alpha_client=_Alpha(response), yahoo_client=_Yahoo(error="must not be called")
    )

    result = asyncio.run(
        service.resolve("EX", forecast_years=(2025,), base_revenue=Decimal("120"))
    )

    assert result.selected_provider == "alphavantage"
    assert result.years == (2025,)
    assert result.estimates[0].average == Decimal("144")
    assert result.estimates[0].analyst_count == 12


def test_missing_or_empty_alpha_falls_through_to_yahoo():
    response = YahooRevenueEstimateResponse.model_validate(
        {"symbol": "EX", "rows": [
            {"period": "0y", "average": "144", "analyst_count": 5},
            {"period": "+1y", "average": "158.4", "analyst_count": 5},
        ]}
    )
    for alpha_client in (None, _Alpha(EarningsEstimatesResponse(symbol="EX"))):
        service = ForwardRevenueEstimateService(
            alphavantage_api_key=None,
            alpha_client=alpha_client,
            yahoo_client=_Yahoo(response),
        )
        result = asyncio.run(
            service.resolve(
                "EX",
                forecast_years=(2025, 2026),
                current_fiscal_year=2025,
                base_revenue=Decimal("120"),
            )
        )
        assert result.selected_provider == "yahoo"
        assert result.years == (2025, 2026)
        assert result.diagnostics[-1].status.value == "success"


def test_both_forward_providers_unavailable_returns_diagnostics_not_an_error():
    service = ForwardRevenueEstimateService(
        alpha_client=_Alpha(error="rate limit"),
        yahoo_client=_Yahoo(error="unsupported ticker"),
    )
    result = asyncio.run(
        service.resolve("EX", forecast_years=(2025,), base_revenue=Decimal("120"))
    )

    assert isinstance(result, ForwardRevenueEstimateResult)
    assert not result.estimates
    assert [item.provider for item in result.diagnostics] == [
        "alphavantage",
        "yahoo",
    ]
    assert "rate limit" in (result.fallback_reason or "")


def test_yahoo_relative_periods_map_to_issuer_fiscal_years():
    service = ForwardRevenueEstimateService(
        alpha_client=_Alpha(error="no key"),
        yahoo_client=_Yahoo(_yahoo_table()),
    )
    result = asyncio.run(
        service.resolve(
            "EX",
            forecast_years=(2027, 2028),
            current_fiscal_year=2027,
            base_revenue=Decimal("120"),
        )
    )
    assert result.years == (2027, 2028)


def test_consensus_path_is_consumed_by_adaptive_multistage():
    financials = _financials()
    base_service = FcffForecastService()
    parameters = FcffForecastParameters(forecast_years=2)
    seed = base_service.forecast(financials, parameters)
    evidence = ForwardGrowthEvidence(
        guidance=False,
        source="analyst_consensus / yahoo",
        growth_path=(Decimal("20"), Decimal("10")),
        growth_path_by_year=((2025, Decimal("20")), (2026, Decimal("10"))),
    )
    forecast, plan = AdaptiveMultistageFcffForecastService(base_service).forecast(
        financials,
        seed,
        parameters,
        Decimal("3"),
        MultistageValuationConfiguration(terminal_return_on_invested_capital=Decimal("15")),
        forward_evidence=evidence,
    )

    assert [item.revenue_growth for item in forecast.observations[:2]] == [
        Decimal("20"),
        Decimal("10"),
    ]
    assert plan.forward_growth_source == "analyst_consensus / yahoo"
    assert plan.forward_estimates_path == evidence.growth_path


def test_management_growth_path_remains_ahead_of_consensus():
    management = ForwardGrowthEvidence(
        guidance=True,
        growth_path=(Decimal("7"),),
        guidance_growth_path=(Decimal("7"),),
        confidence="high",
    )
    consensus = ForwardGrowthEvidence(
        guidance=False,
        growth_path=(Decimal("20"),),
        source="analyst_consensus / alphavantage",
    )
    merged = cli_module._merge_forward_growth_evidence(
        management,
        consensus,
        management_has_revenue_guidance=False,
    )
    assert merged.guidance
    assert merged.growth_path == management.growth_path


def test_management_and_consensus_paths_merge_by_fiscal_year():
    management = ForwardGrowthEvidence(
        guidance=True,
        source="management_guidance",
        growth_path=(Decimal("7"),),
        growth_path_by_year=((2026, Decimal("7")),),
        guidance_growth_path=(Decimal("7"),),
    )
    consensus = ForwardGrowthEvidence(
        guidance=False,
        source="analyst_consensus / yahoo",
        growth_path=(Decimal("9"), Decimal("10")),
        growth_path_by_year=(
            (2026, Decimal("9")),
            (2027, Decimal("10")),
        ),
        forward_estimate_years=(2026, 2027),
        forward_estimate_growth_path=(Decimal("9"), Decimal("10")),
    )

    merged = cli_module._merge_forward_growth_evidence(
        management,
        consensus,
        management_has_revenue_guidance=True,
        forecast_years=(2026, 2027),
    )

    assert merged.guidance
    assert merged.growth_path == (Decimal("7"), Decimal("10"))
    assert merged.growth_path_by_year == (
        (2026, Decimal("7")),
        (2027, Decimal("10")),
    )
    assert merged.forward_estimate_growth_path == (Decimal("9"), Decimal("10"))


def test_cli_forward_helper_composes_actual_resolver_and_yahoo_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cli_module, "ALPHAVANTAGE_API_KEY", None)
    monkeypatch.setattr(
        ForwardRevenueEstimateService,
        "_yahoo_provider",
        lambda _self: (_Yahoo(_yahoo_table()), False),
    )
    args = SimpleNamespace(
        cache_dir=tmp_path,
        ticker="EX",
        provider_symbol=[],
    )
    forecast = SimpleNamespace(
        observations=[SimpleNamespace(fiscal_year=2025), SimpleNamespace(fiscal_year=2026)],
        current_fiscal_year=2025,
        base_revenue=Decimal("120"),
        unit="USD",
    )
    result = asyncio.run(
        cli_module._retrieve_forward_estimates(args, _financials(), forecast, use_cache=False, make_cache=False)
    )

    assert result.selected_provider == "yahoo"
    assert result.years == (2025, 2026)


def test_cli_resolver_output_reaches_adaptive_multistage(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cli_module, "ALPHAVANTAGE_API_KEY", None)
    monkeypatch.setattr(
        ForwardRevenueEstimateService,
        "_yahoo_provider",
        lambda _self: (_Yahoo(_yahoo_table()), False),
    )
    args = SimpleNamespace(
        cache_dir=tmp_path,
        ticker="EX",
        provider_symbol=[],
    )
    financials = _financials()
    base_service = FcffForecastService()
    parameters = FcffForecastParameters(forecast_years=2)
    seed = base_service.forecast(financials, parameters)
    result = asyncio.run(
        cli_module._retrieve_forward_estimates(
            args,
            financials,
            seed,
            use_cache=False,
            make_cache=False,
        )
    )
    evidence = ForwardRevenueEstimateService.to_growth_evidence(
        result,
        forecast_years=tuple(item.fiscal_year for item in seed.observations),
        base_revenue=seed.base_revenue,
        seed_revenues={item.fiscal_year: item.revenue for item in seed.observations},
    )
    forecast, plan = AdaptiveMultistageFcffForecastService(base_service).forecast(
        financials,
        seed,
        parameters,
        Decimal("3"),
        MultistageValuationConfiguration(
            terminal_return_on_invested_capital=Decimal("15")
        ),
        forward_evidence=evidence,
    )

    assert result.selected_provider == "yahoo"
    assert plan.forward_growth_source == "analyst_consensus / yahoo"
    assert plan.forward_estimate_years == result.years
    assert tuple(item.revenue_growth for item in forecast.observations[:2]) == (
        Decimal("20"),
        Decimal("10"),
    )
