import asyncio
import datetime
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import edgarito.cli.__main__ as cli_module
from edgarito.cli import main
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.classification import (
    NormalizedCompanyClassification,
    Sector,
)
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.valuation import (
    BusinessArchetype,
    CompanyLifecycle,
    Cyclicality,
    DataReadiness,
    ForecastProfile,
    ModelRole,
    RelativeValuationBasis,
    ValuationInput,
    ValuationModel,
    ValuationModelSelector,
    ValuationProfileBuilder,
    ValuationProfileOverrides,
)

FIXTURE = Path(__file__).parent / "fixtures" / "aapl_facts.json"


def _observation(
    concept: FinancialConcept, value: str, fiscal_year: int
) -> FinancialObservation:
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(value),
        unit="USD",
        granularity=Granularity.ANNUAL,
        fiscal_year=fiscal_year,
        fiscal_period=FiscalPeriod.FY,
        period_end=datetime.date(fiscal_year, 12, 31),
        provider="test",
        taxonomy="test",
        source_concept=concept.value,
    )


def _financials(
    *,
    equity: str = "100",
    net_income: tuple[str, ...] = ("10", "12", "14"),
    revenue: tuple[str, ...] = ("100", "110", "120"),
    cash_flow: tuple[str, ...] = ("15", "17", "19"),
) -> NormalizedCompanyFinancials:
    years = range(2022, 2025)
    observations = []
    for index, year in enumerate(years):
        values = {
            FinancialConcept.REVENUE: revenue[index],
            FinancialConcept.NET_INCOME: net_income[index],
            FinancialConcept.OPERATING_CASH_FLOW: cash_flow[index],
            FinancialConcept.CAPITAL_EXPENDITURES: "5",
            FinancialConcept.STOCKHOLDERS_EQUITY: equity,
            FinancialConcept.TOTAL_ASSETS: "200",
            FinancialConcept.TOTAL_LIABILITIES: "100",
        }
        observations.extend(
            _observation(concept, value, year) for concept, value in values.items()
        )
    return NormalizedCompanyFinancials(
        provider="test",
        company_id="0000000001",
        company_name="Test Company",
        ticker="TEST",
        observations=observations,
    )


def _classification(sector: Sector, industry: str):
    return NormalizedCompanyClassification(
        provider="test",
        company_id="0000000001",
        company_name="Test Company",
        ticker="TEST",
        sector=sector,
        industry=industry,
        source_sector=sector.value,
        source_industry=industry,
        industry_taxonomy="test",
    )


def _model(selection, model: ValuationModel):
    return next(result for result in selection.models if result.model == model)


def test_general_operating_company_selects_fcff_and_reports_missing_inputs():
    profile = ValuationProfileBuilder().build(
        _financials(), _classification(Sector.TECHNOLOGY, "Software Infrastructure")
    )
    selection = ValuationModelSelector().select(profile)

    assert profile.business_archetype == BusinessArchetype.GENERAL_OPERATING
    assert profile.lifecycle == CompanyLifecycle.MATURE
    assert selection.primary.model == ValuationModel.FCFF_DCF
    assert selection.primary.forecast_profile == ForecastProfile.STANDARD
    assert selection.primary.data_readiness == DataReadiness.BLOCKED
    assert ValuationInput.FCFF_FORECAST in selection.primary.missing_inputs
    assert _model(selection, ValuationModel.COMPARABLE_MULTIPLES).role == (
        ModelRole.CROSSCHECK
    )


def test_bank_selects_residual_income_and_rejects_fcff():
    profile = ValuationProfileBuilder().build(
        _financials(), _classification(Sector.FINANCIALS, "Banks - Diversified")
    )
    selection = ValuationModelSelector().select(profile)

    assert profile.business_archetype == BusinessArchetype.FINANCIAL_INTERMEDIARY
    assert selection.primary.model == ValuationModel.RESIDUAL_INCOME
    fcff = _model(selection, ValuationModel.FCFF_DCF)
    assert fcff.role == ModelRole.NOT_RECOMMENDED
    assert fcff.data_readiness == DataReadiness.NOT_APPLICABLE
    assert fcff.hard_rejections
    multiples = _model(selection, ValuationModel.COMPARABLE_MULTIPLES)
    assert multiples.relative_bases[:2] == (
        RelativeValuationBasis.PRICE_TO_TANGIBLE_BOOK,
        RelativeValuationBasis.PRICE_TO_BOOK,
    )


def test_reit_selects_nav_and_affo_crosschecks():
    profile = ValuationProfileBuilder().build(
        _financials(),
        _classification(Sector.REAL_ESTATE, "Industrial REIT"),
    )
    selection = ValuationModelSelector().select(profile)

    assert profile.business_archetype == BusinessArchetype.REIT_PROPERTY
    assert selection.primary.model == ValuationModel.NAV_SOTP
    assert ValuationInput.AFFO in selection.primary.missing_inputs
    assert (
        RelativeValuationBasis.PRICE_TO_AFFO
        in _model(selection, ValuationModel.COMPARABLE_MULTIPLES).relative_bases
    )


def test_semiconductor_uses_a_normalized_cycle_fcff_profile():
    profile = ValuationProfileBuilder().build(
        _financials(), _classification(Sector.TECHNOLOGY, "Semiconductors")
    )
    selection = ValuationModelSelector().select(profile)

    assert profile.cyclicality == Cyclicality.HIGH
    assert selection.primary.model == ValuationModel.FCFF_DCF
    assert selection.primary.forecast_profile == ForecastProfile.NORMALIZED_CYCLE


def test_unknown_profile_overrides_defer_to_financial_inference():
    profile = ValuationProfileBuilder().build(
        _financials(),
        _classification(Sector.CONSUMER_DISCRETIONARY, "Automobile Manufacturers"),
        ValuationProfileOverrides(
            lifecycle=CompanyLifecycle.UNKNOWN,
            cyclicality=Cyclicality.UNKNOWN,
        ),
    )

    assert profile.lifecycle == CompanyLifecycle.MATURE
    assert profile.cyclicality == Cyclicality.HIGH
    assert "Lifecycle inferred as mature" in profile.inference_notes
    assert "Cyclicality inferred as high" in profile.inference_notes


def test_unprofitable_growth_company_uses_revenue_to_margin_profile():
    profile = ValuationProfileBuilder().build(
        _financials(
            revenue=("100", "130", "170"),
            net_income=("-20", "-10", "-5"),
            cash_flow=("-15", "-8", "-2"),
        ),
        _classification(Sector.TECHNOLOGY, "Software - Application"),
    )
    selection = ValuationModelSelector().select(profile)

    assert profile.lifecycle == CompanyLifecycle.UNPROFITABLE_GROWTH
    assert selection.primary.forecast_profile == ForecastProfile.REVENUE_TO_MARGIN
    multiples = _model(selection, ValuationModel.COMPARABLE_MULTIPLES)
    assert multiples.relative_bases[0] == RelativeValuationBasis.EV_TO_REVENUE


def test_explicit_inputs_can_make_the_primary_model_ready():
    required = {
        ValuationInput.FCFF_FORECAST,
        ValuationInput.WACC,
        ValuationInput.TERMINAL_GROWTH,
        ValuationInput.NET_DEBT,
        ValuationInput.DILUTED_SHARES,
    }
    profile = ValuationProfileBuilder().build(
        _financials(),
        _classification(Sector.TECHNOLOGY, "Software Infrastructure"),
        ValuationProfileOverrides(available_inputs=required),
    )
    selection = ValuationModelSelector().select(profile)

    assert selection.primary.data_readiness == DataReadiness.READY
    assert selection.primary.missing_inputs == set()


def test_profile_uses_latest_metrics_and_reported_diluted_shares_for_readiness():
    financials = _financials()
    financials.observations.extend(
        [
            _observation(FinancialConcept.CASH_AND_EQUIVALENTS, "20", 2024),
            _observation(FinancialConcept.SHORT_TERM_DEBT, "5", 2024),
            _observation(FinancialConcept.LONG_TERM_DEBT_CURRENT, "3", 2024),
            _observation(FinancialConcept.LONG_TERM_DEBT_NONCURRENT, "40", 2024),
            _observation(FinancialConcept.GOODWILL, "10", 2024),
            _observation(FinancialConcept.INTANGIBLE_ASSETS_NET, "5", 2024),
            _observation(
                FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES, "50", 2024
            ).model_copy(update={"unit": "shares"}),
        ]
    )

    profile = ValuationProfileBuilder().build(
        financials,
        _classification(Sector.TECHNOLOGY, "Software Infrastructure"),
    )

    assert {
        ValuationInput.NET_DEBT,
        ValuationInput.TANGIBLE_BOOK_EQUITY,
        ValuationInput.DILUTED_SHARES,
    } <= profile.available_inputs


def test_negative_book_equity_rejects_residual_income():
    profile = ValuationProfileBuilder().build(
        _financials(equity="-10"),
        _classification(Sector.FINANCIALS, "Insurance - Diversified"),
    )
    selection = ValuationModelSelector().select(profile)

    residual_income = _model(selection, ValuationModel.RESIDUAL_INCOME)
    assert residual_income.role == ModelRole.NOT_RECOMMENDED
    assert residual_income.hard_rejections
    assert selection.primary.model == ValuationModel.EQUITY_DCF


def test_economic_override_takes_precedence_over_official_sector():
    profile = ValuationProfileBuilder().build(
        _financials(),
        _classification(Sector.TECHNOLOGY, "Software Infrastructure"),
        ValuationProfileOverrides(business_archetype=BusinessArchetype.HOLDING_COMPANY),
    )

    assert profile.business_archetype == BusinessArchetype.HOLDING_COMPANY
    assert ValuationModelSelector().select(profile).primary.model == (
        ValuationModel.NAV_SOTP
    )


def test_broad_sector_without_specific_economics_does_not_choose_a_primary():
    profile = ValuationProfileBuilder().build(
        _financials(), _classification(Sector.FINANCIALS, "Financial Services")
    )
    selection = ValuationModelSelector().select(profile)

    assert profile.business_archetype == BusinessArchetype.UNRESOLVED
    assert selection.primary is None
    assert all(model.role != ModelRole.PRIMARY for model in selection.models)
    assert all(model.limitations for model in selection.models)


def test_profile_rejects_mismatched_company_data():
    classification = _classification(Sector.TECHNOLOGY, "Software")
    classification.company_id = "2"

    with pytest.raises(ValueError, match="different company IDs"):
        ValuationProfileBuilder().build(_financials(), classification)


def test_cli_reports_valuation_suitability_from_cached_data(
    tmp_path, capsys, monkeypatch
):
    ticker_path = (
        tmp_path
        / "providers"
        / "edgar"
        / "www.sec.gov"
        / "files"
        / "company_tickers.json"
    )
    facts_path = (
        tmp_path
        / "providers"
        / "edgar"
        / "data.sec.gov"
        / "api"
        / "xbrl"
        / "companyfacts"
        / "CIK0000320193.json"
    )
    ticker_path.parent.mkdir(parents=True)
    facts_path.parent.mkdir(parents=True)
    ticker_path.write_text(
        json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}),
        encoding="utf-8",
    )
    facts_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    fmp_profile = [
        {
            "symbol": "AAPL",
            "companyName": "Apple Inc.",
            "cik": "320193",
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }
    ]
    profile_path = tmp_path / "providers" / "fmp" / "AAPL" / "profile.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(json.dumps(fmp_profile), encoding="utf-8")
    monkeypatch.setattr(cli_module, "FMP_API_KEY", "test-api-key")

    exit_code = main(
        [
            "valuation-models",
            "--ticker",
            "AAPL",
            "--cache-dir",
            str(tmp_path),
            "--user-agent",
            "Edgarito Tests (tests@example.com)",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Economic profile: General Operating" in output
    assert "PRIMARY" in output
    assert "FCFF DCF — suitability 90/100; data Partial" in output
    assert "Comparable Multiples" in output


def test_automatic_assumptions_use_danish_yield_and_inflation_for_dkk(
    tmp_path, monkeypatch
):
    calls = []
    risk_free = object()
    inflation = object()

    class FakeEcbClient:
        def __init__(self, cache):
            self.cache = cache

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get_series(self, flow_ref, key, **kwargs):
            calls.append((flow_ref, key, kwargs))
            return risk_free if flow_ref == "IRS" else inflation

    monkeypatch.setattr(cli_module, "EcbClient", FakeEcbClient)
    args = SimpleNamespace(cache_dir=tmp_path, refresh=False, ticker="NVO")

    inputs = asyncio.run(
        cli_module._retrieve_automatic_assumption_inputs(
            args,
            _financials(),
            "DKK",
            needs_wacc=False,
            needs_terminal=True,
        )
    )

    assert inputs["risk_free_series"] is risk_free
    assert inputs["inflation_series"] is inflation
    assert [(flow, key) for flow, key, _ in calls] == [
        ("IRS", "M.DK.L.L40.CI.0000.DKK.N.Z"),
        ("HICP", "M.DK.N.000000.4D0.ANR"),
    ]
    assert all(call[2]["end_period"] == datetime.date.today() for call in calls)
