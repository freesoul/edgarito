import datetime
from decimal import Decimal

import pytest

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


def test_profile_rejects_mismatched_company_data():
    classification = _classification(Sector.TECHNOLOGY, "Software")
    classification.company_id = "2"

    with pytest.raises(ValueError, match="different company IDs"):
        ValuationProfileBuilder().build(_financials(), classification)
