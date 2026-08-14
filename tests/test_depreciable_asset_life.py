import datetime
from decimal import Decimal
from types import SimpleNamespace

from edgarito.cli.__main__ import _resolve_depreciable_asset_life_configuration
from edgarito.config.valuation import MultistageValuationConfiguration
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.valuation.selection import BusinessArchetype
from edgarito.services.valuation import (
    DepreciableAssetLifeResolver,
)


def _observation(concept, value, fiscal_year, *, unit="USD"):
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(value),
        unit=unit,
        granularity=Granularity.ANNUAL,
        fiscal_year=fiscal_year,
        fiscal_period=FiscalPeriod.FY,
        period_end=datetime.date(fiscal_year, 12, 31),
        provider="test",
        taxonomy="test",
        source_concept=concept.value,
    )


def _financials(pairs=()):
    observations = [
        observation
        for year, capex, depreciation in pairs
        for observation in (
            _observation(FinancialConcept.CAPITAL_EXPENDITURES, capex, year),
            _observation(
                FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
                depreciation,
                year,
            ),
        )
    ]
    return NormalizedCompanyFinancials(
        provider="test",
        company_id="TEST",
        company_name="Test Company",
        observations=observations,
    )


def _profile_context(*, industry="Automobiles"):
    return SimpleNamespace(
        industry=industry,
        business_archetype=BusinessArchetype.GENERAL_OPERATING,
        sector=None,
    )


def test_resolver_uses_median_of_positive_annual_implied_lives():
    result = DepreciableAssetLifeResolver().resolve(
        _financials(
            (
                (2022, "10", "2"),
                (2023, "18", "3"),
                (2024, "1000", "1"),
            )
        ),
        business_archetype=BusinessArchetype.GENERAL_OPERATING,
    )

    assert result.value == 6
    assert result.historical_lives == (
        Decimal("5"),
        Decimal("6"),
        Decimal("1000"),
    )
    assert result.historical_median == Decimal("6")
    assert "normalized annual CAPEX/D&A history" in result.source
    assert result.warnings


def test_resolver_bounds_historical_life_to_the_forecast_range():
    result = DepreciableAssetLifeResolver().resolve(
        _financials(((2024, "100", "1"),)),
        business_archetype=BusinessArchetype.GENERAL_OPERATING,
    )

    assert result.value == 30


def test_resolver_uses_automotive_and_general_operating_priors():
    resolver = DepreciableAssetLifeResolver()

    automotive = resolver.resolve(
        _financials(),
        industry="Automobile Manufacturers",
        business_archetype=BusinessArchetype.GENERAL_OPERATING,
    )
    general = resolver.resolve(
        _financials(),
        business_archetype=BusinessArchetype.GENERAL_OPERATING,
    )

    assert automotive.value == 7
    assert "automotive/manufacturing" in automotive.source
    assert general.value == 7
    assert "general-operating" in general.source


def test_resolver_does_not_infer_for_financial_or_asset_archetypes():
    financial = DepreciableAssetLifeResolver().resolve(
        _financials(((2024, "70", "10"),)),
        business_archetype=BusinessArchetype.FINANCIAL_INTERMEDIARY,
    )
    asset = DepreciableAssetLifeResolver().resolve(
        _financials(((2024, "70", "10"),)),
        business_archetype=BusinessArchetype.REIT_PROPERTY,
    )

    assert financial.value is None
    assert asset.value is None


def test_cli_configuration_preserves_explicit_profile_life():
    configuration = MultistageValuationConfiguration(
        depreciable_asset_life_years=4,
    )

    resolved_configuration, resolution = _resolve_depreciable_asset_life_configuration(
        _financials(), _profile_context(), configuration
    )

    assert resolved_configuration.depreciable_asset_life_years == 4
    assert resolution is None


def test_cli_configuration_applies_inferred_life_and_keeps_audit_warning():
    configuration = MultistageValuationConfiguration()

    resolved_configuration, resolution = _resolve_depreciable_asset_life_configuration(
        _financials(), _profile_context(), configuration
    )

    assert resolved_configuration.depreciable_asset_life_years == 7
    assert resolution is not None
    assert resolution.warnings
    assert "inferred automatically" in resolution.audit_message
