"""Focused coverage for the staged company OPEX/EBIT layer."""

import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.forecasting import (
    ForecastDecision,
    ForecastMetric,
    ForecastOverride,
    ForecastValueBasis,
)
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.operating import (
    CompanyOperatingEconomicsForecast,
    EvidenceReference,
    OperatingDriverObservation,
    OperatingEconomicsForecastConfig,
    OperatingSegment,
    SegmentRevenueForecast,
)
from edgarito.services.operating import (
    OperatingEconomicsForecastService,
    OperatingForecastService,
    normalized_company_financials_to_operating_observations,
)

D = Decimal


def _segment(name="platform", **kwargs):
    return OperatingSegment(segment_id=name, name=name.title(), currency="USD", **kwargs)


def _revenue(segment, values):
    years = tuple(values)
    amounts = tuple(D(value) for value in values.values())
    growth = (None,) + tuple(
        (current / previous - 1) * 100
        for previous, current in zip(amounts[:-1], amounts[1:], strict=True)
    )
    return SegmentRevenueForecast(
        segment=segment,
        fiscal_years=years,
        revenue=amounts,
        revenue_growth=growth,
        source_by_year={year: "independent_operating" for year in years},
        confidence_by_year={year: "high" for year in years},
        unit="USD",
    )


def _observation(segment_id, metric, year, value, *, origin="reported", unit="USD", currency=None, fiscal_period="FY", period_key=None, scope="company", evidence=None):
    return OperatingDriverObservation(
        segment_id=segment_id,
        driver_id=metric,
        fiscal_year=year,
        fiscal_period=fiscal_period,
        period_key=period_key,
        value=D(value),
        unit=unit,
        currency=currency,
        origin=origin,
        confidence="high",
        scope=scope,
        is_total=scope == "company",
        evidence=evidence,
    )


def test_expense_paths_require_basis_and_preserve_one_value_repeat():
    segment = _segment()
    result = OperatingEconomicsForecastService().forecast(
        (segment,),
        _revenue(segment, {2025: "100", 2026: "200"}),
        fiscal_years=(2025, 2026),
        overrides=(
            ForecastOverride(
                scope="company", metric="r_and_d", strategy="explicit",
                explicit_path=(D("10"),), basis=ForecastValueBasis.ABSOLUTE,
            ),
            ForecastOverride(
                scope="company", metric="sg_and_a", strategy="ratio",
                explicit_path=(D("5"),), basis=ForecastValueBasis.PERCENT_OF_REVENUE,
            ),
        ),
    )
    assert result.r_and_d == (D("10"), D("10"))
    assert result.sg_and_a == (D("5"), D("10"))
    assert result.r_and_d_source_by_year[2025] == "explicit"
    with pytest.raises(ValidationError, match="value basis"):
        ForecastOverride(
            scope="company", metric="r_and_d", strategy="explicit", explicit_path=(1,)
        )
    with pytest.raises(ValidationError, match="basis=percent_of_revenue"):
        ForecastOverride(
            scope="company", metric="sg_and_a", strategy="ratio",
            explicit_path=(1,), basis=ForecastValueBasis.ABSOLUTE,
        )
    with pytest.raises(ValueError, match="fiscal horizon"):
        OperatingEconomicsForecastService().forecast(
            (segment,), _revenue(segment, {2025: "100", 2026: "200"}),
            fiscal_years=(2025, 2026), overrides=ForecastOverride(
                scope="company", metric="r_and_d", strategy="explicit",
                explicit_path=(1, 2, 3), basis="absolute",
            ),
        )


def test_expense_precedence_and_weighted_historical_ratio_are_independent():
    segment = _segment()
    observations = (
        _observation("company", "revenue", 2022, 100),
        _observation("company", "r_and_d", 2022, 10),
        _observation("company", "sg_and_a", 2022, 20),
        _observation("company", "revenue", 2023, 200),
        _observation("company", "r_and_d", 2023, 40),
        _observation("company", "sg_and_a", 2023, 40),
        _observation("company", "r_and_d", 2025, 30, origin="first_party_observation"),
        _observation("company", "sg_and_a", 2025, 60, origin="management_guidance"),
    )
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "300", 2026: "400"}),
        observations=observations, fiscal_years=(2025, 2026),
        config=OperatingEconomicsForecastConfig(normalization_method="weighted_recent"),
    )
    assert result.r_and_d[0] == D("30")
    assert result.r_and_d[1].quantize(D("0.01")) == D("66.67")
    assert result.sg_and_a == (D("60"), D("80"))
    assert result.diagnostics.r_and_d.historical_years == (2022, 2023)
    assert result.diagnostics.sg_and_a.normalized_ratio == D("20")


def test_company_expenses_are_not_allocated_or_double_counted_with_segments():
    first, second = _segment("first"), _segment("second")
    result = OperatingEconomicsForecastService().forecast(
        (first, second), (_revenue(first, {2025: "100"}), _revenue(second, {2025: "100"})),
        fiscal_years=(2025,), observations=(
            _observation("company", "r_and_d", 2025, 50),
            _observation("first", "r_and_d", 2025, 10, scope="segment"),
            _observation("second", "r_and_d", 2025, 20, scope="segment"),
        ),
    )
    assert result.r_and_d == (D("50"),)
    assert result.segment_economics[0].r_and_d == (D("10"),)
    assert result.segment_economics[1].r_and_d == (D("20"),)
    assert result.segment_economics[0].diagnostics.r_and_d.coverage == D("1")
    assert any("overlaps segment evidence" in item for item in result.warnings)


def test_exhaustive_segment_sum_and_company_residual_are_explicit_scope_modes():
    first, second = _segment("first"), _segment("second")
    result = OperatingEconomicsForecastService().forecast(
        (first, second), (_revenue(first, {2025: "100"}), _revenue(second, {2025: "100"})),
        fiscal_years=(2025,), overrides=(
            ForecastOverride(scope="segment", scope_id="first", metric="r_and_d", strategy="explicit", explicit_path=(10,), basis="absolute"),
            ForecastOverride(scope="segment", scope_id="second", metric="r_and_d", strategy="explicit", explicit_path=(20,), basis="absolute"),
        ),
    )
    assert result.r_and_d == (D("30"),)
    with pytest.raises(ValueError, match="ambiguous"):
        OperatingEconomicsForecastService().forecast(
            (first, second), (_revenue(first, {2025: "100"}), _revenue(second, {2025: "100"})),
            fiscal_years=(2025,), overrides=(
                ForecastOverride(scope="segment", scope_id="first", metric="r_and_d", strategy="explicit", explicit_path=(10,), basis="absolute"),
                ForecastOverride(scope="segment", scope_id="first", metric="r_and_d", strategy="explicit", explicit_path=(11,), basis="absolute"),
            ),
        )


def test_signed_other_items_stability_zero_policy_and_ebit_identity():
    segment = _segment()
    years = (2022, 2023, 2024, 2025)
    revenues = (D(100), D(110), D(120), D(130))
    observations = []
    for year, revenue in zip(years[:3], revenues[:3], strict=True):
        observations.extend((
            _observation("company", "revenue", year, revenue),
            _observation("company", "gross_profit", year, revenue * D(".6")),
            _observation("company", "r_and_d", year, revenue * D(".1")),
            _observation("company", "sg_and_a", year, revenue * D(".2")),
            _observation("company", "operating_income", year, revenue * D(".35")),
        ))
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, dict(zip(years, revenues, strict=True))),
        observations=observations, fiscal_years=years,
    )
    assert result.other_operating_items == (D("5"), D("5.5"), D("6"), D("6.5"))
    assert result.ebit == (D("35"), D("38.5"), D("42"), D("45.5"))
    assert result.years[-1].ebit == (
        result.years[-1].gross_profit - result.years[-1].r_and_d
        - result.years[-1].sg_and_a + result.years[-1].other_operating_items
    )
    assert result.diagnostics.ebit.supported_years == years

    unavailable = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}), fiscal_years=(2025,)
    )
    assert unavailable.other_operating_items == (None,)
    assert unavailable.ebit == (None,)


def test_reported_ebit_is_reconciliation_target_and_event_items_need_support():
    segment = _segment()
    evidence = EvidenceReference(provider="sec", accession="a", document_name="10-K")
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2024: "100"}), fiscal_years=(2024,), observations=(
            _observation("company", "gross_profit", 2024, 60, evidence=evidence),
            _observation("company", "r_and_d", 2024, 10, evidence=evidence),
            _observation("company", "sg_and_a", 2024, 20, evidence=evidence),
            _observation("company", "operating_income", 2024, 25, evidence=evidence),
        ),
    )
    assert result.years[0].reported_ebit == D("25")
    assert result.years[0].ebit_reconstruction_error == D("0")
    assert result.years[0].other_operating_items == D("-5")
    assert result.years[0].other_operating_items_provenance == evidence


def test_period_unit_currency_and_negative_expense_evidence_do_not_enter_selection():
    segment = _segment()
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}), fiscal_years=(2025,), observations=(
            _observation("company", "r_and_d", 2025, 10, unit="EUR"),
            _observation("company", "sg_and_a", 2025, 10, unit="USD/year"),
            _observation("company", "other_operating_items", 2025, 3, unit="USD", scope="company"),
        ),
    )
    assert result.r_and_d == (None,)
    assert result.sg_and_a == (None,)
    assert result.other_operating_items == (D("3"),)
    assert any("incompatible period, unit, currency" in item for item in result.warnings)
    with pytest.raises(ValueError, match="cannot be negative"):
        OperatingEconomicsForecastService().forecast(
            (segment,), _revenue(segment, {2025: "100"}), fiscal_years=(2025,), observations=(
                _observation("company", "r_and_d", 2025, -1),
            ),
        )


def test_normalized_financials_adapter_is_company_only_and_preserves_metadata():
    reference_date = datetime.date(2025, 2, 1)
    facts = []
    for concept, value in (
        (FinancialConcept.REVENUE, 100),
        (FinancialConcept.GROSS_PROFIT, 60),
        (FinancialConcept.RESEARCH_AND_DEVELOPMENT_EXPENSE, -10),
        (FinancialConcept.SELLING_GENERAL_AND_ADMINISTRATIVE_EXPENSE, 20),
        (FinancialConcept.OPERATING_INCOME, 25),
    ):
        facts.append(FinancialObservation(
            concept=concept,
            statement=concept.statement,
            value=D(value),
            unit="USD",
            granularity=Granularity.ANNUAL,
            fiscal_year=2024,
            fiscal_period=FiscalPeriod.FY,
            period_end=datetime.date(2024, 12, 31),
            provider="sec",
            taxonomy="us-gaap",
            source_concept=concept.value,
            accession_number="abc",
            filed=reference_date,
        ))
    normalized = NormalizedCompanyFinancials(
        provider="sec", company_id="1", company_name="Example", observations=facts
    )
    observations = normalized_company_financials_to_operating_observations(
        normalized, as_of=datetime.date(2025, 2, 2)
    )
    assert {item.segment_id for item in observations} == {"company"}
    assert next(item for item in observations if item.driver_id == "r_and_d").value == D("10")
    assert next(item for item in observations if item.driver_id == "r_and_d").provenance.accession == "abc"
    assert ForecastMetric.OTHER_OPERATING_ITEMS.value == "other_operating_items"


def test_operating_forecast_service_attaches_company_economics_without_segments():
    observations = (
        _observation("company", "revenue", 2024, 100),
        _observation("company", "gross_profit", 2024, 60),
        _observation("company", "r_and_d", 2024, 10),
        _observation("company", "sg_and_a", 2024, 20),
        _observation("company", "operating_income", 2024, 25),
    )
    result = OperatingForecastService().forecast(
        (), (), observations=observations,
        historical_revenue={2024: D("100"), 2025: D("110")},
        fiscal_years=(2024, 2025),
    )
    assert result.operating_economics is not None
    assert result.operating_economics.r_and_d == (D("10"), D("11"))
    assert result.operating_economics.ebit == (D("25"), D("27.5"))


def test_segment_ratio_paths_use_matching_segment_revenue_each_year():
    first, second = _segment("first"), _segment("second")
    result = OperatingEconomicsForecastService().forecast(
        (first, second),
        (
            _revenue(first, {2025: "100", 2026: "200"}),
            _revenue(second, {2025: "300", 2026: "400"}),
        ),
        fiscal_years=(2025, 2026),
        overrides=(
            ForecastOverride(
                scope="segment", scope_id="first", metric="r_and_d",
                strategy="ratio", explicit_path=(10,), basis="percent_of_revenue",
            ),
            ForecastOverride(
                scope="segment", scope_id="second", metric="r_and_d",
                strategy="ratio", explicit_path=(10,), basis="percent_of_revenue",
            ),
        ),
    )
    assert result.segment_economics[0].r_and_d == (D("10"), D("20"))
    assert result.segment_economics[1].r_and_d == (D("30"), D("40"))
    assert result.r_and_d == (D("40"), D("60"))


def test_segment_ratio_sum_requires_every_selected_segment_denominator():
    valid, missing = _segment("valid"), _segment("missing")
    result = OperatingEconomicsForecastService().forecast(
        (valid, missing),
        (
            _revenue(valid, {2025: "100"}),
            _revenue(missing, {2025: "0"}),
        ),
        fiscal_years=(2025,),
        overrides=(
            ForecastOverride(
                scope="segment", scope_id="valid", metric="r_and_d",
                strategy="ratio", explicit_path=(10,), basis="percent_of_revenue",
            ),
            ForecastOverride(
                scope="segment", scope_id="missing", metric="r_and_d",
                strategy="ratio", explicit_path=(10,), basis="percent_of_revenue",
            ),
        ),
    )
    assert result.r_and_d == (None,)
    assert result.segment_economics[1].r_and_d == (None,)
    assert any("not exhaustive" in warning for warning in result.warnings)


def test_ebit_alias_and_explicit_target_never_bypass_identity():
    segment = _segment()
    assert _observation("company", "ebit", 2025, 25).driver_id == "operating_income"
    reported = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}), fiscal_years=(2025,),
        observations=(
            _observation("company", "gross_profit", 2025, 60),
            _observation("company", "r_and_d", 2025, 10),
            _observation("company", "ebit", 2025, 25),
        ),
        overrides=ForecastOverride(
            scope="company", metric="ebit", strategy="explicit",
            explicit_path=(42,), basis="absolute",
        ),
    )
    assert reported.years[0].reported_ebit == D("25")
    assert reported.ebit == (None,)
    assert reported.explicit_ebit_target_by_year == {2025: D("42")}
    assert "calculated_identity_not_overridden" in reported.ebit_audit_by_year.get(2025, ())
    assert any("reconciliation target" in warning for warning in reported.warnings)
    with pytest.raises(ValidationError, match="only segment R&D and SG&A"):
        ForecastOverride(
            scope="segment", scope_id="platform", metric="ebit",
            strategy="explicit", explicit_path=(1,), basis="absolute",
        )
    with pytest.raises(ValidationError, match="only segment R&D and SG&A"):
        ForecastDecision(
            scope="segment", scope_id="platform", metric="operating_income",
            strategy="explicit", explicit_path=(1,), basis="absolute",
        )
    with pytest.raises(ValidationError, match="only segment R&D and SG&A"):
        ForecastOverride(
            scope="segment", scope_id="platform", metric="other_operating_items",
            strategy="explicit", explicit_path=(1,), basis="absolute",
        )


def test_company_gross_repair_updates_metadata_and_removes_stale_diagnostics():
    segment = _segment()
    reference = EvidenceReference(provider="sec", accession="gp-1", document_name="10-K")
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}), fiscal_years=(2025,),
        observations=(_observation("company", "gross_profit", 2025, 60, evidence=reference),),
    )
    assert result.consolidated_gross_profit == (D("60"),)
    assert result.diagnostics.gross_profit.coverage == D("1")
    assert result.source_by_year[2025] == "reported"
    assert result.method_by_year[2025] == "reported_operating_economics"
    assert result.confidence_by_year[2025] == "high"
    assert result.gross_profit_provenance_by_year[2025] == reference
    assert result.gross_profit_source_provenance_by_year[2025] == (reference,)
    assert result.years[0].gross_profit_source_provenance == (reference,)
    assert result.years[0].audit == (
        "company_gross_repair_source=reported",
        "company_gross_repair_method=reported_operating_economics",
    )
    assert not any("consolidated gross profit unavailable" in warning for warning in result.warnings)


def test_currency_conflicts_are_rejected_and_duplicate_denominators_are_deterministic():
    segment = _segment()
    incompatible = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100", 2026: "100"}), fiscal_years=(2025, 2026),
        observations=(
            _observation("company", "r_and_d", 2025, 10, unit="USD", currency="EUR"),
            _observation("company", "r_and_d", 2025, 10, unit="EUR"),
        ),
    )
    assert incompatible.r_and_d == (None, None)
    assert any("currency" in warning for warning in incompatible.warnings)

    duplicate = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100", 2026: "100"}), fiscal_years=(2025, 2026),
        observations=(
            _observation("company", "revenue", 2024, 100, scope="company"),
            _observation("company", "revenue", 2024, 200, scope="company"),
            _observation("company", "r_and_d", 2024, 20),
        ),
    )
    assert duplicate.r_and_d[0] == D("20")

    period_mismatch = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100", 2026: "100"}), fiscal_years=(2025, 2026),
        observations=(
            _observation("company", "revenue", 2024, 100, fiscal_period="FQ", period_key="Q1"),
            _observation("company", "r_and_d", 2024, 20),
        ),
    )
    assert period_mismatch.r_and_d == (None, None)


def test_residual_window_excludes_old_outlier_from_stability_and_audit():
    segment = _segment()
    years = tuple(range(2019, 2026))
    revenues = {year: D("100") for year in years}
    observations = []
    for year in years[:-1]:
        residual_ratio = D("80") if year == 2019 else D("5")
        observations.extend((
            _observation("company", "revenue", year, 100),
            _observation("company", "gross_profit", year, 60),
            _observation("company", "r_and_d", year, 10),
            _observation("company", "sg_and_a", year, 20),
            _observation("company", "operating_income", year, 35 + residual_ratio - 5),
        ))
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, revenues), observations=observations,
        fiscal_years=years,
        config=OperatingEconomicsForecastConfig(historical_window=2),
    )
    assert result.other_operating_items[-1] == D("5")
    assert result.diagnostics.other_operating_items.normalized_ratio == D("5")
    assert result.diagnostics.other_operating_items.historical_years == (2023, 2024)


def test_explicit_ebit_target_completes_other_items_without_bypassing_identity():
    segment = _segment()
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}), fiscal_years=(2025,),
        observations=(_observation("company", "gross_profit", 2025, 60),),
        overrides=(
            ForecastOverride(scope="company", metric="r_and_d", strategy="explicit", explicit_path=(10,), basis="absolute"),
            ForecastOverride(scope="company", metric="sg_and_a", strategy="explicit", explicit_path=(20,), basis="absolute"),
            ForecastOverride(scope="company", metric="ebit", strategy="explicit", explicit_path=(40,), basis="absolute"),
        ),
    )
    assert result.other_operating_items == (D("10"),)
    assert result.ebit == (D("40"),)
    assert result.other_operating_items_method_by_year[2025] == "explicit_ebit_target_residual"
    assert result.explicit_ebit_reconstruction_error_by_year == {2025: D("0")}


def test_explicit_other_items_are_retained_and_explicit_ebit_mismatch_is_audited():
    segment = _segment()
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}), fiscal_years=(2025,),
        observations=(_observation("company", "gross_profit", 2025, 60),),
        overrides=(
            ForecastOverride(scope="company", metric="r_and_d", strategy="explicit", explicit_path=(10,), basis="absolute"),
            ForecastOverride(scope="company", metric="sg_and_a", strategy="explicit", explicit_path=(20,), basis="absolute"),
            ForecastOverride(scope="company", metric="other_operating_items", strategy="explicit", explicit_path=(5,), basis="absolute"),
            ForecastOverride(scope="company", metric="ebit", strategy="explicit", explicit_path=(40,), basis="absolute"),
        ),
    )
    assert result.other_operating_items == (D("5"),)
    assert result.ebit == (D("35"),)
    assert result.explicit_ebit_reconstruction_error_by_year == {2025: D("5")}
    assert result.ebit_reconstruction_error_by_year == {2025: D("5")}
    assert any("selected EBIT target" in warning for warning in result.warnings)


def test_explicit_ebit_precedes_reported_target_but_keeps_separate_errors():
    segment = _segment()
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}), fiscal_years=(2025,),
        observations=(
            _observation("company", "gross_profit", 2025, 60),
            _observation("company", "operating_income", 2025, 25),
        ),
        overrides=(
            ForecastOverride(scope="company", metric="r_and_d", strategy="explicit", explicit_path=(10,), basis="absolute"),
            ForecastOverride(scope="company", metric="sg_and_a", strategy="explicit", explicit_path=(20,), basis="absolute"),
            ForecastOverride(scope="company", metric="ebit", strategy="explicit", explicit_path=(40,), basis="absolute"),
        ),
    )
    assert result.ebit == (D("40"),)
    assert result.other_operating_items == (D("10"),)
    assert result.reported_ebit_by_year == {2025: D("25")}
    assert result.explicit_ebit_reconstruction_error_by_year == {2025: D("0")}
    assert result.reported_ebit_reconstruction_error_by_year == {2025: D("15")}


def test_opex_stage_preserves_selected_gross_and_blocks_future_company_gross_leakage():
    segment = _segment()
    selected = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}), fiscal_years=(2025,),
        observations=(_observation("company", "gross_profit", 2025, 60),),
        overrides=(
            ForecastOverride(scope="segment", scope_id="platform", metric="gross_profit", strategy="explicit", explicit_path=(50,), provenance="gross plan"),
            ForecastOverride(scope="company", metric="r_and_d", strategy="explicit", explicit_path=(10,), basis="absolute"),
            ForecastOverride(scope="company", metric="sg_and_a", strategy="explicit", explicit_path=(20,), basis="absolute"),
            ForecastOverride(scope="company", metric="other_operating_items", strategy="explicit", explicit_path=(0,), basis="absolute"),
        ),
    )
    assert selected.consolidated_gross_profit == (D("50"),)
    assert selected.years[0].gross_profit_provenance == "gross plan"

    leakage = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100", 2026: "200"}), fiscal_years=(2025, 2026),
        observations=(
            _observation("company", "revenue", 2024, 100),
            _observation("company", "gross_profit", 2024, 50),
            _observation("company", "revenue", 2026, 200),
            _observation("company", "gross_profit", 2026, 180),
            _observation("company", "r_and_d", 2025, 10),
            _observation("company", "sg_and_a", 2025, 20),
            _observation("company", "other_operating_items", 2025, 0),
        ),
    )
    assert leakage.consolidated_gross_profit == (D("50"), D("180"))


def test_company_explicit_margin_precedes_reported_gp_and_survives_opex_stage():
    segment = _segment()
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}), fiscal_years=(2025,),
        observations=(
            _observation("company", "gross_profit", 2025, 30),
            _observation("company", "r_and_d", 2025, 10),
            _observation("company", "sg_and_a", 2025, 20),
        ),
        overrides=(
            ForecastOverride(
                scope="company", metric="gross_margin", strategy="explicit",
                explicit_path=(50,), provenance="company margin",
            ),
        ),
    )
    assert result.consolidated_gross_margin == (D("50"),)
    assert result.consolidated_gross_profit == (D("50"),)
    assert result.gross_margin_provenance_by_year[2025] == "company margin"
    assert result.gross_profit_provenance_by_year[2025] == "company margin"
    assert result.years[0].method == "forecast_plan_explicit_company_gross_margin"


def test_company_explicit_gp_precedes_reported_gp_and_derives_margin():
    segment = _segment()
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}), fiscal_years=(2025,),
        observations=(_observation("company", "gross_profit", 2025, 20),),
        overrides=(
            ForecastOverride(
                scope="company", metric="gross_profit", strategy="explicit",
                explicit_path=(60,), provenance="company GP",
            ),
        ),
    )
    assert result.consolidated_gross_profit == (D("60"),)
    assert result.consolidated_gross_margin == (D("60"),)
    assert result.gross_profit_provenance_by_year[2025] == "company GP"
    assert result.gross_margin_provenance_by_year[2025] == "company GP"


def test_company_explicit_gross_without_provenance_round_trips_without_none_entries():
    segment = _segment()
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}), fiscal_years=(2025,),
        overrides=(
            ForecastOverride(
                scope="company", metric="gross_margin", strategy="explicit",
                explicit_path=(50,),
            ),
        ),
    )

    assert result.gross_margin_provenance_by_year == {}
    assert result.gross_profit_provenance_by_year == {}
    assert CompanyOperatingEconomicsForecast.model_validate_json(
        result.model_dump_json()
    ) == result


def test_company_explicit_margin_and_gp_keep_both_values_and_audit_mismatch():
    segment = _segment()
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}), fiscal_years=(2025,),
        overrides=(
            ForecastOverride(
                scope="company", metric="gross_margin", strategy="explicit",
                explicit_path=(50,), provenance="company margin",
            ),
            ForecastOverride(
                scope="company", metric="gross_profit", strategy="explicit",
                explicit_path=(60,), provenance="company GP",
            ),
        ),
    )
    assert result.consolidated_gross_margin == (D("50"),)
    assert result.consolidated_gross_profit == (D("60"),)
    assert result.years[0].identity_error == D("10")
    assert result.diagnostics.gross_profit.reconstruction_error == D("10")
    assert result.diagnostics.gross_margin.reconstruction_error == D("10")
    assert any("differ by 10" in warning for warning in result.warnings)
    assert "company_gross_identity_error=10" in result.audit_by_year[2025]
    round_trip = CompanyOperatingEconomicsForecast.model_validate_json(
        result.model_dump_json()
    )
    assert round_trip == result
