from decimal import Decimal

import pytest
from pydantic import ValidationError

from edgarito.schemas import (
    CompanyOperatingEconomicsForecast,
    OperatingEconomicsForecast,
    SegmentOperatingEconomicsForecast,
)
from edgarito.schemas.forecasting import FcffForecastParameters, ForecastOverride
from edgarito.schemas.operating import (
    CompanyOperatingForecast,
    EvidenceReference,
    OperatingDriverObservation,
    OperatingSegment,
    SegmentRevenueForecast,
)
from edgarito.services.operating import (
    OperatingEconomicsForecastService,
    OperatingForecastIntegrationService,
    OperatingForecastService,
    RevenueForecastReconciler,
)

D = Decimal


def _segment(name="platform", *, parent_id=None, scope="segment", currency="USD"):
    return OperatingSegment(
        segment_id=name,
        name=name.title(),
        parent_id=parent_id,
        scope=scope,
        currency=currency,
    )


def _revenue(segment, values, *, sources=None):
    years = tuple(values)
    path = tuple(D(value) for value in values.values())
    growth = [None]
    for previous, current in zip(path[:-1], path[1:], strict=True):
        growth.append((current / previous - 1) * 100 if previous else None)
    return SegmentRevenueForecast(
        segment=segment,
        fiscal_years=years,
        revenue=path,
        revenue_growth=tuple(growth),
        source_by_year=sources or {year: "normalized_historical" for year in years},
        confidence_by_year={year: "high" for year in years},
        unit=segment.currency or "currency",
    )


def _observation(
    segment,
    driver,
    year,
    value,
    *,
    unit="USD",
    origin="reported",
    fiscal_period="FY",
    period_key=None,
    scope=None,
    evidence=None,
):
    return OperatingDriverObservation(
        segment_id=segment.segment_id,
        driver_id=driver,
        fiscal_year=year,
        fiscal_period=fiscal_period,
        period_key=period_key,
        value=D(value),
        unit=unit,
        origin=origin,
        confidence="high",
        scope=scope,
        evidence=evidence,
    )


def test_revenue_only_forecast_is_model_dump_identical_and_hybrid_revenue_is_unchanged():
    segment = _segment()
    revenue_only = OperatingForecastService().forecast(
        (segment,), (),
        observations=(_observation(segment, "revenue", 2025, "100"),),
        fiscal_years=(2025,),
    )
    with_gross = OperatingForecastService().forecast(
        (segment,), (),
        observations=(
            _observation(segment, "revenue", 2025, "100"),
            _observation(segment, "gross_margin", 2025, "40", unit="percent"),
        ),
        fiscal_years=(2025,),
    )
    assert "operating_economics" not in revenue_only.model_dump()
    assert revenue_only.consolidated_revenue == with_gross.consolidated_revenue
    assert with_gross.operating_economics is not None


def test_reported_margin_precedes_normalization_and_future_gp_uses_identity():
    segment = _segment()
    forecast = _revenue(segment, {2024: "100", 2025: "125", 2026: "150"})
    result = OperatingEconomicsForecastService().forecast(
        (segment,), forecast,
        observations=(_observation(segment, "gross_margin", 2024, "40", unit="percent"),),
        fiscal_years=(2024, 2025, 2026),
    )
    economics = result.segment_economics[0]
    assert economics.gross_margin == (D("40"), D("40"), D("40"))
    assert economics.gross_profit == (D("40"), D("50"), D("60"))
    assert economics.years[0].source == "mixed"
    assert economics.years[2].expected_gross_profit == D("60")


def test_historical_gp_over_revenue_and_cost_of_revenue_are_deterministic():
    segment = _segment()
    forecast = _revenue(segment, {2024: "100", 2025: "120"})
    result = OperatingEconomicsForecastService().forecast(
        (segment,), forecast,
        observations=(
            _observation(segment, "revenue", 2024, "100"),
            _observation(segment, "gross_profit", 2024, "25"),
        ),
        fiscal_years=(2024, 2025),
    )
    assert result.segment_economics[0].gross_margin == (D("25"), D("25"))
    assert result.segment_economics[0].gross_profit == (D("25"), D("30"))

    cost_result = OperatingEconomicsForecastService().forecast(
        (segment,), forecast,
        observations=(
            _observation(segment, "revenue", 2024, "100"),
            _observation(segment, "cost_of_sales", 2024, "70"),
        ),
        fiscal_years=(2024, 2025),
    )
    assert cost_result.segment_economics[0].gross_profit[0] == D("30")
    assert cost_result.segment_economics[0].gross_margin[0] == D("30")


def test_modeled_revenue_is_never_labeled_historical_derivation():
    segment = _segment()
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2024: "100", 2025: "120"}),
        observations=(_observation(segment, "gross_profit", 2024, "30"),),
        fiscal_years=(2024, 2025),
    )
    economics = result.segment_economics[0]
    assert economics.gross_profit[0] == D("30")
    assert economics.gross_margin == (None, None)
    assert economics.years[0].method == "reported_gross_economics"
    assert economics.years[0].source != "derived_historical"
    assert result.consolidated_gross_margin == (None, None)


@pytest.mark.parametrize(
    "mismatched",
    [
        {"driver_id": "gross_profit", "unit": "EUR"},
        {"driver_id": "gross_profit", "unit": "USD", "scope": "company"},
        {"driver_id": "gross_profit", "unit": "USD", "fiscal_period": "FQ", "period_key": "Q1"},
    ],
)
def test_scope_unit_and_period_mismatches_are_unavailable(mismatched):
    segment = _segment()
    base = _observation(segment, "revenue", 2024, "100")
    mismatch = dict(mismatched)
    driver = mismatch.pop("driver_id")
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2024: "100"}),
        observations=(base, _observation(segment, driver, 2024, "30", **mismatch)),
        fiscal_years=(2024,),
    )
    assert result.segment_economics[0].gross_profit == (None,)
    assert result.segment_economics[0].gross_margin == (None,)


def test_explicit_margin_and_gp_paths_precede_guidance_and_audit_mismatch():
    segment = _segment()
    forecast = _revenue(segment, {2025: "100", 2026: "120"})
    override = ForecastOverride(
        scope="segment",
        scope_id="platform",
        metric="gross_margin",
        strategy="explicit",
        explicit_path=(D("60"),),
        provenance="manual margin scenario",
    )
    result = OperatingEconomicsForecastService().forecast(
        (segment,), forecast,
        observations=(_observation(segment, "gross_margin", 2025, "20", unit="percent", origin="management_guidance"),),
        fiscal_years=(2025, 2026),
        overrides=(override,),
    )
    assert result.segment_economics[0].gross_margin == (D("60"), D("60"))
    assert result.segment_economics[0].gross_profit == (D("60"), D("72"))
    assert result.segment_economics[0].years[0].provenance == "manual margin scenario"

    inconsistent = ForecastOverride(
        scope="segment", scope_id="platform", metric="gross_profit",
        strategy="explicit", explicit_path=(D("50"),),
    )
    mismatch = OperatingEconomicsForecastService().forecast(
        (segment,), forecast, fiscal_years=(2025, 2026),
        overrides=(override, inconsistent),
    )
    assert mismatch.segment_economics[0].gross_profit[0] == D("50")
    assert mismatch.diagnostics.identity_warnings

    with pytest.raises(ValidationError):
        ForecastOverride(
            scope="segment", scope_id="platform", metric="gross_margin",
            strategy="explicit", explicit_path=(D("101"),),
        )
    with pytest.raises(ValueError, match="fiscal horizon"):
        OperatingEconomicsForecastService().forecast(
            (segment,), forecast, fiscal_years=(2025, 2026),
            overrides=(ForecastOverride(
                scope="segment", scope_id="platform", metric="gross_profit",
                strategy="explicit", explicit_path=(D("1"), D("2"), D("3")),
            ),),
        )

    gp_only = OperatingEconomicsForecastService().forecast(
        (segment,), forecast,
        observations=(_observation(segment, "gross_margin", 2025, "20", unit="percent"),),
        fiscal_years=(2025, 2026),
        overrides=(ForecastOverride(
            scope="segment", scope_id="platform", metric="gross_profit",
            strategy="explicit", explicit_path=(D("60"),),
        ),),
    )
    assert gp_only.segment_economics[0].gross_margin == (D("20"), D("20"))
    assert gp_only.segment_economics[0].gross_profit == (D("60"), D("60"))
    assert gp_only.segment_economics[0].diagnostics.identity_warnings


def test_identity_warning_does_not_modify_explicit_gp_and_provenance_is_chained():
    segment = _segment()
    first = EvidenceReference(provider="sec", accession="a", document_name="a.htm")
    second = EvidenceReference(provider="sec", accession="b", document_name="b.htm")
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2024: "100", 2025: "120"}),
        observations=(
            _observation(segment, "revenue", 2024, "100", evidence=first),
            _observation(segment, "gross_profit", 2024, "30", evidence=second),
        ),
        fiscal_years=(2024, 2025),
    )
    year = result.segment_economics[0].years[0]
    assert year.gross_profit == D("30")
    assert year.identity_error == 0
    assert {item.accession for item in year.source_provenance} == {"a", "b"}


def test_margin_and_profit_provenance_are_retained_separately_through_company_sum():
    first, second = _segment("first"), _segment("second")
    first_ref = EvidenceReference(provider="sec", accession="first")
    second_ref = EvidenceReference(provider="sec", accession="second")
    result = OperatingEconomicsForecastService().forecast(
        (first, second),
        (_revenue(first, {2025: "100"}), _revenue(second, {2025: "100"})),
        observations=(
            _observation(first, "gross_margin", 2025, "20", unit="percent", evidence=first_ref),
            _observation(second, "gross_margin", 2025, "30", unit="percent", evidence=second_ref),
        ),
        fiscal_years=(2025,),
        overrides=(
            ForecastOverride(
                scope="segment", scope_id="first", metric="gross_margin",
                strategy="explicit", explicit_path=(D("25"),), provenance="margin override",
            ),
            ForecastOverride(
                scope="segment", scope_id="first", metric="gross_profit",
                strategy="explicit", explicit_path=(D("21"),), provenance="profit override",
            ),
        ),
    )
    segment_year = result.segment_economics[0].years[0]
    company_year = result.years[0]
    assert segment_year.gross_margin_provenance == "margin override"
    assert segment_year.gross_profit_provenance == "profit override"
    assert {item.accession for item in company_year.gross_margin_source_provenance} == {
        "second"
    }
    assert {item.accession for item in company_year.gross_profit_source_provenance} == {
        "second"
    }
    assert "margin override" in company_year.provenance_chain
    assert "profit override" in company_year.provenance_chain


def test_management_cost_guidance_derives_future_gp_with_truthful_method():
    segment = _segment()
    result = OperatingEconomicsForecastService().forecast(
        (segment,),
        _revenue(segment, {2026: "100"}, sources={2026: "independent_operating"}),
        observations=(_observation(
            segment, "cost_of_sales", 2026, "40", origin="management_guidance"
        ), _observation(segment, "gross_profit", 2026, "50")),
        fiscal_years=(2026,),
    )
    year = result.segment_economics[0].years[0]
    assert year.gross_profit == D("60")
    assert year.gross_margin == D("60")
    assert year.source == "management_guidance"
    assert year.method == (
        "forecast_revenue_less_management_cost_over_revenue+"
        "forecast_revenue_less_management_cost_of_revenue"
    )
    assert "gross_profit_precedence=management_guided_cost_over_direct_reported" in year.audit

    inconsistent = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2026: "100"}),
        observations=(
            _observation(segment, "cost_of_sales", 2026, "40", origin="management_guidance"),
            _observation(segment, "gross_margin", 2026, "20", unit="percent"),
            _observation(segment, "gross_profit", 2026, "50"),
        ),
        fiscal_years=(2026,),
    )
    inconsistent_year = inconsistent.segment_economics[0].years[0]
    assert inconsistent_year.gross_margin == D("20")
    assert inconsistent_year.gross_profit == D("60")
    assert inconsistent_year.identity_error == D("40")
    assert inconsistent_year.audit


def test_consolidation_is_revenue_weighted_and_uses_exact_parent_child_selection():
    first, second = _segment("first"), _segment("second")
    forecasts = (
        _revenue(first, {2025: "100"}),
        _revenue(second, {2025: "300"}),
    )
    observations = (
        _observation(first, "gross_margin", 2025, "10", unit="percent"),
        _observation(second, "gross_margin", 2025, "20", unit="percent"),
    )
    result = OperatingEconomicsForecastService().forecast(
        (first, second), forecasts, observations=observations, fiscal_years=(2025,)
    )
    assert result.consolidated_gross_profit == (D("70"),)
    assert result.consolidated_gross_margin == (D("17.5"),)

    parent, child = _segment("parent"), _segment("child", parent_id="parent", scope="product")
    parent_result = OperatingEconomicsForecastService().forecast(
        (parent, child),
        (_revenue(parent, {2025: "100"}), _revenue(child, {2025: "20"})),
        observations=(_observation(parent, "gross_margin", 2025, "50", unit="percent"),),
        fiscal_years=(2025,),
    )
    assert parent_result.consolidated_gross_profit == (D("50"),)
    assert len(parent_result.segment_economics) == 2


def test_consolidated_margin_rejects_uncovered_company_revenue_residual():
    first, second = _segment("first"), _segment("second")
    first_revenue = _revenue(first, {2025: "100"})
    second_revenue = _revenue(second, {2025: "300"})
    company = CompanyOperatingForecast(
        company_id="company",
        fiscal_years=(2025,),
        segment_forecasts=(first_revenue, second_revenue),
        consolidated_revenue=(D("500"),),
        consolidated_growth=(None,),
        source_by_year={2025: "analyst_consensus"},
        confidence_by_year={2025: "high"},
    )
    result = OperatingEconomicsForecastService().forecast(
        (first, second), company,
        observations=(
            _observation(first, "gross_margin", 2025, "10", unit="percent"),
            _observation(second, "gross_margin", 2025, "20", unit="percent"),
        ),
        fiscal_years=(2025,),
    )
    assert result.consolidated_gross_profit == (D("70"),)
    assert result.consolidated_gross_margin == (None,)
    assert any("uncovered residual" in warning for warning in result.warnings)


def test_multiple_consolidated_scopes_are_not_summed_by_economics():
    first = _segment("total_one", scope="consolidated")
    second = _segment("total_two", scope="consolidated")
    result = OperatingEconomicsForecastService().forecast(
        (first, second),
        (_revenue(first, {2025: "100"}), _revenue(second, {2025: "200"})),
        observations=(
            _observation(first, "gross_margin", 2025, "10", unit="percent"),
            _observation(second, "gross_margin", 2025, "20", unit="percent"),
        ),
        fiscal_years=(2025,),
    )
    assert result.consolidated_gross_profit == (None,)
    assert result.consolidated_gross_margin == (None,)
    assert any("Multiple consolidated" in warning for warning in result.warnings)


def test_consolidated_segment_rejects_product_scope_evidence():
    segment = _segment("reported_total", scope="consolidated")
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}),
        observations=(_observation(
            segment, "gross_margin", 2025, "40", unit="percent", scope="product"
        ),),
        fiscal_years=(2025,),
    )
    assert result.segment_economics[0].gross_margin == (None,)
    assert result.consolidated_gross_profit == (None,)
    assert any("scope mismatch" in warning for warning in result.warnings)


def test_unknown_generic_currency_cannot_be_consolidated_with_known_usd():
    generic = _segment("generic", currency=None)
    usd = _segment("usd", currency="USD")
    result = OperatingEconomicsForecastService().forecast(
        (generic, usd),
        (_revenue(generic, {2025: "100"}), _revenue(usd, {2025: "200"})),
        observations=(
            _observation(generic, "gross_margin", 2025, "10", unit="percent"),
            _observation(usd, "gross_margin", 2025, "20", unit="percent"),
        ),
        fiscal_years=(2025,),
    )
    assert result.consolidated_gross_profit == (None,)
    assert result.consolidated_gross_margin == (None,)
    assert any("currency/unit mismatch" in warning for warning in result.warnings)

    all_generic_first = _segment("generic_first", currency=None)
    all_generic_second = _segment("generic_second", currency=None)
    all_generic = OperatingEconomicsForecastService().forecast(
        (all_generic_first, all_generic_second),
        (_revenue(all_generic_first, {2025: "100"}), _revenue(all_generic_second, {2025: "200"})),
        observations=(
            _observation(all_generic_first, "gross_margin", 2025, "10", unit="percent"),
            _observation(all_generic_second, "gross_margin", 2025, "20", unit="percent"),
        ),
        fiscal_years=(2025,),
    )
    assert all_generic.consolidated_gross_profit == (None,)
    assert all_generic.consolidated_gross_margin == (None,)


def test_explicit_economics_targets_must_match_one_supplied_canonical_segment():
    segment = _segment("platform")
    override = ForecastOverride(
        scope="segment", scope_id="missing", metric="gross_margin",
        strategy="explicit", explicit_path=(D("40"),),
    )
    with pytest.raises(ValueError, match="does not match"):
        OperatingEconomicsForecastService().forecast(
            (segment,), _revenue(segment, {2025: "100"}),
            fiscal_years=(2025,), overrides=override,
        )

    with pytest.raises(ValueError, match="ambiguous"):
        OperatingForecastService().forecast(
            (segment, _segment("platform")), (),
            observations=(_observation(segment, "revenue", 2025, "100"),),
            fiscal_years=(2025,),
            overrides=ForecastOverride(
                scope="segment", scope_id="platform", metric="gross_margin",
                strategy="explicit", explicit_path=(D("40"),),
            ),
        )

    with pytest.raises(ValueError, match="ambiguous"):
        OperatingEconomicsForecastService().forecast(
            (segment, _segment("platform")), _revenue(segment, {2025: "100"}),
            fiscal_years=(2025,), overrides=ForecastOverride(
                scope="segment", scope_id="platform", metric="gross_margin",
                strategy="explicit", explicit_path=(D("40"),),
            ),
        )


def test_reconciliation_clears_stale_economics_and_integration_exposes_independent():
    segment = _segment()
    independent = OperatingForecastService().forecast(
        (segment,), (),
        observations=(
            _observation(segment, "revenue", 2025, "100"),
            _observation(segment, "revenue", 2026, "120"),
            _observation(segment, "gross_margin", 2025, "40", unit="percent"),
        ),
        fiscal_years=(2025, 2026),
    )
    assert independent.operating_economics is not None
    reconciled = RevenueForecastReconciler().reconcile_with_details(
        independent, explicit_anchors={2026: D("200")}
    )
    assert reconciled.forecast.operating_economics is None
    assert reconciled.forecast.segment_forecasts[0].operating_economics is None
    assert any("Gross economics cleared" in warning for warning in reconciled.forecast.warnings)

    integrated = OperatingForecastIntegrationService().integrate(
        (segment,), (),
        observations=(
            _observation(segment, "revenue", 2025, "100"),
            _observation(segment, "revenue", 2026, "120"),
            _observation(segment, "gross_margin", 2025, "40", unit="percent"),
        ),
        fiscal_years=(2025, 2026),
        explicit_anchors={2026: D("200")},
        fcff_parameters=FcffForecastParameters(forecast_years=2),
    )
    assert integrated.operating_economics is integrated.independent_forecast.operating_economics
    assert integrated.reconciled_forecast.operating_economics is None


def test_metric_diagnostics_are_independent_and_incomplete_gp_is_not_zero():
    first, second = _segment("first"), _segment("second")
    result = OperatingEconomicsForecastService().forecast(
        (first, second),
        (_revenue(first, {2025: "100"}), _revenue(second, {2025: "200"})),
        observations=(_observation(first, "gross_margin", 2025, "40", unit="percent"),),
        fiscal_years=(2025,),
    )
    assert result.consolidated_gross_profit == (None,)
    assert result.consolidated_gross_margin == (None,)
    assert result.segment_economics[0].diagnostics.gross_margin.coverage == D("1")
    assert result.diagnostics.gross_profit.coverage == D("0")
    assert any("every selected segment" in item for item in result.warnings)


def test_public_economics_contracts_round_trip_and_operating_service_attachment():
    segment = _segment()
    result = OperatingEconomicsForecastService().forecast(
        (segment,), _revenue(segment, {2025: "100"}),
        observations=(_observation(segment, "gross_margin", 2025, "40", unit="percent"),),
        fiscal_years=(2025,),
    )
    assert OperatingEconomicsForecast is CompanyOperatingEconomicsForecast
    assert isinstance(result, CompanyOperatingEconomicsForecast)
    assert isinstance(result.segment_economics[0], SegmentOperatingEconomicsForecast)
    assert CompanyOperatingEconomicsForecast.model_validate_json(result.model_dump_json()) == result


def test_extraction_boundary_remains_evidence_only_and_accepts_negative_margin():
    from edgarito.schemas.operating import (
        ExtractedOperatingEvidenceResponse,
        ExtractedOperatingObservation,
    )

    response = ExtractedOperatingEvidenceResponse(
        observations=[ExtractedOperatingObservation(
            segment_id="platform",
            driver_id="gross_margin",
            fiscal_year=2025,
            value=-20,
            unit="percent",
            supporting_text="Platform gross margin was -20% in FY2025.",
        )]
    )
    assert response.observations[0].value == -20
    negative_profit = ExtractedOperatingEvidenceResponse(
        observations=[ExtractedOperatingObservation(
            segment_id="platform",
            driver_id="gross_income",
            fiscal_year=2025,
            value=-10,
            unit="USD",
            supporting_text="Platform gross income was -10 USD in FY2025.",
        )]
    )
    assert negative_profit.observations[0].value == -10
    with pytest.raises(ValidationError):
        ExtractedOperatingEvidenceResponse.model_validate(
            {"forecasts": [{"fiscal_year": 2026, "gross_profit": 1}]}
        )
