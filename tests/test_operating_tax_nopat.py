import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from edgarito.schemas.forecasting import (
    ForecastDecision,
    ForecastMetric,
    ForecastOverride,
    ForecastPlan,
    ForecastValueBasis,
)
from edgarito.schemas.guidance.management import (
    GuidanceBasis,
    GuidanceMetric,
    GuidancePeriodType,
    GuidanceQualifier,
    GuidanceScope,
    GuidanceStatus,
    GuidanceValueKind,
    ManagementGuidance,
)
from edgarito.schemas.operating import (
    CompanyOperatingEconomicsForecast,
    EvidenceReference,
    OperatingDriverObservation,
    OperatingEconomicsForecastConfig,
    OperatingEconomicsYear,
    OperatingSegment,
    SegmentOperatingEconomicsForecast,
)
from edgarito.services.forecasting.plan import FcffForecastPlanService
from edgarito.services.guidance.resolver import ManagementGuidanceResolver
from edgarito.services.operating import OperatingTaxNopatEngine
from edgarito.services.operating._forecast.normalization import (
    _management_guidance_to_observation,
    _normalize_management_constraints,
)

D = Decimal


def _guidance(
    *,
    period_type=GuidancePeriodType.FISCAL_YEAR,
    fiscal_quarter=None,
    status=GuidanceStatus.ISSUED,
    evidence_verified=True,
    extraction_confidence=None,
):
    return ManagementGuidance(
        metric=GuidanceMetric.TAX_RATE,
        fiscal_year=2024,
        fiscal_quarter=fiscal_quarter,
        period_type=period_type,
        point=D("30"),
        value_kind=GuidanceValueKind.PERCENTAGE,
        currency=None,
        unit="percent",
        basis=GuidanceBasis.GAAP,
        scope=GuidanceScope.CONSOLIDATED,
        qualifier=GuidanceQualifier.POINT,
        status=status,
        filing_date=datetime.date(2024, 2, 1),
        accession_number="000-guidance",
        filing_form="8-K",
        source_document="ex991.htm",
        source_document_type="EX-99.1",
        supporting_text="We expect a tax rate of 30 percent.",
        evidence_verified=evidence_verified,
        extraction_model="test",
        extraction_confidence=extraction_confidence,
    )


def _unsafe_guidance(status):
    payload = _guidance().model_dump()
    payload["status"] = status
    return ManagementGuidance.model_construct(**payload)


def _base(*, ebit=("20", "25"), years=(2024, 2025)):
    return CompanyOperatingEconomicsForecast(
        company_id="company",
        fiscal_years=years,
        consolidated_revenue=tuple(D("100") for _ in years),
        consolidated_gross_profit=tuple(D("60") for _ in years),
        consolidated_gross_margin=tuple(D("60") for _ in years),
        consolidated_ebit=tuple(D(value) for value in ebit),
    )


def _observation(
    metric,
    year,
    value,
    *,
    unit="USD",
    origin="reported",
    reference=None,
    currency=None,
    confidence="high",
):
    return OperatingDriverObservation(
        segment_id="company",
        driver_id=metric,
        fiscal_year=year,
        value=D(value),
        unit=unit,
        scope="company",
        is_total=True,
        origin=origin,
        confidence=confidence,
        currency=currency,
        evidence=reference,
        provenance=reference,
    )


def test_historical_tax_rates_normalize_and_calculate_operating_nopat():
    result = OperatingTaxNopatEngine().apply(
        _base(),
        (
            _observation("pretax_income", 2022, "100"),
            _observation("income_tax_expense", 2022, "20"),
            _observation("pretax_income", 2023, "200"),
            _observation("income_tax_expense", 2023, "40"),
        ),
    )

    assert result.tax_rate == (D("20"), D("20"))
    assert result.tax == (D("4"), D("5"))
    assert result.nopat == (D("16"), D("20"))
    assert result.diagnostics.tax_rate.historical_rates == (D("20"), D("20"))
    assert result.diagnostics.nopat.completeness == D("1")


def test_historical_normalization_round_trips_outside_forecast_horizon():
    result = OperatingTaxNopatEngine().apply(
        _base(),
        (
            _observation("pretax_income", 2022, "100"),
            _observation("income_tax_expense", 2022, "20"),
        ),
    )
    restored = CompanyOperatingEconomicsForecast.model_validate_json(
        result.model_dump_json()
    )

    assert restored.historical_pretax_income_by_year == {2022: D("100")}
    assert restored.historical_income_tax_expense_by_year == {2022: D("20")}
    assert restored.historical_effective_tax_rate_by_year == {2022: D("20")}
    assert restored == result

    with pytest.raises(ValueError, match="strictly before"):
        CompanyOperatingEconomicsForecast(
            company_id="company",
            fiscal_years=(2024, 2025),
            consolidated_revenue=(D("100"), D("100")),
            consolidated_gross_profit=(D("60"), D("60")),
            consolidated_gross_margin=(D("60"), D("60")),
            historical_effective_tax_rate_by_year={2024: D("20")},
        )


def test_historical_tax_resolves_one_like_for_like_pair_per_year_and_rejects_alternatives():
    same_year_best = (
        _observation("pretax_income", 2022, "100", confidence="low", currency="USD"),
        _observation("income_tax_expense", 2022, "10", confidence="low", currency="USD"),
        _observation("pretax_income", 2022, "100", currency="USD"),
        _observation("income_tax_expense", 2022, "20", currency="USD"),
    )
    selected = OperatingTaxNopatEngine().apply(
        _base(years=(2024,), ebit=("20",)), same_year_best
    )
    assert selected.historical_effective_tax_rate_by_year == {2022: D("20")}

    ambiguous = OperatingTaxNopatEngine().apply(
        _base(years=(2024,), ebit=("20",)),
        (
            _observation("pretax_income", 2022, "100", currency="USD"),
            _observation("income_tax_expense", 2022, "20", currency="USD"),
            _observation("pretax_income", 2022, "100", currency="EUR"),
            _observation("income_tax_expense", 2022, "30", currency="EUR"),
        ),
        config=OperatingEconomicsForecastConfig(
            tax_rate_dispersion_threshold=D("1")
        ),
    )
    assert ambiguous.historical_effective_tax_rate_by_year == {}
    assert any("alternatives are ambiguous" in warning for warning in ambiguous.warnings)


def test_invalid_history_is_excluded_and_median_or_weighted_recent_is_deterministic():
    observations = (
        _observation("pretax_income", 2020, "0"),
        _observation("income_tax_expense", 2020, "1"),
        _observation("pretax_income", 2021, "100"),
        _observation("income_tax_expense", 2021, "-1"),
        _observation("pretax_income", 2022, "100"),
        _observation("income_tax_expense", 2022, "10"),
        _observation("pretax_income", 2023, "100"),
        _observation("income_tax_expense", 2023, "30"),
        _observation("pretax_income", 2019, "100"),
        _observation("income_tax_expense", 2019, "110"),
    )
    config = OperatingEconomicsForecastConfig(
        tax_rate_dispersion_threshold=D("1"),
        tax_rate_normalization_method="weighted_recent",
    )
    result = OperatingTaxNopatEngine().apply(_base(), observations, config=config)

    assert result.historical_effective_tax_rate_by_year == {2022: D("10"), 2023: D("30")}
    assert result.tax_rate == (D("23.33333333333333333333333333"),) * 2
    assert any("not positive" in warning for warning in result.warnings)
    assert any("negative" in warning for warning in result.warnings)
    assert any("exceeds 100" in warning for warning in result.warnings)


def test_explicit_tax_rate_override_requires_percentage_points_and_planner_keeps_basis():
    reference = "manual tax plan"
    override = ForecastOverride(
        scope="company",
        metric=ForecastMetric.TAX_RATE,
        strategy="explicit",
        explicit_path=(D("25"),),
        basis=ForecastValueBasis.PERCENTAGE_POINTS,
        provenance=reference,
    )
    plan = FcffForecastPlanService().plan("normalized", overrides=(override,))
    result = OperatingTaxNopatEngine().apply(_base(), plan=plan)

    assert plan.decision("company", "tax_rate").basis == ForecastValueBasis.PERCENTAGE_POINTS
    assert result.tax_rate == (D("25"), D("25"))
    assert result.tax == (D("5"), D("6.25"))
    assert result.tax_rate_provenance_by_year[2024] == reference

    with pytest.raises(ValidationError, match="basis=percentage_points"):
        ForecastOverride(
            scope="company",
            metric="tax_rate",
            strategy="explicit",
            explicit_path=(25,),
            basis=ForecastValueBasis.PERCENT_OF_REVENUE,
        )


def test_management_tax_rate_precedes_forward_evidence_and_history():
    observations = (
        _observation("pretax_income", 2022, "100"),
        _observation("income_tax_expense", 2022, "10"),
        _observation("tax_rate", 2024, "30", unit="percent", origin="first_party_observation"),
        _observation("tax_rate", 2024, "25", unit="percent", origin="management_guidance"),
    )
    result = OperatingTaxNopatEngine().apply(
        _base(years=(2024,), ebit=("20",)), observations
    )
    assert result.tax_rate == (D("25"),)
    assert result.tax_rate_source_by_year == {2024: "management_guidance"}


@pytest.mark.parametrize(
    "guidance",
    [
        _guidance(status=GuidanceStatus.WITHDRAWN),
        _guidance(evidence_verified=False),
        _unsafe_guidance("superseded"),
        _unsafe_guidance("inactive"),
    ],
)
def test_ineligible_tax_guidance_is_not_adapted(guidance):
    assert _management_guidance_to_observation(guidance) is None


def test_tax_guidance_preserves_fiscal_period_and_provenance():
    quarterly = _guidance(
        period_type=GuidancePeriodType.QUARTER,
        fiscal_quarter=2,
    )
    annual = _guidance()
    quarterly_observation = _management_guidance_to_observation(quarterly)
    annual_observation = _management_guidance_to_observation(annual)

    assert quarterly_observation is not None
    assert quarterly_observation.fiscal_period == "FQ"
    assert quarterly_observation.period_key == "Q2"
    assert annual_observation is not None
    assert annual_observation.fiscal_period == "FY"
    assert annual_observation.period_key is None
    assert quarterly_observation.evidence is not None
    assert quarterly_observation.evidence.accession == "000-guidance"
    assert quarterly_observation.confidence == "medium"


def test_eligible_current_guidance_is_preferred_without_unsupported_confidence():
    guidance = _guidance(extraction_confidence=D("0.95"))
    resolved = ManagementGuidanceResolver().resolve(
        [guidance], as_of=datetime.date(2024, 12, 31)
    )
    observations = _normalize_management_constraints(
        resolved, segments=(), definitions=()
    )
    result = OperatingTaxNopatEngine().apply(_base(years=(2024,), ebit=("20",)), observations)

    assert result.tax_rate == (D("30"),)
    assert result.tax_rate_source_by_year == {2024: "management_guidance"}
    assert result.tax_rate_confidence_by_year == {2024: "high"}


def test_quarterly_tax_guidance_cannot_outrank_fiscal_year_history():
    quarter = _management_guidance_to_observation(
        _guidance(period_type=GuidancePeriodType.QUARTER, fiscal_quarter=2)
    )
    assert quarter is not None
    result = OperatingTaxNopatEngine().apply(
        _base(years=(2024,), ebit=("20",)),
        (
            quarter,
            _observation("pretax_income", 2022, "100"),
            _observation("income_tax_expense", 2022, "20"),
        ),
    )

    assert result.tax_rate == (D("20"),)
    assert result.tax_rate_source_by_year == {2024: "normalized_historical"}


def test_historical_pair_keeps_separate_input_provenance_and_derived_audit():
    pretax_reference = EvidenceReference(provider="sec", accession="pretax")
    tax_reference = EvidenceReference(provider="sec", accession="tax")
    result = OperatingTaxNopatEngine().apply(
        _base(years=(2024,), ebit=("20",)),
        (
            _observation("pretax_income", 2022, "100", reference=pretax_reference),
            _observation("income_tax_expense", 2022, "20", reference=tax_reference),
        ),
    )

    assert result.historical_pretax_income_provenance_by_year[2022] == pretax_reference
    assert result.historical_income_tax_expense_provenance_by_year[2022] == tax_reference
    assert set(result.historical_effective_tax_rate_provenance_chain_by_year[2022]) == {
        pretax_reference,
        tax_reference,
    }
    assert "effective_tax_rate_inputs=pretax_income,income_tax_expense" in result.historical_effective_tax_rate_audit_by_year[2022]
    assert pretax_reference in result.diagnostics.tax_rate.provenance
    assert tax_reference in result.diagnostics.tax_rate.provenance
    assert pretax_reference in result.diagnostics.tax.provenance
    assert tax_reference in result.diagnostics.nopat.provenance


def test_explicit_and_management_provenance_reach_modeled_diagnostics():
    ebit_reference = EvidenceReference(provider="sec", accession="ebit")
    management_reference = EvidenceReference(provider="sec", accession="guidance")
    explicit = ForecastOverride(
        scope="company",
        metric="tax_rate",
        strategy="explicit",
        explicit_path=(D("25"),),
        basis="percentage_points",
        provenance="explicit tax plan",
    )
    base = _base(years=(2024,), ebit=("20",)).model_copy(
        update={
            "ebit_provenance_by_year": {2024: ebit_reference},
            "source_provenance_by_year": {2024: (ebit_reference,)},
        }
    )
    explicit_result = OperatingTaxNopatEngine().apply(base, overrides=explicit)
    assert "explicit tax plan" in explicit_result.diagnostics.tax_rate.provenance
    assert ebit_reference in explicit_result.diagnostics.tax.provenance

    management_result = OperatingTaxNopatEngine().apply(
        base,
        (_observation("tax_rate", 2024, "30", unit="percent", origin="management_guidance", reference=management_reference),),
    )
    assert management_reference in management_result.diagnostics.tax_rate.provenance
    assert management_reference in management_result.diagnostics.nopat.provenance


def test_nopat_uses_only_ebit_and_audits_negative_ebit_without_clamping():
    result = OperatingTaxNopatEngine().apply(
        _base(ebit=("-10", "20")),
        overrides=ForecastOverride(
            scope="company",
            metric="tax_rate",
            strategy="explicit",
            explicit_path=(20,),
            basis="percentage_points",
        ),
    )
    assert result.tax == (D("-2"), D("4"))
    assert result.nopat == (D("-8"), D("16"))
    assert any("negative EBIT" in warning for warning in result.warnings)
    assert "net_income" not in result.years[0].nopat_method


def test_tax_is_company_only_and_tax_stage_round_trips():
    with pytest.raises(ValidationError, match="company-only"):
        ForecastOverride(
            scope="segment",
            scope_id="platform",
            metric="tax_rate",
            strategy="explicit",
            explicit_path=(20,),
            basis="percentage_points",
        )

    result = OperatingTaxNopatEngine().apply(
        _base(),
        (_observation("tax_rate", 2024, "20", unit="percent"),),
    )
    restored = CompanyOperatingEconomicsForecast.model_validate_json(result.model_dump_json())
    assert restored == result
    assert result.segment_economics == ()

    segment = SegmentOperatingEconomicsForecast(
        segment=OperatingSegment(segment_id="platform", name="Platform"),
        fiscal_years=(2024,),
        revenue=(D("100"),),
        gross_margin=(D("50"),),
        gross_profit=(D("50"),),
    )
    assert not hasattr(segment, "tax_rate")
    with pytest.raises(ValueError, match="company-only tax or NOPAT"):
        SegmentOperatingEconomicsForecast(
            segment=OperatingSegment(segment_id="platform", name="Platform"),
            fiscal_years=(2024,),
            revenue=(D("100"),),
            gross_margin=(D("50"),),
            gross_profit=(D("50"),),
            years=(OperatingEconomicsYear(fiscal_year=2024, tax_rate=D("20")),),
        )


def test_tax_and_nopat_overrides_are_identity_derived_but_non_explicit_tax_decision_is_valid():
    with pytest.raises(ValidationError, match="Explicit TAX overrides"):
        ForecastOverride(
            scope="company",
            metric=ForecastMetric.TAX,
            strategy="explicit",
            explicit_path=(20,),
            basis="absolute",
        )
    with pytest.raises(ValidationError, match="Explicit NOPAT overrides"):
        ForecastOverride(
            scope="company",
            metric=ForecastMetric.NOPAT,
            strategy="explicit",
            explicit_path=(D("16"),),
            basis="absolute",
        )

    decision = ForecastDecision(
        scope="company", metric=ForecastMetric.TAX, strategy="consolidated"
    )
    plan = ForecastPlan(
        requested="normalized", resolved="normalized", decisions=(decision,)
    )
    assert OperatingTaxNopatEngine().apply(_base(), plan=plan) == _base()
