"""Focused deterministic tests for the operating reinvestment stage."""

import datetime
from decimal import Decimal

import pytest

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.forecasting import (
    FcffForecast,
    FcffForecastDcfStub,
    FcffForecastParameters,
    FcffForecastYtdAnchor,
    ForecastOverride,
    ForecastSeedType,
)
from edgarito.schemas.guidance.management import MonetaryForecastConstraint
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.operating import (
    CompanyOperatingEconomicsForecast,
    EvidenceReference,
    OperatingDriverObservation,
    OperatingEconomicsForecastConfig,
    OperatingInvestmentProgram,
    OperatingReinvestmentSeed,
    OperatingSegment,
    SegmentOperatingEconomicsForecast,
)
from edgarito.services.forecasting.validation import (
    ForecastValidationService,
    driver_economics_to_validation_context,
)
from edgarito.services.operating import (
    DriverBasedCanonicalFcffAdapter,
    OperatingReinvestmentEngine,
    normalized_company_financials_to_operating_observations,
)

D = Decimal


def _base(years=(2025, 2026), *, ebit=None):
    ebit = ebit or tuple(str(20 + 2 * i) for i, _ in enumerate(years))
    return CompanyOperatingEconomicsForecast(
        company_id="company",
        fiscal_years=years,
        consolidated_revenue=tuple(D("100") + D("10") * i for i, _ in enumerate(years)),
        consolidated_gross_profit=tuple(D("60") + D("6") * i for i, _ in enumerate(years)),
        consolidated_gross_margin=tuple(D("60") for _ in years),
        consolidated_ebit=tuple(D(value) for value in ebit),
        tax_rate=tuple(D("20") for _ in years),
        tax=tuple(D(value) * D(".2") for value in ebit),
        nopat=tuple(D(value) * D(".8") for value in ebit),
        unit="USD",
    )


def _observation(metric, year, value, *, unit="USD", origin="reported", **kwargs):
    return OperatingDriverObservation(
        segment_id=kwargs.pop("segment_id", "company"),
        driver_id=metric,
        fiscal_year=year,
        value=D(value),
        unit=unit,
        scope=kwargs.pop("scope", "company"),
        is_total=kwargs.pop("is_total", True),
        origin=origin,
        confidence=kwargs.pop("confidence", "high"),
        **kwargs,
    )


def _complete_observations():
    return (
        _observation("revenue", 2024, "100"),
        _observation("depreciation_and_amortization", 2024, "4"),
        _observation("depreciation_and_amortization", 2025, "5"),
        _observation("depreciation_and_amortization", 2026, "5.5"),
        _observation("capital_expenditures", 2024, "4"),
        _observation("capital_expenditures", 2025, "6"),
        _observation("capital_expenditures", 2026, "6.6"),
        _observation("operating_working_capital", 2024, "10"),
        _observation("operating_working_capital", 2025, "11"),
        _observation("operating_working_capital", 2026, "9"),
    )


def _ytd_template():
    anchor = FcffForecastYtdAnchor(
        fiscal_year=2025,
        ytd_period_end=datetime.date(2025, 6, 30),
        fiscal_year_end=datetime.date(2025, 12, 31),
        actual_quarters=2,
        actual_revenue=D("60"),
        actual_operating_income=D("12"),
        actual_pretax_income=D("12"),
        actual_income_tax_expense=D("2.4"),
        actual_tax_rate=D("20"),
        actual_depreciation_and_amortization=D("3"),
        actual_capital_expenditures=D("8"),
        actual_operating_working_capital=D("130"),
        latest_annual_revenue=D("100"),
        revenue_growth=D("0"),
        operating_margin=D("20"),
        tax_rate=D("20"),
        depreciation_to_revenue=D("5"),
        capex_to_revenue=D("6"),
        operating_working_capital_to_revenue=D("140"),
    )
    stub = FcffForecastDcfStub(
        forecast_year=1,
        fiscal_year=2025,
        period_start=datetime.date(2025, 6, 30),
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
        annual_nopat=D("16"),
        actual_ytd_nopat=D("9.6"),
        annual_depreciation_and_amortization=D("5"),
        actual_ytd_depreciation_and_amortization=D("3"),
        annual_capital_expenditures=D("6"),
        actual_ytd_capital_expenditures=D("8"),
        fiscal_year_end_operating_working_capital=D("140"),
        actual_ytd_operating_working_capital=D("130"),
        fcff=D("0.4"),
    )
    return FcffForecast(
        provider="fixture",
        company_id="company",
        company_name="Company",
        seed_type=ForecastSeedType.YTD_PLUS_FORECAST,
        seed_methodology="fixture YTD plus forecast",
        seed_period_end=datetime.date(2025, 6, 30),
        current_fiscal_year=2025,
        base_fiscal_year=2025,
        base_period_end=datetime.date(2025, 6, 30),
        base_revenue=D("100"),
        base_operating_income=D("20"),
        base_tax_rate=D("20"),
        base_nopat=D("16"),
        base_depreciation_and_amortization=D("5"),
        base_capital_expenditures=D("6"),
        base_operating_working_capital=D("100"),
        unit="USD",
        parameters=FcffForecastParameters(forecast_years=1),
        historical_fiscal_years=(2024,),
        assumption_sources={},
        ytd_anchor=anchor,
        dcf_stub=stub,
    )


def test_explicit_absolute_and_ratio_paths_repeat_and_manual_wins():
    base = _base()
    result = OperatingReinvestmentEngine().apply(
        base,
        _complete_observations(),
        overrides=(
            ForecastOverride(
                scope="company",
                metric="depreciation_and_amortization",
                strategy="ratio",
                explicit_path=(D("7"),),
                basis="percent_of_revenue",
            ),
            ForecastOverride(
                scope="company",
                metric="capital_expenditures",
                strategy="explicit",
                explicit_path=(D("8"), D("9")),
                basis="absolute",
            ),
        ),
        seed=OperatingReinvestmentSeed(fiscal_year=2024, unit="USD", value=D("10")),
    )
    assert result.depreciation_and_amortization == (D("7"), D("7.7"))
    assert result.capital_expenditures == (D("8"), D("9"))


@pytest.mark.parametrize("metric", ["depreciation_and_amortization", "capital_expenditures", "operating_working_capital"])
def test_reinvestment_path_basis_is_unambiguous(metric):
    with pytest.raises(ValueError, match="basis"):
        ForecastOverride(
            scope="company",
            metric=metric,
            strategy="ratio",
            explicit_path=(D("5"),),
            basis="absolute",
        )


@pytest.mark.parametrize("metric", ["delta_nwc", "fcff"])
def test_derived_delta_and_fcff_overrides_are_rejected(metric):
    with pytest.raises(ValueError, match="derived"):
        ForecastOverride(
            scope="company",
            metric=metric,
            strategy="explicit",
            explicit_path=(D("1"),),
            basis="absolute",
        )


def test_historical_normalization_keeps_years_and_weighted_recent_is_configurable():
    result = OperatingReinvestmentEngine().apply(
        _base(years=(2025,)),
        (
            _observation("revenue", 2022, "100"),
            _observation("revenue", 2023, "100"),
            _observation("revenue", 2024, "100"),
            _observation("depreciation_and_amortization", 2022, "2"),
            _observation("depreciation_and_amortization", 2023, "4"),
            _observation("depreciation_and_amortization", 2024, "6"),
            _observation("capital_expenditures", 2022, "2"),
            _observation("capital_expenditures", 2023, "4"),
            _observation("capital_expenditures", 2024, "6"),
            _observation("operating_working_capital", 2022, "10"),
            _observation("operating_working_capital", 2023, "10"),
            _observation("operating_working_capital", 2024, "10"),
        ),
        config=OperatingEconomicsForecastConfig(normalization_method="weighted_recent"),
        seed=OperatingReinvestmentSeed(fiscal_year=2024, unit="USD", value=D("10")),
    )
    assert result.depreciation_and_amortization == (D("4.666666666666666666666666667"),)
    assert result.diagnostics.depreciation_and_amortization.historical_years == (2022, 2023, 2024)


def test_missing_component_does_not_zero_fill_fcff():
    observations = tuple(item for item in _complete_observations() if item.driver_id != "capital_expenditures")
    result = OperatingReinvestmentEngine().apply(_base(), observations)
    assert result.depreciation_and_amortization[0] == D("5")
    assert result.capital_expenditures[0] is None
    assert result.fcff[0] is None


def test_simple_capex_life_cohort_is_conservative_and_explicitly_audited():
    result = OperatingReinvestmentEngine().apply(
        _base(years=(2025,)),
        (
            _observation("revenue", 2024, "200"),
            _observation("depreciation_and_amortization", 2024, "10"),
            _observation("capital_expenditures", 2024, "8"),
            _observation("capital_expenditures", 2025, "12"),
            _observation("operating_working_capital", 2024, "20"),
            _observation("operating_working_capital", 2025, "20"),
        ),
        config=OperatingEconomicsForecastConfig(depreciable_asset_life_years=4),
        seed=OperatingReinvestmentSeed(fiscal_year=2024, unit="USD", value=D("20")),
    )
    assert result.depreciation_and_amortization == (D("10"),)
    assert "simple_cohort_method=true" in result.depreciation_and_amortization_audit_by_year[2025]


@pytest.mark.parametrize(
    ("constraint", "expected"),
    [
        (MonetaryForecastConstraint(point=D("8")), D("8")),
        (MonetaryForecastConstraint(minimum=D("8")), D("8")),
        (MonetaryForecastConstraint(maximum=D("5")), D("4")),
        (MonetaryForecastConstraint(minimum=D("5"), maximum=D("7")), D("5")),
    ],
)
def test_capex_constraints_preserve_point_and_bound_semantics(constraint, expected):
    result = OperatingReinvestmentEngine().apply(
        _base(years=(2025,)),
        tuple(
            item
            for item in _complete_observations()
            if not (item.driver_id == "capital_expenditures" and item.fiscal_year == 2025)
        ),
        capex_constraints={2025: constraint},
        seed=OperatingReinvestmentSeed(fiscal_year=2024, unit="USD", value=D("10")),
    )
    assert result.capital_expenditures[0] == expected
    assert f"capex_constraint_method={constraint.methodology}" in result.capital_expenditures_audit_by_year[2025]


def test_capex_program_scales_money_but_capacity_only_program_is_audit_only():
    result = OperatingReinvestmentEngine().apply(
        _base(years=(2025,)),
        tuple(
            item
            for item in _complete_observations()
            if not (item.driver_id == "capital_expenditures" and item.fiscal_year == 2025)
        ),
        investment_programs=(
            OperatingInvestmentProgram(
                program_id="facility",
                name="Facility expansion investment",
                fiscal_year=2025,
                value=D("3"),
                scale=D("2"),
                unit="USD",
                purpose="capital spending",
            ),
        ),
        seed=OperatingReinvestmentSeed(fiscal_year=2024, unit="USD", value=D("10")),
    )
    assert result.capital_expenditures[0] == D("6")
    assert "investment_program=facility" in result.capital_expenditures_audit_by_year[2025][1]

    capacity_only = OperatingInvestmentProgram(
        program_id="capacity",
        name="Capacity expansion",
        fiscal_year=2025,
        value=D("20"),
        unit="units",
        purpose="capacity",
    )
    unavailable = OperatingReinvestmentEngine().apply(
        _base(years=(2025,)),
        tuple(item for item in _complete_observations() if item.driver_id != "capital_expenditures"),
        investment_programs=(capacity_only,),
    )
    assert unavailable.capital_expenditures[0] is None


def test_same_filing_programs_are_deduplicated_within_period_but_retained_across_years():
    filing = EvidenceReference(provider="sec", accession="program-filing")
    observations = tuple(
        item
        for item in _complete_observations()
        if not (
            item.driver_id == "capital_expenditures"
            and item.fiscal_year in {2025, 2026}
        )
    )
    programs = (
        OperatingInvestmentProgram(
            program_id="same-program",
            name="Facility investment",
            fiscal_year=2025,
            value=D("3"),
            unit="USD",
            purpose="capital spending",
            evidence=filing,
        ),
        OperatingInvestmentProgram(
            program_id="same-program",
            name="Facility investment",
            fiscal_year=2026,
            value=D("4"),
            unit="USD",
            purpose="capital spending",
            evidence=filing,
        ),
        OperatingInvestmentProgram(
            program_id="same-program",
            name="Facility investment duplicate",
            fiscal_year=2025,
            value=D("99"),
            unit="USD",
            purpose="capital spending",
            evidence=filing,
        ),
    )
    result = OperatingReinvestmentEngine().apply(
        _base(), observations, investment_programs=programs
    )
    assert result.capital_expenditures == (D("3"), D("4"))


@pytest.mark.parametrize(
    ("value", "low", "high", "expected"),
    [
        (D("-20"), None, None, (D("20"), None, None)),
        (None, D("-100"), D("-50"), (None, D("50"), D("100"))),
    ],
)
def test_monetary_program_outflows_use_positive_spending_semantics(
    value, low, high, expected
):
    program = OperatingInvestmentProgram(
        program_id="outflow",
        name="Capital investment",
        fiscal_year=2025,
        value=value,
        low=low,
        high=high,
        unit="USD",
        purpose="capital spending",
    )
    assert (program.value, program.low, program.high) == expected


def test_monetary_program_mixed_sign_range_is_ambiguous():
    with pytest.raises(ValueError, match="mixed signs"):
        OperatingInvestmentProgram(
            program_id="ambiguous",
            name="Capital investment",
            fiscal_year=2025,
            low=D("-10"),
            high=D("20"),
            unit="USD",
            purpose="capital spending",
        )


def test_owc_seed_and_delta_signs_follow_balance_identity():
    result = OperatingReinvestmentEngine().apply(_base(), _complete_observations())
    assert result.change_in_operating_working_capital == (D("1"), D("-2"))
    assert result.fcff[0] == D("14")
    assert result.fcff[1] == D("18.5")


def test_segment_evidence_is_not_allocated_and_requires_exhaustive_nonoverlap():
    segments = tuple(
        SegmentOperatingEconomicsForecast(
            segment=OperatingSegment(segment_id=segment_id, name=segment_id),
            fiscal_years=(2025,),
            revenue=(D("50"),),
            gross_margin=(D("50"),),
            gross_profit=(D("25"),),
        )
        for segment_id in ("a", "b")
    )
    records = (
        _observation("depreciation_and_amortization", 2025, "2", segment_id="a", scope="segment", is_total=False, is_component=True, exhaustive=True),
        _observation("depreciation_and_amortization", 2025, "3", segment_id="b", scope="segment", is_total=False, is_component=True, exhaustive=True),
    )
    result = OperatingReinvestmentEngine().apply(
        _base(years=(2025,)).model_copy(update={"segment_economics": segments}),
        records,
        segments=(segments[0].segment, segments[1].segment),
    )
    assert result.depreciation_and_amortization[0] == D("5")


def test_no_overlap_between_company_and_segment_amounts():
    segment = SegmentOperatingEconomicsForecast(
        segment=OperatingSegment(segment_id="a", name="A"),
        fiscal_years=(2025,), revenue=(D("50"),), gross_margin=(D("50"),), gross_profit=(D("25"),),
    )
    with pytest.raises(ValueError, match="overlap"):
        OperatingReinvestmentEngine().apply(
            _base(years=(2025,)).model_copy(update={"segment_economics": (segment,)}),
            (
                _observation("depreciation_and_amortization", 2025, "5"),
                _observation("depreciation_and_amortization", 2025, "2", segment_id="a", scope="segment", is_total=False, is_component=True),
            ),
        )


def test_segment_ancestor_chain_rejects_parent_and_grandchild_overlap():
    segments = tuple(
        SegmentOperatingEconomicsForecast(
            segment=OperatingSegment(segment_id=segment_id, name=segment_id, parent_id=parent),
            fiscal_years=(2025,),
            revenue=(D("30"),),
            gross_margin=(D("50"),),
            gross_profit=(D("15"),),
        )
        for segment_id, parent in (("parent", None), ("child", "parent"), ("grandchild", "child"))
    )
    records = (
        _observation("depreciation_and_amortization", 2025, "2", segment_id="parent", scope="segment", is_total=False, is_component=True, exhaustive=True),
        _observation("depreciation_and_amortization", 2025, "1", segment_id="grandchild", scope="segment", is_total=False, is_component=True, exhaustive=True),
    )
    with pytest.raises(ValueError, match="Overlapping"):
        OperatingReinvestmentEngine().apply(
            _base(years=(2025,)).model_copy(update={"segment_economics": segments}),
            records,
        )


def test_canonical_mapping_keeps_exact_identity_ratios_and_provenance():
    reference = EvidenceReference(provider="sec", accession="abc")
    result = OperatingReinvestmentEngine().apply(
        _base(years=(2025,)),
        tuple(item.model_copy(update={"provenance": reference, "evidence": reference, "source_provenance": (reference,)}) for item in _complete_observations()),
    )
    mapped = DriverBasedCanonicalFcffAdapter().observations(result, _ytd_template())
    assert mapped[0].fcff == D("14")
    assert mapped[0].depreciation_to_revenue == D("5")
    assert mapped[0].capex_to_revenue == D("6")
    assert mapped[0].cell_audits["capital_expenditures"].source == "reported"
    assert "abc" in mapped[0].cell_audits["capital_expenditures"].method


def test_validation_adapter_is_read_only_and_runs_existing_fcff_rules():
    base = _base(years=(2025,))
    result = OperatingReinvestmentEngine().apply(base, _complete_observations())
    before = result.model_dump()
    context = driver_economics_to_validation_context(result)
    assert context.rows[0].fcff == result.fcff[0]
    assert ForecastValidationService().validate(result).error_count == 0
    assert result.model_dump() == before


def test_stage_not_attempted_serialization_is_unchanged_shape():
    base = _base(years=(2025,))
    dumped = base.model_dump()
    assert "depreciation_and_amortization" not in dumped
    assert "capital_expenditures" not in dumped
    assert "fcff" not in dumped


def test_normalized_financials_map_negative_da_and_capex_as_positive_company_facts():
    filing_date = datetime.date(2025, 2, 1)
    facts = [
        {
            "concept": concept,
            "statement": concept.statement,
            "value": value,
            "unit": "USD",
            "granularity": Granularity.ANNUAL,
            "fiscal_year": 2024,
            "fiscal_period": FiscalPeriod.FY,
            "period_end": datetime.date(2024, 12, 31),
            "provider": "sec",
            "taxonomy": "us-gaap",
            "source_concept": concept.value,
            "accession_number": "accession-1",
            "filed": filing_date,
        }
        for concept, value in (
            (FinancialConcept.REVENUE, D("100")),
            (FinancialConcept.DEPRECIATION_AND_AMORTIZATION, D("-12")),
            (FinancialConcept.CAPITAL_EXPENDITURES, D("-20")),
            (FinancialConcept.GROSS_PROFIT, D("60")),
        )
    ]
    observations = normalized_company_financials_to_operating_observations(
        NormalizedCompanyFinancials(
            provider="sec",
            company_id="company",
            company_name="Company",
            observations=facts,
        ),
        as_of=datetime.date(2025, 2, 2),
    )
    by_driver = {item.driver_id: item for item in observations}
    assert by_driver["depreciation_and_amortization"].value == D("12")
    assert by_driver["capital_expenditures"].value == D("20")
    assert by_driver["depreciation_and_amortization"].provenance.accession == "accession-1"
    assert by_driver["capital_expenditures"].unit == "usd"
    assert all(item.driver_id is not None for item in observations)


def test_ytd_template_uses_full_year_base_owc_and_preserves_stub():
    template = _ytd_template()
    base = _base(years=(2025,)).model_copy(
        update={"consolidated_revenue": (D("100"),)}
    )
    economics = OperatingReinvestmentEngine().apply(
        base,
        (
            _observation("depreciation_and_amortization", 2025, "5"),
            _observation("capital_expenditures", 2025, "6"),
            _observation("operating_working_capital", 2025, "140"),
        ),
        seed=template,
    )
    assert economics.change_in_operating_working_capital == (D("40"),)
    stale_template = template.model_copy(
        update={
            "dcf_stub": template.dcf_stub.model_copy(
                update={"fcff": D("999")}
            )
        }
    )
    mapped = DriverBasedCanonicalFcffAdapter().adapt(economics, stale_template)
    assert mapped.ytd_anchor.actual_operating_working_capital == D("130")
    assert mapped.dcf_stub.fcff == D("0.4")
    assert mapped.dcf_stub != stale_template.dcf_stub
    assert mapped.observations[0].operating_working_capital == D("140")
    assert mapped.observations[0].revenue_growth == D("0")
    assert FcffForecast.model_validate(mapped.model_dump(mode="python")) == mapped
    with pytest.raises(ValueError, match="below reported YTD CAPEX"):
        OperatingReinvestmentEngine().apply(
            base,
            (
                _observation("depreciation_and_amortization", 2025, "5"),
                _observation("capital_expenditures", 2025, "6"),
                _observation("operating_working_capital", 2025, "140"),
            ),
            seed=template,
            capex_constraints={
                2025: MonetaryForecastConstraint(point=D("7"))
            },
        )


def test_seed_selection_returns_the_exact_filtered_high_confidence_observation():
    valid_ref = EvidenceReference(provider="sec", accession="owc-valid")
    low_ref = EvidenceReference(provider="sec", accession="owc-low")
    result = OperatingReinvestmentEngine().apply(
        _base(years=(2025,)),
        (
            _observation("depreciation_and_amortization", 2025, "5"),
            _observation("capital_expenditures", 2025, "6"),
            _observation("operating_working_capital", 2025, "140"),
            _observation(
                "operating_working_capital",
                2024,
                "999",
                unit="EUR",
                currency="EUR",
                evidence=EvidenceReference(provider="sec", accession="wrong-currency"),
            ),
            _observation(
                "operating_working_capital",
                2024,
                "999",
                unit="percent",
                evidence=EvidenceReference(provider="sec", accession="ratio"),
            ),
            _observation(
                "operating_working_capital",
                2024,
                "200",
                confidence="low",
                evidence=low_ref,
            ),
            _observation(
                "operating_working_capital",
                2024,
                "100",
                evidence=valid_ref,
                source_provenance=(valid_ref,),
            ),
        ),
    )
    assert result.change_in_operating_working_capital == (D("40"),)
    assert result.reinvestment_seed.value == D("100")
    assert result.reinvestment_seed.provenance == valid_ref
    assert result.reinvestment_seed.model_dump()["provenance"]["accession"] == "owc-valid"
    assert result.change_in_operating_working_capital_provenance_by_year[2025] == valid_ref


def test_canonical_adapter_requires_base_revenue_and_audits_all_ratios():
    economics = OperatingReinvestmentEngine().apply(
        _base(years=(2025,)),
        (
            _observation("depreciation_and_amortization", 2025, "5"),
            _observation("capital_expenditures", 2025, "6"),
            _observation("operating_working_capital", 2024, "10"),
            _observation("operating_working_capital", 2025, "11"),
        ),
    )
    with pytest.raises(ValueError, match="base_period_end"):
        DriverBasedCanonicalFcffAdapter().observations(economics)
    with pytest.raises(ValueError, match="base revenue anchor"):
        DriverBasedCanonicalFcffAdapter().observations(
            economics, base_period_end=datetime.date(2024, 12, 31)
        )
    standalone = DriverBasedCanonicalFcffAdapter().observations(
        economics,
        base_period_end=datetime.date(2024, 12, 31),
        base_revenue=D("100"),
    )
    assert standalone[0].period_end == datetime.date(2025, 12, 31)
    template = _ytd_template()
    observation = DriverBasedCanonicalFcffAdapter().observations(economics, template)[0]
    for field in (
        "depreciation_to_revenue",
        "capex_to_revenue",
        "operating_working_capital_to_revenue",
    ):
        audit = observation.cell_audits[field]
        assert audit.source != "unavailable"
        assert "ratio_numerator=" in audit.method
        assert "ratio_denominator=" in audit.method


def test_canonical_adapter_rejects_incompatible_template_unit_and_horizon():
    economics = OperatingReinvestmentEngine().apply(
        _base(years=(2025,)),
        (
            _observation("depreciation_and_amortization", 2025, "5"),
            _observation("capital_expenditures", 2025, "6"),
            _observation("operating_working_capital", 2024, "10"),
            _observation("operating_working_capital", 2025, "11"),
        ),
    )
    template = _ytd_template()
    with pytest.raises(ValueError, match="units"):
        DriverBasedCanonicalFcffAdapter().adapt(
            economics, template.model_copy(update={"unit": "EUR"})
        )
    with pytest.raises(ValueError, match="horizons"):
        DriverBasedCanonicalFcffAdapter().adapt(
            economics,
            template.model_copy(
                update={
                    "parameters": FcffForecastParameters(forecast_years=2)
                }
            ),
        )


def test_ytd_adapter_clears_stale_stub_when_mapped_annual_inputs_are_incomplete():
    incomplete = OperatingReinvestmentEngine().apply(
        _base(years=(2025,)),
        (_observation("operating_working_capital", 2025, "140"),),
        seed=_ytd_template(),
    )
    mapped = DriverBasedCanonicalFcffAdapter().adapt(incomplete, _ytd_template())
    assert mapped.observations == []
    assert mapped.dcf_stub is None


def test_canonical_adapter_rejects_company_identity_mismatch():
    economics = OperatingReinvestmentEngine().apply(
        _base(years=(2025,)), _complete_observations()
    ).model_copy(update={"company_id": "economics-company"})
    template = _ytd_template().model_copy(update={"company_id": "template-company"})
    with pytest.raises(ValueError, match="company_id"):
        DriverBasedCanonicalFcffAdapter().adapt(economics, template)


def test_canonical_adapter_preserves_non_calendar_fiscal_year_end_and_leap_safety():
    economics = OperatingReinvestmentEngine().apply(
        _base(years=(2025, 2026)), _complete_observations()
    )
    template = _ytd_template().model_copy(
        update={
            "base_fiscal_year": 2024,
            "base_period_end": datetime.date(2024, 9, 30),
            "current_fiscal_year": None,
            "seed_type": ForecastSeedType.FISCAL_YEAR,
            "ytd_anchor": None,
            "dcf_stub": None,
            "parameters": FcffForecastParameters(forecast_years=2),
        }
    )
    mapped = DriverBasedCanonicalFcffAdapter().adapt(economics, template)
    assert [item.period_end for item in mapped.observations] == [
        datetime.date(2025, 9, 30),
        datetime.date(2026, 9, 30),
    ]

    leap_economics = OperatingReinvestmentEngine().apply(
        _base(years=(2028,)),
        _complete_observations(),
        seed=OperatingReinvestmentSeed(fiscal_year=2027, unit="USD", value=D("10")),
    )
    leap_template = template.model_copy(
        update={
            "base_fiscal_year": 2027,
            "base_period_end": datetime.date(2027, 2, 28),
            "parameters": FcffForecastParameters(forecast_years=1),
            "current_fiscal_year": None,
        }
    )
    leap_mapped = DriverBasedCanonicalFcffAdapter().adapt(
        leap_economics, leap_template
    )
    assert leap_mapped.observations[0].period_end == datetime.date(2028, 2, 28)
