from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from edgarito.schemas import CompanyOperatingForecast as PublicCompanyOperatingForecast
from edgarito.schemas import (
    DependencyAudit,
    EconomicComponentRole,
    EconomicEvaluationResult,
    EconomicMateriality,
    EconomicModel,
    EconomicNode,
    EconomicNodeType,
    EconomicObservation,
    EconomicProvenance,
    EconomicRelationship,
    EconomicRelationshipType,
    EconomicSourceEdge,
    EconomicUnitKind,
    EconomicValue,
    GraphDiagnostic,
    GraphDiagnostics,
    UnresolvedLeafRequirement,
)
from edgarito.schemas import OperatingSegment as PublicOperatingSegment
from edgarito.schemas.operating import (
    CompanyOperatingForecast,
    EvidenceReference,
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverForecast,
    OperatingDriverObservation,
    OperatingSegment,
    SegmentRevenueForecast,
    operating_periods_compatible,
)
from edgarito.schemas.valuation import AssumptionOrigin, AssumptionProvenance


def _segment() -> OperatingSegment:
    return OperatingSegment(
        segment_id="cloud",
        name=" Cloud ",
        scope="segment",
        currency="usd",
        dimensions={"product": "infrastructure"},
    )


def _definition() -> OperatingDriverDefinition:
    return OperatingDriverDefinition(
        driver_id="cloud-volume-price",
        archetype=OperatingArchetype.VOLUME_PRICE,
        segment_id="cloud",
        output_metric="revenue",
        input_metrics=("volume", "price"),
        units={"volume": "million users", "price": "USD per user"},
        formula_id="volume_times_price",
        required_inputs=("volume", "price"),
    )


def _observation() -> OperatingDriverObservation:
    return OperatingDriverObservation(
        segment_id="cloud",
        driver_id="cloud-volume-price",
        fiscal_year=2026,
        fiscal_period="FY",
        value=Decimal("12.5"),
        unit="million users",
        origin="extracted_evidence",
        confidence="HIGH",
        provenance=AssumptionProvenance(origin=AssumptionOrigin.EXPLICIT),
        evidence=EvidenceReference(
            provider="sec",
            accession="0000000000-26-000001",
            filing_date="2026-02-01",
            document_name="10-K.htm",
            supporting_text="The cloud business served 12.5 million users.",
        ),
    )


def _driver_forecast() -> OperatingDriverForecast:
    return OperatingDriverForecast(
        segment_id="cloud",
        driver_id="cloud-volume-price",
        fiscal_year=2026,
        value=Decimal("12.5"),
        unit="million users",
        source="independent_operating",
        method="volume_times_price",
        confidence="medium",
    )


def _segment_forecast() -> SegmentRevenueForecast:
    return SegmentRevenueForecast(
        segment=_segment(),
        fiscal_years=(2026, 2027),
        revenue=(Decimal("100"), Decimal("110")),
        revenue_growth=(None, Decimal("10")),
        driver_forecasts=(_driver_forecast(),),
        explicit_years=(2026, 2027),
        source_by_year={2026: "independent_operating", 2027: "independent_operating"},
        confidence_by_year={2026: "high", 2027: "medium"},
        unit="USD",
    )


def test_operating_contracts_are_public_frozen_and_round_trip():
    assert PublicOperatingSegment is OperatingSegment
    assert PublicCompanyOperatingForecast is CompanyOperatingForecast
    assert {item.value for item in OperatingArchetype} == {
        "volume_price",
        "subscribers_arpu",
        "capacity_utilization_price",
        "transactions_take_rate",
        "backlog_conversion",
        "store_count_sales_per_store",
        "generic_segment_growth",
    }

    definition = _definition()
    observation = _observation()
    segment_forecast = _segment_forecast()
    company_forecast = CompanyOperatingForecast(
        company_id="example",
        fiscal_years=(2026, 2027),
        segment_forecasts=(segment_forecast,),
        consolidated_revenue=(Decimal("100"), Decimal("110")),
        consolidated_growth=(None, Decimal("10")),
        explicit_years=(2026, 2027),
        transition_start_year=2028,
        source_by_year={2026: "independent_operating", 2027: "independent_operating"},
        confidence_by_year={2026: "high", 2027: "medium"},
        unit="USD",
    )

    assert definition.units["volume"] == "million users"
    assert observation.confidence == "high"
    assert observation.evidence is not None
    assert observation.evidence.accession_number == "0000000000-26-000001"
    assert company_forecast.transition_start_year == 2028
    assert (
        CompanyOperatingForecast.model_validate_json(company_forecast.model_dump_json())
        == company_forecast
    )

    with pytest.raises(ValidationError):
        segment_forecast.revenue = (Decimal("101"), Decimal("111"))


def test_economic_graph_contracts_are_public_and_round_trip():
    provenance = EconomicProvenance(
        source="sec",
        available_on=date(2026, 2, 1),
        evidence_ids=("evidence-1",),
    )
    node = EconomicNode(
        node_id="revenue",
        node_type=EconomicNodeType.INPUT,
        metric="Revenue",
        unit="USD",
        unit_kind=EconomicUnitKind.MONETARY,
        currency="USD",
        provenance=provenance,
        materiality=EconomicMateriality.MATERIAL,
        component_role=EconomicComponentRole.STANDARD,
    )
    source_edge = EconomicSourceEdge(node_id="revenue")
    relationship = EconomicRelationship(
        target="revenue",
        relationship_type=EconomicRelationshipType.IDENTITY,
        sources=(source_edge,),
        provenance=provenance,
    )
    observation = EconomicObservation(
        node_id="revenue",
        fiscal_year=2025,
        value=Decimal("100"),
        unit="USD",
        currency="USD",
        provenance=provenance,
    )
    model = EconomicModel(
        nodes=(node,),
        relationships=(relationship,),
        observations=(observation,),
        revenue_root="revenue",
    )
    value = EconomicValue(
        node_id="revenue",
        fiscal_year=2025,
        value=Decimal("100"),
        unit="USD",
        currency="USD",
        available=True,
        provenance=provenance,
    )
    audit = DependencyAudit(
        node_id="revenue",
        fiscal_year=2025,
        available=True,
        dependency_chain=("revenue",),
    )
    unresolved = UnresolvedLeafRequirement(
        node_id="revenue", fiscal_year=2025, reason="missing observation"
    )
    diagnostic = GraphDiagnostic(code="example", message="Example diagnostic")
    diagnostics = GraphDiagnostics(diagnostic_messages=(diagnostic,))
    result = EconomicEvaluationResult(
        target_years=(2025,),
        values={"revenue": {2025: Decimal("100")}},
        cells=(value,),
        dependency_audits=(audit,),
        unresolved_leaf_requirements=(unresolved,),
        diagnostics=diagnostics,
    )

    assert model.nodes == (node,)
    assert EconomicModel.model_validate_json(model.model_dump_json()) == model
    assert EconomicEvaluationResult.model_validate_json(result.model_dump_json()) == result


def test_operating_contracts_reject_invalid_decimal_unit_year_and_range_values():
    with pytest.raises(ValidationError, match="must be finite"):
        OperatingDriverObservation(
            segment_id="cloud",
            driver_id="users",
            fiscal_year=2026,
            value=Decimal("NaN"),
            unit="users",
            origin="reported",
            confidence="high",
        )

    with pytest.raises(ValidationError, match="cannot be blank"):
        OperatingDriverForecast(
            segment_id="cloud",
            driver_id="users",
            fiscal_year=2026,
            value=Decimal("10"),
            unit=" ",
            source="independent_operating",
            method="reported",
            confidence="high",
        )

    with pytest.raises(ValidationError, match="greater than or equal to 1900"):
        OperatingDriverForecast(
            segment_id="cloud",
            driver_id="users",
            fiscal_year=1899,
            value=Decimal("10"),
            unit="users",
            source="independent_operating",
            method="reported",
            confidence="high",
        )

    with pytest.raises(ValidationError, match="low cannot exceed high"):
        OperatingDriverObservation(
            segment_id="cloud",
            driver_id="users",
            fiscal_year=2026,
            low=Decimal("12"),
            high=Decimal("10"),
            unit="users",
            origin="management_guidance",
            confidence="medium",
        )


def test_operating_segment_ids_and_periods_are_canonical_and_compatible():
    segment = OperatingSegment(
        segment_id="Automotive business",
        name="Automotive business",
        parent_id="Total operating segment",
    )
    definition = OperatingDriverDefinition(
        driver_id="automotive-growth",
        archetype=OperatingArchetype.GENERIC_SEGMENT_GROWTH,
        segment_id="Automotive business",
        output_metric="revenue",
        input_metrics=("growth",),
        units={"growth": "ratio"},
        formula_id="generic_segment_growth",
        required_inputs=("growth",),
    )
    observation = OperatingDriverObservation(
        segment_id="Automotive business",
        driver_id="segment_revenue",
        fiscal_year=2025,
        fiscal_period="Q1",
        value=Decimal("10"),
        unit="USD millions",
        origin="reported",
        confidence="high",
    )

    assert segment.segment_id == "automotive"
    assert segment.parent_id == "total"
    assert definition.segment_id == "automotive"
    assert observation.segment_id == "automotive"
    assert observation.fiscal_period == "FQ"
    assert observation.period_key == "Q1"
    assert operating_periods_compatible("Q1", "FQ", "Q1", "Q1")
    assert not operating_periods_compatible("Q1", "Q2", "Q1", "Q2")
    assert not operating_periods_compatible("FY", "FQ")


def test_operating_forecast_paths_preserve_absolute_revenue_invariants():
    with pytest.raises(ValidationError, match="must match growth"):
        SegmentRevenueForecast(
            segment=_segment(),
            fiscal_years=(2026, 2027),
            revenue=(Decimal("100"), Decimal("110")),
            revenue_growth=(None, Decimal("9")),
        )

    with pytest.raises(ValidationError, match="equal length"):
        CompanyOperatingForecast(
            company_id="example",
            fiscal_years=(2026, 2027),
            consolidated_revenue=(Decimal("100"),),
            consolidated_growth=(None,),
        )

    with pytest.raises(ValidationError, match="after the last explicit"):
        CompanyOperatingForecast(
            company_id="example",
            fiscal_years=(2026, 2027),
            consolidated_revenue=(Decimal("100"), Decimal("110")),
            consolidated_growth=(None, Decimal("10")),
            explicit_years=(2026, 2027),
            transition_start_year=2027,
        )
