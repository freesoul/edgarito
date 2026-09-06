"""Focused tests for evidence-only economic-structure discovery."""

import hashlib
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from edgarito.schemas.operating_graph import (
    EconomicModel,
    EconomicNode,
    EconomicObservation,
    EconomicProvenance,
    EconomicRelationship,
    EconomicSourceEdge,
)
from edgarito.services.operating._graph import (
    EconomicModelDiscoveryService,
    evaluate_graph,
)

ROOT = Path(__file__).resolve().parents[1]
V_MODEL_FIXTURE = ROOT / "tests" / "fixtures" / "evaluation" / "v_economic_model.json"
V_EXPERIMENT = ROOT / "tests" / "fixtures" / "evaluation" / "v_experiment.json"
AS_OF = date(2023, 11, 15)


def _visa_model() -> EconomicModel:
    return EconomicModel.model_validate(json.loads(V_MODEL_FIXTURE.read_text()))


def test_visa_structure_has_truthful_components_contra_and_lag_paths():
    model = _visa_model()
    nodes = {node.node_id: node for node in model.nodes}
    relationships = {item.target: item for item in model.relationships}

    assert nodes["service_revenue"].node_type == "component"
    assert nodes["data_processing_revenue"].node_type == "component"
    assert nodes["international_transaction_revenue"].node_type == "component"
    assert nodes["other_revenue"].node_type == "component"
    assert nodes["client_incentives"].component_role == "contra_revenue"

    net_sources = {
        edge.node_id: edge.sign for edge in relationships["net_revenue"].sources
    }
    assert net_sources == {
        "service_revenue": Decimal("1"),
        "data_processing_revenue": Decimal("1"),
        "international_transaction_revenue": Decimal("1"),
        "other_revenue": Decimal("1"),
        "client_incentives": Decimal("-1"),
    }
    assert relationships["service_revenue"].relationship_type.value == "multiply"
    assert {edge.node_id for edge in relationships["service_revenue"].sources} == {
        "lagged_payments_volume",
        "service_yield",
    }
    assert relationships["lagged_payments_volume"].relationship_type.value == "lag"
    assert relationships["lagged_payments_volume"].sources[0].fiscal_lag == 1
    assert (
        relationships["data_processing_revenue"].relationship_type.value == "multiply"
    )
    assert relationships[
        "international_transaction_revenue"
    ].relationship_type.value == ("multiply")


def test_visa_fixture_has_no_fabricated_segments_or_blended_take_rate():
    model = _visa_model()
    assert all(node.node_type.value != "segment" for node in model.nodes)
    assert not any("take_rate" in node.node_id.casefold() for node in model.nodes)
    assert not any("take_rate" in node.metric.casefold() for node in model.nodes)
    assert all("take_rate" not in str(item).casefold() for item in model.relationships)


def test_visa_discovery_retains_only_point_in_time_evidence_and_unresolved_leaves():
    result = EconomicModelDiscoveryService().discover(model=_visa_model(), as_of=AS_OF)

    assert result.valid
    assert result.audit.evidence_only is True
    assert result.audit.forecasts_added == 0
    assert result.audit.coefficients_added == 0
    assert len(result.rejected_records) == 0
    assert {
        (item.node_id, item.fiscal_year, str(item.value))
        for item in result.model.observations
        if item.node_id in {"payments_volume", "processed_transactions"}
    } == {
        ("payments_volume", 2022, "11607"),
        ("payments_volume", 2023, "12338"),
        ("processed_transactions", 2022, "192530"),
        ("processed_transactions", 2023, "212579"),
    }
    component_ids = {
        "service_revenue",
        "data_processing_revenue",
        "international_transaction_revenue",
        "other_revenue",
        "client_incentives",
    }
    assert not any(item.node_id in component_ids for item in result.model.observations)
    assert not any(
        item.node_id
        in {"service_yield", "revenue_per_transaction", "international_yield"}
        for item in result.model.observations
    )
    assert all(item.available_on <= AS_OF for item in result.model.observations)
    assert {item.node_id for item in result.unresolved_leaves} >= {
        "service_yield",
        "revenue_per_transaction",
        "cross_border_volume",
        "international_yield",
        "other_revenue",
        "client_incentives",
    }
    assert {
        item.node_id
        for item in result.unresolved_leaves
        if item.category == "coefficient"
    } == {"service_yield", "revenue_per_transaction", "international_yield"}
    assert any("service_revenue" in item for item in result.missing_evidence)
    assert any(
        "cannot be reconstructed or forecast" in item
        for item in result.missing_evidence
    )


def test_current_frozen_evidence_cannot_reconstruct_or_forecast_total_revenue():
    result = EconomicModelDiscoveryService().discover(model=_visa_model(), as_of=AS_OF)
    evaluation = evaluate_graph(
        result.model,
        (2022, 2023, 2024),
        as_of=AS_OF,
    )

    assert [evaluation.value("net_revenue", year) for year in (2022, 2023, 2024)] == [
        None,
        None,
        None,
    ]
    assert evaluation.diagnostics.historical_reconstructable_share == Decimal("0")
    assert evaluation.diagnostics.forecastable_share == Decimal("0")
    required = {item.node_id for item in evaluation.unresolved_leaf_requirements}
    assert {"service_yield", "revenue_per_transaction", "other_revenue"} <= required
    assert any("historical component" in item for item in result.missing_evidence)


def test_visa_fy2024_requirements_follow_root_paths_and_source_year_lags():
    model = _visa_model()
    evaluation = evaluate_graph(model, (2024,), as_of=AS_OF)

    requirements = {
        item.node_id: item
        for item in evaluation.unresolved_leaf_requirements
        if item.fiscal_year == 2024
    }
    assert set(requirements) == {
        "service_yield",
        "processed_transactions",
        "revenue_per_transaction",
        "cross_border_volume",
        "international_yield",
        "other_revenue",
        "client_incentives",
    }
    assert requirements["service_yield"].path == (
        "net_revenue",
        "service_revenue",
        "service_yield",
    )
    assert requirements["service_yield"].required_by_relationship_ids == (
        "net_revenue_components",
        "service_revenue_lagged_volume",
    )
    assert requirements["processed_transactions"].path == (
        "net_revenue",
        "data_processing_revenue",
        "processed_transactions",
    )
    assert requirements["processed_transactions"].required_by_relationship_ids == (
        "net_revenue_components",
        "data_processing_transactions",
    )
    assert requirements["other_revenue"].path == ("net_revenue", "other_revenue")
    assert requirements["other_revenue"].required_by_relationship_ids == (
        "net_revenue_components",
    )
    assert evaluation.value("lagged_payments_volume", 2024) == Decimal("12338")
    assert not any(
        item.node_id == "payments_volume" and item.fiscal_year == 2024
        for item in evaluation.unresolved_leaf_requirements
    )
    nodes = model.node_by_id
    assert nodes["payments_volume"].forecast_assumption_allowed is True
    assert nodes["processed_transactions"].forecast_assumption_allowed is True
    assert all("take_rate" not in item.node_id for item in model.nodes)


def test_discovery_rejects_future_observations_at_the_as_of_boundary():
    model = _visa_model()
    future = EconomicObservation(
        node_id="payments_volume",
        fiscal_year=2024,
        value=Decimal("13190"),
        unit="USD billions",
        currency="USD",
        origin="first_party_observation",
        provenance=EconomicProvenance(
            source="visa_ir",
            origin="first_party_observation",
            reference="post-cutoff-fixture",
        ),
        available_on=date(2024, 10, 29),
    )
    result = EconomicModelDiscoveryService().discover(
        model=model.model_copy(update={"observations": (*model.observations, future)}),
        as_of=AS_OF,
    )

    assert all(
        not (item.node_id == "payments_volume" and item.fiscal_year == 2024)
        for item in result.model.observations
    )
    assert any(
        item.code == "future_evidence" and item.record_id == "payments_volume:2024:FY"
        for item in result.rejected_records
    )
    assert result.audit.future_record_count == 1


def test_generic_discovery_result_is_immutable_and_retains_reasons():
    provenance = EconomicProvenance(source="fixture", reference="generic")
    nodes = (
        EconomicNode(
            node_id="input",
            node_type="input",
            metric="input",
            unit="USD",
            currency="USD",
            unit_kind="monetary",
            provenance=provenance,
        ),
        EconomicNode(
            node_id="root",
            node_type="aggregate",
            metric="root",
            unit="USD",
            currency="USD",
            unit_kind="monetary",
            provenance=provenance,
        ),
    )
    relationship = EconomicRelationship(
        target="root",
        relationship_type="add",
        sources=(EconomicSourceEdge(node_id="input", sign=1),),
        provenance=provenance,
    )
    observation = EconomicObservation(
        node_id="input",
        fiscal_year=2024,
        value=1,
        unit="USD",
        currency="USD",
        provenance=provenance,
        available_on=date(2025, 1, 1),
    )

    result = EconomicModelDiscoveryService().discover(
        nodes=nodes,
        relationships=(relationship,),
        observations=(observation,),
        revenue_root="root",
        as_of=date(2024, 12, 31),
    )

    assert result.accepted_model.revenue_root == "root"
    assert result.rejected_count == 1
    assert result.rejected[0].code == "future_evidence"
    assert result.rejected[0].reason
    with pytest.raises(ValidationError):
        result.accepted_model = result.accepted_model


def test_frozen_v_experiment_artifact_bytes_remain_unchanged():
    assert hashlib.sha256(V_EXPERIMENT.read_bytes()).hexdigest() == (
        "79604a71ae0cc153a95434b3e02147cb62f081a747f6c872a8a0ec36a2faaecb"
    )
