import asyncio
from datetime import date
from decimal import Decimal

import pytest

from edgarito.schemas.operating_graph import (
    EconomicModel,
    EconomicNode,
    EconomicObservation,
    EconomicRelationship,
    EconomicSourceEdge,
)
from edgarito.services.forecasting.reasoning import (
    EconomicGraphReasoningInput,
    EconomicLeafReasoningAssumption,
    EconomicLeafReasoningResponse,
    ForecastReasoner,
    ForecastReasoningCache,
)
from edgarito.services.forecasting.reasoning.economic_graph import (
    EconomicGraphReasoningValidator,
)
from edgarito.services.research.contracts import MarketSizeEvidence


def _input(*, research=()):
    from edgarito.services.forecasting.reasoning import ForecastReasoningInput

    return ForecastReasoningInput(
        company_id="graph-company",
        company_name="Graph Company",
        unit="USD",
        as_of=date(2025, 1, 1),
        forecast_years=(2025, 2026),
        research_evidence=research,
    )


def _node(node_id, node_type="input", *, allowed=False, metric=None):
    return EconomicNode(
        node_id=node_id,
        node_type=node_type,
        metric=metric or node_id,
        unit="USD",
        unit_kind="monetary",
        currency="USD",
        forecast_assumption_allowed=allowed,
    )


def _edge(node_id, sign=1):
    return EconomicSourceEdge(node_id=node_id, sign=sign)


def _model(*, leaf_type="input", observations=(), shared=False):
    nodes = [_node("leaf", leaf_type, allowed=True)]
    relationships = []
    if shared:
        nodes.extend((_node("left", "derived"), _node("right", "derived")))
        relationships.extend(
            (
                EconomicRelationship(
                    target="left", relationship_type="identity", sources=(_edge("leaf"),)
                ),
                EconomicRelationship(
                    target="right", relationship_type="identity", sources=(_edge("leaf"),)
                ),
            )
        )
        nodes.append(_node("root", "aggregate"))
        relationships.append(
            EconomicRelationship(
                target="root",
                relationship_type="add",
                sources=(_edge("left"), _edge("right")),
            )
        )
        root = "root"
    else:
        nodes.append(_node("root", "aggregate"))
        relationships.append(
            EconomicRelationship(
                target="root", relationship_type="identity", sources=(_edge("leaf"),)
            )
        )
        root = "root"
    return EconomicModel(
        nodes=tuple(nodes), relationships=tuple(relationships), observations=observations, revenue_root=root
    )


def _assumption(node_id="leaf", *, evidence_ids=(), base=(10, 12)):
    return EconomicLeafReasoningAssumption(
        node_id=node_id,
        fiscal_years=(2025, 2026),
        low=(base[0] - 1, base[1] - 1),
        base=base,
        high=(base[0] + 1, base[1] + 1),
        unit="USD",
        evidence_ids=evidence_ids,
        rationale="bounded leaf proposal",
        confidence="medium",
        evidence_based=bool(evidence_ids),
        model_assumption=not bool(evidence_ids),
    )


class FakeOpenAI:
    model = "graph-fake"
    reasoning_effort = "low"

    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def extract_structured(self, **kwargs):
        self.calls += 1
        assert kwargs["response_model"] is EconomicLeafReasoningResponse
        return self.response


def _run(fake, forecast_input, model, tmp_path):
    return asyncio.run(
        ForecastReasoner(fake, cache=ForecastReasoningCache(tmp_path)).reason_economic_model(
            forecast_input, model
        )
    )


def test_missing_leaf_compiles_base_and_re_evaluates_graph(tmp_path):
    fake = FakeOpenAI(EconomicLeafReasoningResponse(assumptions=(_assumption(),)))
    result = _run(fake, _input(), _model(), tmp_path)

    assert [item.value for item in result.compiled_observations] == [Decimal("10"), Decimal("12")]
    assert [result.evaluation.value("root", year) for year in (2025, 2026)] == [
        Decimal("10"),
        Decimal("12"),
    ]
    assert all(item.origin == "model_assumption" for item in result.compiled_observations)
    assert all(item.available_on == date(2025, 1, 1) for item in result.compiled_observations)
    assert result.retained_ranges["leaf"][0] == (Decimal("9"), Decimal("10"), Decimal("11"))
    assert not result.evaluation.unresolved_leaf_requirements


@pytest.mark.parametrize("node_id, leaf_type", [("root", "input"), ("leaf", "derived")])
def test_aggregate_root_and_derived_leaf_assumptions_are_rejected(node_id, leaf_type):
    model = _model(leaf_type=leaf_type)
    fake_response = EconomicLeafReasoningResponse(assumptions=(_assumption(node_id),))
    graph_input = EconomicGraphReasoningInput(forecast_input=_input(), economic_model=model)
    validation = EconomicGraphReasoningValidator().validate(fake_response, graph_input)

    assert not validation.accepted_assumptions
    assert {item.code for item in validation.rejected_assumptions} & {
        "UNSAFE_NODE_TYPE",
        "REVENUE_ROOT_FORBIDDEN",
    }


def test_manual_graph_observation_wins_and_collision_is_audit_only(tmp_path):
    manual = EconomicObservation(
        node_id="leaf",
        year=2026,
        value=Decimal("99"),
        unit="USD",
        currency="USD",
        origin="manual",
        available_on=date(2024, 1, 1),
    )
    fake = FakeOpenAI(EconomicLeafReasoningResponse(assumptions=(_assumption(),)))
    result = _run(fake, _input(), _model(observations=(manual,)), tmp_path)

    assert result.evaluation.value("root", 2025) == Decimal("10")
    assert result.evaluation.value("root", 2026) == Decimal("99")
    assert [item.fiscal_year for item in result.compiled_observations] == [2025]
    assert any(item.code == "MANUAL_GRAPH_OBSERVATION_PRECEDENCE" for item in result.collisions)


def test_shared_leaf_is_compiled_once_and_used_by_both_paths(tmp_path):
    fake = FakeOpenAI(EconomicLeafReasoningResponse(assumptions=(_assumption(),)))
    result = _run(fake, _input(), _model(shared=True), tmp_path)

    assert len(result.compiled_observations) == 2
    assert result.evaluation.value("left", 2025) == Decimal("10")
    assert result.evaluation.value("right", 2025) == Decimal("10")
    assert result.evaluation.value("root", 2025) == Decimal("20")


def test_future_point_in_time_citation_is_rejected():
    evidence = MarketSizeEvidence(
        evidence_id="future-size",
        source_date=date(2026, 1, 1),
        source_type="analyst_estimate",
        low=1,
        base=2,
        high=3,
        provenance={"source": "future"},
        unit="USD",
        market="graph",
    )
    graph_input = EconomicGraphReasoningInput(
        forecast_input=_input(research=(evidence,)), economic_model=_model()
    )
    response = EconomicLeafReasoningResponse(
        assumptions=(_assumption(evidence_ids=("future-size",)),)
    )
    validation = EconomicGraphReasoningValidator().validate(response, graph_input)

    assert any(item.code == "EXCLUDED_EVIDENCE" for item in validation.rejected_assumptions)


def test_reasoner_can_target_parameter_and_residual_forecast_leaves():
    parameter = EconomicNode(
        node_id="revenue_per_transaction",
        node_type="input",
        metric="revenue_per_transaction",
        unit="USD/transactions",
        unit_kind="monetary_per_unit",
        denominator_unit="transactions",
        currency="USD",
        forecast_assumption_allowed=True,
    )
    residual = EconomicNode(
        node_id="residual",
        node_type="component",
        metric="residual",
        unit="USD",
        unit_kind="monetary",
        currency="USD",
        forecast_assumption_allowed=True,
    )
    model = EconomicModel(
        nodes=(
            _node("revenue"),
            EconomicNode(
                node_id="transactions",
                node_type="input",
                metric="transactions",
                unit="transactions",
                unit_kind="count",
                currency=None,
            ),
            _node("total"),
            _node("known"),
            parameter,
            residual,
        ),
        relationships=(
            EconomicRelationship(
                target=parameter.node_id,
                relationship_type="ratio",
                sources=(_edge("revenue"), _edge("transactions")),
                relationship_id="parameter:ratio",
                historical_parameter_derivation=True,
            ),
            EconomicRelationship(
                target=residual.node_id,
                relationship_type="residual",
                sources=(_edge("total"), _edge("known", sign=-1)),
                relationship_id="residual:total_known",
            ),
        ),
        observations=tuple(
            EconomicObservation(
                node_id="known",
                year=year,
                value=Decimal("70"),
                unit="USD",
                currency="USD",
                origin="reported",
            )
            for year in (2025, 2026)
        ),
    )
    graph_input = EconomicGraphReasoningInput(
        forecast_input=_input(), economic_model=model
    )
    response = EconomicLeafReasoningResponse(
        assumptions=(
            EconomicLeafReasoningAssumption(
                node_id=parameter.node_id,
                fiscal_years=(2025, 2026),
                low=(9, 9),
                base=(10, 10),
                high=(11, 11),
                unit=parameter.unit,
                rationale="bounded parameter",
                confidence="medium",
                model_assumption=True,
            ),
            EconomicLeafReasoningAssumption(
                node_id=residual.node_id,
                fiscal_years=(2025, 2026),
                low=(20, 20),
                base=(30, 30),
                high=(40, 40),
                unit=residual.unit,
                rationale="bounded residual",
                confidence="medium",
                model_assumption=True,
            ),
        )
    )

    validation = EconomicGraphReasoningValidator().validate(response, graph_input)

    assert validation.is_valid
    assert {
        item.node_id for item in validation.accepted_assumptions
    } == {parameter.node_id, residual.node_id}
