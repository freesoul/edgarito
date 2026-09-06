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
from edgarito.services.operating._graph import (
    GraphValidationError,
    evaluate_graph,
    inspect_graph,
    validate_graph,
)


def _node(
    node_id: str,
    node_type: str = "input",
    *,
    unit: str = "USD",
    unit_kind: str = "monetary",
    currency: str | None = "USD",
    scope: str = "consolidated",
    scope_id: str = "company",
    role: str = "standard",
    **kwargs,
) -> EconomicNode:
    return EconomicNode(
        node_id=node_id,
        node_type=node_type,
        scope=scope,
        scope_id=scope_id,
        metric=node_id,
        unit=unit,
        unit_kind=unit_kind,
        currency=currency,
        component_role=role,
        **kwargs,
    )


def _observation(
    node_id: str,
    year: int,
    value: str,
    *,
    unit: str = "USD",
    currency: str | None = "USD",
    origin: str = "reported",
    available_on: date | None = None,
) -> EconomicObservation:
    return EconomicObservation(
        node_id=node_id,
        year=year,
        value=Decimal(value),
        unit=unit,
        currency=currency,
        origin=origin,
        available_on=available_on,
    )


def _edge(node_id: str, *, sign: int = 1, weight: str = "1", lag: int = 0):
    return EconomicSourceEdge(
        node_id=node_id,
        sign=Decimal(sign),
        weight=Decimal(weight),
        fiscal_lag=lag,
    )


def test_add_subtract_and_contra_revenue_use_explicit_signs():
    model = EconomicModel(
        nodes=(
            _node("gross"),
            _node("discounts", "component", role="contra_revenue"),
            _node("net", "aggregate"),
        ),
        relationships=(
            EconomicRelationship(
                target="net",
                relationship_type="subtract",
                sources=(_edge("gross", sign=1), _edge("discounts", sign=-1)),
            ),
        ),
        observations=(
            _observation("gross", 2024, "100"),
            _observation("discounts", 2024, "8"),
        ),
        revenue_root="net",
    )

    result = evaluate_graph(model, [2024])

    assert result.value("net", 2024) == Decimal("92")
    assert result.diagnostics.unresolved_count == 0


def test_multiply_ratio_lag_and_growth_are_deterministic():
    model = EconomicModel(
        nodes=(
            _node("volume", unit="units", unit_kind="count", currency=None),
            _node(
                "price",
                unit="USD/unit",
                unit_kind="monetary_per_unit",
                currency="USD",
                denominator_unit="units",
            ),
            _node("revenue", "derived"),
            _node("rate", unit="ratio", unit_kind="rate", currency=None),
            _node("growth_revenue", "derived"),
            _node("lagged_revenue", "derived"),
            _node(
                "unit_price",
                "derived",
                unit="USD/unit",
                unit_kind="monetary_per_unit",
                denominator_unit="units",
            ),
        ),
        relationships=(
            EconomicRelationship(
                target="revenue",
                relationship_type="multiply",
                sources=(_edge("volume"), _edge("price")),
            ),
            EconomicRelationship(
                target="unit_price",
                relationship_type="ratio",
                sources=(_edge("revenue"), _edge("volume")),
            ),
            EconomicRelationship(
                target="growth_revenue",
                relationship_type="growth",
                sources=(_edge("revenue"), _edge("rate")),
            ),
            EconomicRelationship(
                target="lagged_revenue",
                relationship_type="lag",
                sources=(_edge("revenue", lag=1),),
            ),
        ),
        observations=(
            _observation("volume", 2024, "10", unit="units", currency=None),
            _observation("volume", 2025, "12", unit="units", currency=None),
            _observation("price", 2024, "3", unit="USD/unit"),
            _observation("rate", 2024, "0.1", unit="ratio", currency=None),
        ),
    )

    result = evaluate_graph(model, [2024, 2025])

    assert result.value("revenue", 2024) == Decimal("30")
    assert result.value("unit_price", 2024) == Decimal("3")
    assert result.value("growth_revenue", 2024) == Decimal("33")
    assert result.value("lagged_revenue", 2025) == Decimal("30")


@pytest.mark.parametrize(
    ("rate_unit", "rate_value", "expected"),
    [("percent", "10", "1.10"), ("ratio", "10", "11")],
)
def test_growth_scales_only_explicit_percentage_rates(
    rate_unit, rate_value, expected
):
    model = EconomicModel(
        nodes=(
            _node("base"),
            _node("rate", unit=rate_unit, unit_kind="rate", currency=None),
            _node("grown", "derived"),
        ),
        relationships=(
            EconomicRelationship(
                target="grown",
                relationship_type="growth",
                sources=(_edge("base"), _edge("rate")),
            ),
        ),
        observations=(
            _observation("base", 2024, "1"),
            _observation("rate", 2024, rate_value, unit=rate_unit, currency=None),
        ),
    )

    assert evaluate_graph(model, [2024]).value("grown", 2024) == Decimal(expected)


def test_multiply_scales_percent_take_rate_but_not_dimensional_coefficients():
    take_rate_model = EconomicModel(
        nodes=(
            _node("transactions", unit="transactions", unit_kind="count", currency=None),
            _node("take_rate", unit="%", unit_kind="rate", currency=None),
            _node("taken", "derived", unit="transactions", unit_kind="count", currency=None),
        ),
        relationships=(
            EconomicRelationship(
                target="taken",
                relationship_type="multiply",
                sources=(_edge("transactions"), _edge("take_rate")),
            ),
        ),
        observations=(
            _observation(
                "transactions", 2024, "1", unit="transactions", currency=None
            ),
            _observation("take_rate", 2024, "2", unit="%", currency=None),
        ),
    )
    assert evaluate_graph(take_rate_model, [2024]).value("taken", 2024) == Decimal(
        ".02"
    )

    coefficient_model = EconomicModel(
        nodes=(
            _node(
                "coefficient",
                unit="USD millions/USD billions",
                unit_kind="rate",
                currency=None,
            ),
            _node("base", unit="USD billions"),
            _node("output", "derived", unit="USD millions"),
        ),
        relationships=(
            EconomicRelationship(
                target="output",
                relationship_type="multiply",
                sources=(_edge("coefficient"), _edge("base")),
            ),
        ),
        observations=(
            _observation(
                "coefficient",
                2024,
                "2",
                unit="USD millions/USD billions",
                currency=None,
            ),
            _observation("base", 2024, "3", unit="USD billions"),
        ),
    )
    assert evaluate_graph(coefficient_model, [2024]).value("output", 2024) == Decimal(
        "6"
    )


def test_validator_rejects_ambiguous_rate_units_instead_of_guessing():
    model = EconomicModel(
        nodes=(
            _node("base"),
            _node("rate", unit="rate", unit_kind="rate", currency=None),
            _node("grown", "derived"),
        ),
        relationships=(
            EconomicRelationship(
                target="grown",
                relationship_type="growth",
                sources=(_edge("base"), _edge("rate")),
            ),
        ),
        observations=(
            _observation("base", 2024, "1"),
            _observation("rate", 2024, "10", unit="rate", currency=None),
        ),
    )

    report = inspect_graph(model)
    assert any(item.code == "rate_unit_ambiguous" for item in report.errors)
    with pytest.raises(GraphValidationError, match="rate_unit_ambiguous"):
        evaluate_graph(model, [2024])


def test_shared_driver_is_included_once_in_the_backward_audit():
    model = EconomicModel(
        nodes=(_node("driver"), _node("left", "derived"), _node("right", "derived")),
        relationships=(
            EconomicRelationship(
                target="left", relationship_type="identity", sources=(_edge("driver"),)
            ),
            EconomicRelationship(
                target="right", relationship_type="identity", sources=(_edge("driver"),)
            ),
        ),
        observations=(_observation("driver", 2024, "7"),),
    )

    result = evaluate_graph(model, [2024])

    left = next(item for item in result.dependency_audits if item.node_id == "left")
    right = next(item for item in result.dependency_audits if item.node_id == "right")
    assert left.leaf_nodes == right.leaf_nodes == ("driver",)
    assert left.dependency_chain.count("driver") == 1


def test_residual_reconstructs_history_but_requires_future_evidence():
    model = EconomicModel(
        nodes=(_node("total"), _node("known"), _node("residual", "component")),
        relationships=(
            EconomicRelationship(
                target="residual",
                relationship_type="residual",
                sources=(_edge("total", sign=1), _edge("known", sign=-1)),
            ),
        ),
        observations=(
            _observation("total", 2024, "100"),
            _observation("known", 2024, "70"),
        ),
    )

    result = evaluate_graph(model, [2024, 2025])

    assert result.value("residual", 2024) == Decimal("30")
    assert result.value("residual", 2025) is None
    assert result.diagnostics.unresolved_count > 0

    explicit_future = model.model_copy(
        update={
            "observations": (
                *model.observations,
                _observation("residual", 2025, "31"),
            )
        }
    )
    assert evaluate_graph(explicit_future, [2025]).value("residual", 2025) == Decimal(
        "31"
    )


def test_derived_historical_parameter_requires_its_declared_origin():
    model = EconomicModel(
        nodes=(
            _node("margin", "derived", unit="ratio", unit_kind="rate", currency=None),
        ),
        observations=(
            _observation(
                "margin",
                2024,
                "0.25",
                unit="ratio",
                currency=None,
                origin="derived_historical_parameter",
            ),
        ),
    )

    assert evaluate_graph(model, [2024]).value("margin", 2024) == Decimal("0.25")


def test_validator_rejects_reference_dimension_and_zero_lag_cycle_errors():
    mismatch = EconomicModel(
        nodes=(
            _node("left"),
            _node("right", scope_id="other"),
            _node("sum", "derived"),
        ),
        relationships=(
            EconomicRelationship(
                target="sum",
                relationship_type="add",
                sources=(_edge("left", sign=1), _edge("right", sign=1)),
            ),
        ),
    )
    report = inspect_graph(mismatch)
    assert any(item.code == "scope_mismatch" for item in report.errors)
    with pytest.raises(GraphValidationError):
        validate_graph(mismatch)

    cycle = EconomicModel(
        nodes=(_node("a", "derived"), _node("b", "derived")),
        relationships=(
            EconomicRelationship(
                target="a", relationship_type="identity", sources=(_edge("b"),)
            ),
            EconomicRelationship(
                target="b", relationship_type="identity", sources=(_edge("a"),)
            ),
        ),
    )
    with pytest.raises(GraphValidationError, match="zero_lag_cycle"):
        validate_graph(cycle)

    positive_lag = EconomicModel(
        nodes=(_node("lagged", "derived"),),
        relationships=(
            EconomicRelationship(
                target="lagged",
                relationship_type="lag",
                sources=(_edge("lagged", lag=1),),
            ),
        ),
    )
    assert validate_graph(positive_lag).valid


def test_point_in_time_leakage_and_provenance_chain_are_visible():
    model = EconomicModel(
        nodes=(
            _node("input", provenance="filing:input"),
            _node("output", "derived", provenance="model:output"),
        ),
        relationships=(
            EconomicRelationship(
                target="output",
                relationship_type="identity",
                sources=(_edge("input"),),
                provenance="formula:identity",
            ),
        ),
        observations=(
            _observation("input", 2025, "10", available_on=date(2026, 2, 1)),
        ),
    )

    result = evaluate_graph(model, [2025], as_of=date(2026, 1, 31))

    assert result.value("output", 2025) is None
    assert any(
        item.code == "future_evidence"
        for item in result.diagnostics.diagnostic_messages
    )
    audit = next(item for item in result.dependency_audits if item.node_id == "output")
    assert "model:output" in audit.provenance_chain
    assert "filing:input" in audit.provenance_chain


def test_missing_leaf_makes_aggregate_unavailable_not_zero():
    model = EconomicModel(
        nodes=(_node("missing"), _node("aggregate", "aggregate")),
        relationships=(
            EconomicRelationship(
                target="aggregate",
                relationship_type="add",
                sources=(_edge("missing", sign=1),),
            ),
        ),
    )

    result = evaluate_graph(model, [2024])

    assert result.value("aggregate", 2024) is None
    assert result.value("aggregate", 2024) != Decimal("0")
    assert result.diagnostics.aggregate_coverage == Decimal("0")


def test_historical_parameter_ratio_derives_history_and_collapses_future_sources():
    model = EconomicModel(
        nodes=(
            _node("revenue", unit="USD", unit_kind="monetary"),
            _node("transactions", unit="transactions", unit_kind="count", currency=None),
            _node(
                "revenue_per_transaction",
                unit="USD/transactions",
                unit_kind="monetary_per_unit",
                denominator_unit="transactions",
                materiality="material",
                forecast_assumption_allowed=True,
            ),
        ),
        relationships=(
            EconomicRelationship(
                target="revenue_per_transaction",
                relationship_type="ratio",
                sources=(_edge("revenue"), _edge("transactions")),
                relationship_id="parameter:revenue_per_transaction",
                provenance="formula:reported_revenue_per_transaction",
                historical_parameter_derivation=True,
            ),
        ),
        observations=(
            _observation("revenue", 2024, "100"),
            _observation(
                "transactions", 2024, "10", unit="transactions", currency=None
            ),
        ),
    )

    result = evaluate_graph(model, (2024, 2025))

    historical = result.cell("revenue_per_transaction", 2024)
    future = result.cell("revenue_per_transaction", 2025)
    assert historical is not None
    assert historical.value == Decimal("10")
    assert historical.origin == "derived_historical_parameter"
    assert "formula:reported_revenue_per_transaction" in historical.provenance_chain
    assert future is not None and not future.is_available
    requirements = result.unresolved_leaf_requirements
    assert {(item.node_id, item.fiscal_year) for item in requirements} == {
        ("revenue_per_transaction", 2025)
    }
    requirement = requirements[0]
    assert requirement.scope == "consolidated"
    assert requirement.scope_id == "company"
    assert requirement.metric == "revenue_per_transaction"
    assert requirement.unit == "USD/transactions"
    assert requirement.currency == "USD"
    assert requirement.materiality.value == "material"
    assert requirement.required_by_relationship_ids == (
        "parameter:revenue_per_transaction",
    )


@pytest.mark.parametrize("mismatch", ("period", "scope", "unit"))
def test_historical_parameter_ratio_rejects_incompatible_graph_dimensions(mismatch):
    transactions = _node(
        "transactions", unit="transactions", unit_kind="count", currency=None
    )
    parameter = _node(
        "parameter",
        unit="USD/transactions",
        unit_kind="monetary_per_unit",
        denominator_unit="transactions",
        forecast_assumption_allowed=True,
    )
    relationship = EconomicRelationship(
        target="parameter",
        relationship_type="ratio",
        sources=(_edge("revenue"), _edge("transactions")),
        relationship_id="parameter:ratio",
        fiscal_period="FQ" if mismatch == "period" else "FY",
        historical_parameter_derivation=True,
    )
    if mismatch == "scope":
        transactions = transactions.model_copy(update={"scope_id": "other"})
    if mismatch == "unit":
        parameter = parameter.model_copy(update={"denominator_unit": "units"})
    model = EconomicModel(
        nodes=(_node("revenue"), transactions, parameter),
        relationships=(relationship,),
        fiscal_period="FY",
    )

    report = inspect_graph(model)

    assert not report.valid
    assert any(
        item.code
        in {
            "relationship_period_mismatch",
            "scope_mismatch",
            "ratio_denominator_unit",
        }
        for item in report.errors
    )


def test_residual_derives_history_and_targets_component_without_total_forecast():
    model = EconomicModel(
        nodes=(
            _node("total"),
            _node("known"),
            _node(
                "residual",
                "component",
                materiality="material",
                forecast_assumption_allowed=True,
            ),
        ),
        relationships=(
            EconomicRelationship(
                target="residual",
                relationship_type="residual",
                sources=(_edge("total"), _edge("known", sign=-1)),
                relationship_id="residual:from_total",
                provenance="formula:residual",
            ),
        ),
        observations=(
            _observation("total", 2024, "100"),
            _observation("known", 2024, "70"),
            _observation("known", 2025, "72"),
        ),
    )

    result = evaluate_graph(model, (2024, 2025))

    historical = result.cell("residual", 2024)
    assert historical is not None
    assert historical.value == Decimal("30")
    assert historical.origin == "derived_historical_residual"
    future = result.cell("residual", 2025)
    assert future is not None and not future.is_available
    assert {
        (item.node_id, item.fiscal_year)
        for item in result.unresolved_leaf_requirements
    } == {("residual", 2025)}
    requirement = result.unresolved_leaf_requirements[0]
    assert requirement.metric == "residual"
    assert requirement.required_by_relationship_ids == ("residual:from_total",)
