import asyncio
import datetime
from decimal import Decimal

import pytest
from test_driver_based_fcff import _financials

from edgarito.schemas.forecasting import FcffForecastParameters
from edgarito.schemas.operating import (
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
)
from edgarito.services.forecasting import (
    DriverBasedFcffForecastService,
    DriverBasedForecastIncompleteError,
    EconomicGraphDriverBasedForecastService,
    EconomicLeafReasoningAssumption,
    EconomicLeafReasoningResponse,
    ForecastReasoner,
)
from edgarito.services.operating._graph import build_legacy_economic_graph

D = Decimal


def _graph_fixture(*, forecast_observations=True):
    segment = OperatingSegment(segment_id="saas", name="SaaS", currency="USD")
    definitions = (
        OperatingDriverDefinition(
            driver_id="subscriptions",
            archetype=OperatingArchetype.SUBSCRIBERS_ARPU,
            segment_id="saas",
            output_metric="revenue",
            input_metrics=("subscribers", "arpu"),
            units={"subscribers": "users", "arpu": "USD/user"},
            formula_id="subscribers_arpu",
            required_inputs=("subscribers", "arpu"),
        ),
        OperatingDriverDefinition(
            driver_id="api",
            archetype=OperatingArchetype.VOLUME_PRICE,
            segment_id="saas",
            output_metric="revenue",
            input_metrics=("volume", "price"),
            units={"volume": "call", "price": "USD/call"},
            formula_id="volume_price",
            required_inputs=("volume", "price"),
        ),
    )
    values = {
        "subscribers": ("20", "100", "110"),
        "arpu": ("10", "10", "10"),
        "volume": ("100", "1000", "1200"),
        "price": ("0.1", "0.1", "0.1"),
    }
    observations = []
    for metric, (historical, first, second), unit in (
        ("subscribers", values["subscribers"], "users"),
        ("arpu", values["arpu"], "USD/user"),
            ("volume", values["volume"], "call"),
        ("price", values["price"], "USD/call"),
    ):
        year_values = [(2024, historical)]
        if forecast_observations:
            year_values.extend(((2025, first), (2026, second)))
        observations.extend(
            OperatingDriverObservation(
                segment_id="saas",
                driver_id=metric,
                fiscal_year=year,
                value=D(value),
                unit=unit,
                origin="reported",
                confidence="high",
            )
            for year, value in year_values
        )
    return build_legacy_economic_graph(
        (segment,), definitions, tuple(observations), company_id="fixture-company"
    ).model


def _economic_observations():
    values = {
        2025: {
            "gross_profit": "660",
            "r_and_d": "50",
            "sg_and_a": "100",
            "other_operating_items": "0",
            "tax_rate": "20",
            "depreciation_and_amortization": "20",
            "capital_expenditures": "30",
            "operating_working_capital": "50",
        },
        2026: {
            "gross_profit": "732",
            "r_and_d": "55",
            "sg_and_a": "110",
            "other_operating_items": "0",
            "tax_rate": "20",
            "depreciation_and_amortization": "22",
            "capital_expenditures": "33",
            "operating_working_capital": "55",
        },
    }
    return tuple(
        OperatingDriverObservation(
            segment_id="company",
            driver_id=metric,
            fiscal_year=year,
            value=D(value),
            unit="percent" if metric == "tax_rate" else "USD",
            scope="company",
            is_total=True,
            origin="first_party_observation",
            confidence="high",
        )
        for year, metrics in values.items()
        for metric, value in metrics.items()
    )


class _GraphClient:
    model = "graph-test"
    reasoning_effort = "low"

    def __init__(self, model):
        self.model_definition = model

    async def extract_structured(self, **_kwargs):
        nodes = self.model_definition.node_by_id
        paths = {
            "subscribers": ("100", "110"),
            "arpu": ("10", "10"),
            "volume": ("1000", "1200"),
            "price": ("0.1", "0.1"),
        }
        assumptions = tuple(
            EconomicLeafReasoningAssumption(
                node_id=node.node_id,
                fiscal_years=(2025, 2026),
                low=tuple(D(value) - 1 for value in paths[node.metric]),
                base=tuple(D(value) for value in paths[node.metric]),
                high=tuple(D(value) + 1 for value in paths[node.metric]),
                unit=node.unit,
                rationale="bounded test leaf",
                confidence="high",
                model_assumption=True,
            )
            for node in nodes.values()
            if node.metric in paths and node.node_type.value == "input"
        )
        return EconomicLeafReasoningResponse(assumptions=assumptions)


def test_graph_reasoning_composes_leaf_revenue_with_existing_fcff_economics(tmp_path):
    model = _graph_fixture(forecast_observations=False)
    result = asyncio.run(
        EconomicGraphDriverBasedForecastService(
            reasoner=ForecastReasoner(
                _GraphClient(model), cache=tmp_path
            )
        ).forecast(
            _financials(),
            economic_model=model,
            parameters=FcffForecastParameters(forecast_years=2),
            forecast_years=(2025, 2026),
            as_of=datetime.date(2025, 1, 1),
            observations=_economic_observations(),
        )
    )

    assert result.readiness.ready
    assert len(result.graph_result.compiled_observations) == 8
    assert result.graph_result.evaluation.value(model.revenue_root, 2025) == D("1100.0")
    assert result.graph_result.evaluation.value(model.revenue_root, 2026) == D("1220.0")
    assert not any(
        item.node_id == model.revenue_root
        for item in result.graph_result.accepted_assumptions
    )
    operating = result.company_operating_forecast
    assert operating.consolidated_revenue == (D("1100.0"), D("1220.0"))
    assert operating.source_by_year == {2025: "economic_graph", 2026: "economic_graph"}
    for item in result.forecast.observations:
        assert item.fcff == item.nopat + item.depreciation_and_amortization - item.capital_expenditures - item.change_in_operating_working_capital
    assert "revenue_source=economic_graph" in result.audit


def test_missing_graph_leaf_raises_structured_driver_incomplete_error():
    model = _graph_fixture(forecast_observations=False)
    with pytest.raises(DriverBasedForecastIncompleteError) as error:
        DriverBasedFcffForecastService().forecast(
            _financials(),
            FcffForecastParameters(forecast_years=2),
            economic_model=model,
            observations=_economic_observations(),
        )

    assert not error.value.readiness.ready
    assert any("Economic graph unresolved leaves" in item for item in error.value.readiness.canonical_errors)
    assert any("price" in item for item in error.value.readiness.canonical_errors)
