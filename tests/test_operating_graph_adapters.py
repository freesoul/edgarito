from decimal import Decimal

import pytest

from edgarito.schemas.operating import (
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
)
from edgarito.services.operating._forecast.service import OperatingForecastService
from edgarito.services.operating._graph import (
    EconomicForecastAdaptationError,
    adapt_economic_forecast,
    build_legacy_economic_graph,
    evaluate_graph,
)


def _segment(segment_id: str = "segment") -> OperatingSegment:
    return OperatingSegment(segment_id=segment_id, name="Segment", currency="USD")


def _definition(
    archetype: OperatingArchetype,
    metrics: tuple[str, ...],
    *,
    driver_id: str = "revenue-driver",
    units: dict[str, str] | None = None,
) -> OperatingDriverDefinition:
    return OperatingDriverDefinition(
        driver_id=driver_id,
        archetype=archetype,
        segment_id="segment",
        output_metric="revenue",
        input_metrics=metrics,
        units=units or {metric: "unit" for metric in metrics},
        formula_id=archetype.value,
        required_inputs=metrics,
    )


def _observation(driver_id: str, value: str, year: int = 2026):
    return OperatingDriverObservation(
        segment_id="segment",
        driver_id=driver_id,
        fiscal_year=year,
        value=Decimal(value),
        unit="unit",
        origin="reported",
        confidence="high",
    )


@pytest.mark.parametrize(
    ("archetype", "metrics", "values", "units", "expected"),
    [
        (
            OperatingArchetype.VOLUME_PRICE,
            ("volume", "price"),
            ("10", "2"),
            {"volume": "units", "price": "USD/unit"},
            "20",
        ),
        (
            OperatingArchetype.SUBSCRIBERS_ARPU,
            ("subscribers", "arpu"),
            ("10", "3"),
            {"subscribers": "users", "arpu": "USD/user"},
            "30",
        ),
        (
            OperatingArchetype.CAPACITY_UTILIZATION_PRICE,
            ("capacity", "utilization", "price"),
            ("10", "0.8", "2"),
            {
                "capacity": "units",
                "utilization": "ratio",
                "price": "USD/unit",
            },
            "16",
        ),
        (
            OperatingArchetype.TRANSACTIONS_TAKE_RATE,
            ("transactions", "take_rate"),
            ("10", "0.1"),
            {"transactions": "transactions", "take_rate": "ratio"},
            "1.0",
        ),
        (
            OperatingArchetype.BACKLOG_CONVERSION,
            ("backlog", "conversion_rate"),
            ("100", "0.25"),
            {"backlog": "USD", "conversion_rate": "ratio"},
            "25",
        ),
        (
            OperatingArchetype.STORE_COUNT_SALES_PER_STORE,
            ("store_count", "sales_per_store"),
            ("5", "4"),
            {"store_count": "stores", "sales_per_store": "USD/store"},
            "20",
        ),
    ],
)
def test_legacy_graph_adapter_matches_legacy_numeric_dump(
    archetype, metrics, values, units, expected
):
    definition = _definition(archetype, metrics, units=units)
    observations = tuple(
        _observation(metric, value)
        for metric, value in zip(metrics, values, strict=True)
    )
    legacy = OperatingForecastService().forecast(
        (_segment(),), (definition,), observations, fiscal_years=(2026,)
    )
    fragments = build_legacy_economic_graph((_segment(),), (definition,), observations)
    evaluation = evaluate_graph(fragments.model, (2026,))
    adapted = adapt_economic_forecast(
        fragments.model,
        evaluation,
        business_roots={fragments.model.business_roots[0]: _segment()},
    ).company_forecast

    assert (
        adapted.model_dump()["consolidated_revenue"]
        == legacy.model_dump()["consolidated_revenue"]
        == (Decimal(expected),)
    )
    assert (
        adapted.model_dump()["consolidated_growth"]
        == legacy.model_dump()["consolidated_growth"]
    )
    assert adapted.source_by_year == legacy.source_by_year


def test_generic_growth_uses_legacy_historical_revenue_seed():
    definition = _definition(
        OperatingArchetype.GENERIC_SEGMENT_GROWTH,
        ("growth",),
        units={"growth": "ratio"},
    )
    observations = (_observation("growth", "0.10"),)
    fragments = build_legacy_economic_graph(
        (_segment(),),
        (definition,),
        observations,
        historical_revenue={2025: Decimal("100")},
    )
    evaluation = evaluate_graph(fragments.model, (2026,))
    adapted = adapt_economic_forecast(
        fragments.model,
        evaluation,
        business_roots={fragments.model.business_roots[0]: _segment()},
    ).company_forecast

    assert adapted.consolidated_revenue == (Decimal("110.0"),)
    assert adapted.consolidated_growth == (None,)


def test_multiple_definitions_add_and_shared_inputs_are_explicit_and_deduplicated():
    definitions = (
        _definition(
            OperatingArchetype.VOLUME_PRICE,
            ("volume", "price"),
            driver_id="volume-revenue",
        ),
        _definition(
            OperatingArchetype.SUBSCRIBERS_ARPU,
            ("subscribers", "arpu"),
            driver_id="subscriber-revenue",
        ),
    )
    observations = (
        _observation("volume", "10"),
        _observation("price", "2"),
        _observation("subscribers", "10"),
        _observation("arpu", "3"),
    )
    fragments = build_legacy_economic_graph((_segment(),), definitions, observations)
    evaluation = evaluate_graph(fragments.model, (2026,))

    segment_id = _segment().segment_id
    segment_root = fragments.revenue_root_by_segment[segment_id]
    segment_adds = [
        relationship
        for relationship in fragments.model.relationships
        if relationship.target == segment_root
    ]
    assert len(segment_adds) == 1
    assert segment_adds[0].relationship_type.value == "add"
    assert (
        len(
            [
                item
                for item in fragments.model.observations
                if item.node_id == fragments.input_node_by_key[segment_id, "volume"]
            ]
        )
        == 1
    )
    assert evaluation.value(fragments.model.revenue_root, 2026) == Decimal("50")


def test_missing_root_fails_without_zero_and_company_only_has_no_fake_segment():
    definition = _definition(OperatingArchetype.VOLUME_PRICE, ("volume", "price"))
    fragments = build_legacy_economic_graph(
        (_segment(),), (definition,), (_observation("volume", "10"),)
    )
    evaluation = evaluate_graph(fragments.model, (2026,))
    assert evaluation.value(fragments.model.revenue_root, 2026) is None
    with pytest.raises(EconomicForecastAdaptationError, match="unavailable"):
        adapt_economic_forecast(
            fragments.model,
            evaluation,
            business_roots={fragments.model.business_roots[0]: _segment()},
        )
