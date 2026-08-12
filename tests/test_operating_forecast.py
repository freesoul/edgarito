from decimal import Decimal

import pytest

from edgarito.schemas.operating import (
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
)
from edgarito.schemas.valuation import AssumptionOrigin, AssumptionProvenance
from edgarito.services.operating import OperatingForecastService


def _segment(segment_id: str = "segment") -> OperatingSegment:
    return OperatingSegment(
        segment_id=segment_id,
        name=segment_id.title(),
        currency="USD",
    )


def _definition(
    archetype: OperatingArchetype,
    input_metrics: tuple[str, ...],
    *,
    segment_id: str = "segment",
    driver_id: str = "revenue-driver",
    units: dict[str, str] | None = None,
) -> OperatingDriverDefinition:
    return OperatingDriverDefinition(
        driver_id=driver_id,
        archetype=archetype,
        segment_id=segment_id,
        output_metric="revenue",
        input_metrics=input_metrics,
        units=units or {metric: "unit" for metric in input_metrics},
        formula_id=archetype.value,
        required_inputs=input_metrics,
    )


def _observation(
    driver_id: str,
    value: str,
    year: int = 2026,
    *,
    segment_id: str = "segment",
    origin: str = "reported",
    confidence: str = "high",
    provenance: AssumptionProvenance | None = None,
    low: str | None = None,
    high: str | None = None,
) -> OperatingDriverObservation:
    return OperatingDriverObservation(
        segment_id=segment_id,
        driver_id=driver_id,
        fiscal_year=year,
        value=Decimal(value) if value is not None else None,
        low=Decimal(low) if low is not None else None,
        high=Decimal(high) if high is not None else None,
        unit="unit",
        origin=origin,
        confidence=confidence,
        provenance=provenance,
    )


@pytest.mark.parametrize(
    ("archetype", "metrics", "values", "units", "expected"),
    [
        (
            OperatingArchetype.VOLUME_PRICE,
            ("volume", "price"),
            {"volume": "10", "price": "2"},
            {"volume": "units", "price": "USD/unit"},
            "20",
        ),
        (
            OperatingArchetype.SUBSCRIBERS_ARPU,
            ("subscribers", "arpu"),
            {"subscribers": "10", "arpu": "3"},
            {"subscribers": "users", "arpu": "USD/user"},
            "30",
        ),
        (
            OperatingArchetype.CAPACITY_UTILIZATION_PRICE,
            ("capacity", "utilization", "price"),
            {"capacity": "10", "utilization": "0.8", "price": "2"},
            {"capacity": "units", "utilization": "ratio", "price": "USD/unit"},
            "16",
        ),
        (
            OperatingArchetype.TRANSACTIONS_TAKE_RATE,
            ("transactions", "take_rate"),
            {"transactions": "10", "take_rate": "0.1"},
            {"transactions": "transactions", "take_rate": "ratio"},
            "1",
        ),
        (
            OperatingArchetype.BACKLOG_CONVERSION,
            ("backlog", "conversion_rate"),
            {"backlog": "100", "conversion_rate": "0.25"},
            {"backlog": "USD", "conversion_rate": "ratio"},
            "25",
        ),
        (
            OperatingArchetype.STORE_COUNT_SALES_PER_STORE,
            ("store_count", "sales_per_store"),
            {"store_count": "5", "sales_per_store": "4"},
            {"store_count": "stores", "sales_per_store": "USD/store"},
            "20",
        ),
    ],
)
def test_operating_archetype_formulas_are_deterministic(
    archetype,
    metrics,
    values,
    units,
    expected,
):
    result = OperatingForecastService().forecast(
        segments=(_segment(),),
        definitions=(_definition(archetype, metrics, units=units),),
        observations=tuple(
            _observation(metric, value) for metric, value in values.items()
        ),
        fiscal_years=(2026,),
    )

    assert result.consolidated_revenue == (Decimal(expected),)
    assert result.segment_forecasts[0].source_by_year == {2026: "independent_operating"}
    assert result.segment_forecasts[0].driver_forecasts


def test_generic_segment_growth_uses_prior_absolute_revenue():
    result = OperatingForecastService().forecast(
        segments=(_segment(),),
        definitions=(
            _definition(
                OperatingArchetype.GENERIC_SEGMENT_GROWTH,
                ("growth",),
                units={"growth": "ratio"},
            ),
        ),
        observations=(_observation("growth", "0.10"),),
        historical_revenue={2025: Decimal("100")},
        fiscal_years=(2026,),
    )

    assert result.consolidated_revenue == (Decimal("110.00"),)
    assert result.consolidated_growth == (None,)


def test_missing_required_input_is_a_warning_not_an_invented_forecast():
    result = OperatingForecastService().forecast(
        segments=(_segment(),),
        definitions=(
            _definition(
                OperatingArchetype.VOLUME_PRICE,
                ("volume", "price"),
            ),
        ),
        observations=(_observation("volume", "10"),),
        fiscal_years=(2026,),
    )

    segment = result.segment_forecasts[0]
    assert segment.revenue == (Decimal("0"),)
    assert segment.source_by_year[2026] == "unavailable"
    assert segment.confidence_by_year[2026] == "low"
    assert any(
        "missing required input 'price'" in warning for warning in segment.warnings
    )


def test_management_observation_anchors_same_metric_and_preserves_provenance():
    provenance = AssumptionProvenance(
        origin=AssumptionOrigin.EXPLICIT,
        methodology="filed management guidance",
    )
    result = OperatingForecastService().forecast(
        segments=(_segment(),),
        definitions=(
            _definition(
                OperatingArchetype.VOLUME_PRICE,
                ("volume", "price"),
            ),
        ),
        observations=(
            _observation("volume", "10"),
            _observation("price", "2"),
        ),
        management_constraints=(
            _observation(
                "price",
                "3",
                origin="management_guidance",
                provenance=provenance,
            ),
        ),
        fiscal_years=(2026,),
    )

    segment = result.segment_forecasts[0]
    price_forecast = next(
        item for item in segment.driver_forecasts if item.driver_id == "price"
    )
    assert result.consolidated_revenue == (Decimal("30"),)
    assert segment.source_by_year[2026] == "management_guidance"
    assert price_forecast.value == Decimal("3")
    assert price_forecast.source == "management_guidance"
    assert price_forecast.provenance == provenance


def test_management_output_constraint_anchors_revenue_point():
    result = OperatingForecastService().forecast(
        segments=(_segment(),),
        definitions=(
            _definition(
                OperatingArchetype.VOLUME_PRICE,
                ("volume", "price"),
            ),
        ),
        observations=(
            _observation("volume", "10"),
            _observation("price", "2"),
        ),
        management_constraints=(
            _observation(
                "revenue-driver",
                "50",
                origin="management_guidance",
            ),
        ),
        fiscal_years=(2026,),
    )

    assert result.consolidated_revenue == (Decimal("50"),)
    revenue_forecast = next(
        item
        for item in result.segment_forecasts[0].driver_forecasts
        if item.driver_id == "revenue-driver"
    )
    assert revenue_forecast.constraint == "management revenue point constraint"


def test_consolidation_and_growth_are_derived_from_segment_absolute_revenue():
    segments = (_segment("cloud"), _segment("stores"))
    definitions = (
        _definition(
            OperatingArchetype.VOLUME_PRICE,
            ("volume", "price"),
            segment_id="cloud",
            driver_id="cloud-revenue",
        ),
        _definition(
            OperatingArchetype.STORE_COUNT_SALES_PER_STORE,
            ("store_count", "sales_per_store"),
            segment_id="stores",
            driver_id="stores-revenue",
        ),
    )
    observations = (
        _observation("volume", "10", segment_id="cloud"),
        _observation("price", "2", segment_id="cloud"),
        _observation("store_count", "5", segment_id="stores"),
        _observation("sales_per_store", "6", segment_id="stores"),
    )
    result = OperatingForecastService().forecast(
        segments=segments,
        definitions=definitions,
        observations=observations,
        fiscal_years=(2026,),
    )

    assert [item.revenue for item in result.segment_forecasts] == [
        (Decimal("20"),),
        (Decimal("30"),),
    ]
    assert result.consolidated_revenue == (Decimal("50"),)
    assert result.consolidated_growth == (None,)


def test_transition_starts_after_last_explicit_operating_year():
    definition = _definition(
        OperatingArchetype.VOLUME_PRICE,
        ("volume", "price"),
    )
    observations = tuple(
        item
        for year in (2026, 2027)
        for item in (
            _observation("volume", "10", year),
            _observation("price", str(year - 2024), year),
        )
    )
    result = OperatingForecastService().forecast(
        segments=(_segment(),),
        definitions=(definition,),
        observations=observations,
        historical_revenue={2025: Decimal("15")},
        fiscal_years=(2025, 2026, 2027, 2028),
    )

    assert result.explicit_years == (2026, 2027)
    assert result.transition_start_year == 2028
    assert result.segment_forecasts[0].source_by_year[2025] == "normalized_historical"
    assert result.segment_forecasts[0].source_by_year[2028] == "unavailable"
