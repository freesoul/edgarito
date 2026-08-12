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


def test_management_segment_revenue_constraint_applies_once_after_two_components():
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
    result = OperatingForecastService().forecast(
        segments=(_segment(),),
        definitions=definitions,
        observations=(
            _observation("volume", "10"),
            _observation("price", "2"),
            _observation("subscribers", "10"),
            _observation("arpu", "3"),
        ),
        management_constraints=(
            _observation("revenue", "50", origin="management_guidance"),
        ),
        fiscal_years=(2026,),
    )

    assert result.segment_forecasts[0].revenue == (Decimal("50"),)
    assert result.consolidated_revenue == (Decimal("50"),)
    aggregate = next(
        item
        for item in result.segment_forecasts[0].driver_forecasts
        if item.driver_id == "revenue"
    )
    assert aggregate.method == "segment aggregate revenue constraint"


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


@pytest.mark.parametrize(
    "segments",
    [
        (
            OperatingSegment(
                segment_id="company",
                name="Company",
                scope="consolidated",
                currency="USD",
            ),
            OperatingSegment(
                segment_id="cloud",
                name="Cloud",
                parent_id="company",
                scope="segment",
                currency="USD",
            ),
        ),
        (
            OperatingSegment(
                segment_id="parent",
                name="Parent",
                scope="segment",
                currency="USD",
            ),
            OperatingSegment(
                segment_id="child",
                name="Child",
                parent_id="parent",
                scope="product",
                currency="USD",
            ),
        ),
    ],
)
def test_consolidation_excludes_overlapping_parent_child_scopes(segments):
    definitions = tuple(
        _definition(
            OperatingArchetype.VOLUME_PRICE,
            ("volume", "price"),
            segment_id=segment.segment_id,
            driver_id=f"{segment.segment_id}-revenue",
        )
        for segment in segments
    )
    observations = tuple(
        item
        for segment in segments
        for item in (
            _observation("volume", "10", segment_id=segment.segment_id),
            _observation("price", "2", segment_id=segment.segment_id),
        )
    )
    result = OperatingForecastService().forecast(
        segments=segments,
        definitions=definitions,
        observations=observations,
        fiscal_years=(2026,),
    )

    assert result.consolidated_revenue == (Decimal("20"),)
    assert any("overlap" in warning for warning in result.warnings)


def test_consolidation_with_missing_child_is_unavailable_not_partial_total():
    segments = (
        _segment("parent"),
        OperatingSegment(
            segment_id="child",
            name="Child",
            parent_id="parent",
            scope="product",
            currency="USD",
        ),
    )
    result = OperatingForecastService().forecast(
        segments=segments,
        definitions=(
            _definition(
                OperatingArchetype.VOLUME_PRICE,
                ("volume", "price"),
                segment_id="parent",
                driver_id="parent-revenue",
            ),
        ),
        observations=(
            _observation("volume", "10", segment_id="parent"),
            _observation("price", "2", segment_id="parent"),
        ),
        fiscal_years=(2026,),
    )

    assert result.consolidated_revenue == (Decimal("20"),)
    assert result.source_by_year[2026] == "independent_operating"


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


@pytest.mark.parametrize(
    ("archetype", "metrics", "history", "observed", "expected"),
    [
        (
            OperatingArchetype.VOLUME_PRICE,
            ("volume", "price"),
            {2024: "20", 2025: "30"},
            {
                2024: {"volume": "10", "price": "2"},
                2025: {"volume": "12", "price": "2.5"},
                2026: {"volume": "14", "price": "3"},
            },
            "42",
        ),
        (
            OperatingArchetype.SUBSCRIBERS_ARPU,
            ("subscribers", "arpu"),
            {2024: "30", 2025: "48"},
            {
                2024: {"subscribers": "10", "arpu": "3"},
                2025: {"subscribers": "12", "arpu": "4"},
                2026: {"subscribers": "15", "arpu": "5"},
            },
            "75",
        ),
    ],
)
def test_historical_driver_reconstruction_is_audited_before_forward_use(
    archetype,
    metrics,
    history,
    observed,
    expected,
):
    result = OperatingForecastService().forecast(
        segments=(_segment(),),
        definitions=(_definition(archetype, metrics),),
        observations=tuple(
            _observation(metric, value, year)
            for year, values in observed.items()
            for metric, value in values.items()
        ),
        historical_revenue=history,
        fiscal_years=(2024, 2025, 2026),
    )

    segment = result.segment_forecasts[0]
    assert segment.driver_coverage == Decimal("1")
    assert segment.coverage_ratio == Decimal("1")
    assert segment.reconstruction_error == Decimal("0")
    assert segment.reconstruction_error_by_year == {
        2024: Decimal("0"),
        2025: Decimal("0"),
    }
    assert segment.supported_years == (2024, 2025)
    assert segment.confidence == "high"
    assert segment.confidence_by_year[2026] == "high"
    assert segment.revenue[-1] == Decimal(expected)
    assert any("supported years: FY2024, FY2025" in item for item in segment.warnings)


def test_low_driver_coverage_reduces_confidence_and_emits_warning():
    result = OperatingForecastService().forecast(
        segments=(_segment(),),
        definitions=(
            _definition(OperatingArchetype.VOLUME_PRICE, ("volume", "price")),
        ),
        observations=(
            _observation("volume", "10", 2024),
            _observation("price", "2", 2024),
            _observation("volume", "12", 2025),
            _observation("volume", "20", 2028),
            _observation("price", "3", 2028),
        ),
        historical_revenue={
            2024: Decimal("20"),
            2025: Decimal("24"),
            2026: Decimal("30"),
            2027: Decimal("36"),
        },
        fiscal_years=(2024, 2025, 2026, 2027, 2028),
    )

    segment = result.segment_forecasts[0]
    assert segment.driver_coverage == Decimal(1) / Decimal(4)
    assert segment.reconstruction_error == Decimal("0")
    assert segment.supported_years == (2024,)
    assert segment.confidence == "low"
    assert any("driver coverage is low" in item for item in segment.warnings)
    assert segment.confidence_by_year[2028] == "low"


def test_high_reconstruction_error_reduces_confidence_and_emits_warning():
    result = OperatingForecastService().forecast(
        segments=(_segment(),),
        definitions=(
            _definition(OperatingArchetype.SUBSCRIBERS_ARPU, ("subscribers", "arpu")),
        ),
        observations=tuple(
            _observation(metric, value, year)
            for year, values in {
                2024: {"subscribers": "10", "arpu": "3"},
                2025: {"subscribers": "12", "arpu": "4"},
                2026: {"subscribers": "15", "arpu": "5"},
            }.items()
            for metric, value in values.items()
        ),
        historical_revenue={2024: Decimal("40"), 2025: Decimal("60")},
        fiscal_years=(2024, 2025, 2026),
    )

    segment = result.segment_forecasts[0]
    assert segment.driver_coverage == Decimal("1")
    assert segment.reconstruction_error == Decimal("0.225")
    assert segment.supported_years == (2024, 2025)
    assert segment.confidence == "low"
    assert any("reconstruction error is high" in item for item in segment.warnings)
    assert segment.confidence_by_year[2026] == "low"


def test_company_reconstruction_audit_compares_consolidated_revenue():
    segments = (_segment("cloud"), _segment("subscriptions"))
    definitions = (
        _definition(
            OperatingArchetype.VOLUME_PRICE,
            ("volume", "price"),
            segment_id="cloud",
            driver_id="cloud-revenue",
        ),
        _definition(
            OperatingArchetype.SUBSCRIBERS_ARPU,
            ("subscribers", "arpu"),
            segment_id="subscriptions",
            driver_id="subscriptions-revenue",
        ),
    )
    observations = (
        _observation("volume", "10", 2024, segment_id="cloud"),
        _observation("price", "2", 2024, segment_id="cloud"),
        _observation("subscribers", "10", 2024, segment_id="subscriptions"),
        _observation("arpu", "3", 2024, segment_id="subscriptions"),
    )

    result = OperatingForecastService().forecast(
        segments=segments,
        definitions=definitions,
        observations=observations,
        historical_revenue={2024: Decimal("50")},
        fiscal_years=(2024, 2025),
    )

    assert result.driver_coverage == Decimal("1")
    assert result.reconstruction_error == Decimal("0")
    assert result.supported_years == (2024,)
    assert result.confidence == "high"
    assert result.reconstruction_error_by_year == {2024: Decimal("0")}
