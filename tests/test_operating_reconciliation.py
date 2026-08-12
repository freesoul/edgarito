from decimal import Decimal

from edgarito.schemas.forward import ForwardRevenueEstimate
from edgarito.schemas.operating import CompanyOperatingForecast
from edgarito.services.forecasting import FcffForecastParameters
from edgarito.services.operating import (
    RevenueForecastReconciler,
    materialize_revenue_anchors,
)


def _growth(values: tuple[Decimal, ...]) -> tuple[Decimal | None, ...]:
    result: list[Decimal | None] = [None]
    for previous, current in zip(values[:-1], values[1:], strict=True):
        if previous == 0:
            result.append(Decimal(0) if current == 0 else None)
        else:
            result.append((current / previous - Decimal(1)) * Decimal(100))
    return tuple(result)


def _independent(
    years: tuple[int, ...] = (2026, 2027, 2028),
    revenue: tuple[str, ...] = ("100", "110", "120"),
    *,
    sources: dict[int, str] | None = None,
    confidences: dict[int, str] | None = None,
) -> CompanyOperatingForecast:
    values = tuple(Decimal(value) for value in revenue)
    return CompanyOperatingForecast(
        company_id="example",
        fiscal_years=years,
        consolidated_revenue=values,
        consolidated_growth=_growth(values),
        explicit_years=tuple(
            year
            for year in years
            if (sources or {}).get(year) == "independent_operating"
        ),
        source_by_year=sources or {year: "independent_operating" for year in years},
        confidence_by_year=confidences or {year: "high" for year in years},
    )


def test_reconciler_preserves_explicit_and_management_precedence():
    result = RevenueForecastReconciler().reconcile(
        _independent(),
        consensus=(
            ForwardRevenueEstimate.from_value(2026, Decimal("200"), source="yahoo"),
            ForwardRevenueEstimate.from_value(2027, Decimal("210"), source="yahoo"),
        ),
        historical_revenue={2026: Decimal("90"), 2027: Decimal("95")},
        explicit_anchors={2026: Decimal("125")},
        management_anchors={2027: Decimal("135")},
    )

    assert result.consolidated_revenue == (
        Decimal("125"),
        Decimal("135"),
        Decimal("120"),
    )
    assert result.source_by_year == {
        2026: "explicit",
        2027: "management_guidance",
        2028: "independent_operating",
    }


def test_partial_segments_do_not_demote_management_selection_to_consensus():
    forecast = _independent(
        years=(2026,),
        revenue=("100",),
        sources={2026: "management_guidance"},
        confidences={2026: "high"},
    )
    result = RevenueForecastReconciler().reconcile(
        forecast,
        consensus={2026: Decimal("250")},
        management_anchors={2026: Decimal("125")},
    )

    assert result.consolidated_revenue == (Decimal("125"),)
    assert result.source_by_year[2026] == "management_guidance"


def test_reconciler_uses_consensus_only_to_fill_missing_independent_years():
    result = RevenueForecastReconciler().reconcile(
        _independent(
            sources={
                2026: "independent_operating",
                2027: "unavailable",
                2028: "independent_operating",
            },
            confidences={2026: "high", 2027: "low", 2028: "high"},
            revenue=("100", "0", "120"),
        ),
        consensus=(
            ForwardRevenueEstimate.from_value(2027, Decimal("115"), source="yahoo"),
            ForwardRevenueEstimate.from_value(2028, Decimal("999"), source="yahoo"),
        ),
    )

    assert result.consolidated_revenue == (
        Decimal("100"),
        Decimal("115"),
        Decimal("120"),
    )
    assert result.source_by_year[2027] == "analyst_consensus"
    assert result.source_by_year[2028] == "independent_operating"


def test_reconciler_falls_back_to_normalized_history_after_consensus():
    result = RevenueForecastReconciler().reconcile(
        _independent(
            sources={2026: "unavailable", 2027: "unavailable", 2028: "unavailable"},
            confidences={2026: "low", 2027: "low", 2028: "low"},
            revenue=("0", "0", "0"),
        ),
        consensus=(
            ForwardRevenueEstimate.from_value(2026, Decimal("105"), source="yahoo"),
        ),
        historical_revenue={2027: Decimal("100"), 2028: Decimal("102")},
    )

    assert result.consolidated_revenue == (
        Decimal("105"),
        Decimal("100"),
        Decimal("102"),
    )
    assert result.source_by_year[2027] == "normalized_historical"
    assert result.source_by_year[2028] == "normalized_historical"


def test_reconciler_derives_growth_from_selected_absolute_revenue():
    result = RevenueForecastReconciler().reconcile(
        _independent(
            revenue=("100", "100", "100"),
            sources={2026: "unavailable", 2027: "unavailable", 2028: "unavailable"},
            confidences={2026: "low", 2027: "low", 2028: "low"},
        ),
        consensus=(
            ForwardRevenueEstimate.from_value(2026, Decimal("120"), source="yahoo"),
            ForwardRevenueEstimate.from_value(2027, Decimal("150"), source="yahoo"),
        ),
        historical_revenue={2028: Decimal("165")},
    )

    assert result.consolidated_revenue == (
        Decimal("120"),
        Decimal("150"),
        Decimal("165"),
    )
    assert result.consolidated_growth == (
        None,
        Decimal("25"),
        Decimal("10"),
    )


def test_transition_starts_after_final_selected_explicit_year():
    result = RevenueForecastReconciler().reconcile(
        _independent(
            years=(2026, 2027, 2028, 2029),
            revenue=("0", "0", "0", "0"),
            sources={
                2026: "unavailable",
                2027: "unavailable",
                2028: "unavailable",
                2029: "unavailable",
            },
            confidences={year: "low" for year in (2026, 2027, 2028, 2029)},
        ),
        consensus=(
            ForwardRevenueEstimate.from_value(2026, Decimal("100"), source="yahoo"),
            ForwardRevenueEstimate.from_value(2027, Decimal("110"), source="yahoo"),
            ForwardRevenueEstimate.from_value(2028, Decimal("120"), source="yahoo"),
        ),
        historical_revenue={2029: Decimal("125")},
    )

    assert result.explicit_years == (2026, 2027, 2028)
    assert result.transition_start_year == 2029


def test_materialize_selected_revenue_preserves_existing_fcff_anchor_priority():
    parameters = FcffForecastParameters(
        forecast_years=3,
        revenue_anchors={2026: Decimal("125")},
        revenue_anchor_sources={2026: "explicit"},
    )
    selected = RevenueForecastReconciler().reconcile_with_details(
        _independent(
            revenue=("100", "110", "120"),
            sources={year: "independent_operating" for year in (2026, 2027, 2028)},
        ),
    )

    materialized = materialize_revenue_anchors(parameters, selected)

    assert materialized.revenue_anchors == {
        2026: Decimal("125"),
        2027: Decimal("110"),
        2028: Decimal("120"),
    }
    assert materialized.revenue_anchor_sources[2026].value == "explicit"
    assert materialized.revenue_anchor_sources[2027].value == "forward_evidence"


def test_materialize_selected_consensus_outranks_normalized_history_anchor():
    parameters = FcffForecastParameters(
        forecast_years=1,
        revenue_anchors={2026: Decimal("90")},
        revenue_anchor_sources={2026: "normalized_historical"},
    )

    materialized = materialize_revenue_anchors(
        parameters,
        {2026: (Decimal("105"), "analyst_consensus")},
    )

    assert materialized.revenue_anchors[2026] == Decimal("105")
    assert materialized.revenue_anchor_sources[2026].value == "forward_evidence"


def test_reconciliation_retains_per_year_selected_and_candidate_revenue_audit():
    result = RevenueForecastReconciler().reconcile(
        _independent(
            years=(2026, 2027),
            revenue=("100", "0"),
            sources={2026: "independent_operating", 2027: "unavailable"},
            confidences={2026: "high", 2027: "low"},
        ),
        consensus={2027: Decimal("115")},
        management_anchors={2026: Decimal("110")},
    )

    assert result.selected_revenue_by_year == {
        2026: Decimal("110"),
        2027: Decimal("115"),
    }
    assert result.selected_source_by_year == {
        2026: "management_guidance",
        2027: "analyst_consensus",
    }
    assert result.independent_revenue_by_year == {
        2026: Decimal("100"),
        2027: Decimal("0"),
    }
    assert result.consensus_revenue_by_year == {2027: Decimal("115")}
    assert result.management_revenue_by_year == {2026: Decimal("110")}


def test_reconciliation_exposes_supported_consensus_divergence_and_transition_audit():
    result = RevenueForecastReconciler().reconcile_with_details(
        _independent(
            years=(2026, 2027, 2028),
            revenue=("100", "0", "120"),
            sources={
                2026: "independent_operating",
                2027: "unavailable",
                2028: "independent_operating",
            },
            confidences={2026: "high", 2027: "low", 2028: "high"},
        ),
        consensus=(
            ForwardRevenueEstimate.from_value(2027, Decimal("125"), source="yahoo"),
            ForwardRevenueEstimate.from_value(2028, Decimal("999"), source="yahoo"),
        ),
    )

    assert result.own_supported_years == (2026, 2028)
    assert result.consensus_years == (2027,)
    assert result.divergence_by_year == {
        2028: (Decimal("999") / Decimal("120") - Decimal("1")) * Decimal("100")
    }
    assert result.divergence == result.divergence_by_year[2028].copy_abs()
    assert result.transition_start_year == 2029
    assert any(
        "Consensus revenue years: FY2027" in warning for warning in result.warnings
    )
