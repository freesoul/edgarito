from decimal import Decimal

from edgarito.services.forecasting import FcffForecastParameters
from edgarito.services.operating import OperatingForecastIntegrationService


def test_integration_returns_independent_selected_details_and_materialized_parameters():
    result = OperatingForecastIntegrationService().integrate(
        segments=(),
        definitions=(),
        historical_revenue={2026: Decimal("100"), 2027: Decimal("110")},
        explicit_anchors={2027: Decimal("125")},
        fiscal_years=(2026, 2027),
        fcff_parameters=FcffForecastParameters(forecast_years=2),
    )

    assert result.independent_forecast.consolidated_revenue == (
        Decimal("100"),
        Decimal("110"),
    )
    assert result.reconciled_forecast.consolidated_revenue == (
        Decimal("100"),
        Decimal("125"),
    )
    assert result.details.resolved_years[1].source == "explicit"
    assert result.fcff_parameters.revenue_anchors == {
        2026: Decimal("100"),
        2027: Decimal("125"),
    }


def test_integration_preserves_explicit_fcff_anchor_during_materialization():
    result = OperatingForecastIntegrationService().integrate(
        segments=(),
        definitions=(),
        historical_revenue={2026: Decimal("100")},
        fiscal_years=(2026,),
        parameters=FcffForecastParameters(
            forecast_years=1,
            revenue_anchors={2026: Decimal("90")},
            revenue_anchor_sources={2026: "explicit"},
        ),
    )

    assert result.parameters.revenue_anchors[2026] == Decimal("90")
    assert result.parameters.revenue_anchor_sources[2026].value == "explicit"
