import datetime
from decimal import Decimal

from edgarito.config.valuation import MultistageValuationConfiguration
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.forward import ForwardRevenueEstimate
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.operating import (
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
)
from edgarito.services.forecasting import FcffForecastParameters
from edgarito.services.operating import (
    OperatingForecastIntegrationService,
    OperatingForecastPipelineService,
)


def _financial_observation(concept, value, year):
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(value),
        unit="USD",
        granularity=Granularity.ANNUAL,
        fiscal_year=year,
        fiscal_period=FiscalPeriod.FY,
        period_end=datetime.date(year, 12, 31),
        provider="fixture",
        taxonomy="fixture",
        source_concept=concept.value,
    )


def _fcff_financials():
    values = {
        2023: {
            "revenue": "100",
            "operating_income": "20",
            "pretax_income": "18",
            "income_tax_expense": "3.6",
            "depreciation_and_amortization": "4",
            "capital_expenditures": "5",
            "accounts_receivable": "15",
            "inventory": "10",
            "prepaid_and_other_current_assets": "5",
            "accounts_payable": "8",
            "accrued_liabilities": "4",
            "deferred_revenue_current": "2",
        },
        2024: {
            "revenue": "120",
            "operating_income": "30",
            "pretax_income": "24",
            "income_tax_expense": "4.8",
            "depreciation_and_amortization": "5",
            "capital_expenditures": "6",
            "accounts_receivable": "18",
            "inventory": "12",
            "prepaid_and_other_current_assets": "6",
            "accounts_payable": "9",
            "accrued_liabilities": "5",
            "deferred_revenue_current": "2",
        },
    }
    concepts = {item.value: item for item in FinancialConcept}
    return NormalizedCompanyFinancials(
        provider="fixture",
        company_id="fixture-company",
        company_name="Fixture Company",
        observations=[
            _financial_observation(concepts[concept], value, year)
            for year, year_values in values.items()
            for concept, value in year_values.items()
        ],
    )


def _operating_fixture():
    segment = OperatingSegment(segment_id="cloud", name="Cloud", currency="USD")
    definition = OperatingDriverDefinition(
        driver_id="cloud-revenue",
        archetype=OperatingArchetype.VOLUME_PRICE,
        segment_id="cloud",
        output_metric="revenue",
        input_metrics=("volume", "price"),
        units={"volume": "units", "price": "USD/unit"},
        formula_id="volume_price",
        required_inputs=("volume", "price"),
    )
    observations = tuple(
        OperatingDriverObservation(
            segment_id="cloud",
            driver_id=metric,
            fiscal_year=year,
            value=Decimal(value),
            unit="units" if metric == "volume" else "USD/unit",
            origin="reported",
            confidence="high",
        )
        for year, values in {
            2024: {"volume": "60", "price": "2"},
            2025: {"volume": "70", "price": "2"},
        }.items()
        for metric, value in values.items()
    )
    return segment, definition, observations


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


def test_integration_normalizes_nested_segment_history_before_reconciliation():
    cloud, definition, observations = _operating_fixture()
    result = OperatingForecastIntegrationService().integrate(
        segments=(cloud,),
        definitions=(definition,),
        observations=observations,
        historical_revenue={"cloud": {2024: Decimal("120")}},
        fiscal_years=(2024, 2025, 2026),
        parameters=FcffForecastParameters(forecast_years=3),
    )

    assert result.reconciliation.resolved_years[0].historical_revenue == Decimal("120")
    assert result.parameters.revenue_anchors[2024] == Decimal("120")


def test_pipeline_composes_operating_reconciliation_into_fcff_and_adaptive_plan():
    segment, definition, observations = _operating_fixture()
    result = OperatingForecastPipelineService().forecast(
        _fcff_financials(),
        evidence={
            "segments": (segment,),
            "definitions": (definition,),
            "observations": observations,
            "historical_revenue": {2024: Decimal("120")},
        },
        parameters=FcffForecastParameters(forecast_years=3),
        consensus_estimates=(
            ForwardRevenueEstimate.from_value(2026, Decimal("150"), source="fixture"),
        ),
        terminal_growth_rate=Decimal("3"),
        adaptive_configuration=MultistageValuationConfiguration(
            terminal_return_on_invested_capital=Decimal("15")
        ),
    )

    assert result.reconciled_forecast.source_by_year[2025] == "independent_operating"
    assert result.reconciled_forecast.source_by_year[2026] == "analyst_consensus"
    assert result.reconciled_forecast.consensus_years == (2026,)
    assert result.forecast.observations[0].revenue == Decimal("140")
    assert result.forecast.observations[1].revenue == Decimal("150")
    assert result.forecast.observations[1].fcff == (
        result.forecast.observations[1].nopat
        + result.forecast.observations[1].depreciation_and_amortization
        - result.forecast.observations[1].capital_expenditures
        - result.forecast.observations[1].change_in_operating_working_capital
    )
    assert result.plan is not None
    assert result.plan.operating_consensus_years == (2026,)
    assert result.forecast.operating_driver_coverage == Decimal("1")
