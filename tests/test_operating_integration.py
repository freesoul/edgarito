import asyncio
import datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

import edgarito.cli.__main__ as cli_main
from edgarito.config.valuation import MultistageValuationConfiguration
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.enums.market import Market
from edgarito.schemas.forecasting import FcffForecastParameters
from edgarito.schemas.forward import ForwardRevenueEstimate
from edgarito.schemas.guidance.management import (
    GuidanceApplication,
    GuidanceBasis,
    GuidanceMetric,
    GuidanceOverlayResult,
    GuidancePeriodType,
    GuidanceQualifier,
    GuidanceScope,
    GuidanceStatus,
    GuidanceValueKind,
    ManagementGuidance,
)
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
from edgarito.services.financials.availability import ObservationAvailabilityMode
from edgarito.services.forecasting._fcff.service import FcffForecastService
from edgarito.services.guidance.resolver import ManagementGuidanceResolver
from edgarito.services.operating._discovery.service import (
    OperatingEvidenceDiscoveryService,
)
from edgarito.services.operating.contracts import OperatingForecastQualityError
from edgarito.services.operating.integration import (
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


def _tax_guidance(value, filed):
    return ManagementGuidance(
        metric=GuidanceMetric.TAX_RATE,
        fiscal_year=2025,
        period_type=GuidancePeriodType.FISCAL_YEAR,
        point=Decimal(value),
        value_kind=GuidanceValueKind.PERCENTAGE,
        unit="percent",
        basis=GuidanceBasis.GAAP,
        scope=GuidanceScope.CONSOLIDATED,
        qualifier=GuidanceQualifier.POINT,
        status=GuidanceStatus.ISSUED,
        filing_date=filed,
        accession_number=f"guidance-{value}",
        filing_form="8-K",
        source_document="ex991.htm",
        source_document_type="EX-99.1",
        supporting_text=f"We expect a tax rate of {value} percent.",
        evidence_verified=True,
        extraction_model="test",
    )


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


def test_integration_resolves_raw_management_guidance_at_as_of_boundary():
    pre = _tax_guidance("20", datetime.date(2024, 1, 15))
    post = _tax_guidance("30", datetime.date(2024, 3, 15))
    result = OperatingForecastIntegrationService().integrate(
        segments=(),
        definitions=(),
        management_constraints=(pre, post),
        historical_revenue={2025: Decimal("100")},
        fiscal_years=(2025,),
        parameters=FcffForecastParameters(forecast_years=1),
        as_of=datetime.date(2024, 2, 1),
    )

    economics = result.independent_forecast.operating_economics
    assert economics is not None
    assert economics.tax_rate == (Decimal("20"),)
    assert economics.tax_rate_provenance_by_year[2025].accession == "guidance-20"

    without_snapshot = OperatingForecastIntegrationService().integrate(
        segments=(),
        definitions=(),
        management_constraints=(post,),
        historical_revenue={2025: Decimal("100")},
        fiscal_years=(2025,),
        parameters=FcffForecastParameters(forecast_years=1),
    )
    no_snapshot_economics = without_snapshot.independent_forecast.operating_economics
    assert no_snapshot_economics is not None
    assert no_snapshot_economics.tax_rate == (Decimal("30"),)


def test_integration_re_resolves_later_guidance_container_at_current_as_of():
    pre = _tax_guidance("20", datetime.date(2024, 1, 15))
    post = _tax_guidance("30", datetime.date(2024, 3, 15))
    later_snapshot = ManagementGuidanceResolver().resolve(
        [pre, post], as_of=datetime.date(2024, 4, 1)
    )

    current_snapshot = OperatingForecastIntegrationService().integrate(
        segments=(),
        definitions=(),
        management_constraints=later_snapshot,
        historical_revenue={2025: Decimal("100")},
        fiscal_years=(2025,),
        parameters=FcffForecastParameters(forecast_years=1),
        as_of=datetime.date(2024, 2, 1),
    )

    economics = current_snapshot.independent_forecast.operating_economics
    assert economics is None


def test_integration_filters_post_as_of_guidance_overlay_application():
    post = _tax_guidance("30", datetime.date(2024, 3, 15))
    overlay = GuidanceOverlayResult(
        applications=(
            GuidanceApplication(
                driver="tax_rate",
                fiscal_year=2025,
                value=Decimal("30"),
                guidance=post,
                methodology="fixture application",
            ),
        )
    )

    result = OperatingForecastIntegrationService().integrate(
        segments=(),
        definitions=(),
        management_constraints=overlay,
        historical_revenue={2025: Decimal("100")},
        fiscal_years=(2025,),
        parameters=FcffForecastParameters(forecast_years=1),
        as_of=datetime.date(2024, 2, 1),
    )

    assert result.independent_forecast.operating_economics is None


@pytest.mark.parametrize(
    "availability_mode",
    [ObservationAvailabilityMode.POINT_IN_TIME, ObservationAvailabilityMode.CURRENT_SNAPSHOT],
)
def test_pipeline_as_of_remains_compatible_with_availability_modes(availability_mode):
    segment, definition, observations = _operating_fixture()
    result = OperatingForecastPipelineService().forecast(
        _fcff_financials(),
        evidence={
            "segments": (segment,),
            "definitions": (definition,),
            "observations": observations,
            "historical_revenue": {2024: Decimal("120")},
        },
        parameters=FcffForecastParameters(forecast_years=2),
        as_of=datetime.date(2026, 1, 1),
        availability_mode=availability_mode,
    )

    assert result.quality is not None
    assert result.quality.accepted


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
    assert result.quality is not None
    assert result.quality.accepted
    assert result.quality.driver_coverage == Decimal("1")


def test_pipeline_rejects_operating_evidence_below_activation_quality_gate():
    segment, definition, observations = _operating_fixture()

    with pytest.raises(OperatingForecastQualityError) as caught:
        OperatingForecastPipelineService().forecast(
            _fcff_financials(),
            evidence={
                "segments": (segment,),
                "definitions": (definition,),
                "observations": observations,
            },
            parameters=FcffForecastParameters(forecast_years=2),
        )

    rejection = caught.value.result
    assert not rejection.accepted
    assert rejection.driver_coverage == Decimal("0.5")
    assert "driver coverage=0.5" in rejection.reason


def test_cli_operating_provider_is_invoked_with_company_and_forecast_context():
    calls = {}

    class _Provider:
        async def discover(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(warnings=("discovery warning",))

    forecast = FcffForecastService().forecast(
        _fcff_financials(),
        FcffForecastParameters(forecast_years=2),
    )
    evidence, warnings = asyncio.run(
        cli_main._retrieve_operating_evidence(
            _fcff_financials(),
            forecast,
            datetime.date(2026, 1, 1),
            provider=_Provider(),
            args=SimpleNamespace(ticker="FIX", cik=123, refresh=False),
        )
    )

    assert evidence is not None
    assert calls["company_id"] == "fixture-company"
    assert calls["ticker"] == "FIX"
    assert calls["cik"] == 123
    assert calls["fiscal_years"] == (2025, 2026)
    assert warnings == ("discovery warning",)


def test_cli_missing_openai_configuration_returns_structured_operating_rejection(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cli_main, "OPENAI_API_KEY", None)
    args = SimpleNamespace(
        cache_dir=tmp_path,
        user_agent="Tests (tests@example.com)",
    )

    async def resolve():
        async with cli_main._operating_evidence_provider(
            args, _fcff_financials()
        ) as result:
            return result

    provider, rejection = asyncio.run(resolve())

    assert provider is None
    assert rejection == "OpenAI API key missing"


def test_cli_operating_provider_skips_sec_for_eu_market(monkeypatch, tmp_path):
    class _ShouldNotConstruct:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("SEC/OpenAI operating clients must not be constructed")

    monkeypatch.setattr(cli_main, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli_main, "OpenAIClient", _ShouldNotConstruct)
    monkeypatch.setattr(cli_main, "EdgarClient", _ShouldNotConstruct)
    args = SimpleNamespace(
        market=Market.EU.value,
        cache_dir=tmp_path,
        user_agent="Tests (tests@example.com)",
    )

    async def resolve():
        async with cli_main._operating_evidence_provider(
            args, _fcff_financials()
        ) as result:
            return result

    provider, rejection = asyncio.run(resolve())

    assert provider is None
    assert rejection == "SEC-backed operating evidence skipped for the eu market"


def test_cli_operating_provider_factory_closes_openai_and_sec_clients(
    monkeypatch, tmp_path
):
    state = {}

    class _OpenAI:
        def __init__(self, **_kwargs):
            self.closed = False
            state["openai"] = self

        async def close(self):
            self.closed = True

    class _Edgar:
        def __init__(self, cache, user_agent):
            state["edgar_args"] = (cache, user_agent)

        async def __aenter__(self):
            state["edgar"] = self
            return self

        async def __aexit__(self, *_args):
            state["edgar_closed"] = True

    monkeypatch.setattr(cli_main, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli_main, "OpenAIClient", _OpenAI)
    monkeypatch.setattr(cli_main, "EdgarClient", _Edgar)
    args = SimpleNamespace(
        cache_dir=tmp_path,
        user_agent="Tests (tests@example.com)",
    )

    async def resolve():
        async with cli_main._operating_evidence_provider(
            args, _fcff_financials()
        ) as result:
            assert isinstance(result[0], OperatingEvidenceDiscoveryService)
            assert result[1] is None
            return result

    asyncio.run(resolve())

    assert state["edgar_args"][1] == "Tests (tests@example.com)"
    assert state["edgar_closed"]
    assert state["openai"].closed
