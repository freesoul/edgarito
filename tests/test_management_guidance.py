import asyncio
import datetime
from decimal import Decimal
from types import SimpleNamespace

import edgarito.cli.__main__ as cli_module
from edgarito.cli.presentation.valuation_report import ValuationReportConsolePresenter
from edgarito.schemas.guidance.management import (
    ExtractedGuidanceItem,
    ExtractedGuidanceResponse,
    GuidanceBasis,
    GuidanceMetric,
    GuidanceOverlayResult,
    GuidancePeriodType,
    GuidanceScope,
    GuidanceStatus,
    GuidanceUnit,
    GuidanceValueKind,
    ManagementGuidance,
)
from edgarito.schemas.providers.edgar.filing import SecFiling, SecFilingDocument
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.forecasting.models import (
    FcffForecast,
    FcffForecastDriver,
    FcffForecastObservation,
    FcffForecastParameters,
    ForecastAssumptionSource,
)
from edgarito.services.guidance.extraction import ManagementGuidanceExtractor
from edgarito.services.guidance.overlay import GuidanceForecastOverlay
from edgarito.services.guidance.resolver import ManagementGuidanceResolver
from edgarito.services.guidance.service import (
    GuidanceDiscoveryResult,
    ManagementGuidanceService,
)


def _observation(year, revenue, growth="10", margin="20", tax="20", capex="5"):
    revenue = Decimal(revenue)
    return FcffForecastObservation(
        forecast_year=year - 2024,
        fiscal_year=year,
        period_end=datetime.date(year, 12, 31),
        revenue_growth=Decimal(growth),
        revenue=revenue,
        operating_margin=Decimal(margin),
        operating_income=revenue * Decimal(margin) / 100,
        tax_rate=Decimal(tax),
        nopat=revenue * Decimal(margin) / 100 * (1 - Decimal(tax) / 100),
        depreciation_to_revenue=Decimal("4"),
        depreciation_and_amortization=revenue * Decimal(".04"),
        capex_to_revenue=Decimal(capex),
        capital_expenditures=revenue * Decimal(capex) / 100,
        operating_working_capital_to_revenue=Decimal("10"),
        operating_working_capital=revenue * Decimal(".1"),
        change_in_operating_working_capital=Decimal("1"),
        fcff=Decimal("10"),
        unit="USD",
    )


def _baseline():
    return FcffForecast(
        provider="test",
        company_id="1",
        company_name="Test",
        ticker="TEST",
        base_fiscal_year=2024,
        base_period_end=datetime.date(2024, 12, 31),
        base_revenue=Decimal("105"),
        base_operating_income=Decimal("21"),
        base_tax_rate=Decimal("20"),
        base_nopat=Decimal("16.8"),
        base_depreciation_and_amortization=Decimal("4"),
        base_capital_expenditures=Decimal("5"),
        base_operating_working_capital=Decimal("10"),
        unit="USD",
        parameters=FcffForecastParameters(forecast_years=2),
        historical_fiscal_years=(2023, 2024),
        assumption_sources={
            driver: ForecastAssumptionSource.TRAILING_AVERAGE
            for driver in FcffForecastDriver
        },
        observations=[_observation(2025, "115.5"), _observation(2026, "127.05")],
    )


def _guidance(
    metric,
    *,
    metric_name=None,
    year=2025,
    low=None,
    high=None,
    point=None,
    kind=GuidanceValueKind.MONETARY,
    currency="USD",
    basis=GuidanceBasis.GAAP,
    status=GuidanceStatus.ISSUED,
    filed=datetime.date(2025, 2, 1),
):
    return ManagementGuidance(
        metric=metric,
        metric_name=metric_name,
        fiscal_year=year,
        period_type=GuidancePeriodType.FISCAL_YEAR,
        point=Decimal(point) if point is not None else None,
        low=Decimal(low) if low is not None else None,
        high=Decimal(high) if high is not None else None,
        value_kind=kind,
        currency=currency,
        unit=currency or kind.value,
        basis=basis,
        scope=GuidanceScope.CONSOLIDATED,
        status=status,
        filing_date=filed,
        accession_number=f"accession-{filed}",
        filing_form="8-K",
        source_document="ex991.htm",
        source_document_type="EX-99.1",
        supporting_text="We expect this range.",
        evidence_verified=True,
        extraction_model="gpt-test",
    )


def test_revenue_guidance_midpoint_becomes_absolute_anchor():
    revenue = _guidance(GuidanceMetric.REVENUE, low="120", high="130")
    gross_margin = _guidance(
        GuidanceMetric.GROSS_MARGIN,
        low="50",
        high="52",
        kind=GuidanceValueKind.PERCENTAGE,
        currency=None,
    )

    parameters, result = GuidanceForecastOverlay().apply(
        [revenue, gross_margin],
        baseline=_baseline(),
        parameters=FcffForecastParameters(forecast_years=2),
    )

    assert parameters.revenue_anchors == {2025: Decimal("125")}
    assert parameters.revenue_anchor_sources == {
        2025: ForecastAssumptionSource.MANAGEMENT_GUIDANCE
    }
    assert parameters.assumption_source_overrides == {
        FcffForecastDriver.REVENUE_GROWTH: ForecastAssumptionSource.MANAGEMENT_GUIDANCE
    }
    assert result.applications[0].methodology == (
        "management guidance midpoint revenue anchor"
    )
    assert gross_margin in result.evidence_only


def test_named_revenue_component_cannot_replace_total_company_anchor():
    total_revenue = _guidance(
        GuidanceMetric.REVENUE,
        low="51000000000",
        high="51400000000",
    )
    ads_revenue = _guidance(
        GuidanceMetric.REVENUE,
        metric_name="ads revenue",
        point="3000000000",
    )

    parameters, result = GuidanceForecastOverlay().apply(
        [total_revenue, ads_revenue],
        baseline=_baseline(),
        parameters=FcffForecastParameters(forecast_years=2),
    )

    assert parameters.revenue_anchors == {2025: Decimal("51200000000")}
    assert [item.guidance for item in result.applications] == [total_revenue]
    assert ads_revenue in result.evidence_only
    assert any(
        "named revenue component 'ads revenue'" in reason
        for reason in result.rejected_reasons
    )


def test_reported_total_revenue_wins_over_unknown_basis_regardless_of_order():
    total_revenue = _guidance(
        GuidanceMetric.REVENUE,
        low="51000000000",
        high="51400000000",
        basis=GuidanceBasis.REPORTED,
    )
    misleading_revenue = _guidance(
        GuidanceMetric.REVENUE,
        point="3000000000",
        basis=GuidanceBasis.UNKNOWN,
    )

    for records in (
        [total_revenue, misleading_revenue],
        [misleading_revenue, total_revenue],
    ):
        parameters, result = GuidanceForecastOverlay().apply(
            records,
            baseline=_baseline(),
            parameters=FcffForecastParameters(forecast_years=2),
        )

        assert parameters.revenue_anchors == {2025: Decimal("51200000000")}
        assert parameters.revenue_anchor_sources == {
            2025: ForecastAssumptionSource.MANAGEMENT_GUIDANCE
        }
        assert [item.guidance for item in result.applications] == [total_revenue]
        assert misleading_revenue in result.evidence_only
        assert any(
            "lower-priority guidance" in reason for reason in result.rejected_reasons
        )


def test_capex_uses_guided_revenue_but_gross_margin_never_maps_to_operating_margin():
    revenue = _guidance(GuidanceMetric.REVENUE, low="120", high="130")
    capex = _guidance(GuidanceMetric.CAPEX, point="15")
    gross = _guidance(
        GuidanceMetric.GROSS_MARGIN,
        point="55",
        kind=GuidanceValueKind.PERCENTAGE,
        currency=None,
    )

    parameters, result = GuidanceForecastOverlay().apply(
        [revenue, capex, gross],
        baseline=_baseline(),
        parameters=FcffForecastParameters(forecast_years=2),
    )

    assert parameters.capex_to_revenue[0] == Decimal("12")
    assert parameters.operating_margin is None
    assert gross in result.evidence_only


def test_explicit_cli_or_profile_driver_has_precedence_over_guidance():
    requested = FcffForecastParameters(
        forecast_years=2,
        revenue_growth=Decimal("7"),
        operating_margin=Decimal("23"),
    )
    revenue = _guidance(GuidanceMetric.REVENUE, low="120", high="130")
    margin = _guidance(
        GuidanceMetric.OPERATING_MARGIN,
        low="20",
        high="22",
        kind=GuidanceValueKind.PERCENTAGE,
        currency=None,
    )

    parameters, result = GuidanceForecastOverlay().apply(
        [revenue, margin], baseline=_baseline(), parameters=requested
    )

    assert parameters == requested
    assert result.applications == ()
    assert any("explicit CLI/profile" in reason for reason in result.rejected_reasons)


def test_current_resolver_uses_latest_filing_and_honors_withdrawal_and_lookahead():
    old = _guidance(
        GuidanceMetric.REVENUE,
        low="100",
        high="110",
        filed=datetime.date(2025, 1, 1),
    )
    new = _guidance(
        GuidanceMetric.REVENUE,
        low="120",
        high="130",
        filed=datetime.date(2025, 2, 1),
    )
    future = _guidance(
        GuidanceMetric.REVENUE,
        low="140",
        high="150",
        filed=datetime.date(2025, 4, 1),
    )

    resolved = ManagementGuidanceResolver().resolve(
        [old, new, future], as_of=datetime.date(2025, 3, 1)
    )
    assert resolved.records == (new,)

    withdrawn = new.model_copy(update={"status": GuidanceStatus.WITHDRAWN})
    resolved = ManagementGuidanceResolver().resolve(
        [old, withdrawn], as_of=datetime.date(2025, 3, 1)
    )
    assert resolved.records == ()


class _Ai:
    model = "gpt-test"
    reasoning_effort = "low"

    def __init__(self, response=None):
        self.calls = 0
        self.contents = []
        self.response = response

    async def extract_structured(self, **kwargs):
        self.calls += 1
        self.contents.append(kwargs["content"])
        return self.response or ExtractedGuidanceResponse(
            guidance=[
                ExtractedGuidanceItem(
                    metric=GuidanceMetric.REVENUE,
                    fiscal_year=2026,
                    period_type=GuidancePeriodType.FISCAL_YEAR,
                    low=Decimal("10"),
                    high=Decimal("11"),
                    value_kind=GuidanceValueKind.MONETARY,
                    currency="USD",
                    unit=GuidanceUnit.BILLIONS,
                    scope=GuidanceScope.CONSOLIDATED,
                    supporting_text="We expect FY2026 revenue of $10-$11 billion.",
                )
            ]
        )


class _Edgar:
    def __init__(self, filing):
        self.filing = filing
        self.refresh_values = []

    async def get_cik(self, ticker, use_cache=True, make_cache=True):
        return 1

    async def get_guidance_filings(self, cik, **kwargs):
        self.refresh_values.append(kwargs["use_cache"])
        return [self.filing.model_copy(update={"documents": ()})]

    async def get_filing_documents(self, filing, **kwargs):
        return self.filing


def test_refresh_with_unchanged_sec_content_reuses_normalized_extraction(tmp_path):
    document = SecFilingDocument(
        filename="ex991.htm",
        document_type="EX-99.1",
        description="Financial results",
        content="We expect FY2026 revenue of $10-$11 billion.",
    )
    filing = SecFiling(
        cik=1,
        accession_number="0000000001-26-000001",
        form="8-K",
        filing_date=datetime.date(2026, 1, 2),
        items=("2.02",),
        primary_document="primary.htm",
        documents=(document,),
    )
    ai = _Ai()
    edgar = _Edgar(filing)
    service = ManagementGuidanceService(
        edgar, ManagementGuidanceExtractor(ai, FileSystemCache(tmp_path))
    )

    first = asyncio.run(
        service.retrieve(ticker="TEST", cik=None, as_of=datetime.date(2026, 2, 1))
    )
    second = asyncio.run(
        service.retrieve(
            ticker="TEST",
            cik=None,
            as_of=datetime.date(2026, 2, 1),
            refresh_sec=True,
        )
    )

    assert ai.calls == 1
    assert first.cache_misses == 1
    assert second.cache_hits == 1
    assert edgar.refresh_values == [True, False]
    assert first.filings_inspected == second.filings_inspected == 1
    assert first.documents_inspected == second.documents_inspected == 1
    assert first.extracted_guidance_records == second.extracted_guidance_records == 1
    assert first.rejected_records == second.rejected_records == 0


def test_service_sends_bounded_periodic_filing_context_but_validates_full_text(
    tmp_path,
):
    phrase = "We currently expect capital expenditures to exceed $25 billion in 2026."
    response = ExtractedGuidanceResponse(
        guidance=[
            ExtractedGuidanceItem(
                metric=GuidanceMetric.CAPEX,
                fiscal_year=2026,
                period_type=GuidancePeriodType.FISCAL_YEAR,
                point=25,
                value_kind=GuidanceValueKind.MONETARY,
                currency="USD",
                unit=GuidanceUnit.BILLIONS,
                scope=GuidanceScope.CONSOLIDATED,
                supporting_text=phrase,
            )
        ]
    )
    document = SecFilingDocument(
        filename="primary.htm",
        document_type="10-Q",
        description="Quarterly report",
        content=("Historical discussion without current information. " * 10_000)
        + phrase,
    )
    filing = SecFiling(
        cik=1,
        accession_number="0000000001-26-000002",
        form="10-Q",
        filing_date=datetime.date(2026, 7, 15),
        primary_document="primary.htm",
        documents=(document,),
    )
    ai = _Ai(response)
    service = ManagementGuidanceService(
        _Edgar(filing),
        ManagementGuidanceExtractor(ai, FileSystemCache(tmp_path)),
        max_filings=1,
        max_documents=1,
    )

    result = asyncio.run(
        service.retrieve(ticker="TEST", cik=1, as_of=datetime.date(2026, 8, 1))
    )

    assert ai.calls == 1
    assert len(ai.contents[0]) < len(document.content)
    assert phrase in ai.contents[0]
    assert [item.metric for item in result.records] == [GuidanceMetric.CAPEX]
    assert result.filings_inspected == 1
    assert result.documents_inspected == 1
    assert result.extracted_guidance_records == 1
    assert result.rejected_records == 0


def test_service_counts_rejected_extraction_records(tmp_path):
    accepted_text = "We expect FY2026 revenue of $10-$11 billion."
    response = ExtractedGuidanceResponse(
        guidance=[
            ExtractedGuidanceItem(
                metric=GuidanceMetric.REVENUE,
                fiscal_year=2026,
                period_type=GuidancePeriodType.FISCAL_YEAR,
                low=10,
                high=11,
                value_kind=GuidanceValueKind.MONETARY,
                currency="USD",
                unit=GuidanceUnit.BILLIONS,
                scope=GuidanceScope.CONSOLIDATED,
                supporting_text=accepted_text,
            ),
            ExtractedGuidanceItem(
                metric=GuidanceMetric.CAPEX,
                fiscal_year=2026,
                period_type=GuidancePeriodType.FISCAL_YEAR,
                point=25,
                value_kind=GuidanceValueKind.MONETARY,
                currency="USD",
                unit=GuidanceUnit.BILLIONS,
                scope=GuidanceScope.CONSOLIDATED,
                supporting_text="This evidence is not in the filing.",
            ),
        ]
    )
    document = SecFilingDocument(
        filename="primary.htm",
        document_type="10-Q",
        description="Quarterly report",
        content=accepted_text,
    )
    filing = SecFiling(
        cik=1,
        accession_number="0000000001-26-000003",
        form="10-Q",
        filing_date=datetime.date(2026, 7, 15),
        primary_document="primary.htm",
        documents=(document,),
    )
    result = asyncio.run(
        ManagementGuidanceService(
            _Edgar(filing),
            ManagementGuidanceExtractor(_Ai(response), FileSystemCache(tmp_path)),
            max_filings=1,
            max_documents=1,
        ).retrieve(ticker="TEST", cik=1, as_of=datetime.date(2026, 8, 1))
    )

    assert len(result.records) == 1
    assert len(result.rejected) == 1
    assert result.extracted_guidance_records == 1
    assert result.rejected_records == 1


def test_guidance_counters_transfer_from_discovery_to_overlay(monkeypatch, tmp_path):
    discovery = GuidanceDiscoveryResult(
        filings_inspected=3,
        documents_inspected=4,
        extracted_guidance_records=5,
        rejected_records=2,
    )

    class _OpenAI:
        def __init__(self, **kwargs):
            pass

        async def close(self):
            pass

    class _Edgar:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class _Service:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, **kwargs):
            return discovery

    monkeypatch.setattr(cli_module, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli_module, "OpenAIClient", _OpenAI)
    monkeypatch.setattr(cli_module, "EdgarClient", _Edgar)
    monkeypatch.setattr(cli_module, "ManagementGuidanceService", _Service)

    parameters, overlay = asyncio.run(
        cli_module._management_guidance_overlay(
            SimpleNamespace(
                user_agent="test@example.com",
                cache_dir=tmp_path,
                ticker="TEST",
                cik=1,
                refresh=False,
            ),
            SimpleNamespace(ticker="TEST"),
            FcffForecastParameters(forecast_years=2),
            _baseline(),
            datetime.date(2026, 8, 1),
        )
    )

    assert parameters.forecast_years == 2
    assert overlay.filings_inspected == 3
    assert overlay.documents_inspected == 4
    assert overlay.extracted_guidance_records == 5
    assert overlay.rejected_records == 2


def test_valuation_report_renders_audit_counters_without_source_records():
    result = GuidanceOverlayResult(
        rejected_reasons=("Extraction rejected: malformed guidance",),
        filings_inspected=2,
        documents_inspected=3,
        extracted_guidance_records=1,
        rejected_records=4,
        cache_hits=1,
        cache_misses=2,
    )

    rendered = ValuationReportConsolePresenter().render(
        intrinsic=None,
        peer_report=None,
        relative=None,
        provider_relative=None,
        decision=None,
        profile_name=None,
        show_scenarios=False,
        show_sensitivity=False,
        show_reverse_dcf=False,
        verbose=True,
        management_guidance=result,
    )

    assert "MANAGEMENT GUIDANCE" in rendered
    assert "Filings inspected: 2" in rendered
    assert "Documents inspected: 3" in rendered
    assert "Extracted guidance records: 1" in rendered
    assert "Applied: 0" in rendered
    assert "Evidence only: 0" in rendered
    assert "Rejected: 4" in rendered
    assert "Extraction rejected: malformed guidance" in rendered
    assert "Source:" not in rendered
    assert "Extraction: OpenAI /" not in rendered


def test_valuation_report_renders_evidence_only_guidance_without_applications():
    record = _guidance(GuidanceMetric.REVENUE, low="120", high="130")
    result = GuidanceOverlayResult(
        evidence_only=(record,),
        documents_inspected=1,
        extracted_guidance_records=1,
    )

    rendered = ValuationReportConsolePresenter().render(
        intrinsic=None,
        peer_report=None,
        relative=None,
        provider_relative=None,
        decision=None,
        profile_name=None,
        show_scenarios=False,
        show_sensitivity=False,
        show_reverse_dcf=False,
        verbose=False,
        management_guidance=result,
    )

    assert "MANAGEMENT GUIDANCE" in rendered
    assert "Evidence only: 1" in rendered
    assert "FY2025 revenue" in rendered
