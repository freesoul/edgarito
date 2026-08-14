import asyncio
import datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

import edgarito.cli.__main__ as cli_module
from edgarito.cli.presentation.valuation_report import ValuationReportConsolePresenter
from edgarito.enums.market import Market
from edgarito.schemas.guidance.management import (
    ExtractedGuidanceItem,
    ExtractedGuidanceResponse,
    GuidanceBasis,
    GuidanceDocumentAudit,
    GuidanceMetric,
    GuidanceOverlayResult,
    GuidancePeriodType,
    GuidanceQualifier,
    GuidanceScope,
    GuidanceStatus,
    GuidanceUnit,
    GuidanceValueKind,
    ManagementGuidance,
    MonetaryForecastConstraint,
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
    period_type=GuidancePeriodType.FISCAL_YEAR,
    fiscal_quarter=None,
    qualifier=GuidanceQualifier.UNKNOWN,
    basis=GuidanceBasis.GAAP,
    status=GuidanceStatus.ISSUED,
    filed=datetime.date(2025, 2, 1),
):
    return ManagementGuidance(
        metric=metric,
        metric_name=metric_name,
        fiscal_year=year,
        fiscal_quarter=fiscal_quarter,
        period_type=period_type,
        point=Decimal(point) if point is not None else None,
        low=Decimal(low) if low is not None else None,
        high=Decimal(high) if high is not None else None,
        value_kind=kind,
        currency=currency,
        unit=currency or kind.value,
        qualifier=qualifier,
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


def test_applied_revenue_growth_guidance_remains_quantitative_forward_evidence():
    growth = _guidance(
        GuidanceMetric.REVENUE_GROWTH,
        point="12",
        kind=GuidanceValueKind.PERCENTAGE,
        currency=None,
    )

    _parameters, result = GuidanceForecastOverlay().apply(
        [growth],
        baseline=_baseline(),
        parameters=FcffForecastParameters(forecast_years=2),
    )

    evidence = cli_module._forward_growth_evidence(
        "growth", set(), result, (2025, 2026)
    )

    assert evidence.growth_path == (Decimal("12"),)
    assert evidence.growth_anchor == Decimal("12")
    assert evidence.confidence == "high"


def test_later_year_growth_guidance_does_not_shift_into_first_forecast_year():
    growth = _guidance(
        GuidanceMetric.REVENUE_GROWTH,
        year=2026,
        point="12",
        kind=GuidanceValueKind.PERCENTAGE,
        currency=None,
    )

    _parameters, result = GuidanceForecastOverlay().apply(
        [growth],
        baseline=_baseline(),
        parameters=FcffForecastParameters(forecast_years=2),
    )
    evidence = cli_module._forward_growth_evidence(
        "growth", set(), result, (2025, 2026)
    )

    assert [application.guidance for application in result.applications] == [growth]
    assert evidence.growth_path == ()
    assert evidence.guidance


@pytest.mark.parametrize(
    "guidance",
    [
        _guidance(
            GuidanceMetric.REVENUE_GROWTH,
            period_type=GuidancePeriodType.QUARTER,
            fiscal_quarter=2,
            point="12",
            kind=GuidanceValueKind.PERCENTAGE,
            currency=None,
        ),
        _guidance(
            GuidanceMetric.REVENUE_GROWTH,
            year=2027,
            point="13",
            kind=GuidanceValueKind.PERCENTAGE,
            currency=None,
        ),
        _guidance(
            GuidanceMetric.REVENUE_GROWTH,
            year=None,
            point="14",
            kind=GuidanceValueKind.PERCENTAGE,
            currency=None,
        ),
        _guidance(
            GuidanceMetric.REVENUE_GROWTH,
            year=2030,
            period_type=GuidancePeriodType.LONG_TERM_TARGET,
            point="15",
            kind=GuidanceValueKind.PERCENTAGE,
            currency=None,
        ),
    ],
)
def test_ineligible_growth_guidance_remains_qualitative_not_quantitative(guidance):
    _parameters, result = GuidanceForecastOverlay().apply(
        [guidance],
        baseline=_baseline(),
        parameters=FcffForecastParameters(forecast_years=2),
    )
    evidence = cli_module._forward_growth_evidence(
        "growth", set(), result, (2025, 2026)
    )

    assert result.applications == ()
    assert guidance in result.evidence_only
    assert evidence.growth_path == ()
    assert evidence.guidance


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


def test_absolute_capex_guidance_maps_without_revenue_guidance():
    capex = _guidance(
        GuidanceMetric.CAPEX,
        point="15",
        qualifier=GuidanceQualifier.POINT,
    )
    gross = _guidance(
        GuidanceMetric.GROSS_MARGIN,
        point="55",
        kind=GuidanceValueKind.PERCENTAGE,
        currency=None,
    )

    parameters, result = GuidanceForecastOverlay().apply(
        [capex, gross],
        baseline=_baseline(),
        parameters=FcffForecastParameters(forecast_years=2),
    )

    assert parameters.capex_to_revenue is None
    assert parameters.capex_constraints == {
        2025: MonetaryForecastConstraint(
            point=Decimal("15"), source="management_guidance"
        )
    }
    capex_application = next(
        application
        for application in result.applications
        if application.guidance is capex
    )
    assert capex_application.methodology == (
        "management guidance point capex constraint"
    )
    assert capex_application.source == "management_guidance"
    assert parameters.operating_margin is None
    assert gross in result.evidence_only


def test_capex_constraint_qualifiers_preserve_points_and_bounds():
    records = [
        _guidance(GuidanceMetric.CAPEX, point="15", qualifier=GuidanceQualifier.POINT),
        _guidance(
            GuidanceMetric.CAPEX,
            year=2026,
            point="12",
            qualifier=GuidanceQualifier.AT_LEAST,
        ),
        _guidance(
            GuidanceMetric.CAPEX,
            year=2027,
            point="8",
            qualifier=GuidanceQualifier.AT_MOST,
        ),
        _guidance(
            GuidanceMetric.CAPEX,
            year=2028,
            low="10",
            high="20",
            qualifier=GuidanceQualifier.RANGE,
        ),
    ]
    baseline = _baseline().model_copy(
        update={
            "observations": [_observation(year, "115") for year in range(2025, 2029)]
        }
    )
    parameters, result = GuidanceForecastOverlay().apply(
        records,
        baseline=baseline,
        parameters=FcffForecastParameters(forecast_years=4),
    )

    assert parameters.capex_constraints == {
        2025: MonetaryForecastConstraint(point=Decimal("15")),
        2026: MonetaryForecastConstraint(minimum=Decimal("12")),
        2027: MonetaryForecastConstraint(maximum=Decimal("8")),
        2028: MonetaryForecastConstraint(
            point=Decimal("15"), minimum=Decimal("10"), maximum=Decimal("20")
        ),
    }
    assert [application.methodology for application in result.applications] == [
        "management guidance point capex constraint",
        "management guidance floor capex constraint",
        "management guidance ceiling capex constraint",
        "management guidance range capex constraint",
    ]


def test_capex_more_than_guidance_is_a_lower_bound_not_a_point():
    capex = _guidance(
        GuidanceMetric.CAPEX,
        point="25",
        qualifier=GuidanceQualifier.MORE_THAN,
    )

    parameters, result = GuidanceForecastOverlay().apply(
        [capex],
        baseline=_baseline(),
        parameters=FcffForecastParameters(forecast_years=2),
    )

    assert parameters.capex_constraints == {
        2025: MonetaryForecastConstraint(minimum=Decimal("25"))
    }
    assert result.applications[0].methodology == (
        "management guidance floor capex constraint"
    )


def test_absolute_capex_guidance_does_not_cross_currency_boundary():
    capex = _guidance(GuidanceMetric.CAPEX, point="15", currency="EUR")

    parameters, result = GuidanceForecastOverlay().apply(
        [capex],
        baseline=_baseline(),
        parameters=FcffForecastParameters(forecast_years=2),
    )

    assert parameters.capex_constraints == {}
    assert result.applications == ()
    assert capex in result.evidence_only
    assert any(
        "does not match forecast unit" in reason for reason in result.rejected_reasons
    )


def test_explicit_capex_ratio_path_wins_and_records_guidance_evidence():
    requested = FcffForecastParameters(forecast_years=2, capex_to_revenue=Decimal("6"))
    capex = _guidance(GuidanceMetric.CAPEX, point="15")

    parameters, result = GuidanceForecastOverlay().apply(
        [capex], baseline=_baseline(), parameters=requested
    )

    assert parameters == requested
    assert result.applications == ()
    assert capex in result.evidence_only
    assert any(
        "explicit CLI/profile capex_to_revenue wins" in reason
        for reason in result.rejected_reasons
    )


def test_monetary_forecast_constraint_validates_bounds_and_point():
    with pytest.raises(ValueError, match="minimum cannot exceed maximum"):
        MonetaryForecastConstraint(minimum=Decimal("10"), maximum=Decimal("9"))
    with pytest.raises(ValueError, match="point cannot exceed maximum"):
        MonetaryForecastConstraint(point=Decimal("11"), maximum=Decimal("10"))
    with pytest.raises(ValueError, match="fiscal years are invalid"):
        FcffForecastParameters(
            forecast_years=1,
            capex_constraints={1800: MonetaryForecastConstraint(point=Decimal("1"))},
        )


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


class _MultiFilingEdgar:
    def __init__(self, filings):
        self.filings = tuple(filings)
        self.fetched_accessions = []

    async def get_guidance_filings(self, cik, **kwargs):
        return [filing.model_copy(update={"documents": ()}) for filing in self.filings]

    async def get_filing_documents(self, filing, **kwargs):
        self.fetched_accessions.append(filing.accession_number)
        return next(
            item
            for item in self.filings
            if item.accession_number == filing.accession_number
        )


def test_periodic_primary_is_processed_before_current_documents_consume_budget(
    tmp_path,
):
    current = SecFiling(
        cik=1,
        accession_number="0000000001-26-000010",
        form="8-K",
        filing_date=datetime.date(2026, 7, 14),
        items=("2.02",),
        primary_document="current.htm",
        documents=(
            SecFilingDocument(
                filename="current.htm",
                document_type="8-K",
                content="We expect FY2026 revenue of $10-$11 billion.",
            ),
        ),
    )
    periodic = SecFiling(
        cik=1,
        accession_number="0000000001-26-000011",
        form="10-Q",
        filing_date=datetime.date(2026, 7, 15),
        primary_document="primary.htm",
        documents=(
            SecFilingDocument(
                filename="primary.htm",
                document_type="10-Q",
                content="We expect FY2026 revenue of $10-$11 billion.",
            ),
        ),
    )
    result = asyncio.run(
        ManagementGuidanceService(
            _MultiFilingEdgar([current, periodic]),
            ManagementGuidanceExtractor(_Ai(), FileSystemCache(tmp_path)),
            max_filings=2,
            max_documents=1,
        ).retrieve(ticker="TEST", cik=1, as_of=datetime.date(2026, 8, 1))
    )

    assert [item.filing_form for item in result.records] == ["10-Q"]
    assert result.documents_inspected == 1
    assert result.document_audits[0].filing_form == "10-Q"
    assert result.document_audits[0].filename == "primary.htm"
    assert result.document_audits[0].is_primary

def _budget_filing(form, accession, filing_date):
    primary_document = f"{accession}-primary.htm"
    documents = tuple(
        SecFilingDocument(
            filename=(
                primary_document if index == 0 else f"{accession}-ex99-{index}.htm"
            ),
            document_type=form if index == 0 else "EX-99.1",
            description=(
                "Quarterly report" if index == 0 else "Financial results press release"
            ),
            content="We expect FY2026 revenue of $10-$11 billion.",
        )
        for index in range(3)
    )
    return SecFiling(
        cik=1,
        accession_number=accession,
        form=form,
        filing_date=filing_date,
        items=("2.02",) if form == "8-K" else (),
        primary_document=primary_document,
        primary_document_description=(
            "Current financial results" if form == "8-K" else "Quarterly report"
        ),
        documents=documents,
    )


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


def test_service_reserves_document_capacity_for_selected_periodic_filing(tmp_path):
    current_filings = [
        _budget_filing(
            "8-K",
            f"0000000001-26-00000{index}",
            datetime.date(2026, 7, 10 + index),
        )
        for index in (1, 2)
    ]
    periodic = _budget_filing(
        "10-Q", "0000000001-26-000003", datetime.date(2026, 7, 13)
    )
    edgar = _MultiFilingEdgar([*current_filings, periodic])
    ai = _Ai()

    result = asyncio.run(
        ManagementGuidanceService(
            edgar,
            ManagementGuidanceExtractor(ai, FileSystemCache(tmp_path)),
            max_filings=3,
            max_documents_per_filing=3,
            max_documents=6,
        ).retrieve(ticker="TEST", cik=1, as_of=datetime.date(2026, 8, 1))
    )

    assert periodic.accession_number in edgar.fetched_accessions
    assert len(edgar.fetched_accessions) == 3
    assert ai.calls == 6
    assert result.filings_inspected == 3
    assert result.documents_inspected == 6


def test_service_does_not_count_filings_skipped_by_document_budget(tmp_path):
    filings = [
        _budget_filing(
            "8-K",
            f"0000000001-26-00000{index}",
            datetime.date(2026, 7, 10 + index),
        )
        for index in (1, 2)
    ]
    edgar = _MultiFilingEdgar(filings)

    result = asyncio.run(
        ManagementGuidanceService(
            edgar,
            ManagementGuidanceExtractor(_Ai(), FileSystemCache(tmp_path)),
            max_filings=2,
            max_documents_per_filing=3,
            max_documents=1,
        ).retrieve(ticker="TEST", cik=1, as_of=datetime.date(2026, 8, 1))
    )

    assert len(edgar.fetched_accessions) == 1
    assert result.filings_inspected == 1
    assert result.documents_inspected == 1


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
    assert result.records[0].is_primary
    assert result.filings_inspected == 1
    assert result.documents_inspected == 1
    assert result.extracted_guidance_records == 1
    assert result.rejected_records == 0
    assert len(result.document_audits) == 1
    audit = result.document_audits[0]
    assert audit.filing_form == "10-Q"
    assert audit.accession_number == filing.accession_number
    assert audit.filename == "primary.htm"
    assert audit.is_primary
    assert audit.cleaned_size == len(document.content)
    assert audit.bounded_context_size == len(ai.contents[0])
    assert audit.keyword_hits == {
        "expect": 1,
        "capex": 0,
        "capital expenditures": 1,
        "revenue": 0,
        "margin": 0,
    }
    assert audit.accepted_records == 1
    assert audit.rejected_records == 0


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
    audit = GuidanceDocumentAudit(
        filing_form="10-Q",
        filing_date=datetime.date(2026, 7, 15),
        accession_number="accession",
        filename="primary.htm",
        document_type="10-Q",
        is_primary=True,
        cleaned_size=100,
        bounded_context_size=80,
        keyword_hits={
            "expect": 1,
            "capex": 1,
            "capital expenditures": 1,
            "revenue": 2,
            "margin": 1,
        },
        accepted_records=1,
    )
    discovery = GuidanceDiscoveryResult(
        filings_inspected=3,
        documents_inspected=4,
        extracted_guidance_records=5,
        rejected_records=2,
        document_audits=(audit,),
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
    assert overlay.document_audits == (audit,)


def test_management_guidance_skips_sec_for_eu_market(monkeypatch, tmp_path):
    class _ShouldNotConstruct:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("SEC/OpenAI guidance clients must not be constructed")

    monkeypatch.setattr(cli_module, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli_module, "OpenAIClient", _ShouldNotConstruct)
    monkeypatch.setattr(cli_module, "EdgarClient", _ShouldNotConstruct)
    parameters = FcffForecastParameters(forecast_years=2)

    returned_parameters, overlay = asyncio.run(
        cli_module._management_guidance_overlay(
            SimpleNamespace(
                market=Market.EU.value,
                user_agent="test@example.com",
                cache_dir=tmp_path,
                ticker="TEST",
                cik=1,
                refresh=False,
            ),
            SimpleNamespace(ticker="TEST"),
            parameters,
            _baseline(),
            datetime.date(2026, 8, 1),
        )
    )

    assert returned_parameters is parameters
    assert overlay.applications == ()
    assert overlay.warnings == (
        "SEC/EDGAR management guidance skipped for the eu market",
    )


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


def test_valuation_report_renders_document_audit_without_document_contents():
    result = GuidanceOverlayResult(
        document_audits=(
            GuidanceDocumentAudit(
                filing_form="10-Q",
                filing_date=datetime.date(2026, 7, 15),
                accession_number="0000000001-26-000002",
                filename="primary.htm",
                document_type="10-Q",
                is_primary=True,
                cleaned_size=50_000,
                bounded_context_size=24_000,
                keyword_hits={
                    "expect": 2,
                    "capex": 1,
                    "capital expenditures": 1,
                    "revenue": 3,
                    "margin": 4,
                },
                accepted_records=1,
                rejected_records=2,
            ),
        ),
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

    assert "SEC document audit (contents omitted):" in rendered
    assert "form=10-Q" in rendered
    assert "accession=0000000001-26-000002" in rendered
    assert "filename=primary.htm type=10-Q" in rendered
    assert "primary=yes" in rendered
    assert "cleaned=50000 chars context=24000 chars" in rendered
    assert "expect=2 capex=1 capital expenditures=1 revenue=3 margin=4" in rendered
    assert "accepted=1 rejected=2" in rendered
    assert "We currently expect" not in rendered


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
