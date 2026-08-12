import asyncio
import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from edgarito.schemas.operating import (
    ExtractedOperatingDriverDefinition,
    ExtractedOperatingEvidenceResponse,
    ExtractedOperatingInvestmentProgram,
    ExtractedOperatingObservation,
    ExtractedOperatingSegment,
    OperatingArchetype,
)
from edgarito.schemas.providers.edgar.filing import SecFiling, SecFilingDocument
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.operating.discovery import OperatingEvidenceDiscoveryService
from edgarito.services.operating.extraction import OperatingEvidenceExtractor


class _FakeOpenAI:
    model = "gpt-operating-test"
    reasoning_effort = "low"

    def __init__(self, response=None, error=None):
        self.response = response or ExtractedOperatingEvidenceResponse()
        self.error = error
        self.calls = 0
        self.contents = []

    async def extract_structured(self, **kwargs):
        self.calls += 1
        self.contents.append(kwargs["content"])
        if self.error is not None:
            raise self.error
        return self.response


def _filing() -> SecFiling:
    return SecFiling(
        cik=1,
        accession_number="0000000001-26-000001",
        form="10-K",
        filing_date=datetime.date(2026, 2, 1),
        primary_document="annual.htm",
    )


def _document(text: str) -> SecFilingDocument:
    return SecFilingDocument(
        filename="annual.htm",
        document_type="10-K",
        description="Annual report",
        content=text,
    )


def _evidence_response(text: str) -> ExtractedOperatingEvidenceResponse:
    return ExtractedOperatingEvidenceResponse(
        segments=[
            ExtractedOperatingSegment(
                segment_id="platform",
                name="Platform",
                scope="segment",
                supporting_text="Our Platform segment serves customers.",
                confidence="high",
            )
        ],
        definitions=[
            ExtractedOperatingDriverDefinition(
                driver_id="platform-volume-price",
                archetype="volume_price",
                segment_id="platform",
                input_metrics=("volume", "price"),
                required_inputs=("volume", "price"),
                units={"volume": "million units", "price": "USD/unit"},
                supporting_text="Our Platform revenue relationship is volume times price.",
                confidence="high",
            ),
            ExtractedOperatingDriverDefinition(
                driver_id="platform-subscribers-arpu",
                archetype="subscribers_arpu",
                segment_id="platform",
                input_metrics=("subscribers", "arpu"),
                required_inputs=("subscribers", "arpu"),
                units={"subscribers": "million users", "arpu": "USD/user"},
                supporting_text="Our Platform relationship uses subscribers and ARPU.",
                confidence="medium",
            ),
            ExtractedOperatingDriverDefinition(
                driver_id="platform-capacity",
                archetype="capacity_utilization_price",
                segment_id="platform",
                input_metrics=("capacity", "utilization", "price"),
                required_inputs=("capacity", "utilization", "price"),
                units={
                    "capacity": "units",
                    "utilization": "percent",
                    "price": "USD/unit",
                },
                supporting_text="Our Platform relationship uses capacity, utilization, and price.",
                confidence="medium",
            ),
        ],
        observations=[
            ExtractedOperatingObservation(
                segment_id="platform",
                driver_id="volume",
                fiscal_year=2026,
                value=12.5,
                unit="million units",
                supporting_text=text,
                confidence="high",
            )
        ],
        investment_programs=[
            ExtractedOperatingInvestmentProgram(
                program_id="facility-expansion",
                name="Facility expansion",
                segment_id="platform",
                fiscal_year=2027,
                value=25,
                unit="USD millions",
                currency="USD",
                status="planned",
                supporting_text="We plan a $25 million facility expansion in FY2027.",
                confidence="high",
            )
        ],
    )


def test_operating_extraction_maps_volume_subscriber_and_capacity_archetypes(tmp_path):
    text = (
        "Our Platform segment serves customers. "
        "Our Platform revenue relationship is volume times price. "
        "Our Platform relationship uses subscribers and ARPU. "
        "Our Platform relationship uses capacity, utilization, and price. "
        "We served 12.5 million units in FY2026. "
        "We plan a $25 million facility expansion in FY2027."
    )
    ai = _FakeOpenAI(_evidence_response("We served 12.5 million units in FY2026."))
    extractor = OperatingEvidenceExtractor(ai, FileSystemCache(tmp_path))

    entry, cache_hit = asyncio.run(
        extractor.extract(
            _filing(),
            _document(text),
            text,
            as_of=datetime.date(2026, 3, 1),
            fiscal_years=(2026, 2027),
        )
    )

    assert not cache_hit
    assert [item.archetype for item in entry.definitions] == [
        OperatingArchetype.VOLUME_PRICE,
        OperatingArchetype.SUBSCRIBERS_ARPU,
        OperatingArchetype.CAPACITY_UTILIZATION_PRICE,
    ]
    assert entry.observations[0].value == Decimal("12.5")
    assert entry.investment_programs[0].value == Decimal("25")
    assert entry.audit_records[-1].record_type == "investment_program"


def test_poor_disclosure_returns_empty_evidence_without_inventing_driver_values(tmp_path):
    filing = _filing()
    ai = _FakeOpenAI(ExtractedOperatingEvidenceResponse())
    service = OperatingEvidenceDiscoveryService(
        _Edgar(filing),
        OperatingEvidenceExtractor(ai, FileSystemCache(tmp_path)),
        max_filings=1,
        max_documents=1,
    )

    result = asyncio.run(
        service.discover(cik=1, as_of=datetime.date(2026, 3, 1), fiscal_years=(2026,))
    )

    assert not result.available
    assert result.segments == ()
    assert result.definitions == ()
    assert result.observations == ()
    assert result.investment_programs == ()
    assert result.documents_inspected == 1


def test_source_and_number_validation_rejects_unmatched_or_missing_values(tmp_path):
    text = "We served 12.5 million units in FY2026."
    response = ExtractedOperatingEvidenceResponse(
        observations=[
            ExtractedOperatingObservation(
                segment_id="platform",
                driver_id="volume",
                fiscal_year=2026,
                value=12.5,
                low=12,
                high=13.5,
                unit="million units",
                supporting_text=text,
            ),
            ExtractedOperatingObservation(
                segment_id="platform",
                driver_id="price",
                fiscal_year=2026,
                value=3,
                unit="USD/unit",
                supporting_text="This excerpt is not in the filing.",
            ),
        ]
    )
    entry, _ = asyncio.run(
        OperatingEvidenceExtractor(_FakeOpenAI(response), FileSystemCache(tmp_path)).extract(
            _filing(),
            _document(text),
            text,
            as_of=datetime.date(2026, 3, 1),
        )
    )

    assert entry.observations == ()
    assert len(entry.rejected) == 2
    assert any("numerical values" in item.reason for item in entry.rejected)
    assert any("not found" in item.reason for item in entry.rejected)
    assert entry.missing_evidence


def test_analyst_and_unsupported_revenue_forecast_claims_are_rejected(tmp_path):
    text = (
        "Analyst consensus expects FY2027 revenue of $100 million. "
        "We expect FY2027 revenue of $110 million."
    )
    response = ExtractedOperatingEvidenceResponse(
        observations=[
            ExtractedOperatingObservation(
                segment_id="company",
                driver_id="revenue",
                fiscal_year=2027,
                value=100,
                unit="USD millions",
                supporting_text="Analyst consensus expects FY2027 revenue of $100 million.",
            ),
            ExtractedOperatingObservation(
                segment_id="company",
                driver_id="revenue",
                fiscal_year=2027,
                value=110,
                unit="USD millions",
                supporting_text="We expect FY2027 revenue of $110 million.",
            ),
        ]
    )
    entry, _ = asyncio.run(
        OperatingEvidenceExtractor(_FakeOpenAI(response), FileSystemCache(tmp_path)).extract(
            _filing(),
            _document(text),
            text,
            as_of=datetime.date(2026, 3, 1),
        )
    )

    assert entry.observations == ()
    assert len(entry.unsupported_evidence) == 2
    assert all("revenue" in item.reason.casefold() or "expectations" in item.reason for item in entry.rejected)


def test_openai_response_forbids_unapproved_forecast_collection():
    with pytest.raises(ValidationError):
        ExtractedOperatingEvidenceResponse.model_validate(
            {"forecasts": [{"fiscal_year": 2027, "revenue": 100}]}
        )


def test_operating_extraction_cache_hit_is_deterministic(tmp_path):
    text = "We served 12.5 million units in FY2026."
    ai = _FakeOpenAI(
        ExtractedOperatingEvidenceResponse(
            observations=[
                ExtractedOperatingObservation(
                    segment_id="platform",
                    driver_id="volume",
                    fiscal_year=2026,
                    value=12.5,
                    unit="million units",
                    supporting_text=text,
                    confidence="high",
                )
            ]
        )
    )
    filing = _filing()
    document = _document(text)
    first, first_hit = asyncio.run(
        OperatingEvidenceExtractor(ai, FileSystemCache(tmp_path)).extract(
            filing, document, text, as_of=datetime.date(2026, 3, 1)
        )
    )
    second, second_hit = asyncio.run(
        OperatingEvidenceExtractor(ai, FileSystemCache(tmp_path)).extract(
            filing, document, text, as_of=datetime.date(2026, 3, 1)
        )
    )

    assert not first_hit
    assert second_hit
    assert first == second
    assert ai.calls == 1


def test_discovery_is_generic_for_ticker_labels_and_is_failure_isolated(tmp_path):
    filing = _filing()
    ai = _FakeOpenAI(error=RuntimeError("optional model unavailable"))
    edgar = _Edgar(filing)
    service = OperatingEvidenceDiscoveryService(
        edgar,
        OperatingEvidenceExtractor(ai, FileSystemCache(tmp_path)),
        max_filings=1,
        max_documents=1,
    )

    results = [
        asyncio.run(
            service.discover(
                ticker=ticker,
                cik=None,
                as_of=datetime.date(2026, 3, 1),
            )
        )
        for ticker in ("AAA", "TSLA")
    ]

    assert edgar.tickers == ["AAA", "TSLA"]
    assert results[0].segments == results[1].segments == ()
    assert all("extraction" in warning for result in results for warning in result.warnings)


class _Edgar:
    def __init__(self, filing):
        self.filing = filing
        self.tickers = []

    async def get_cik(self, ticker, **kwargs):
        self.tickers.append(ticker)
        return self.filing.cik

    async def get_guidance_filings(self, cik, **kwargs):
        return [self.filing]

    async def get_filing_documents(self, filing, **kwargs):
        return self.filing.model_copy(
            update={
                "documents": (
                    SecFilingDocument(
                        filename="annual.htm",
                        document_type="10-K",
                        description="Annual report",
                        content="The company operates in several markets.",
                    ),
                )
            }
        )
