import asyncio
import datetime
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError
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
from edgarito.services.guidance.documents import extract_operating_context
from edgarito.services.openai import OpenAIClient, OpenAIExtractionError
from edgarito.services.operating._discovery.service import (
    OperatingEvidenceDiscoveryService,
)
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


def test_poor_disclosure_returns_empty_evidence_without_inventing_driver_values(
    tmp_path,
):
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
        OperatingEvidenceExtractor(
            _FakeOpenAI(response), FileSystemCache(tmp_path)
        ).extract(
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
        OperatingEvidenceExtractor(
            _FakeOpenAI(response), FileSystemCache(tmp_path)
        ).extract(
            _filing(),
            _document(text),
            text,
            as_of=datetime.date(2026, 3, 1),
        )
    )

    assert entry.observations == ()
    assert len(entry.unsupported_evidence) == 2
    assert all(
        "revenue" in item.reason.casefold() or "expectations" in item.reason
        for item in entry.rejected
    )


def test_openai_response_forbids_unapproved_forecast_collection():
    with pytest.raises(ValidationError):
        ExtractedOperatingEvidenceResponse.model_validate(
            {"forecasts": [{"fiscal_year": 2027, "revenue": 100}]}
        )


def test_openai_client_accepts_nested_parsed_response_shape_without_forecasts():
    payload = {
        "operating_segments": [
            {
                "segment_id": "platform",
                "name": "Platform",
                "supporting_text": "Our Platform segment serves customers.",
                "dimensions": [{"key": "product", "value": "software"}],
            }
        ],
        "driver_definitions": [
            {
                "driver_id": "platform-volume-price",
                "archetype": "volume_price",
                "segment_id": "platform",
                "inputs": ["volume", "price"],
                "units": [
                    {"key": "volume", "value": "units"},
                    {"key": "price", "value": "USD/unit"},
                ],
                "supporting_text": "Our Platform relationship is volume times price.",
            }
        ],
        "driver_observations": [
            {
                "segment_id": "platform",
                "driver_id": "volume",
                "fiscal_year": 2026,
                "value": 12.5,
                "unit": "units",
                "supporting_text": "We served 12.5 units in FY2026.",
            }
        ],
        "investment_program_facts": [],
    }
    response = SimpleNamespace(
        output_parsed=None,
        output=[
            SimpleNamespace(
                content=[SimpleNamespace(parsed=payload)],
            )
        ],
    )

    parsed = OpenAIClient._structured_output(response)
    normalized = ExtractedOperatingEvidenceResponse.model_validate(parsed)

    assert normalized.segments[0].dimensions == {"product": "software"}
    assert normalized.definitions[0].units == {
        "volume": "units",
        "price": "USD/unit",
    }
    assert normalized.observations[0].value == 12.5
    assert normalized.investment_programs == []

    with pytest.raises(ValidationError):
        ExtractedOperatingEvidenceResponse.model_validate(
            {**payload, "forecasts": [{"fiscal_year": 2027, "revenue": 100}]}
        )


def test_openai_client_reads_output_text_and_preserves_validation_error_detail():
    payload = {
        "segments": [],
        "definitions": [],
        "observations": [],
        "investment_programs": [],
    }
    response = SimpleNamespace(output_text=json.dumps(payload))
    assert OpenAIClient._structured_output(response) == payload

    incomplete = SimpleNamespace(
        output_parsed=None,
        output=[],
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    with pytest.raises(OpenAIExtractionError, match="max_output_tokens"):

        class _IncompleteClient:
            class responses:
                @staticmethod
                async def parse(**_kwargs):
                    return incomplete

        asyncio.run(
            OpenAIClient(client=_IncompleteClient(), max_attempts=1).extract_structured(
                instructions="extract evidence",
                content="source",
                response_model=ExtractedOperatingEvidenceResponse,
            )
        )

    class _Client:
        class responses:
            @staticmethod
            async def parse(**_kwargs):
                return SimpleNamespace(output_parsed={"forecasts": [{"revenue": 100}]})

    client = OpenAIClient(client=_Client())
    with pytest.raises(OpenAIExtractionError, match="forecasts"):
        asyncio.run(
            client.extract_structured(
                instructions="extract evidence",
                content="source",
                response_model=ExtractedOperatingEvidenceResponse,
            )
        )


def test_openai_client_reports_api_schema_rejection_reason():
    error = BadRequestError(
        "structured schema rejected",
        response=httpx.Response(
            400,
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        ),
        body={
            "error": {"message": "Invalid schema: additionalProperties must be false"}
        },
    )

    class _Client:
        class responses:
            @staticmethod
            async def parse(**_kwargs):
                raise error

    with pytest.raises(
        OpenAIExtractionError,
        match="additionalProperties must be false",
    ):
        asyncio.run(
            OpenAIClient(client=_Client(), max_attempts=1).extract_structured(
                instructions="extract evidence",
                content="source",
                response_model=ExtractedOperatingEvidenceResponse,
            )
        )


def test_existing_operating_fixture_maps_to_compatible_evidence_response_shape():
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "operating"
            / "structured_evidence.json"
        ).read_text()
    )
    response = ExtractedOperatingEvidenceResponse.model_validate(
        {
            "operating_segments": [
                {**item, "supporting_text": "The company reports a Cloud segment."}
                for item in fixture["segments"]
            ],
            "driver_definitions": [
                {
                    **item,
                    "supporting_text": "Cloud revenue is volume times price.",
                }
                for item in fixture["definitions"]
            ],
            "driver_observations": [
                {
                    **item,
                    "supporting_text": (
                        f"Cloud {item['driver_id']} was reported for FY"
                        f"{item['fiscal_year']}."
                    ),
                }
                for item in fixture["observations"]
            ],
            "investment_program_facts": [],
        }
    )

    assert len(response.segments) == len(fixture["segments"])
    assert len(response.definitions) == len(fixture["definitions"])
    assert len(response.observations) == len(fixture["observations"])


def test_discovery_warning_includes_structured_extraction_reason(tmp_path):
    filing = _filing()
    ai = _FakeOpenAI(error=OpenAIExtractionError("schema field 'units' is invalid"))
    result = asyncio.run(
        OperatingEvidenceDiscoveryService(
            _Edgar(filing),
            OperatingEvidenceExtractor(ai, FileSystemCache(tmp_path)),
            max_filings=1,
            max_documents=1,
        ).discover(cik=1, as_of=datetime.date(2026, 3, 1))
    )

    assert any(
        "schema field 'units' is invalid" in warning for warning in result.warnings
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
    assert all(
        "extraction" in warning for result in results for warning in result.warnings
    )


def test_operating_context_retains_historical_segment_table_rows_and_period_headers():
    text = "\n".join(
        [
            "Outlook and other narrative " * 80,
            "Segment revenue table",
            "Fiscal year     FY2024     FY2025",
            "Automotive      80,000     95,000",
            "Energy           6,000      8,000",
        ]
    )

    context = extract_operating_context(text, max_chars=1_000, window_chars=120)

    assert "Fiscal year" in context
    assert "Automotive" in context
    assert "FY2025" in context


def test_grounded_segment_revenue_creates_generic_growth_fallback(tmp_path):
    text = "The Automotive segment reported revenue of $100 million in FY2025."
    response = ExtractedOperatingEvidenceResponse(
        segments=[
            ExtractedOperatingSegment(
                segment_id="Automotive business",
                name="Automotive business",
                supporting_text=text,
            )
        ],
        observations=[
            ExtractedOperatingObservation(
                segment_id="Automotive business",
                driver_id="segment_revenue",
                fiscal_year=2025,
                value=100,
                unit="USD millions",
                supporting_text=text,
            )
        ],
    )

    entry, _ = asyncio.run(
        OperatingEvidenceExtractor(
            _FakeOpenAI(response), FileSystemCache(tmp_path)
        ).extract(_filing(), _document(text), text, as_of=datetime.date(2026, 3, 1))
    )

    assert entry.segments[0].segment_id == "automotive"
    assert entry.observations[0].segment_id == "automotive"
    assert entry.observations[0].normalized_value == Decimal("100000000")
    assert entry.definitions[0].archetype == OperatingArchetype.GENERIC_SEGMENT_GROWTH
    assert any(
        "generic segment-growth fallback" in (item.reason or "")
        for item in entry.audit_records
    )


def test_implied_price_and_arpu_use_only_same_period_reported_revenue_pairs(tmp_path):
    text = (
        "The segment reported revenue of $100 million, 20 million units, "
        "and 10 million subscribers in FY2025."
    )
    response = ExtractedOperatingEvidenceResponse(
        observations=[
            ExtractedOperatingObservation(
                segment_id="segment",
                driver_id=driver_id,
                fiscal_year=2025,
                value=value,
                unit=unit,
                supporting_text=text,
            )
            for driver_id, value, unit in (
                ("segment_revenue", 100, "USD millions"),
                ("volume", 20, "million units"),
                ("subscribers", 10, "million users"),
            )
        ]
    )

    entry, _ = asyncio.run(
        OperatingEvidenceExtractor(
            _FakeOpenAI(response), FileSystemCache(tmp_path)
        ).extract(_filing(), _document(text), text, as_of=datetime.date(2026, 3, 1))
    )

    derived = {
        item.driver_id: item for item in entry.observations if item.origin == "derived"
    }
    assert derived["implied_price"].value == Decimal("5")
    assert derived["implied_arpu"].value == Decimal("10")
    assert all(item.fiscal_period == "FY" for item in derived.values())
    assert all("same-period" not in (item.reason or "") for item in entry.audit_records)


def test_extractor_preserves_scope_evidence_and_total_flags(tmp_path):
    text = "The total segment reported $100 million in FY2025."
    response = ExtractedOperatingEvidenceResponse(
        observations=[
            ExtractedOperatingObservation(
                segment_id="segment",
                driver_id="revenue",
                fiscal_year=2025,
                value=100,
                unit="USD millions",
                scope="segment",
                scope_evidence="The total segment",
                is_total=True,
                supporting_text=text,
            )
        ]
    )

    entry, _ = asyncio.run(
        OperatingEvidenceExtractor(
            _FakeOpenAI(response), FileSystemCache(tmp_path)
        ).extract(_filing(), _document(text), text, as_of=datetime.date(2026, 3, 1))
    )

    observation = entry.observations[0]
    assert observation.scope == "segment"
    assert observation.scope_evidence == "The total segment"
    assert observation.is_total


def test_incompatible_observation_units_are_rejected_with_item_audit_reason(tmp_path):
    text = (
        "The segment uses volume and price to describe revenue. "
        "The segment reported 20 USD millions of volume in FY2025."
    )
    response = ExtractedOperatingEvidenceResponse(
        definitions=[
            ExtractedOperatingDriverDefinition(
                driver_id="segment-volume-price",
                archetype="volume_price",
                segment_id="segment",
                input_metrics=("volume", "price"),
                required_inputs=("volume", "price"),
                units={"volume": "units", "price": "USD/unit"},
                supporting_text="The segment uses volume and price to describe revenue.",
            )
        ],
        observations=[
            ExtractedOperatingObservation(
                segment_id="segment",
                driver_id="volume",
                fiscal_year=2025,
                value=20,
                unit="USD millions",
                supporting_text=text,
            )
        ],
    )

    entry, _ = asyncio.run(
        OperatingEvidenceExtractor(
            _FakeOpenAI(response), FileSystemCache(tmp_path)
        ).extract(_filing(), _document(text), text, as_of=datetime.date(2026, 3, 1))
    )

    assert entry.observations == ()
    assert entry.unusable_reasons
    assert any(
        audit.status == "unusable" and "incompatible" in (audit.reason or "")
        for audit in entry.audit_records
    )


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


def test_gap_targeted_retry_inspects_ranked_exhibit_outside_initial_document_budget(
    tmp_path,
):
    filing = _filing().model_copy(
        update={
            "form": "10-K",
            "documents": (
                SecFilingDocument(
                    filename="annual.htm",
                    document_type="10-K",
                    description="Annual report",
                    content="The platform reported revenue of $100 million in FY2025.",
                ),
                SecFilingDocument(
                    filename="kpi.htm",
                    document_type="EX-99.1",
                    description="Platform volume KPI",
                    content="The platform reported 20 million units in FY2025.",
                ),
            ),
        }
    )

    class _GapOpenAI(_FakeOpenAI):
        async def extract_structured(self, **kwargs):
            self.calls += 1
            content = kwargs["content"]
            self.contents.append(content)
            if "20 million units" in content:
                return ExtractedOperatingEvidenceResponse(
                    observations=[
                        ExtractedOperatingObservation(
                            segment_id="platform",
                            driver_id="volume",
                            fiscal_year=2025,
                            value=20,
                            unit="million units",
                            supporting_text="The platform reported 20 million units in FY2025.",
                        )
                    ]
                )
            return ExtractedOperatingEvidenceResponse(
                observations=[
                    ExtractedOperatingObservation(
                        segment_id="platform",
                        driver_id="revenue",
                        fiscal_year=2025,
                        value=100,
                        unit="USD millions",
                        supporting_text="The platform reported revenue of $100 million in FY2025.",
                    )
                ]
            )

    class _GapEdgar(_Edgar):
        async def get_guidance_filings(self, cik, **kwargs):
            return [filing]

        async def get_filing_documents(self, filing, **kwargs):
            return filing

    ai = _GapOpenAI()
    result = asyncio.run(
        OperatingEvidenceDiscoveryService(
            _GapEdgar(filing),
            OperatingEvidenceExtractor(ai, FileSystemCache(tmp_path)),
            max_filings=1,
            max_documents_per_filing=1,
            max_documents=1,
        ).discover(cik=1, as_of=datetime.date(2026, 3, 1), fiscal_years=(2025,))
    )

    assert result.exhibits_found == 1
    assert any(item.driver_id == "volume" for item in result.observations)
    assert result.gaps_resolved_sec
    assert any("20 million units" in content for content in ai.contents)


def test_ir_fallback_requires_profile_url_and_records_provider_resolution(tmp_path):
    filing = _filing().model_copy(
        update={
            "documents": (
                SecFilingDocument(
                    filename="annual.htm",
                    document_type="10-K",
                    description="Annual report",
                    content="The platform reported revenue of $100 million in FY2025.",
                ),
            )
        }
    )

    class _Ir:
        def __init__(self):
            self.urls = []

        async def retrieve(self, *, url, gaps, as_of):
            self.urls.append(url)
            ir_text = "The platform reported 20 million units in FY2025."
            return (
                (
                    filing,
                    SecFilingDocument(
                        filename="ir-kpi.htm",
                        document_type="IR-KPI",
                        description="Investor relations KPI release",
                        content=ir_text,
                    ),
                    ir_text,
                ),
            )

    class _IrEdgar(_Edgar):
        async def get_filing_documents(self, filing, **kwargs):
            return filing

    text = "The platform reported revenue of $100 million in FY2025."
    ai_response = ExtractedOperatingEvidenceResponse(
        observations=[
            ExtractedOperatingObservation(
                segment_id="platform",
                driver_id="revenue",
                fiscal_year=2025,
                value=100,
                unit="USD millions",
                supporting_text=text,
            )
        ]
    )

    class _IrOpenAI(_FakeOpenAI):
        async def extract_structured(self, **kwargs):
            self.calls += 1
            content = kwargs["content"]
            self.contents.append(content)
            if "20 million units" in content:
                return ExtractedOperatingEvidenceResponse(
                    observations=[
                        ExtractedOperatingObservation(
                            segment_id="platform",
                            driver_id="volume",
                            fiscal_year=2025,
                            value=20,
                            unit="million units",
                            supporting_text="The platform reported 20 million units in FY2025.",
                        )
                    ]
                )
            return ai_response

    ir = _Ir()
    result = asyncio.run(
        OperatingEvidenceDiscoveryService(
            _IrEdgar(filing),
            OperatingEvidenceExtractor(
                _IrOpenAI(),
                FileSystemCache(tmp_path),
            ),
            ir_fallback=ir,
            max_filings=1,
            max_documents=1,
        ).discover(
            cik=1,
            as_of=datetime.date(2026, 3, 1),
            fiscal_years=(2025,),
            profile_metadata={"investorWebsite": "https://example.test/investors"},
        )
    )

    assert ir.urls == ["https://example.test/investors"]
    assert result.ir_diagnostic is None
    assert result.gaps_resolved_ir
    assert not result.gaps_resolved_sec
    assert any(
        item.evidence is not None and item.evidence.provider == "company_ir"
        for item in result.observations
    )
