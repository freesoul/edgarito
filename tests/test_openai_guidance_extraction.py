import asyncio
import datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from edgarito.schemas.guidance.management import (
    ExtractedGuidanceItem,
    ExtractedGuidanceResponse,
    GuidanceBasis,
    GuidanceMetric,
    GuidancePeriodType,
    GuidanceQualifier,
    GuidanceScope,
    GuidanceStatus,
    GuidanceUnit,
    GuidanceValueKind,
)
from edgarito.schemas.providers.edgar.filing import SecFiling, SecFilingDocument
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.guidance.documents import (
    GUIDANCE_CONTEXT_MAX_CHARS,
    extract_guidance_context,
)
from edgarito.services.guidance.extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    ManagementGuidanceExtractor,
)
from edgarito.services.openai import (
    OpenAIClient,
    OpenAIExtractionError,
    OpenAIUnavailableError,
)


class _Responses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(output_parsed=output, output=[])


class _Sdk:
    def __init__(self, outputs):
        self.responses = _Responses(outputs)


def test_openai_client_uses_async_responses_structured_outputs():
    class Result(ExtractedGuidanceResponse):
        pass

    expected = Result(guidance=[])
    sdk = _Sdk([expected])
    client = OpenAIClient(client=sdk, model="test-model", reasoning_effort="low")

    result = asyncio.run(
        client.extract_structured(
            instructions="instructions", content="content", response_model=Result
        )
    )

    assert result == expected
    call = sdk.responses.calls[0]
    assert call["model"] == "test-model"
    assert call["text_format"] is Result
    assert call["store"] is False
    assert call["reasoning"] == {"effort": "low"}


def test_openai_client_missing_key_disables_extraction():
    client = OpenAIClient(api_key=None, client=None)
    with pytest.raises(OpenAIUnavailableError):
        asyncio.run(
            client.extract_structured(
                instructions="x",
                content="y",
                response_model=ExtractedGuidanceResponse,
            )
        )


def test_openai_client_retries_transient_failure(monkeypatch):
    class Transient(Exception):
        pass

    sdk = _Sdk([Transient("rate limit"), ExtractedGuidanceResponse(guidance=[])])
    client = OpenAIClient(client=sdk, max_attempts=2, retry_delay=0)
    monkeypatch.setattr(OpenAIClient, "_TRANSIENT_ERRORS", (Transient,))

    asyncio.run(
        client.extract_structured(
            instructions="x",
            content="y",
            response_model=ExtractedGuidanceResponse,
        )
    )

    assert len(sdk.responses.calls) == 2


def test_openai_client_rejects_refused_or_malformed_output():
    sdk = _Sdk([None])
    client = OpenAIClient(client=sdk)
    with pytest.raises(OpenAIExtractionError, match="no structured output"):
        asyncio.run(
            client.extract_structured(
                instructions="x",
                content="y",
                response_model=ExtractedGuidanceResponse,
            )
        )


class _DomainOpenAI:
    def __init__(self, response, *, model="gpt-test", effort="low"):
        self.response = response
        self.model = model
        self.reasoning_effort = effort
        self.calls = 0
        self.contents = []

    async def extract_structured(self, **kwargs):
        self.calls += 1
        self.contents.append(kwargs["content"])
        return self.response


def _filing() -> SecFiling:
    return SecFiling(
        cik=1,
        accession_number="0000000001-26-000001",
        form="6-K",
        filing_date=datetime.date(2026, 7, 15),
        primary_document="wrapper.htm",
    )


def _document(content=None) -> SecFilingDocument:
    return SecFilingDocument(
        filename="ex991.htm",
        document_type="EX-99.1",
        description="Financial results press release",
        content=content
        or "We expect full-year 2026 revenue of $10.0 billion to $10.5 billion.",
    )


def _response(supporting_text=None) -> ExtractedGuidanceResponse:
    return ExtractedGuidanceResponse(
        guidance=[
            ExtractedGuidanceItem(
                metric=GuidanceMetric.REVENUE,
                fiscal_year=2026,
                period_type=GuidancePeriodType.FISCAL_YEAR,
                low=Decimal("10.0"),
                high=Decimal("10.5"),
                value_kind=GuidanceValueKind.MONETARY,
                currency="USD",
                unit=GuidanceUnit.BILLIONS,
                basis=GuidanceBasis.GAAP,
                scope=GuidanceScope.CONSOLIDATED,
                qualifier=GuidanceQualifier.RANGE,
                status=GuidanceStatus.ISSUED,
                supporting_text=supporting_text
                or "We expect full-year 2026 revenue of $10.0 billion to $10.5 billion.",
                extraction_confidence=Decimal("0.95"),
            )
        ]
    )


def test_normalized_extraction_cache_is_one_time_per_identity(tmp_path):
    cache = FileSystemCache(tmp_path)
    ai = _DomainOpenAI(_response())
    extractor = ManagementGuidanceExtractor(ai, cache)
    filing = _filing()
    document = _document()

    first, first_hit = asyncio.run(
        extractor.extract(
            filing, document, document.content, valuation_date=datetime.date(2026, 8, 1)
        )
    )
    second, second_hit = asyncio.run(
        ManagementGuidanceExtractor(ai, cache).extract(
            filing, document, document.content, valuation_date=datetime.date(2026, 8, 1)
        )
    )

    assert not first_hit
    assert second_hit
    assert ai.calls == 1
    assert first == second
    assert first.accepted[0].low == Decimal("10000000000.0")
    assert first.accepted[0].midpoint == Decimal("10250000000.00")
    assert "gpt-test" in next(tmp_path.rglob("*.json")).read_text()
    assert "api_key" not in next(tmp_path.rglob("*.json")).read_text().casefold()


@pytest.mark.parametrize("change", ["content", "model", "prompt", "schema"])
def test_extraction_cache_invalidates_for_identity_changes(tmp_path, change):
    cache = FileSystemCache(tmp_path)
    ai = _DomainOpenAI(_response())
    filing = _filing()
    document = _document()
    asyncio.run(
        ManagementGuidanceExtractor(ai, cache).extract(
            filing, document, document.content, valuation_date=datetime.date(2026, 8, 1)
        )
    )

    changed_document = document
    changed_ai = ai
    prompt = PROMPT_VERSION
    schema = SCHEMA_VERSION
    if change == "content":
        changed_document = _document(document.content + " Updated outlook.")
    elif change == "model":
        changed_ai = _DomainOpenAI(_response(), model="gpt-other")
    elif change == "prompt":
        prompt = f"{PROMPT_VERSION}-changed"
    else:
        schema = f"{SCHEMA_VERSION}-changed"
    asyncio.run(
        ManagementGuidanceExtractor(
            changed_ai, cache, prompt_version=prompt, schema_version=schema
        ).extract(
            filing,
            changed_document,
            changed_document.content,
            valuation_date=datetime.date(2026, 8, 1),
        )
    )

    assert sum(item.calls for item in {ai, changed_ai}) == 2


def test_unmatched_supporting_evidence_is_rejected_and_cached(tmp_path):
    ai = _DomainOpenAI(_response("We invented revenue guidance of $99 billion."))
    entry, _ = asyncio.run(
        ManagementGuidanceExtractor(ai, FileSystemCache(tmp_path)).extract(
            _filing(),
            _document(),
            _document().content,
            valuation_date=datetime.date(2026, 8, 1),
        )
    )

    assert entry.accepted == ()
    assert (
        entry.rejected[0].reason == "Supporting text was not found in the SEC document"
    )


def test_long_document_uses_bounded_guidance_context_for_llm_and_full_text_for_evidence(
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
    full_text = (
        "Historical discussion without current information. " * 10_000
    ) + phrase
    document = _document(full_text)
    ai = _DomainOpenAI(response)

    entry, cache_hit = asyncio.run(
        ManagementGuidanceExtractor(ai, FileSystemCache(tmp_path)).extract(
            _filing(), document, full_text, valuation_date=datetime.date(2026, 8, 1)
        )
    )

    assert not cache_hit
    assert len(ai.contents[0]) <= GUIDANCE_CONTEXT_MAX_CHARS
    assert len(ai.contents[0]) < len(full_text)
    assert phrase in ai.contents[0]
    assert extract_guidance_context(full_text) == ai.contents[0]
    assert len(entry.accepted) == 1
    assert entry.accepted[0].supporting_text == phrase


@pytest.mark.parametrize(
    ("text", "expected_reason"),
    [
        (
            "Revenue for the quarter was $10.0 billion.",
            "actual result, not guidance",
        ),
        (
            "Analyst consensus expects revenue of $10.0 billion.",
            "third-party expectations",
        ),
        (
            "We expect high-single-digit growth.",
            "numerical values are absent",
        ),
    ],
)
def test_results_analyst_estimates_and_qualitative_text_cannot_validate(
    tmp_path, text, expected_reason
):
    response = _response(text).model_copy(
        update={
            "guidance": [
                _response(text)
                .guidance[0]
                .model_copy(
                    update={
                        "point": Decimal("10.0"),
                        "low": None,
                        "high": None,
                    }
                )
            ]
        }
    )
    document = _document(text)
    entry, _ = asyncio.run(
        ManagementGuidanceExtractor(
            _DomainOpenAI(response), FileSystemCache(tmp_path)
        ).extract(
            _filing(),
            document,
            document.content,
            valuation_date=datetime.date(2026, 8, 1),
        )
    )

    assert entry.accepted == ()
    assert expected_reason in entry.rejected[0].reason
