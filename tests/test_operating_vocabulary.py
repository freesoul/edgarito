import asyncio
import datetime

from edgarito.schemas.vocabulary import (
    DiscoveredKpiTerm,
    DiscoveredKpiVocabularyResponse,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.operating.vocabulary import (
    KpiVocabularyProvider,
    normalize_industry_namespace,
)


class _OpenAI:
    def __init__(self):
        self.calls = 0

    async def extract_structured(self, **_kwargs):
        self.calls += 1
        return DiscoveredKpiVocabularyResponse(
            terms=(
                DiscoveredKpiTerm(
                    raw_term="vehicle deliveries",
                    canonical_metric="deliveries",
                    rationale="Vehicle deliveries were 1.8 million",
                    source_document="release.html",
                    supporting_text="Vehicle deliveries were 1.8 million.",
                    confidence="high",
                ),
            )
        )


def test_automobile_classification_alias_uses_automotive_registry():
    provider = KpiVocabularyProvider()

    assert normalize_industry_namespace("Auto Manufacturers") == "automotive"
    assert any(
        raw == "asp" for raw, _metric in provider.normal_terms("Auto Manufacturers")
    )


def test_global_vocabulary_is_available_without_classification():
    provider = KpiVocabularyProvider()

    terms = provider.normal_terms()

    assert len(terms) > 0
    assert any(raw == "deliveries" for raw, _metric in terms)


def test_fallback_can_be_forced_after_industry_search(tmp_path):
    client = _OpenAI()
    provider = KpiVocabularyProvider(client, FileSystemCache(tmp_path))

    terms, audit = asyncio.run(
        provider.discover(
            context="The company reports vehicle deliveries.",
            source_document="release.html",
            source_text="Vehicle deliveries were 1.8 million.",
            industry="Auto Manufacturers",
            business_archetype="general_operating",
            as_of=datetime.date(2026, 1, 1),
            force=True,
            fallback_reason="no usable reconstruction pairs after industry search",
        )
    )

    assert client.calls == 1
    assert terms[0].canonical_metric == "deliveries"
    assert audit.fallback_triggered
    assert audit.fallback_reason.startswith("no usable")
    assert audit.normalized_industry == "automotive"


def test_automotive_registry_expansion_is_not_collapsed():
    provider = KpiVocabularyProvider()

    terms = provider.normal_terms("Auto Manufacturers")
    metrics = {raw: metric for raw, metric in terms}

    assert provider.industry_term_count("Auto Manufacturers") == 3
    assert metrics["deliveries"] == "deliveries"
    assert metrics["production"] == "production"
    assert metrics["asp"] == "price"
