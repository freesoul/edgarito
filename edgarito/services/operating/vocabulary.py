"""Layered, grounded KPI terminology discovery."""

from __future__ import annotations

import datetime
import hashlib
import json
import re

from edgarito.config.operating import OPERATING_VOCABULARY
from edgarito.schemas.vocabulary import DiscoveredKpiTerm, KpiVocabularyAudit
from edgarito.services.cache.filesystem_cache import FileSystemCache

VOCABULARY_SCHEMA_VERSION = "kpi-vocabulary-v1"
VOCABULARY_PROMPT_VERSION = "kpi-terminology-v1"

# Deliberately small global vocabulary. Industry additions and provider-label
# aliases are versioned data; the compiled view retains their source order.
GLOBAL_KPI_TERMS = OPERATING_VOCABULARY.global_terms
INDUSTRY_KPI_TERMS = OPERATING_VOCABULARY.industry_terms
INDUSTRY_ALIASES = OPERATING_VOCABULARY.industry_aliases
VOCABULARY_CONFIG_VERSION = OPERATING_VOCABULARY.cache_version


def normalize_vocabulary_key(value: object | None) -> str:
    return re.sub(
        r"[^a-z0-9]+", "_", str(getattr(value, "value", value) or "").casefold()
    ).strip("_")


def normalize_industry_namespace(value: object | None) -> str:
    """Return the stable vocabulary namespace for a provider industry label."""
    key = normalize_vocabulary_key(value)
    return INDUSTRY_ALIASES.get(key, key)


class KpiVocabularyProvider:
    """Combines deterministic taxonomy terms and optionally grounded LLM terms."""

    GLOBAL_KPI_TERMS = GLOBAL_KPI_TERMS

    def __init__(
        self,
        openai_client=None,
        cache: FileSystemCache | None = None,
        *,
        model: str | None = None,
    ):
        self.openai = openai_client
        self.cache = cache
        self.model = model or getattr(openai_client, "model", "unknown")

    def normal_terms(
        self, industry: object | None = None, business_archetype: object | None = None
    ) -> tuple[tuple[str, str], ...]:
        key = normalize_industry_namespace(industry)
        archetype = normalize_vocabulary_key(business_archetype)
        additions = (
            *INDUSTRY_KPI_TERMS.get(key, ()),
            *INDUSTRY_KPI_TERMS.get(archetype, ()),
        )
        return tuple(dict.fromkeys((*GLOBAL_KPI_TERMS, *additions)))

    def context(self, industry=None, business_archetype=None) -> str:
        return ", ".join(
            raw for raw, _ in self.normal_terms(industry, business_archetype)
        )

    def industry_term_count(
        self, industry: object | None = None, business_archetype: object | None = None
    ) -> int:
        key = normalize_industry_namespace(industry)
        archetype = normalize_vocabulary_key(business_archetype)
        return len(
            tuple(
                dict.fromkeys(
                    (
                        *INDUSTRY_KPI_TERMS.get(key, ()),
                        *INDUSTRY_KPI_TERMS.get(archetype, ()),
                    )
                )
            )
        )

    async def discover(
        self,
        *,
        context: str,
        source_document: str,
        source_text: str,
        industry=None,
        business_archetype=None,
        as_of: datetime.date,
        document_hash: str | None = None,
        force: bool = False,
        fallback_reason: str | None = None,
    ) -> tuple[tuple[DiscoveredKpiTerm, ...], KpiVocabularyAudit]:
        normal = self.normal_terms(industry, business_archetype)
        hits = {
            raw
            for raw, _ in normal
            if re.search(rf"\b{re.escape(raw)}\b", source_text, re.I)
        }
        base = KpiVocabularyAudit(
            global_count=len(GLOBAL_KPI_TERMS),
            industry_count=max(0, len(normal) - len(GLOBAL_KPI_TERMS)),
            terms=tuple(raw for raw, _ in normal),
            cache_status="not_needed",
            raw_industry=str(industry or ""),
            normalized_industry=normalize_industry_namespace(industry),
            selected_archetype=normalize_vocabulary_key(business_archetype),
            fallback_triggered=force,
            fallback_reason=fallback_reason or "",
        )
        if (len(hits) >= 2 and not force) or self.openai is None or self.cache is None:
            return (), base
        digest = document_hash or hashlib.sha256(source_text.encode()).hexdigest()
        key = hashlib.sha256(
            f"{digest}|{self.model}|{VOCABULARY_PROMPT_VERSION}|{VOCABULARY_SCHEMA_VERSION}|{VOCABULARY_CONFIG_VERSION}".encode()
        ).hexdigest()
        path = f"kpi-vocabulary/{normalize_industry_namespace(industry) or normalize_vocabulary_key(business_archetype) or 'unresolved'}/{key}.json"
        cached = self.cache.read(path)
        if cached is not None:
            terms = tuple(
                DiscoveredKpiTerm.model_validate(item) for item in json.loads(cached)
            )
            # An earlier deterministic pass may have cached an empty result.
            # An evidence-quality retry must be allowed to ask the terminology
            # model again rather than treating that empty cache as a discovery hit.
            if force and not terms:
                cached = None
            else:
                return terms, base.model_copy(
                    update={
                        "discovered_count": len(terms),
                        "cache_status": "hit",
                        "terms": tuple(item.raw_term for item in terms),
                        "validated_terms": tuple(item.raw_term for item in terms),
                    }
                )
        from edgarito.schemas.vocabulary import DiscoveredKpiVocabularyResponse

        instructions = "Return terminology only. No values, forecasts, or estimates. Every term must be copied from the source and rationale/support must be grounded in it."
        response = await self.openai.extract_structured(
            instructions=instructions,
            content=context[:24000],
            response_model=DiscoveredKpiVocabularyResponse,
        )
        accepted: list[DiscoveredKpiTerm] = []
        rejected = 0
        for item in response.terms:
            if not _grounded(item, source_text, source_document):
                rejected += 1
                continue
            accepted.append(
                item.model_copy(
                    update={
                        "industry": normalize_industry_namespace(industry),
                        "business_archetype": normalize_vocabulary_key(
                            business_archetype
                        ),
                        "source_document": source_document,
                        "first_validated": as_of,
                        "last_validated": as_of,
                        "schema_version": VOCABULARY_SCHEMA_VERSION,
                    }
                )
            )
        self.cache.save(
            path,
            json.dumps(
                [item.model_dump(mode="json") for item in accepted], sort_keys=True
            ),
        )
        return tuple(accepted), base.model_copy(
            update={
                "discovered_count": len(accepted),
                "rejected_count": rejected,
                "cache_status": "miss",
                "terms": tuple(item.raw_term for item in accepted),
                "validated_terms": tuple(item.raw_term for item in accepted),
                "raw_industry": str(industry or ""),
                "normalized_industry": normalize_industry_namespace(industry),
                "selected_archetype": normalize_vocabulary_key(business_archetype),
                "fallback_reason": fallback_reason or "",
            }
        )


def _grounded(item: DiscoveredKpiTerm, source_text: str, source_document: str) -> bool:
    source = " ".join(source_text.split()).casefold()
    support = " ".join(item.supporting_text.split()).casefold()
    rationale = " ".join(item.rationale.split()).casefold()
    rationale_tokens = {
        token for token in re.findall(r"[a-z0-9]+", rationale) if len(token) > 3
    }
    source_tokens = set(re.findall(r"[a-z0-9]+", source))
    return bool(
        source_document
        and support in source
        and item.raw_term.casefold() in support
        and rationale_tokens
        and rationale_tokens <= source_tokens
    )
