"""Layered, grounded KPI terminology discovery."""

from __future__ import annotations

import datetime
import hashlib
import json
import re

from edgarito.schemas.vocabulary import DiscoveredKpiTerm, KpiVocabularyAudit
from edgarito.services.cache.filesystem_cache import FileSystemCache

VOCABULARY_SCHEMA_VERSION = "kpi-vocabulary-v1"
VOCABULARY_PROMPT_VERSION = "kpi-terminology-v1"

# Deliberately small global vocabulary. Industry additions live below by
# taxonomy/archetype, never by issuer or ticker.
GLOBAL_KPI_TERMS = (
    ("revenue", "revenue"),
    ("sales", "revenue"),
    ("volume", "volume"),
    ("price", "price"),
    ("customers", "customers"),
    ("capacity", "capacity"),
    ("utilization", "utilization"),
    ("production", "production"),
    ("deliveries", "deliveries"),
    ("shipments", "shipments"),
    ("units", "volume"),
    ("deployments", "deployments"),
    ("subscribers", "subscribers"),
    ("users", "users"),
    ("members", "members"),
    ("arpu", "arpu"),
    ("backlog", "backlog"),
    ("orders", "orders"),
    ("transactions", "transactions"),
    ("stores", "stores"),
    ("sites", "sites"),
)
INDUSTRY_KPI_TERMS = {
    "general_operating": (("backlog", "backlog"), ("orders", "orders")),
    "automobile_manufacturers": (
        ("deliveries", "deliveries"),
        ("production", "production"),
    ),
    "automotive": (
        ("deliveries", "deliveries"),
        ("production", "production"),
        ("asp", "price"),
    ),
    "semiconductors": (
        ("wafer shipments", "volume"),
        ("bit shipments", "volume"),
        ("utilization", "utilization"),
        ("asp", "price"),
    ),
    "cloud_software": (
        ("seats", "users"),
        ("arr", "revenue"),
        ("mrr", "revenue"),
        ("workloads", "volume"),
    ),
    "bank": (
        ("loans", "volume"),
        ("deposits", "volume"),
        ("nim", "price"),
        ("aum", "volume"),
    ),
    "energy_storage": (
        ("gwh", "capacity"),
        ("mwh", "capacity"),
        ("deployments", "deployments"),
        ("installed capacity", "capacity"),
    ),
    "technology_platform": (
        ("subscribers", "subscribers"),
        ("average revenue per user", "arpu"),
    ),
    "resource_producer": (("production", "production"), ("reserves", "reserves")),
    "reit_property": (("occupancy", "occupancy"), ("same-store NOI", "same_store_noi")),
    "project_pipeline": (("bookings", "bookings"), ("backlog", "backlog")),
}


def normalize_vocabulary_key(value: object | None) -> str:
    return re.sub(
        r"[^a-z0-9]+", "_", str(getattr(value, "value", value) or "").casefold()
    ).strip("_")


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
        key = normalize_vocabulary_key(industry)
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
        )
        if len(hits) >= 2 or self.openai is None or self.cache is None:
            return (), base
        digest = document_hash or hashlib.sha256(source_text.encode()).hexdigest()
        key = hashlib.sha256(
            f"{digest}|{self.model}|{VOCABULARY_PROMPT_VERSION}|{VOCABULARY_SCHEMA_VERSION}".encode()
        ).hexdigest()
        path = f"kpi-vocabulary/{normalize_vocabulary_key(industry) or normalize_vocabulary_key(business_archetype) or 'unresolved'}/{key}.json"
        cached = self.cache.read(path)
        if cached is not None:
            terms = tuple(
                DiscoveredKpiTerm.model_validate(item) for item in json.loads(cached)
            )
            return terms, base.model_copy(
                update={
                    "discovered_count": len(terms),
                    "cache_status": "hit",
                    "terms": tuple(item.raw_term for item in terms),
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
                        "industry": normalize_vocabulary_key(industry),
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
