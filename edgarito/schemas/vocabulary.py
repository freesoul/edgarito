"""Provider-neutral KPI vocabulary contracts."""

from __future__ import annotations

import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DiscoveredKpiTerm(BaseModel):
    """A terminology hint grounded in a first-party document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_term: str = Field(min_length=1, max_length=160)
    canonical_metric: str = Field(min_length=1, max_length=160)
    industry: str = ""
    business_archetype: str = ""
    rationale: str = Field(min_length=1, max_length=500)
    source_document: str = Field(min_length=1, max_length=300)
    supporting_text: str = Field(min_length=1, max_length=2_000)
    confidence: str = "medium"
    first_validated: datetime.date = datetime.date(2000, 1, 1)
    last_validated: datetime.date = datetime.date(2000, 1, 1)
    schema_version: str = "kpi-vocabulary-v1"
    company_specific: bool = False

    @field_validator(
        "raw_term",
        "canonical_metric",
        "industry",
        "business_archetype",
        "rationale",
        "source_document",
        "supporting_text",
    )
    @classmethod
    def clean_text(cls, value: str) -> str:
        return " ".join(str(value).split())

    @field_validator("confidence", mode="before")
    @classmethod
    def clean_confidence(cls, value: str) -> str:
        value = str(getattr(value, "value", value)).casefold().strip()
        if value not in {"high", "medium", "low"}:
            raise ValueError("confidence must be high, medium, or low")
        return value


class KpiVocabularyAudit(BaseModel):
    """Content-free vocabulary discovery diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    global_count: int = 0
    industry_count: int = 0
    discovered_count: int = 0
    rejected_count: int = 0
    cache_status: str = "miss"
    terms: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


class DiscoveredKpiVocabularyResponse(BaseModel):
    """Strict terminology-only model exposed to the language model."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    terms: tuple[DiscoveredKpiTerm, ...] = ()
