"""Async OpenAI proposal boundary for ForecastReasoner v1."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.forecasting.reasoning.cache import ForecastReasoningCache
from edgarito.services.forecasting.reasoning.contracts import (
    ForecastReasoningInput,
    ForecastReasoningMetadata,
    ForecastReasoningResponse,
)
from edgarito.services.forecasting.reasoning.evidence import (
    EvidenceCatalog,
    build_evidence_catalog,
    canonical_json,
    compact_reasoning_input,
    compact_structured,
    content_hash,
    manual_inputs_hash,
    research_hash,
)
from edgarito.services.openai import OpenAIClient

PROMPT_VERSION = "forecast_reasoner_v1_prompt_2"
SCHEMA_VERSION = "forecast_reasoner_v1_response_2"
VALIDATOR_VERSION = "forecast_reasoner_v1_validator_2"
CONTEXT_VERSION = "forecast_reasoner_v1_context_1"


FORECAST_REASONER_INSTRUCTIONS = """You are ForecastReasoner v1. Return only the strict structured response schema.

You propose auditable operating-driver and supported financial-metric assumptions;
you do not execute them. Use only the supplied compact structured input and
evidence catalog. Never browse, make web claims, invent URLs as citations, or
cite anything not identified by a catalog evidence_id. Evidence-based
assumptions MUST cite one or more catalog IDs. Model assumptions may cite
supporting catalog IDs but must be explicitly labeled model_assumption.

Evidence precedence is: accepted first-party observations and management
constraints, then normalized historical facts, then typed research evidence and
consensus, then a clearly labeled model assumption. Explain conflicts in the
rationale or unresolved items; do not silently choose an unsupported fact.

Every assumption must set assumption_type to exactly evidence_based or
model_assumption. Do not include a method field; target_type, basis, and the
target determine the proposed path while provenance is derived deterministically.
Modeling decisions are audit-only. If provided, each decision must use one
target string and target_type set to forecast_metric or operating_driver; never
send separate metric and driver_id fields. Use only the supported strategy
values in the schema.

For forecast years 1-2, prefer observable guidance and near-term operating
drivers. For years 3-5, explain a transition toward supported historical or
stable operating behavior. For years 6+, use only an accepted generic-growth
definition or a conservative explicitly supported regime. Every path must list
the exact supplied fiscal years and low/base/high values.

Do not return formulas or accounting arithmetic. In particular, do not propose
gross profit, EBIT, tax amount, NOPAT, delta NWC, FCFF, DCF, enterprise value,
or valuation assumptions. Do not calculate GP/EBIT/tax/NOPAT/delta/FCFF. Do not
invent an arbitrary formula_id, archetype, alternative formula, or direct
segment-revenue forecast. Operating-driver proposals must use the supplied
registry-backed definition and one of its required inputs. Financial targets
are limited to supported gross margin, R&D, SG&A, other operating items, tax
rate, D&A, CAPEX, OWC, and safe consolidated revenue paths with explicit units
and value bases. The deterministic compiler calculates accounting and FCFF.
"""


class ForecastReasoningProposal(BaseModel):
    """Response plus identity artifacts from one model/cache proposal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    response: ForecastReasoningResponse
    catalog: EvidenceCatalog
    metadata: ForecastReasoningMetadata
    cache_hit: bool = False
    cache_key: str

    @property
    def assumptions(self):
        return self.response.assumptions

    @property
    def modeling_decisions(self):
        return self.response.modeling_decisions

    @property
    def unresolved_items(self):
        return self.response.unresolved_items

    @property
    def warnings(self):
        return self.response.warnings

    @property
    def overall_confidence(self):
        return self.response.overall_confidence


def build_reasoning_prompt() -> str:
    """Return the versioned prompt separately for identity tests and callers."""

    return FORECAST_REASONER_INSTRUCTIONS


def build_reasoning_content(
    input_value: ForecastReasoningInput, catalog: EvidenceCatalog
) -> str:
    """Build compact canonical JSON; no source filings or raw text are sent."""

    if not isinstance(input_value, ForecastReasoningInput):
        input_value = ForecastReasoningInput.model_validate(input_value)
    if not isinstance(catalog, EvidenceCatalog):
        catalog = EvidenceCatalog.model_validate(catalog)

    return canonical_json(
        {
            "context_version": CONTEXT_VERSION,
            "input": compact_reasoning_input(input_value),
            # Exclusions remain in the returned audit catalog, but are not
            # model input: unavailable evidence must not influence a response.
            "evidence_catalog": compact_structured(catalog.items),
        }
    )


class ForecastReasoner:
    """Ask OpenAI for proposals, never for executable forecast output."""

    def __init__(
        self,
        client: OpenAIClient | Any | None = None,
        *,
        openai_client: OpenAIClient | Any | None = None,
        cache: ForecastReasoningCache | Any | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.client = client or openai_client or OpenAIClient()
        self.model = model or getattr(self.client, "model", "")
        self.reasoning_effort = reasoning_effort or getattr(
            self.client, "reasoning_effort", "medium"
        )
        if isinstance(cache, (str, Path, FileSystemCache)):
            self.cache = ForecastReasoningCache(cache)
        else:
            self.cache = cache if cache is not None else ForecastReasoningCache()

    async def reason(
        self,
        input_value: ForecastReasoningInput | Any,
        *,
        force_refresh: bool = False,
    ) -> ForecastReasoningProposal:
        input_value = (
            input_value
            if isinstance(input_value, ForecastReasoningInput)
            else ForecastReasoningInput.model_validate(input_value)
        )
        catalog = build_evidence_catalog(input_value)
        prompt = build_reasoning_prompt()
        schema_hash = content_hash(ForecastReasoningResponse.model_json_schema())
        prompt_hash = content_hash(prompt)
        validator_hash = content_hash(VALIDATOR_VERSION)
        context_hash = content_hash(
            {
                "version": CONTEXT_VERSION,
                "company_name": input_value.company_name,
                "ticker": input_value.ticker,
                "unit": input_value.unit,
            }
        )
        metadata = ForecastReasoningMetadata(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            validator_version=VALIDATOR_VERSION,
            context_version=CONTEXT_VERSION,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            validator_hash=validator_hash,
            context_hash=context_hash,
            evidence_bundle_hash=catalog.bundle_hash,
            research_hash=research_hash(input_value),
            manual_inputs_hash=manual_inputs_hash(input_value),
        )
        identity = ForecastReasoningCache.identity(
            input_value,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            validator_version=VALIDATOR_VERSION,
            context_version=CONTEXT_VERSION,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            validator_hash=validator_hash,
            context_hash=context_hash,
        )
        cache_key = identity.digest
        if not force_refresh:
            envelope = self._cache_load(identity)
            if envelope is not None:
                return ForecastReasoningProposal(
                    response=envelope.response,
                    catalog=catalog,
                    metadata=envelope.metadata,
                    cache_hit=True,
                    cache_key=cache_key,
                )

        raw = self.client.extract_structured(
            instructions=prompt,
            content=build_reasoning_content(input_value, catalog),
            response_model=ForecastReasoningResponse,
            model=self.model or None,
        )
        if inspect.isawaitable(raw):
            raw = await raw
        response = (
            raw
            if isinstance(raw, ForecastReasoningResponse)
            else ForecastReasoningResponse.model_validate(raw)
        )
        self._cache_save(identity, response, metadata)
        return ForecastReasoningProposal(
            response=response,
            catalog=catalog,
            metadata=metadata,
            cache_hit=False,
            cache_key=cache_key,
        )

    async def propose(self, input_value: ForecastReasoningInput | Any, **kwargs: Any):
        return await self.reason(input_value, **kwargs)

    async def forecast(self, input_value: ForecastReasoningInput | Any, **kwargs: Any):
        return await self.reason(input_value, **kwargs)

    extract = reason
    areason = reason

    def _cache_load(self, identity):
        loader = getattr(self.cache, "load", None) or getattr(self.cache, "get", None)
        return loader(identity) if loader is not None else None

    def _cache_save(self, identity, response, metadata):
        saver = getattr(self.cache, "save", None) or getattr(self.cache, "put", None)
        if saver is not None:
            saver(identity, response, metadata)


__all__ = [
    "PROMPT_VERSION",
    "SCHEMA_VERSION",
    "VALIDATOR_VERSION",
    "CONTEXT_VERSION",
    "FORECAST_REASONER_INSTRUCTIONS",
    "ForecastReasoningProposal",
    "ForecastReasoner",
    "build_reasoning_prompt",
    "build_reasoning_content",
]
