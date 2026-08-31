"""Deterministic ForecastReasoner cache, isolated from extractor caches."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.forecasting.reasoning.contracts import (
    ForecastReasoningCacheIdentity,
    ForecastReasoningInput,
    ForecastReasoningMetadata,
    ForecastReasoningResponse,
)
from edgarito.services.forecasting.reasoning.evidence import (
    build_evidence_catalog,
    manual_inputs_hash,
    research_hash,
)


class ForecastReasoningCacheEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    identity: ForecastReasoningCacheIdentity
    response: ForecastReasoningResponse
    metadata: ForecastReasoningMetadata


class ForecastReasoningCache:
    """Store only strict response envelopes under a reasoner-specific prefix."""

    prefix = "forecast_reasoning_v1"

    def __init__(
        self,
        cache: FileSystemCache | str | Path | None = None,
        *,
        root_directory: str | Path | None = None,
    ) -> None:
        cache = cache if cache is not None else root_directory
        self.cache = (
            cache
            if isinstance(cache, FileSystemCache)
            else FileSystemCache(cache or Path("cache"))
        )

    @staticmethod
    def identity(
        input_value: ForecastReasoningInput,
        *,
        model: str,
        reasoning_effort: str,
        prompt_version: str,
        schema_version: str,
        validator_version: str,
        context_version: str,
        prompt_hash: str,
        schema_hash: str,
        validator_hash: str,
        context_hash: str,
    ) -> ForecastReasoningCacheIdentity:
        catalog = build_evidence_catalog(input_value)
        return ForecastReasoningCacheIdentity(
            company_id=input_value.company_id,
            as_of=input_value.as_of,
            forecast_years=input_value.forecast_years,
            evidence_bundle_hash=catalog.bundle_hash,
            research_hash=research_hash(input_value),
            manual_inputs_hash=manual_inputs_hash(input_value),
            model=model,
            reasoning_effort=reasoning_effort,
            prompt_version=prompt_version,
            schema_version=schema_version,
            validator_version=validator_version,
            context_version=context_version,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            validator_hash=validator_hash,
            context_hash=context_hash,
        )

    key = identity
    make_identity = identity

    def path(self, identity: ForecastReasoningCacheIdentity | str) -> str:
        digest = identity if isinstance(identity, str) else identity.digest
        return f"{self.prefix}/{digest}.json"

    def load(
        self, identity: ForecastReasoningCacheIdentity | str
    ) -> ForecastReasoningCacheEnvelope | None:
        raw = self.cache.read(self.path(identity))
        if raw is None:
            return None
        try:
            envelope = ForecastReasoningCacheEnvelope.model_validate_json(raw)
        except Exception:
            # A partial/corrupt cache is a miss, never a source of executable
            # assumptions.  Do not touch any extractor cache.
            return None
        if (
            isinstance(identity, ForecastReasoningCacheIdentity)
            and envelope.identity != identity
        ):
            return None
        return envelope

    def get(
        self, identity: ForecastReasoningCacheIdentity | str
    ) -> ForecastReasoningCacheEnvelope | None:
        return self.load(identity)

    def save(
        self,
        identity: ForecastReasoningCacheIdentity,
        response: ForecastReasoningResponse,
        metadata: ForecastReasoningMetadata,
    ) -> ForecastReasoningCacheEnvelope:
        envelope = ForecastReasoningCacheEnvelope(
            identity=identity,
            response=response,
            metadata=metadata,
        )
        self.cache.save(self.path(identity), envelope.model_dump_json())
        return envelope

    def put(
        self,
        identity: ForecastReasoningCacheIdentity,
        response: ForecastReasoningResponse,
        metadata: ForecastReasoningMetadata,
    ) -> ForecastReasoningCacheEnvelope:
        return self.save(identity, response, metadata)


__all__ = ["ForecastReasoningCache", "ForecastReasoningCacheEnvelope"]
