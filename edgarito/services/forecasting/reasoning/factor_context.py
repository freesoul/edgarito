"""Opt-in ForecastReasoner boundary for resolved valuation factors.

Factor estimates are synthesized context, not executable forecast artifacts.
This module deliberately owns a separate prompt, context version, and cache
identity so importing or calling the ordinary v1 reasoner remains unchanged.
"""

from __future__ import annotations

import datetime as dt
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
    EvidenceCatalogItem,
    EvidenceCatalogProvenance,
    build_evidence_catalog,
    canonical_json,
    compact_reasoning_input,
    compact_structured,
    content_hash,
    manual_inputs_hash,
    research_hash,
)
from edgarito.services.forecasting.reasoning.reasoner import (
    FORECAST_REASONER_INSTRUCTIONS,
    ForecastReasoningProposal,
)
from edgarito.services.forecasting.reasoning.validation import (
    ForecastReasoningValidator,
)
from edgarito.services.valuation.factors.adapters.reasoning import (
    FactorAugmentedReasoningInput,
    FactorReasoningContextItem,
)
from edgarito.services.valuation.factors.contracts import FactorDomain
from edgarito.services.valuation.factors.identity import (
    canonicalize_currency,
    canonicalize_token,
    canonicalize_unit,
)

FACTOR_CONTEXT_PROMPT_VERSION = "forecast_reasoner_factor_context_v1_prompt_1"
FACTOR_CONTEXT_SCHEMA_VERSION = "forecast_reasoner_factor_context_v1_response_1"
FACTOR_CONTEXT_VALIDATOR_VERSION = "forecast_reasoner_factor_context_v1_validator_2"
FACTOR_CONTEXT_VERSION = "forecast_reasoner_factor_context_v1_context_1"


FACTOR_CONTEXT_REASONER_INSTRUCTIONS = (
    FORECAST_REASONER_INSTRUCTIONS
    + """

Resolved factor estimates are synthesized reusable context, not reported facts.
ForecastReasoner decides company-specific assumptions from that context; it does
not directly compile a factor into a ForecastOverride or OperatingObservation.
Factor records are citable only by their supplied FACTOR-<fingerprint> IDs, and
their methodology, resolver, provenance, dependencies, range, and information
date must remain visible in the rationale. Global factors remain global context;
do not turn a global factor into a company-specific reported fact. To support an
assumption, cite a company, business, or operating factor whose canonical factor
key matches the target, scope, unit, and currency. Cite external factors only
when they are canonical dependencies of that bridge factor.
"""
)


class FactorContextForecastReasoningCacheIdentity(BaseModel):
    """Versioned cache identity that cannot collide with ordinary v1 reason."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    company_id: str
    as_of: dt.date
    forecast_years: tuple[int, ...]
    evidence_bundle_hash: str
    research_hash: str
    manual_inputs_hash: str
    model: str
    reasoning_effort: str
    prompt_version: str
    schema_version: str
    validator_version: str
    context_version: str
    prompt_hash: str
    schema_hash: str
    validator_hash: str
    context_hash: str
    factor_context_hash: str
    factor_fingerprints: tuple[str, ...]

    @property
    def digest(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class FactorContextForecastReasoningCacheEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    identity: FactorContextForecastReasoningCacheIdentity
    response: ForecastReasoningResponse
    metadata: ForecastReasoningMetadata


class FactorContextForecastReasoningCache:
    """Use a factor-only namespace while reusing the supplied cache storage."""

    prefix = "forecast_reasoning_factor_context_v1"

    def __init__(
        self,
        cache: ForecastReasoningCache | FileSystemCache | str | Path | Any | None = None,
    ) -> None:
        if isinstance(cache, ForecastReasoningCache):
            cache = cache.cache
        if cache is None:
            cache = FileSystemCache(Path("cache"))
        elif isinstance(cache, (str, Path)):
            cache = FileSystemCache(cache)
        self.cache = cache

    @classmethod
    def identity(
        cls,
        input_value: ForecastReasoningInput,
        *,
        factor_context_hash: str,
        factor_fingerprints: tuple[str, ...],
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
        evidence_bundle_hash: str,
    ) -> FactorContextForecastReasoningCacheIdentity:
        return FactorContextForecastReasoningCacheIdentity(
            company_id=input_value.company_id,
            as_of=input_value.as_of,
            forecast_years=input_value.forecast_years,
            evidence_bundle_hash=evidence_bundle_hash,
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
            factor_context_hash=factor_context_hash,
            factor_fingerprints=factor_fingerprints,
        )

    key = identity
    make_identity = identity

    def path(self, identity: FactorContextForecastReasoningCacheIdentity | str) -> str:
        digest = identity if isinstance(identity, str) else identity.digest
        return f"{self.prefix}/{digest}.json"

    def load(
        self, identity: FactorContextForecastReasoningCacheIdentity | str
    ) -> FactorContextForecastReasoningCacheEnvelope | None:
        if hasattr(self.cache, "read"):
            raw = self.cache.read(self.path(identity))
            if raw is None:
                return None
            try:
                envelope = FactorContextForecastReasoningCacheEnvelope.model_validate_json(raw)
            except Exception:
                return None
        else:
            loader = getattr(self.cache, "load", None) or getattr(self.cache, "get", None)
            envelope = loader(identity) if loader is not None else None
            if envelope is None:
                return None
            if not isinstance(envelope, FactorContextForecastReasoningCacheEnvelope):
                try:
                    envelope = FactorContextForecastReasoningCacheEnvelope.model_validate(
                        envelope
                    )
                except Exception:
                    return None
        if (
            isinstance(identity, FactorContextForecastReasoningCacheIdentity)
            and envelope.identity != identity
        ):
            return None
        return envelope

    get = load

    def save(
        self,
        identity: FactorContextForecastReasoningCacheIdentity,
        response: ForecastReasoningResponse,
        metadata: ForecastReasoningMetadata,
    ) -> FactorContextForecastReasoningCacheEnvelope:
        envelope = FactorContextForecastReasoningCacheEnvelope(
            identity=identity,
            response=response,
            metadata=metadata,
        )
        if hasattr(self.cache, "read"):
            self.cache.save(self.path(identity), envelope.model_dump_json())
        else:
            saver = getattr(self.cache, "save", None) or getattr(self.cache, "put", None)
            if saver is not None:
                saver(identity, response, metadata)
        return envelope

    put = save


def _date(value: Any, label: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    raise ValueError(f"factor {label} must be a date")


def _identity_token(value: str) -> str:
    return canonicalize_token(value)


def _company_tokens(input_value: ForecastReasoningInput) -> set[str]:
    values = {input_value.company_id}
    if input_value.company_name:
        values.add(input_value.company_name)
    return {_identity_token(value) for value in values}


def _validate_scope(item: FactorReasoningContextItem, input_value: ForecastReasoningInput) -> None:
    key = item.key
    domain = key.domain
    company_tokens = _company_tokens(input_value)
    if domain in {FactorDomain.COMPANY, FactorDomain.OPERATING}:
        if _identity_token(key.subject_id) not in company_tokens:
            raise ValueError(
                f"factor {item.computation_fingerprint} has incompatible company scope"
            )
    elif domain == FactorDomain.BUSINESS:
        company, separator, business = key.subject_id.partition(":")
        if not separator or _identity_token(company) not in company_tokens:
            raise ValueError(
                f"factor {item.computation_fingerprint} has incompatible business scope"
            )
        known_segments = {
            _identity_token(segment.segment_id) for segment in input_value.segments
        }
        if known_segments and _identity_token(business) not in known_segments:
            raise ValueError(
                f"factor {item.computation_fingerprint} has unknown business scope"
            )


def _validate_item(
    item: FactorReasoningContextItem, input_value: ForecastReasoningInput
) -> None:
    key = item.key
    if item.target_period != key.period:
        raise ValueError("factor target period must match factor key period")
    if item.target_period.target_year is not None and (
        item.target_period.target_year not in input_value.forecast_years
    ):
        raise ValueError(
            f"factor {item.computation_fingerprint} has an incompatible target period"
        )
    try:
        item_unit = canonicalize_unit(item.unit)
    except (TypeError, ValueError) as exc:
        raise ValueError("factor unit is invalid") from exc
    if item_unit != key.unit:
        raise ValueError("factor unit must match factor key unit")
    item_currency = (
        canonicalize_currency(item.currency) if item.currency is not None else None
    )
    if item_currency != key.currency:
        raise ValueError("factor currency must match factor key currency")
    info_as_of = _date(item.info_as_of, "info_as_of")
    dates = tuple(item.all_availability_dates) or (info_as_of,)
    for available_on in dates:
        if _date(available_on, "availability date") > input_value.as_of:
            raise ValueError(
                f"factor {item.computation_fingerprint} is unavailable after "
                f"base as_of={input_value.as_of.isoformat()}"
            )
    if info_as_of > input_value.as_of:
        raise ValueError(
            f"factor {item.computation_fingerprint} info_as_of is after base as_of"
        )
    if len(item.evidence_refs) != len(set(item.evidence_refs)):
        raise ValueError("factor evidence IDs must be unique")
    _validate_scope(item, input_value)


def _factor_scope(
    item: FactorReasoningContextItem,
) -> tuple[str, str | None]:
    if item.key.domain == FactorDomain.BUSINESS:
        return "business", item.key.subject_id.rsplit(":", 1)[-1]
    if item.key.domain in {FactorDomain.COMPANY, FactorDomain.OPERATING}:
        return "company", "company"
    # Market, commodity, macro, regulatory, and other non-company factors are
    # deliberately left global rather than being attributed to the requester.
    return "global", "global"


def _factor_context_values(
    item: FactorReasoningContextItem, scope: str, scope_id: str | None
) -> dict[str, str]:
    key = item.key
    period = item.target_period
    values: dict[str, str] = {
        "scope": scope,
        "domain": key.domain.value,
        "subject_type": key.subject_type,
        "subject_id": key.subject_id,
        "metric": key.metric,
        "period_type": period.period_type.value,
        "period_key": period.period_key,
        "target_period": canonical_json(period.canonical),
        "factor_key": canonical_json(key.canonical),
        "confidence": item.confidence.value,
        "methodology": item.method,
        "resolver": item.resolver or "",
        "evidence_refs": canonical_json(item.evidence_refs),
        "dependencies": canonical_json(tuple(key.canonical for key in item.dependencies)),
        "dependency_fingerprints": canonical_json(item.dependency_fingerprints),
        "info_as_of": _date(item.info_as_of, "info_as_of").isoformat(),
        "availability_dates": canonical_json(
            tuple(_date(value, "availability date") for value in item.all_availability_dates)
        ),
    }
    if period.target_year is not None:
        values["target_year"] = str(period.target_year)
    for name in (
        "geography",
        "market",
        "industry",
        "product",
        "business",
        "basis",
        "qualifier",
    ):
        value = getattr(key, name)
        if value is not None:
            values[name] = str(value)
    if scope_id is not None:
        values["scope_id"] = scope_id
        if scope == "business":
            values["segment"] = scope_id
    if item.provenance is not None:
        values["provenance"] = canonical_json(item.provenance)
    return values


def build_factor_evidence_item(
    item: FactorReasoningContextItem, input_value: ForecastReasoningInput
) -> EvidenceCatalogItem:
    """Project one resolved estimate into a citable, non-executable record."""

    if not isinstance(item, FactorReasoningContextItem):
        item = FactorReasoningContextItem.model_validate(item)
    if not isinstance(input_value, ForecastReasoningInput):
        input_value = ForecastReasoningInput.model_validate(input_value)
    _validate_item(item, input_value)
    scope, scope_id = _factor_scope(item)
    fingerprint = item.computation_fingerprint
    provenance_source = None
    if item.provenance is not None:
        provenance_source = getattr(item.provenance, "source", item.provenance)
    return EvidenceCatalogItem(
        evidence_id=f"FACTOR-{fingerprint}",
        category="FACTOR",
        scope=scope,
        scope_id=scope_id,
        context=tuple(_factor_context_values(item, scope, scope_id).items()),
        fiscal_year=item.target_period.target_year,
        period=item.target_period.period_key,
        metric=item.key.metric,
        unit=item.unit,
        low=item.range.low,
        base=item.range.base,
        high=item.range.high,
        currency=item.currency,
        provenance=EvidenceCatalogProvenance(
            source_type="factor_estimate",
            source=str(provenance_source) if provenance_source is not None else None,
            source_date=_date(item.info_as_of, "info_as_of"),
            payload_type="FactorEstimate",
            payload_hash=fingerprint,
            reference=item.resolver,
        ),
        payload_identity=fingerprint,
    )


def build_factor_evidence_catalog(
    augmented_input: FactorAugmentedReasoningInput | Any,
) -> EvidenceCatalog:
    augmented_input = _coerce_augmented_input(augmented_input)
    base = augmented_input.base_input
    catalog = build_evidence_catalog(base)
    factor_items = augmented_input.factor_context.items
    seen_fingerprints: set[str] = set()
    seen_evidence_refs: set[str] = set()
    projected: list[EvidenceCatalogItem] = []
    for item in factor_items:
        _validate_item(item, base)
        fingerprint = item.computation_fingerprint
        if fingerprint in seen_fingerprints:
            raise ValueError(f"duplicate factor fingerprint: {fingerprint}")
        seen_fingerprints.add(fingerprint)
        duplicate_refs = seen_evidence_refs.intersection(item.evidence_refs)
        if duplicate_refs:
            duplicate = sorted(duplicate_refs)[0]
            raise ValueError(f"duplicate factor evidence ID: {duplicate}")
        seen_evidence_refs.update(item.evidence_refs)
        evidence_id = f"FACTOR-{fingerprint}"
        if evidence_id in catalog.evidence_ids:
            raise ValueError(f"duplicate evidence ID: {evidence_id}")
        projected.append(build_factor_evidence_item(item, base))
    return EvidenceCatalog(
        items=tuple(sorted((*catalog.items, *projected), key=lambda value: value.evidence_id)),
        exclusions=catalog.exclusions,
        duplicate_explicit_ids=catalog.duplicate_explicit_ids,
    )


def build_factor_reasoning_prompt() -> str:
    return FACTOR_CONTEXT_REASONER_INSTRUCTIONS


def build_factor_reasoning_content(
    augmented_input: FactorAugmentedReasoningInput | Any,
    catalog: EvidenceCatalog | None = None,
) -> str:
    augmented_input = _coerce_augmented_input(augmented_input)
    catalog = catalog or build_factor_evidence_catalog(augmented_input)
    return canonical_json(
        {
            "context_version": FACTOR_CONTEXT_VERSION,
            "input": compact_reasoning_input(augmented_input.base_input),
            "factor_context_hash": augmented_input.factor_context.context_hash,
            "factor_fingerprints": tuple(
                item.computation_fingerprint
                for item in augmented_input.factor_context.items
            ),
            "evidence_catalog": compact_structured(catalog.items),
        }
    )


def _coerce_augmented_input(value: FactorAugmentedReasoningInput | Any):
    return (
        value
        if isinstance(value, FactorAugmentedReasoningInput)
        else FactorAugmentedReasoningInput.model_validate(value)
    )


class FactorContextForecastReasoner:
    """Reason over a base forecast input plus validated factor context only."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        openai_client: Any | None = None,
        cache: Any | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        validator: ForecastReasoningValidator | None = None,
    ) -> None:
        self.client = client or openai_client
        if self.client is None:
            from edgarito.services.openai import OpenAIClient

            self.client = OpenAIClient()
        self.model = model or getattr(self.client, "model", "")
        self.reasoning_effort = reasoning_effort or getattr(
            self.client, "reasoning_effort", "medium"
        )
        self.cache = FactorContextForecastReasoningCache(cache)
        self.validator = validator or ForecastReasoningValidator()

    async def reason(
        self,
        augmented_input: FactorAugmentedReasoningInput | Any,
        *,
        force_refresh: bool = False,
    ) -> ForecastReasoningProposal:
        augmented_input = _coerce_augmented_input(augmented_input)
        base = augmented_input.base_input
        catalog = build_factor_evidence_catalog(augmented_input)
        prompt = build_factor_reasoning_prompt()
        schema_hash = content_hash(ForecastReasoningResponse.model_json_schema())
        prompt_hash = content_hash(prompt)
        validator_hash = content_hash(FACTOR_CONTEXT_VALIDATOR_VERSION)
        fingerprints = tuple(
            sorted(
                item.computation_fingerprint
                for item in augmented_input.factor_context.items
            )
        )
        context_hash = content_hash(
            {
                "version": FACTOR_CONTEXT_VERSION,
                "company_name": base.company_name,
                "ticker": base.ticker,
                "unit": base.unit,
                "factor_context_hash": augmented_input.factor_context.context_hash,
                "factor_fingerprints": fingerprints,
            }
        )
        metadata = ForecastReasoningMetadata(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            prompt_version=FACTOR_CONTEXT_PROMPT_VERSION,
            schema_version=FACTOR_CONTEXT_SCHEMA_VERSION,
            validator_version=FACTOR_CONTEXT_VALIDATOR_VERSION,
            context_version=FACTOR_CONTEXT_VERSION,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            validator_hash=validator_hash,
            context_hash=context_hash,
            evidence_bundle_hash=catalog.bundle_hash,
            research_hash=research_hash(base),
            manual_inputs_hash=manual_inputs_hash(base),
        )
        identity = self.cache.identity(
            base,
            factor_context_hash=augmented_input.factor_context.context_hash,
            factor_fingerprints=fingerprints,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            prompt_version=FACTOR_CONTEXT_PROMPT_VERSION,
            schema_version=FACTOR_CONTEXT_SCHEMA_VERSION,
            validator_version=FACTOR_CONTEXT_VALIDATOR_VERSION,
            context_version=FACTOR_CONTEXT_VERSION,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            validator_hash=validator_hash,
            context_hash=context_hash,
            evidence_bundle_hash=catalog.bundle_hash,
        )
        cache_key = identity.digest
        if not force_refresh:
            envelope = self.cache.load(identity)
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
            content=build_factor_reasoning_content(augmented_input, catalog),
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
        self.cache.save(identity, response, metadata)
        return ForecastReasoningProposal(
            response=response,
            catalog=catalog,
            metadata=metadata,
            cache_hit=False,
            cache_key=cache_key,
        )

    async def propose(self, augmented_input: FactorAugmentedReasoningInput | Any, **kwargs):
        return await self.reason(augmented_input, **kwargs)

    def validate(self, proposal: ForecastReasoningProposal, augmented_input: Any):
        """Run the same deterministic validator against the combined catalog."""

        augmented_input = _coerce_augmented_input(augmented_input)
        return self.validator.validate(
            proposal.response,
            augmented_input.base_input,
            proposal.catalog,
        )


FactorReasoningForecastReasoner = FactorContextForecastReasoner


__all__ = [
    "FACTOR_CONTEXT_PROMPT_VERSION",
    "FACTOR_CONTEXT_SCHEMA_VERSION",
    "FACTOR_CONTEXT_VALIDATOR_VERSION",
    "FACTOR_CONTEXT_VERSION",
    "FACTOR_CONTEXT_REASONER_INSTRUCTIONS",
    "FactorContextForecastReasoningCacheIdentity",
    "FactorContextForecastReasoningCacheEnvelope",
    "FactorContextForecastReasoningCache",
    "FactorContextForecastReasoner",
    "FactorReasoningForecastReasoner",
    "build_factor_evidence_item",
    "build_factor_evidence_catalog",
    "build_factor_reasoning_prompt",
    "build_factor_reasoning_content",
]
