"""Opt-in context projection for factor-aware reasoning experiments.

This module deliberately wraps, rather than modifies, ForecastReasoner v1.
Factor estimates are explanatory context here; they are not executable forecast
overrides and do not enter the existing prompt or cache identity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from edgarito.services.forecasting.reasoning.contracts import ForecastReasoningInput
from edgarito.services.valuation.factors.contracts import (
    FactorConfidence,
    FactorEstimate,
    FactorKey,
    FactorPeriod,
    FactorProvenance,
    FactorRange,
)
from edgarito.services.valuation.factors.identity import canonical_json, stable_digest


class FactorReasoningContextItem(BaseModel):
    """The auditable, non-executable projection of one FactorEstimate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: FactorKey
    range: FactorRange = Field(
        validation_alias=AliasChoices("range", "factor_range", "estimate_range")
    )
    method: str = Field(
        validation_alias=AliasChoices("method", "methodology")
    )
    confidence: FactorConfidence
    evidence_refs: tuple[str, ...] = ()
    dependencies: tuple[FactorKey, ...] = ()
    dependency_fingerprints: tuple[tuple[str, str], ...] = ()
    provenance: FactorProvenance | str | None = None
    info_as_of: Any = None
    target_period: FactorPeriod
    unit: str
    currency: str | None = None
    resolver: str | None = None
    # Keep the computation identity and every availability boundary when a
    # resolved estimate crosses into a reasoning request.  The old context
    # projection intentionally omitted these fields because it was not yet a
    # provider boundary; factor-aware reasoning needs them to reject stale
    # context before a model call.
    fingerprint: str | None = None
    all_availability_dates: tuple[Any, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "all_availability_dates", "availability_dates"
        ),
    )
    version: int = 1

    @field_validator("method")
    @classmethod
    def require_method(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("factor reasoning method cannot be blank")
        return value

    @property
    def methodology(self) -> str:
        return self.method

    @property
    def identity(self) -> FactorKey:
        return self.key

    @property
    def factor_range(self) -> FactorRange:
        return self.range

    @property
    def evidence(self) -> tuple[str, ...]:
        return self.evidence_refs

    @property
    def dependency_keys(self) -> tuple[FactorKey, ...]:
        return self.dependencies

    @classmethod
    def from_estimate(cls, estimate: FactorEstimate) -> "FactorReasoningContextItem":
        if not isinstance(estimate, FactorEstimate):
            estimate = FactorEstimate.model_validate(estimate)
        return cls(
            key=estimate.key,
            range=estimate.range,
            method=estimate.methodology,
            confidence=estimate.confidence,
            evidence_refs=estimate.evidence_refs,
            dependencies=estimate.dependencies,
            dependency_fingerprints=estimate.dependency_fingerprints,
            provenance=estimate.source,
            info_as_of=estimate.info_as_of,
            target_period=estimate.target_period,
            unit=estimate.unit,
            currency=estimate.currency,
            resolver=estimate.resolver,
            fingerprint=estimate.fingerprint,
            all_availability_dates=estimate.all_availability_dates,
            version=estimate.version,
        )

    @property
    def compact_payload(self) -> dict[str, Any]:
        return {
            "key": self.key.canonical,
            "range": self.range.model_dump(mode="python"),
            "method": self.method,
            "confidence": self.confidence.value,
            "evidence_refs": self.evidence_refs,
            "dependencies": tuple(key.canonical for key in self.dependencies),
            "dependency_fingerprints": self.dependency_fingerprints,
            "provenance": self.provenance,
            "info_as_of": self.info_as_of,
            "target_period": self.target_period.canonical,
            "unit": self.unit,
            "currency": self.currency,
            "resolver": self.resolver,
            "fingerprint": self.fingerprint,
            "all_availability_dates": self.all_availability_dates,
            "version": self.version,
        }

    @property
    def availability_dates(self) -> tuple[Any, ...]:
        return self.all_availability_dates

    @property
    def computation_fingerprint(self) -> str:
        """Return the originating estimate fingerprint when available."""

        return self.fingerprint or stable_digest(self.compact_payload)


class FactorReasoningContext(BaseModel):
    """Compact deterministic context that can be shown to a future reasoner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[FactorReasoningContextItem, ...] = ()

    @classmethod
    def from_estimates(
        cls,
        estimates: FactorEstimate | Iterable[FactorEstimate] | Mapping[Any, FactorEstimate],
    ) -> "FactorReasoningContext":
        if isinstance(estimates, FactorEstimate):
            values = (estimates,)
        elif isinstance(estimates, Mapping):
            values = tuple(estimates.values())
        else:
            values = tuple(estimates)
        items = tuple(
            sorted(
                (FactorReasoningContextItem.from_estimate(item) for item in values),
                key=lambda item: (
                    item.key.digest,
                    item.key.semantic_id,
                    canonical_json(item.compact_payload),
                ),
            )
        )
        return cls(items=items)

    from_factor_estimates = from_estimates

    @property
    def compact_payload(self) -> tuple[dict[str, Any], ...]:
        return tuple(item.compact_payload for item in self.items)

    @property
    def compact(self) -> str:
        return canonical_json(self.compact_payload)

    @property
    def context_hash(self) -> str:
        return stable_digest(self.compact_payload)

    @property
    def hash(self) -> str:
        return self.context_hash

    @property
    def digest(self) -> str:
        return self.context_hash

    @property
    def compact_context(self) -> str:
        return self.compact

    @property
    def estimates(self) -> tuple[FactorReasoningContextItem, ...]:
        return self.items

    @property
    def fingerprints(self) -> tuple[str, ...]:
        return tuple(item.computation_fingerprint for item in self.items)


class FactorAugmentedReasoningInput(BaseModel):
    """A non-invasive wrapper around existing ForecastReasoningInput."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    input: ForecastReasoningInput = Field(
        validation_alias=AliasChoices("input", "base_input", "forecast_input")
    )
    factor_context: FactorReasoningContext = Field(default_factory=FactorReasoningContext)

    @property
    def base_input(self) -> ForecastReasoningInput:
        return self.input

    @property
    def forecast_input(self) -> ForecastReasoningInput:
        return self.input

    @property
    def manual_overrides(self):
        """Expose only the original overrides; factors are never compiled into them."""

        return self.input.manual_overrides

    @property
    def context_hash(self) -> str:
        return self.factor_context.context_hash

    @property
    def reasoning_context(self) -> FactorReasoningContext:
        return self.factor_context

    def to_forecast_reasoning_input(self) -> ForecastReasoningInput:
        return self.input


class ResolvedFactorReasoningAdapter:
    """Build factor context without changing ForecastReasoner contracts."""

    def to_context(
        self,
        estimates: FactorEstimate | Iterable[FactorEstimate] | Mapping[Any, FactorEstimate],
    ) -> FactorReasoningContext:
        return FactorReasoningContext.from_estimates(estimates)

    adapt = to_context
    to_reasoning_context = to_context

    def augment(
        self,
        input_value: ForecastReasoningInput,
        estimates: FactorReasoningContext
        | FactorEstimate
        | Iterable[FactorEstimate]
        | Mapping[Any, FactorEstimate],
    ) -> FactorAugmentedReasoningInput:
        if not isinstance(input_value, ForecastReasoningInput):
            input_value = ForecastReasoningInput.model_validate(input_value)
        context = (
            estimates
            if isinstance(estimates, FactorReasoningContext)
            else self.to_context(estimates)
        )
        return FactorAugmentedReasoningInput(input=input_value, factor_context=context)

    wrap = augment

    def to_research_evidence(
        self,
        estimate: FactorEstimate,
        *,
        explicit: bool = False,
        explicitly_requested: bool | None = None,
        evidence: Iterable[Any] = (),
    ) -> tuple[Any, ...]:
        """Return supplied raw research only after an explicit opt-in.

        A computed factor does not contain enough semantic coordinates to infer
        a market-evidence kind.  Therefore this seam accepts an already typed,
        semantically identical source observation rather than inventing one.
        """

        if explicitly_requested is not None:
            explicit = explicitly_requested
        if not explicit:
            return ()
        values = tuple(evidence)
        if not values:
            raise ValueError(
                "explicit research conversion requires semantically identical source evidence"
            )
        from edgarito.services.valuation.factors.adapters.research import (
            ResearchFactorAdapter,
        )

        for item in values:
            try:
                converted = ResearchFactorAdapter().to_factor_evidence(
                    item,
                    target_key=estimate.key,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "research source evidence is not semantically identical to factor"
                ) from exc
            if converted.range != estimate.range:
                raise ValueError("research source evidence does not match factor range")
        return values

    map_to_research_evidence = to_research_evidence


__all__ = [
    "FactorAugmentedReasoningInput",
    "FactorReasoningContext",
    "FactorReasoningContextItem",
    "ResolvedFactorReasoningAdapter",
]
