"""Contracts and deterministic static decomposers."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from edgarito.services.valuation.factors.contracts import (
    FactorCost,
    FactorDependencyProposal,
    FactorKey,
    FactorMateriality,
    FactorPriority,
)


class FactorDecompositionProposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    child_key: FactorKey = Field(
        validation_alias=AliasChoices("child_key", "key", "child")
    )
    relationship_role: str = Field(
        default="dependency",
        validation_alias=AliasChoices("relationship_role", "role"),
    )
    rationale: str = ""
    materiality: FactorMateriality = FactorMateriality.MEDIUM
    priority: FactorPriority = FactorPriority.NORMAL
    weight: Any = 1
    cost: FactorCost = FactorCost()
    required: bool = True

    @field_validator("materiality", mode="before")
    @classmethod
    def normalize_materiality(cls, value):
        return FactorMateriality(value)

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value):
        return FactorPriority(value)

    @field_validator("relationship_role", "rationale")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return str(value).strip()


class FactorDecomposition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parent_key: FactorKey
    proposals: tuple[FactorDecompositionProposal, ...]
    operation: str = "IDENTITY"
    rationale: str = ""
    decomposer: str = "static_mapping"
    cost: FactorCost = FactorCost()
    external_calls: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)

    @field_validator("operation")
    @classmethod
    def normalize_operation(cls, value: str) -> str:
        return value.strip().upper()

    @property
    def children(self) -> tuple[FactorDecompositionProposal, ...]:
        return self.proposals

    @property
    def dependency_proposals(self) -> tuple[FactorDecompositionProposal, ...]:
        return self.proposals


@runtime_checkable
class FactorDecomposer(Protocol):
    decomposer_id: str

    def can_decompose(self, request, context=None) -> bool: ...

    def decompose(self, request, context=None) -> FactorDecomposition: ...


def _proposal(value: Any) -> FactorDecompositionProposal:
    if isinstance(value, FactorDecompositionProposal):
        return value
    if isinstance(value, FactorDependencyProposal):
        return FactorDecompositionProposal(
            child_key=value.key,
            relationship_role=getattr(value, "relationship_role", "dependency"),
            rationale=getattr(value, "rationale", value.reason),
            materiality=getattr(value, "materiality", FactorMateriality.MEDIUM),
            priority=getattr(value, "priority", FactorPriority.NORMAL),
            weight=getattr(value, "weight", 1),
            cost=getattr(value, "cost", None) or FactorCost(),
            required=value.required,
        )
    if isinstance(value, Mapping):
        return FactorDecompositionProposal.model_validate(value)
    return FactorDecompositionProposal(child_key=value)


class StaticMappingDecomposer:
    """A side-effect-free mapping decomposer intended for tests and config."""

    decomposer_id = "static_mapping"

    def __init__(self, mapping: Mapping[Any, Any], *, operation: str = "IDENTITY"):
        self.mapping = dict(mapping)
        self.operation = operation

    @staticmethod
    def _lookup(mapping, request):
        for candidate in (request.key, request.key.digest, request.key.semantic_id):
            try:
                if candidate in mapping:
                    return mapping[candidate]
            except TypeError:
                continue
        return None

    def can_decompose(self, request, context=None) -> bool:
        return self._lookup(self.mapping, request) is not None

    def decompose(self, request, context=None) -> FactorDecomposition:
        raw = self._lookup(self.mapping, request)
        if raw is None:
            raise KeyError(request.key.digest)
        if isinstance(raw, FactorDecomposition):
            return raw.model_copy(update={"parent_key": request.key})
        operation = self.operation
        if isinstance(raw, Mapping):
            operation = raw.get("operation", operation)
            raw = raw.get("proposals", raw.get("children", ()))
        if isinstance(raw, (FactorDecompositionProposal, FactorDependencyProposal)):
            raw = (raw,)
        proposals = tuple(_proposal(item) for item in raw)
        return FactorDecomposition(
            parent_key=request.key,
            proposals=proposals,
            operation=operation,
            decomposer=self.decomposer_id,
        )


DecompositionProposal = FactorDecompositionProposal
DecompositionResult = FactorDecomposition


__all__ = [
    "FactorDecomposer",
    "FactorDecomposition",
    "FactorDecompositionProposal",
    "DecompositionProposal",
    "DecompositionResult",
    "StaticMappingDecomposer",
]
