"""Resolver protocol and typed resolver results."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgarito.services.valuation.factors.contracts import (
    FactorCost,
    FactorEstimate,
    FactorResolutionStatus,
)


class ResolverResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimate: Optional[FactorEstimate] = None
    unresolved: bool = False
    status: FactorResolutionStatus = FactorResolutionStatus.UNRESOLVED
    reason: Optional[str] = None
    cost: FactorCost = FactorCost()
    external_calls: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    warnings: tuple[str, ...] = ()

    @field_validator("cost", mode="before")
    @classmethod
    def normalize_cost(cls, value):
        if isinstance(value, (Decimal, int, float, str)):
            return FactorCost(amount=Decimal(str(value)))
        return value

    @property
    def resolution_cost(self) -> Decimal:
        return self.cost.amount

    @model_validator(mode="after")
    def normalize_result(self) -> "ResolverResult":
        if self.estimate is not None:
            object.__setattr__(self, "unresolved", False)
            if self.status == FactorResolutionStatus.UNRESOLVED:
                object.__setattr__(self, "status", FactorResolutionStatus.DIRECTLY_RESOLVED)
        else:
            object.__setattr__(self, "unresolved", True)
            if self.status not in {
                FactorResolutionStatus.STOPPED,
                FactorResolutionStatus.FAILED,
            }:
                object.__setattr__(self, "status", FactorResolutionStatus.UNRESOLVED)
        return self

    @property
    def resolved(self) -> bool:
        return self.estimate is not None and not self.unresolved

    @classmethod
    def success(cls, estimate: FactorEstimate, **kwargs: Any) -> "ResolverResult":
        return cls(estimate=estimate, status=FactorResolutionStatus.DIRECTLY_RESOLVED, **kwargs)

    @classmethod
    def missing(cls, reason: Optional[str] = None, **kwargs: Any) -> "ResolverResult":
        return cls(unresolved=True, reason=reason, **kwargs)


@runtime_checkable
class FactorResolver(Protocol):
    resolver_id: str

    def can_resolve(self, request, context=None) -> bool: ...

    def resolve(self, request, context=None, **kwargs: Any) -> ResolverResult: ...


FactorResolverResult = ResolverResult
ResolutionResult = ResolverResult


__all__ = [
    "FactorResolver",
    "FactorResolverResult",
    "ResolutionResult",
    "ResolverResult",
]
