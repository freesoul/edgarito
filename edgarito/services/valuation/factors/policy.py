"""Explicit limits for recursive factor expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from edgarito.services.valuation.factors.contracts import (
    FactorCost,
    FactorMateriality,
    FactorRequest,
    StopReason,
)
from edgarito.services.valuation.factors.identity import canonicalize_currency

_MATERIALITY_RANK = {
    FactorMateriality.UNKNOWN: -1,
    FactorMateriality.IMMATERIAL: 0,
    FactorMateriality.LOW: 1,
    FactorMateriality.MATERIAL: 2,
    FactorMateriality.MEDIUM: 3,
    FactorMateriality.HIGH: 4,
    FactorMateriality.CRITICAL: 5,
}


class FactorResolutionMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    resolver_calls: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    external_calls: int = Field(default=0, ge=0)
    resolution_cost: Decimal = Field(default=Decimal("0"), ge=0)
    elapsed_seconds: Optional[float] = Field(default=None, ge=0)

    @property
    def cost(self) -> Decimal:
        return self.resolution_cost

    @property
    def total_cost(self) -> Decimal:
        return self.resolution_cost

    @property
    def resolution_calls(self) -> int:
        return self.resolver_calls


class FactorExpansionDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    reason: Optional[StopReason] = None
    detail: Optional[str] = None

    @property
    def should_expand(self) -> bool:
        return self.allowed

    @property
    def stop_reason(self) -> Optional[StopReason]:
        return self.reason


@dataclass
class _Usage:
    resolver_calls: int = 0
    model_calls: int = 0
    external_calls: int = 0
    resolution_cost: Decimal = field(default_factory=lambda: Decimal("0"))

    def metrics(self, elapsed_seconds: Optional[float] = None) -> FactorResolutionMetrics:
        return FactorResolutionMetrics(
            resolver_calls=self.resolver_calls,
            model_calls=self.model_calls,
            external_calls=self.external_calls,
            resolution_cost=self.resolution_cost,
            elapsed_seconds=elapsed_seconds,
        )

    def add_cost(
        self,
        cost: Optional[FactorCost],
        *,
        expected_currency: str = "USD",
    ) -> None:
        if cost is None:
            return
        if cost.currency != expected_currency:
            raise ValueError(
                f"cost currency {cost.currency} does not match "
                f"configured cost currency {expected_currency}"
            )
        self.resolution_cost += cost.amount

    def record_resolver_call(self) -> None:
        self.resolver_calls += 1

    def add(
        self,
        cost: Optional[FactorCost],
        *,
        external=0,
        model=0,
        expected_currency: str = "USD",
        count_resolver: bool = True,
    ) -> None:
        if count_resolver:
            self.record_resolver_call()
        self.external_calls += int(external)
        self.model_calls += int(model)
        self.add_cost(cost, expected_currency=expected_currency)


class FactorExpansionPolicy(BaseModel):
    """Immutable policy; the service keeps usage separately per resolution."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    max_depth: int = Field(default=4, ge=0)
    max_nodes: Optional[int] = Field(default=None, ge=0)
    max_resolution_cost: Optional[Decimal] = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("max_resolution_cost", "max_cost"),
    )
    max_external_calls: Optional[int] = Field(default=None, ge=0)
    max_model_calls: Optional[int] = Field(default=None, ge=0)
    max_attempts: Optional[int] = Field(default=None, ge=0)
    cost_currency: str = "USD"
    min_materiality: FactorMateriality = FactorMateriality.LOW

    @field_validator("min_materiality", mode="before")
    @classmethod
    def normalize_materiality(cls, value):
        return FactorMateriality(value)

    @field_validator("cost_currency")
    @classmethod
    def normalize_cost_currency(cls, value: str) -> str:
        return canonicalize_currency(value)

    def decision(
        self,
        request: Optional[FactorRequest] = None,
        *,
        depth: int = 0,
        nodes: int = 0,
        resolution_cost: Decimal = Decimal("0"),
        external_calls: int = 0,
        model_calls: int = 0,
        resolver_calls: int = 0,
        materiality: Optional[FactorMateriality] = None,
    ) -> FactorExpansionDecision:
        budget = request.budget if request is not None else None
        effective_depth = self._minimum(
            self._minimum(
                self.max_depth, request.max_depth if request is not None else None
            ),
            budget.max_depth if budget else None,
        )
        effective_nodes = self._minimum(self.max_nodes, budget.max_nodes if budget else None)
        effective_external = self._minimum(
            self.max_external_calls, budget.max_external_calls if budget else None
        )
        effective_model = self._minimum(
            self.max_model_calls, budget.max_model_calls if budget else None
        )
        effective_attempts = self._minimum(
            self.max_attempts, budget.max_attempts if budget else None
        )
        budget_cost = None
        if budget is not None:
            budget_cost = (
                budget.max_resolution_cost
                if budget.max_resolution_cost is not None
                else budget.max_cost
            )
        effective_cost = self._minimum_decimal(self.max_resolution_cost, budget_cost)
        requested_materiality = materiality or (
            request.materiality if request is not None else FactorMateriality.MEDIUM
        )
        if request is not None and request.remaining_depth == 0:
            return FactorExpansionDecision(
                allowed=False,
                reason=StopReason.MAX_DEPTH,
                detail="remaining depth exhausted",
            )
        if _MATERIALITY_RANK[FactorMateriality(requested_materiality)] < _MATERIALITY_RANK[
            self.min_materiality
        ]:
            return FactorExpansionDecision(
                allowed=False,
                reason=StopReason.BELOW_MATERIALITY,
                detail=f"materiality below {self.min_materiality.value}",
            )
        if depth >= effective_depth:
            return FactorExpansionDecision(
                allowed=False,
                reason=StopReason.MAX_DEPTH,
                detail=f"depth {depth} reached limit {effective_depth}",
            )
        if effective_nodes is not None and nodes >= effective_nodes:
            return FactorExpansionDecision(
                allowed=False,
                reason=StopReason.MAX_NODES,
                detail=f"node count {nodes} reached limit {effective_nodes}",
            )
        if effective_cost is not None and resolution_cost > effective_cost:
            return FactorExpansionDecision(
                allowed=False,
                reason=StopReason.MAX_RESOLUTION_COST,
                detail="resolution cost budget exhausted",
            )
        if (
            effective_attempts is not None
            and resolver_calls >= effective_attempts
        ):
            return FactorExpansionDecision(
                allowed=False,
                reason=StopReason.BUDGET_EXHAUSTED,
                detail="resolver-attempt budget exhausted",
            )
        if effective_external is not None and external_calls >= effective_external:
            return FactorExpansionDecision(
                allowed=False,
                reason=StopReason.MAX_EXTERNAL_CALLS,
                detail="external-call budget exhausted",
            )
        if effective_model is not None and model_calls >= effective_model:
            return FactorExpansionDecision(
                allowed=False,
                reason=StopReason.BUDGET_EXHAUSTED,
                detail="model-call budget exhausted",
            )
        return FactorExpansionDecision(allowed=True)

    def attempt_decision(
        self,
        request: Optional[FactorRequest] = None,
        *,
        resolver_calls: int = 0,
    ) -> FactorExpansionDecision:
        """Check the limit that applies immediately before invoking a resolver."""

        budget = request.budget if request is not None else None
        effective_attempts = self._minimum(
            self.max_attempts, budget.max_attempts if budget else None
        )
        if effective_attempts is not None and resolver_calls >= effective_attempts:
            return FactorExpansionDecision(
                allowed=False,
                reason=StopReason.BUDGET_EXHAUSTED,
                detail="resolver-attempt budget exhausted",
            )
        return FactorExpansionDecision(allowed=True)

    def can_expand(self, *args, **kwargs) -> bool:
        return self.decision(*args, **kwargs).allowed

    def evaluate(self, *args, **kwargs) -> FactorExpansionDecision:
        return self.decision(*args, **kwargs)

    def check(self, *args, **kwargs) -> FactorExpansionDecision:
        return self.decision(*args, **kwargs)

    def stop_reason(self, *args, **kwargs) -> Optional[StopReason]:
        return self.decision(*args, **kwargs).reason

    @staticmethod
    def _minimum(first, second):
        if first is None:
            return second
        if second is None:
            return first
        return min(first, second)

    @staticmethod
    def _minimum_decimal(first, second):
        if first is None:
            return second
        if second is None:
            return first
        return min(first, second)


__all__ = [
    "ExpansionDecision",
    "ExpansionPolicy",
    "FactorExpansionDecision",
    "FactorExpansionPolicy",
    "FactorResolutionMetrics",
    "_Usage",
]


ExpansionPolicy = FactorExpansionPolicy
ExpansionDecision = FactorExpansionDecision
