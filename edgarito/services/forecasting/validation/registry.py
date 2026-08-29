"""Small immutable rule protocol and registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from .config import ForecastValidationConfig
from .contracts import ForecastValidationContext, ForecastValidationFinding
from .rules.compounding import CompoundingScaleRule
from .rules.consistency import CrossMetricConsistencyRule
from .rules.growth import FcffGrowthRule
from .rules.horizon import HorizonIntegrityRule
from .rules.margins import OperatingMarginRule, RevenueFcffMarginRule
from .rules.reinvestment import ReinvestmentRule
from .rules.terminal import (
    TerminalDiscontinuityRule,
    TerminalEconomicConsistencyRule,
    TerminalValueRule,
)
from .rules.working_capital import WorkingCapitalRule


class ValidationRule(Protocol):
    """Protocol implemented by one deterministic validation rule."""

    name: str

    def evaluate(
        self,
        context: ForecastValidationContext,
        config: ForecastValidationConfig,
    ) -> tuple[ForecastValidationFinding, ...]: ...


@dataclass(frozen=True)
class RuleRegistry:
    """An ordered, immutable collection of validation rules."""

    rules: tuple[ValidationRule, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        names = [rule.name for rule in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("Validation rule names must be unique")

    @classmethod
    def default(cls) -> "RuleRegistry":
        return cls(
            rules=(
                HorizonIntegrityRule(),
                FcffGrowthRule(),
                RevenueFcffMarginRule(),
                OperatingMarginRule(),
                ReinvestmentRule(),
                WorkingCapitalRule(),
                TerminalValueRule(),
                TerminalEconomicConsistencyRule(),
                TerminalDiscontinuityRule(),
                CompoundingScaleRule(),
                CrossMetricConsistencyRule(),
            )
        )

    def register(self, rule: ValidationRule) -> "RuleRegistry":
        """Return a registry with ``rule`` appended; do not mutate this one."""

        if rule.name in {item.name for item in self.rules}:
            raise ValueError(f"Validation rule {rule.name!r} is already registered")
        return RuleRegistry((*self.rules, rule))

    def extend(self, rules: Iterable[ValidationRule]) -> "RuleRegistry":
        result = self
        for rule in rules:
            result = result.register(rule)
        return result


DEFAULT_RULE_REGISTRY = RuleRegistry.default()


__all__ = ["DEFAULT_RULE_REGISTRY", "RuleRegistry", "ValidationRule"]
