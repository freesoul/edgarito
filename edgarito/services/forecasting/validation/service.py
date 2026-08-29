"""Public service for deterministic, read-only forecast validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .config import ForecastValidationConfig
from .contracts import (
    ForecastValidationContext,
    ForecastValidationResult,
)
from .registry import RuleRegistry


class ForecastValidationService:
    """Run independent checks over an already-produced forecast artifact."""

    def __init__(
        self,
        config: ForecastValidationConfig | Mapping[str, Any] | None = None,
        registry: RuleRegistry | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, ForecastValidationConfig)
            else ForecastValidationConfig.model_validate(config or {})
        )
        self.registry = registry or RuleRegistry.default()

    def validate(self, artifact: Any) -> ForecastValidationResult:
        context = ForecastValidationContext.from_artifact(artifact)
        findings = tuple(
            finding
            for rule in self.registry.rules
            for finding in rule.evaluate(context, self.config)
        )
        return ForecastValidationResult(findings=findings)

    check = validate
    run = validate


ForecastSanityCheckService = ForecastValidationService


__all__ = ["ForecastSanityCheckService", "ForecastValidationService"]
