"""Deterministic FCFF method planning.

This module only resolves a method and describes the decisions that an
executor should make.  It deliberately does not retrieve evidence, perform
forecast arithmetic, or know about the command line.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from typing import Any, Callable

from edgarito.schemas.forecasting import (
    FcffForecastMethod,
    ForecastDecision,
    ForecastMetric,
    ForecastOverride,
    ForecastPlan,
    ForecastScope,
    ForecastStrategy,
)

_CONSOLIDATED_COMPANY_METRICS = (
    ForecastMetric.REVENUE,
    ForecastMetric.OPERATING_MARGIN,
    ForecastMetric.TAX,
    ForecastMetric.DEPRECIATION_AND_AMORTIZATION,
    ForecastMetric.CAPEX,
    ForecastMetric.OPERATING_WORKING_CAPITAL,
    ForecastMetric.DELTA_NWC,
)
_FUTURE_DRIVER_BASED_COMPANY_METRICS = (
    ForecastMetric.GROSS_MARGIN,
    ForecastMetric.GROSS_PROFIT,
    ForecastMetric.R_AND_D,
    ForecastMetric.SG_AND_A,
)
_DRIVER_BASED_COMPANY_METRICS = (
    *_CONSOLIDATED_COMPANY_METRICS,
    *_FUTURE_DRIVER_BASED_COMPANY_METRICS,
)
_DRIVER_BASED_DRIVER_METRICS = frozenset(
    {
        ForecastMetric.R_AND_D,
        ForecastMetric.SG_AND_A,
        ForecastMetric.TAX,
        ForecastMetric.DEPRECIATION_AND_AMORTIZATION,
        ForecastMetric.CAPEX,
        ForecastMetric.OPERATING_WORKING_CAPITAL,
        ForecastMetric.DELTA_NWC,
        ForecastMetric.GROSS_MARGIN,
        ForecastMetric.GROSS_PROFIT,
    }
)


class FcffForecastPlanService:
    """Resolve FCFF methods and produce an auditable decision plan.

    ``quality_gate`` and ``capability`` are optional, provider-neutral
    callables.  Supplying them lets an application reuse its existing
    operating activation assessment without making this planner depend on a
    provider or on the FCFF implementation.
    """

    def __init__(
        self,
        *,
        quality_gate: Callable[..., Any] | None = None,
        capability: Callable[..., Any] | None = None,
        operating_quality_gate: Callable[..., Any] | None = None,
        operating_capability: Callable[..., Any] | None = None,
    ) -> None:
        self.quality_gate = quality_gate or operating_quality_gate
        self.capability = capability or operating_capability

    def plan(
        self,
        requested_method: FcffForecastMethod | str = FcffForecastMethod.NORMALIZED,
        *,
        requested: FcffForecastMethod | str | None = None,
        default_method: FcffForecastMethod | str = FcffForecastMethod.NORMALIZED,
        method: FcffForecastMethod | str | None = None,
        quality_gate: Callable[..., Any] | None = None,
        evidence: Any = None,
        operating_evidence: Any = None,
        operating_quality: Any = None,
        quality: Any = None,
        current_operating_quality: Any = None,
        operating_result: Any = None,
        operating_capability: Any = None,
        capability: Any = None,
        current_operating_capability: Any = None,
        overrides: Iterable[ForecastOverride] | Mapping[Any, Any] | None = None,
        manual_overrides: Iterable[ForecastOverride] | Mapping[Any, Any] | None = None,
    ) -> ForecastPlan:
        """Return a concrete plan without evaluating any forecast values."""

        requested_value = (
            default_method if requested_method is None else requested_method
        )
        if requested is not None:
            requested_value = requested
        if method is not None:
            requested_value = method
        requested = FcffForecastMethod(requested_value)
        evidence = operating_evidence if operating_evidence is not None else evidence
        supplied_quality = current_operating_quality
        if supplied_quality is None:
            supplied_quality = (
                operating_quality if operating_quality is not None else quality
            )
        if supplied_quality is None and operating_result is not None:
            embedded_quality = _value(operating_result, "quality")
            supplied_quality = (
                embedded_quality if embedded_quality is not None else operating_result
            )
        supplied_capability = current_operating_capability
        if supplied_capability is None:
            supplied_capability = (
                operating_capability if operating_capability is not None else capability
            )
        override_values = (
            manual_overrides if manual_overrides is not None else overrides
        )
        normalized_overrides = _normalize_overrides(override_values)

        quality_result = self._quality_assessment(
            evidence, supplied_quality, quality_gate=quality_gate
        )
        if quality_result is None:
            selected_capability = supplied_capability
            if selected_capability is None and self.capability is not None:
                selected_capability = self.capability
            if callable(selected_capability):
                selected_capability = _call_assessment(selected_capability, evidence)
            quality_result = selected_capability

        if requested == FcffForecastMethod.AUTO:
            accepted = _assessment_accepted(quality_result)
            resolved = (
                FcffForecastMethod.HYBRID if accepted else FcffForecastMethod.NORMALIZED
            )
            rationale = _auto_rationale(resolved, quality_result)
            confidence = _assessment_confidence(quality_result, "medium")
        else:
            resolved = requested
            rationale = (
                f"Explicit FCFF forecast method '{resolved.value}' was requested; "
                "method resolution did not reinterpret it"
            )
            confidence = (
                "low" if resolved == FcffForecastMethod.DRIVER_BASED else "medium"
            )

        decisions = self._decisions_for(
            resolved,
            evidence,
            normalized_overrides,
        )
        warnings: list[str] = []
        audit = [
            f"requested_method={requested.value}",
            f"resolved_method={resolved.value}",
        ]
        if quality_result is not None:
            reason = _assessment_reason(quality_result)
            audit.append(
                "operating_quality="
                + ("accepted" if _assessment_accepted(quality_result) else "rejected")
                + (f": {reason}" if reason else "")
            )
            warnings.extend(_assessment_warnings(quality_result))
        if (
            requested == FcffForecastMethod.AUTO
            and resolved != FcffForecastMethod.HYBRID
        ):
            warnings.append(
                "AUTO did not activate HYBRID operating forecasting; normalized FCFF "
                "was selected"
            )
        if resolved == FcffForecastMethod.DRIVER_BASED:
            warnings.append(
                "driver_based is representable in the plan but execution is not "
                "implemented"
            )
        if normalized_overrides:
            audit.append(f"manual_override_count={len(normalized_overrides)}")
        audit.append(f"decision_count={len(decisions)}")

        return ForecastPlan(
            requested=requested,
            resolved=resolved,
            decisions=decisions,
            overrides=normalized_overrides,
            rationale=rationale,
            warnings=tuple(dict.fromkeys(warnings)),
            audit=tuple(audit),
            confidence=confidence,
        )

    def _quality_assessment(
        self,
        evidence: Any,
        supplied: Any,
        *,
        quality_gate: Callable[..., Any] | None = None,
    ) -> Any:
        if supplied is not None:
            return supplied
        if evidence is None:
            return None
        embedded = _value(evidence, "quality")
        if embedded is not None:
            return embedded
        selected_gate = quality_gate or self.quality_gate
        if selected_gate is not None:
            return _call_assessment(selected_gate, evidence)
        # A raw evidence payload is not itself a quality assessment.  The
        # orchestration/application layer injects the existing operating gate;
        # without it AUTO conservatively remains normalized.
        return None

    def _decisions_for(
        self,
        method: FcffForecastMethod,
        evidence: Any,
        overrides: tuple[ForecastOverride, ...],
    ) -> tuple[ForecastDecision, ...]:
        if method == FcffForecastMethod.DRIVER_BASED:
            decisions = [
                ForecastDecision(
                    scope=ForecastScope.COMPANY,
                    metric=metric,
                    strategy=(
                        ForecastStrategy.DRIVER
                        if metric in _DRIVER_BASED_DRIVER_METRICS
                        else ForecastStrategy.CONSOLIDATED
                    ),
                    rationale="Driver-based FCFF planning decision",
                    confidence="low",
                )
                for metric in _DRIVER_BASED_COMPANY_METRICS
            ]
            for segment_id in _segment_ids(evidence):
                decisions.extend(
                    ForecastDecision(
                        scope=ForecastScope.SEGMENT,
                        scope_id=segment_id,
                        metric=metric,
                        strategy=ForecastStrategy.DRIVER,
                        rationale=(
                            "Driver-based planning represents segment revenue and "
                            "gross-profit economics"
                        ),
                        confidence="low",
                    )
                    for metric in (
                        ForecastMetric.REVENUE,
                        ForecastMetric.GROSS_MARGIN,
                        ForecastMetric.GROSS_PROFIT,
                    )
                )
        else:
            decisions = [
                ForecastDecision(
                    scope=ForecastScope.COMPANY,
                    metric=metric,
                    strategy=ForecastStrategy.CONSOLIDATED,
                    rationale=(
                        "Normalized consolidated FCFF decision"
                        if method == FcffForecastMethod.NORMALIZED
                        else "Hybrid company FCFF decision remains consolidated"
                    ),
                    confidence="medium",
                )
                for metric in _CONSOLIDATED_COMPANY_METRICS
            ]
            if method == FcffForecastMethod.HYBRID:
                decisions.extend(
                    ForecastDecision(
                        scope=ForecastScope.SEGMENT,
                        scope_id=segment_id,
                        metric=ForecastMetric.REVENUE,
                        strategy=ForecastStrategy.DRIVER,
                        rationale="Hybrid operating evidence supplies segment revenue",
                        confidence="medium",
                    )
                    for segment_id in _segment_ids(evidence)
                )

        by_key = {_record_key(item): item for item in decisions}
        for override in overrides:
            by_key[override.key] = ForecastDecision(
                scope=override.scope,
                scope_id=override.scope_id,
                metric=override.metric,
                strategy=override.strategy,
                explicit_path=override.explicit_path,
                basis=override.basis,
                provenance=override.provenance,
                rationale="Manual FCFF forecast override takes precedence",
                confidence="high",
            )
        return tuple(sorted(by_key.values(), key=_record_key))

    # Common spellings retained for programmatic callers.
    build = plan
    create = plan
    create_plan = plan
    build_plan = plan
    resolve = plan
    resolve_plan = plan
    forecast_plan = plan


def _normalize_overrides(
    values: Iterable[ForecastOverride] | Mapping[Any, Any] | None,
) -> tuple[ForecastOverride, ...]:
    if values is None:
        return ()
    if isinstance(values, Mapping):
        records = []
        for key, value in values.items():
            if isinstance(value, ForecastOverride):
                records.append(value)
                continue
            payload = dict(value) if isinstance(value, Mapping) else {}
            if isinstance(key, tuple):
                if len(key) == 2:
                    payload.setdefault("scope", key[0])
                    payload.setdefault("metric", key[1])
                elif len(key) == 3:
                    payload.setdefault("scope", key[0])
                    payload.setdefault("scope_id", key[1])
                    payload.setdefault("metric", key[2])
            elif isinstance(key, str) and ":" in key:
                parts = key.split(":")
                if len(parts) == 2:
                    payload.setdefault("scope", parts[0])
                    payload.setdefault("metric", parts[1])
                elif len(parts) == 3:
                    payload.setdefault("scope", parts[0])
                    payload.setdefault("scope_id", parts[1])
                    payload.setdefault("metric", parts[2])
            records.append(payload)
    elif isinstance(values, (str, bytes)):
        records = (values,)
    else:
        records = tuple(values)
    return tuple(
        sorted(
            (ForecastOverride.model_validate(item) for item in records), key=_record_key
        )
    )


def _segment_ids(evidence: Any) -> tuple[str, ...]:
    segments = _value(evidence, "segments") or ()
    result = set()
    for segment in segments:
        segment_id = _value(segment, "segment_id")
        if segment_id:
            result.add(str(segment_id).strip())
    return tuple(sorted(result))


def _record_key(value: ForecastDecision | ForecastOverride) -> tuple[str, str, str]:
    metric = (
        value.metric.value
        if isinstance(value.metric, ForecastMetric)
        else str(value.metric)
    )
    return (value.scope.value, value.scope_id, metric)


def _value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _call_assessment(callable_: Callable[..., Any], evidence: Any) -> Any:
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return callable_(evidence)
    positional = tuple(
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    accepts_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if not positional and not accepts_varargs:
        return callable_()
    return callable_(evidence)


def _assessment_accepted(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(value.get("accepted", value.get("available", False)))
    return bool(
        getattr(
            value,
            "accepted",
            getattr(
                value,
                "available",
                getattr(
                    value,
                    "supported",
                    getattr(
                        value,
                        "operating_capable",
                        getattr(
                            value,
                            "can_forecast",
                            getattr(
                                value,
                                "can_activate",
                                getattr(
                                    value, "enabled", getattr(value, "usable", False)
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )


def _assessment_reason(value: Any) -> str:
    if isinstance(value, Mapping):
        reason = value.get("reason", "")
    else:
        reason = getattr(value, "reason", "")
    return str(reason).strip() if reason else ""


def _assessment_confidence(value: Any, default: str) -> str:
    if isinstance(value, Mapping):
        confidence = value.get("confidence")
    else:
        confidence = getattr(value, "confidence", None)
    normalized = str(confidence).strip().casefold() if confidence else default
    return normalized if normalized in {"high", "medium", "low"} else default


def _assessment_warnings(value: Any) -> tuple[str, ...]:
    warnings = (
        value.get("warnings", ())
        if isinstance(value, Mapping)
        else getattr(value, "warnings", ())
    )
    return tuple(str(item).strip() for item in (warnings or ()) if str(item).strip())


def _auto_rationale(
    resolved: FcffForecastMethod,
    quality: Any,
) -> str:
    if resolved == FcffForecastMethod.HYBRID:
        reason = _assessment_reason(quality)
        return (
            "AUTO selected HYBRID because the existing operating capability/quality "
            "gate was accepted" + (f": {reason}" if reason else "")
        )
    if quality is None:
        return (
            "AUTO selected NORMALIZED because no operating capability or quality "
            "assessment was available"
        )
    reason = _assessment_reason(quality)
    return (
        "AUTO selected NORMALIZED because the existing operating capability/quality "
        "gate was not accepted" + (f": {reason}" if reason else "")
    )


ForecastPlanService = FcffForecastPlanService


__all__ = ["FcffForecastPlanService", "ForecastPlanService"]
