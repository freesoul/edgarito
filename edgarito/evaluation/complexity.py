"""Descriptive model-complexity diagnostics for reasoned forecasts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .contracts import ComplexityDiagnostics


class ComplexityAnalyzer:
    def analyze(
        self,
        reasoning_input: Any = None,
        reasoned_result: Any = None,
        *,
        validation_findings: Iterable[Any] = (),
    ) -> ComplexityDiagnostics:
        return complexity_diagnostics(
            reasoning_input,
            reasoned_result,
            validation_findings=validation_findings,
        )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def complexity_diagnostics(
    reasoning_input: Any = None,
    reasoned_result: Any = None,
    *,
    validation_findings: Iterable[Any] = (),
) -> ComplexityDiagnostics:
    """Count model structure and findings without declaring a model wrong."""

    if reasoned_result is None and (
        _field(reasoning_input, "reasoning") is not None
        or _field(reasoning_input, "accepted_assumptions") is not None
    ):
        reasoned_result = reasoning_input
        reasoning_input = None
    input_value = reasoning_input
    reasoning = _field(reasoned_result, "reasoning")
    accepted = _tuple(_field(reasoning, "accepted_assumptions")) or _tuple(_field(reasoned_result, "accepted_assumptions"))
    rejected = _tuple(_field(reasoning, "rejected_assumptions")) + _tuple(_field(reasoning, "rejected_decisions"))
    unresolved = _tuple(_field(reasoning, "unresolved_items"))
    if not unresolved:
        unresolved = _tuple(_field(reasoned_result, "unresolved_items"))
    if input_value is None:
        input_value = _field(reasoned_result, "reasoning_input")
    segments = len(_tuple(_field(input_value, "segments")))
    driver_assumptions = sum(_field(item, "target_type") == "operating_driver" for item in accepted)
    financial_assumptions = sum(_field(item, "target_type") == "forecast_metric" for item in accepted)
    model_assumptions = sum(
        str(_field(item, "assumption_type", "")).casefold() in {"model_assumption", "model"}
        or bool(_field(item, "model_assumption", False))
        for item in accepted
    )
    evidence_based = sum(
        str(_field(item, "assumption_type", "")).casefold() in {"evidence_based", "evidence"}
        or bool(_field(item, "evidence_based", False))
        for item in accepted
    )
    manual = len(_tuple(_field(input_value, "manual_overrides"))) + len(
        _tuple(_field(input_value, "manual_forward_driver_observations"))
    )
    findings = tuple(validation_findings)
    if not findings:
        validation = _field(reasoned_result, "validation") or _field(reasoning, "validation")
        findings = (
            _tuple(_field(validation, "findings"))
            + _tuple(_field(validation, "issues"))
            + _tuple(_field(validation, "warnings"))
        )
        findings += _tuple(_field(reasoned_result, "warnings"))
    indicators = (
        f"segments={segments}",
        f"driver_assumptions={driver_assumptions}",
        f"financial_assumptions={financial_assumptions}",
        f"model_assumptions={model_assumptions}",
        f"evidence_based={evidence_based}",
        f"unresolved={len(unresolved)}",
        f"rejected={len(rejected)}",
        f"manual={manual}",
        f"validation_findings={len(findings)}",
    )
    return ComplexityDiagnostics(
        segments=segments,
        driver_assumptions=driver_assumptions,
        financial_assumptions=financial_assumptions,
        model_assumptions=model_assumptions,
        evidence_based=evidence_based,
        unresolved=len(unresolved),
        rejected=len(rejected),
        manual=manual,
        validation_findings=len(findings),
        over_modeling_indicators=indicators,
    )


diagnose_complexity = complexity_diagnostics


__all__ = ["ComplexityAnalyzer", "complexity_diagnostics", "diagnose_complexity"]
