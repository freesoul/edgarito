"""Deterministic machine and fixed-section human backtest reports."""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .contracts import ForecastBacktestResult


class ReportBuilder:
    def machine(self, result: ForecastBacktestResult | Any) -> dict[str, Any]:
        return deterministic_machine_report(result)

    def human(self, result: ForecastBacktestResult | Any) -> str:
        return human_report(result)

HUMAN_SECTIONS = (
    "MODEL STRUCTURE",
    "REASONED ASSUMPTIONS",
    "EVIDENCE USED",
    "UNRESOLVED ITEMS",
    "FORECAST OUTPUT",
    "ACTUAL OUTPUT",
    "ERRORS",
    "LOW/HIGH COVERAGE",
    "VALIDATION FINDINGS",
    "BASELINE COMPARISON",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="python"))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key).casefold() != "generated_at"
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        values = [_jsonable(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(
                values,
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":"), default=str
                ),
            )
        return values
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    values = getattr(value, "__dict__", None)
    if isinstance(values, dict):
        return _jsonable(values)
    return str(value)


def deterministic_machine_report(result: ForecastBacktestResult | Any) -> dict[str, Any]:
    """Return a recursively sorted, deterministic JSON-compatible dictionary."""

    result = result if isinstance(result, ForecastBacktestResult) else ForecastBacktestResult.model_validate(result)
    payload = {
        "case_id": result.case_id,
        "ticker": result.ticker,
        "company": result.company,
        "as_of": result.as_of,
        "fiscal_years": result.fiscal_years,
        "report_identity": result.report_identity,
        "leakage_audit": result.leakage_audit,
        "reasoned_result": result.reasoned_result,
        "canonical_forecast": result.canonical_forecast,
        "actual_outcomes": result.actual_outcomes,
        "actual_outcome_audit": result.actual_outcome_audit,
        "assumption_scores": result.assumption_scores,
        "financial_scores": result.financial_scores,
        "calibration": result.calibration,
        "complexity": result.complexity,
        "validation": result.validation,
        "normalized_comparison": result.normalized_comparison,
        "hybrid_comparison": result.hybrid_comparison,
        "diagnostics": result.diagnostics,
        "routes": result.routes,
    }
    # Sorting through a JSON round trip also guarantees a stable scalar
    # representation for Decimal values held by arbitrary fake collaborators.
    return json.loads(json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")))


class EvaluationMachineReport(BaseModel):
    """Pydantic envelope for deterministic machine output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    payload: dict[str, Any] = Field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return deterministic_sorted_dict(self.payload)

    def json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


def deterministic_sorted_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): deterministic_sorted_dict(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [deterministic_sorted_dict(item) for item in value]
    return value


def machine_report(result: ForecastBacktestResult | Any) -> dict[str, Any]:
    return deterministic_machine_report(result)


def machine_report_model(result: ForecastBacktestResult | Any) -> EvaluationMachineReport:
    return EvaluationMachineReport(payload=deterministic_machine_report(result))


def _section_payload(result: ForecastBacktestResult, section: str) -> Any:
    mapping = {
        "MODEL STRUCTURE": result.complexity,
        "REASONED ASSUMPTIONS": result.assumption_scores,
        "EVIDENCE USED": {
            "included": result.leakage_audit.included,
            "excluded": result.leakage_audit.excluded,
        },
        "UNRESOLVED ITEMS": getattr(getattr(result.reasoned_result, "reasoning", None), "unresolved_items", ()),
        "FORECAST OUTPUT": result.canonical_forecast,
        "ACTUAL OUTPUT": result.actual_outcomes,
        "ERRORS": result.financial_scores,
        "LOW/HIGH COVERAGE": result.calibration,
        "VALIDATION FINDINGS": {
            "leakage": result.leakage_audit.issues,
            "validation": result.validation,
        },
        "BASELINE COMPARISON": {
            "normalized": result.normalized_comparison,
            "hybrid": result.hybrid_comparison,
        },
    }
    return mapping[section]


def human_report(result: ForecastBacktestResult | Any) -> str:
    """Render exactly the ten stable section headings, with no wall clock."""

    result = result if isinstance(result, ForecastBacktestResult) else ForecastBacktestResult.model_validate(result)
    lines: list[str] = []
    for section in HUMAN_SECTIONS:
        lines.append(section)
        lines.append(json.dumps(_jsonable(_section_payload(result, section)), sort_keys=True, separators=(",", ":")))
    return "\n".join(lines)


render_human_report = human_report
build_machine_report = deterministic_machine_report


__all__ = [
    "HUMAN_SECTIONS",
    "EvaluationMachineReport",
    "deterministic_machine_report",
    "machine_report",
    "machine_report_model",
    "human_report",
    "render_human_report",
    "build_machine_report",
]
