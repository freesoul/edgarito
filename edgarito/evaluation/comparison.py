"""Method-aware comparisons between reasoned and baseline forecasts."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from .contracts import BaselineComparison, FinancialScoreReport, RouteIdentity
from .scoring import _forecast_rows, score_financials


def _errors(report: FinancialScoreReport) -> dict[str, dict[int, Decimal]]:
    return {
        metric: {
            item.fiscal_year: item.absolute_error
            for item in report.scores
            if item.metric == metric and item.absolute_error is not None
        }
        for metric in sorted({item.metric for item in report.scores})
    }


def _row_year(row: Any) -> int | None:
    value = row.get("fiscal_year") if isinstance(row, Mapping) else getattr(row, "fiscal_year", None)
    return int(value) if value is not None else None


def compare_forecasts(
    reasoned_forecast: Any,
    baseline_forecast: Any,
    actuals: Any,
    *,
    route: str = "normalized",
    method: str | None = None,
    collaborator: str | None = None,
    required_years: tuple[int, ...] | list[int] | None = None,
    reasoned_evidence_identity: str | None = None,
    baseline_evidence_identity: str | None = None,
) -> BaselineComparison:
    """Return descriptive per-metric error deltas, never a method-string guess."""

    route_text = str(getattr(route, "value", route))
    reasoned_years = tuple(
        sorted(
            _row_year(row)
            for row in _forecast_rows(reasoned_forecast)
            if _row_year(row) is not None
        )
    )
    required = tuple(required_years) if required_years is not None else reasoned_years
    if tuple(sorted(required)) != required or len(required) != len(set(required)):
        raise ValueError("Comparison required_years must be sorted and unique")
    if baseline_forecast is None:
        return BaselineComparison(
            route=RouteIdentity(
                route=route_text,
                collaborator=collaborator,
                available=False,
                reason=f"{route_text} baseline returned no forecast",
            ),
            method=method or route_text,
            unavailable_reason=f"{route_text} baseline returned no forecast",
        )
    baseline_years = tuple(
        sorted(
            _row_year(row)
            for row in _forecast_rows(baseline_forecast)
            if _row_year(row) is not None
        )
    )
    if baseline_years != required:
        raise ValueError(
            f"{route_text} baseline fiscal horizon {baseline_years} does not match "
            f"required years {required}"
        )
    if reasoned_years != required:
        raise ValueError(
            f"reasoned fiscal horizon {reasoned_years} does not match required years {required}"
        )
    if (
        reasoned_evidence_identity is not None
        and baseline_evidence_identity is not None
        and reasoned_evidence_identity != baseline_evidence_identity
    ):
        return BaselineComparison(
            route=RouteIdentity(
                route=route_text,
                collaborator=collaborator,
                available=False,
                reason="evidence identity differs; baseline is non-comparable",
                evidence_identity=baseline_evidence_identity,
            ),
            method=method or route_text,
            unavailable_reason="evidence identity differs; baseline is non-comparable",
        )
    baseline_report = score_financials(
        baseline_forecast,
        actuals,
        method=method or route_text,
        required_years=required,
    )
    reasoned_report = score_financials(
        reasoned_forecast, actuals, method="reasoned", required_years=required
    )
    baseline_coverage = {
        (item.metric, item.fiscal_year)
        for item in baseline_report.scores
        if item.scored
    }
    reasoned_coverage = {
        (item.metric, item.fiscal_year)
        for item in reasoned_report.scores
        if item.scored
    }
    if not baseline_coverage:
        raise ValueError(f"{route_text} baseline has no scoreable required metrics")
    if baseline_coverage != reasoned_coverage:
        raise ValueError(
            f"{route_text} baseline score coverage does not match reasoned coverage"
        )
    reasoned_by = _errors(reasoned_report)
    baseline_by = _errors(baseline_report)
    metrics = sorted(set(reasoned_by) | set(baseline_by))
    deltas: dict[str, Any] = {}
    for metric in metrics:
        years = sorted(set(reasoned_by.get(metric, {})) & set(baseline_by.get(metric, {})))
        year_values: dict[str, Any] = {}
        for year in years:
            reasoned_error = reasoned_by[metric][year]
            baseline_error = baseline_by[metric][year]
            delta = reasoned_error - baseline_error
            status = "improved" if delta < 0 else "worsened" if delta > 0 else "tied"
            year_values[str(year)] = {
                "reasoned_absolute_error": reasoned_error,
                "baseline_absolute_error": baseline_error,
                "delta_reasoned_minus_baseline": delta,
                "status": status,
            }
        reasoned_mean = (
            sum((reasoned_by[metric][year] for year in years), Decimal(0)) / len(years)
            if years
            else None
        )
        baseline_mean = (
            sum((baseline_by[metric][year] for year in years), Decimal(0)) / len(years)
            if years
            else None
        )
        if reasoned_mean is None or baseline_mean is None:
            status = "unscored"
            delta = None
        else:
            delta = reasoned_mean - baseline_mean
            status = "improved" if delta < 0 else "worsened" if delta > 0 else "tied"
        deltas[metric] = {
            "status": status,
            "reasoned_absolute_error": reasoned_mean,
            "baseline_absolute_error": baseline_mean,
            "delta_reasoned_minus_baseline": delta,
            "years": year_values,
        }
    return BaselineComparison(
        route=RouteIdentity(
            route=route_text,
            collaborator=collaborator,
            evidence_identity=baseline_evidence_identity,
        ),
        method=method or route_text,
        financial_scores=baseline_report,
        metric_deltas=deltas,
    )


compare_baseline = compare_forecasts
compare_reasoned_to_baseline = compare_forecasts


__all__ = ["compare_forecasts", "compare_baseline", "compare_reasoned_to_baseline"]
