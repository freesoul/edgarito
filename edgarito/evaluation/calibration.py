"""Descriptive interval calibration aggregates; no inferential statistics."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from .contracts import AssumptionScore, CalibrationStratum, CalibrationSummary


class CalibrationAggregator:
    def aggregate(self, values: Any) -> CalibrationSummary:
        return aggregate_calibration(values)


def _score_values(value: Any) -> tuple[AssumptionScore, ...]:
    if value is None:
        return ()
    if isinstance(value, AssumptionScore):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        result: list[AssumptionScore] = []
        for item in value:
            result.extend(_score_values(item))
        return tuple(result)
    case_scores = getattr(value, "assumption_scores", None)
    if case_scores is not None and case_scores is not value:
        return _score_values(case_scores)
    scores = getattr(value, "scores", None)
    if scores is not None:
        return tuple(item for item in scores if isinstance(item, AssumptionScore))
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        result: list[AssumptionScore] = []
        for item in value:
            result.extend(_score_values(item))
        return tuple(result)
    return ()


def aggregate_calibration(values: Any) -> CalibrationSummary:
    """Aggregate interval hits across assumptions or complete backtest cases."""

    scores = tuple(item for item in _score_values(values) if item.scored and item.interval_hit is not None)
    hits = sum(item.interval_hit is True for item in scores)
    by_confidence: dict[str, list[AssumptionScore]] = {}
    for item in scores:
        by_confidence.setdefault(item.confidence, []).append(item)
    strata = []
    for confidence, items in sorted(by_confidence.items()):
        count = len(items)
        strata.append(
            CalibrationStratum(
                confidence=confidence,
                sample_size=count,
                hit_count=sum(item.interval_hit is True for item in items),
                interval_coverage=Decimal(sum(item.interval_hit is True for item in items)) / Decimal(count),
                warning=(
                    "Small sample; coverage is descriptive and has no statistical significance."
                    if count < 30
                    else "Coverage is descriptive; no statistical significance is claimed."
                ),
            )
        )
    return CalibrationSummary(
        sample_size=len(scores),
        hit_count=hits,
        interval_coverage=Decimal(hits) / Decimal(len(scores)) if scores else None,
        strata=tuple(strata),
    )


calibrate = aggregate_calibration
aggregate_interval_coverage = aggregate_calibration


__all__ = [
    "CalibrationAggregator",
    "aggregate_calibration",
    "aggregate_interval_coverage",
    "calibrate",
]
