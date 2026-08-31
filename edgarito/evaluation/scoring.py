"""Deterministic assumption and financial scoring for backtest outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from .contracts import (
    ActualAssumptionOutcome,
    ActualOutcomeData,
    AssumptionScore,
    AssumptionScoreReport,
    FinancialMetricScore,
    FinancialScoreReport,
    _decimal_key,
    _unit_key,
    canonical_financial_metric,
)

_FINANCIAL_METRICS = (
    "revenue",
    "ebit",
    "tax_rate",
    "nopat",
    "depreciation_and_amortization",
    "capex",
    "operating_working_capital",
    "delta_nwc",
    "fcff",
)


class AssumptionScorer:
    def score(self, reasoned: Any, actuals: ActualOutcomeData | Any) -> AssumptionScoreReport:
        return score_assumptions(reasoned, actuals)


class FinancialScorer:
    def score(
        self,
        forecast: Any,
        actuals: ActualOutcomeData | Any,
        *,
        method: str = "reasoned",
        required_years: tuple[int, ...] | list[int] | None = None,
    ) -> FinancialScoreReport:
        return score_financials(
            forecast, actuals, method=method, required_years=required_years
        )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    value = getattr(value, "value", value)
    try:
        value = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception:
        return None
    return value if value.is_finite() else None


def _metric(value: Any) -> str:
    return canonical_financial_metric(value)


def _scope(value: Any) -> str:
    return str(getattr(value, "value", value)).casefold()


def _basis(value: Any) -> str:
    raw = str(getattr(value, "value", value)).strip().casefold().replace("-", "_").replace(" ", "_")
    return {
        "amount": "absolute",
        "currency": "absolute",
        "ratio": "percent_of_revenue",
        "percent": "percent_of_revenue",
        "percentage": "percentage_points",
        "pp": "percentage_points",
    }.get(raw, raw)


def _safe_percentage_error(forecast: Decimal | None, actual: Decimal | None) -> Decimal | None:
    """Return percentage error only where a sign/zero denominator is stable."""

    if forecast is None or actual is None or actual == 0:
        return None
    if forecast != 0 and ((forecast < 0) != (actual < 0)):
        return None
    return abs(forecast - actual) / abs(actual) * Decimal(100)


def _sign(value: Decimal | None) -> int | None:
    if value is None:
        return None
    return 1 if value > 0 else -1 if value < 0 else 0


def _sign_error(forecast: Decimal | None, actual: Decimal | None) -> bool | None:
    if forecast is None or actual is None:
        return None
    return _sign(forecast) != _sign(actual)


def _assumptions(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    for candidate in (
        _field(value, "accepted_assumptions"),
        _field(_field(value, "reasoning"), "accepted_assumptions"),
        _field(_field(value, "response"), "assumptions"),
        _field(value, "assumptions"),
    ):
        if candidate is not None:
            return tuple(candidate)
    return ()


def _actual_assumption_key(
    item: ActualAssumptionOutcome,
) -> tuple[str, str, str, str, int, str, str, str]:
    return (
        item.target_type,
        _scope(item.scope),
        item.scope_id,
        _metric(item.driver_id or item.metric),
        item.fiscal_year,
        _basis(item.basis),
        item.unit_key,
        _decimal_key(item.scale),
    )


def _reasoned_assumption_key(
    item: Any, year: int
) -> tuple[str, str, str, str, int, str, str, str]:
    target_type = str(_field(item, "target_type", "")).casefold()
    target = _field(item, "driver_id") if target_type == "operating_driver" else _field(item, "metric")
    return (
        target_type,
        _scope(_field(item, "scope", "company")),
        str(_field(item, "scope_id", "company")),
        _metric(target),
        year,
        _basis(_field(item, "basis", "absolute")),
        _unit_key(str(_field(item, "unit", "currency"))),
        "1",
    )


def score_assumptions(
    reasoned: Any,
    actuals: ActualOutcomeData | Any,
) -> AssumptionScoreReport:
    """Match each retained assumption path to a distinct realized target."""

    actuals = actuals if isinstance(actuals, ActualOutcomeData) else ActualOutcomeData.model_validate(actuals)
    by_key = {_actual_assumption_key(item): item for item in actuals.assumption_outcomes}
    scores: list[AssumptionScore] = []
    for assumption in _assumptions(reasoned):
        years = tuple(_field(assumption, "fiscal_years", ()))
        low_path = tuple(_decimal(item) for item in (_field(assumption, "low", ()) or ()))
        base_path = tuple(_decimal(item) for item in (_field(assumption, "base", ()) or ()))
        high_path = tuple(_decimal(item) for item in (_field(assumption, "high", ()) or ()))
        confidence = str(_field(assumption, "confidence", "medium")).casefold()
        kind = str(_field(assumption, "assumption_type", ""))
        if not kind:
            kind = "evidence_based" if _field(assumption, "evidence_based", False) else "model_assumption"
        for index, year in enumerate(years):
            low, base, high = low_path[index], base_path[index], high_path[index]
            # A malformed external object is unscored rather than causing a
            # partial scoring report.  Typed reasoner contracts normally make
            # these values non-null and ordered.
            if None in (low, base, high):
                continue
            key = _reasoned_assumption_key(assumption, year)
            actual = by_key.get(key)
            if actual is None:
                compatible_target = tuple(key[:6])
                incompatible = tuple(
                    item
                    for actual_key, item in by_key.items()
                    if tuple(actual_key[:6]) == compatible_target
                )
                scores.append(
                    AssumptionScore(
                        assumption_id=str(_field(assumption, "assumption_id", "unknown")),
                        target_type=key[0],
                        scope=key[1],
                        scope_id=key[2],
                        target=key[3],
                        fiscal_year=year,
                        basis=key[5],
                        reasoned_unit=str(_field(assumption, "unit", "currency")),
                        actual_unit=(incompatible[0].unit if incompatible else None),
                        low=low,
                        base=base,
                        high=high,
                        confidence=confidence,
                        kind=kind,
                        evidence_ids=tuple(_field(assumption, "evidence_ids", ()) or ()),
                        scored=False,
                        unmatched_reason=(
                            "incompatible unit or scale in ActualAssumptionOutcome"
                            if incompatible
                            else "no ActualAssumptionOutcome matched target, scope, year, and basis"
                        ),
                    )
                )
                continue
            actual_value = actual.actual
            width = high - low
            scores.append(
                AssumptionScore(
                    assumption_id=str(_field(assumption, "assumption_id", "unknown")),
                    target_type=key[0],
                    scope=key[1],
                    scope_id=key[2],
                    target=key[3],
                    fiscal_year=year,
                    basis=key[5],
                    reasoned_unit=str(_field(assumption, "unit", "currency")),
                    actual_unit=actual.unit,
                    low=low,
                    base=base,
                    high=high,
                    actual=actual_value,
                    absolute_error=abs(base - actual_value) if actual_value is not None else None,
                    percentage_error=_safe_percentage_error(base, actual_value),
                    interval_hit=(low <= actual_value <= high) if actual_value is not None else None,
                    normalized_interval_position=(actual_value - low) / width if actual_value is not None and width != 0 else None,
                    confidence=confidence,
                    kind=kind,
                    evidence_ids=tuple(_field(assumption, "evidence_ids", ()) or ()),
                    scored=actual_value is not None,
                    unmatched_reason=(
                        "matched outcome has no realized value"
                        if actual_value is None
                        else None
                    ),
                )
            )
    scores.sort(key=lambda item: (item.fiscal_year, item.target_type, item.scope_id, item.target, item.assumption_id))
    return AssumptionScoreReport(
        scores=tuple(scores),
        scored_count=sum(item.scored for item in scores),
        unscored_count=sum(not item.scored for item in scores),
    )


def _forecast_rows(forecast: Any) -> tuple[Any, ...]:
    if forecast is None:
        return ()
    rows = _field(forecast, "observations")
    if rows is not None:
        return tuple(rows)
    if isinstance(forecast, Mapping):
        rows = forecast.get("forecast_observations", forecast.get("rows", ()))
        return tuple(rows or ())
    return ()


def _canonical_forecast(value: Any) -> Any:
    for candidate in (value, _field(value, "canonical_forecast"), _field(value, "forecast"), _field(value, "driver_result")):
        if candidate is not None and _forecast_rows(candidate):
            return candidate
    return value


def _forecast_value(row: Any, metric: str) -> Decimal | None:
    names = {
        "revenue": ("revenue",),
        "ebit": ("ebit", "operating_income"),
        "tax_rate": ("tax_rate",),
        "nopat": ("nopat",),
        "depreciation_and_amortization": ("depreciation_and_amortization", "depreciation"),
        "capex": ("capex", "capital_expenditures"),
        "operating_working_capital": ("operating_working_capital", "owc"),
        "delta_nwc": ("delta_nwc", "change_in_operating_working_capital", "change_in_working_capital"),
        "fcff": ("fcff",),
    }[metric]
    return next((_decimal(_field(row, name)) for name in names if _field(row, name) is not None), None)


def _unit_metadata(
    value: Any, metric: str, parent: Any = None
) -> tuple[str, str | None, Decimal]:
    if metric == "tax_rate":
        default_unit = "percent"
    else:
        default_unit = "currency"
    raw = value
    if raw is None:
        raw = parent
    nested_unit = _field(raw, "unit") if raw is not None else None
    if nested_unit is None and parent is not None and metric != "tax_rate":
        nested_unit = _field(parent, "unit")
    unit = str(nested_unit or default_unit)
    currency = _field(raw, "currency") if raw is not None else None
    if currency is None and parent is not None:
        currency = _field(parent, "currency")
    scale = _decimal(_field(raw, "scale")) if raw is not None else None
    if scale is None and parent is not None:
        scale = _decimal(_field(parent, "scale"))
    return unit, str(currency) if currency is not None else None, scale or Decimal(1)


def _forecast_metadata(row: Any, metric: str, parent: Any) -> tuple[str, str | None, Decimal]:
    names = {
        "revenue": ("revenue",),
        "ebit": ("ebit", "operating_income"),
        "tax_rate": ("tax_rate",),
        "nopat": ("nopat",),
        "depreciation_and_amortization": ("depreciation_and_amortization", "depreciation"),
        "capex": ("capex", "capital_expenditures"),
        "operating_working_capital": ("operating_working_capital", "owc"),
        "delta_nwc": ("delta_nwc", "change_in_operating_working_capital", "change_in_working_capital"),
        "fcff": ("fcff",),
    }[metric]
    item = next((_field(row, name) for name in names if _field(row, name) is not None), None)
    row_unit = _field(row, "unit") if row is not None else None
    item_unit = _field(item, "unit") if item is not None else None
    if metric != "tax_rate" and item_unit is None and row_unit is None:
        return _unit_metadata(item, metric, parent)
    return _unit_metadata(item, metric, row or parent)


def _actual_financials(actuals: ActualOutcomeData) -> dict[tuple[int, str], Decimal | None]:
    return {(item.fiscal_year, _metric(item.metric)): item.value for item in actuals.observations}


def _actual_financial_metadata(
    actuals: ActualOutcomeData,
) -> dict[tuple[int, str], tuple[str, str | None, Decimal]]:
    return {
        (item.fiscal_year, _metric(item.metric)): (
            item.unit,
            item.currency,
            item.scale,
        )
        for item in actuals.observations
    }


def score_financials(
    forecast: Any,
    actuals: ActualOutcomeData | Any,
    *,
    method: str = "reasoned",
    required_years: tuple[int, ...] | list[int] | None = None,
) -> FinancialScoreReport:
    """Score independent fiscal years without changing the forecast artifact."""

    actuals = actuals if isinstance(actuals, ActualOutcomeData) else ActualOutcomeData.model_validate(actuals)
    forecast = _canonical_forecast(forecast)
    rows = _forecast_rows(forecast)
    by_year = {int(_field(row, "fiscal_year")): row for row in rows if _field(row, "fiscal_year") is not None}
    if required_years is not None:
        years = tuple(int(item) for item in required_years)
        if tuple(sorted(years)) != years or len(years) != len(set(years)):
            raise ValueError("required_years must be sorted and unique")
    else:
        years = tuple(
            sorted(
                set(by_year)
                if by_year
                else {item.fiscal_year for item in actuals.observations}
            )
        )
    actual_by_key = _actual_financials(actuals)
    actual_metadata = _actual_financial_metadata(actuals)
    values: dict[tuple[int, str], Decimal | None] = {
        (year, metric): _forecast_value(by_year.get(year), metric) if year in by_year else None
        for year in years
        for metric in _FINANCIAL_METRICS
    }
    metadata: dict[tuple[int, str], tuple[str, str | None, Decimal]] = {
        (year, metric): _forecast_metadata(by_year.get(year), metric, forecast)
        for year in years
        for metric in _FINANCIAL_METRICS
    }
    scores: list[FinancialMetricScore] = []
    for year in years:
        for metric in _FINANCIAL_METRICS:
            forecast_value = values[(year, metric)]
            actual_value = actual_by_key.get((year, metric))
            forecast_unit, forecast_currency, forecast_scale = metadata[(year, metric)]
            actual_unit_data = actual_metadata.get((year, metric))
            actual_unit, actual_currency, actual_scale = actual_unit_data or (
                None,
                None,
                None,
            )
            units_compatible = (
                forecast_value is not None
                and actual_value is not None
                and actual_unit is not None
                and _unit_key(forecast_unit, forecast_currency)
                == _unit_key(actual_unit, actual_currency)
                and forecast_scale == actual_scale
            )
            prior_forecast = values.get((year - 1, metric))
            prior_actual = actual_by_key.get((year - 1, metric))
            prior_units_compatible = (
                prior_forecast is not None
                and prior_actual is not None
                and (year - 1, metric) in actual_metadata
                and _unit_key(
                    metadata[(year - 1, metric)][0], metadata[(year - 1, metric)][1]
                )
                == _unit_key(
                    actual_metadata[(year - 1, metric)][0],
                    actual_metadata[(year - 1, metric)][1],
                )
                and metadata[(year - 1, metric)][2]
                == actual_metadata[(year - 1, metric)][2]
            )
            forecast_direction = (
                _sign(forecast_value - prior_forecast)
                if units_compatible and prior_units_compatible
                else None
            )
            actual_direction = (
                _sign(actual_value - prior_actual)
                if units_compatible and prior_units_compatible
                else None
            )
            unmatched_reason = (
                "forecast value unavailable"
                if forecast_value is None
                else "actual value unavailable"
                if actual_value is None
                else "incompatible forecast/actual unit, currency, or scale"
                if not units_compatible
                else None
            )
            scores.append(
                FinancialMetricScore(
                    method=method,
                    metric=metric,
                    fiscal_year=year,
                    forecast=forecast_value,
                    actual=actual_value,
                    forecast_unit=forecast_unit,
                    forecast_currency=forecast_currency,
                    forecast_scale=forecast_scale,
                    actual_unit=actual_unit,
                    actual_currency=actual_currency,
                    actual_scale=actual_scale,
                    absolute_error=abs(forecast_value - actual_value)
                    if units_compatible
                    else None,
                    percentage_error=(
                        _safe_percentage_error(forecast_value, actual_value)
                        if units_compatible
                        else None
                    ),
                    sign_error=_sign_error(forecast_value, actual_value)
                    if units_compatible
                    else None,
                    yoy_direction_error=(forecast_direction != actual_direction) if forecast_direction is not None and actual_direction is not None else None,
                    scored=units_compatible,
                    unmatched_reason=unmatched_reason,
                )
            )
    report = FinancialScoreReport(
        method=method,
        required_years=years,
        scores=tuple(scores),
        per_method={method: tuple(scores)},
    )
    return report


def score_financials_by_method(
    forecasts: Mapping[str, Any],
    actuals: ActualOutcomeData | Any,
    *,
    required_years: tuple[int, ...] | list[int] | None = None,
) -> dict[str, FinancialScoreReport]:
    return {
        method: score_financials(
            forecast, actuals, method=method, required_years=required_years
        )
        for method, forecast in sorted(forecasts.items())
    }


score_assumptions_for_result = score_assumptions
score_financial_metrics = score_financials
score_assumption_paths = score_assumptions


__all__ = [
    "score_assumptions",
    "AssumptionScorer",
    "FinancialScorer",
    "score_assumptions_for_result",
    "score_assumption_paths",
    "score_financials",
    "score_financial_metrics",
    "score_financials_by_method",
]
