"""Repeatability diagnostics for reasoner proposals, without scenario execution."""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from edgarito.services.forecasting.reasoning.evidence import content_hash

from .contracts import (
    ForecastBacktestCase,
    StabilityObservation,
    StabilityReport,
    _stable_value,
)
from .leakage import enforce_leakage


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception:
        return None
    return result if result.is_finite() else None


def _assumptions(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    response = _field(value, "response")
    return tuple(
        _field(value, "accepted_assumptions")
        or _field(response, "assumptions")
        or _field(value, "assumptions")
        or ()
    )


def _target(assumption: Any) -> str:
    target_type = str(_field(assumption, "target_type", "")).casefold()
    target = _field(assumption, "driver_id") if target_type == "operating_driver" else _field(assumption, "metric")
    return ":".join(
        (
            target_type,
            str(getattr(_field(assumption, "scope", "company"), "value", _field(assumption, "scope", "company"))),
            str(_field(assumption, "scope_id", "company")),
            str(getattr(target, "value", target)),
        )
    )


def _reasoner_callable(reasoner: Any) -> Any:
    return getattr(reasoner, "reason", None) or getattr(reasoner, "propose", None) or reasoner


def _supports_force_refresh(reasoner: Any) -> bool:
    runner = _reasoner_callable(reasoner)
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or (
            name == "force_refresh"
            and parameter.kind != inspect.Parameter.POSITIONAL_ONLY
        )
        for name, parameter in signature.parameters.items()
    )


async def _call_reasoner(
    reasoner: Any, input_value: Any, *, supports_force_refresh: bool
) -> Any:
    runner = _reasoner_callable(reasoner)
    accepts_refresh = supports_force_refresh
    raw = runner(input_value, force_refresh=True) if accepts_refresh else runner(input_value)
    return await raw if inspect.isawaitable(raw) else raw


class StabilityEvaluator:
    """Repeat one frozen reasoner input and report dispersion only."""

    def __init__(self, reasoner: Any):
        if reasoner is None:
            raise ValueError("StabilityEvaluator requires an injected reasoner")
        self.reasoner = reasoner

    async def evaluate(
        self,
        case_or_input: ForecastBacktestCase | Any,
        *,
        runs: int = 3,
    ) -> StabilityReport:
        if runs < 1:
            raise ValueError("Stability runs must be at least one")
        if not isinstance(case_or_input, ForecastBacktestCase):
            raise TypeError(
                "StabilityEvaluator requires a complete ForecastBacktestCase; "
                "raw ForecastReasoningInput is not auditable"
            )
        enforce_leakage(case_or_input)
        supports_force_refresh = _supports_force_refresh(self.reasoner)
        if runs > 1 and not supports_force_refresh:
            raise ValueError(
                "StabilityEvaluator requires explicit force_refresh or **kwargs "
                "support when runs > 1"
            )
        input_value = case_or_input.reasoning_input
        if input_value is None:
            raise ValueError("Stability requires a ForecastReasoningInput in the case")
        # A model_copy is intentionally not made per run: each invocation gets
        # the same frozen object and force_refresh controls cache bypass.
        observations: list[StabilityObservation] = []
        for index in range(runs):
            proposal = await _call_reasoner(
                self.reasoner,
                input_value,
                supports_force_refresh=supports_force_refresh,
            )
            run_identity = str(
                _field(proposal, "proposal_identity")
                or _field(proposal, "cache_key")
                or content_hash(_stable_value(proposal))
            )
            # A fake may return the same identity on every refresh; retain a
            # deterministic run suffix rather than collapsing the audit trail.
            run_identity = f"{run_identity}:run-{index + 1}"
            for assumption in _assumptions(proposal):
                years = tuple(_field(assumption, "fiscal_years", ()) or ())
                bases = tuple(_decimal(value) for value in (_field(assumption, "base", ()) or ()))
                lows = tuple(_decimal(value) for value in (_field(assumption, "low", ()) or ()))
                highs = tuple(_decimal(value) for value in (_field(assumption, "high", ()) or ()))
                target = _target(assumption)
                for position, year in enumerate(years):
                    observations.append(
                        StabilityObservation(
                            run_identity=run_identity,
                            target=target,
                            fiscal_year=int(year),
                            base=bases[position] if position < len(bases) else None,
                            low=lows[position] if position < len(lows) else None,
                            high=highs[position] if position < len(highs) else None,
                        )
                    )
        grouped: defaultdict[str, list[Decimal]] = defaultdict(list)
        for item in observations:
            if item.base is not None:
                grouped[f"{item.target}:{item.fiscal_year}"].append(item.base)
        minimum: dict[str, Decimal | None] = {}
        maximum: dict[str, Decimal | None] = {}
        mean: dict[str, Decimal | None] = {}
        variance: dict[str, Decimal | None] = {}
        dispersion: dict[str, Decimal | None] = {}
        for key, values in sorted(grouped.items()):
            minimum[key] = min(values)
            maximum[key] = max(values)
            mean[key] = sum(values, Decimal(0)) / Decimal(len(values))
            variance[key] = sum((value - mean[key]) ** 2 for value in values) / Decimal(len(values))
            dispersion[key] = maximum[key] - minimum[key]
        return StabilityReport(
            run_count=runs,
            observations=tuple(observations),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            variance=variance,
            dispersion=dispersion,
        )


async def evaluate_stability(
    case_or_input: ForecastBacktestCase | Any,
    reasoner: Any,
    *,
    runs: int = 3,
) -> StabilityReport:
    return await StabilityEvaluator(reasoner).evaluate(case_or_input, runs=runs)


__all__ = ["StabilityEvaluator", "evaluate_stability"]
