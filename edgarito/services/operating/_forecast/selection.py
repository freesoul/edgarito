"""Select authoritative operating observations for each forecast input."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal

from edgarito.config.operating import OPERATING_VOCABULARY
from edgarito.schemas.operating import (
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    operating_periods_compatible,
)
from edgarito.services.operating._forecast.contracts import (
    _MANAGEMENT_SOURCE,
    _SelectedObservation,
)

_SEGMENT_REVENUE_DRIVERS = frozenset(OPERATING_VOCABULARY.revenue_driver_priority)
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_ORIGIN_RANK = {
    "management_guidance": 2,
    "reported": 1,
    "first_party_observation": 1,
    "extracted_evidence": 1,
    "forward_evidence": 1,
    "derived": 1,
    "reasoned_assumption": 0,
    "model_assumption": -1,
}


def _select_observations(
    records: Iterable[OperatingDriverObservation],
) -> dict[tuple[str, str, int], _SelectedObservation]:
    grouped: dict[
        tuple[str, str, int, str, str | None], list[OperatingDriverObservation]
    ] = defaultdict(list)
    for record in records:
        grouped[
            (
                record.segment_id,
                record.driver_id,
                record.fiscal_year,
                record.fiscal_period,
                record.period_key,
            )
        ].append(record)
    selected: dict[tuple[str, str, int], _SelectedObservation] = {}
    by_metric_period: dict[tuple[str, str, int], list[OperatingDriverObservation]] = (
        defaultdict(list)
    )
    for (
        segment_id,
        driver_id,
        year,
        _period,
        _period_key,
    ), candidates in grouped.items():
        by_metric_period[(segment_id, driver_id, year)].extend(candidates)
    for key, candidates in by_metric_period.items():
        winner = max(
            enumerate(candidates),
            key=lambda pair: (
                _ORIGIN_RANK.get(pair[1].origin, 0),
                _CONFIDENCE_RANK[pair[1].confidence],
                -pair[0],
            ),
        )[1]
        value = _observation_value(winner)
        constraint = _constraint_label(winner)
        selected[key] = _SelectedObservation(winner, value, constraint)
    return selected


def _find_input_observation(
    selected: Mapping[tuple[str, str, int], _SelectedObservation],
    segment_id: str,
    metric: str,
    year: int,
    definition: OperatingDriverDefinition,
) -> _SelectedObservation | None:
    candidates = [metric]
    normalized = metric.casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "arpu": ("average_revenue_per_user",),
        "average_revenue_per_user": ("arpu",),
        "subscribers": ("subscriber_count",),
        "subscriber_count": ("subscribers",),
        "transactions": ("transaction_count",),
        "transaction_count": ("transactions",),
        "stores": ("store_count",),
        "store_count": ("stores",),
        "sales_per_location": ("sales_per_store",),
        "sales_per_store": ("sales_per_location",),
        "conversion": ("conversion_rate",),
        "conversion_rate": ("conversion",),
        "growth_rate": ("growth",),
        "growth": ("growth_rate",),
    }
    candidates.extend(aliases.get(normalized, ()))
    if definition.archetype == OperatingArchetype.GENERIC_SEGMENT_GROWTH:
        candidates.extend((definition.driver_id, "segment_growth"))
    for candidate in candidates:
        result = selected.get((segment_id, candidate, year))
        if result is not None:
            return result
    return None


def _selected_periods_compatible(
    observations: Iterable[_SelectedObservation | None],
) -> bool:
    values = tuple(item for item in observations if item is not None)
    if len(values) < 2:
        return True
    first = values[0]
    return all(
        operating_periods_compatible(
            first.period,
            item.period,
            first.period_key,
            item.period_key,
        )
        for item in values[1:]
    )


def _generic_growth_fallback(
    definition: OperatingDriverDefinition,
    metric: str,
    segment_id: str,
    year: int,
    historical_revenue: Mapping[int, Decimal],
) -> _SelectedObservation | None:
    if definition.archetype != OperatingArchetype.GENERIC_SEGMENT_GROWTH:
        return None
    normalized_metric = metric.casefold().replace("-", "_").replace(" ", "_")
    if normalized_metric not in {"growth", "growth_rate", "segment_growth"}:
        return None
    current = historical_revenue.get(year - 1)
    previous = historical_revenue.get(year - 2)
    if current is None or previous is None or previous == 0:
        return None
    growth = current / previous - Decimal(1)
    unit = definition.units.get(metric, "ratio").casefold()
    if "%" in unit or "percent" in unit or "percentage" in unit:
        value = growth * Decimal(100)
    else:
        value = growth
    observation = OperatingDriverObservation(
        segment_id=segment_id,
        driver_id=metric,
        fiscal_year=year,
        value=value,
        unit=definition.units.get(metric, "ratio"),
        origin="derived",
        confidence="medium",
        method="generic_segment_growth_from_reported_revenue",
    )
    return _SelectedObservation(observation, value)


def _find_output_constraint(
    selected: Mapping[tuple[str, str, int], _SelectedObservation],
    segment_id: str,
    definition: OperatingDriverDefinition,
    year: int,
) -> _SelectedObservation | None:
    if definition.driver_id.casefold() in _SEGMENT_REVENUE_DRIVERS:
        return None
    candidates = (definition.driver_id, f"{definition.driver_id}_revenue")
    for candidate in candidates:
        result = selected.get((segment_id, candidate, year))
        if (
            result is not None
            and result.observation.origin == "management_guidance"
            and result.observation.driver_id not in definition.input_metrics
        ):
            return result
    return None


def _find_segment_revenue_constraint(
    selected: Mapping[tuple[str, str, int], _SelectedObservation],
    segment_id: str,
    years: Sequence[int],
) -> dict[int, _SelectedObservation]:
    """Return generic management revenue constraints once per segment/year."""

    allowed_years = set(years)
    by_year: dict[int, _SelectedObservation] = {}
    for (selected_segment, driver_id, year), candidate in selected.items():
        if (
            selected_segment != segment_id
            or year not in allowed_years
            or candidate.source != _MANAGEMENT_SOURCE
            or driver_id.casefold() not in _SEGMENT_REVENUE_DRIVERS
        ):
            continue
        previous = by_year.get(year)
        if previous is None or _segment_revenue_driver_rank(
            driver_id
        ) > _segment_revenue_driver_rank(previous.observation.driver_id):
            by_year[year] = candidate
    return by_year


def _segment_revenue_driver_rank(driver_id: str) -> int:
    return OPERATING_VOCABULARY.revenue_driver_priority.get(driver_id.casefold(), 0)


def _direct_revenue_observation(
    selected: Mapping[tuple[str, str, int], _SelectedObservation],
    segment_id: str,
    year: int,
    definitions: Sequence[OperatingDriverDefinition],
) -> _SelectedObservation | None:
    candidates = [definition.driver_id for definition in definitions]
    candidates.extend(("revenue", "segment_revenue"))
    for candidate in candidates:
        result = selected.get((segment_id, candidate, year))
        if result is not None and candidate not in {
            metric for definition in definitions for metric in definition.input_metrics
        }:
            return result
    return None


def _reported_revenue_observation_map(
    selected: Mapping[tuple[str, str, int], _SelectedObservation],
    segment_id: str,
) -> dict[int, Decimal]:
    result: dict[int, Decimal] = {}
    for (selected_segment, driver_id, year), selected_item in selected.items():
        if (
            selected_segment == segment_id
            and driver_id.casefold() in _SEGMENT_REVENUE_DRIVERS
            and selected_item.source != _MANAGEMENT_SOURCE
        ):
            result[year] = selected_item.value
    return result


def _apply_output_constraint(
    value: Decimal,
    constraint: _SelectedObservation,
) -> tuple[Decimal, str]:
    observation = constraint.observation
    if observation.value is not None:
        return observation.value, "management revenue point constraint"
    low = observation.low
    high = observation.high
    if low is not None and value < low:
        value = low
    if high is not None and value > high:
        value = high
    return value, constraint.constraint or "management revenue range constraint"


def _previous_revenue(
    year: int,
    selected_revenue_by_year: Mapping[int, Decimal],
    historical_revenue: Mapping[int, Decimal],
) -> Decimal | None:
    previous_year = year - 1
    if previous_year in selected_revenue_by_year:
        return selected_revenue_by_year[previous_year]
    return historical_revenue.get(previous_year)


def _observation_value(observation: OperatingDriverObservation) -> Decimal:
    if observation.value is not None:
        return observation.value * observation.scale
    if observation.low is not None and observation.high is not None:
        return (observation.low + observation.high) / Decimal(2) * observation.scale
    if observation.low is not None:
        return observation.low * observation.scale
    if observation.high is not None:
        return observation.high * observation.scale
    raise ValueError("Operating observation has no usable value")


def _constraint_label(observation: OperatingDriverObservation) -> str | None:
    if observation.origin != "management_guidance":
        return None
    if observation.value is not None:
        return "point"
    if observation.low is not None and observation.high is not None:
        return "range"
    if observation.low is not None:
        return "floor"
    if observation.high is not None:
        return "ceiling"
    return None


def _is_generic_unit_observation(observation: OperatingDriverObservation) -> bool:
    return observation.scale == Decimal(1) and observation.unit.casefold() == "units"


__all__ = [name for name in globals() if name.startswith("_")]
