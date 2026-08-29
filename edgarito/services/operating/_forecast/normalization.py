"""Normalize operating forecast inputs and management constraints."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from edgarito.schemas.operating import (
    EvidenceReference,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
    canonical_operating_segment_id,
)
from edgarito.services.operating._forecast.consolidation import (
    _select_consolidation_segments,
)
from edgarito.services.operating._forecast.contracts import (
    _YEAR_MAX,
    _YEAR_MIN,
    _HistoricalRevenue,
)


def _coerce_segment(value: OperatingSegment | Mapping[str, Any]) -> OperatingSegment:
    return (
        value
        if isinstance(value, OperatingSegment)
        else OperatingSegment.model_validate(value)
    )


def _coerce_definition(
    value: OperatingDriverDefinition | Mapping[str, Any],
) -> OperatingDriverDefinition:
    return (
        value
        if isinstance(value, OperatingDriverDefinition)
        else OperatingDriverDefinition.model_validate(value)
    )


def _coerce_observation(
    value: OperatingDriverObservation | Mapping[str, Any],
) -> OperatingDriverObservation:
    return (
        value
        if isinstance(value, OperatingDriverObservation)
        else OperatingDriverObservation.model_validate(value)
    )


def _normalize_years(fiscal_years: Iterable[int]) -> tuple[int, ...]:
    years = tuple(int(year) for year in fiscal_years)
    if not years:
        raise ValueError("Operating forecast fiscal_years cannot be empty")
    if tuple(sorted(years)) != years or len(years) != len(set(years)):
        raise ValueError("Operating forecast fiscal_years must be sorted and unique")
    if any(year < _YEAR_MIN or year > _YEAR_MAX for year in years):
        raise ValueError(
            "Operating forecast fiscal_years are outside the supported range"
        )
    return years


def _to_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not decimal.is_finite():
        raise ValueError(f"{label} must be finite")
    return decimal


def _normalize_historical_revenue(
    value: Any,
    segments: Sequence[OperatingSegment],
) -> _HistoricalRevenue:
    company: dict[int, Decimal] = {}
    by_segment: dict[str, dict[int, Decimal]] = defaultdict(dict)
    if value is None:
        return _HistoricalRevenue(company, dict(by_segment))

    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            if _is_year_key(raw_key):
                company[_year_key(raw_key)] = _non_negative_revenue(
                    raw_value, f"historical revenue FY{raw_key}"
                )
                continue
            if isinstance(raw_key, tuple) and len(raw_key) == 2:
                segment_id, raw_year = raw_key
                year = _year_key(raw_year)
                canonical_segment_id = canonical_operating_segment_id(str(segment_id))
                by_segment[canonical_segment_id][year] = _non_negative_revenue(
                    raw_value,
                    f"historical revenue {segment_id} FY{year}",
                )
                continue
            if isinstance(raw_value, Mapping):
                segment_id = canonical_operating_segment_id(str(raw_key).strip())
                if not segment_id:
                    raise ValueError("Historical revenue segment ID cannot be blank")
                for raw_year, amount in raw_value.items():
                    year = _year_key(raw_year)
                    by_segment[segment_id][year] = _non_negative_revenue(
                        amount,
                        f"historical revenue {segment_id} FY{year}",
                    )
                continue
            raise ValueError(
                "Historical revenue mappings must be keyed by year or segment/year"
            )
    else:
        for item in value:
            if isinstance(item, OperatingDriverObservation):
                if item.driver_id.casefold() not in {
                    "revenue",
                    "segment_revenue",
                }:
                    continue
                by_segment[canonical_operating_segment_id(item.segment_id)][
                    item.fiscal_year
                ] = _non_negative_revenue(
                    _observation_value(item),
                    f"historical revenue {item.segment_id} FY{item.fiscal_year}",
                )
            elif isinstance(item, tuple) and len(item) == 3:
                segment_id, raw_year, amount = item
                year = _year_key(raw_year)
                canonical_segment_id = canonical_operating_segment_id(str(segment_id))
                by_segment[canonical_segment_id][year] = _non_negative_revenue(
                    amount,
                    f"historical revenue {segment_id} FY{year}",
                )
            else:
                raise ValueError(
                    "Historical revenue iterables must contain revenue observations"
                )

    if len(segments) == 1 and company:
        by_segment.setdefault(segments[0].segment_id, {}).update(company)
    return _HistoricalRevenue(company, dict(by_segment))


def normalize_company_historical_revenue(
    value: Any,
    segments: Iterable[OperatingSegment],
) -> dict[int, Decimal]:
    """Normalize nested/tuple history to a company-only fiscal-year mapping."""

    normalized_segments = tuple(_coerce_segment(item) for item in segments)
    historical = _normalize_historical_revenue(value, normalized_segments)
    company = dict(historical.company)
    selected = _select_consolidation_segments(normalized_segments).segments
    if company or not selected:
        return company
    histories = [
        historical.by_segment.get(segment.segment_id, {}) for segment in selected
    ]
    years = (
        set.intersection(*(set(history) for history in histories))
        if histories
        else set()
    )
    for year in years:
        company[year] = sum((history[year] for history in histories), Decimal(0))
    return dict(sorted(company.items()))


def _normalize_management_constraints(
    value: Any,
    *,
    segments: Sequence[OperatingSegment],
    definitions: Sequence[OperatingDriverDefinition],
) -> tuple[OperatingDriverObservation, ...]:
    if value is None:
        return ()
    if isinstance(value, (OperatingDriverObservation, Mapping)):
        if isinstance(value, OperatingDriverObservation):
            items: list[Any] = [value]
        elif {"driver_id", "fiscal_year"}.issubset(value):
            items = [value]
        else:
            items = list(_mapping_constraint_items(value, segments))
    else:
        if hasattr(value, "records"):
            items = list(value.records)
        elif hasattr(value, "eligible_records"):
            items = list(value.eligible_records)
        elif hasattr(value, "applications"):
            items = [item.guidance for item in value.applications]
        elif hasattr(value, "metric") and hasattr(value, "fiscal_year"):
            items = [value]
        else:
            items = list(value)

    units_by_key: dict[tuple[str, str], str] = {}
    for definition in definitions:
        for metric, unit in definition.units.items():
            units_by_key[(definition.segment_id, metric)] = unit
        units_by_key[(definition.segment_id, definition.driver_id)] = next(
            iter(definition.units.values()), "currency"
        )
    result: list[OperatingDriverObservation] = []
    for item in items:
        if isinstance(item, OperatingDriverObservation):
            result.append(item.model_copy(update={"origin": "management_guidance"}))
            continue
        if _is_tax_rate_guidance(item):
            observation = _management_guidance_to_observation(item)
            if observation is not None:
                result.append(observation)
            continue
        if isinstance(item, Mapping):
            result.append(
                _constraint_mapping_to_observation(
                    item,
                    units_by_key=units_by_key,
                    segments=segments,
                )
            )
            continue
        if isinstance(item, tuple) and len(item) in {3, 4}:
            result.append(
                _constraint_tuple_to_observation(
                    item,
                    units_by_key=units_by_key,
                    segments=segments,
                )
            )
            continue
        raise ValueError(
            "Management constraints must be operating observations or mappings"
        )
    return tuple(result)


def _is_tax_rate_guidance(value: Any) -> bool:
    metric = getattr(getattr(value, "metric", None), "value", getattr(value, "metric", None))
    return str(metric).strip().casefold() == "tax_rate" and hasattr(value, "period_type")


def _management_guidance_to_observation(
    value: Any,
) -> OperatingDriverObservation | None:
    """Convert only eligible, unambiguous tax guidance into operating evidence."""

    if not _is_tax_rate_guidance(value):
        return None
    if getattr(value, "evidence_verified", False) is not True:
        return None
    status = str(
        getattr(getattr(value, "status", None), "value", getattr(value, "status", ""))
    ).strip().casefold()
    if status in {"withdrawn", "superseded", "inactive"}:
        return None
    if getattr(value, "eligible", True) is False or getattr(value, "is_eligible", True) is False:
        return None
    scope = str(
        getattr(getattr(value, "scope", None), "value", getattr(value, "scope", ""))
    ).strip().casefold()
    if scope != "consolidated":
        return None

    period_type = str(
        getattr(
            getattr(value, "period_type", None),
            "value",
            getattr(value, "period_type", ""),
        )
    ).strip().casefold()
    fiscal_year = value.fiscal_year
    if fiscal_year is None:
        return None
    if period_type == "quarter":
        fiscal_quarter = value.fiscal_quarter
        if fiscal_quarter not in {1, 2, 3, 4}:
            return None
        fiscal_period = "FQ"
        period_key = f"Q{fiscal_quarter}"
    elif period_type == "fiscal_year":
        if value.fiscal_quarter is not None:
            return None
        fiscal_period = "FY"
        period_key = None
    else:
        # Multi-year and long-term guidance are not period-compatible tax-rate
        # evidence and must never silently become an FY observation.
        return None

    unit = str(getattr(value, "unit", "")).strip().casefold()
    if unit not in {"%", "percent", "percentage", "percentage_points", "pp"}:
        return None
    point = getattr(value, "point", None)
    if point is None:
        low = getattr(value, "low", None)
        high = getattr(value, "high", None)
        point = low if high is None else high if low is None else (low + high) / Decimal(2)
    reference = EvidenceReference(
        provider="sec",
        accession=getattr(value, "accession_number", None),
        filing_date=getattr(value, "filing_date", None),
        document_name=getattr(value, "source_document", None),
        supporting_text=getattr(value, "supporting_text", None),
    )
    extraction_confidence = getattr(value, "extraction_confidence", None)
    if extraction_confidence is None:
        confidence = "medium"
    elif extraction_confidence >= Decimal("0.9"):
        confidence = "high"
    elif extraction_confidence >= Decimal("0.5"):
        confidence = "medium"
    else:
        confidence = "low"
    return OperatingDriverObservation(
        segment_id="company",
        driver_id="tax_rate",
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        period_key=period_key,
        value=point,
        unit=str(getattr(value, "unit", "percent")),
        scope="company",
        scope_evidence="verified consolidated management tax-rate guidance",
        basis=str(
            getattr(getattr(value, "basis", None), "value", getattr(value, "basis", ""))
        ),
        origin="management_guidance",
        confidence=confidence,
        provenance=reference,
        evidence=reference,
        source_provenance=(reference,),
    )


def _mapping_constraint_items(
    value: Mapping[Any, Any],
    segments: Sequence[OperatingSegment],
) -> Iterable[Any]:
    for key, constraint in value.items():
        if isinstance(constraint, OperatingDriverObservation):
            yield constraint
            continue
        if isinstance(key, tuple) and len(key) == 3:
            segment_id, driver_id, year = key
            yield {
                "segment_id": segment_id,
                "driver_id": driver_id,
                "fiscal_year": year,
                "constraint": constraint,
            }
            continue
        if isinstance(key, tuple) and len(key) == 2:
            driver_id, year = key
            segment_ids = [segment.segment_id for segment in segments]
            if len(segment_ids) != 1:
                raise ValueError(
                    "Two-part management constraint keys require one segment"
                )
            yield {
                "segment_id": segment_ids[0],
                "driver_id": driver_id,
                "fiscal_year": year,
                "constraint": constraint,
            }
            continue
        if isinstance(constraint, Mapping):
            for year, nested_constraint in constraint.items():
                segment_id = str(key)
                yield {
                    "segment_id": segment_id,
                    "driver_id": nested_constraint.get("driver_id", "revenue"),
                    "fiscal_year": year,
                    "constraint": nested_constraint,
                }
            continue
        raise ValueError("Unsupported management constraint mapping key")


def _constraint_mapping_to_observation(
    value: Mapping[str, Any],
    *,
    units_by_key: Mapping[tuple[str, str], str],
    segments: Sequence[OperatingSegment],
) -> OperatingDriverObservation:
    segment_id = str(value.get("segment_id", "")).strip()
    driver_id = str(value.get("driver_id", "revenue")).strip()
    declared_scope = str(value.get("scope", "")).strip().casefold()
    fiscal_year = value.get("fiscal_year")
    if not segment_id and declared_scope in {"company", "consolidated", "total"}:
        segment_id = "company"
    elif not segment_id and len(segments) == 1:
        segment_id = segments[0].segment_id
    if fiscal_year is None:
        raise ValueError("Management constraint requires fiscal_year")
    raw_constraint = value.get("constraint", value)
    kwargs: dict[str, Any] = {
        "segment_id": segment_id,
        "driver_id": driver_id,
        "fiscal_year": fiscal_year,
        "unit": value.get(
            "unit",
            units_by_key.get(
                (segment_id, driver_id),
                "percent"
                if str(driver_id).strip().casefold().replace("-", "_").replace(" ", "_")
                in {"tax_rate", "effective_tax_rate"}
                else "currency",
            ),
        ),
        "origin": "management_guidance",
        "confidence": value.get("confidence", "high"),
    }
    if declared_scope:
        kwargs["scope"] = declared_scope
    if isinstance(raw_constraint, Mapping):
        for name in (
            "value",
            "low",
            "high",
            "currency",
            "basis",
            "provenance",
            "evidence",
        ):
            if name in raw_constraint:
                kwargs[name] = raw_constraint[name]
        if "point" in raw_constraint and "value" not in raw_constraint:
            kwargs["value"] = raw_constraint["point"]
    elif isinstance(raw_constraint, tuple) and len(raw_constraint) == 2:
        kwargs["low"], kwargs["high"] = raw_constraint
    else:
        kwargs["value"] = raw_constraint
    return OperatingDriverObservation(**kwargs)


def _constraint_tuple_to_observation(
    value: tuple[Any, ...],
    *,
    units_by_key: Mapping[tuple[str, str], str],
    segments: Sequence[OperatingSegment],
) -> OperatingDriverObservation:
    if len(value) == 3:
        driver_id, fiscal_year, constraint = value
        if len(segments) != 1:
            raise ValueError(
                "Three-part management constraint tuples require one segment"
            )
        segment_id = segments[0].segment_id
    elif len(value) == 4:
        segment_id, driver_id, fiscal_year, constraint = value
    else:
        raise ValueError(
            "Management constraint tuples must contain driver/year/value or "
            "segment/driver/year/value"
        )
    return _constraint_mapping_to_observation(
        {
            "segment_id": segment_id,
            "driver_id": driver_id,
            "fiscal_year": fiscal_year,
            "constraint": constraint,
            "unit": units_by_key.get(
                (str(segment_id), str(driver_id)),
                "percent"
                if str(driver_id).strip().casefold().replace("-", "_").replace(" ", "_")
                in {"tax_rate", "effective_tax_rate"}
                else "currency",
            ),
        },
        units_by_key=units_by_key,
        segments=segments,
    )


def _is_year_key(value: Any) -> bool:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return False
    return str(value).strip() == str(year) or isinstance(value, int)


def _year_key(value: Any) -> int:
    year = int(value)
    if year < _YEAR_MIN or year > _YEAR_MAX:
        raise ValueError(
            "Historical revenue fiscal year is outside the supported range"
        )
    return year


def _non_negative_revenue(value: Any, label: str) -> Decimal:
    decimal = _to_decimal(value, label)
    if decimal < 0:
        raise ValueError(f"{label} cannot be negative")
    return decimal


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


__all__ = [
    "_coerce_segment",
    "_coerce_definition",
    "_coerce_observation",
    "_normalize_years",
    "_normalize_historical_revenue",
    "_normalize_management_constraints",
    "normalize_company_historical_revenue",
]
