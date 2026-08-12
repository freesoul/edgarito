"""Small provider-neutral deterministic operating forecast service."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from edgarito.schemas.operating import (
    CompanyOperatingForecast,
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverForecast,
    OperatingDriverObservation,
    OperatingSegment,
    SegmentRevenueForecast,
)
from edgarito.services.operating.registry import (
    ARCHETYPE_FORMULAS,
    FORMULA_REGISTRY,
    ArchetypeFormulaRegistry,
)

_INDEPENDENT_SOURCE = "independent_operating"
_MANAGEMENT_SOURCE = "management_guidance"
_HISTORICAL_SOURCE = "normalized_historical"
_UNAVAILABLE_SOURCE = "unavailable"
_CONSOLIDATION_SOURCE = "mixed"
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_YEAR_MIN = 1900
_YEAR_MAX = 2200


@dataclass(frozen=True)
class _HistoricalRevenue:
    company: dict[int, Decimal]
    by_segment: dict[str, dict[int, Decimal]]


@dataclass(frozen=True)
class _SelectedObservation:
    observation: OperatingDriverObservation
    value: Decimal
    constraint: str | None = None

    @property
    def source(self) -> str:
        return _observation_source(self.observation)

    @property
    def confidence(self) -> str:
        return self.observation.confidence

    @property
    def provenance(self) -> Any:
        return self.observation.provenance or self.observation.evidence


@dataclass(frozen=True)
class _FormulaResult:
    definition: OperatingDriverDefinition
    value: Decimal
    source: str
    confidence: str
    provenance: Any
    method: str
    constraint: str | None
    inputs: tuple[_SelectedObservation, ...]


class OperatingForecastService:
    """Build deterministic segment and consolidated operating revenue paths.

    The service consumes already-normalized operating evidence.  It has no
    analyst-consensus input and never calls a provider.  Each requested fiscal
    year is represented in the returned contracts; when a formula cannot be
    evaluated, the service keeps a zero placeholder and emits a warning rather
    than fabricating a driver value.
    """

    def __init__(self, registry: ArchetypeFormulaRegistry | None = None):
        self.registry = registry or FORMULA_REGISTRY

    def forecast(
        self,
        segments: Iterable[OperatingSegment],
        definitions: Iterable[OperatingDriverDefinition],
        observations: Iterable[OperatingDriverObservation] = (),
        management_constraints: Any = (),
        historical_revenue: Any = None,
        fiscal_years: Iterable[int] = (),
        *,
        company_id: str = "company",
    ) -> CompanyOperatingForecast:
        """Return segment forecasts and their deterministic consolidation.

        ``historical_revenue`` accepts the architecture's company-level
        ``{year: value}`` mapping and also the useful segment-level forms
        ``{segment_id: {year: value}}`` and ``{(segment_id, year): value}``.
        Management constraints are normally
        ``OperatingDriverObservation`` instances with
        ``origin='management_guidance'``; simple mappings are accepted as a
        convenience for deterministic callers.
        """

        years = _normalize_years(fiscal_years)
        normalized_segments = tuple(_coerce_segment(item) for item in segments)
        normalized_definitions = tuple(_coerce_definition(item) for item in definitions)
        normalized_observations = tuple(
            _coerce_observation(item) for item in observations
        )

        segment_ids = [segment.segment_id for segment in normalized_segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("Operating segments must have unique segment IDs")
        unknown_definition_segments = {
            definition.segment_id
            for definition in normalized_definitions
            if definition.segment_id not in set(segment_ids)
        }
        if unknown_definition_segments:
            raise ValueError(
                "Operating definitions reference unknown segments: "
                + ", ".join(sorted(unknown_definition_segments))
            )

        historical = _normalize_historical_revenue(
            historical_revenue,
            normalized_segments,
        )
        constraint_records = _normalize_management_constraints(
            management_constraints,
            segments=normalized_segments,
            definitions=normalized_definitions,
        )
        records = (*normalized_observations, *constraint_records)
        selected_records = _select_observations(records)
        definitions_by_segment: dict[str, tuple[OperatingDriverDefinition, ...]] = {
            segment.segment_id: tuple(
                sorted(
                    (
                        definition
                        for definition in normalized_definitions
                        if definition.segment_id == segment.segment_id
                        and definition.output_metric == "revenue"
                    ),
                    key=lambda item: item.driver_id,
                )
            )
            for segment in normalized_segments
        }

        segment_forecasts = tuple(
            self.forecast_segment(
                segment,
                definitions_by_segment[segment.segment_id],
                selected_records=selected_records,
                fiscal_years=years,
                historical_revenue=historical.by_segment.get(segment.segment_id, {}),
            )
            for segment in normalized_segments
        )

        consolidated_revenue: list[Decimal] = []
        consolidated_sources: dict[int, str] = {}
        consolidated_confidences: dict[int, str] = {}
        warnings: list[str] = []
        explicit_years = sorted(
            {year for forecast in segment_forecasts for year in forecast.explicit_years}
        )

        for forecast in segment_forecasts:
            warnings.extend(
                f"{forecast.segment.segment_id}: {warning}"
                for warning in forecast.warnings
            )

        for index, year in enumerate(years):
            segment_values = [forecast.revenue[index] for forecast in segment_forecasts]
            segment_sources = [
                forecast.source_by_year[year] for forecast in segment_forecasts
            ]
            segment_confidences = [
                forecast.confidence_by_year[year] for forecast in segment_forecasts
            ]
            usable_sources = [
                source for source in segment_sources if source != _UNAVAILABLE_SOURCE
            ]
            if segment_forecasts and usable_sources:
                value = sum(segment_values, Decimal(0))
                source = _combine_sources(segment_sources)
                confidence = _worst_confidence(segment_confidences)
            elif year in historical.company:
                value = historical.company[year]
                source = _HISTORICAL_SOURCE
                confidence = "medium"
            else:
                value = Decimal(0)
                source = _UNAVAILABLE_SOURCE
                confidence = "low"
                if not segment_forecasts:
                    warnings.append(
                        f"FY{year}: no operating segments or historical revenue"
                    )
            consolidated_revenue.append(value)
            consolidated_sources[year] = source
            consolidated_confidences[year] = confidence

        consolidated_growth = _growth_path(tuple(consolidated_revenue))
        transition_start_year = explicit_years[-1] + 1 if explicit_years else None
        return CompanyOperatingForecast(
            company_id=company_id,
            fiscal_years=years,
            segment_forecasts=segment_forecasts,
            consolidated_revenue=tuple(consolidated_revenue),
            consolidated_growth=consolidated_growth,
            explicit_years=tuple(explicit_years),
            transition_start_year=transition_start_year,
            source_by_year=consolidated_sources,
            confidence_by_year=consolidated_confidences,
            warnings=tuple(dict.fromkeys(warnings)),
            unit=_company_unit(normalized_segments),
        )

    def forecast_segment(
        self,
        segment: OperatingSegment,
        definitions: Iterable[OperatingDriverDefinition],
        observations: Iterable[OperatingDriverObservation] = (),
        management_constraints: Any = (),
        historical_revenue: Any = None,
        fiscal_years: Iterable[int] = (),
        *,
        selected_records: Mapping[tuple[str, str, int], _SelectedObservation]
        | None = None,
    ) -> SegmentRevenueForecast:
        """Build one segment path; public for formula-focused callers."""

        normalized_segment = _coerce_segment(segment)
        years = _normalize_years(fiscal_years)
        normalized_definitions = tuple(_coerce_definition(item) for item in definitions)
        if any(
            definition.segment_id != normalized_segment.segment_id
            for definition in normalized_definitions
        ):
            raise ValueError("Operating definition segment_id must match the segment")

        if selected_records is None:
            records = tuple(_coerce_observation(item) for item in observations)
            records += _normalize_management_constraints(
                management_constraints,
                segments=(normalized_segment,),
                definitions=normalized_definitions,
            )
            selected_records = _select_observations(records)
        historical = _normalize_historical_revenue(
            historical_revenue,
            (normalized_segment,),
        )
        segment_history = historical.by_segment.get(normalized_segment.segment_id, {})
        return self._forecast_segment(
            normalized_segment,
            normalized_definitions,
            selected_records,
            years,
            segment_history,
        )

    # ``build`` is a small compatibility seam for callers that use builder
    # terminology while keeping ``forecast`` as the primary API.
    build = forecast
    forecast_company = forecast

    def _forecast_segment(
        self,
        segment: OperatingSegment,
        definitions: tuple[OperatingDriverDefinition, ...],
        selected_records: Mapping[tuple[str, str, int], _SelectedObservation],
        years: tuple[int, ...],
        historical_revenue: Mapping[int, Decimal],
    ) -> SegmentRevenueForecast:
        revenues: list[Decimal] = []
        sources: dict[int, str] = {}
        confidences: dict[int, str] = {}
        explicit_years: list[int] = []
        driver_forecasts: dict[tuple[str, int], OperatingDriverForecast] = {}
        warnings: list[str] = []
        selected_revenue_by_year: dict[int, Decimal] = {}

        for year in years:
            formula_results: list[_FormulaResult] = []
            for definition in definitions:
                result, result_warnings = self._evaluate_definition(
                    segment,
                    definition,
                    selected_records,
                    year,
                    selected_revenue_by_year,
                    historical_revenue,
                )
                warnings.extend(result_warnings)
                if result is not None:
                    formula_results.append(result)
                    self._record_input_forecasts(
                        driver_forecasts,
                        segment,
                        definition,
                        result.inputs,
                        year,
                        warnings,
                    )

            historical_value = historical_revenue.get(year)
            if historical_value is not None:
                revenue = historical_value
                source = _HISTORICAL_SOURCE
                confidence = "medium"
            elif formula_results:
                revenue = sum((result.value for result in formula_results), Decimal(0))
                source = _combine_formula_sources(formula_results)
                confidence = _worst_confidence(
                    [result.confidence for result in formula_results]
                )
                if source != _HISTORICAL_SOURCE:
                    explicit_years.append(year)
            else:
                direct = _direct_revenue_observation(
                    selected_records,
                    segment.segment_id,
                    year,
                    definitions,
                )
                if direct is not None:
                    revenue = direct.value
                    source = direct.source
                    confidence = direct.confidence
                    if source == _MANAGEMENT_SOURCE:
                        explicit_years.append(year)
                else:
                    revenue = Decimal(0)
                    source = _UNAVAILABLE_SOURCE
                    confidence = "low"
                    warnings.append(f"FY{year}: no usable operating revenue formula")

            if revenue < 0 or not revenue.is_finite():
                warnings.append(
                    f"FY{year}: operating revenue result was negative or non-finite"
                )
                revenue = historical_revenue.get(year, Decimal(0))
                source = (
                    _HISTORICAL_SOURCE
                    if year in historical_revenue
                    else _UNAVAILABLE_SOURCE
                )
                confidence = "medium" if source == _HISTORICAL_SOURCE else "low"

            revenues.append(revenue)
            selected_revenue_by_year[year] = revenue
            sources[year] = source
            confidences[year] = confidence

            for result in formula_results:
                output_forecast = OperatingDriverForecast(
                    segment_id=segment.segment_id,
                    driver_id=result.definition.driver_id,
                    fiscal_year=year,
                    value=result.value,
                    unit=segment.currency or "currency",
                    source=result.source,
                    method=result.method,
                    confidence=result.confidence,
                    constraint=result.constraint,
                    provenance=result.provenance,
                )
                self._record_driver_forecast(
                    driver_forecasts,
                    output_forecast,
                    warnings,
                )

            direct = _direct_revenue_observation(
                selected_records,
                segment.segment_id,
                year,
                definitions,
            )
            if direct is not None and not formula_results:
                direct_forecast = OperatingDriverForecast(
                    segment_id=segment.segment_id,
                    driver_id=direct.observation.driver_id,
                    fiscal_year=year,
                    value=direct.value,
                    unit=segment.currency or "currency",
                    source=direct.source,
                    method="observed revenue",
                    confidence=direct.confidence,
                    constraint=direct.constraint,
                    provenance=direct.provenance,
                )
                self._record_driver_forecast(
                    driver_forecasts,
                    direct_forecast,
                    warnings,
                )

        revenue_path = tuple(revenues)
        return SegmentRevenueForecast(
            segment=segment,
            fiscal_years=years,
            revenue=revenue_path,
            revenue_growth=_growth_path(revenue_path),
            driver_forecasts=tuple(
                sorted(
                    driver_forecasts.values(),
                    key=lambda item: (item.fiscal_year, item.driver_id),
                )
            ),
            explicit_years=tuple(sorted(set(explicit_years))),
            source_by_year=sources,
            confidence_by_year=confidences,
            warnings=tuple(dict.fromkeys(warnings)),
            unit=segment.currency or "currency",
        )

    def _evaluate_definition(
        self,
        segment: OperatingSegment,
        definition: OperatingDriverDefinition,
        selected_records: Mapping[tuple[str, str, int], _SelectedObservation],
        year: int,
        selected_revenue_by_year: Mapping[int, Decimal],
        historical_revenue: Mapping[int, Decimal],
    ) -> tuple[_FormulaResult | None, list[str]]:
        warnings: list[str] = []
        inputs: dict[str, Decimal] = {}
        selected_inputs: list[_SelectedObservation] = []
        for metric in definition.required_inputs:
            selected = _find_input_observation(
                selected_records,
                segment.segment_id,
                metric,
                year,
                definition,
            )
            if selected is None:
                warnings.append(
                    f"FY{year} {definition.driver_id}: missing required input "
                    f"'{metric}'"
                )
                continue
            inputs[metric] = selected.value
            selected_inputs.append(selected)
        for metric in definition.optional_inputs:
            selected = _find_input_observation(
                selected_records,
                segment.segment_id,
                metric,
                year,
                definition,
            )
            if selected is not None:
                inputs[metric] = selected.value
                selected_inputs.append(selected)

        previous_revenue = None
        if definition.archetype == OperatingArchetype.GENERIC_SEGMENT_GROWTH:
            previous_revenue = _previous_revenue(
                year,
                selected_revenue_by_year,
                historical_revenue,
            )
            if previous_revenue is None:
                warnings.append(
                    f"FY{year} {definition.driver_id}: missing required input "
                    "'previous_revenue'"
                )

        if any(metric not in inputs for metric in definition.required_inputs):
            return None, warnings
        if definition.archetype == OperatingArchetype.GENERIC_SEGMENT_GROWTH and (
            previous_revenue is None
        ):
            return None, warnings

        output_constraint = _find_output_constraint(
            selected_records,
            segment.segment_id,
            definition,
            year,
        )
        try:
            value = self.registry.evaluate(
                definition.archetype,
                inputs,
                units=definition.units,
                previous_revenue=previous_revenue,
            )
        except (KeyError, ValueError) as error:
            warnings.append(
                f"FY{year} {definition.driver_id}: formula unavailable ({error})"
            )
            return None, warnings

        constraint = None
        source = _INDEPENDENT_SOURCE
        confidence = _worst_confidence(
            [selected.confidence for selected in selected_inputs]
        )
        provenance = _first_observation_provenance(selected_inputs)
        if any(selected.source == _MANAGEMENT_SOURCE for selected in selected_inputs):
            source = _MANAGEMENT_SOURCE
            confidence = _worst_confidence(
                [selected.confidence for selected in selected_inputs]
            )
            constraint = "management input constraint"

        if output_constraint is not None:
            value, constraint = _apply_output_constraint(
                value,
                output_constraint,
            )
            source = _MANAGEMENT_SOURCE
            confidence = output_constraint.confidence
            provenance = output_constraint.provenance

        return (
            _FormulaResult(
                definition=definition,
                value=value,
                source=source,
                confidence=confidence,
                provenance=provenance,
                method=f"formula:{definition.formula_id}",
                constraint=constraint,
                inputs=tuple(selected_inputs),
            ),
            warnings,
        )

    @staticmethod
    def _record_input_forecasts(
        forecasts: dict[tuple[str, int], OperatingDriverForecast],
        segment: OperatingSegment,
        definition: OperatingDriverDefinition,
        inputs: tuple[_SelectedObservation, ...],
        year: int,
        warnings: list[str],
    ) -> None:
        for selected in inputs:
            metric = selected.observation.driver_id
            forecast = OperatingDriverForecast(
                segment_id=segment.segment_id,
                driver_id=metric,
                fiscal_year=year,
                value=selected.value,
                unit=definition.units.get(metric, "unit"),
                source=selected.source,
                method=(
                    "management constraint"
                    if selected.source == _MANAGEMENT_SOURCE
                    else "observed input"
                ),
                confidence=selected.confidence,
                constraint=selected.constraint,
                provenance=selected.provenance,
            )
            OperatingForecastService._record_driver_forecast(
                forecasts,
                forecast,
                warnings,
            )

    @staticmethod
    def _record_driver_forecast(
        forecasts: dict[tuple[str, int], OperatingDriverForecast],
        forecast: OperatingDriverForecast,
        warnings: list[str],
    ) -> None:
        key = (forecast.driver_id, forecast.fiscal_year)
        previous = forecasts.get(key)
        if previous is None:
            forecasts[key] = forecast
            return
        if previous.value != forecast.value:
            warnings.append(
                f"FY{forecast.fiscal_year} {forecast.driver_id}: conflicting "
                "driver forecasts; first deterministic value retained"
            )


DeterministicOperatingForecastService = OperatingForecastService
OperatingForecastEngine = OperatingForecastService


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
                by_segment[str(segment_id)][year] = _non_negative_revenue(
                    raw_value,
                    f"historical revenue {segment_id} FY{year}",
                )
                continue
            if isinstance(raw_value, Mapping):
                segment_id = str(raw_key).strip()
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
                by_segment[item.segment_id][item.fiscal_year] = _non_negative_revenue(
                    _observation_value(item),
                    f"historical revenue {item.segment_id} FY{item.fiscal_year}",
                )
            elif isinstance(item, tuple) and len(item) == 3:
                segment_id, raw_year, amount = item
                year = _year_key(raw_year)
                by_segment[str(segment_id)][year] = _non_negative_revenue(
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
        else:
            items = list(_mapping_constraint_items(value, segments))
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
        if isinstance(item, Mapping):
            observation = _constraint_mapping_to_observation(
                item,
                units_by_key=units_by_key,
                segments=segments,
            )
            result.append(observation)
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
    fiscal_year = value.get("fiscal_year")
    if not segment_id and len(segments) == 1:
        segment_id = segments[0].segment_id
    if fiscal_year is None:
        raise ValueError("Management constraint requires fiscal_year")
    raw_constraint = value.get("constraint", value)
    kwargs: dict[str, Any] = {
        "segment_id": segment_id,
        "driver_id": driver_id,
        "fiscal_year": fiscal_year,
        "unit": value.get(
            "unit", units_by_key.get((segment_id, driver_id), "currency")
        ),
        "origin": "management_guidance",
        "confidence": value.get("confidence", "high"),
    }
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
            "unit": units_by_key.get((str(segment_id), str(driver_id)), "currency"),
        },
        units_by_key=units_by_key,
        segments=segments,
    )


def _select_observations(
    records: Iterable[OperatingDriverObservation],
) -> dict[tuple[str, str, int], _SelectedObservation]:
    grouped: dict[tuple[str, str, int], list[OperatingDriverObservation]] = defaultdict(
        list
    )
    for record in records:
        grouped[(record.segment_id, record.driver_id, record.fiscal_year)].append(
            record
        )
    selected: dict[tuple[str, str, int], _SelectedObservation] = {}
    for key, candidates in grouped.items():
        winner = max(
            enumerate(candidates),
            key=lambda pair: (
                2 if pair[1].origin == "management_guidance" else 1,
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


def _find_output_constraint(
    selected: Mapping[tuple[str, str, int], _SelectedObservation],
    segment_id: str,
    definition: OperatingDriverDefinition,
    year: int,
) -> _SelectedObservation | None:
    candidates = (
        definition.driver_id,
        definition.output_metric,
        "revenue",
        f"{definition.driver_id}_revenue",
    )
    for candidate in candidates:
        result = selected.get((segment_id, candidate, year))
        if (
            result is not None
            and result.observation.origin == "management_guidance"
            and result.observation.driver_id not in definition.input_metrics
        ):
            return result
    return None


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
        return observation.value
    if observation.low is not None and observation.high is not None:
        return (observation.low + observation.high) / Decimal(2)
    if observation.low is not None:
        return observation.low
    if observation.high is not None:
        return observation.high
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


def _observation_source(observation: OperatingDriverObservation) -> str:
    if observation.origin == "management_guidance":
        return _MANAGEMENT_SOURCE
    return observation.origin


def _first_observation_provenance(
    observations: Sequence[_SelectedObservation],
) -> Any:
    for observation in observations:
        if observation.provenance is not None:
            return observation.provenance
    return None


def _first_provenance(results: Sequence[_FormulaResult]) -> Any:
    for result in results:
        if result.provenance is not None:
            return result.provenance
    return None


def _combine_formula_sources(results: Sequence[_FormulaResult]) -> str:
    sources = {result.source for result in results}
    if len(sources) == 1:
        return next(iter(sources))
    return _CONSOLIDATION_SOURCE


def _combine_sources(sources: Sequence[str]) -> str:
    unique = set(sources)
    if len(unique) == 1:
        return next(iter(unique))
    return _CONSOLIDATION_SOURCE


def _worst_confidence(confidences: Iterable[str]) -> str:
    values = tuple(confidences)
    if not values:
        return "low"
    return min(values, key=lambda value: _CONFIDENCE_RANK[value])


def _growth_path(revenue: tuple[Decimal, ...]) -> tuple[Decimal | None, ...]:
    growth: list[Decimal | None] = [None]
    for previous, current in zip(revenue[:-1], revenue[1:], strict=True):
        if previous == 0:
            growth.append(Decimal(0) if current == 0 else None)
        else:
            growth.append((current / previous - Decimal(1)) * Decimal(100))
    return tuple(growth)


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


def _company_unit(segments: Sequence[OperatingSegment]) -> str:
    currencies = {segment.currency for segment in segments if segment.currency}
    return next(iter(currencies)) if len(currencies) == 1 else "currency"


__all__ = [
    "ARCHETYPE_FORMULAS",
    "ArchetypeFormulaRegistry",
    "CompanyOperatingForecast",
    "DeterministicOperatingForecastService",
    "FORMULA_REGISTRY",
    "OperatingForecastEngine",
    "OperatingForecastService",
    "SegmentRevenueForecast",
]
