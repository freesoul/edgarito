"""Evaluate operating formulas and build segment revenue paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal
from typing import Any

from edgarito.schemas.operating import (
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverForecast,
    OperatingSegment,
    SegmentRevenueForecast,
    operating_units_compatible,
)
from edgarito.services.operating._forecast.consolidation import (
    _combine_formula_sources,
    _forecast_confidence,
    _growth_path,
    _is_own_operating_source,
    _segment_modeled_revenue_share,
)
from edgarito.services.operating._forecast.contracts import (
    _CONFIDENCE_RANK,
    _HISTORICAL_SOURCE,
    _MANAGEMENT_SOURCE,
    _UNAVAILABLE_SOURCE,
    _FormulaResult,
    _SelectedObservation,
    _worst_confidence,
)
from edgarito.services.operating._forecast.reconstruction import (
    _apply_reconstruction_confidence,
    _historical_reconstruction_audit,
)
from edgarito.services.operating._forecast.selection import (
    _apply_output_constraint,
    _direct_revenue_observation,
    _find_segment_revenue_constraint,
    _previous_revenue,
    _reported_revenue_observation_map,
)


def calculate_segment_forecast(
    service: Any,
    segment: OperatingSegment,
    definitions: tuple[OperatingDriverDefinition, ...],
    selected_records: Mapping[tuple[str, str, int], _SelectedObservation],
    historical_selected_records: Mapping[
        tuple[str, str, int], _SelectedObservation
    ],
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
    segment_revenue_constraints = _find_segment_revenue_constraint(
        selected_records,
        segment.segment_id,
        years,
    )
    reconstruction_audit = _historical_reconstruction_audit(
        segment.segment_id,
        definitions,
        historical_selected_records,
        historical_revenue,
        service.registry,
    )
    warnings.extend(reconstruction_audit.warnings)
    formula_history = dict(historical_revenue)
    formula_history.update(
        _reported_revenue_observation_map(selected_records, segment.segment_id)
    )
    direct_revenue_history = _reported_revenue_observation_map(
        historical_selected_records, segment.segment_id
    )

    for year in years:
        formula_results: list[_FormulaResult] = []
        segment_constraint = segment_revenue_constraints.get(year)
        for definition in definitions:
            result, result_warnings = service._evaluate_definition(
                segment,
                definition,
                selected_records,
                year,
                selected_revenue_by_year,
                formula_history,
            )
            warnings.extend(result_warnings)
            if result is not None:
                if result.source != _MANAGEMENT_SOURCE:
                    result = replace(
                        result,
                        confidence=_apply_reconstruction_confidence(
                            result.confidence,
                            reconstruction_audit,
                        ),
                    )
                formula_results.append(result)
                service._record_input_forecasts(
                    driver_forecasts,
                    segment,
                    definition,
                    result.inputs,
                    year,
                    warnings,
                )

        historical_value = historical_revenue.get(year)
        applied_segment_constraint = False
        direct = _direct_revenue_observation(
            selected_records,
            segment.segment_id,
            year,
            definitions,
        )
        if historical_value is not None:
            revenue = historical_value
            source = _HISTORICAL_SOURCE
            confidence = "medium"
        elif direct is not None and direct.source != _MANAGEMENT_SOURCE:
            revenue = direct.value
            source = direct.source
            confidence = direct.confidence
            if segment_constraint is not None:
                revenue, constraint_label = _apply_output_constraint(
                    revenue,
                    segment_constraint,
                )
                source = _MANAGEMENT_SOURCE
                confidence = segment_constraint.confidence
                applied_segment_constraint = True
            if source != _HISTORICAL_SOURCE:
                explicit_years.append(year)
        elif formula_results:
            revenue = sum((result.value for result in formula_results), Decimal(0))
            source = _combine_formula_sources(formula_results)
            confidence = _worst_confidence(
                [result.confidence for result in formula_results]
            )
            if segment_constraint is not None:
                revenue, constraint_label = _apply_output_constraint(
                    revenue,
                    segment_constraint,
                )
                source = _MANAGEMENT_SOURCE
                confidence = segment_constraint.confidence
                applied_segment_constraint = True
            if source != _HISTORICAL_SOURCE:
                explicit_years.append(year)
        else:
            if direct is not None:
                revenue = direct.value
                source = direct.source
                confidence = direct.confidence
                if segment_constraint is not None:
                    revenue, constraint_label = _apply_output_constraint(
                        revenue,
                        segment_constraint,
                    )
                    source = _MANAGEMENT_SOURCE
                    confidence = segment_constraint.confidence
                    applied_segment_constraint = True
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
            service._record_driver_forecast(
                driver_forecasts,
                output_forecast,
                warnings,
            )

        if applied_segment_constraint and segment_constraint is not None:
            constraint_forecast = OperatingDriverForecast(
                segment_id=segment.segment_id,
                driver_id=segment_constraint.observation.driver_id,
                fiscal_year=year,
                value=revenue,
                unit=segment.currency or "currency",
                source=_MANAGEMENT_SOURCE,
                method="segment aggregate revenue constraint",
                confidence=segment_constraint.confidence,
                constraint=constraint_label,
                provenance=segment_constraint.provenance,
            )
            service._record_driver_forecast(
                driver_forecasts,
                constraint_forecast,
                warnings,
            )

        direct = _direct_revenue_observation(
            selected_records,
            segment.segment_id,
            year,
            definitions,
        )
        if direct is not None and not formula_results and not applied_segment_constraint:
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
            service._record_driver_forecast(
                driver_forecasts,
                direct_forecast,
                warnings,
            )

    revenue_path = tuple(revenues)
    modeled_revenue_share = _segment_modeled_revenue_share(
        historical_revenue,
        direct_revenue_history,
        reconstruction_audit.supported_years,
        forward_revenue=revenue_path,
        forward_sources=sources,
        generic_fallback=any(
            definition.archetype == OperatingArchetype.GENERIC_SEGMENT_GROWTH
            for definition in definitions
        ),
        fiscal_years=years,
        historical_years=tuple(historical_revenue),
    )
    segment_confidence = _forecast_confidence(
        reconstruction_audit.confidence,
        modeled_revenue_share,
    )
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
        driver_coverage=reconstruction_audit.coverage,
        modeled_revenue_share=modeled_revenue_share,
        genuine_coverage=reconstruction_audit.genuine_coverage,
        reconstruction_error=reconstruction_audit.error,
        reconstruction_error_by_year=reconstruction_audit.error_by_year,
        derived_reconstruction_years=reconstruction_audit.derived_reconstruction_years,
        supported_years=reconstruction_audit.supported_years,
        own_supported_years=tuple(
            year
            for year, source in sources.items()
            if _is_own_operating_source(source)
            and _CONFIDENCE_RANK[confidences[year]] >= _CONFIDENCE_RANK["medium"]
        ),
        confidence=segment_confidence,
    )


def evaluate_definition(
    service: Any,
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
        selected = service._find_input_observation(
            selected_records,
            segment.segment_id,
            metric,
            year,
            definition,
        )
        if selected is None:
            fallback = service._generic_growth_fallback(
                definition,
                metric,
                segment.segment_id,
                year,
                historical_revenue,
            )
            if fallback is not None:
                selected = fallback
                inputs[metric] = selected.value
                selected_inputs.append(selected)
                continue
            warnings.append(
                f"FY{year} {definition.driver_id}: missing required input "
                f"'{metric}'"
            )
            continue
        inputs[metric] = selected.value
        selected_inputs.append(selected)
    for metric in definition.optional_inputs:
        selected = service._find_input_observation(
            selected_records,
            segment.segment_id,
            metric,
            year,
            definition,
        )
        if selected is not None:
            inputs[metric] = selected.value
            selected_inputs.append(selected)

    if selected_inputs and not service._selected_periods_compatible(selected_inputs):
        periods = ", ".join(
            f"{item.observation.driver_id}={item.observation.fiscal_period}"
            for item in selected_inputs
        )
        warnings.append(
            f"FY{year} {definition.driver_id}: incompatible operating evidence "
            f"periods ({periods})"
        )
        return None, warnings

    incompatible_units = [
        selected
        for metric in (*definition.required_inputs, *definition.optional_inputs)
        if (
            selected := service._find_input_observation(
                selected_records,
                segment.segment_id,
                metric,
                year,
                definition,
            )
        )
        is not None
        and metric in definition.units
        and definition.units[metric].casefold() not in {"unit", "unspecified"}
        and not service._is_generic_unit_observation(selected.observation)
        and not operating_units_compatible(
            definition.units[metric], selected.observation.unit
        )
    ]
    if incompatible_units:
        details = ", ".join(
            f"{item.observation.driver_id}={item.observation.unit}"
            for item in incompatible_units
        )
        warnings.append(
            f"FY{year} {definition.driver_id}: incompatible operating units "
            f"({details})"
        )
        return None, warnings

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

    output_constraint = service._find_output_constraint(
        selected_records,
        segment.segment_id,
        definition,
        year,
    )
    try:
        value = service.registry.evaluate(
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
    source = "independent_operating"
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


def record_input_forecasts(
    service: Any | None,
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
        if service is None:
            record_driver_forecast(forecasts, forecast, warnings)
        else:
            service._record_driver_forecast(forecasts, forecast, warnings)


def record_driver_forecast(
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


def _first_observation_provenance(
    observations: tuple[_SelectedObservation, ...] | list[_SelectedObservation],
) -> Any:
    for observation in observations:
        if observation.provenance is not None:
            return observation.provenance
    return None


__all__ = [
    "calculate_segment_forecast",
    "evaluate_definition",
    "record_input_forecasts",
    "record_driver_forecast",
]
