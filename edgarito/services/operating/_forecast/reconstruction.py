"""Audit historical operating drivers before using them for forecasting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from edgarito.schemas.operating import (
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingSegment,
)
from edgarito.services.operating._forecast.contracts import (
    _HIGH_DRIVER_COVERAGE,
    _HIGH_RECONSTRUCTION_ERROR,
    _MANAGEMENT_SOURCE,
    _MEDIUM_DRIVER_COVERAGE,
    _MEDIUM_RECONSTRUCTION_ERROR,
    _ReconstructionAudit,
    _SelectedObservation,
    _worst_confidence,
)
from edgarito.services.operating._forecast.selection import (
    _find_input_observation,
    _selected_periods_compatible,
)
from edgarito.services.operating.registry import ArchetypeFormulaRegistry


def _historical_reconstruction_audit(
    segment_id: str,
    definitions: Sequence[OperatingDriverDefinition],
    selected_records: Mapping[tuple[str, str, int], _SelectedObservation],
    historical_revenue: Mapping[int, Decimal],
    registry: ArchetypeFormulaRegistry,
) -> _ReconstructionAudit:
    """Validate a segment's driver formulas against reported revenue history."""

    reported = dict(historical_revenue)
    _add_reported_revenue_observations(
        reported,
        segment_id=segment_id,
        definitions=definitions,
        selected_records=selected_records,
    )
    if not reported:
        return _ReconstructionAudit(
            coverage=None,
            error=None,
            error_by_year={},
            supported_years=(),
            confidence="low",
            warnings=(
                "historical driver reconstruction unavailable: no reported "
                "segment revenue history",
            ),
        )

    supported: list[int] = []
    genuine_supported: list[int] = []
    derived_years: list[int] = []
    errors: dict[int, Decimal] = {}
    warnings: list[str] = []
    for year in sorted(reported):
        reconstructed = _reconstruct_segment_revenue(
            segment_id,
            definitions,
            selected_records,
            year,
            reported,
            registry,
        )
        if reconstructed is None:
            warnings.append(
                f"FY{year}: reported segment revenue could not be reconstructed "
                "from complete historical driver inputs"
            )
            continue
        supported.append(year)
        if _has_genuine_reconstruction_inputs(
            segment_id, definitions, selected_records, year
        ):
            genuine_supported.append(year)
        else:
            derived_years.append(year)
        errors[year] = _relative_reconstruction_error(
            reconstructed,
            reported[year],
        )

    coverage = Decimal(len(supported)) / Decimal(len(reported))
    genuine_coverage = Decimal(len(genuine_supported)) / Decimal(len(reported))
    error = _mean(tuple(errors.values()))
    confidence = _reconstruction_confidence(coverage, error)
    warnings.extend(
        _reconstruction_quality_warnings(
            coverage,
            error,
            supported_years=tuple(supported),
            subject="segment",
            confidence=confidence,
        )
    )
    return _ReconstructionAudit(
        coverage=coverage,
        error=error,
        error_by_year=errors,
        supported_years=tuple(supported),
        confidence=confidence,
        warnings=tuple(warnings),
        genuine_coverage=genuine_coverage,
        derived_reconstruction_years=tuple(derived_years),
    )


def _historical_company_reconstruction_audit(
    segments: Sequence[OperatingSegment],
    definitions_by_segment: Mapping[str, Sequence[OperatingDriverDefinition]],
    selected_records: Mapping[tuple[str, str, int], _SelectedObservation],
    historical,
    registry: ArchetypeFormulaRegistry,
) -> _ReconstructionAudit:
    """Validate consolidated driver revenue against reported company history."""

    if not segments:
        return _ReconstructionAudit(
            coverage=None,
            error=None,
            error_by_year={},
            supported_years=(),
            confidence="low",
            warnings=(
                "historical company driver reconstruction unavailable: no "
                "operating segments were supplied",
            ),
        )

    reported = dict(historical.company)
    if not reported:
        segment_histories: dict[str, dict[int, Decimal]] = {}
        for segment in segments:
            segment_history = dict(historical.by_segment.get(segment.segment_id, {}))
            _add_reported_revenue_observations(
                segment_history,
                segment_id=segment.segment_id,
                definitions=definitions_by_segment.get(segment.segment_id, ()),
                selected_records=selected_records,
            )
            if segment_history:
                segment_histories[segment.segment_id] = segment_history
        for year in sorted(
            {year for values in segment_histories.values() for year in values}
        ):
            if all(year in values for values in segment_histories.values()) and len(
                segment_histories
            ) == len(segments):
                reported[year] = sum(
                    (values[year] for values in segment_histories.values()),
                    Decimal(0),
                )

    if not reported:
        return _ReconstructionAudit(
            coverage=None,
            error=None,
            error_by_year={},
            supported_years=(),
            confidence="low",
            warnings=(
                "historical company driver reconstruction unavailable: no "
                "reported company revenue history",
            ),
        )

    segment_histories = {
        segment.segment_id: dict(historical.by_segment.get(segment.segment_id, {}))
        for segment in segments
    }
    for segment in segments:
        _add_reported_revenue_observations(
            segment_histories[segment.segment_id],
            segment_id=segment.segment_id,
            definitions=definitions_by_segment.get(segment.segment_id, ()),
            selected_records=selected_records,
        )

    supported: list[int] = []
    genuine_supported: list[int] = []
    derived_years: list[int] = []
    errors: dict[int, Decimal] = {}
    warnings: list[str] = []
    for year in sorted(reported):
        reconstructed_values: list[Decimal] = []
        complete = True
        for segment in segments:
            reconstructed = _reconstruct_segment_revenue(
                segment.segment_id,
                definitions_by_segment.get(segment.segment_id, ()),
                selected_records,
                year,
                segment_histories[segment.segment_id],
                registry,
            )
            if reconstructed is None:
                complete = False
                break
            reconstructed_values.append(reconstructed)
        if not complete:
            warnings.append(
                f"FY{year}: reported company revenue could not be reconstructed "
                "from complete historical segment drivers"
            )
            continue
        supported.append(year)
        if all(
            _has_genuine_reconstruction_inputs(
                segment.segment_id,
                definitions_by_segment.get(segment.segment_id, ()),
                selected_records,
                year,
            )
            for segment in segments
        ):
            genuine_supported.append(year)
        else:
            derived_years.append(year)
        errors[year] = _relative_reconstruction_error(
            sum(reconstructed_values, Decimal(0)),
            reported[year],
        )

    coverage = Decimal(len(supported)) / Decimal(len(reported))
    genuine_coverage = Decimal(len(genuine_supported)) / Decimal(len(reported))
    error = _mean(tuple(errors.values()))
    confidence = _reconstruction_confidence(coverage, error)
    warnings.extend(
        _reconstruction_quality_warnings(
            coverage,
            error,
            supported_years=tuple(supported),
            subject="company",
            confidence=confidence,
        )
    )
    return _ReconstructionAudit(
        coverage=coverage,
        error=error,
        error_by_year=errors,
        supported_years=tuple(supported),
        confidence=confidence,
        warnings=tuple(warnings),
        genuine_coverage=genuine_coverage,
        derived_reconstruction_years=tuple(derived_years),
    )


def _reconstruct_segment_revenue(
    segment_id: str,
    definitions: Sequence[OperatingDriverDefinition],
    selected_records: Mapping[tuple[str, str, int], _SelectedObservation],
    year: int,
    historical_revenue: Mapping[int, Decimal],
    registry: ArchetypeFormulaRegistry,
) -> Decimal | None:
    """Evaluate every revenue definition for one historical fiscal year."""

    if not definitions:
        return None
    values: list[Decimal] = []
    for definition in definitions:
        inputs: dict[str, Decimal] = {}
        for metric in definition.required_inputs:
            selected = _find_input_observation(
                selected_records,
                segment_id,
                metric,
                year,
                definition,
            )
            if selected is None or selected.source == _MANAGEMENT_SOURCE:
                return None
            inputs[metric] = selected.value
            if not _selected_periods_compatible(
                tuple(
                    _find_input_observation(
                        selected_records,
                        segment_id,
                        required_metric,
                        year,
                        definition,
                    )
                    for required_metric in definition.required_inputs
                )
            ):
                return None
        for metric in definition.optional_inputs:
            selected = _find_input_observation(
                selected_records,
                segment_id,
                metric,
                year,
                definition,
            )
            if selected is not None and selected.source != _MANAGEMENT_SOURCE:
                inputs[metric] = selected.value

        previous_revenue = None
        if definition.archetype == OperatingArchetype.GENERIC_SEGMENT_GROWTH:
            previous_revenue = historical_revenue.get(year - 1)
            if previous_revenue is None:
                return None
        try:
            value = registry.evaluate(
                definition.archetype,
                inputs,
                units=definition.units,
                previous_revenue=previous_revenue,
            )
        except (KeyError, ValueError):
            return None
        if value < 0 or not value.is_finite():
            return None
        values.append(value)
    return sum(values, Decimal(0))


def _has_genuine_reconstruction_inputs(
    segment_id: str,
    definitions: Sequence[OperatingDriverDefinition],
    selected_records: Mapping[tuple[str, str, int], _SelectedObservation],
    year: int,
) -> bool:
    """Return whether reconstruction used non-derived reported inputs."""

    for definition in definitions:
        for metric in definition.required_inputs:
            selected = _find_input_observation(
                selected_records, segment_id, metric, year, definition
            )
            if selected is None or selected.source in {
                _MANAGEMENT_SOURCE,
                "derived",
            }:
                return False
    return bool(definitions)


def _add_reported_revenue_observations(
    target: dict[int, Decimal],
    *,
    segment_id: str,
    definitions: Sequence[OperatingDriverDefinition],
    selected_records: Mapping[tuple[str, str, int], _SelectedObservation],
) -> None:
    """Use explicit reported revenue observations when no mapping supplied it."""

    candidates = ["revenue", "segment_revenue"]
    candidates.extend(
        definition.driver_id
        for definition in definitions
        if definition.driver_id not in definition.input_metrics
    )
    for year_key, selected in selected_records.items():
        selected_segment, driver_id, year = year_key
        if selected_segment != segment_id or driver_id not in candidates:
            continue
        if selected.source == _MANAGEMENT_SOURCE:
            continue
        target.setdefault(year, selected.value)


def _relative_reconstruction_error(
    reconstructed: Decimal,
    reported: Decimal,
) -> Decimal:
    if reported == 0:
        return Decimal(0) if reconstructed == 0 else Decimal(1)
    return abs(reconstructed - reported) / abs(reported)


def _mean(values: Sequence[Decimal]) -> Decimal | None:
    if not values:
        return None
    return sum(values, Decimal(0)) / Decimal(len(values))


def _reconstruction_confidence(
    coverage: Decimal,
    error: Decimal | None,
) -> str:
    if error is None:
        return "low"
    if coverage >= _HIGH_DRIVER_COVERAGE and error <= _HIGH_RECONSTRUCTION_ERROR:
        return "high"
    if coverage >= _MEDIUM_DRIVER_COVERAGE and error <= _MEDIUM_RECONSTRUCTION_ERROR:
        return "medium"
    return "low"


def _apply_reconstruction_confidence(
    confidence: str,
    audit: _ReconstructionAudit,
) -> str:
    """Avoid treating an unvalidated segment path as high-confidence."""

    if audit.coverage is None:
        return "low"
    return _worst_confidence((confidence, audit.confidence))


def _apply_company_reconstruction_confidence(
    confidence: str,
    audit: _ReconstructionAudit,
) -> str:
    """Require a validated company history for an independent company path."""

    if audit.coverage is None:
        return "low"
    return _worst_confidence((confidence, audit.confidence))


def _reconstruction_quality_warnings(
    coverage: Decimal,
    error: Decimal | None,
    *,
    supported_years: tuple[int, ...],
    subject: str,
    confidence: str,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if coverage < _MEDIUM_DRIVER_COVERAGE:
        warnings.append(
            f"historical {subject} driver coverage is low ({coverage:.1%}); "
            "forward driver revenue is not fully supported"
        )
    elif coverage < _HIGH_DRIVER_COVERAGE:
        warnings.append(
            f"historical {subject} driver coverage is partial ({coverage:.1%})"
        )
    if error is not None and error > _MEDIUM_RECONSTRUCTION_ERROR:
        warnings.append(
            f"historical {subject} driver reconstruction error is high ({error:.1%})"
        )
    elif error is not None and error > _HIGH_RECONSTRUCTION_ERROR:
        warnings.append(
            f"historical {subject} driver reconstruction error is elevated "
            f"({error:.1%})"
        )
    if supported_years:
        years = ", ".join(f"FY{year}" for year in supported_years)
        warnings.append(f"historical {subject} driver supported years: {years}")
    warnings.append(f"historical {subject} driver confidence: {confidence}")
    return tuple(warnings)


__all__ = [name for name in globals() if name.startswith("_")]
