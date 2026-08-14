"""Small provider-neutral deterministic operating forecast service."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Any

from edgarito.config.operating import OPERATING_VOCABULARY
from edgarito.schemas.operating import (
    CompanyOperatingForecast,
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverForecast,
    OperatingDriverObservation,
    OperatingSegment,
    SegmentRevenueForecast,
    canonical_operating_segment_id,
    operating_periods_compatible,
    operating_units_compatible,
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
_SEGMENT_REVENUE_DRIVERS = frozenset(OPERATING_VOCABULARY.revenue_driver_priority)
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_YEAR_MIN = 1900
_YEAR_MAX = 2200
_HIGH_DRIVER_COVERAGE = Decimal("0.80")
_MEDIUM_DRIVER_COVERAGE = Decimal("0.50")
_HIGH_RECONSTRUCTION_ERROR = Decimal("0.05")
_MEDIUM_RECONSTRUCTION_ERROR = Decimal("0.20")


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

    @property
    def period(self) -> str:
        return self.observation.fiscal_period

    @property
    def period_key(self) -> str | None:
        return self.observation.period_key


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


@dataclass(frozen=True)
class _ReconstructionAudit:
    """Deterministic validation of driver formulas against reported revenue."""

    coverage: Decimal | None
    error: Decimal | None
    error_by_year: dict[int, Decimal]
    supported_years: tuple[int, ...]
    confidence: str
    warnings: tuple[str, ...]
    genuine_coverage: Decimal | None = None
    derived_reconstruction_years: tuple[int, ...] = ()


@dataclass(frozen=True)
class _ConsolidationSelection:
    """Non-overlapping segment scopes eligible for company aggregation."""

    segments: tuple[OperatingSegment, ...]
    warnings: tuple[str, ...] = ()


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
        normalized_segments_by_id: dict[str, OperatingSegment] = {}
        for item in segments:
            segment = _coerce_segment(item)
            previous = normalized_segments_by_id.get(segment.segment_id)
            if previous is None or (
                previous.name == previous.segment_id
                and segment.name != segment.segment_id
            ):
                normalized_segments_by_id[segment.segment_id] = segment
        normalized_segments = tuple(normalized_segments_by_id.values())
        normalized_definitions = tuple(_coerce_definition(item) for item in definitions)
        for definition in normalized_definitions:
            if definition.segment_id not in normalized_segments_by_id:
                normalized_segments_by_id[definition.segment_id] = OperatingSegment(
                    segment_id=definition.segment_id,
                    name=definition.segment_id.replace("_", " ").title(),
                    scope="segment",
                    source="first_party_filing",
                    confidence="medium",
                )
        normalized_segments = tuple(normalized_segments_by_id.values())
        normalized_observations = tuple(
            _coerce_observation(item) for item in observations
        )

        normalized_segments = tuple(
            {segment.segment_id: segment for segment in normalized_segments}.values()
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
            # Definitions may arrive from a filing whose segment identity was
            # normalized after the segment collection was assembled. Materialize
            # a generic segment rather than aborting the valuation.
            for segment_id in sorted(unknown_definition_segments):
                normalized_segments += (
                    OperatingSegment(
                        segment_id=segment_id,
                        name=segment_id.replace("_", " ").title(),
                        scope="segment",
                        source="first_party_filing",
                        confidence="medium",
                    ),
                )
            normalized_segments = tuple(
                {
                    segment.segment_id: segment for segment in normalized_segments
                }.values()
            )

        consolidation = _select_consolidation_segments(normalized_segments)
        consolidation_ids = {segment.segment_id for segment in consolidation.segments}

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
        historical_selected_records = _select_observations(
            record for record in records if record.origin != "management_guidance"
        )
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
                historical_selected_records=historical_selected_records,
                fiscal_years=years,
                historical_revenue=historical.by_segment.get(segment.segment_id, {}),
            )
            for segment in normalized_segments
        )

        consolidated_forecasts = tuple(
            forecast
            for forecast in segment_forecasts
            if forecast.segment.segment_id in consolidation_ids
        )

        company_audit = _historical_company_reconstruction_audit(
            consolidation.segments,
            definitions_by_segment,
            historical_selected_records,
            historical,
            self.registry,
        )
        modeled_revenue_share = _modeled_revenue_share(
            consolidation.segments,
            segment_forecasts,
            historical,
            historical_selected_records,
            company_audit,
        )

        consolidated_revenue: list[Decimal] = []
        consolidated_sources: dict[int, str] = {}
        consolidated_confidences: dict[int, str] = {}
        warnings: list[str] = list(consolidation.warnings)

        for forecast in segment_forecasts:
            warnings.extend(
                f"{forecast.segment.segment_id}: {warning}"
                for warning in forecast.warnings
            )
        warnings.extend(f"company: {warning}" for warning in company_audit.warnings)

        for index, year in enumerate(years):
            segment_values = [
                forecast.revenue[index] for forecast in consolidated_forecasts
            ]
            segment_sources = [
                forecast.source_by_year[year] for forecast in consolidated_forecasts
            ]
            segment_confidences = [
                forecast.confidence_by_year[year] for forecast in consolidated_forecasts
            ]
            if consolidated_forecasts and all(
                source != _UNAVAILABLE_SOURCE for source in segment_sources
            ):
                value = sum(segment_values, Decimal(0))
                source = _combine_sources(segment_sources)
                confidence = _worst_confidence(segment_confidences)
                if source in {_INDEPENDENT_SOURCE, _CONSOLIDATION_SOURCE}:
                    confidence = _apply_company_reconstruction_confidence(
                        confidence,
                        company_audit,
                    )
            elif consolidated_forecasts and segment_sources:
                # A partial segment set is not a company total.  In particular,
                # do not let a supported management-constrained segment plus a
                # missing sibling become a low-confidence number that can be
                # mistaken for a complete independent company forecast.
                missing = ", ".join(
                    forecast.segment.segment_id
                    for forecast in consolidated_forecasts
                    if forecast.source_by_year[year] == _UNAVAILABLE_SOURCE
                )
                warnings.append(
                    f"FY{year}: consolidated operating revenue is incomplete; "
                    f"unavailable segment scope(s): {missing}"
                )
                if year in historical.company:
                    value = historical.company[year]
                    source = _HISTORICAL_SOURCE
                    confidence = "medium"
                else:
                    value = Decimal(0)
                    source = _UNAVAILABLE_SOURCE
                    confidence = "low"
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

        explicit_years = sorted(
            year
            for year, source in consolidated_sources.items()
            if source not in {_HISTORICAL_SOURCE, _UNAVAILABLE_SOURCE}
        )
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
            driver_coverage=company_audit.coverage,
            modeled_revenue_share=modeled_revenue_share,
            genuine_coverage=company_audit.genuine_coverage,
            reconstruction_error=company_audit.error,
            reconstruction_error_by_year=company_audit.error_by_year,
            derived_reconstruction_years=company_audit.derived_reconstruction_years,
            supported_years=company_audit.supported_years,
            own_supported_years=tuple(
                year
                for year, source in consolidated_sources.items()
                if _is_own_operating_source(source)
                and _CONFIDENCE_RANK[consolidated_confidences[year]]
                >= _CONFIDENCE_RANK["medium"]
            ),
            confidence=company_audit.confidence,
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
        historical_selected_records: Mapping[tuple[str, str, int], _SelectedObservation]
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
            historical_selected_records = _select_observations(
                record for record in records if record.origin != "management_guidance"
            )
        elif historical_selected_records is None:
            historical_selected_records = {
                key: selected
                for key, selected in selected_records.items()
                if selected.source != _MANAGEMENT_SOURCE
            }
        historical = _normalize_historical_revenue(
            historical_revenue,
            (normalized_segment,),
        )
        segment_history = historical.by_segment.get(normalized_segment.segment_id, {})
        return self._forecast_segment(
            normalized_segment,
            normalized_definitions,
            selected_records,
            historical_selected_records,
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
            self.registry,
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
                result, result_warnings = self._evaluate_definition(
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
                    self._record_input_forecasts(
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
                # A reported segment-revenue observation is a valid direct
                # operating path.  It must outrank a generic formula or a
                # formula reconstructed from the same revenue row.
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
                self._record_driver_forecast(
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
                self._record_driver_forecast(
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
            if (
                direct is not None
                and not formula_results
                and not applied_segment_constraint
            ):
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
                fallback = _generic_growth_fallback(
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

        if selected_inputs and not _selected_periods_compatible(selected_inputs):
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
                selected := _find_input_observation(
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
            and not _is_generic_unit_observation(selected.observation)
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
    """Normalize nested/tuple history to a company-only fiscal-year mapping.

    The operating engine may retain segment history for reconstruction, but
    reconciliation and FCFF fallback consume only consolidated company history.
    Segment totals are derived only from the same non-overlapping scope set used
    by consolidation; incomplete segment years are omitted rather than treated
    as zero.
    """

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


def _select_consolidation_segments(
    segments: Sequence[OperatingSegment],
) -> _ConsolidationSelection:
    """Choose a deterministic, non-overlapping company consolidation set.

    A consolidated scope is already a company total and therefore wins over
    all child scopes.  Otherwise only hierarchy roots are summed; supplied
    descendants remain available in ``segment_forecasts`` for audit purposes
    but are never added to their parent.
    """

    if not segments:
        return _ConsolidationSelection(())

    warnings: list[str] = []
    consolidated = tuple(
        segment for segment in segments if segment.scope == "consolidated"
    )
    if consolidated:
        selected = consolidated
        omitted = tuple(
            segment.segment_id
            for segment in segments
            if segment.segment_id not in {item.segment_id for item in selected}
        )
        if len(consolidated) > 1:
            warnings.append(
                "Multiple consolidated operating scopes supplied; only "
                "one consolidated scope can be used for the company total; "
                "consolidated revenue is unavailable"
            )
        if omitted:
            warnings.append(
                "Consolidated operating scope overlaps supplied child/other "
                "scopes; excluded from company sum: " + ", ".join(omitted)
            )
        return _ConsolidationSelection(selected, tuple(warnings))

    supplied_ids = {segment.segment_id for segment in segments}
    roots = tuple(
        segment
        for segment in segments
        if segment.parent_id is None or segment.parent_id not in supplied_ids
    )
    if not roots:
        return _ConsolidationSelection(
            (),
            (
                "Operating segment hierarchy has no supplied root scope; "
                "consolidated revenue is unavailable",
            ),
        )

    descendants = tuple(
        segment.segment_id
        for segment in segments
        if segment.segment_id not in {root.segment_id for root in roots}
    )
    if descendants:
        warnings.append(
            "Parent/child operating scopes overlap; excluded descendants from "
            "company sum: " + ", ".join(descendants)
        )
    root_scopes = {segment.scope for segment in roots}
    if len(root_scopes) > 1:
        return _ConsolidationSelection(
            (),
            tuple(
                [
                    *warnings,
                    "Operating segment scopes overlap across independent roots "
                    f"({', '.join(sorted(root_scopes))}); consolidated revenue is "
                    "unavailable",
                ]
            ),
        )
    return _ConsolidationSelection(roots, tuple(warnings))


def _find_output_constraint(
    selected: Mapping[tuple[str, str, int], _SelectedObservation],
    segment_id: str,
    definition: OperatingDriverDefinition,
    year: int,
) -> _SelectedObservation | None:
    # Generic revenue constraints are applied after all revenue components for
    # the segment have been aggregated.  Only a definition-specific output
    # constraint belongs inside this component evaluation.
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


def _historical_reconstruction_audit(
    segment_id: str,
    definitions: Sequence[OperatingDriverDefinition],
    selected_records: Mapping[tuple[str, str, int], _SelectedObservation],
    historical_revenue: Mapping[int, Decimal],
    registry: ArchetypeFormulaRegistry,
) -> _ReconstructionAudit:
    """Validate a segment's driver formulas against reported revenue history.

    The audit is intentionally separate from the forward path.  A historical
    revenue value remains the reported value even when its driver inputs do
    not reconstruct it; the result only determines whether the driver model
    is sufficiently supported for forward use.
    """

    reported = dict(historical_revenue)
    # A mapping explicitly supplied by the caller is authoritative.  Driver
    # observations fill only years that are otherwise absent.
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
    historical: _HistoricalRevenue,
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
            # A management constraint is not historical reported evidence.
            # Refusing to use it here is conservative and prevents a guidance
            # value from making an unverified driver model appear supported.
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


def _observation_source(observation: OperatingDriverObservation) -> str:
    if observation.origin == "management_guidance":
        return _MANAGEMENT_SOURCE
    return observation.origin


def _is_generic_unit_observation(observation: OperatingDriverObservation) -> bool:
    """Keep legacy fixture observations with an intentionally generic unit usable."""

    return observation.scale == Decimal(1) and observation.unit.casefold() == "units"


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


def _is_own_operating_source(source: str) -> bool:
    return source == _INDEPENDENT_SOURCE


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


def _modeled_revenue_share(
    consolidation_segments: Sequence[OperatingSegment],
    segment_forecasts: Sequence[SegmentRevenueForecast],
    historical: _HistoricalRevenue,
    historical_selected_records: Mapping[tuple[str, str, int], _SelectedObservation],
    company_audit: _ReconstructionAudit,
) -> Decimal | None:
    """Return revenue-weighted coverage for the consolidated segment scope.

    Historical formula coverage is a useful audit, but it is not the same as
    the portion of revenue represented by a usable segment path.  A direct
    reported segment-revenue observation, a generic growth path, or a complete
    formula path can therefore contribute to modeled share without making a
    derived driver look like independent evidence.
    """

    if not consolidation_segments or not segment_forecasts:
        return None

    forecast_by_id = {
        forecast.segment.segment_id: forecast for forecast in segment_forecasts
    }
    selected_ids = {segment.segment_id for segment in consolidation_segments}
    totals: dict[str, Decimal] = {}
    modeled: dict[str, Decimal] = {}
    historical_total = sum(historical.company.values(), Decimal(0))
    historical_supported = sum(
        historical.company[year]
        for year in company_audit.supported_years
        if year in historical.company
    )
    for segment in consolidation_segments:
        forecast = forecast_by_id.get(segment.segment_id)
        if forecast is None:
            continue
        history = dict(historical.by_segment.get(segment.segment_id, {}))
        history.update(
            _reported_revenue_observation_map(
                historical_selected_records, segment.segment_id
            )
        )
        if history:
            total = sum(history.values(), Decimal(0))
            share = forecast.modeled_revenue_share
            supported = total * share if share is not None else Decimal(0)
        else:
            total = sum(forecast.revenue, Decimal(0))
            share = forecast.modeled_revenue_share
            supported = total * share if share is not None else Decimal(0)
        if total > 0:
            totals[segment.segment_id] = total
            modeled[segment.segment_id] = supported

    if not totals:
        if historical_total == 0:
            return None
        return historical_supported / historical_total
    total = sum(totals.values(), Decimal(0))
    modeled_total = sum(
        modeled[segment_id] for segment_id in selected_ids if segment_id in modeled
    )
    if total == 0:
        return None
    return min(Decimal(1), modeled_total / total)


def _segment_modeled_revenue_share(
    historical_revenue: Mapping[int, Decimal],
    reported_revenue: Mapping[int, Decimal],
    supported_years: Sequence[int],
    *,
    forward_revenue: Sequence[Decimal] = (),
    forward_sources: Mapping[int, str] | None = None,
    generic_fallback: bool = False,
    fiscal_years: Sequence[int] = (),
    historical_years: Sequence[int] = (),
) -> Decimal | None:
    """Return revenue-weighted modeled coverage for one segment.

    Historical reconstruction coverage remains the legacy metric when history
    is present.  Forward direct revenue and generic-growth paths contribute
    modeled share without changing the causal reconstruction denominator.
    """

    reported = dict(historical_revenue)
    reported.update(reported_revenue)
    if reported:
        total = sum(reported.values(), Decimal(0))
        if total == 0:
            return None
        if generic_fallback and not supported_years:
            modeled = sum(reported.values(), Decimal(0))
        else:
            modeled = sum(
                reported[year] for year in supported_years if year in reported
            )
        if not supported_years and not generic_fallback and historical_years:
            # Preserve the legacy fiscal-history denominator when no usable
            # historical formula path exists; forward direct revenue still
            # contributes only when there is no history to audit.
            modeled = sum(
                reported[year] for year in reported_revenue if year in reported
            )
        elif reported_revenue:
            modeled += sum(
                reported[year]
                for year in reported_revenue
                if year in reported and year not in supported_years
            )
        if forward_revenue and forward_sources:
            # Keep coverage revenue-weighted across the complete comparable
            # path.  A history-only denominator can make forward modeled
            # revenue exceed 100%; skip years already represented by reported
            # observations to avoid double counting.
            reported_years = set(reported)
            for year, value in zip(fiscal_years, forward_revenue, strict=True):
                if year in reported_years:
                    continue
                total += value
                if (
                    value > 0
                    and forward_sources.get(year) != _UNAVAILABLE_SOURCE
                    and (
                        generic_fallback
                        or forward_sources.get(year) != _HISTORICAL_SOURCE
                    )
                ):
                    modeled += value
        if total == 0:
            return None
        return min(Decimal(1), max(Decimal(0), modeled / total))

    if not forward_revenue or not forward_sources:
        return None
    total = sum(forward_revenue, Decimal(0))
    if total == 0:
        return None
    modeled = sum(
        value
        for year, value in zip(fiscal_years, forward_revenue, strict=True)
        if forward_sources.get(year) != _UNAVAILABLE_SOURCE
        and (generic_fallback or forward_sources.get(year) != _HISTORICAL_SOURCE)
    )
    return min(Decimal(1), max(Decimal(0), modeled / total))


def _forecast_confidence(
    reconstruction_confidence: str,
    modeled_revenue_share: Decimal | None,
) -> str:
    """Keep the historical audit confidence separate from revenue coverage."""

    if modeled_revenue_share is None:
        return reconstruction_confidence
    if modeled_revenue_share < _MEDIUM_DRIVER_COVERAGE:
        return "low"
    if modeled_revenue_share < _HIGH_DRIVER_COVERAGE:
        return _worst_confidence((reconstruction_confidence, "medium"))
    return reconstruction_confidence


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
