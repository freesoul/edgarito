"""Orchestration for deterministic operating forecasts.

Input normalization, observation selection, historical reconstruction, and
consolidation live in focused modules. This service owns the workflow and the
small delegation methods used by the calculation stages.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any

from edgarito.schemas.operating import (
    CompanyOperatingForecast,
    OperatingDriverDefinition,
    OperatingDriverForecast,
    OperatingDriverObservation,
    OperatingSegment,
    SegmentRevenueForecast,
)
from edgarito.services.operating._forecast.calculation import (
    calculate_segment_forecast,
    evaluate_definition,
    record_driver_forecast,
    record_input_forecasts,
)
from edgarito.services.operating._forecast.consolidation import (
    _CONSOLIDATION_SOURCE,
    _combine_sources,
    _company_unit,
    _growth_path,
    _is_own_operating_source,
    _modeled_revenue_share,
    _select_consolidation_segments,
    _worst_confidence,
)
from edgarito.services.operating._forecast.contracts import (
    _CONFIDENCE_RANK,
    _HISTORICAL_SOURCE,
    _INDEPENDENT_SOURCE,
    _MANAGEMENT_SOURCE,
    _UNAVAILABLE_SOURCE,
    _FormulaResult,
    _SelectedObservation,
)
from edgarito.services.operating._forecast.economics import (
    OperatingEconomicsForecastService,
)
from edgarito.services.operating._forecast.normalization import (
    _coerce_definition,
    _coerce_observation,
    _coerce_segment,
    _normalize_historical_revenue,
    _normalize_management_constraints,
    _normalize_years,
    normalize_company_historical_revenue,
)
from edgarito.services.operating._forecast.reconstruction import (
    _apply_company_reconstruction_confidence,
    _historical_company_reconstruction_audit,
)
from edgarito.services.operating._forecast.selection import (
    _find_input_observation,
    _find_output_constraint,
    _find_segment_revenue_constraint,
    _generic_growth_fallback,
    _is_generic_unit_observation,
    _select_observations,
    _selected_periods_compatible,
)
from edgarito.services.operating.registry import (
    ARCHETYPE_FORMULAS,
    FORMULA_REGISTRY,
    ArchetypeFormulaRegistry,
)


class OperatingForecastService:
    """Build deterministic segment and consolidated operating revenue paths."""

    def __init__(
        self,
        registry: ArchetypeFormulaRegistry | None = None,
        economics_service: OperatingEconomicsForecastService | None = None,
    ):
        self.registry = registry or FORMULA_REGISTRY
        self.economics_service = economics_service or OperatingEconomicsForecastService()

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
        plan: Any | None = None,
        forecast_plan: Any | None = None,
        overrides: Any = (),
        forecast_overrides: Any | None = None,
        economics_config: Any | None = None,
        operating_economics_config: Any | None = None,
    ) -> CompanyOperatingForecast:
        """Normalize evidence, forecast each segment, and consolidate roots."""

        years = _normalize_years(fiscal_years)
        normalized_segments_by_id: dict[str, OperatingSegment] = {}
        ambiguous_segment_ids: set[str] = set()
        for item in segments:
            segment = _coerce_segment(item)
            previous = normalized_segments_by_id.get(segment.segment_id)
            if previous is not None:
                ambiguous_segment_ids.add(segment.segment_id)
            if previous is None or (
                previous.name == previous.segment_id and segment.name != segment.segment_id
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
                {segment.segment_id: segment for segment in normalized_segments}.values()
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
        company_forecast = CompanyOperatingForecast(
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
        economics_requested = _has_economics_inputs(
            records,
            plan if plan is not None else forecast_plan,
            overrides if forecast_overrides is None else forecast_overrides,
        )
        if economics_requested and not normalized_segments and _has_explicit_economics_target(
            plan if plan is not None else forecast_plan,
            overrides if forecast_overrides is None else forecast_overrides,
        ):
            raise ValueError(
                "Explicit gross-economics target does not match a supplied canonical segment"
            )
        if economics_requested and (normalized_segments or company_forecast is not None):
            economics = self.economics_service.forecast(
                segments=normalized_segments,
                segment_revenue_forecasts=segment_forecasts,
                observations=records,
                fiscal_years=years,
                revenue_forecast=company_forecast,
                plan=plan,
                forecast_plan=forecast_plan,
                overrides=overrides,
                forecast_overrides=forecast_overrides,
                company_id=company_id,
                config=(
                    economics_config
                    if economics_config is not None
                    else operating_economics_config
                ),
                ambiguous_segment_ids=ambiguous_segment_ids,
            )
            economics_by_id = {
                item.segment.segment_id: item for item in economics.segment_economics
            }
            company_forecast = company_forecast.model_copy(
                update={
                    "segment_forecasts": tuple(
                        item.model_copy(
                            update={
                                "operating_economics": economics_by_id.get(
                                    item.segment.segment_id
                                )
                            }
                        )
                        for item in segment_forecasts
                    ),
                    "operating_economics": economics,
                }
            )
        return company_forecast

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
        historical_selected_records: Mapping[
            tuple[str, str, int], _SelectedObservation
        ]
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
        return self._forecast_segment(
            normalized_segment,
            normalized_definitions,
            selected_records,
            historical_selected_records,
            years,
            historical.by_segment.get(normalized_segment.segment_id, {}),
        )

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
        return calculate_segment_forecast(
            self,
            segment,
            definitions,
            selected_records,
            historical_selected_records,
            years,
            historical_revenue,
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
        return evaluate_definition(
            self,
            segment,
            definition,
            selected_records,
            year,
            selected_revenue_by_year,
            historical_revenue,
        )

    @staticmethod
    def _first_observation_provenance(observations):
        for observation in observations:
            if observation.provenance is not None:
                return observation.provenance
        return None

    @staticmethod
    def _first_provenance(results):
        for result in results:
            if result.provenance is not None:
                return result.provenance
        return None

    # Calculation delegates through these methods rather than importing the
    # helpers directly at each call site.
    @staticmethod
    def _find_input_observation(*args, **kwargs):
        return _find_input_observation(*args, **kwargs)

    @staticmethod
    def _find_output_constraint(*args, **kwargs):
        return _find_output_constraint(*args, **kwargs)

    @staticmethod
    def _find_segment_revenue_constraint(*args, **kwargs):
        return _find_segment_revenue_constraint(*args, **kwargs)

    @staticmethod
    def _is_generic_unit_observation(*args, **kwargs):
        return _is_generic_unit_observation(*args, **kwargs)

    @staticmethod
    def _selected_periods_compatible(*args, **kwargs):
        return _selected_periods_compatible(*args, **kwargs)

    @staticmethod
    def _generic_growth_fallback(*args, **kwargs):
        return _generic_growth_fallback(*args, **kwargs)

    @staticmethod
    def _record_input_forecasts(
        forecasts: dict[tuple[str, int], OperatingDriverForecast],
        segment: OperatingSegment,
        definition: OperatingDriverDefinition,
        inputs: tuple[_SelectedObservation, ...],
        year: int,
        warnings: list[str],
    ) -> None:
        record_input_forecasts(
            None,
            forecasts,
            segment,
            definition,
            inputs,
            year,
            warnings,
        )

    @staticmethod
    def _record_driver_forecast(
        forecasts: dict[tuple[str, int], OperatingDriverForecast],
        forecast: OperatingDriverForecast,
        warnings: list[str],
    ) -> None:
        record_driver_forecast(forecasts, forecast, warnings)


def _first_observation_provenance(observations):
    for observation in observations:
        if observation.provenance is not None:
            return observation.provenance
    return None


def _first_provenance(results):
    for result in results:
        if result.provenance is not None:
            return result.provenance
    return None


__all__ = [
    "ARCHETYPE_FORMULAS",
    "ArchetypeFormulaRegistry",
    "CompanyOperatingForecast",
    "FORMULA_REGISTRY",
    "OperatingForecastService",
    "SegmentRevenueForecast",
    "normalize_company_historical_revenue",
]


def _has_economics_inputs(observations, plan, overrides) -> bool:
    """Keep the revenue-only model dump unchanged unless economics are requested."""

    economics_metrics = {
        "gross_margin",
        "gross_margin_percent",
        "gross_margin_percentage",
        "gross_margin_rate",
        "gross_margin_pct",
        "gross_profit_margin",
        "gross_profit",
        "gross_profit_amount",
        "gross_income",
        "gross_income_amount",
        "cost_of_revenue",
        "cost_of_sales",
        "cost_of_goods_sold",
        "cost_of_goods",
        "cogs",
        "r_and_d",
        "research_and_development",
        "research_and_development_expense",
        "sg_and_a",
        "selling_general_and_administrative",
        "selling_general_and_administrative_expense",
        "other_operating_items",
        "other_operating_item",
        "other_operating_income",
        "other_operating_expense",
        "recurring_other_operating_items",
        "ebit",
        "operating_income",
        "tax",
        "tax_rate",
        "effective_tax_rate",
        "tax_rate_percentage",
        "forward_tax_rate",
        "tax_rate_guidance",
        "nopat",
        "pretax_income",
        "pretax",
        "pre_tax_income",
        "income_tax_expense",
        "tax_expense",
    }
    if any(
        str(getattr(item, "driver_id", "")).strip().casefold().replace("-", "_").replace(" ", "_")
        in economics_metrics
        for item in observations
    ):
        return True
    records = []
    if plan is not None:
        records.extend(getattr(plan, "decisions", ()) or ())
        records.extend(getattr(plan, "overrides", ()) or ())
        if isinstance(plan, Mapping):
            records.extend(plan.get("decisions", ()) or ())
            records.extend(plan.get("overrides", ()) or ())
    if isinstance(overrides, Mapping):
        records.extend(overrides.values())
    elif hasattr(overrides, "metric") and hasattr(overrides, "strategy"):
        records.append(overrides)
    else:
        try:
            records.extend(overrides or ())
        except TypeError:
            records.append(overrides)
    return any(
        str(
            getattr(
                getattr(item, "metric", item.get("metric", "") if isinstance(item, Mapping) else ""),
                "value",
                getattr(item, "metric", item.get("metric", "") if isinstance(item, Mapping) else ""),
            )
        )
        .strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
        in economics_metrics
        for item in records
    )


def _has_explicit_economics_target(plan, overrides) -> bool:
    records = []
    if plan is not None:
        if isinstance(plan, Mapping):
            records.extend(plan.get("decisions", ()) or ())
            records.extend(plan.get("overrides", ()) or ())
        else:
            records.extend(getattr(plan, "decisions", ()) or ())
            records.extend(getattr(plan, "overrides", ()) or ())
    if isinstance(overrides, Mapping):
        records.extend(overrides.values())
    elif hasattr(overrides, "metric") and hasattr(overrides, "strategy"):
        records.append(overrides)
    else:
        try:
            records.extend(overrides or ())
        except TypeError:
            records.append(overrides)
    for item in records:
        metric = (
            item.get("metric", "") if isinstance(item, Mapping) else getattr(item, "metric", "")
        )
        strategy = (
            item.get("strategy", "")
            if isinstance(item, Mapping)
            else getattr(item, "strategy", "")
        )
        metric = str(getattr(metric, "value", metric)).strip().casefold()
        strategy = str(getattr(strategy, "value", strategy)).strip().casefold()
        scope = (
            item.get("scope", "")
            if isinstance(item, Mapping)
            else getattr(getattr(item, "scope", ""), "value", getattr(item, "scope", ""))
        )
        scope = str(scope).strip().casefold()
        if metric in {
            "gross_margin",
            "gross_profit",
            "r_and_d",
            "sg_and_a",
            "other_operating_items",
            "ebit",
            "tax",
            "tax_rate",
            "nopat",
        } and strategy in {"explicit", "ratio", "residual"} and scope == "segment":
            return True
    return False
