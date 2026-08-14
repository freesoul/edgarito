"""Consolidate non-overlapping operating scopes and revenue coverage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal

from edgarito.schemas.operating import OperatingSegment, SegmentRevenueForecast
from edgarito.services.operating._forecast.contracts import (
    _CONFIDENCE_RANK,
    _CONSOLIDATION_SOURCE,
    _HISTORICAL_SOURCE,
    _INDEPENDENT_SOURCE,
    _UNAVAILABLE_SOURCE,
    _ConsolidationSelection,
    _ReconstructionAudit,
    _SelectedObservation,
)
from edgarito.services.operating._forecast.selection import (
    _reported_revenue_observation_map,
)


def _select_consolidation_segments(
    segments: Sequence[OperatingSegment],
) -> _ConsolidationSelection:
    """Choose a deterministic, non-overlapping company consolidation set."""

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


def _combine_formula_sources(results) -> str:
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


def _company_unit(segments: Sequence[OperatingSegment]) -> str:
    currencies = {segment.currency for segment in segments if segment.currency}
    return next(iter(currencies)) if len(currencies) == 1 else "currency"


def _modeled_revenue_share(
    consolidation_segments: Sequence[OperatingSegment],
    segment_forecasts: Sequence[SegmentRevenueForecast],
    historical,
    historical_selected_records: Mapping[tuple[str, str, int], _SelectedObservation],
    company_audit: _ReconstructionAudit,
) -> Decimal | None:
    """Return revenue-weighted coverage for the consolidated segment scope."""

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
    """Return revenue-weighted modeled coverage for one segment."""

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
    if modeled_revenue_share < Decimal("0.50"):
        return "low"
    if modeled_revenue_share < Decimal("0.80"):
        return _worst_confidence((reconstruction_confidence, "medium"))
    return reconstruction_confidence


__all__ = [name for name in globals() if name.startswith("_")]
