"""Selection and historical normalization for operating reinvestment inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from edgarito.schemas.forecasting import ForecastStrategy
from edgarito.schemas.guidance.management import MonetaryForecastConstraint
from edgarito.schemas.operating import (
    CompanyOperatingEconomicsForecast,
    OperatingDriverObservation,
    OperatingInvestmentProgram,
    SegmentOperatingEconomicsForecast,
    operating_periods_compatible,
)

from .contracts import (
    _CAPEX,
    _CONFIDENCE_RANK,
    _DA,
    _DIRECT_ORIGINS,
    _HUNDRED,
    _OWC,
    _Candidate,
    _Path,
)


class _ReinvestmentSelectionMixin:
    """Metric precedence, direct evidence, and historical ratio selection."""

    @classmethod
    def _resolve_metric(
        cls,
        metric: str,
        base: CompanyOperatingEconomicsForecast,
        records: Sequence[OperatingDriverObservation],
        paths: Mapping[tuple[str, str], _Path],
        history: Mapping[str, tuple[Decimal | None, _Candidate | None]],
        *,
        fiscal_period: str,
        period_key: str | None,
        constraints: Mapping[int, MonetaryForecastConstraint] | None = None,
        programs: Sequence[OperatingInvestmentProgram] = (),
        program_range_policy: str = "none",
        segment_records: Sequence[OperatingDriverObservation] = (),
        warnings: list[str],
    ) -> tuple[list[Decimal | None], list[_Candidate | None]]:
        values: list[Decimal | None] = []
        candidates: list[_Candidate | None] = []
        for index, year in enumerate(base.fiscal_years):
            revenue = base.consolidated_revenue[index]
            path = paths.get(("company", metric))
            candidate = cls._path_candidate(metric, path, index, revenue)
            direct = cls._direct_candidate(
                metric, year, records, base.unit, fiscal_period, period_key
            )
            if direct is not None and direct.ratio is not None and revenue is not None:
                direct = direct.__class__(
                    revenue * direct.ratio / _HUNDRED,
                    direct.source,
                    direct.method,
                    direct.confidence,
                    direct.provenance,
                    direct.references,
                    (*direct.audit, f"ratio_amount={revenue}*{direct.ratio}/100"),
                    direct.ratio,
                    direct.historical_years,
                    direct.provenance_chain,
                    direct.segment_id,
                    direct.is_component,
                    direct.exhaustive,
                    direct.residual,
                )
            segment_candidates = cls._segment_candidates(
                metric,
                year,
                base.segment_economics,
                segment_records,
                paths,
                base.unit,
                fiscal_period,
                period_key,
            )
            company_path = path
            residual_path_candidate = (
                candidate if candidate is not None and candidate.residual else None
            )
            residual_direct_candidate = (
                direct if direct is not None and direct.residual else None
            )
            if not segment_candidates and residual_path_candidate is not None:
                candidate = None
            if not segment_candidates and residual_direct_candidate is not None:
                direct = None
            if residual_direct_candidate is not None and segment_candidates:
                direct = None
            if residual_path_candidate is not None and segment_candidates:
                candidate = None
            if (candidate is not None or direct is not None) and segment_candidates:
                if not (company_path is not None and company_path.residual):
                    raise ValueError(
                        f"Company and segment {metric} evidence overlap in FY{year}; "
                        "do not double count company amounts"
                    )
            if candidate is None and direct is None and segment_candidates:
                if not cls._non_overlapping_segments(
                    segment_candidates, base.segment_economics
                ):
                    raise ValueError(f"Overlapping segment {metric} evidence in FY{year}")
                residual = residual_path_candidate or (
                    cls._path_candidate(metric, company_path, index, revenue)
                    if company_path and company_path.residual
                    else None
                )
                residual = residual or residual_direct_candidate
                if residual is None:
                    component_values = segment_candidates
                    if (
                        len(component_values) == len(base.segment_economics)
                        and all(
                            item.exhaustive and item.is_component
                            for item in component_values
                        )
                    ):
                        candidate = _Candidate(
                            sum((item.value for item in component_values), Decimal(0)),
                            "derived",
                            f"exhaustive_non_overlapping_segment_{metric}_sum",
                            cls._worst_confidence(
                                tuple(item.confidence for item in component_values)
                            ),
                            next(
                                (
                                    item.provenance
                                    for item in component_values
                                    if item.provenance is not None
                                ),
                                None,
                            ),
                            tuple(
                                dict.fromkeys(
                                    ref
                                    for item in component_values
                                    for ref in item.references
                                )
                            ),
                            (
                                "segment_sum_is_exhaustive=true",
                                "segment_sum_non_overlapping=true",
                            ),
                            provenance_chain=cls._provenance_items(
                                *(item.provenance for item in component_values)
                            ),
                        )
                    else:
                        warnings.append(
                            f"FY{year}: company {metric} unavailable; segment evidence "
                            "is not explicitly exhaustive and non-overlapping"
                        )
                else:
                    candidate = _Candidate(
                        residual.value
                        + sum((item.value for item in segment_candidates), Decimal(0)),
                        "derived",
                        f"segment_{metric}_sum_plus_explicit_company_residual",
                        cls._worst_confidence(
                            (residual.confidence, *(item.confidence for item in segment_candidates))
                        ),
                        residual.provenance,
                        tuple(
                            dict.fromkeys(
                                (
                                    *residual.references,
                                    *(
                                        ref
                                        for item in segment_candidates
                                        for ref in item.references
                                    ),
                                )
                            )
                        ),
                        (
                            "segment_sum_non_overlapping=true",
                            "explicit_company_residual=true",
                        ),
                        provenance_chain=cls._provenance_items(
                            residual.provenance,
                            *(item.provenance for item in segment_candidates),
                        ),
                    )
            history_value, history_candidate = history.get(metric, (None, None))
            if candidate is None and direct is not None:
                candidate = direct

            provisional = candidate
            constraint = (constraints or {}).get(year) if metric == _CAPEX else None
            if (
                metric == _CAPEX
                and constraint is not None
                and candidate is not None
                and path is None
            ):
                constrained = candidate.value
                if constraint.point is not None:
                    constrained = constraint.point
                else:
                    if constraint.minimum is not None:
                        constrained = max(constrained, constraint.minimum)
                    if constraint.maximum is not None:
                        constrained = min(constrained, constraint.maximum)
                candidate = _Candidate(
                    constrained,
                    constraint.source,
                    f"{constraint.methodology}_capex_constraint",
                    candidate.confidence,
                    candidate.provenance,
                    candidate.references,
                    (
                        *candidate.audit,
                        f"capex_constraint_method={constraint.methodology}",
                        f"capex_constraint_point={constraint.point}",
                        f"capex_constraint_minimum={constraint.minimum}",
                        f"capex_constraint_maximum={constraint.maximum}",
                    ),
                    provenance_chain=candidate.provenance_chain,
                )
            if metric == _CAPEX and candidate is None:
                program = cls._program_candidate(
                    programs,
                    year,
                    base.unit,
                    fiscal_period,
                    period_key,
                    warnings,
                    range_policy=program_range_policy,
                )
                if constraint is not None:
                    if program is not None:
                        provisional = program
                    elif (
                        history_candidate is not None
                        and history_value is not None
                        and revenue is not None
                    ):
                        provisional = _Candidate(
                            revenue * history_candidate.ratio / _HUNDRED,
                            history_candidate.source,
                            history_candidate.method,
                            history_candidate.confidence,
                            history_candidate.provenance,
                            history_candidate.references,
                            history_candidate.audit,
                            history_candidate.ratio,
                            history_candidate.historical_years,
                            history_candidate.provenance_chain,
                        )
                    if constraint.point is not None:
                        constrained = constraint.point
                    elif provisional is not None:
                        constrained = provisional.value
                        if constraint.minimum is not None:
                            constrained = max(constrained, constraint.minimum)
                        if constraint.maximum is not None:
                            constrained = min(constrained, constraint.maximum)
                    else:
                        constrained = None
                    if constrained is not None:
                        candidate = _Candidate(
                            constrained,
                            constraint.source,
                            f"{constraint.methodology}_capex_constraint",
                            "high",
                            constraint.source,
                            (),
                            (
                                f"capex_constraint_method={constraint.methodology}",
                                f"capex_constraint_point={constraint.point}",
                                f"capex_constraint_minimum={constraint.minimum}",
                                f"capex_constraint_maximum={constraint.maximum}",
                            ),
                        )
                elif program is not None:
                    candidate = program
                elif (
                    history_candidate is not None
                    and history_value is not None
                    and revenue is not None
                ):
                    candidate = _Candidate(
                        revenue * history_candidate.ratio / _HUNDRED,
                        "normalized_historical",
                        history_candidate.method,
                        history_candidate.confidence,
                        history_candidate.provenance,
                        history_candidate.references,
                        history_candidate.audit,
                        history_candidate.ratio,
                        history_candidate.historical_years,
                        history_candidate.provenance_chain,
                    )
            elif (
                candidate is None
                and history_candidate is not None
                and history_value is not None
                and revenue is not None
            ):
                candidate = _Candidate(
                    revenue * history_candidate.ratio / _HUNDRED,
                    "normalized_historical",
                    history_candidate.method,
                    history_candidate.confidence,
                    history_candidate.provenance,
                    history_candidate.references,
                    history_candidate.audit,
                    history_candidate.ratio,
                    history_candidate.historical_years,
                    history_candidate.provenance_chain,
                )
            values.append(candidate.value if candidate is not None else None)
            candidates.append(candidate)
        return values, candidates

    @classmethod
    def _segment_candidates(
        cls,
        metric: str,
        year: int,
        existing: Sequence[SegmentOperatingEconomicsForecast],
        records: Sequence[OperatingDriverObservation],
        paths: Mapping[tuple[str, str], _Path],
        expected_unit: str,
        fiscal_period: str,
        period_key: str | None,
    ) -> tuple[_Candidate, ...]:
        result = []
        known = {item.segment.segment_id for item in existing}
        for segment_id in sorted(known):
            values = tuple(
                item
                for item in records
                if cls._metric(item.driver_id) == metric
                and item.fiscal_year == year
                and item.segment_id == segment_id
                and cls._is_segment(item)
                and item.origin
                in {
                    "reported",
                    "first_party_observation",
                    "extracted_evidence",
                    "forward_evidence",
                }
            )
            segment = next(
                item for item in existing if item.segment.segment_id == segment_id
            )
            path = paths.get((segment_id, metric))
            if not values and path is None:
                continue
            candidate = cls._direct_candidate(
                metric, year, values, expected_unit, fiscal_period, period_key
            )
            path_candidate = (
                cls._path_candidate(
                    metric,
                    path,
                    segment.fiscal_years.index(year),
                    segment.revenue[segment.fiscal_years.index(year)],
                )
                if path is not None
                else None
            )
            if (
                candidate is not None
                and path_candidate is not None
                and candidate.value != path_candidate.value
            ):
                raise ValueError(
                    f"Overlapping segment {metric} path and evidence for "
                    f"{segment_id} FY{year}"
                )
            if path_candidate is not None:
                candidate = _Candidate(
                    path_candidate.value,
                    path_candidate.source,
                    path_candidate.method,
                    path_candidate.confidence,
                    path_candidate.provenance,
                    path_candidate.references,
                    path_candidate.audit,
                    path_candidate.ratio,
                    path_candidate.historical_years,
                    path_candidate.provenance_chain,
                    segment_id,
                    True,
                    True,
                    False,
                )
            if candidate is not None:
                if candidate.ratio is not None:
                    revenue = segment.revenue[segment.fiscal_years.index(year)]
                    if revenue is not None:
                        candidate = _Candidate(
                            revenue * candidate.ratio / _HUNDRED,
                            candidate.source,
                            candidate.method,
                            candidate.confidence,
                            candidate.provenance,
                            candidate.references,
                            (*candidate.audit, f"ratio_amount={revenue}*{candidate.ratio}/100"),
                            candidate.ratio,
                            candidate.historical_years,
                            candidate.provenance_chain,
                            candidate.segment_id,
                            candidate.is_component,
                            candidate.exhaustive,
                            candidate.residual,
                        )
                result.append(candidate)
        return tuple(result)

    @staticmethod
    def _non_overlapping_segments(candidates, existing) -> bool:
        ids = {item.segment_id for item in candidates}
        by_id = {item.segment.segment_id: item.segment for item in existing}
        for segment_id in ids:
            ancestor = by_id.get(segment_id).parent_id if by_id.get(segment_id) else None
            visited: set[str] = set()
            while ancestor is not None and ancestor not in visited:
                if ancestor in ids:
                    return False
                visited.add(ancestor)
                ancestor = (
                    by_id.get(ancestor).parent_id
                    if by_id.get(ancestor) is not None
                    else None
                )
        return True

    @classmethod
    def _direct_candidate(
        cls,
        metric: str,
        year: int,
        records: Sequence[OperatingDriverObservation],
        expected_unit: str,
        fiscal_period: str,
        period_key: str | None,
    ) -> _Candidate | None:
        choices: list[
            tuple[int, OperatingDriverObservation, Decimal, Decimal | None]
        ] = []
        for position, item in enumerate(records):
            if cls._metric(item.driver_id) != metric or item.fiscal_year != year:
                continue
            if not operating_periods_compatible(
                item.fiscal_period,
                fiscal_period,
                item.period_key,
                period_key,
            ):
                continue
            try:
                value = item.normalized_value
            except ValueError:
                continue
            ratio = cls._ratio_value(item)
            if ratio is None and not cls._compatible_currency(
                expected_unit, item.unit, item.currency
            ):
                continue
            if ratio is not None:
                value = ratio
            if metric in {_DA, _CAPEX} and value < 0:
                continue
            if item.origin not in _DIRECT_ORIGINS and item.origin != "management_guidance":
                continue
            choices.append((position, item, value, ratio))
        if not choices:
            return None
        selected = max(
            choices,
            key=lambda choice: (
                1 if choice[1].is_total else 0,
                1 if choice[1].origin == "management_guidance" else 0,
                _CONFIDENCE_RANK.get(choice[1].confidence, 0),
                -choice[0],
            ),
        )
        _, item, value, ratio = selected
        source = (
            "management_guidance"
            if item.origin == "management_guidance"
            else "first_party_observation"
            if item.origin
            in {"first_party_observation", "extracted_evidence", "forward_evidence"}
            else "reported"
        )
        method = item.method or f"direct_{metric}_observation"
        if ratio is not None:
            method = f"{method}_ratio"
        provenance = item.provenance or item.evidence
        refs = tuple(
            dict.fromkeys(
                (
                    *item.source_provenance,
                    *((item.evidence,) if item.evidence else ()),
                )
            )
        )
        return _Candidate(
            value,
            source,
            method,
            item.confidence,
            provenance,
            refs,
            (f"direct_observation_unit={item.unit}",),
            ratio,
            provenance_chain=tuple(
                value for value in (item.provenance, item.evidence) if value is not None
            ),
            segment_id=item.segment_id,
            is_component=item.is_component,
            exhaustive=item.exhaustive,
            residual=item.is_component and not item.is_total,
        )

    @classmethod
    def _historical(
        cls,
        records: Sequence[OperatingDriverObservation],
        cutoff: int | None,
        fiscal_period: str,
        period_key: str | None,
        *,
        normalization_method: str = "median",
        historical_window: int = 3,
    ) -> dict[str, tuple[Decimal | None, _Candidate | None]]:
        if cutoff is None:
            return {}
        historical_records = tuple(
            item for item in records if item.origin != "management_guidance"
        )
        revenue: dict[int, OperatingDriverObservation] = {}
        for item in historical_records:
            if item.fiscal_year >= cutoff:
                continue
            if not operating_periods_compatible(
                item.fiscal_period,
                fiscal_period,
                item.period_key,
                period_key,
            ):
                continue
            if cls._metric(item.driver_id) == "revenue":
                if item.normalized_value > 0 and cls._compatible_currency(
                    "currency", item.unit, item.currency
                ):
                    revenue.setdefault(item.fiscal_year, item)
        ratios: dict[str, list[tuple[int, Decimal, _Candidate]]] = {
            metric: [] for metric in {_DA, _CAPEX, _OWC}
        }
        for metric in ratios:
            for year in sorted(
                {
                    item.fiscal_year
                    for item in historical_records
                    if item.fiscal_year < cutoff
                }
            ):
                rev = revenue.get(year)
                if rev is None:
                    continue
                direct = cls._direct_candidate(
                    metric,
                    year,
                    historical_records,
                    rev.unit,
                    fiscal_period,
                    period_key,
                )
                if direct is None or (metric == _OWC and direct.ratio is not None):
                    continue
                amount = direct.value
                ratio = amount / rev.normalized_value * _HUNDRED
                ratios[metric].append(
                    (
                        year,
                        ratio,
                        _Candidate(
                            ratio,
                            "normalized_historical",
                            f"median_recent_historical_{metric}_to_revenue",
                            direct.confidence,
                            direct.provenance,
                            direct.references,
                            (
                                f"historical_{metric}_years={year}",
                                f"historical_{metric}_ratio={ratio}",
                            ),
                            ratio,
                            (year,),
                            direct.provenance_chain,
                            historical_amount=amount,
                        ),
                    )
                )
        result: dict[str, tuple[Decimal | None, _Candidate | None]] = {}
        for metric, values in ratios.items():
            if not values:
                result[metric] = (None, None)
                continue
            selected_values = values[-historical_window:]
            ordered = sorted(item[1] for item in selected_values)
            middle = len(ordered) // 2
            if normalization_method == "weighted_recent":
                denominator = Decimal(sum(range(1, len(selected_values) + 1)))
                normalized = sum(
                    (
                        item[1] * Decimal(position)
                        for position, item in enumerate(selected_values, 1)
                    ),
                    Decimal(0),
                ) / denominator
            else:
                normalized = ordered[middle] if len(ordered) % 2 else (
                    ordered[middle - 1] + ordered[middle]
                ) / Decimal(2)
            sources = tuple(item[2] for item in selected_values)
            candidate = _Candidate(
                normalized,
                "normalized_historical",
                f"{normalization_method}_recent_historical_{metric}_to_revenue",
                cls._worst_confidence(tuple(item.confidence for item in sources)),
                next(
                    (item.provenance for item in sources if item.provenance is not None),
                    None,
                ),
                tuple(
                    dict.fromkeys(ref for item in sources for ref in item.references)
                ),
                (
                    f"historical_{metric}_years={','.join(str(item[0]) for item in selected_values)}",
                    f"historical_{metric}_ratios={','.join(str(item[1]) for item in selected_values)}",
                ),
                normalized,
                tuple(item[0] for item in selected_values),
                cls._provenance_items(*(item.provenance for item in sources)),
                historical_amount=sources[-1].historical_amount,
            )
            result[metric] = (normalized, candidate)
        return result

    @classmethod
    def _path_candidate(
        cls,
        metric: str,
        path: _Path | None,
        index: int,
        revenue: Decimal | None,
    ) -> _Candidate | None:
        if path is None:
            return None
        raw = path.values[index]
        value = (
            raw
            if path.strategy in {ForecastStrategy.EXPLICIT, ForecastStrategy.RESIDUAL}
            else revenue * raw / _HUNDRED if revenue is not None else None
        )
        if value is None:
            return None
        return _Candidate(
            value,
            "explicit",
            f"forecast_plan_{path.strategy.value}_{metric}",
            "high",
            path.provenance,
            path.references,
            (f"path_basis={path.basis.value}",),
            raw if path.strategy == ForecastStrategy.RATIO else None,
            provenance_chain=(path.provenance,) if path.provenance is not None else (),
            residual=path.residual,
        )


__all__ = ["_ReinvestmentSelectionMixin"]
