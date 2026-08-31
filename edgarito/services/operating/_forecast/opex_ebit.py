"""Focused deterministic operating-expense and EBIT stage.

The gross economics service owns gross-margin selection.  This module consumes
that result and adds the company OPEX layer without ever turning a company
expense into a segment allocation.  It intentionally has no provider or LLM
dependencies; normalized financial facts enter as ordinary observations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from edgarito.schemas.forecasting import (
    ForecastDecision,
    ForecastOverride,
    ForecastPlan,
    ForecastStrategy,
    ForecastValueBasis,
)
from edgarito.schemas.operating import (
    CompanyOperatingEconomicsForecast,
    EvidenceReference,
    OperatingDriverObservation,
    OperatingEconomicsForecastConfig,
    OperatingEconomicsMetricDiagnostics,
    OperatingSegment,
    canonical_operating_segment_id,
    operating_periods_compatible,
)
from edgarito.services.operating._forecast.consolidation import (
    _select_consolidation_segments,
    _worst_confidence,
)

_RD = "r_and_d"
_SGA = "sg_and_a"
_OTHER = "other_operating_items"
_EBIT = "ebit"
_GP = "gross_profit"
_GM = "gross_margin"
_REVENUE = "revenue"
_OPERATING_INCOME = "operating_income"
_DIRECT_ORIGINS = frozenset({
    "reported",
    "first_party_observation",
    "extracted_evidence",
    "reasoned_assumption",
    "model_assumption",
})
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
_EVENT_TERMS = frozenset({"restructuring", "impairment", "write_down", "writeoff", "reorganization"})
_CURRENCY_TERMS = ("usd", "eur", "gbp", "jpy", "cny", "cad", "aud", "chf", "currency", "dollar", "$", "€", "£")
_EXPENSE_METRICS = frozenset({_RD, _SGA})
_SUPPORTED = frozenset({_RD, _SGA, _OTHER, _EBIT})


@dataclass(frozen=True)
class _Candidate:
    metric: str
    value: Decimal
    source: str
    confidence: str
    method: str
    provenance: Any = None
    references: tuple[EvidenceReference, ...] = ()
    observations: tuple[OperatingDriverObservation, ...] = ()
    ratio: Decimal | None = None
    historical_years: tuple[int, ...] = ()
    residual_magnitude: Decimal | None = None


@dataclass(frozen=True)
class _Path:
    values: tuple[Decimal, ...]
    strategy: ForecastStrategy
    basis: ForecastValueBasis
    provenance: Any = None
    references: tuple[EvidenceReference, ...] = ()


@dataclass(frozen=True)
class _MetricResult:
    values: tuple[Decimal | None, ...]
    candidates: tuple[_Candidate | None, ...]
    warnings: tuple[str, ...]
    attempted: bool


class OperatingOpexEbitEngine:
    """Apply company OPEX, signed other items, and the EBIT identity."""

    def apply(
        self,
        base: CompanyOperatingEconomicsForecast,
        observations: Iterable[OperatingDriverObservation] = (),
        *,
        segments: Sequence[OperatingSegment] = (),
        plan: ForecastPlan | Mapping[str, Any] | None = None,
        overrides: Iterable[ForecastOverride] | Mapping[Any, Any] = (),
        config: OperatingEconomicsForecastConfig | None = None,
        fiscal_period: str = "FY",
        period_key: str | None = None,
        ambiguous_segment_ids: Iterable[str] = (),
    ) -> CompanyOperatingEconomicsForecast:
        policy = config or OperatingEconomicsForecastConfig()
        self._expected_unit = base.unit
        if self._expected_unit == "currency":
            segment_currencies = {
                item.segment.currency
                for item in base.segment_economics
                if item.segment.currency
            }
            if len(segment_currencies) == 1:
                self._expected_unit = next(iter(segment_currencies))
        years = base.fiscal_years
        records = tuple(
            item
            if isinstance(item, OperatingDriverObservation)
            else OperatingDriverObservation.model_validate(item)
            for item in observations
        )
        paths = self._paths(plan, overrides, years, segments, ambiguous_segment_ids)
        company_records = tuple(item for item in records if self._is_company(item))
        company_revenue = base.consolidated_revenue
        segment_revenues = {
            item.segment.segment_id: (item.revenue, item.unit)
            for item in base.segment_economics
        }

        # A directly reported company gross-profit fact is a company truth and
        # must win over a segment sum. This is still a gross-selection seam;
        # the OPEX stage only repairs the base result when company facts exist.
        gross_profit, gross_margin, gross_updates, gross_candidates = self._company_gross(
            base,
            company_records,
            company_revenue,
            policy,
            fiscal_period=fiscal_period,
            period_key=period_key,
        )
        opex_requested = bool(
            any(self._metric(item.driver_id) in _SUPPORTED | {_OPERATING_INCOME} for item in records)
            or any(metric in _SUPPORTED for _scope, metric in paths)
        )
        if not opex_requested:
            return base.model_copy(update=gross_updates) if gross_updates else base

        rd = self._select_expense(
            _RD, years, company_revenue, company_records, records, segments,
            segment_revenues,
            paths, policy, fiscal_period, period_key,
        )
        sga = self._select_expense(
            _SGA, years, company_revenue, company_records, records, segments,
            segment_revenues,
            paths, policy, fiscal_period, period_key,
        )
        other = self._select_other(
            years, company_revenue, gross_profit, rd, sga, company_records,
            records, paths, policy, fiscal_period, period_key,
        )
        ebit_values: list[Decimal | None] = []
        ebit_candidates: list[_Candidate | None] = []
        ebit_warnings: list[str] = []
        reported: dict[int, Decimal] = {}
        explicit_targets: dict[int, Decimal] = {}
        ebit_errors: dict[int, Decimal] = {}
        explicit_ebit_errors: dict[int, Decimal] = {}
        reported_ebit_errors: dict[int, Decimal] = {}
        explicit_ebit_path = paths.get(("company", _EBIT))
        for index, year in enumerate(years):
            target = self._reported_candidate(
                _OPERATING_INCOME, year, company_records, policy,
                fiscal_period=fiscal_period, period_key=period_key,
            )
            if target is not None:
                reported[year] = target.value
            explicit_target = (
                self._path_candidate(
                    _EBIT,
                    explicit_ebit_path,
                    index,
                    None,
                    year,
                )
                if explicit_ebit_path is not None
                else None
            )
            if explicit_target is not None:
                explicit_targets[year] = explicit_target.value
                ebit_warnings.append(
                    f"FY{year}: explicit EBIT is retained as a reconciliation target; calculated EBIT remains the component identity"
                )
            # Explicit EBIT is the selected forecast target. Reported EBIT is
            # retained independently as a reconciliation target and is never
            # silently replaced in the audit trail.
            reconciliation_target = explicit_target or target
            if (
                gross_profit[index] is None
                or rd.values[index] is None
                or sga.values[index] is None
                or other.values[index] is None
            ):
                ebit_values.append(None)
                ebit_candidates.append(None)
                ebit_warnings.append(
                    f"FY{year}: EBIT unavailable because GP, R&D, SG&A, and signed other operating items are all required"
                )
                continue
            value = (
                gross_profit[index]
                - rd.values[index]
                - sga.values[index]
                + other.values[index]
            )
            component_candidates = tuple(
                item
                for item in (
                    gross_candidates[index],
                    rd.candidates[index],
                    sga.candidates[index],
                    other.candidates[index],
                )
                if item is not None
            )
            candidate = _Candidate(
                _EBIT,
                value,
                self._combine_sources((base.source_by_year.get(year, "unavailable"), *[item.source for item in component_candidates])),
                _worst_confidence((base.confidence_by_year.get(year, "low"), *[item.confidence for item in component_candidates])),
                "gross_profit_minus_r_and_d_minus_sg_and_a_plus_other_operating_items",
                provenance=next((item.provenance for item in component_candidates if item.provenance is not None), None),
                references=self._references(component_candidates),
            )
            ebit_values.append(value)
            ebit_candidates.append(candidate)
            if reconciliation_target is not None:
                ebit_errors[year] = abs(value - reconciliation_target.value)
                if explicit_target is not None:
                    explicit_ebit_errors[year] = abs(value - explicit_target.value)
                if target is not None:
                    reported_ebit_errors[year] = abs(value - target.value)
                if ebit_errors[year] != 0:
                    ebit_warnings.append(
                        f"FY{year}: reconstructed EBIT differs from selected EBIT target by {ebit_errors[year]}"
                    )
                if target is not None and reported_ebit_errors[year] != 0:
                    ebit_warnings.append(
                        f"FY{year}: reconstructed EBIT differs from reported operating income by {reported_ebit_errors[year]}"
                    )

        # Segment OPEX is deliberately a separate, optional view. It is never
        # used to fill a company result when a company candidate exists.
        segment_updates = self._segment_updates(
            base.segment_economics,
            segments,
            years,
            company_revenue,
            records,
            paths,
            fiscal_period,
            period_key,
        )

        gross_diagnostics = gross_updates.get("diagnostics", base.diagnostics)
        diagnostics = gross_diagnostics.model_copy(
            update={
                _RD: self._diagnostics(_RD, years, rd, policy),
                _SGA: self._diagnostics(_SGA, years, sga, policy),
                _OTHER: self._diagnostics(_OTHER, years, other, policy),
                _EBIT: self._diagnostics(_EBIT, years, _MetricResult(tuple(ebit_values), tuple(ebit_candidates), tuple(ebit_warnings), True), policy, errors=ebit_errors),
            }
        )
        all_warnings = tuple(
            dict.fromkeys(
                (
                    *gross_updates.get("warnings", base.warnings),
                    *rd.warnings,
                    *sga.warnings,
                    *other.warnings,
                    *ebit_warnings,
                )
            )
        )
        updates: dict[str, Any] = {
            **gross_updates,
            "consolidated_r_and_d": rd.values,
            "consolidated_sg_and_a": sga.values,
            "consolidated_other_operating_items": other.values,
            "consolidated_ebit": tuple(ebit_values),
            "reported_ebit_by_year": reported,
            "explicit_ebit_target_by_year": explicit_targets,
            "ebit_reconstruction_error_by_year": ebit_errors,
            "explicit_ebit_reconstruction_error_by_year": explicit_ebit_errors,
            "reported_ebit_reconstruction_error_by_year": reported_ebit_errors,
            "r_and_d_source_by_year": self._source_map(rd, years),
            "r_and_d_method_by_year": self._method_map(rd, years),
            "r_and_d_confidence_by_year": self._confidence_map(rd, years),
            "r_and_d_provenance_by_year": self._provenance_map(rd, years),
            "r_and_d_audit_by_year": self._audit_map(rd, years),
            "sg_and_a_source_by_year": self._source_map(sga, years),
            "sg_and_a_method_by_year": self._method_map(sga, years),
            "sg_and_a_confidence_by_year": self._confidence_map(sga, years),
            "sg_and_a_provenance_by_year": self._provenance_map(sga, years),
            "sg_and_a_audit_by_year": self._audit_map(sga, years),
            "other_operating_items_source_by_year": self._source_map(other, years),
            "other_operating_items_method_by_year": self._method_map(other, years),
            "other_operating_items_confidence_by_year": self._confidence_map(other, years),
            "other_operating_items_provenance_by_year": self._provenance_map(other, years),
            "other_operating_items_audit_by_year": self._audit_map(other, years),
            "ebit_source_by_year": self._source_map(_MetricResult(tuple(ebit_values), tuple(ebit_candidates), tuple(ebit_warnings), True), years),
            "ebit_method_by_year": self._method_map(_MetricResult(tuple(ebit_values), tuple(ebit_candidates), tuple(ebit_warnings), True), years),
            "ebit_confidence_by_year": self._confidence_map(_MetricResult(tuple(ebit_values), tuple(ebit_candidates), tuple(ebit_warnings), True), years),
            "ebit_provenance_by_year": self._provenance_map(_MetricResult(tuple(ebit_values), tuple(ebit_candidates), tuple(ebit_warnings), True), years),
            "ebit_audit_by_year": self._ebit_audit_map(
                years,
                tuple(ebit_candidates),
                explicit_targets,
                reported,
                ebit_errors,
                explicit_ebit_errors,
                reported_ebit_errors,
            ),
            "diagnostics": diagnostics,
            "warnings": all_warnings,
            "segment_economics": segment_updates,
        }
        updates["years"] = self._company_years(
            base,
            gross_profit,
            gross_margin,
            rd,
            sga,
            other,
            tuple(ebit_values),
            tuple(ebit_candidates),
            reported,
            ebit_errors,
            explicit_targets,
            explicit_ebit_errors,
            reported_ebit_errors,
            gross_years=gross_updates.get("years"),
        )
        return base.model_copy(update=updates)

    @staticmethod
    def _metric(value: Any) -> str:
        normalized = str(getattr(value, "value", value)).strip().casefold().replace("-", "_").replace(" ", "_")
        aliases = {
            "gross_income": _GP,
            "gross_profit_amount": _GP,
            "gross_profit_margin": _GM,
            "gross_margin_percent": _GM,
            "gross_margin_percentage": _GM,
            "gross_margin_rate": _GM,
            "gross_margin_pct": _GM,
            "research_and_development": _RD,
            "research_and_development_expense": _RD,
            "research_development": _RD,
            "selling_general_and_administrative": _SGA,
            "selling_general_and_administrative_expense": _SGA,
            "other_operating_item": _OTHER,
            "other_operating_income_expense": _OTHER,
            "other_operating_income": _OTHER,
            "other_operating_expense": _OTHER,
            "recurring_other_operating_items": _OTHER,
            "recurring_other_operating_income": _OTHER,
            "operating_income_loss": _OPERATING_INCOME,
            "ebit": _OPERATING_INCOME,
        }
        return aliases.get(normalized, normalized)

    @classmethod
    def _path_metric(cls, value: Any) -> str:
        normalized = str(getattr(value, "value", value)).strip().casefold().replace("-", "_").replace(" ", "_")
        # EBIT is an output/target path, while an observed `ebit` is the same
        # reported operating-income reconciliation fact.
        return (
            _EBIT
            if normalized in {_EBIT, _OPERATING_INCOME, "operating_income_loss"}
            else cls._metric(value)
        )

    @staticmethod
    def _coerce_plan(value):
        if value is None:
            return None
        return value if isinstance(value, ForecastPlan) else ForecastPlan.model_validate(value)

    def _paths(self, plan, overrides, years, segments, ambiguous_segment_ids=()) -> dict[tuple[str, str], _Path]:
        records: list[ForecastDecision | ForecastOverride] = []
        normalized_plan = self._coerce_plan(plan)
        if normalized_plan is not None:
            records.extend(normalized_plan.decisions)
            records.extend(normalized_plan.overrides)
        records.extend(self._coerce_overrides(overrides))
        supplied = {item.segment_id for item in segments}
        ambiguous = {
            canonical_operating_segment_id(item) or str(item)
            for item in ambiguous_segment_ids
        }
        result: dict[tuple[str, str], _Path] = {}
        for record in records:
            metric = self._path_metric(record.metric)
            if metric not in _SUPPORTED:
                continue
            if record.scope.value == "segment" and metric in {_OTHER, _EBIT}:
                raise ValueError(
                    f"Segment {metric} decisions/overrides are unsupported; only segment R&D and SG&A are supported in this phase"
                )
            path = record.explicit_path
            if record.strategy in {ForecastStrategy.EXPLICIT, ForecastStrategy.RATIO} and path is None:
                raise ValueError(f"{record.strategy.value} {metric} decision requires explicit_path")
            if path is None:
                continue
            values = tuple(Decimal(item) for item in path)
            if len(values) not in {1, len(years)}:
                raise ValueError(f"Explicit {metric} path must contain one value or exactly the fiscal horizon")
            if any(not item.is_finite() for item in values):
                raise ValueError(f"Explicit {metric} path must contain finite values")
            basis = record.basis
            if metric in _EXPENSE_METRICS:
                expected = ForecastValueBasis.PERCENT_OF_REVENUE if record.strategy == ForecastStrategy.RATIO else ForecastValueBasis.ABSOLUTE
                if basis != expected:
                    raise ValueError(f"{record.strategy.value} {metric} paths require basis={expected.value}")
                if any(item < 0 for item in values):
                    raise ValueError(f"{metric} paths cannot be negative")
            elif metric == _GM and record.strategy == ForecastStrategy.EXPLICIT and basis not in {ForecastValueBasis.PERCENT_OF_REVENUE, ForecastValueBasis.PERCENTAGE_POINTS}:
                raise ValueError(f"{record.strategy.value} {metric} paths require a percentage basis")
            elif metric != _GM and record.strategy in {ForecastStrategy.EXPLICIT, ForecastStrategy.RESIDUAL} and basis != ForecastValueBasis.ABSOLUTE:
                raise ValueError(f"{record.strategy.value} {metric} paths require basis=absolute")
            elif metric in {_OTHER, _EBIT} and record.strategy == ForecastStrategy.RATIO:
                raise ValueError(f"Ratio {metric} paths are not supported")
            if record.scope.value == "segment":
                target = canonical_operating_segment_id(record.scope_id) or record.scope_id
                if target in ambiguous:
                    raise ValueError(
                        f"Explicit {metric} target '{record.scope_id}' is ambiguous among supplied canonical segments"
                    )
                if target not in supplied:
                    raise ValueError(f"Explicit {metric} target '{record.scope_id}' does not match a supplied canonical segment")
                key = (target, metric)
            else:
                key = ("company", metric)
            expanded = values * len(years) if len(values) == 1 else values
            candidate = _Path(expanded, record.strategy, basis or ForecastValueBasis.ABSOLUTE, record.provenance, self._references_for_record(record))
            previous = result.get(key)
            if previous is not None and previous != candidate:
                raise ValueError(f"ambiguous overlapping {metric} forecast paths for {key[0]}")
            result[key] = candidate
        return result

    @staticmethod
    def _coerce_overrides(value) -> tuple[ForecastOverride, ...]:
        if value is None:
            return ()
        if isinstance(value, ForecastOverride):
            return (value,)
        if isinstance(value, Mapping):
            if {"scope", "metric", "strategy"}.issubset(value):
                return (ForecastOverride.model_validate(value),)
            records = []
            for key, item in value.items():
                if isinstance(item, ForecastOverride):
                    records.append(item)
                    continue
                payload = dict(item) if isinstance(item, Mapping) else {}
                if isinstance(key, tuple):
                    if len(key) == 2:
                        payload.setdefault("scope", key[0])
                        payload.setdefault("metric", key[1])
                    elif len(key) == 3:
                        payload.setdefault("scope", key[0])
                        payload.setdefault("scope_id", key[1])
                        payload.setdefault("metric", key[2])
                elif isinstance(key, str) and ":" in key:
                    parts = key.split(":")
                    if len(parts) == 2:
                        payload.setdefault("scope", parts[0])
                        payload.setdefault("metric", parts[1])
                    elif len(parts) == 3:
                        payload.setdefault("scope", parts[0])
                        payload.setdefault("scope_id", parts[1])
                        payload.setdefault("metric", parts[2])
                records.append(payload)
            value = records
        if isinstance(value, (str, bytes, Mapping)):
            return ()
        try:
            items = tuple(value)
        except TypeError:
            items = (value,)
        return tuple(item if isinstance(item, ForecastOverride) else ForecastOverride.model_validate(item) for item in items)

    @staticmethod
    def _references_for_record(record) -> tuple[EvidenceReference, ...]:
        return (record.provenance,) if isinstance(record.provenance, EvidenceReference) else ()

    def _select_expense(
        self,
        metric,
        years,
        revenue,
        company_records,
        records,
        segments,
        segment_revenues,
        paths,
        policy,
        fiscal_period,
        period_key,
    ) -> _MetricResult:
        segment_paths = {segment_id: path for (segment_id, name), path in paths.items() if name == metric and segment_id != "company"}
        company_path = paths.get(("company", metric))
        warnings: list[str] = []
        candidates: list[_Candidate | None] = []
        values: list[Decimal | None] = []
        selected_segments = _select_consolidation_segments(tuple(segments)).segments
        for index, year in enumerate(years):
            company_candidate = None
            if company_path is not None:
                company_candidate = self._path_candidate(
                    metric, company_path, index, revenue[index], year
                )
                if company_candidate is None:
                    warnings.append(
                        f"FY{year}: explicit company {metric} path has no valid revenue denominator"
                    )
            if company_path is None and company_candidate is None:
                company_candidate = self._observation_candidate(metric, year, company_records, policy, fiscal_period=fiscal_period, period_key=period_key, management=True)
            if company_path is None and company_candidate is None:
                company_candidate = self._observation_candidate(metric, year, company_records, policy, fiscal_period=fiscal_period, period_key=period_key, direct=True)
            if company_path is None and company_candidate is None:
                company_candidate = self._historical_ratio(
                    metric,
                    year,
                    company_records,
                    policy,
                    fiscal_period=fiscal_period,
                    period_key=period_key,
                )
            if (
                company_candidate is not None
                and company_candidate.ratio is not None
                and company_candidate.source == "normalized_historical"
            ):
                if revenue[index] is not None:
                    company_candidate = _Candidate(
                        metric,
                        revenue[index] * company_candidate.ratio / Decimal(100),
                        company_candidate.source,
                        company_candidate.confidence,
                        company_candidate.method,
                        company_candidate.provenance,
                        company_candidate.references,
                        company_candidate.observations,
                        company_candidate.ratio,
                        company_candidate.historical_years,
                        company_candidate.residual_magnitude,
                    )
                else:
                    company_candidate = None

            segment_candidates = []
            for segment in selected_segments:
                path = segment_paths.get(segment.segment_id)
                denominator = self._segment_revenue(
                    segment,
                    segment_revenues,
                    records,
                    index,
                    year,
                    policy,
                    fiscal_period,
                    period_key,
                )
                candidate = (
                    self._path_candidate(metric, path, index, denominator, year)
                    if path is not None
                    else None
                )
                if path is None and candidate is None:
                    candidate = self._observation_candidate(metric, year, tuple(item for item in records if item.segment_id == segment.segment_id), policy, fiscal_period=fiscal_period, period_key=period_key, direct=True, segment=segment)
                if candidate is not None:
                    segment_candidates.append(candidate)
            if (
                company_path is not None
                and company_path.strategy == ForecastStrategy.RESIDUAL
                and not segment_candidates
            ):
                company_candidate = None
                warnings.append(
                    f"FY{year}: {metric} company residual ignored because no segment sum is available"
                )
            if (
                company_path is not None
                and company_path.strategy == ForecastStrategy.RESIDUAL
                and segment_candidates
            ):
                if len(segment_candidates) != len(selected_segments):
                    company_candidate = None
                    warnings.append(
                        f"FY{year}: {metric} unavailable because company residual requires an exhaustive segment sum"
                    )
                else:
                    segment_total = sum(
                        item.value for item in segment_candidates
                    )
                    company_candidate = _Candidate(
                        metric,
                        segment_total + company_candidate.value
                        if company_candidate is not None
                        else segment_total,
                        "mixed",
                        _worst_confidence(
                            tuple(item.confidence for item in segment_candidates)
                            + ((company_candidate.confidence,) if company_candidate else ())
                        ),
                        "sum_exhaustive_non_overlapping_segment_opex_plus_company_residual",
                        provenance=(
                            company_candidate.provenance
                            if company_candidate is not None
                            else segment_candidates[0].provenance
                        ),
                        references=self._references(
                            (*segment_candidates, company_candidate)
                            if company_candidate is not None
                            else segment_candidates
                        ),
                    )
            elif company_candidate is not None and segment_candidates:
                warnings.append(f"FY{year}: company {metric} evidence overlaps segment evidence; company scope selected and segment values were not added")
            if company_candidate is None and segment_candidates:
                if len(segment_candidates) != len(selected_segments):
                    warnings.append(f"FY{year}: {metric} unavailable because segment evidence is not exhaustive")
                else:
                    company_candidate = _Candidate(metric, sum(item.value for item in segment_candidates), "mixed" if len({item.source for item in segment_candidates}) > 1 else segment_candidates[0].source, _worst_confidence(tuple(item.confidence for item in segment_candidates)), "sum_exhaustive_non_overlapping_segment_opex", provenance=segment_candidates[0].provenance, references=tuple(ref for item in segment_candidates for ref in item.references))
            elif company_candidate is None and segment_paths:
                warnings.append(
                    f"FY{year}: {metric} unavailable because segment ratio paths lack exhaustive valid denominators"
                )
            if company_candidate is None and any(
                self._metric(item.driver_id) == metric
                and self._is_company(item)
                and item.fiscal_year == year
                for item in records
            ):
                warnings.append(
                    f"FY{year}: {metric} evidence unavailable due to incompatible period, unit, currency, or value"
                )
            if company_candidate is not None and company_candidate.value < 0:
                raise ValueError(f"{metric} cannot be negative")
            candidates.append(company_candidate)
            values.append(company_candidate.value if company_candidate is not None else None)
        return _MetricResult(tuple(values), tuple(candidates), tuple(dict.fromkeys(warnings)), bool(company_path or any(self._metric(item.driver_id) == metric for item in records)))

    def _segment_revenue(
        self,
        segment,
        segment_revenues,
        records,
        index,
        year,
        policy,
        fiscal_period,
        period_key,
    ):
        path_info = segment_revenues.get(segment.segment_id)
        if path_info is not None:
            path, unit = path_info
            if (
                self._currency_compatible(unit, segment.currency)
                and index < len(path)
            ):
                denominator = path[index]
                if denominator is not None and denominator != 0:
                    return denominator
        candidate = self._observation_candidate(
            _REVENUE,
            year,
            tuple(item for item in records if item.segment_id == segment.segment_id),
            policy,
            fiscal_period=fiscal_period,
            period_key=period_key,
            direct=True,
            segment=segment,
        )
        return candidate.value if candidate is not None and candidate.value != 0 else None

    def _select_other(self, years, revenue, gross_profit, rd, sga, company_records, records, paths, policy, fiscal_period, period_key):
        values: list[Decimal | None] = []
        candidates: list[_Candidate | None] = []
        warnings: list[str] = []
        path = paths.get(("company", _OTHER))
        explicit_ebit_path = paths.get(("company", _EBIT))
        for index, year in enumerate(years):
            candidate = self._path_candidate(_OTHER, path, index, revenue, year) if path is not None else None
            if candidate is None and path is None and explicit_ebit_path is not None:
                target = self._path_candidate(
                    _EBIT, explicit_ebit_path, index, None, year
                )
                if (
                    target is not None
                    and gross_profit[index] is not None
                    and rd.values[index] is not None
                    and sga.values[index] is not None
                ):
                    candidate = _Candidate(
                        _OTHER,
                        target.value
                        - (gross_profit[index] - rd.values[index] - sga.values[index]),
                        "explicit",
                        "high",
                        "explicit_ebit_target_residual",
                        target.provenance,
                        target.references,
                    )
            if candidate is None:
                candidate = self._observation_candidate(_OTHER, year, company_records, policy, fiscal_period=fiscal_period, period_key=period_key, management=True)
            if candidate is None:
                candidate = self._observation_candidate(_OTHER, year, company_records, policy, fiscal_period=fiscal_period, period_key=period_key, direct=True, exclude_events=True)
            if candidate is None:
                candidate = self._reported_residual(year, revenue[index], gross_profit[index], rd.values[index], sga.values[index], company_records, policy, fiscal_period, period_key)
            if candidate is None:
                candidate = self._historical_residual(
                    year,
                    revenue[index],
                    company_records,
                    policy,
                    fiscal_period,
                    period_key,
                )
            if candidate is None:
                # No evidence is intentionally different from a supported zero.
                warnings.append(f"FY{year}: other operating items unavailable; no qualifying evidence supports zero")
            candidates.append(candidate)
            values.append(candidate.value if candidate is not None else None)
        return _MetricResult(tuple(values), tuple(candidates), tuple(dict.fromkeys(warnings)), bool(path or any(self._metric(item.driver_id) == _OTHER or self._metric(item.driver_id) == _OPERATING_INCOME for item in company_records)))

    def _reported_residual(self, year, revenue, gross_profit, rd, sga, records, policy, fiscal_period, period_key):
        reported = self._reported_candidate(_OPERATING_INCOME, year, records, policy, fiscal_period=fiscal_period, period_key=period_key)
        if reported is None or gross_profit is None or rd is None or sga is None:
            return None
        value = reported.value - (gross_profit - rd - sga)
        return _Candidate(
            _OTHER,
            value,
            "reported",
            reported.confidence,
            "reported_operating_income_residual",
            reported.provenance,
            reported.references,
            reported.observations,
            ratio=value / revenue * Decimal(100) if revenue not in (None, Decimal(0)) else None,
            historical_years=(year,),
            residual_magnitude=abs(value),
        )

    def _historical_residual(
        self,
        before_year,
        current_revenue,
        records,
        policy,
        fiscal_period,
        period_key,
    ):
        residuals: list[_Candidate] = []
        by_year: dict[int, dict[str, _Candidate]] = {}
        for item in records:
            if not self._is_company(item) or item.fiscal_year >= before_year or item.fiscal_period != fiscal_period:
                continue
            metric = self._metric(item.driver_id)
            if metric not in {_GP, _RD, _SGA, _OPERATING_INCOME}:
                continue
            candidate = self._observation_candidate(metric, item.fiscal_year, records, policy, fiscal_period=fiscal_period, period_key=period_key, direct=True)
            if candidate is not None:
                by_year.setdefault(item.fiscal_year, {})[metric] = candidate
        for year, facts in sorted(by_year.items()):
            revenue_candidate = self._observation_candidate(
                _REVENUE,
                year,
                records,
                policy,
                fiscal_period=fiscal_period,
                period_key=period_key,
                direct=True,
            )
            rev = revenue_candidate.value if revenue_candidate is not None else None
            if rev in (None, Decimal(0)) or not {_GP, _RD, _SGA, _OPERATING_INCOME}.issubset(facts):
                continue
            value = facts[_OPERATING_INCOME].value - (facts[_GP].value - facts[_RD].value - facts[_SGA].value)
            residuals.append(
                _Candidate(
                    _OTHER,
                    value,
                    "normalized_historical",
                    _worst_confidence(
                        tuple(item.confidence for item in facts.values())
                        + (revenue_candidate.confidence,)
                    ),
                    "historical_reported_ebit_residual",
                    facts[_OPERATING_INCOME].provenance,
                    tuple(
                        ref
                        for item in (*facts.values(), revenue_candidate)
                        for ref in item.references
                    ),
                    historical_years=(year,),
                    ratio=value / rev * Decimal(100),
                )
            )
        if not residuals:
            return None
        residuals = residuals[-policy.historical_window :]
        ratios = [item.ratio for item in residuals if item.ratio is not None]
        normalized = self._aggregate(ratios, policy)
        years = tuple(item.historical_years[0] for item in residuals)
        magnitude = max((abs(item) for item in ratios), default=Decimal(0))
        if magnitude <= policy.other_operating_items_materiality_threshold:
            return _Candidate(_OTHER, Decimal(0), "normalized_historical", _worst_confidence(tuple(item.confidence for item in residuals)), "immaterial_historical_other_operating_items_zero", residuals[-1].provenance, tuple(ref for item in residuals for ref in item.references), ratio=Decimal(0), historical_years=years, residual_magnitude=max((abs(item.value) for item in residuals), default=Decimal(0)))
        if not self._stable(ratios, policy.other_operating_items_stability_threshold):
            return None
        value = (current_revenue or Decimal(0)) * normalized / Decimal(100)
        return _Candidate(_OTHER, value, "normalized_historical", _worst_confidence(tuple(item.confidence for item in residuals)), f"{policy.normalization_method}_stable_historical_other_operating_items_residual", residuals[-1].provenance, tuple(ref for item in residuals for ref in item.references), ratio=normalized, historical_years=years, residual_magnitude=max((abs(item.value) for item in residuals), default=Decimal(0)))

    def _historical_ratio(
        self,
        metric,
        before_year,
        records,
        policy,
        *,
        fiscal_period,
        period_key,
    ):
        ratios: list[tuple[int, Decimal, _Candidate]] = []
        historical_years = sorted(
            {
                item.fiscal_year
                for item in records
                if self._is_company(item)
                and item.fiscal_period == fiscal_period
                and self._metric(item.driver_id) == _REVENUE
            }
        )
        for year in historical_years:
            revenue_candidate = self._observation_candidate(
                _REVENUE,
                year,
                records,
                policy,
                fiscal_period=fiscal_period,
                period_key=period_key,
                direct=True,
            )
            if revenue_candidate is None or year >= before_year or revenue_candidate.value == 0:
                continue
            candidate = self._observation_candidate(metric, year, records, policy, fiscal_period=fiscal_period, period_key=period_key, direct=True)
            if candidate is not None:
                candidate = _Candidate(
                    candidate.metric,
                    candidate.value,
                    candidate.source,
                    candidate.confidence,
                    candidate.method,
                    candidate.provenance,
                    tuple(
                        dict.fromkeys(
                            (*candidate.references, *revenue_candidate.references)
                        )
                    ),
                    candidate.observations,
                    candidate.ratio,
                    candidate.historical_years,
                    candidate.residual_magnitude,
                )
                ratios.append(
                    (year, candidate.value / revenue_candidate.value * Decimal(100), candidate)
                )
        if not ratios:
            return None
        ratios = ratios[-policy.historical_window :]
        normalized = self._aggregate([item[1] for item in ratios], policy)
        return _Candidate(metric, Decimal(0), "normalized_historical", _worst_confidence(tuple(item[2].confidence for item in ratios)), f"{policy.normalization_method}_recent_historical_{metric}_ratio", ratios[-1][2].provenance, tuple(ref for item in ratios for ref in item[2].references), ratio=normalized, historical_years=tuple(item[0] for item in ratios))

    @staticmethod
    def _aggregate(values, policy):
        values = tuple(values)[-policy.historical_window:]
        if policy.normalization_method == "weighted_recent":
            denominator = Decimal(sum(range(1, len(values) + 1)))
            return sum((value * Decimal(position) for position, value in enumerate(values, 1)), Decimal(0)) / denominator
        ordered = sorted(values)
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / Decimal(2)

    @staticmethod
    def _stable(values, threshold):
        if len(values) < 2:
            return True
        scale = max(max(abs(value) for value in values), Decimal(1))
        return (max(values) - min(values)) / scale <= threshold

    def _path_candidate(self, metric, path, index, revenue, year):
        if path is None:
            return None
        value = path.values[index]
        if path.strategy == ForecastStrategy.RATIO:
            if revenue is None or revenue == 0:
                return None
            value = revenue * value / Decimal(100)
            ratio = path.values[index]
        else:
            ratio = None
        return _Candidate(metric, value, "explicit", "high", f"forecast_plan_{path.strategy.value}_{metric}", path.provenance, path.references, ratio=ratio)

    def _observation_candidate(self, metric, year, records, policy, *, fiscal_period, period_key, management=False, direct=False, segment=None, exclude_events=False):
        choices: list[tuple[int, OperatingDriverObservation, Decimal]] = []
        for position, observation in enumerate(records):
            if self._metric(observation.driver_id) != metric or observation.fiscal_year != year:
                continue
            if management and observation.origin != "management_guidance":
                continue
            if direct and observation.origin not in _DIRECT_ORIGINS:
                continue
            if exclude_events and _EVENT_TERMS.intersection(
                {
                    *self._metric(observation.driver_id).split("_"),
                    *(observation.scope_evidence or "").casefold().replace("-", "_").split("_"),
                    *(observation.basis or "").casefold().replace("-", "_").split("_"),
                    *(observation.method or "").casefold().replace("-", "_").split("_"),
                }
            ):
                continue
            if segment is not None and not self._segment_scope(observation, segment):
                continue
            expected_unit = segment.currency if segment is not None else getattr(self, "_expected_unit", None)
            if not self._currency_compatible(
                observation.unit, expected_unit, observation.currency
            ):
                continue
            value = self._monetary_value(observation, policy)
            if value is None:
                continue
            if metric in _EXPENSE_METRICS and value < 0:
                raise ValueError(f"{metric} cannot be negative")
            if not operating_periods_compatible(observation.fiscal_period, fiscal_period, observation.period_key, period_key):
                continue
            choices.append((position, observation, value))
        if not choices:
            return None
        _, observation, value = max(choices, key=lambda item: (_ORIGIN_RANK.get(item[1].origin, 0), item[1].is_total, item[1].confidence == "high", item[1].evidence is not None, -item[0]))
        return _Candidate(metric, value, "management_guidance" if observation.origin == "management_guidance" else observation.origin, observation.confidence, "management_guidance_observation" if management else "reported_operating_economics", observation.provenance or observation.evidence, self._references((observation,)), (observation,))

    def _reported_candidate(self, metric, year, records, policy, *, fiscal_period, period_key):
        return self._observation_candidate(metric, year, records, policy, fiscal_period=fiscal_period, period_key=period_key, direct=True)

    @staticmethod
    def _monetary_value(observation, policy):
        if OperatingOpexEbitEngine._metric(observation.driver_id) == _GM:
            unit = observation.unit.casefold().replace(" ", "_")
            try:
                value = observation.normalized_value
            except ValueError:
                return None
            if "bp" in unit or "basis_point" in unit:
                value /= Decimal(100)
            elif unit in {"ratio", "rate", "decimal", "fraction"}:
                value *= Decimal(100)
            return value if Decimal("-100") <= value <= Decimal("100") else None
        if "/" in observation.unit and not any(
            scale in observation.unit.casefold()
            for scale in ("thousand", "million", "billion")
        ):
            return None
        if not any(term in observation.unit.casefold() for term in _CURRENCY_TERMS):
            return None
        try:
            return observation.normalized_value
        except ValueError:
            return None

    @staticmethod
    def _currency_compatible(
        actual: str,
        expected: str | None,
        observation_currency: str | None = None,
    ) -> bool:
        if expected is None:
            return True
        def code(value: str) -> str | None:
            folded = value.casefold()
            return next((item for item in ("usd", "eur", "gbp", "jpy", "cny", "cad", "aud", "chf") if item in folded), None)
        unit_code = code(actual)
        declared_code = code(observation_currency or "")
        expected_code = code(expected)
        if unit_code and declared_code and unit_code != declared_code:
            return False
        actual_codes = {item for item in (unit_code, declared_code) if item}
        return not expected_code or not actual_codes or actual_codes == {expected_code}

    @staticmethod
    def _is_company(observation):
        return observation.segment_id == "company" or (observation.scope or "").casefold() in {"company", "consolidated", "total"}

    @staticmethod
    def _segment_scope(observation, segment):
        if OperatingOpexEbitEngine._is_company(observation) or observation.is_component:
            return False
        return not observation.scope or observation.scope.casefold() == segment.scope.casefold() or (observation.scope.casefold() == "segment" and segment.scope == "segment")

    @staticmethod
    def _references(items):
        return tuple(dict.fromkeys(ref for item in items for ref in (*getattr(item, "references", ()), getattr(item, "evidence", None), getattr(item, "provenance", None) if isinstance(getattr(item, "provenance", None), EvidenceReference) else None) if ref is not None))

    @staticmethod
    def _combine_sources(sources):
        values = {source for source in sources if source and source != "unavailable"}
        return next(iter(values)) if len(values) == 1 else ("mixed" if values else "unavailable")

    def _company_gross(self, base, records, revenue, policy, *, fiscal_period, period_key):
        profits = list(base.consolidated_gross_profit)
        margins = list(base.consolidated_gross_margin)
        updates: dict[str, Any] = {}
        candidates: list[_Candidate | None] = []
        repaired: list[bool] = []
        for index, year in enumerate(base.fiscal_years):
            if profits[index] is not None or margins[index] is not None:
                candidates.append(
                    _Candidate(
                        _GP if profits[index] is not None else _GM,
                        profits[index] if profits[index] is not None else margins[index],
                        base.source_by_year.get(year, "unavailable"),
                        base.confidence_by_year.get(year, "low"),
                        base.method_by_year.get(year, "unavailable"),
                        base.provenance_by_year.get(year),
                        base.source_provenance_by_year.get(year, ()),
                    )
                )
                repaired.append(False)
                continue
            gp = self._observation_candidate(_GP, year, records, policy, fiscal_period=fiscal_period, period_key=period_key, direct=True)
            gm = self._observation_candidate(_GM, year, records, policy, fiscal_period=fiscal_period, period_key=period_key, direct=True)
            if gp is not None:
                profits[index] = gp.value
                candidates.append(gp)
                repaired.append(True)
                if revenue[index] not in (None, Decimal(0)):
                    margins[index] = gp.value / revenue[index] * Decimal(100)
            elif gm is not None and revenue[index] is not None:
                margins[index] = gm.value
                profits[index] = revenue[index] * gm.value / Decimal(100)
                candidates.append(gm)
                repaired.append(True)
            else:
                historical_margins: list[Decimal] = []
                for historical_year in sorted(
                    {
                        item.fiscal_year
                        for item in records
                        if item.fiscal_period == fiscal_period
                        and item.fiscal_year < year
                    }
                ):
                    historical_gp = self._observation_candidate(
                        _GP,
                        historical_year,
                        records,
                        policy,
                        fiscal_period=fiscal_period,
                        period_key=period_key,
                        direct=True,
                    )
                    historical_revenue = self._observation_candidate(
                        _REVENUE,
                        historical_year,
                        records,
                        policy,
                        fiscal_period=fiscal_period,
                        period_key=period_key,
                        direct=True,
                    )
                    if (
                        historical_gp is not None
                        and historical_revenue is not None
                        and historical_revenue.value
                    ):
                        historical_margins.append(
                            historical_gp.value
                            / historical_revenue.value
                            * Decimal(100)
                        )
                if not historical_margins or revenue[index] is None:
                    candidates.append(None)
                    repaired.append(False)
                    continue
                normalized = self._aggregate(historical_margins, policy)
                if policy.gross_margin_min <= normalized <= policy.gross_margin_max:
                    margins[index] = normalized
                    profits[index] = revenue[index] * normalized / Decimal(100)
                    candidates.append(
                        _Candidate(
                            _GM,
                            normalized,
                            "normalized_historical",
                            "medium",
                            f"{policy.normalization_method}_historical_company_gross_margin",
                            ratio=normalized,
                        )
                    )
                    repaired.append(True)
                else:
                    candidates.append(None)
                    repaired.append(False)
        selected_years = tuple(
            index
            for index, is_repaired in enumerate(repaired)
            if is_repaired
        )
        if (
            tuple(profits) != base.consolidated_gross_profit
            or tuple(margins) != base.consolidated_gross_margin
            or selected_years
        ):
            source_by_year = dict(base.source_by_year)
            confidence_by_year = dict(base.confidence_by_year)
            provenance_by_year = dict(base.provenance_by_year)
            provenance_chain_by_year = dict(base.provenance_chain_by_year)
            source_provenance_by_year = dict(base.source_provenance_by_year)
            margin_provenance = dict(base.gross_margin_provenance_by_year)
            margin_chain = dict(base.gross_margin_provenance_chain_by_year)
            margin_references = dict(base.gross_margin_source_provenance_by_year)
            profit_provenance = dict(base.gross_profit_provenance_by_year)
            profit_chain = dict(base.gross_profit_provenance_chain_by_year)
            profit_references = dict(base.gross_profit_source_provenance_by_year)
            methods = dict(base.method_by_year)
            audits = dict(base.audit_by_year)
            for index in selected_years:
                year = base.fiscal_years[index]
                candidate = candidates[index]
                assert candidate is not None
                source_by_year[year] = candidate.source
                confidence_by_year[year] = candidate.confidence
                methods[year] = candidate.method
                audits[year] = (
                    f"company_gross_repair_source={candidate.source}",
                    f"company_gross_repair_method={candidate.method}",
                )
                for mapping in (
                    provenance_by_year,
                    provenance_chain_by_year,
                    source_provenance_by_year,
                    margin_provenance,
                    margin_chain,
                    margin_references,
                    profit_provenance,
                    profit_chain,
                    profit_references,
                ):
                    mapping.pop(year, None)
                if candidate.provenance is not None:
                    provenance_by_year[year] = candidate.provenance
                    provenance_chain_by_year[year] = (candidate.provenance,)
                    margin_provenance[year] = candidate.provenance
                    margin_chain[year] = (candidate.provenance,)
                    profit_provenance[year] = candidate.provenance
                    profit_chain[year] = (candidate.provenance,)
                source_provenance_by_year[year] = candidate.references
                margin_references[year] = candidate.references
                profit_references[year] = candidate.references
            updates.update(
                {
                    "consolidated_gross_profit": tuple(profits),
                    "consolidated_gross_margin": tuple(margins),
                    "source_by_year": source_by_year,
                    "confidence_by_year": confidence_by_year,
                    "provenance_by_year": provenance_by_year,
                    "provenance_chain_by_year": provenance_chain_by_year,
                    "source_provenance_by_year": source_provenance_by_year,
                    "gross_margin_provenance_by_year": margin_provenance,
                    "gross_margin_provenance_chain_by_year": margin_chain,
                    "gross_margin_source_provenance_by_year": margin_references,
                    "gross_profit_provenance_by_year": profit_provenance,
                    "gross_profit_provenance_chain_by_year": profit_chain,
                    "gross_profit_source_provenance_by_year": profit_references,
                    "method_by_year": methods,
                    "audit_by_year": audits,
                }
            )
            stale = tuple(
                warning
                for warning in base.warnings
                if not any(
                    f"FY{base.fiscal_years[index]}: consolidated gross" in warning
                    for index in selected_years
                )
            )
            updates["warnings"] = stale
            gross_margin_supported = tuple(
                year
                for year, value in zip(base.fiscal_years, margins, strict=True)
                if value is not None
            )
            gross_profit_supported = tuple(
                year
                for year, value in zip(base.fiscal_years, profits, strict=True)
                if value is not None
            )
            confidence = _worst_confidence(
                tuple(
                    candidate.confidence
                    for candidate in candidates
                    if candidate is not None
                )
            )
            gross_warnings = tuple(
                warning
                for warning in base.diagnostics.gross_profit.warnings
                if not any(
                    f"FY{base.fiscal_years[index]}: consolidated gross" in warning
                    for index in selected_years
                )
            )
            margin_warnings = tuple(
                warning
                for warning in base.diagnostics.gross_margin.warnings
                if not any(
                    f"FY{base.fiscal_years[index]}: consolidated gross" in warning
                    for index in selected_years
                )
            )
            gross_identity_warnings = tuple(
                warning
                for warning in base.diagnostics.gross_profit.identity_warnings
                if not any(
                    f"FY{base.fiscal_years[index]}: consolidated gross" in warning
                    for index in selected_years
                )
            )
            margin_identity_warnings = tuple(
                warning
                for warning in base.diagnostics.gross_margin.identity_warnings
                if not any(
                    f"FY{base.fiscal_years[index]}: consolidated gross" in warning
                    for index in selected_years
                )
            )
            gross_diagnostics = base.diagnostics.model_copy(
                update={
                    "completeness": (
                        Decimal(
                            len(
                                set(gross_margin_supported)
                                & set(gross_profit_supported)
                            )
                        )
                        / Decimal(len(base.fiscal_years))
                        if base.fiscal_years
                        else None
                    ),
                    "warnings": stale,
                    "identity_warnings": tuple(
                        warning
                        for warning in base.diagnostics.identity_warnings
                        if not any(
                            f"FY{base.fiscal_years[index]}: consolidated gross"
                            in warning
                            for index in selected_years
                        )
                    ),
                    "gross_margin": base.diagnostics.gross_margin.model_copy(
                        update={
                            "coverage": (
                                Decimal(len(gross_margin_supported))
                                / Decimal(len(base.fiscal_years))
                                if base.fiscal_years
                                else None
                            ),
                            "supported_years": gross_margin_supported,
                            "confidence": confidence,
                            "reconstruction_error": Decimal(0),
                            "completeness": (
                                Decimal(len(gross_margin_supported))
                                / Decimal(len(base.fiscal_years))
                                if base.fiscal_years
                                else None
                            ),
                            "warnings": margin_warnings,
                            "identity_warnings": margin_identity_warnings,
                        }
                    ),
                    "gross_profit": base.diagnostics.gross_profit.model_copy(
                        update={
                            "coverage": (
                                Decimal(len(gross_profit_supported))
                                / Decimal(len(base.fiscal_years))
                                if base.fiscal_years
                                else None
                            ),
                            "supported_years": gross_profit_supported,
                            "confidence": confidence,
                            "reconstruction_error": Decimal(0),
                            "completeness": (
                                Decimal(len(gross_profit_supported))
                                / Decimal(len(base.fiscal_years))
                                if base.fiscal_years
                                else None
                            ),
                            "warnings": gross_warnings,
                            "identity_warnings": gross_identity_warnings,
                        }
                    ),
                }
            )
            updates["diagnostics"] = gross_diagnostics
            updates["years"] = tuple(
                year.model_copy(
                    update={
                        "gross_profit": profits[index],
                        "gross_margin": margins[index],
                        "source": source_by_year.get(year.fiscal_year, "unavailable"),
                        "confidence": confidence_by_year.get(year.fiscal_year, "low"),
                        "provenance": provenance_by_year.get(year.fiscal_year),
                        "provenance_chain": provenance_chain_by_year.get(year.fiscal_year, ()),
                        "source_provenance": source_provenance_by_year.get(year.fiscal_year, ()),
                        "gross_margin_provenance": margin_provenance.get(year.fiscal_year),
                        "gross_margin_provenance_chain": margin_chain.get(year.fiscal_year, ()),
                        "gross_margin_source_provenance": margin_references.get(year.fiscal_year, ()),
                        "gross_profit_provenance": profit_provenance.get(year.fiscal_year),
                        "gross_profit_provenance_chain": profit_chain.get(year.fiscal_year, ()),
                        "gross_profit_source_provenance": profit_references.get(year.fiscal_year, ()),
                        "method": methods.get(year.fiscal_year, "unavailable"),
                        "audit": audits.get(year.fiscal_year, ()),
                    }
                )
                for index, year in enumerate(base.years)
            )
        return tuple(profits), tuple(margins), updates, tuple(candidates)

    def _segment_updates(self, economics, segments, years, company_revenue, records, paths, fiscal_period, period_key):
        by_id = {item.segment.segment_id: item for item in economics}
        result = []
        for segment in segments:
            base = by_id.get(segment.segment_id)
            if base is None:
                continue
            rd = self._segment_metric(_RD, segment, base.revenue, base.unit, years, records, paths, fiscal_period, period_key)
            sga = self._segment_metric(_SGA, segment, base.revenue, base.unit, years, records, paths, fiscal_period, period_key)
            rd_sources = self._source_map(rd, years)
            rd_methods = self._method_map(rd, years)
            rd_confidences = self._confidence_map(rd, years)
            rd_provenance = self._provenance_map(rd, years)
            sga_sources = self._source_map(sga, years)
            sga_methods = self._method_map(sga, years)
            sga_confidences = self._confidence_map(sga, years)
            sga_provenance = self._provenance_map(sga, years)
            result.append(base.model_copy(update={
                "r_and_d": rd.values,
                "sg_and_a": sga.values,
                "r_and_d_source_by_year": rd_sources,
                "r_and_d_method_by_year": rd_methods,
                "r_and_d_confidence_by_year": rd_confidences,
                "r_and_d_provenance_by_year": rd_provenance,
                "r_and_d_audit_by_year": self._audit_map(rd, years),
                "sg_and_a_source_by_year": sga_sources,
                "sg_and_a_method_by_year": sga_methods,
                "sg_and_a_confidence_by_year": sga_confidences,
                "sg_and_a_provenance_by_year": sga_provenance,
                "sg_and_a_audit_by_year": self._audit_map(sga, years),
                "diagnostics": base.diagnostics.model_copy(update={
                    _RD: self._diagnostics(_RD, years, rd, OperatingEconomicsForecastConfig()),
                    _SGA: self._diagnostics(_SGA, years, sga, OperatingEconomicsForecastConfig()),
                }),
                "years": tuple(
                    item.model_copy(
                        update={
                            "r_and_d": rd.values[index],
                            "sg_and_a": sga.values[index],
                            "r_and_d_source": rd_sources.get(year, "unavailable"),
                            "r_and_d_method": rd_methods.get(year, "unavailable"),
                            "r_and_d_confidence": rd_confidences.get(year, "low"),
                            "r_and_d_provenance": rd_provenance.get(year),
                            "r_and_d_audit": self._audit_map(rd, years).get(year, ()),
                            "sg_and_a_source": sga_sources.get(year, "unavailable"),
                            "sg_and_a_method": sga_methods.get(year, "unavailable"),
                            "sg_and_a_confidence": sga_confidences.get(year, "low"),
                            "sg_and_a_provenance": sga_provenance.get(year),
                            "sg_and_a_audit": self._audit_map(sga, years).get(year, ()),
                        }
                    )
                    for index, (item, year) in enumerate(
                        zip(base.years, years, strict=True)
                    )
                ),
            }))
        return tuple(result) if result else economics

    def _segment_metric(self, metric, segment, revenue, revenue_unit, years, records, paths, fiscal_period, period_key):
        path = paths.get((segment.segment_id, metric))
        values = []
        candidates = []
        for index, year in enumerate(years):
            denominator = None
            if path is not None and (
                path.strategy != ForecastStrategy.RATIO
                or self._currency_compatible(revenue_unit, segment.currency)
            ):
                denominator = revenue[index]
            candidate = self._path_candidate(metric, path, index, denominator, year) if path else None
            if candidate is None:
                candidate = self._observation_candidate(metric, year, tuple(item for item in records if item.segment_id == segment.segment_id), OperatingEconomicsForecastConfig(), fiscal_period=fiscal_period, period_key=period_key, direct=True, segment=segment)
            candidates.append(candidate)
            values.append(candidate.value if candidate else None)
        return _MetricResult(tuple(values), tuple(candidates), (), bool(path or any(self._metric(item.driver_id) == metric and item.segment_id == segment.segment_id for item in records)))

    @staticmethod
    def _diagnostics(metric, years, result, policy, errors=None):
        supported = tuple(year for year, value in zip(years, result.values, strict=True) if value is not None)
        candidates = tuple(item for item in result.candidates if item is not None)
        all_historical = tuple(
            sorted({year for item in candidates for year in item.historical_years})
        )
        historical = all_historical[-policy.historical_window :]
        recent_set = set(historical)
        historical_candidates = tuple(
            item
            for item in candidates
            if not item.historical_years
            or recent_set.intersection(item.historical_years)
        )
        ratios = tuple(
            item.ratio
            for item in historical_candidates
            if item.ratio is not None
        )
        residual = max(
            (
                item.residual_magnitude
                if item.residual_magnitude is not None
                else abs(item.value)
                for item in historical_candidates
                if item.historical_years
            ),
            default=None,
        )
        return OperatingEconomicsMetricDiagnostics(
            metric=metric,
            coverage=Decimal(len(supported)) / Decimal(len(years)) if result.attempted and years else None,
            supported_years=supported,
            confidence=_worst_confidence(tuple(item.confidence for item in candidates)) if candidates else "low",
            reconstruction_error=(
                sum(errors.values(), Decimal(0)) / Decimal(len(errors))
                if errors
                else Decimal(0)
                if metric == _OTHER and historical
                else None
            ),
            completeness=Decimal(len(supported)) / Decimal(len(years)) if result.attempted and years else None,
            normalized_ratio=ratios[-1] if ratios else None,
            historical_years=historical,
            residual_magnitude=residual,
            provenance=tuple(
                dict.fromkeys(
                    item.provenance
                    for item in historical_candidates
                    if item.provenance is not None
                )
            ),
            warnings=result.warnings,
        )

    @staticmethod
    def _source_map(result, years):
        return {year: item.source for year, item in zip(years, result.candidates, strict=True) if item is not None}

    @staticmethod
    def _method_map(result, years):
        return {year: item.method for year, item in zip(years, result.candidates, strict=True) if item is not None}

    @staticmethod
    def _confidence_map(result, years):
        return {year: item.confidence for year, item in zip(years, result.candidates, strict=True) if item is not None}

    @staticmethod
    def _provenance_map(result, years):
        return {year: item.provenance for year, item in zip(years, result.candidates, strict=True) if item is not None and item.provenance is not None}

    @staticmethod
    def _audit_map(result, years):
        return {
            year: (f"selected_source={item.source}",)
            for year, item in zip(years, result.candidates, strict=True)
            if item is not None
        }

    @staticmethod
    def _ebit_audit_map(
        years,
        candidates,
        explicit_targets,
        reported,
        errors,
        explicit_errors,
        reported_errors,
    ):
        return {
            year: OperatingOpexEbitEngine._ebit_audit(
                year,
                explicit_targets,
                reported,
                errors,
                explicit_errors,
                reported_errors,
                candidate is not None,
            )
            for year, candidate in zip(years, candidates, strict=True)
            if OperatingOpexEbitEngine._ebit_audit(
                year,
                explicit_targets,
                reported,
                errors,
                explicit_errors,
                reported_errors,
                candidate is not None,
            )
        }

    @staticmethod
    def _references(candidates):
        return tuple(
            dict.fromkeys(
                ref
                for item in candidates
                for ref in (
                    *getattr(item, "references", ()),
                    *getattr(item, "source_provenance", ()),
                    getattr(item, "evidence", None),
                    getattr(item, "provenance", None)
                    if isinstance(getattr(item, "provenance", None), EvidenceReference)
                    else None,
                )
                if ref is not None
            )
        )

    @staticmethod
    def _company_years(
        base,
        gp,
        gm,
        rd,
        sga,
        other,
        ebit,
        candidates,
        reported,
        errors,
        explicit_targets,
        explicit_errors,
        reported_errors,
        gross_years=None,
    ):
        rd_sources = base.r_and_d_source_by_year
        sga_sources = base.sg_and_a_source_by_year
        other_sources = base.other_operating_items_source_by_year
        ebit_result = _MetricResult(ebit, candidates, (), True)
        return tuple(
            (gross_years[index] if gross_years is not None else item).model_copy(update={
                "gross_profit": gp[index],
                "gross_margin": gm[index],
                "r_and_d": rd.values[index],
                "sg_and_a": sga.values[index],
                "other_operating_items": other.values[index],
                "ebit": ebit[index],
                "reported_ebit": reported.get(year),
                "explicit_ebit_target": explicit_targets.get(year),
                "ebit_reconstruction_error": errors.get(year),
                "explicit_ebit_reconstruction_error": explicit_errors.get(year),
                "reported_ebit_reconstruction_error": reported_errors.get(year),
                "r_and_d_source": rd_sources.get(year, "unavailable") if not rd.candidates[index] else rd.candidates[index].source,
                "r_and_d_method": rd.candidates[index].method if rd.candidates[index] else "unavailable",
                "r_and_d_confidence": rd.candidates[index].confidence if rd.candidates[index] else "low",
                "r_and_d_provenance": rd.candidates[index].provenance if rd.candidates[index] else None,
                "r_and_d_audit": (f"selected_source={rd.candidates[index].source}",) if rd.candidates[index] else (),
                "sg_and_a_source": sga_sources.get(year, "unavailable") if not sga.candidates[index] else sga.candidates[index].source,
                "sg_and_a_method": sga.candidates[index].method if sga.candidates[index] else "unavailable",
                "sg_and_a_confidence": sga.candidates[index].confidence if sga.candidates[index] else "low",
                "sg_and_a_provenance": sga.candidates[index].provenance if sga.candidates[index] else None,
                "sg_and_a_audit": (f"selected_source={sga.candidates[index].source}",) if sga.candidates[index] else (),
                "other_operating_items_source": other_sources.get(year, "unavailable") if not other.candidates[index] else other.candidates[index].source,
                "other_operating_items_method": other.candidates[index].method if other.candidates[index] else "unavailable",
                "other_operating_items_confidence": other.candidates[index].confidence if other.candidates[index] else "low",
                "other_operating_items_provenance": other.candidates[index].provenance if other.candidates[index] else None,
                "other_operating_items_audit": (f"selected_source={other.candidates[index].source}",) if other.candidates[index] else (),
                "ebit_source": ebit_result.candidates[index].source if ebit_result.candidates[index] else "unavailable",
                "ebit_method": ebit_result.candidates[index].method if ebit_result.candidates[index] else "unavailable",
                "ebit_confidence": ebit_result.candidates[index].confidence if ebit_result.candidates[index] else "low",
                "ebit_provenance": ebit_result.candidates[index].provenance if ebit_result.candidates[index] else None,
                "ebit_audit": OperatingOpexEbitEngine._ebit_audit(
                    year,
                    explicit_targets,
                    reported,
                    errors,
                    explicit_errors,
                    reported_errors,
                    ebit_result.candidates[index] is not None,
                ),
            })
            for index, (item, year) in enumerate(zip(base.years, base.fiscal_years, strict=True))
        )

    @staticmethod
    def _ebit_audit(
        year,
        explicit_targets,
        reported,
        errors,
        explicit_errors,
        reported_errors,
        calculated,
    ):
        audit: list[str] = []
        if year in explicit_targets:
            audit.extend(
                (
                    f"explicit_reconciliation_target={explicit_targets[year]}",
                    "calculated_identity_not_overridden",
                )
            )
        if calculated and year in errors:
            if year in explicit_targets:
                audit.append(f"explicit_reconstruction_error={explicit_errors[year]}")
            if year in reported:
                audit.append(f"reported_reconstruction_error={reported_errors[year]}")
            audit.append(f"selected_reconstruction_error={errors[year]}")
        elif not calculated and year in reported:
            audit.append(f"reported_ebit_target={reported[year]}")
        return tuple(audit)


OperatingOpexForecastService = OperatingOpexEbitEngine
OperatingOpexEbitForecastService = OperatingOpexEbitEngine

__all__ = [
    "OperatingOpexEbitEngine",
    "OperatingOpexForecastService",
    "OperatingOpexEbitForecastService",
]
