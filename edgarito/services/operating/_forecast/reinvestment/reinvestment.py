"""Deterministic operating reinvestment and FCFF accounting.

This stage is intentionally narrower than the FCFF forecast service.  It only
selects D&A, CAPEX, and operating working capital inputs, derives the working
capital change, and applies the FCFF identity to an existing operating
economics artifact.  It does not retrieve evidence, allocate company values,
or perform discounting.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from typing import Any

from edgarito.schemas.forecasting import (
    ForecastOverride,
    ForecastPlan,
)
from edgarito.schemas.guidance.management import MonetaryForecastConstraint
from edgarito.schemas.operating import (
    CompanyOperatingEconomicsForecast,
    EvidenceReference,
    OperatingDriverObservation,
    OperatingEconomicsForecastConfig,
    OperatingEconomicsMetricDiagnostics,
    OperatingInvestmentProgram,
    OperatingReinvestmentSeed,
    OperatingSegment,
)

from .canonical import DriverBasedCanonicalFcffAdapter
from .contracts import (
    _CAPEX,
    _CONFIDENCE_RANK,
    _DA,
    _DELTA,
    _FCFF,
    _METRICS,
    _OWC,
    _UNAVAILABLE,
    _Candidate,
)
from .programs import _ReinvestmentProgramsMixin
from .selection import _ReinvestmentSelectionMixin


class OperatingReinvestmentEngine(
    _ReinvestmentSelectionMixin,
    _ReinvestmentProgramsMixin,
):
    """Apply deterministic D&A, CAPEX, OWC, delta, and FCFF accounting."""

    def __init__(self, *, asset_life_resolver: Any | None = None) -> None:
        # The resolver is an optional provider-neutral seam.  The stage never
        # imports or instantiates a valuation resolver, which keeps the simple
        # accounting path usable in isolation.
        self.asset_life_resolver = asset_life_resolver

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
        capex_constraints: Mapping[int, MonetaryForecastConstraint] | None = None,
        investment_programs: Iterable[OperatingInvestmentProgram] = (),
        seed: OperatingReinvestmentSeed | Mapping[str, Any] | None = None,
        reinvestment_seed: OperatingReinvestmentSeed | Mapping[str, Any] | None = None,
        evidence: Any | None = None,
        constraints: Mapping[int, MonetaryForecastConstraint] | None = None,
        programs: Iterable[OperatingInvestmentProgram] | None = None,
    ) -> CompanyOperatingEconomicsForecast:
        policy = (
            config
            if isinstance(config, OperatingEconomicsForecastConfig)
            else OperatingEconomicsForecastConfig.model_validate(config or {})
        )
        years = base.fiscal_years
        evidence_observations = self._evidence_value(evidence, "observations")
        records = tuple(
            item
            if isinstance(item, OperatingDriverObservation)
            else OperatingDriverObservation.model_validate(item)
            for item in (*self._items(observations), *self._items(evidence_observations))
        )
        normalized_seed = self._coerce_seed(reinvestment_seed or seed)
        paths = self._paths(plan, overrides, years)
        selected_constraints = dict(policy.capex_constraints)
        selected_constraints.update(capex_constraints or {})
        selected_constraints.update(constraints or {})
        selected_constraints = {
            int(year): item
            if isinstance(item, MonetaryForecastConstraint)
            else MonetaryForecastConstraint.model_validate(item)
            for year, item in selected_constraints.items()
        }
        self._validate_ytd_capex_constraint(
            reinvestment_seed or seed,
            years[0] if years else None,
            selected_constraints,
        )
        programs = self._programs(
            programs
            if programs is not None
            else investment_programs
            if investment_programs
            else self._evidence_value(evidence, "investment_programs")
            or self._evidence_value(evidence, "programs")
        )
        attempted = bool(
            paths
            or any(self._metric(item.driver_id) in _METRICS for item in records)
            or selected_constraints
            or programs
            or normalized_seed
            or policy.depreciable_asset_life_years is not None
            or self.asset_life_resolver is not None
        )
        if not attempted:
            return base

        warnings: list[str] = []
        company_records = tuple(item for item in records if self._is_company(item))
        first_year = years[0] if years else None
        history = self._historical(
            company_records,
            first_year,
            fiscal_period,
            period_key,
            normalization_method=policy.normalization_method,
            historical_window=policy.historical_window,
        )

        # CAPEX is deliberately resolved before the optional life method for
        # D&A.  This ordering is an accounting contract, not an implementation
        # convenience.
        capex, capex_candidates = self._resolve_metric(
            _CAPEX,
            base,
            company_records,
            paths,
            history,
            fiscal_period=fiscal_period,
            period_key=period_key,
            constraints=selected_constraints,
            programs=programs,
            program_range_policy=policy.investment_program_range_policy,
            segment_records=records,
            warnings=warnings,
        )
        depreciation, depreciation_candidates = self._resolve_metric(
            _DA,
            base,
            company_records,
            paths,
            history,
            fiscal_period=fiscal_period,
            period_key=period_key,
            segment_records=records,
            warnings=warnings,
        )
        life = policy.depreciable_asset_life_years or self._injected_asset_life(
            base, records, warnings
        )
        if life is not None:
            depreciation, depreciation_candidates = self._apply_life_method(
                base,
                capex,
                capex_candidates,
                depreciation,
                depreciation_candidates,
                history,
                life,
                warnings,
            )

        working_capital, working_capital_candidates = self._resolve_metric(
            _OWC,
            base,
            company_records,
            paths,
            history,
            fiscal_period=fiscal_period,
            period_key=period_key,
            segment_records=records,
            warnings=warnings,
        )

        segment_economics = self._apply_segments(
            base.segment_economics,
            segments,
            records,
            paths,
            years,
            fiscal_period,
            period_key,
            warnings,
        )

        delta: list[Decimal | None] = []
        delta_candidates: list[_Candidate | None] = []
        fcff: list[Decimal | None] = []
        fcff_candidates: list[_Candidate | None] = []
        seed_selection = self._seed_selection(
            normalized_seed,
            company_records,
            first_year,
            base.unit,
            fiscal_period,
            period_key,
            warnings,
        )
        previous_owc = seed_selection[1]
        selected_seed = seed_selection[0]
        resolved_seed = self._resolved_seed(seed_selection)
        seed_provenance = getattr(selected_seed, "provenance", None) or getattr(
            selected_seed, "evidence", None
        )
        seed_references = (
            tuple(
                dict.fromkeys(
                    (
                        *getattr(selected_seed, "source_provenance", ()),
                        *(
                            (selected_seed.provenance,)
                            if isinstance(
                                getattr(selected_seed, "provenance", None),
                                EvidenceReference,
                            )
                            else ()
                        ),
                        *(
                            (selected_seed.evidence,)
                            if getattr(selected_seed, "evidence", None) is not None
                            else ()
                        ),
                    )
                )
            )
            if selected_seed is not None
            else ()
        )
        seed_confidence = getattr(selected_seed, "confidence", "high")
        for index, year in enumerate(years):
            current = working_capital[index]
            if current is not None and previous_owc is not None:
                change = current - previous_owc
                component = self._component_candidate(
                    _OWC,
                    working_capital_candidates[index],
                )
                candidate = _Candidate(
                    change,
                    "derived",
                    "operating_working_capital_minus_prior_balance",
                    self._worst_confidence(
                        tuple(
                            item
                            for item in (
                                component.confidence if component is not None else None,
                                seed_confidence if previous_owc is not None else None,
                            )
                            if item is not None
                        )
                    ),
                    component.provenance if component is not None and component.provenance is not None else seed_provenance,
                    tuple(dict.fromkeys((*seed_references, *(component.references if component is not None else ())))),
                    audit=(f"delta_owc_identity={current}-{previous_owc}={change}",),
                    provenance_chain=self._provenance_items(seed_provenance, component.provenance if component is not None else None),
                )
                delta.append(change)
                delta_candidates.append(candidate)
            else:
                delta.append(None)
                delta_candidates.append(None)
                if current is None:
                    warnings.append(f"FY{year}: change in operating working capital unavailable because OWC is unavailable")
                elif previous_owc is None:
                    warnings.append(f"FY{year}: change in operating working capital unavailable because no compatible real OWC seed was supplied")

            nopat = base.nopat[index] if index < len(base.nopat) else None
            if (
                nopat is not None
                and depreciation[index] is not None
                and capex[index] is not None
                and delta[index] is not None
            ):
                value = nopat + depreciation[index] - capex[index] - delta[index]
                nopat_candidate = _Candidate(
                    nopat,
                    base.nopat_source_by_year.get(year, "modeled_nopat"),
                    base.nopat_method_by_year.get(year, "operating_nopat_output"),
                    base.nopat_confidence_by_year.get(year, base.confidence_by_year.get(year, "low")),
                    base.nopat_provenance_by_year.get(year),
                    base.source_provenance_by_year.get(year, ()),
                    base.nopat_audit_by_year.get(year, ()),
                    provenance_chain=(base.nopat_provenance_by_year.get(year),)
                    if base.nopat_provenance_by_year.get(year) is not None
                    else (),
                )
                components = tuple(
                    item
                    for item in (
                        nopat_candidate,
                        depreciation_candidates[index],
                        capex_candidates[index],
                        delta_candidates[index],
                    )
                    if item is not None
                )
                provenance = next(
                    (item.provenance for item in components if item.provenance is not None),
                    None,
                )
                refs = tuple(dict.fromkeys(ref for item in components for ref in item.references))
                chain = self._provenance_items(
                    *[item.provenance for item in components],
                    *[item.provenance_chain for item in components],
                )
                candidate = _Candidate(
                    value,
                    "derived",
                    "nopat_plus_da_minus_capex_minus_delta_owc",
                    self._worst_confidence(tuple(item.confidence for item in components)),
                    provenance,
                    refs,
                    audit=(f"fcff_identity={nopat}+{depreciation[index]}-{capex[index]}-{delta[index]}={value}",),
                    provenance_chain=chain,
                )
                fcff.append(value)
                fcff_candidates.append(candidate)
            else:
                fcff.append(None)
                fcff_candidates.append(None)
                if any(value is None for value in (nopat, depreciation[index], capex[index], delta[index])):
                    warnings.append(f"FY{year}: FCFF unavailable because NOPAT, D&A, CAPEX, and ΔOWC are all required")
            if current is not None:
                previous_owc = current

        metric_candidates = {
            _DA: depreciation_candidates,
            _CAPEX: capex_candidates,
            _OWC: working_capital_candidates,
            _DELTA: delta_candidates,
            _FCFF: fcff_candidates,
        }
        metric_values = {
            _DA: depreciation,
            _CAPEX: capex,
            _OWC: working_capital,
            _DELTA: delta,
            _FCFF: fcff,
        }
        updates: dict[str, Any] = {}
        for metric, values in metric_values.items():
            stem = metric
            updates[stem] = tuple(values)
            updates[f"{stem}_source_by_year"] = self._source_map(metric_candidates[metric], years)
            updates[f"{stem}_method_by_year"] = self._method_map(metric_candidates[metric], years)
            updates[f"{stem}_confidence_by_year"] = self._confidence_map(metric_candidates[metric], years)
            updates[f"{stem}_provenance_by_year"] = self._provenance_map(metric_candidates[metric], years)
            updates[f"{stem}_audit_by_year"] = self._audit_map(metric_candidates[metric], years)

        diagnostics = base.diagnostics.model_copy(
            update={
                metric: self._diagnostics(metric, years, metric_values[metric], metric_candidates[metric], history, warnings)
                for metric in metric_values
            }
        )
        complete_counts = sum(
            all(metric_values[metric][index] is not None for metric in metric_values)
            for index in range(len(years))
        )
        diagnostics = diagnostics.model_copy(
            update={
                "completeness": Decimal(complete_counts) / Decimal(len(years))
                if years
                else None,
            }
        )
        updates["diagnostics"] = diagnostics
        updates["warnings"] = tuple(dict.fromkeys((*base.warnings, *warnings)))
        updates["segment_economics"] = segment_economics
        updates["reinvestment_seed"] = resolved_seed
        updates["years"] = tuple(
            item.model_copy(
                update={
                    _DA: depreciation[index],
                    _CAPEX: capex[index],
                    _OWC: working_capital[index],
                    _DELTA: delta[index],
                    _FCFF: fcff[index],
                    f"{_DA}_source": updates[f"{_DA}_source_by_year"].get(year, _UNAVAILABLE),
                    f"{_DA}_method": updates[f"{_DA}_method_by_year"].get(year, _UNAVAILABLE),
                    f"{_DA}_confidence": updates[f"{_DA}_confidence_by_year"].get(year, "low"),
                    f"{_DA}_provenance": updates[f"{_DA}_provenance_by_year"].get(year),
                    f"{_DA}_audit": updates[f"{_DA}_audit_by_year"].get(year, ()),
                    f"{_CAPEX}_source": updates[f"{_CAPEX}_source_by_year"].get(year, _UNAVAILABLE),
                    f"{_CAPEX}_method": updates[f"{_CAPEX}_method_by_year"].get(year, _UNAVAILABLE),
                    f"{_CAPEX}_confidence": updates[f"{_CAPEX}_confidence_by_year"].get(year, "low"),
                    f"{_CAPEX}_provenance": updates[f"{_CAPEX}_provenance_by_year"].get(year),
                    f"{_CAPEX}_audit": updates[f"{_CAPEX}_audit_by_year"].get(year, ()),
                    f"{_OWC}_source": updates[f"{_OWC}_source_by_year"].get(year, _UNAVAILABLE),
                    f"{_OWC}_method": updates[f"{_OWC}_method_by_year"].get(year, _UNAVAILABLE),
                    f"{_OWC}_confidence": updates[f"{_OWC}_confidence_by_year"].get(year, "low"),
                    f"{_OWC}_provenance": updates[f"{_OWC}_provenance_by_year"].get(year),
                    f"{_OWC}_audit": updates[f"{_OWC}_audit_by_year"].get(year, ()),
                    f"{_DELTA}_source": updates[f"{_DELTA}_source_by_year"].get(year, _UNAVAILABLE),
                    f"{_DELTA}_method": updates[f"{_DELTA}_method_by_year"].get(year, _UNAVAILABLE),
                    f"{_DELTA}_confidence": updates[f"{_DELTA}_confidence_by_year"].get(year, "low"),
                    f"{_DELTA}_provenance": updates[f"{_DELTA}_provenance_by_year"].get(year),
                    f"{_DELTA}_audit": updates[f"{_DELTA}_audit_by_year"].get(year, ()),
                    f"{_FCFF}_source": updates[f"{_FCFF}_source_by_year"].get(year, _UNAVAILABLE),
                    f"{_FCFF}_method": updates[f"{_FCFF}_method_by_year"].get(year, _UNAVAILABLE),
                    f"{_FCFF}_confidence": updates[f"{_FCFF}_confidence_by_year"].get(year, "low"),
                    f"{_FCFF}_provenance": updates[f"{_FCFF}_provenance_by_year"].get(year),
                    f"{_FCFF}_audit": updates[f"{_FCFF}_audit_by_year"].get(year, ()),
                }
            )
            for index, (item, year) in enumerate(zip(base.years, years, strict=True))
        )
        return base.model_copy(update=updates)

    forecast = apply
    build = apply

    @staticmethod
    def _evidence_value(value: Any, name: str):
        if value is None:
            return None
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    def _injected_asset_life(self, base, records, warnings: list[str]) -> int | None:
        """Resolve only through an explicitly injected, provider-neutral seam."""

        resolver = self.asset_life_resolver
        if resolver is None:
            return None
        try:
            result = (
                resolver(base, records)
                if callable(resolver)
                else resolver.resolve(base, observations=records)
            )
        except (TypeError, ValueError, AttributeError):
            warnings.append("Injected depreciable-life resolver could not be represented; D&A cohort method was not used")
            return None
        value = getattr(result, "value", result)
        try:
            value = int(value)
        except (TypeError, ValueError):
            warnings.append("Injected depreciable-life resolver returned no usable life; D&A cohort method was not used")
            return None
        if not 2 <= value <= 30:
            warnings.append("Injected depreciable-life resolver returned a life outside the supported range")
            return None
        return value

    @staticmethod
    def _items(value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, (OperatingDriverObservation, Mapping)):
            return (value,)
        if hasattr(value, "records"):
            return tuple(value.records)
        if hasattr(value, "eligible_records"):
            return tuple(value.eligible_records)
        if isinstance(value, (str, bytes)):
            return (value,)
        try:
            return tuple(value)
        except TypeError:
            return (value,)

    @staticmethod
    def _component_candidate(metric: str, candidate: _Candidate | None) -> _Candidate | None:
        return candidate

    @staticmethod
    def _worst_confidence(values: Sequence[str]) -> str:
        if not values:
            return "low"
        return min(values, key=lambda item: _CONFIDENCE_RANK.get(item, 0))

    @staticmethod
    def _provenance_items(*values) -> tuple[Any, ...]:
        result = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, (tuple, list)):
                for item in value:
                    if item is not None and item not in result:
                        result.append(item)
            elif value not in result:
                result.append(value)
        return tuple(result)

    @staticmethod
    def _source_map(candidates, years):
        return {
            year: item.source if item is not None else _UNAVAILABLE
            for year, item in zip(years, candidates, strict=True)
        }

    @staticmethod
    def _method_map(candidates, years):
        return {
            year: item.method if item is not None else _UNAVAILABLE
            for year, item in zip(years, candidates, strict=True)
        }

    @staticmethod
    def _confidence_map(candidates, years):
        return {
            year: item.confidence if item is not None else "low"
            for year, item in zip(years, candidates, strict=True)
        }

    @staticmethod
    def _provenance_map(candidates, years):
        return {year: item.provenance for year, item in zip(years, candidates, strict=True) if item is not None and item.provenance is not None}

    @staticmethod
    def _audit_map(candidates, years):
        return {
            year: tuple(
                dict.fromkeys(
                    (
                        f"selected_source={item.source if item is not None else _UNAVAILABLE}",
                        *(item.audit if item is not None else ()),
                    )
                )
            )
            for year, item in zip(years, candidates, strict=True)
        }

    @classmethod
    def _diagnostics(cls, metric, years, values, candidates, history, warnings):
        supported = tuple(year for year, value in zip(years, values, strict=True) if value is not None)
        selected = tuple(item for item in candidates if item is not None)
        historical_candidate = history.get(metric, (None, None))[1] if isinstance(history, Mapping) else None
        return OperatingEconomicsMetricDiagnostics(
            metric=metric,
            coverage=Decimal(len(supported)) / Decimal(len(years)) if years else None,
            supported_years=supported,
            confidence=cls._worst_confidence(tuple(item.confidence for item in selected)),
            completeness=Decimal(len(supported)) / Decimal(len(years)) if years else None,
            normalized_ratio=historical_candidate.ratio if historical_candidate else None,
            historical_years=historical_candidate.historical_years if historical_candidate else (),
            provenance=cls._provenance_items(
                *(item.provenance for item in selected),
                *(item.provenance_chain for item in selected),
            ),
            warnings=tuple(dict.fromkeys(warnings)),
        )


OperatingReinvestmentForecastService = OperatingReinvestmentEngine
ReinvestmentForecastService = OperatingReinvestmentEngine

__all__ = [
    "DriverBasedCanonicalFcffAdapter",
    "OperatingReinvestmentEngine",
    "OperatingReinvestmentForecastService",
    "ReinvestmentForecastService",
]
