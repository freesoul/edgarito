"""CAPEX programs, constraints, seed semantics, and segment application."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from edgarito.schemas.forecasting import (
    FcffForecast,
    ForecastDecision,
    ForecastOverride,
    ForecastPlan,
    ForecastStrategy,
    ForecastValueBasis,
)
from edgarito.schemas.operating import (
    EvidenceReference,
    OperatingDriverObservation,
    OperatingInvestmentProgram,
    OperatingReinvestmentSeed,
    OperatingSegment,
    SegmentOperatingEconomicsForecast,
    operating_periods_compatible,
    operating_units_compatible,
)

from .contracts import (
    _CAPEX,
    _CONFIDENCE_RANK,
    _DA,
    _DELTA,
    _FCFF,
    _HUNDRED,
    _METRICS,
    _MONEY_UNITS,
    _OWC,
    _RATE_UNITS,
    _UNAVAILABLE,
    _Candidate,
    _Path,
)


class _ReinvestmentProgramsMixin:
    @classmethod
    def _apply_life_method(
        cls,
        base,
        capex,
        capex_candidates,
        depreciation,
        depreciation_candidates,
        history,
        life: int,
        warnings: list[str],
    ):
        if all(
            item is not None
            and (candidate is None or candidate.source != "normalized_historical")
            for item, candidate in zip(depreciation, depreciation_candidates, strict=True)
        ):
            return depreciation, depreciation_candidates
        history_candidate = history.get(_DA, (None, None))[1]
        if history_candidate is None:
            return depreciation, depreciation_candidates
        current_assets = (
            history_candidate.historical_amount or history_candidate.value
        ) * Decimal(life)
        values = list(depreciation)
        candidates = list(depreciation_candidates)
        for index, amount in enumerate(capex):
            is_normalized_history = (
                candidates[index] is not None
                and candidates[index].source == "normalized_historical"
            )
            if (values[index] is not None and not is_normalized_history) or amount is None:
                if values[index] is not None and not is_normalized_history:
                    current_assets = max(Decimal(0), current_assets - values[index] + (amount or Decimal(0)))
                continue
            dep = current_assets / Decimal(life)
            values[index] = dep
            component = capex_candidates[index]
            candidates[index] = _Candidate(
                dep,
                "derived",
                f"capex_life_cohort_rollforward_{life}_years",
                cls._worst_confidence((history_candidate.confidence, component.confidence if component else "low")),
                component.provenance if component and component.provenance is not None else history_candidate.provenance,
                tuple(dict.fromkeys((*history_candidate.references, *(component.references if component else ())))),
                (f"depreciable_asset_life_years={life}", "simple_cohort_method=true"),
                historical_years=history_candidate.historical_years,
                provenance_chain=cls._provenance_items(history_candidate.provenance, component.provenance if component else None),
            )
            current_assets = max(Decimal(0), current_assets - dep + amount)
        if any(item is not None for item in values):
            warnings.append("D&A used the configured simple CAPEX/life cohort method where no higher-precedence value was available")
        return values, candidates

    @classmethod
    def _program_candidate(
        cls,
        programs: Sequence[OperatingInvestmentProgram],
        year: int,
        expected_unit: str,
        fiscal_period: str,
        period_key: str | None,
        warnings: list[str],
        *,
        range_policy: str = "none",
    ) -> _Candidate | None:
        matching = []
        seen: set[tuple[Any, ...]] = set()
        for program in programs:
            if program.fiscal_year != year or program.segment_id not in {None, "company"}:
                continue
            if not operating_periods_compatible(program.fiscal_period, fiscal_period, program.period_key, period_key):
                continue
            if not cls._is_monetary(program.unit):
                warnings.append(f"FY{year}: investment program {program.program_id} excluded because it is not monetary")
                continue
            if not cls._is_investment_spending(program):
                warnings.append(f"FY{year}: investment program {program.program_id} excluded because it is not clearly investment spending")
                continue
            if not cls._compatible_currency(expected_unit, program.unit, program.currency):
                warnings.append(f"FY{year}: investment program {program.program_id} excluded because currency/scope is incompatible")
                continue
            key = (
                program.program_id,
                program.fiscal_year,
                program.fiscal_period,
                program.period_key,
                program.segment_id,
                program.currency,
                program.unit,
            )
            if key in seen:
                continue
            seen.add(key)
            if program.value is None and not (
                range_policy == "midpoint" and program.low is not None and program.high is not None
            ):
                continue
            matching.append(program)
        if not matching:
            return None
        amount = sum(
            (
                (
                    program.value
                    if program.value is not None
                    else (program.low + program.high) / Decimal(2)
                )
                * program.scale
                for program in matching
            ),
            Decimal(0),
        )
        refs = tuple(program.evidence for program in matching if program.evidence is not None)
        return _Candidate(
            amount,
            "first_party_observation",
            "explicit_monetary_operating_investment_program",
            cls._worst_confidence(tuple(program.confidence for program in matching)),
            refs[0] if refs else None,
            refs,
            tuple(
                f"investment_program={program.program_id};amount="
                f"{program.value if program.value is not None else f'midpoint({program.low},{program.high})'}*{program.scale}"
                for program in matching
            )
            + ("positive_spending_semantics=true",),
            provenance_chain=refs,
        )

    @classmethod
    def _apply_segments(
        cls,
        existing: Sequence[SegmentOperatingEconomicsForecast],
        segments: Sequence[OperatingSegment],
        records: Sequence[OperatingDriverObservation],
        paths: Mapping[tuple[str, str], _Path],
        years: Sequence[int],
        fiscal_period: str,
        period_key: str | None,
        warnings: list[str],
    ) -> tuple[SegmentOperatingEconomicsForecast, ...]:
        by_id = {item.segment.segment_id: item for item in existing}
        segment_values: dict[tuple[str, str, int], _Candidate] = {}
        for item in records:
            metric = cls._metric(item.driver_id)
            if metric not in _METRICS or item.segment_id == "company" or not cls._is_segment(item):
                continue
            if item.fiscal_year not in years or not operating_periods_compatible(item.fiscal_period, fiscal_period, item.period_key, period_key):
                continue
            candidate = cls._direct_candidate(metric, item.fiscal_year, (item,), "currency", fiscal_period, period_key)
            if candidate is None:
                continue
            key = (item.segment_id, metric, item.fiscal_year)
            if key in segment_values and segment_values[key].value != candidate.value:
                raise ValueError(f"Overlapping segment {metric} evidence for {item.segment_id} FY{item.fiscal_year}")
            segment_values[key] = candidate
        for (scope, metric), path in paths.items():
            if scope == "company" or metric not in _METRICS:
                continue
            if scope not in by_id:
                raise ValueError(f"Explicit {metric} target '{scope}' does not match a supplied canonical segment")
            for index, year in enumerate(years):
                revenue = by_id[scope].revenue[index]
                segment_values[(scope, metric, year)] = cls._path_candidate(metric, path, index, revenue)

        result = []
        for segment in existing:
            values_by_metric: dict[str, list[Decimal | None]] = {}
            maps: dict[str, dict[int, Any]] = {}
            for metric in _METRICS:
                values: list[Decimal | None] = []
                candidates: list[_Candidate | None] = []
                for year in years:
                    candidate = segment_values.get((segment.segment.segment_id, metric, year))
                    values.append(candidate.value if candidate else None)
                    candidates.append(candidate)
                values_by_metric[metric] = tuple(values)
                maps[f"{metric}_source_by_year"] = cls._source_map(candidates, years)
                maps[f"{metric}_method_by_year"] = cls._method_map(candidates, years)
                maps[f"{metric}_confidence_by_year"] = cls._confidence_map(candidates, years)
                maps[f"{metric}_provenance_by_year"] = cls._provenance_map(candidates, years)
                maps[f"{metric}_audit_by_year"] = cls._audit_map(candidates, years)
            if not any(value is not None for values in values_by_metric.values() for value in values):
                result.append(segment)
                continue
            diagnostics = segment.diagnostics.model_copy(
                update={
                    metric: cls._diagnostics(metric, years, values_by_metric[metric], [
                        cls._candidate_for_segment(segment, metric, year, segment_values)
                        for year in years
                    ], {}, warnings)
                    for metric in _METRICS
                }
            )
            result.append(
                segment.model_copy(
                    update={
                        **values_by_metric,
                        **maps,
                        "diagnostics": diagnostics,
                        "years": tuple(
                            year_record.model_copy(
                                update={
                                    **{
                                        metric: values_by_metric[metric][index]
                                        for metric in _METRICS
                                    },
                                    **{
                                        f"{metric}_source": maps[f"{metric}_source_by_year"].get(year, _UNAVAILABLE)
                                        for metric in _METRICS
                                    },
                                    **{
                                        f"{metric}_method": maps[f"{metric}_method_by_year"].get(year, _UNAVAILABLE)
                                        for metric in _METRICS
                                    },
                                    **{
                                        f"{metric}_confidence": maps[f"{metric}_confidence_by_year"].get(year, "low")
                                        for metric in _METRICS
                                    },
                                    **{
                                        f"{metric}_provenance": maps[f"{metric}_provenance_by_year"].get(year)
                                        for metric in _METRICS
                                    },
                                    **{
                                        f"{metric}_audit": maps[f"{metric}_audit_by_year"].get(year, ())
                                        for metric in _METRICS
                                    },
                                }
                            )
                            for index, (year, year_record) in enumerate(zip(years, segment.years, strict=True))
                        ),
                    }
                )
            )
        return tuple(result)

    @staticmethod
    def _candidate_for_segment(segment, metric, year, values):
        return values.get((segment.segment.segment_id, metric, year))

    @classmethod
    def _paths(cls, plan, overrides, years) -> dict[tuple[str, str], _Path]:
        records: list[tuple[int, ForecastDecision | ForecastOverride]] = []
        if plan is not None:
            normalized = plan if isinstance(plan, ForecastPlan) else ForecastPlan.model_validate(plan)
            records.extend((0, item) for item in normalized.decisions)
            records.extend((1, item) for item in normalized.overrides)
        records.extend((2, item) for item in cls._coerce_overrides(overrides))
        result: dict[tuple[str, str], tuple[int, _Path]] = {}
        for priority, record in records:
            metric = cls._metric(record.metric)
            if metric in {_DELTA, _FCFF}:
                if record.explicit_path is not None or record.strategy in {ForecastStrategy.EXPLICIT, ForecastStrategy.RATIO, ForecastStrategy.RESIDUAL}:
                    raise ValueError(f"Explicit {metric} overrides are unsupported; it is a derived company identity")
                continue
            if metric not in _METRICS or record.explicit_path is None:
                continue
            strategy = record.strategy
            basis = record.basis
            if strategy not in {ForecastStrategy.EXPLICIT, ForecastStrategy.RATIO, ForecastStrategy.RESIDUAL}:
                raise ValueError(f"{strategy.value} {metric} paths are not supported")
            expected = ForecastValueBasis.ABSOLUTE if strategy in {ForecastStrategy.EXPLICIT, ForecastStrategy.RESIDUAL} else ForecastValueBasis.PERCENT_OF_REVENUE
            if basis != expected:
                raise ValueError(f"{strategy.value} {metric} paths require basis={expected.value}")
            values = tuple(Decimal(item) for item in record.explicit_path)
            if len(values) not in {1, len(years)}:
                raise ValueError(f"Explicit {metric} path must contain one value or exactly the fiscal horizon")
            if any(not item.is_finite() for item in values):
                raise ValueError(f"Explicit {metric} path must contain finite values")
            if metric in {_DA, _CAPEX} and any(item < 0 for item in values):
                raise ValueError(f"Explicit {metric} paths cannot be negative")
            expanded = values * len(years) if len(values) == 1 else values
            references = (record.provenance,) if isinstance(record.provenance, EvidenceReference) else ()
            path = _Path(expanded, strategy, basis, record.provenance, references, record.strategy == ForecastStrategy.RESIDUAL)
            key = (record.scope_id, metric)
            previous = result.get(key)
            if previous is not None and priority == previous[0] and previous[1] != path:
                raise ValueError(f"ambiguous overlapping {metric} forecast paths for {record.scope_id}")
            if previous is None or priority >= previous[0]:
                result[key] = (priority, path)
        return {key: value for key, (_priority, value) in result.items()}

    @staticmethod
    def _coerce_overrides(value) -> tuple[ForecastOverride, ...]:
        if value is None:
            return ()
        if isinstance(value, ForecastOverride):
            return (value,)
        if isinstance(value, Mapping):
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
            return tuple(ForecastOverride.model_validate(item) for item in records)
        if hasattr(value, "metric") and hasattr(value, "strategy"):
            return (value if isinstance(value, ForecastOverride) else ForecastOverride.model_validate(value),)
        return tuple(item if isinstance(item, ForecastOverride) else ForecastOverride.model_validate(item) for item in value)

    @staticmethod
    def _metric(value: Any) -> str:
        normalized = str(getattr(value, "value", value)).strip().casefold().replace("-", "_").replace(" ", "_")
        return {
            "da": _DA,
            "depreciation": _DA,
            "depreciation_to_revenue": _DA,
            "capital_expenditure": _CAPEX,
            "capex_to_revenue": _CAPEX,
            "owc": _OWC,
            "operating_working_capital_to_revenue": _OWC,
            "delta_nwc": _DELTA,
            "change_in_working_capital": _DELTA,
            "fcff": _FCFF,
        }.get(normalized, normalized)

    @staticmethod
    def _ratio_value(item: OperatingDriverObservation) -> Decimal | None:
        unit = item.unit.casefold().replace(" ", "_")
        basis = (item.basis or "").casefold().replace("-", "_").replace(" ", "_")
        if unit not in _RATE_UNITS and "percent" not in basis and "ratio" not in basis:
            return None
        value = item.normalized_value
        if "bp" in unit:
            value /= Decimal(100)
        elif unit in {"ratio", "rate", "decimal", "fraction"}:
            value *= _HUNDRED
        return value

    @staticmethod
    def _compatible_currency(expected_unit: str, actual_unit: str, currency: str | None) -> bool:
        if not operating_units_compatible(expected_unit, actual_unit):
            return False
        def code(value: str | None) -> str | None:
            folded = (value or "").casefold()
            return next((item for item in _MONEY_UNITS if item in folded and item != "currency"), None)
        expected = code(expected_unit)
        actual = currency.casefold() if currency else code(actual_unit)
        return expected is None or actual is None or expected == actual

    @staticmethod
    def _is_company(item: OperatingDriverObservation) -> bool:
        return (
            item.segment_id == "company"
            and (item.scope or "").casefold() != "segment"
        ) or (item.scope or "").casefold() in {"company", "consolidated", "total"}

    @staticmethod
    def _is_segment(item: OperatingDriverObservation) -> bool:
        return (item.scope or "").casefold() == "segment" or item.segment_id != "company"

    @staticmethod
    def _is_monetary(unit: str) -> bool:
        folded = unit.casefold().replace("$", "usd")
        return any(token in folded for token in _MONEY_UNITS) and not any(token in folded for token in ("unit", "store", "capacity", "user"))

    @staticmethod
    def _is_investment_spending(program: OperatingInvestmentProgram) -> bool:
        text = f"{program.name} {program.purpose or ''}".casefold()
        return any(term in text for term in ("capex", "capital", "investment", "facility", "expansion", "construction", "build", "spend"))

    @staticmethod
    def _programs(value) -> tuple[OperatingInvestmentProgram, ...]:
        if value is None:
            return ()
        if isinstance(value, OperatingInvestmentProgram):
            value = (value,)
        elif isinstance(value, Mapping):
            value = tuple(value.values())
        result = []
        seen: set[tuple[Any, ...]] = set()
        for item in value:
            program = item if isinstance(item, OperatingInvestmentProgram) else OperatingInvestmentProgram.model_validate(item)
            identity = (
                program.program_id,
                program.fiscal_year,
                program.fiscal_period,
                program.period_key,
                program.segment_id,
                program.currency,
                program.unit,
            )
            if identity in seen:
                continue
            seen.add(identity)
            result.append(program)
        return tuple(result)

    @staticmethod
    def _coerce_seed(value) -> OperatingReinvestmentSeed | None:
        if value is None:
            return None
        if isinstance(value, FcffForecast):
            return OperatingReinvestmentSeed(
                fiscal_year=value.base_fiscal_year,
                # A canonical YTD+forecast artifact still carries a full-year
                # forecast row.  Its base OWC is the accounting seed for that
                # row; the separate ytd_anchor remains exclusively a DCF/stub
                # representation and must not become the full-year delta seed.
                fiscal_period="LTM" if value.seed_type.value == "TTM" else "FY",
                mode=value.seed_type.value,
                unit=value.unit,
                value=value.base_operating_working_capital,
                source="canonical_seed_forecast",
                confidence="high",
            )
        return value if isinstance(value, OperatingReinvestmentSeed) else OperatingReinvestmentSeed.model_validate(value)

    @staticmethod
    def _validate_ytd_capex_constraint(seed, first_year, constraints) -> None:
        if not isinstance(seed, FcffForecast) or seed.ytd_anchor is None:
            return
        if first_year != seed.ytd_anchor.fiscal_year:
            return
        constraint = constraints.get(first_year)
        if constraint is None:
            return
        actual = seed.ytd_anchor.actual_capital_expenditures
        if (
            constraint.point is not None and constraint.point < actual
        ) or (
            constraint.maximum is not None and constraint.maximum < actual
        ):
            raise ValueError(
                f"FY{first_year} CAPEX constraint is below reported YTD CAPEX"
            )

    @classmethod
    def _seed_selection(
        cls,
        seed: OperatingReinvestmentSeed | None,
        records: Sequence[OperatingDriverObservation],
        first_year: int | None,
        expected_unit: str,
        fiscal_period: str,
        period_key: str | None,
        warnings: list[str],
    ) -> tuple[Any | None, Decimal | None]:
        candidate_seed = seed
        if candidate_seed is not None:
            mode = candidate_seed.mode.casefold().replace("_", " ")
            supported = {"fy", "ttm", "ytd+forecast", "ytd run-rate", "ytd", "ltm"}
            if mode not in supported:
                warnings.append(f"OWC seed mode {candidate_seed.mode!r} cannot be represented safely")
                return candidate_seed, None
            if not cls._compatible_currency(expected_unit, candidate_seed.unit, None):
                warnings.append("OWC seed unit/currency is incompatible with the forecast")
                return candidate_seed, None
            if first_year is not None and mode in {"fy", "ttm", "ltm"} and candidate_seed.fiscal_year != first_year - 1:
                warnings.append("OWC seed fiscal year is not the compatible prior period")
                return candidate_seed, None
            if first_year is not None and mode in {"ytd", "ytd+forecast", "ytd run-rate"} and candidate_seed.fiscal_year not in {first_year, first_year - 1}:
                warnings.append("OWC YTD seed fiscal year is not compatible with the forecast")
                return candidate_seed, None
            if mode in {"fy"} and candidate_seed.fiscal_period != "FY":
                warnings.append("FY OWC seed does not have FY period semantics")
                return candidate_seed, None
            if mode in {"ttm", "ltm"} and candidate_seed.fiscal_period not in {"LTM", "FY"}:
                warnings.append("TTM OWC seed does not have LTM period semantics")
                return candidate_seed, None
            if (
                mode in {"ytd", "ytd+forecast", "ytd run-rate"}
                and candidate_seed.fiscal_period != "YTD"
                and candidate_seed.source != "canonical_seed_forecast"
            ):
                warnings.append("YTD OWC seed does not have YTD period semantics")
                return candidate_seed, None
            return candidate_seed, candidate_seed.value
        if first_year is None:
            return None, None
        choices: list[tuple[int, OperatingDriverObservation]] = []
        for position, item in enumerate(records):
            if cls._metric(item.driver_id) != _OWC or item.fiscal_year != first_year - 1:
                continue
            if not operating_periods_compatible(item.fiscal_period, fiscal_period, item.period_key, period_key):
                continue
            if not cls._compatible_currency(expected_unit, item.unit, item.currency):
                continue
            if cls._ratio_value(item) is not None:
                continue
            choices.append((position, item))
        if not choices:
            return None, None
        _, selected = max(
            choices,
            key=lambda item: (
                1 if item[1].is_total else 0,
                _CONFIDENCE_RANK.get(item[1].confidence, 0),
                -item[0],
            ),
        )
        return selected, selected.normalized_value

    @classmethod
    def _resolved_seed(
        cls, selection: tuple[Any | None, Decimal | None]
    ) -> OperatingReinvestmentSeed | None:
        selected, amount = selection
        if selected is None or amount is None:
            return None
        if isinstance(selected, OperatingReinvestmentSeed):
            return selected
        return OperatingReinvestmentSeed(
            fiscal_year=selected.fiscal_year,
            fiscal_period=selected.fiscal_period,
            period_key=selected.period_key,
            mode=(
                "FY"
                if selected.fiscal_period == "FY"
                else "TTM"
                if selected.fiscal_period == "LTM"
                else "YTD"
            ),
            unit=selected.unit,
            value=amount,
            provenance=selected.provenance or selected.evidence,
            source="historical_seed",
            confidence=selected.confidence,
        )


__all__ = ["_ReinvestmentProgramsMixin"]
