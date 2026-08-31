"""Provider-neutral operating tax-rate and NOPAT stage.

The stage consumes the company economics artifact after EBIT selection.  It
never allocates a company tax burden to segments and has no provider, model, or
valuation dependencies.
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
    operating_periods_compatible,
)
from edgarito.services.financials.effective_tax import (
    calculate_effective_tax_rate,
)
from edgarito.services.operating._forecast.contracts import (
    _CONFIDENCE_RANK,
    _worst_confidence,
)

_RATE = "tax_rate"
_TAX = "tax"
_NOPAT = "nopat"
_PRETAX = "pretax_income"
_TAX_EXPENSE = "income_tax_expense"
_PERCENT = Decimal(100)
_RATE_UNITS = frozenset(
    {"%", "percent", "percentage", "percentage_points", "percentage_point", "pp", "bps", "bp", "ratio", "rate", "decimal", "fraction"}
)
_DIRECT_FORWARD_ORIGINS = frozenset({
    "first_party_observation",
    "extracted_evidence",
    "reasoned_assumption",
    "model_assumption",
})
_HISTORICAL_SOURCE_RANK = {
    "reported": 4,
    "first_party_observation": 3,
    "extracted_evidence": 3,
    "management_guidance": 2,
    "derived": 1,
    "forward_evidence": 1,
    "reasoned_assumption": 0,
    "model_assumption": -1,
}
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
_CURRENCY_TERMS = frozenset(
    {"usd", "eur", "gbp", "jpy", "cny", "cad", "aud", "chf", "currency", "dollar", "dollars", "$", "€", "£"}
)


@dataclass(frozen=True)
class _Candidate:
    value: Decimal
    source: str
    method: str
    confidence: str
    provenance: Any = None
    references: tuple[EvidenceReference, ...] = ()
    observations: tuple[OperatingDriverObservation, ...] = ()
    audit: tuple[str, ...] = ()
    provenance_chain: tuple[Any, ...] = ()


@dataclass(frozen=True)
class _Path:
    values: tuple[Decimal, ...]
    strategy: ForecastStrategy
    basis: ForecastValueBasis
    provenance: Any = None
    references: tuple[EvidenceReference, ...] = ()


@dataclass(frozen=True)
class _HistoricalPair:
    year: int
    fiscal_period: str
    period_key: str | None
    unit: str
    currency: str | None
    pretax: Decimal
    tax: Decimal
    pretax_observation: OperatingDriverObservation
    tax_observation: OperatingDriverObservation
    rate: Decimal

    @property
    def signature(self) -> tuple[str, str | None, str, str | None]:
        return (self.fiscal_period, self.period_key, self.unit, self.currency)

    @property
    def confidence(self) -> str:
        return _worst_confidence(
            (self.pretax_observation.confidence, self.tax_observation.confidence)
        )


@dataclass(frozen=True)
class _History:
    rates: tuple[Decimal, ...] = ()
    years: tuple[int, ...] = ()
    pretax_by_year: dict[int, Decimal] | None = None
    tax_by_year: dict[int, Decimal] | None = None
    provenance_by_year: dict[int, Any] | None = None
    pretax_provenance_by_year: dict[int, Any] | None = None
    tax_provenance_by_year: dict[int, Any] | None = None
    effective_provenance_by_year: dict[int, Any] | None = None
    effective_provenance_chain_by_year: dict[int, tuple[Any, ...]] | None = None
    effective_audit_by_year: dict[int, tuple[str, ...]] | None = None
    normalized: Decimal | None = None
    dispersion: Decimal | None = None
    confidence: str = "low"
    provenance: tuple[Any, ...] = ()
    references: tuple[EvidenceReference, ...] = ()
    warnings: tuple[str, ...] = ()


class OperatingTaxNopatEngine:
    """Apply company-only tax-rate, operating-tax, and NOPAT calculations."""

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
    ) -> CompanyOperatingEconomicsForecast:
        policy = (
            config
            if isinstance(config, OperatingEconomicsForecastConfig)
            else OperatingEconomicsForecastConfig.model_validate(config or {})
        )
        records = tuple(
            item
            if isinstance(item, OperatingDriverObservation)
            else OperatingDriverObservation.model_validate(item)
            for item in (observations or ())
        )
        paths = self._paths(plan, overrides, base.fiscal_years)
        history = self._historical_rates(records, base.fiscal_years, policy, fiscal_period, period_key)
        attempted = bool(
            paths
            or any(self._metric(item.driver_id) in {_RATE, _PRETAX, _TAX_EXPENSE} for item in records)
            or history.rates
            or policy.tax_rate_fallback is not None
        )
        if not attempted:
            return base

        rates: list[Decimal | None] = []
        taxes: list[Decimal | None] = []
        nopats: list[Decimal | None] = []
        rate_candidates: list[_Candidate | None] = []
        tax_candidates: list[_Candidate | None] = []
        nopat_candidates: list[_Candidate | None] = []
        warnings = list(history.warnings)

        first_forecast_year = base.fiscal_years[0] if base.fiscal_years else None
        for index, year in enumerate(base.fiscal_years):
            rate_candidate = None
            path = paths.get(("company", _RATE))
            if path is not None:
                rate_candidate = _Candidate(
                    path.values[index],
                    "explicit",
                    f"forecast_plan_{path.strategy.value}_tax_rate",
                    "high",
                    path.provenance,
                    path.references,
                    audit=("tax_rate_basis=percentage_points",),
                    provenance_chain=(path.provenance,) if path.provenance is not None else (),
                )
            if rate_candidate is None:
                rate_candidate = self._forward_candidate(
                    year,
                    records,
                    fiscal_period=fiscal_period,
                    period_key=period_key,
                    first_forecast_year=first_forecast_year,
                )
            if rate_candidate is None and history.normalized is not None:
                historical_provenance = history.effective_provenance_by_year or {}
                rate_candidate = _Candidate(
                    history.normalized,
                    "normalized_historical",
                    f"{policy.tax_rate_normalization_method}_recent_historical_effective_tax_rate",
                    history.confidence,
                    historical_provenance.get(history.years[-1])
                    if history.years
                    else None,
                    history.references,
                    audit=tuple(
                        [
                            f"historical_effective_tax_rate_years={','.join(map(str, history.years))}",
                            f"historical_effective_tax_rates={','.join(map(str, history.rates))}",
                            f"historical_effective_tax_rate_dispersion={history.dispersion}",
                        ]
                    ),
                    provenance_chain=history.provenance,
                )
            if rate_candidate is None and policy.tax_rate_fallback is not None:
                rate_candidate = _Candidate(
                    policy.tax_rate_fallback,
                    "configured_fallback",
                    "configured_tax_rate_fallback",
                    "low",
                    None,
                    audit=("tax_rate_fallback_configured",),
                )
            if rate_candidate is None:
                warnings.append(f"FY{year}: tax rate unavailable; NOPAT requires a supported tax rate")

            rate = rate_candidate.value if rate_candidate is not None else None
            ebit = base.consolidated_ebit[index]
            tax = None
            nopat = None
            tax_candidate = None
            nopat_candidate = None
            if ebit is not None and rate is not None:
                if ebit < 0:
                    warning = (
                        f"FY{year}: negative EBIT tax-shield realization is uncertain"
                        if policy.negative_ebit_policy == "unavailable"
                        else f"FY{year}: negative EBIT uses the mechanical tax-shield identity; "
                        "immediate tax-shield realization is uncertain"
                    )
                    warnings.append(warning)
                if policy.negative_ebit_policy == "unavailable" and ebit < 0:
                    warnings.append(f"FY{year}: tax and NOPAT unavailable under negative EBIT policy")
                else:
                    tax = ebit * rate / _PERCENT
                    # Compute from the tax amount so the reported NOPAT and
                    # modeled tax retain the exact operating identity under
                    # Decimal context rounding.
                    nopat = ebit - tax
                    calculation_confidence = _worst_confidence(
                        (
                            base.ebit_confidence_by_year.get(
                                year, base.confidence_by_year.get(year, "low")
                            ),
                            rate_candidate.confidence,
                        )
                    )
                    calculation_provenance = (
                        rate_candidate.provenance
                        or base.ebit_provenance_by_year.get(year)
                        or base.provenance_by_year.get(year)
                    )
                    ebit_provenance_chain = self._provenance_items(
                        base.ebit_provenance_by_year.get(year),
                        base.provenance_by_year.get(year),
                        base.provenance_chain_by_year.get(year, ()),
                    )
                    calculation_provenance_chain = self._provenance_items(
                        rate_candidate.provenance,
                        rate_candidate.provenance_chain,
                        *ebit_provenance_chain,
                    )
                    calculation_references = tuple(
                        dict.fromkeys(
                            (
                                *base.source_provenance_by_year.get(year, ()),
                                *rate_candidate.references,
                            )
                        )
                    )
                    tax_candidate = _Candidate(
                        tax,
                        "modeled_operating_tax",
                        "ebit_times_tax_rate_divided_by_100",
                        calculation_confidence,
                        calculation_provenance,
                        calculation_references,
                        audit=(f"tax_identity=ebit*tax_rate/100={tax}",),
                        provenance_chain=calculation_provenance_chain,
                    )
                    nopat_candidate = _Candidate(
                        nopat,
                        "modeled_nopat",
                        "ebit_times_one_minus_tax_rate_divided_by_100",
                        calculation_confidence,
                        calculation_provenance,
                        calculation_references,
                        audit=(f"nopat_identity=ebit-tax={nopat}",),
                        provenance_chain=calculation_provenance_chain,
                    )
            elif ebit is None and rate is not None:
                warnings.append(f"FY{year}: tax and NOPAT unavailable because EBIT is unavailable")

            rates.append(rate)
            taxes.append(tax)
            nopats.append(nopat)
            rate_candidates.append(rate_candidate)
            tax_candidates.append(tax_candidate)
            nopat_candidates.append(nopat_candidate)

        tax_diag = self._diagnostics(_TAX, base.fiscal_years, taxes, tax_candidates, warnings)
        rate_diag = self._diagnostics(
            _RATE,
            base.fiscal_years,
            rates,
            rate_candidates,
            warnings,
            history=history,
        )
        nopat_diag = self._diagnostics(_NOPAT, base.fiscal_years, nopats, nopat_candidates, warnings)
        diagnostics = base.diagnostics.model_copy(
            update={
                "tax_rate": rate_diag,
                "tax": tax_diag,
                "nopat": nopat_diag,
            }
        )

        rate_sources = self._source_map(rate_candidates, base.fiscal_years)
        rate_methods = self._method_map(rate_candidates, base.fiscal_years)
        rate_confidences = self._confidence_map(rate_candidates, base.fiscal_years)
        rate_provenance = self._provenance_map(rate_candidates, base.fiscal_years)
        rate_audits = self._audit_map(rate_candidates, base.fiscal_years)
        tax_sources = self._source_map(tax_candidates, base.fiscal_years)
        tax_methods = self._method_map(tax_candidates, base.fiscal_years)
        tax_confidences = self._confidence_map(tax_candidates, base.fiscal_years)
        tax_provenance = self._provenance_map(tax_candidates, base.fiscal_years)
        tax_audits = self._audit_map(tax_candidates, base.fiscal_years)
        nopat_sources = self._source_map(nopat_candidates, base.fiscal_years)
        nopat_methods = self._method_map(nopat_candidates, base.fiscal_years)
        nopat_confidences = self._confidence_map(nopat_candidates, base.fiscal_years)
        nopat_provenance = self._provenance_map(nopat_candidates, base.fiscal_years)
        nopat_audits = self._audit_map(nopat_candidates, base.fiscal_years)
        merged_warnings = tuple(dict.fromkeys((*base.warnings, *warnings)))
        years = tuple(
            item.model_copy(
                update={
                    "tax_rate": rates[index],
                    "tax": taxes[index],
                    "nopat": nopats[index],
                    "tax_rate_source": rate_sources.get(year, "unavailable"),
                    "tax_rate_method": rate_methods.get(year, "unavailable"),
                    "tax_rate_confidence": rate_confidences.get(year, "low"),
                    "tax_rate_provenance": rate_provenance.get(year),
                    "tax_rate_audit": rate_audits.get(year, ()),
                    "tax_source": tax_sources.get(year, "unavailable"),
                    "tax_method": tax_methods.get(year, "unavailable"),
                    "tax_confidence": tax_confidences.get(year, "low"),
                    "tax_provenance": tax_provenance.get(year),
                    "tax_audit": tax_audits.get(year, ()),
                    "nopat_source": nopat_sources.get(year, "unavailable"),
                    "nopat_method": nopat_methods.get(year, "unavailable"),
                    "nopat_confidence": nopat_confidences.get(year, "low"),
                    "nopat_provenance": nopat_provenance.get(year),
                    "nopat_audit": nopat_audits.get(year, ()),
                }
            )
            for index, (item, year) in enumerate(zip(base.years, base.fiscal_years, strict=True))
        )
        return base.model_copy(
            update={
                "tax_rate": tuple(rates),
                "tax": tuple(taxes),
                "nopat": tuple(nopats),
                "tax_rate_source_by_year": rate_sources,
                "tax_rate_method_by_year": rate_methods,
                "tax_rate_confidence_by_year": rate_confidences,
                "tax_rate_provenance_by_year": rate_provenance,
                "tax_rate_audit_by_year": rate_audits,
                "tax_source_by_year": tax_sources,
                "tax_method_by_year": tax_methods,
                "tax_confidence_by_year": tax_confidences,
                "tax_provenance_by_year": tax_provenance,
                "tax_audit_by_year": tax_audits,
                "nopat_source_by_year": nopat_sources,
                "nopat_method_by_year": nopat_methods,
                "nopat_confidence_by_year": nopat_confidences,
                "nopat_provenance_by_year": nopat_provenance,
                "nopat_audit_by_year": nopat_audits,
                "historical_pretax_income_by_year": history.pretax_by_year or {},
                "historical_income_tax_expense_by_year": history.tax_by_year or {},
                "historical_effective_tax_rate_by_year": {
                    year: rate for year, rate in zip(history.years, history.rates, strict=True)
                },
                "historical_pretax_income_provenance_by_year": {
                    **(history.pretax_provenance_by_year or {})
                },
                "historical_income_tax_expense_provenance_by_year": {
                    **(history.tax_provenance_by_year or {})
                },
                "historical_effective_tax_rate_provenance_by_year": {
                    **(history.effective_provenance_by_year or {})
                },
                "historical_effective_tax_rate_provenance_chain_by_year": {
                    **(history.effective_provenance_chain_by_year or {})
                },
                "historical_effective_tax_rate_audit_by_year": {
                    **(history.effective_audit_by_year or {})
                },
                "diagnostics": diagnostics,
                "warnings": merged_warnings,
                "years": years,
            }
        )

    forecast = apply
    build = apply

    @classmethod
    def _paths(cls, plan, overrides, years) -> dict[tuple[str, str], _Path]:
        records: list[ForecastDecision | ForecastOverride] = []
        if plan is not None:
            normalized = plan if isinstance(plan, ForecastPlan) else ForecastPlan.model_validate(plan)
            records.extend(normalized.decisions)
            records.extend(normalized.overrides)
        records.extend(cls._coerce_overrides(overrides))
        result: dict[tuple[str, str], _Path] = {}
        for record in records:
            metric = cls._metric(record.metric)
            if metric not in {_RATE, _TAX, _NOPAT}:
                continue
            if record.scope.value == "segment":
                raise ValueError(
                    f"Segment {metric} decisions/overrides are unsupported; tax, tax rate, and NOPAT are company-only"
                )
            if metric == _TAX:
                if record.explicit_path is not None:
                    raise ValueError(
                        "Explicit TAX overrides are unsupported; TAX is a monetary output "
                        "derived from EBIT and TAX_RATE"
                    )
                continue
            if metric == _NOPAT:
                if record.explicit_path is not None:
                    raise ValueError(
                        "Explicit NOPAT overrides are unsupported; NOPAT is calculated "
                        "from EBIT and TAX_RATE"
                    )
                continue
            if record.explicit_path is None:
                continue
            if record.basis != ForecastValueBasis.PERCENTAGE_POINTS:
                raise ValueError(
                    f"Explicit {metric} paths require basis=percentage_points; percent_of_revenue is invalid"
                )
            values = tuple(Decimal(item) for item in record.explicit_path)
            if len(values) not in {1, len(years)}:
                raise ValueError(f"Explicit {metric} path must contain one value or exactly the fiscal horizon")
            if any(not item.is_finite() for item in values):
                raise ValueError(f"Explicit {metric} path must contain finite values")
            if any(item < 0 or item > 100 for item in values):
                raise ValueError(f"Explicit {metric} paths must be between 0 and 100 percentage points")
            expanded = values * len(years) if len(values) == 1 else values
            references = (record.provenance,) if isinstance(record.provenance, EvidenceReference) else ()
            candidate = _Path(expanded, record.strategy, record.basis, record.provenance, references)
            key = ("company", metric)
            previous = result.get(key)
            if previous is not None and previous != candidate:
                raise ValueError(f"ambiguous overlapping {metric} forecast paths for company")
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
        if isinstance(value, (str, bytes)):
            return ()
        try:
            values = tuple(value)
        except TypeError:
            values = (value,)
        return tuple(item if isinstance(item, ForecastOverride) else ForecastOverride.model_validate(item) for item in values)

    @classmethod
    def _forward_candidate(
        cls,
        year: int,
        records: Sequence[OperatingDriverObservation],
        *,
        fiscal_period: str,
        period_key: str | None,
        first_forecast_year: int | None,
    ) -> _Candidate | None:
        choices: list[tuple[int, OperatingDriverObservation, Decimal, str]] = []
        for position, observation in enumerate(records):
            if cls._metric(observation.driver_id) != _RATE or observation.fiscal_year != year:
                continue
            if not cls._is_company(observation):
                continue
            if not operating_periods_compatible(observation.fiscal_period, fiscal_period, observation.period_key, period_key):
                continue
            value = cls._rate_value(observation)
            if value is None:
                continue
            origin = observation.origin
            if origin == "management_guidance":
                source = "management_guidance"
            elif origin in {"first_party_observation", "extracted_evidence"}:
                source = "first_party_observation"
            elif origin == "reasoned_assumption":
                source = "reasoned_assumption"
            elif origin == "model_assumption":
                source = "model_assumption"
            elif origin == "forward_evidence":
                source = "forward_evidence"
            elif cls._is_forward(observation, first_forecast_year):
                source = "forward_evidence"
            else:
                continue
            choices.append((position, observation, value, source))
        if not choices:
            return None
        _, observation, value, source = max(
            choices,
            key=lambda item: (
                _ORIGIN_RANK.get(item[1].origin, 0),
                {"management_guidance": 5, "first_party_observation": 4, "reasoned_assumption": 3, "forward_evidence": 2, "model_assumption": 1}[item[3]],
                1 if item[1].is_total else 0,
                _CONFIDENCE_RANK[item[1].confidence],
                item[1].evidence is not None,
                -item[0],
            ),
        )
        method = {
            "management_guidance": "management_guidance_tax_rate",
            "first_party_observation": "first_party_forward_tax_rate",
            "reasoned_assumption": "reasoned_forward_tax_rate",
            "model_assumption": "model_forward_tax_rate",
            "forward_evidence": "deterministic_forward_tax_rate_evidence",
        }[source]
        return _Candidate(
            value,
            source,
            method,
            observation.confidence,
            observation.provenance or observation.evidence,
            cls._references((observation,)),
            (observation,),
            audit=("tax_rate_basis=percentage_points",),
            provenance_chain=tuple(
                item
                for item in (observation.provenance, observation.evidence)
                if item is not None
            ),
        )

    @classmethod
    def _historical_rates(
        cls,
        records: Sequence[OperatingDriverObservation],
        years: tuple[int, ...],
        policy: OperatingEconomicsForecastConfig,
        fiscal_period: str,
        period_key: str | None,
    ) -> _History:
        if not years:
            return _History()
        cutoff = years[0]
        grouped: dict[tuple[int, str, str | None, str, str | None], dict[str, tuple[int, OperatingDriverObservation, Decimal]]] = {}
        warnings: list[str] = []
        for position, observation in enumerate(records):
            metric = cls._metric(observation.driver_id)
            if metric not in {_PRETAX, _TAX_EXPENSE} or not cls._is_company(observation):
                continue
            if observation.fiscal_year >= cutoff:
                continue
            if not operating_periods_compatible(observation.fiscal_period, fiscal_period, observation.period_key, period_key):
                continue
            if not cls._is_monetary_unit(observation.unit):
                warnings.append(
                    f"FY{observation.fiscal_year}: historical tax rate excluded because the unit is not monetary"
                )
                continue
            try:
                value = observation.normalized_value
            except ValueError:
                continue
            key = (observation.fiscal_year, observation.fiscal_period, observation.period_key, observation.unit, observation.currency)
            selected = grouped.setdefault(key, {}).get(metric)
            candidate = (position, observation, value)
            if selected is None or cls._observation_key(candidate) > cls._observation_key(selected):
                grouped[key][metric] = candidate
        candidates_by_year: dict[int, list[_HistoricalPair]] = {}
        for key in sorted(
            grouped,
            key=lambda item: (item[0], item[1], item[2] or "", item[3], item[4] or ""),
        ):
            facts = grouped[key]
            pretax_item = facts.get(_PRETAX)
            tax_item = facts.get(_TAX_EXPENSE)
            if pretax_item is None or tax_item is None:
                continue
            pretax = pretax_item[2]
            tax = tax_item[2]
            rate = calculate_effective_tax_rate(pretax, tax)
            if rate is None:
                if pretax <= 0:
                    warnings.append(f"FY{key[0]}: historical tax rate excluded because pretax income is not positive")
                elif tax < 0:
                    warnings.append(f"FY{key[0]}: historical tax rate excluded because tax expense is negative")
                else:
                    warnings.append(f"FY{key[0]}: historical tax rate excluded because effective rate exceeds 100 percentage points")
                continue
            candidates_by_year.setdefault(key[0], []).append(
                _HistoricalPair(
                    year=key[0],
                    fiscal_period=key[1],
                    period_key=key[2],
                    unit=key[3],
                    currency=key[4],
                    pretax=pretax,
                    tax=tax,
                    pretax_observation=pretax_item[1],
                    tax_observation=tax_item[1],
                    rate=rate,
                )
            )
        pairs: list[_HistoricalPair] = []
        for year in sorted(candidates_by_year):
            candidates = candidates_by_year[year]
            if len(candidates) > 1:
                signatures = {item.signature for item in candidates}
                if len(signatures) > 1:
                    descriptions = ", ".join(
                        f"{period}/{period_key or '-'} {unit}/{currency or '-'}"
                        for period, period_key, unit, currency in sorted(
                            signatures,
                            key=lambda item: (
                                item[0],
                                item[1] or "",
                                item[2],
                                item[3] or "",
                            ),
                        )
                    )
                    warnings.append(
                        f"FY{year}: historical tax rate excluded because compatible "
                        f"period/unit/currency alternatives are ambiguous ({descriptions})"
                    )
                    continue
            pairs.append(
                max(
                    candidates,
                    key=lambda item: (
                        cls._observation_precedence(item.pretax_observation),
                        cls._observation_precedence(item.tax_observation),
                    ),
                )
            )
        pairs = pairs[-policy.historical_window :]
        if not pairs:
            return _History(warnings=tuple(dict.fromkeys(warnings)))
        rates = tuple(item.rate for item in pairs)
        years_out = tuple(item.year for item in pairs)
        normalized = cls._aggregate(rates, policy.tax_rate_normalization_method)
        scale = max(max(abs(item) for item in rates), Decimal(1))
        dispersion = (max(rates) - min(rates)) / scale if len(rates) > 1 else Decimal(0)
        if dispersion > policy.tax_rate_dispersion_threshold:
            warnings.append(
                f"Historical tax-rate dispersion {dispersion} exceeds configured threshold {policy.tax_rate_dispersion_threshold}"
            )
            normalized = None
            confidence = "low"
        elif dispersion <= policy.tax_rate_high_confidence_dispersion_threshold:
            confidence = "high"
        elif dispersion <= policy.tax_rate_medium_confidence_dispersion_threshold:
            confidence = "medium"
        else:
            confidence = "low"
            warnings.append(
                f"Historical tax-rate dispersion {dispersion} lowers confidence"
            )
        confidence = _worst_confidence(
            tuple(item.confidence for item in pairs) + (confidence,)
        )
        pretax_provenance_by_year = {
            item.year: item.pretax_observation.provenance
            or item.pretax_observation.evidence
            for item in pairs
            if item.pretax_observation.provenance is not None
            or item.pretax_observation.evidence is not None
        }
        tax_provenance_by_year = {
            item.year: item.tax_observation.provenance
            or item.tax_observation.evidence
            for item in pairs
            if item.tax_observation.provenance is not None
            or item.tax_observation.evidence is not None
        }
        effective_provenance_by_year = {
            item.year: pretax_provenance_by_year.get(item.year)
            or tax_provenance_by_year.get(item.year)
            for item in pairs
            if item.year in pretax_provenance_by_year
            or item.year in tax_provenance_by_year
        }
        effective_provenance_chain_by_year = {
            item.year: cls._provenance_items(
                pretax_provenance_by_year.get(item.year),
                tax_provenance_by_year.get(item.year),
                *cls._references((item.pretax_observation, item.tax_observation)),
            )
            for item in pairs
        }
        effective_audit_by_year = {
            item.year: (
                "effective_tax_rate_inputs=pretax_income,income_tax_expense",
                f"pretax_income_provenance={pretax_provenance_by_year.get(item.year)}",
                f"income_tax_expense_provenance={tax_provenance_by_year.get(item.year)}",
            )
            for item in pairs
        }
        provenance = cls._provenance_items(
            *pretax_provenance_by_year.values(),
            *tax_provenance_by_year.values(),
            *(
                reference
                for item in pairs
                for reference in cls._references(
                    (item.pretax_observation, item.tax_observation)
                )
            ),
        )
        references = cls._dedupe_references(
            reference
            for item in pairs
            for reference in cls._references(
                (item.pretax_observation, item.tax_observation)
            )
        )
        provenance_by_year = dict(effective_provenance_by_year)
        return _History(
            rates=rates,
            years=years_out,
            pretax_by_year={item.year: item.pretax for item in pairs},
            tax_by_year={item.year: item.tax for item in pairs},
            provenance_by_year=provenance_by_year,
            pretax_provenance_by_year=pretax_provenance_by_year,
            tax_provenance_by_year=tax_provenance_by_year,
            effective_provenance_by_year=effective_provenance_by_year,
            effective_provenance_chain_by_year=effective_provenance_chain_by_year,
            effective_audit_by_year=effective_audit_by_year,
            normalized=normalized,
            dispersion=dispersion,
            confidence=confidence,
            provenance=provenance,
            references=references,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _aggregate(values: Sequence[Decimal], method: str) -> Decimal:
        if method == "weighted_recent":
            denominator = Decimal(sum(range(1, len(values) + 1)))
            return sum((value * Decimal(position) for position, value in enumerate(values, 1)), Decimal(0)) / denominator
        ordered = sorted(values)
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / Decimal(2)

    @staticmethod
    def _observation_key(item) -> tuple[int, int, int, int, int]:
        _, observation, _ = item
        return (
            1 if observation.is_total else 0,
            _HISTORICAL_SOURCE_RANK.get(observation.origin, 0),
            _CONFIDENCE_RANK[observation.confidence],
            1 if observation.evidence is not None else 0,
            -item[0],
        )

    @staticmethod
    def _observation_precedence(observation: OperatingDriverObservation) -> tuple[int, int, int, str, str]:
        reference = observation.evidence or observation.provenance
        provider = str(getattr(reference, "provider", ""))
        accession = str(
            getattr(reference, "accession", getattr(reference, "accession_number", ""))
        )
        return (
            1 if observation.is_total else 0,
            _HISTORICAL_SOURCE_RANK.get(observation.origin, 0),
            _CONFIDENCE_RANK[observation.confidence],
            provider,
            accession,
        )

    @staticmethod
    def _is_forward(observation: OperatingDriverObservation, first_forecast_year: int | None) -> bool:
        if first_forecast_year is not None and observation.fiscal_year >= first_forecast_year:
            return True
        text = " ".join(
            item or ""
            for item in (observation.method, observation.basis, observation.scope_evidence)
        ).casefold()
        return "forward" in text or "estimate" in text

    @staticmethod
    def _metric(value: Any) -> str:
        normalized = str(getattr(value, "value", value)).strip().casefold().replace("-", "_").replace(" ", "_")
        return {
            "effective_tax_rate": _RATE,
            "tax_expense": _TAX_EXPENSE,
            "income_tax": _TAX_EXPENSE,
            "pretax": _PRETAX,
            "pre_tax_income": _PRETAX,
            "tax_rate_percentage": _RATE,
            "forward_tax_rate": _RATE,
            "tax_rate_guidance": _RATE,
        }.get(normalized, normalized)

    @classmethod
    def _rate_value(cls, observation: OperatingDriverObservation) -> Decimal | None:
        unit = observation.unit.casefold().replace(" ", "_")
        if unit not in _RATE_UNITS and "basis_point" not in unit:
            return None
        try:
            value = observation.normalized_value
        except ValueError:
            return None
        if "bp" in unit or "basis_point" in unit:
            value /= Decimal(100)
        elif unit in {"ratio", "rate", "decimal", "fraction"}:
            value *= _PERCENT
        return value if value.is_finite() and Decimal(0) <= value <= _PERCENT else None

    @classmethod
    def _is_company(cls, observation: OperatingDriverObservation) -> bool:
        return (
            observation.segment_id == "company"
            and (observation.scope or "").casefold() != "segment"
        ) or (observation.scope or "").casefold() in {"company", "consolidated", "total"}

    @staticmethod
    def _is_monetary_unit(unit: str) -> bool:
        normalized = unit.casefold()
        if "/" in normalized and not any(
            scale in normalized for scale in ("thousand", "million", "billion")
        ):
            return False
        tokens = set(normalized.replace("/", " ").replace("-", " ").split())
        return bool(tokens & _CURRENCY_TERMS) or normalized in _CURRENCY_TERMS

    @classmethod
    def _references(cls, items) -> tuple[EvidenceReference, ...]:
        return cls._dedupe_references(
            ref
            for item in items
            for ref in (
                getattr(item, "evidence", None),
                getattr(item, "provenance", None)
                if isinstance(getattr(item, "provenance", None), EvidenceReference)
                else None,
                *getattr(item, "source_provenance", ()),
            )
            if ref is not None
        )

    @staticmethod
    def _dedupe_references(references) -> tuple[EvidenceReference, ...]:
        result: list[EvidenceReference] = []
        for reference in references:
            if reference not in result:
                result.append(reference)
        return tuple(result)

    @staticmethod
    def _source_map(candidates, years):
        return {year: item.source for year, item in zip(years, candidates, strict=True) if item is not None}

    @staticmethod
    def _method_map(candidates, years):
        return {year: item.method for year, item in zip(years, candidates, strict=True) if item is not None}

    @staticmethod
    def _confidence_map(candidates, years):
        return {year: item.confidence for year, item in zip(years, candidates, strict=True) if item is not None}

    @staticmethod
    def _provenance_items(*values) -> tuple[Any, ...]:
        flattened: list[Any] = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, (tuple, list)):
                flattened.extend(item for item in value if item is not None)
            else:
                flattened.append(value)
        result: list[Any] = []
        for item in flattened:
            if item not in result:
                result.append(item)
        return tuple(result)

    @staticmethod
    def _provenance_map(candidates, years):
        return {year: item.provenance for year, item in zip(years, candidates, strict=True) if item is not None and item.provenance is not None}

    @staticmethod
    def _audit_map(candidates, years):
        return {
            year: tuple(dict.fromkeys((f"selected_source={item.source}", *item.audit)))
            for year, item in zip(years, candidates, strict=True)
            if item is not None
        }

    @classmethod
    def _diagnostics(cls, metric, years, values, candidates, warnings, *, history=None):
        supported = tuple(year for year, value in zip(years, values, strict=True) if value is not None)
        selected = tuple(item for item in candidates if item is not None)
        historical_rates = history.rates if history is not None else ()
        normalized_rate = history.normalized if history is not None else None
        dispersion = history.dispersion if history is not None else None
        provenance = cls._provenance_items(
            *(
                value
                for item in selected
                for value in (
                    item.provenance,
                    *item.provenance_chain,
                    *item.references,
                )
            ),
            *(history.provenance if history is not None else ()),
            *(history.references if history is not None else ()),
        )
        return OperatingEconomicsMetricDiagnostics(
            metric=metric,
            coverage=Decimal(len(supported)) / Decimal(len(years)) if years else None,
            supported_years=supported,
            confidence=_worst_confidence(tuple(item.confidence for item in selected)) if selected else "low",
            completeness=Decimal(len(supported)) / Decimal(len(years)) if years else None,
            normalized_ratio=normalized_rate if metric == _RATE else None,
            historical_rates=historical_rates,
            normalized_rate=normalized_rate,
            dispersion=dispersion,
            historical_years=history.years if history is not None else (),
            provenance=provenance,
            warnings=tuple(dict.fromkeys(warnings)),
        )


OperatingTaxForecastService = OperatingTaxNopatEngine


__all__ = ["OperatingTaxNopatEngine", "OperatingTaxForecastService"]
