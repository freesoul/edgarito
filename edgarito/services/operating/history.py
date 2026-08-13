"""Deterministic first-party operating history assembly."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any

from edgarito.schemas.operating import (
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
    operating_periods_compatible,
    operating_units_compatible,
)
from edgarito.schemas.operating_history import (
    OperatingHistoryAudit,
    OperatingTimeSeries,
)

_REVENUE_METRICS = frozenset(
    {"revenue", "segment_revenue", "sales", "net_sales", "total_revenue", "total_sales"}
)
_VOLUME_METRICS = frozenset(
    {"volume", "units", "deliveries", "shipments", "production"}
)
_COUNT_METRICS = frozenset(
    {"subscribers", "subscriber_count", "users", "stores", "store_count"}
)
_PERIOD_RANK = {"FY": 4, "LTM": 3, "YTD": 2, "FQ": 1}
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


class OperatingHistoryAssembler:
    """Normalize observations from many filings without changing forecast math."""

    def assemble(
        self,
        observations: Iterable[OperatingDriverObservation | Mapping[str, Any]],
        *,
        segments: Iterable[OperatingSegment | Mapping[str, Any]] = (),
        definitions: Iterable[OperatingDriverDefinition | Mapping[str, Any]] = (),
        company_id: str = "company",
    ) -> OperatingTimeSeries:
        normalized_segments = self._unique_segments(segments)
        normalized_definitions = self._unique_definitions(definitions)
        source = tuple(self._canonical_observation(item) for item in observations)
        normalized_segments = self._ensure_referenced_segments(
            normalized_segments,
            normalized_definitions,
            source,
        )
        accepted: list[OperatingDriverObservation] = []
        failures: list[str] = []
        unit_failures: list[str] = []
        deduplicated = 0
        by_key: dict[
            tuple[str, str, int, str, str | None], OperatingDriverObservation
        ] = {}

        for observation in source:
            key = self._key(observation)
            previous = by_key.get(key)
            if previous is not None:
                deduplicated += 1
                if self._winner(observation, previous) is observation:
                    by_key[key] = observation
                continue
            by_key[key] = observation

        for observation in by_key.values():
            if observation.fiscal_period == "FQ" and not observation.period_key:
                failures.append(
                    f"{observation.segment_id}/{observation.driver_id}/FY"
                    f"{observation.fiscal_year}: FQ period missing quarter key"
                )
                continue
            definition_units = self._definition_units(
                normalized_definitions, observation.segment_id, observation.driver_id
            )
            if definition_units and not any(
                operating_units_compatible(expected, observation.unit)
                for expected in definition_units
            ):
                reason = (
                    f"{observation.segment_id}/{observation.driver_id}/FY"
                    f"{observation.fiscal_year}: unit '{observation.unit}' is not "
                    f"compatible with {sorted(definition_units)}"
                )
                unit_failures.append(reason)
                continue
            accepted.append(observation)

        accepted.sort(key=self._sort_key)
        derived, derived_warnings = self._derive_revenue_and_volume(accepted)
        annualized, annualized_warnings = self._annualize_quarterly_observations(
            accepted
        )
        derived.extend(annualized)
        derived_warnings.extend(annualized_warnings)
        period_failures = self._period_failures(normalized_definitions, accepted)
        all_observations = tuple(accepted + derived)
        revenue_by_segment: dict[str, dict[int, Decimal]] = defaultdict(dict)
        revenue_candidates: dict[tuple[str, int], list[OperatingDriverObservation]] = (
            defaultdict(list)
        )
        company_revenue: dict[int, Decimal] = {}
        for observation in all_observations:
            if not self._is_revenue(observation.driver_id):
                continue
            if observation.fiscal_period not in {"FY", "LTM"}:
                continue
            revenue_candidates[
                (observation.segment_id, observation.fiscal_year)
            ].append(observation)
        for (segment, year), candidates in revenue_candidates.items():
            winner = max(
                candidates,
                key=lambda item: (
                    _PERIOD_RANK.get(item.fiscal_period, 0),
                    _CONFIDENCE_RANK[item.confidence],
                    item.evidence is not None,
                ),
            )
            revenue_by_segment[segment][year] = winner.normalized_value

        # Consolidated/company rows are passed separately from segment rows.  For
        # ordinary segment histories, the engine derives company totals only when
        # all selected non-overlapping segments exist for a year.
        segment_ids = {segment.segment_id for segment in normalized_segments}
        for observation in all_observations:
            if not self._is_revenue(observation.driver_id):
                continue
            if observation.segment_id in {
                "company",
                "consolidated",
                "total",
            } and observation.fiscal_period in {"FY", "LTM"}:
                company_revenue[observation.fiscal_year] = observation.normalized_value
        if not company_revenue and segment_ids:
            years = (
                set.intersection(
                    *(
                        set(revenue_by_segment.get(segment_id, {}))
                        for segment_id in segment_ids
                    )
                )
                if segment_ids
                else set()
            )
            for year in years:
                company_revenue[year] = sum(
                    (
                        revenue_by_segment[segment_id][year]
                        for segment_id in segment_ids
                    ),
                    Decimal(0),
                )

        period_order = ("FY", "FQ", "YTD", "LTM")
        accepted_periods = tuple(
            item
            for item in period_order
            if item in {record.fiscal_period for record in all_observations}
        )
        accepted_metrics = tuple(sorted({item.driver_id for item in all_observations}))
        accepted_pairs = tuple(
            sorted({(item.segment_id, item.driver_id) for item in all_observations})
        )
        missing_pairs = self._missing_pairs(
            normalized_segments, normalized_definitions, all_observations
        )
        history_pairs = tuple(
            f"{segment}/FY{year}"
            for segment, values in sorted(revenue_by_segment.items())
            for year in sorted(values)
        )
        failures.extend(period_failures)
        warnings = tuple(dict.fromkeys((*derived_warnings, *failures, *unit_failures)))
        audit = OperatingHistoryAudit(
            accepted_periods=accepted_periods,
            accepted_metrics=accepted_metrics,
            accepted_pairs=accepted_pairs,
            missing_pairs=missing_pairs,
            period_failures=tuple(failures),
            unit_failures=tuple(unit_failures),
            input_observations=len(source),
            accepted_observations=len(accepted),
            deduplicated_observations=deduplicated,
            derived_observations=len(derived),
            historical_revenue_pairs=history_pairs,
            warnings=warnings,
        )
        return OperatingTimeSeries(
            company_id=company_id,
            segments=normalized_segments,
            definitions=normalized_definitions,
            observations=all_observations,
            company_revenue=dict(sorted(company_revenue.items())),
            historical_revenue={
                segment: dict(sorted(values.items()))
                for segment, values in sorted(revenue_by_segment.items())
            },
            audit=audit,
        )

    build = assemble
    normalize = assemble

    @staticmethod
    def _key(
        observation: OperatingDriverObservation,
    ) -> tuple[str, str, int, str, str | None]:
        return (
            observation.segment_id,
            observation.driver_id.casefold(),
            observation.fiscal_year,
            observation.fiscal_period,
            observation.period_key.casefold() if observation.period_key else None,
        )

    @staticmethod
    def _sort_key(observation: OperatingDriverObservation) -> tuple[Any, ...]:
        return (
            observation.segment_id,
            observation.fiscal_year,
            _PERIOD_RANK.get(observation.fiscal_period, 0),
            observation.period_key or "",
            observation.driver_id.casefold(),
        )

    @staticmethod
    def _winner(left: OperatingDriverObservation, right: OperatingDriverObservation):
        left_rank = OperatingHistoryAssembler._winner_key(left)
        right_rank = OperatingHistoryAssembler._winner_key(right)
        return left if left_rank > right_rank else right

    @staticmethod
    def _winner_key(observation: OperatingDriverObservation) -> tuple[Any, ...]:
        evidence = observation.evidence
        return (
            _CONFIDENCE_RANK[observation.confidence],
            evidence is not None,
            evidence.filing_date.isoformat()
            if evidence and evidence.filing_date
            else "",
            evidence.accession if evidence else "",
            evidence.document_name if evidence else "",
            evidence.source_text_hash if evidence else "",
        )

    @staticmethod
    def _unique_segments(items) -> tuple[OperatingSegment, ...]:
        result: dict[str, OperatingSegment] = {}
        for item in items:
            segment = (
                item
                if isinstance(item, OperatingSegment)
                else OperatingSegment.model_validate(item)
            )
            previous = result.get(segment.segment_id)
            if previous is None:
                result[segment.segment_id] = segment
                continue
            # The same canonical segment can be rediscovered in each filing.
            # Retain the first identity while allowing later evidence to fill
            # a generic display name without creating duplicate IDs.
            if (
                previous.name == previous.segment_id
                and segment.name != segment.segment_id
            ):
                result[segment.segment_id] = previous.model_copy(
                    update={"name": segment.name}
                )
        return tuple(result.values())

    @staticmethod
    def _unique_definitions(items) -> tuple[OperatingDriverDefinition, ...]:
        result: dict[tuple[str, str], OperatingDriverDefinition] = {}
        for item in items:
            definition = (
                item
                if isinstance(item, OperatingDriverDefinition)
                else OperatingDriverDefinition.model_validate(item)
            )
            input_metrics = tuple(
                OperatingHistoryAssembler._canonical_metric(metric)
                for metric in definition.input_metrics
            )
            input_metrics = tuple(dict.fromkeys(input_metrics))
            required_inputs = tuple(
                OperatingHistoryAssembler._canonical_metric(metric)
                for metric in definition.required_inputs
            )
            required_inputs = tuple(dict.fromkeys(required_inputs))
            optional_inputs = tuple(
                dict.fromkeys(
                    OperatingHistoryAssembler._canonical_metric(metric)
                    for metric in definition.optional_inputs
                )
            )
            optional_inputs = tuple(
                metric for metric in optional_inputs if metric not in required_inputs
            )
            if (
                definition.archetype.value == "generic_segment_growth"
                and "revenue" in required_inputs
            ):
                required_inputs = ("growth",)
                input_metrics = tuple(
                    dict.fromkeys(
                        metric for metric in input_metrics if metric != "revenue"
                    )
                )
                if "growth" not in input_metrics:
                    input_metrics = (*input_metrics, "growth")
            definition = OperatingDriverDefinition.model_validate(
                {
                    **definition.model_dump(),
                    "input_metrics": input_metrics,
                    "required_inputs": required_inputs,
                    "optional_inputs": tuple(optional_inputs),
                    "units": {
                        OperatingHistoryAssembler._canonical_metric(metric): unit
                        for metric, unit in definition.units.items()
                    },
                }
            )
            result.setdefault((definition.segment_id, definition.driver_id), definition)
        return tuple(result.values())

    @staticmethod
    def _ensure_referenced_segments(
        segments: tuple[OperatingSegment, ...],
        definitions: tuple[OperatingDriverDefinition, ...],
        observations: tuple[OperatingDriverObservation, ...],
    ) -> tuple[OperatingSegment, ...]:
        result = list(segments)
        known = {segment.segment_id for segment in result}
        referenced = [
            *(definition.segment_id for definition in definitions),
            *(observation.segment_id for observation in observations),
        ]
        for segment_id in dict.fromkeys(referenced):
            if segment_id in known:
                continue
            scope = (
                "consolidated"
                if segment_id in {"company", "consolidated", "total"}
                else "segment"
            )
            result.append(
                OperatingSegment(
                    segment_id=segment_id,
                    name=segment_id.replace("_", " ").title(),
                    scope=scope,
                    source="first_party_filing",
                    confidence="medium",
                )
            )
            known.add(segment_id)
        return tuple(result)

    @staticmethod
    def _canonical_metric(value: str) -> str:
        normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
        if normalized.endswith("_revenue") or normalized.endswith("_revenues"):
            return "revenue"
        if normalized.endswith("_revenue_driver") or normalized.endswith("_sales"):
            return "revenue"
        return {
            "segment_revenue": "revenue",
            "historical_revenue": "revenue",
            "historical_segment_revenue": "revenue",
            "segment_sales": "revenue",
            "total_revenue": "revenue",
            "total_sales": "revenue",
            "sales": "revenue",
            "net_sales": "revenue",
            "deliveries": "volume",
            "shipments": "volume",
            "production": "volume",
            "units": "volume",
            "asp": "price",
            "average_selling_price": "price",
            "implied_price": "price",
            "implied_arpu": "arpu",
            "subscriber_count": "subscribers",
            "users": "subscribers",
            "transaction_count": "transactions",
            "take_rate": "take_rate",
            "conversion": "conversion_rate",
            "stores": "store_count",
            "sales_per_location": "sales_per_store",
            "cash_deliveries": "volume",
            "vehicle_deliveries": "volume",
            "megapack_deployments": "volume",
            "powerwall_deployments": "volume",
            "average_selling_price_per_unit": "price",
            "average_selling_price_per_megapack_unit": "price",
            "megapack_average_selling_price": "price",
        }.get(normalized, normalized)

    @classmethod
    def _canonical_observation(
        cls, item: OperatingDriverObservation | Mapping[str, Any]
    ) -> OperatingDriverObservation:
        observation = (
            item
            if isinstance(item, OperatingDriverObservation)
            else OperatingDriverObservation.model_validate(item)
        )
        data = observation.model_dump()
        data["driver_id"] = cls._canonical_metric(observation.driver_id)
        return OperatingDriverObservation.model_validate(data)

    @staticmethod
    def _definition_units(definitions, segment_id: str, driver_id: str) -> set[str]:
        result: set[str] = set()
        for definition in definitions:
            if definition.segment_id != segment_id:
                continue
            for metric, unit in definition.units.items():
                if metric.casefold().replace("-", "_").replace(
                    " ", "_"
                ) == driver_id.casefold().replace("-", "_").replace(" ", "_"):
                    result.add(unit)
        return result

    @staticmethod
    def _is_revenue(driver_id: str) -> bool:
        return OperatingHistoryAssembler._canonical_metric(driver_id) == "revenue"

    @staticmethod
    def _missing_pairs(segments, definitions, observations) -> tuple[str, ...]:
        observed = {
            (item.segment_id, item.driver_id.casefold()) for item in observations
        }
        missing = []
        for definition in definitions:
            for metric in definition.required_inputs:
                if (definition.segment_id, metric.casefold()) not in observed:
                    missing.append(f"{definition.segment_id}/{metric}")
        return tuple(sorted(set(missing)))

    @staticmethod
    def _period_failures(definitions, observations) -> tuple[str, ...]:
        by_metric: dict[tuple[str, str, int], list[OperatingDriverObservation]] = (
            defaultdict(list)
        )
        for item in observations:
            by_metric[
                (item.segment_id, item.driver_id.casefold(), item.fiscal_year)
            ].append(item)
        failures = []
        for definition in definitions:
            years = {
                year
                for segment, _metric, year in by_metric
                if segment == definition.segment_id
            }
            for year in sorted(years):
                candidates = [
                    by_metric.get((definition.segment_id, metric.casefold(), year), [])
                    for metric in definition.required_inputs
                ]
                if not all(candidates) or len(candidates) < 2:
                    continue
                compatible = any(
                    all(
                        operating_periods_compatible(
                            first.fiscal_period,
                            candidate.fiscal_period,
                            first.period_key,
                            candidate.period_key,
                        )
                        for group in candidates[1:]
                        for candidate in group
                    )
                    for first in candidates[0]
                )
                if not compatible:
                    periods = ", ".join(
                        f"{metric}={','.join(item.fiscal_period for item in group)}"
                        for metric, group in zip(
                            definition.required_inputs, candidates, strict=True
                        )
                    )
                    failures.append(
                        f"{definition.segment_id}/{definition.driver_id}/FY{year}: "
                        f"incompatible periods ({periods})"
                    )
        return tuple(failures)

    @staticmethod
    def _derive_revenue_and_volume(observations):
        grouped: dict[
            tuple[str, int, str, str | None], dict[str, OperatingDriverObservation]
        ] = defaultdict(dict)
        for item in observations:
            if item.origin == "management_guidance":
                continue
            grouped[
                (item.segment_id, item.fiscal_year, item.fiscal_period, item.period_key)
            ][item.driver_id.casefold()] = item
        derived = []
        warnings = []
        for (segment, year, period, period_key), values in grouped.items():
            revenue = next(
                (item for key, item in values.items() if key in _REVENUE_METRICS), None
            )
            volume = next(
                (item for key, item in values.items() if key in _VOLUME_METRICS), None
            )
            price = next(
                (
                    item
                    for key, item in values.items()
                    if key in {"price", "asp", "average_selling_price"}
                    and item.origin == "reported"
                ),
                None,
            )
            if revenue is None and volume is not None and price is not None:
                derived.append(
                    OperatingDriverObservation(
                        segment_id=segment,
                        driver_id="segment_revenue",
                        fiscal_year=year,
                        fiscal_period=period,
                        period_key=period_key,
                        value=volume.normalized_value * price.normalized_value,
                        unit=f"{price.unit}*{volume.unit}",
                        original_unit=f"{price.original_unit or price.unit}*{volume.original_unit or volume.unit}",
                        origin="derived",
                        confidence="medium",
                        method="derived_from_reported_volume_and_price",
                        evidence=price.evidence or volume.evidence,
                    )
                )
                continue
            if (
                revenue is not None
                and volume is None
                and price is not None
                and price.normalized_value != 0
            ):
                derived.append(
                    OperatingDriverObservation(
                        segment_id=segment,
                        driver_id="volume",
                        fiscal_year=year,
                        fiscal_period=period,
                        period_key=period_key,
                        value=revenue.normalized_value / price.normalized_value,
                        unit="units",
                        original_unit="units",
                        origin="derived",
                        confidence="medium",
                        method="derived_from_reported_revenue_and_price",
                        evidence=revenue.evidence or price.evidence,
                    )
                )
                continue
            if revenue is None or volume is None or volume.normalized_value == 0:
                continue
            if not operating_periods_compatible(
                revenue.fiscal_period,
                volume.fiscal_period,
                revenue.period_key,
                volume.period_key,
            ):
                warnings.append(
                    f"{segment}/FY{year}: revenue and volume periods are incompatible"
                )
                continue
            if revenue.normalized_value < 0:
                continue
            derived.append(
                OperatingDriverObservation(
                    segment_id=segment,
                    driver_id="implied_price",
                    fiscal_year=year,
                    fiscal_period=period,
                    period_key=period_key,
                    value=revenue.normalized_value / volume.normalized_value,
                    unit=f"{revenue.unit}/{volume.unit}",
                    original_unit=f"{revenue.original_unit or revenue.unit}/{volume.original_unit or volume.unit}",
                    origin="derived",
                    confidence="medium",
                    method="derived_from_reported_revenue_and_volume",
                    evidence=revenue.evidence or volume.evidence,
                )
            )
        return derived, warnings

    @staticmethod
    def _annualize_quarterly_observations(observations):
        """Build FY/LTM flow observations only from four reported quarters."""

        grouped: dict[tuple[str, str, int], dict[str, OperatingDriverObservation]] = (
            defaultdict(dict)
        )
        for item in observations:
            if item.origin == "management_guidance" or item.fiscal_period != "FQ":
                continue
            if item.period_key not in {"Q1", "Q2", "Q3", "Q4"}:
                continue
            metric = OperatingHistoryAssembler._canonical_metric(item.driver_id)
            if metric not in {"revenue", "volume"}:
                continue
            grouped[(item.segment_id, metric, item.fiscal_year)][item.period_key] = item

        derived: list[OperatingDriverObservation] = []
        for (segment, metric, year), quarters in sorted(grouped.items()):
            if set(quarters) != {"Q1", "Q2", "Q3", "Q4"}:
                continue
            first = quarters["Q1"]
            value = sum(
                (quarters[key].normalized_value for key in ("Q1", "Q2", "Q3", "Q4")),
                Decimal(0),
            )
            for period, method in (
                ("FY", "annual_from_four_reported_quarters"),
                ("LTM", "ltm_from_four_reported_quarters"),
            ):
                derived.append(
                    OperatingDriverObservation(
                        segment_id=segment,
                        driver_id=metric,
                        fiscal_year=year,
                        fiscal_period=period,
                        period_key=period,
                        value=value,
                        unit=first.unit,
                        original_unit=first.original_unit or first.unit,
                        origin="derived",
                        confidence="medium",
                        method=method,
                        evidence=first.evidence,
                    )
                )
        return derived, []


NormalizedOperatingHistoryService = OperatingHistoryAssembler
OperatingTimeSeriesService = OperatingHistoryAssembler


__all__ = [
    "NormalizedOperatingHistoryService",
    "OperatingHistoryAssembler",
    "OperatingTimeSeriesService",
]
