"""Deterministic first-party operating history assembly."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any

from edgarito.config.operating import OPERATING_EXTRACTION
from edgarito.schemas.operating import (
    EvidenceReference,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
    operating_periods_compatible,
    operating_units_compatible,
)
from edgarito.schemas.operating_history import (
    OperatingEvidenceGap,
    OperatingHistoryAudit,
    OperatingTimeSeries,
)

_REVENUE_METRICS = OPERATING_EXTRACTION.history_revenue_metric_ids
_VOLUME_METRICS = OPERATING_EXTRACTION.history_volume_metric_ids
_COUNT_METRICS = OPERATING_EXTRACTION.history_count_metric_ids
_PERIOD_RANK = {"FY": 4, "LTM": 3, "YTD": 2, "FQ": 1}
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_EXTREME_DISCONTINUITY_FACTOR = Decimal("10")


def _source_references(
    *observations: OperatingDriverObservation,
) -> tuple[EvidenceReference, ...]:
    references: dict[tuple[str | None, str | None, str | None], EvidenceReference] = {}
    for observation in observations:
        for reference in (*observation.source_provenance, observation.evidence):
            if reference is not None:
                references[
                    (
                        reference.accession,
                        reference.document_name,
                        reference.source_text_hash,
                    )
                ] = reference
    return tuple(
        references[key]
        for key in sorted(
            references, key=lambda item: tuple(value or "" for value in item)
        )
    )


def _source_document(observation: OperatingDriverObservation) -> str | None:
    reference = observation.evidence or (
        observation.source_provenance[0] if observation.source_provenance else None
    )
    if reference is None:
        return None
    return f"{reference.accession or ''}:{reference.document_name or ''}"


def _economic_units_compatible(revenue_unit: str, volume_unit: str) -> bool:
    """Reject clearly malformed cross-document pairs without changing formulas."""
    revenue = revenue_unit.casefold()
    volume = volume_unit.casefold()
    return any(
        token in revenue for token in ("usd", "eur", "gbp", "currency", "dollar")
    ) and not any(
        token in volume for token in ("usd", "eur", "gbp", "currency", "dollar")
    )


def _derived_scope(items: Iterable[OperatingDriverObservation]) -> str | None:
    scopes = tuple(dict.fromkeys(item.scope for item in items if item.scope))
    return scopes[0] if scopes else None


def _derived_scope_evidence(
    items: Iterable[OperatingDriverObservation],
) -> str | None:
    evidence = tuple(
        dict.fromkeys(
            item.scope_evidence
            or item.basis
            or (item.evidence.supporting_text if item.evidence else None)
            for item in items
            if item.scope_evidence
            or item.basis
            or (item.evidence and item.evidence.supporting_text)
        )
    )
    return "; ".join(evidence) if evidence else None


def _scope_join_reason(
    left: OperatingDriverObservation,
    right: OperatingDriverObservation,
) -> str | None:
    """Reject joins that combine broad facts with component facts."""

    if left.is_component != right.is_component:
        return "derived metric rejected: scope mismatch"
    if left.is_component and (
        left.scope != right.scope or left.scope_evidence != right.scope_evidence
    ):
        return "derived metric rejected: scope mismatch"
    scopes = tuple(dict.fromkeys(item.scope for item in (left, right) if item.scope))
    if len(scopes) > 1:
        return "derived metric rejected: scope mismatch"
    return None


def _aggregate_exhaustive_components(
    items: list[OperatingDriverObservation],
) -> OperatingDriverObservation | None:
    if len(items) < 2 or not all(
        item.is_component and item.exhaustive for item in items
    ):
        return None
    first = items[0]
    if any(item.scope != first.scope for item in items[1:]):
        return None
    return OperatingDriverObservation(
        segment_id=first.segment_id,
        driver_id=first.driver_id,
        fiscal_year=first.fiscal_year,
        fiscal_period=first.fiscal_period,
        period_key=first.period_key,
        value=sum((item.normalized_value for item in items), Decimal(0)),
        unit=first.unit,
        original_unit=first.original_unit or first.unit,
        origin="derived",
        confidence="medium",
        method="derived_from_exhaustive_components",
        evidence=first.evidence,
        source_provenance=_source_references(*items),
        basis=_derived_scope_evidence(items),
        scope=first.scope,
        scope_evidence=_derived_scope_evidence(items),
        is_total=True,
        exhaustive=True,
    )


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
            tuple[str, str, int, str, str | None, str | None, str | None, bool, bool],
            OperatingDriverObservation,
        ] = {}
        join_attempts = 0
        join_accepted = 0
        join_rejections: dict[str, int] = defaultdict(int)
        join_diagnostics: list[str] = []

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
        component_totals = self._derive_exhaustive_totals(accepted)
        derivation_input = accepted + component_totals
        (
            derived,
            derived_warnings,
            join_attempts,
            join_accepted,
            join_rejections,
            join_diagnostics,
        ) = self._derive_revenue_and_volume(derivation_input)
        annualized, annualized_warnings = self._annualize_quarterly_observations(
            accepted
        )
        derived.extend(annualized)
        derived_warnings.extend(annualized_warnings)
        derived.extend(component_totals)
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
                    item.is_total,
                    item.origin != "derived",
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
        reconstruction_candidates, reconstruction_rejections = (
            self._reconstruction_pair_diagnostics(all_observations)
        )
        if reconstruction_candidates and not history_pairs:
            derived_warnings.append(
                "Compatible operating reconstruction candidates exist only in "
                "quarterly/YTD periods; no FY/LTM revenue history was created"
            )
        failures.extend(period_failures)
        gaps = self.derive_gaps(
            accepted,
            segments=normalized_segments,
            definitions=normalized_definitions,
        )
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
            derived_totals=len(component_totals),
            scope_mismatch_rejections=(
                join_rejections.get("derived metric rejected: scope mismatch", 0)
                + join_rejections.get("scope_mismatch", 0)
            ),
            historical_revenue_pairs=history_pairs,
            warnings=warnings,
            joins_attempted=join_attempts,
            joins_accepted=join_accepted,
            joins_rejected=sum(join_rejections.values()),
            join_rejections_by_reason=dict(sorted(join_rejections.items())),
            join_diagnostics=tuple(join_diagnostics),
            source_document_count=len(
                {
                    (ref.accession, ref.document_name)
                    for item in all_observations
                    for ref in (*item.source_provenance, item.evidence)
                    if ref is not None
                }
            ),
            reconstruction_candidates=reconstruction_candidates,
            reconstruction_rejections=reconstruction_rejections,
            gaps_detected=gaps,
            gaps_unresolved=gaps,
            gap_diagnostics=tuple(join_diagnostics),
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

    def derive_gaps(
        self,
        observations: Iterable[OperatingDriverObservation | Mapping[str, Any]],
        *,
        segments: Iterable[OperatingSegment | Mapping[str, Any]] = (),
        definitions: Iterable[OperatingDriverDefinition | Mapping[str, Any]] = (),
    ) -> tuple[OperatingEvidenceGap, ...]:
        """Describe missing inputs and same-period revenue/volume mismatches."""

        source = tuple(self._canonical_observation(item) for item in observations)
        normalized_definitions = self._unique_definitions(definitions)
        gaps: dict[tuple[Any, ...], OperatingEvidenceGap] = {}

        def add(
            segment_id: str,
            metric: str,
            fiscal_year: int | None,
            fiscal_period: str = "FY",
            period_key: str | None = None,
            reason: str = "missing_required_input",
        ) -> None:
            gap = OperatingEvidenceGap(
                segment_id=segment_id,
                metric=self._canonical_metric(metric),
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                period_key=period_key,
                reason=reason,
            )
            previous = gaps.get(gap.key)
            if previous is None or (
                previous.reason == "missing_required_input"
                and reason == "revenue_volume_mismatch"
            ):
                gaps[gap.key] = gap

        # Only reported/extracted source observations are used to detect a gap;
        # a derived value must not hide the missing fact it was inferred from.
        source = tuple(item for item in source if item.origin != "derived")
        by_segment_year: dict[tuple[str, int], list[OperatingDriverObservation]] = (
            defaultdict(list)
        )
        by_period: dict[tuple[str, int, str, str | None], set[str]] = defaultdict(set)
        for item in source:
            by_segment_year[(item.segment_id, item.fiscal_year)].append(item)
            metric = self._canonical_metric(item.driver_id)
            if metric in {"revenue", "volume"}:
                by_period[
                    (
                        item.segment_id,
                        item.fiscal_year,
                        item.fiscal_period,
                        item.period_key,
                    )
                ].add(metric)

        for definition in normalized_definitions:
            years = sorted(
                year
                for segment_id, year in by_segment_year
                if segment_id == definition.segment_id
            )
            if not years:
                for metric in definition.required_inputs:
                    add(definition.segment_id, metric, None)
                continue
            for year in years:
                periods = {
                    (item.fiscal_period, item.period_key)
                    for item in by_segment_year[(definition.segment_id, year)]
                }
                for metric in definition.required_inputs:
                    canonical_metric = self._canonical_metric(metric)
                    if not any(
                        self._canonical_metric(item.driver_id) == canonical_metric
                        for item in by_segment_year[(definition.segment_id, year)]
                    ):
                        for fiscal_period, period_key in sorted(periods):
                            add(
                                definition.segment_id,
                                canonical_metric,
                                year,
                                fiscal_period,
                                period_key,
                            )

        for (segment_id, year, period, period_key), metrics in sorted(
            by_period.items()
        ):
            if "revenue" in metrics and "volume" not in metrics:
                add(
                    segment_id,
                    "volume",
                    year,
                    period,
                    period_key,
                    "revenue_volume_mismatch",
                )
            elif "volume" in metrics and "revenue" not in metrics:
                add(
                    segment_id,
                    "revenue",
                    year,
                    period,
                    period_key,
                    "revenue_volume_mismatch",
                )
        for (segment_id, year), items in sorted(by_segment_year.items()):
            revenues = [
                item
                for item in items
                if self._canonical_metric(item.driver_id) == "revenue"
            ]
            volumes = [
                item
                for item in items
                if self._canonical_metric(item.driver_id) == "volume"
            ]
            if (
                revenues
                and volumes
                and not any(
                    operating_periods_compatible(
                        revenue.fiscal_period,
                        volume.fiscal_period,
                        revenue.period_key,
                        volume.period_key,
                    )
                    for revenue in revenues
                    for volume in volumes
                )
            ):
                for revenue in revenues:
                    add(
                        segment_id,
                        "volume",
                        year,
                        revenue.fiscal_period,
                        revenue.period_key,
                        "revenue_volume_mismatch",
                    )
                for volume in volumes:
                    add(
                        segment_id,
                        "revenue",
                        year,
                        volume.fiscal_period,
                        volume.period_key,
                        "revenue_volume_mismatch",
                    )
        result: list[OperatingEvidenceGap] = []
        for gap in gaps.values():
            source_items = (
                by_segment_year.get((gap.segment_id, gap.fiscal_year), ())
                if gap.fiscal_year is not None
                else ()
            )
            result.append(
                gap.model_copy(
                    update={
                        "source_documents": tuple(
                            dict.fromkeys(
                                document
                                for item in source_items
                                for document in (_source_document(item),)
                                if document
                            )
                        ),
                        "diagnostics": (
                            "revenue/volume mismatch"
                            if gap.reason == "revenue_volume_mismatch"
                            else "required input is absent from assembled history",
                        ),
                    }
                )
            )
        return tuple(sorted(result, key=lambda item: (item.key, item.reason)))

    @staticmethod
    def _key(
        observation: OperatingDriverObservation,
    ) -> tuple[str, str, int, str, str | None, str | None, str | None, bool, bool]:
        return (
            observation.segment_id,
            observation.driver_id.casefold(),
            observation.fiscal_year,
            observation.fiscal_period,
            observation.period_key.casefold() if observation.period_key else None,
            observation.scope,
            observation.scope_evidence,
            observation.is_total,
            observation.is_component,
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
    def _reconstruction_pair_diagnostics(
        observations: Iterable[OperatingDriverObservation],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        grouped: dict[
            tuple[str, int, str, str | None], dict[str, OperatingDriverObservation]
        ] = defaultdict(dict)
        for item in observations:
            metric = OperatingHistoryAssembler._canonical_metric(item.driver_id)
            if metric in {"revenue", "volume", "price"}:
                grouped[
                    (
                        item.segment_id,
                        item.fiscal_year,
                        item.fiscal_period,
                        item.period_key,
                    )
                ][metric] = item
        candidates: list[str] = []
        rejections: list[str] = []
        for (segment, year, period, period_key), values in sorted(grouped.items()):
            revenue = values.get("revenue")
            volume = values.get("volume")
            label = (
                f"{segment}/FY{year}/{period}{('/' + period_key) if period_key else ''}"
            )
            if revenue is None or volume is None:
                if revenue is not None or volume is not None:
                    rejections.append(
                        f"{label}: missing {'volume' if revenue is not None else 'revenue'}"
                    )
                continue
            if not _economic_units_compatible(revenue.unit, volume.unit):
                rejections.append(
                    f"{label}: incompatible units ({revenue.unit}, {volume.unit})"
                )
                continue
            candidates.append(
                f"{label}: revenue={revenue.normalized_value} {revenue.unit}; "
                f"volume={volume.normalized_value} {volume.unit}; "
                f"implied_price={revenue.normalized_value / volume.normalized_value if volume.normalized_value else 'unavailable'}; "
                f"revenue_source={_source_document(revenue) or 'unknown'}; "
                f"volume_source={_source_document(volume) or 'unknown'}"
            )
        return tuple(candidates), tuple(rejections)

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
            tuple[str, int, str, str | None],
            dict[str, list[OperatingDriverObservation]],
        ] = defaultdict(dict)
        for item in observations:
            if item.origin == "management_guidance":
                continue
            key = (
                item.segment_id,
                item.fiscal_year,
                item.fiscal_period,
                item.period_key,
            )
            grouped[key].setdefault(item.driver_id.casefold(), []).append(item)
        derived = []
        warnings = []
        join_attempts = 0
        join_accepted = 0
        join_rejections: dict[str, int] = defaultdict(int)
        join_diagnostics: list[str] = []
        derived_history: dict[tuple[str, int, str], list[Decimal]] = defaultdict(list)
        candidates: dict[
            tuple[str, int], dict[str, list[OperatingDriverObservation]]
        ] = defaultdict(lambda: {"revenue": [], "volume": []})
        for item in observations:
            metric = OperatingHistoryAssembler._canonical_metric(item.driver_id)
            if metric in {"revenue", "volume"}:
                candidates[(item.segment_id, item.fiscal_year)][metric].append(item)
        # Inspect all same-year candidates before grouping by period. This makes
        # an FY/FQ or 6M/Q mismatch visible instead of silently looking missing.
        for (segment, year), pair in sorted(candidates.items()):
            for revenue in pair["revenue"]:
                for volume in pair["volume"]:
                    revenue_doc = _source_document(revenue)
                    volume_doc = _source_document(volume)
                    if not revenue_doc or not volume_doc or revenue_doc == volume_doc:
                        continue
                    join_attempts += 1
                    reason = None
                    if not _economic_units_compatible(revenue.unit, volume.unit):
                        reason = "incompatible_units"
                    elif not operating_periods_compatible(
                        revenue.fiscal_period,
                        volume.fiscal_period,
                        revenue.period_key,
                        volume.period_key,
                    ):
                        reason = "incompatible_period"
                    elif _scope_join_reason(revenue, volume):
                        reason = "scope_mismatch"
                    if reason:
                        join_rejections[reason] += 1
                        join_diagnostics.append(
                            f"{segment}/FY{year}: rejected cross-document join ({reason})"
                        )
        for (segment, year, period, period_key), values in grouped.items():
            revenue = OperatingHistoryAssembler._select_metric(values, _REVENUE_METRICS)
            volume = OperatingHistoryAssembler._select_metric(values, _VOLUME_METRICS)
            price = OperatingHistoryAssembler._select_metric(
                values,
                {"price", "asp", "average_selling_price"},
                reported_only=True,
            )
            subscribers = OperatingHistoryAssembler._select_metric(
                values, _COUNT_METRICS
            )
            if revenue is None and volume is not None and price is not None:
                reason = _scope_join_reason(volume, price)
                if reason:
                    join_rejections[reason] += 1
                    join_diagnostics.append(f"{segment}/FY{year}/{period}: {reason}")
                    continue
                reason = OperatingHistoryAssembler._derived_pair_invalid_reason(
                    "revenue",
                    volume.normalized_value * price.normalized_value,
                    (volume, price),
                    observations,
                    derived_history,
                )
                if reason:
                    join_rejections[reason] += 1
                    join_diagnostics.append(f"{segment}/FY{year}/{period}: {reason}")
                    continue
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
                        source_provenance=_source_references(volume, price),
                        basis=_derived_scope_evidence((volume, price)),
                        scope=_derived_scope((volume, price)),
                        scope_evidence=_derived_scope_evidence((volume, price)),
                    )
                )
                derived_history[(segment, year, "revenue")].append(
                    derived[-1].normalized_value
                )
                continue
            if (
                revenue is not None
                and volume is None
                and price is not None
                and price.normalized_value != 0
            ):
                reason = _scope_join_reason(revenue, price)
                if reason:
                    join_rejections[reason] += 1
                    join_diagnostics.append(f"{segment}/FY{year}/{period}: {reason}")
                    continue
                derived_value = revenue.normalized_value / price.normalized_value
                reason = OperatingHistoryAssembler._derived_pair_invalid_reason(
                    "volume",
                    derived_value,
                    (revenue, price),
                    observations,
                    derived_history,
                )
                if reason:
                    join_rejections[reason] += 1
                    join_diagnostics.append(f"{segment}/FY{year}/{period}: {reason}")
                    continue
                derived.append(
                    OperatingDriverObservation(
                        segment_id=segment,
                        driver_id="volume",
                        fiscal_year=year,
                        fiscal_period=period,
                        period_key=period_key,
                        value=derived_value,
                        unit="units",
                        original_unit="units",
                        origin="derived",
                        confidence="medium",
                        method="derived_from_reported_revenue_and_price",
                        evidence=revenue.evidence or price.evidence,
                        source_provenance=_source_references(revenue, price),
                        basis=_derived_scope_evidence((revenue, price)),
                        scope=_derived_scope((revenue, price)),
                        scope_evidence=_derived_scope_evidence((revenue, price)),
                    )
                )
                derived_history[(segment, year, "volume")].append(
                    derived[-1].normalized_value
                )
                continue
            if (
                revenue is not None
                and subscribers is not None
                and subscribers.normalized_value != 0
            ):
                if operating_periods_compatible(
                    revenue.fiscal_period,
                    subscribers.fiscal_period,
                    revenue.period_key,
                    subscribers.period_key,
                ):
                    reason = _scope_join_reason(revenue, subscribers)
                    if reason:
                        join_rejections[reason] += 1
                        join_diagnostics.append(
                            f"{segment}/FY{year}/{period}: {reason}"
                        )
                        continue
                    revenue_doc = _source_document(revenue)
                    subscribers_doc = _source_document(subscribers)
                    derived_value = (
                        revenue.normalized_value / subscribers.normalized_value
                    )
                    reason = OperatingHistoryAssembler._derived_pair_invalid_reason(
                        "arpu",
                        derived_value,
                        (revenue, subscribers),
                        observations,
                        derived_history,
                    )
                    if reason:
                        join_rejections[reason] += 1
                        join_diagnostics.append(
                            f"{segment}/FY{year}/{period}: {reason}"
                        )
                        continue
                    derived.append(
                        OperatingDriverObservation(
                            segment_id=segment,
                            driver_id="implied_arpu",
                            fiscal_year=year,
                            fiscal_period=period,
                            period_key=period_key,
                            value=derived_value,
                            unit=f"{revenue.unit}/{subscribers.unit}",
                            original_unit=f"{revenue.original_unit or revenue.unit}/{subscribers.original_unit or subscribers.unit}",
                            origin="derived",
                            confidence="medium",
                            method=(
                                "derived_from_reported_revenue_and_subscribers_cross_document"
                                if revenue_doc
                                and subscribers_doc
                                and revenue_doc != subscribers_doc
                                else "derived_from_reported_revenue_and_subscribers"
                            ),
                            evidence=revenue.evidence or subscribers.evidence,
                            source_provenance=_source_references(revenue, subscribers),
                            basis=_derived_scope_evidence((revenue, subscribers)),
                            scope=_derived_scope((revenue, subscribers)),
                            scope_evidence=_derived_scope_evidence(
                                (revenue, subscribers)
                            ),
                        )
                    )
                    derived_history[(segment, year, "arpu")].append(
                        derived[-1].normalized_value
                    )
            if revenue is None or volume is None or volume.normalized_value == 0:
                if (revenue is None) != (volume is None):
                    join_rejections["missing_side"] += 1
                    join_diagnostics.append(
                        f"{segment}/FY{year}/{period}: rejected join (missing_side)"
                    )
                continue
            revenue_doc = _source_document(revenue)
            volume_doc = _source_document(volume)
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
            reason = _scope_join_reason(revenue, volume)
            if reason:
                join_rejections[reason] += 1
                join_diagnostics.append(f"{segment}/FY{year}/{period}: {reason}")
                continue
            derived_value = revenue.normalized_value / volume.normalized_value
            reason = OperatingHistoryAssembler._derived_pair_invalid_reason(
                "price",
                derived_value,
                (revenue, volume),
                observations,
                derived_history,
            )
            if reason:
                join_rejections[reason] += 1
                join_diagnostics.append(f"{segment}/FY{year}/{period}: {reason}")
                continue
            derived.append(
                OperatingDriverObservation(
                    segment_id=segment,
                    driver_id="implied_price",
                    fiscal_year=year,
                    fiscal_period=period,
                    period_key=period_key,
                    value=derived_value,
                    unit=f"{revenue.unit}/{volume.unit}",
                    original_unit=f"{revenue.original_unit or revenue.unit}/{volume.original_unit or volume.unit}",
                    origin="derived",
                    confidence="medium",
                    method=(
                        "derived_from_reported_revenue_and_volume_cross_document"
                        if revenue_doc and volume_doc and revenue_doc != volume_doc
                        else "derived_from_reported_revenue_and_volume"
                    ),
                    evidence=revenue.evidence or volume.evidence,
                    source_provenance=_source_references(revenue, volume),
                    basis=_derived_scope_evidence((revenue, volume)),
                    scope=_derived_scope((revenue, volume)),
                    scope_evidence=_derived_scope_evidence((revenue, volume)),
                )
            )
            derived_history[(segment, year, "price")].append(
                derived[-1].normalized_value
            )
            if revenue_doc and volume_doc and revenue_doc != volume_doc:
                join_accepted += 1
                join_diagnostics.append(
                    f"{segment}/FY{year}/{period}: accepted cross-document join "
                    f"revenue={revenue_doc} volume={volume_doc} method=derived_from_reported_revenue_and_volume"
                )
        return (
            derived,
            warnings,
            join_attempts,
            join_accepted,
            join_rejections,
            join_diagnostics,
        )

    @staticmethod
    def _select_metric(
        values: Mapping[
            str, OperatingDriverObservation | list[OperatingDriverObservation]
        ],
        metric_names: set[str] | frozenset[str],
        *,
        reported_only: bool = False,
    ) -> OperatingDriverObservation | None:
        candidates = []
        for key, value in values.items():
            if key not in metric_names:
                continue
            items = value if isinstance(value, list) else [value]
            candidates.extend(
                item for item in items if not reported_only or item.origin == "reported"
            )
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                item.is_total,
                not item.is_component,
                item.confidence == "high",
                item.evidence is not None,
            ),
        )

    @classmethod
    def _derive_exhaustive_totals(
        cls, observations: Iterable[OperatingDriverObservation]
    ) -> list[OperatingDriverObservation]:
        grouped: dict[
            tuple[str, str, int, str, str | None], list[OperatingDriverObservation]
        ] = defaultdict(list)
        for item in observations:
            if item.origin == "management_guidance" or not item.is_component:
                continue
            metric = cls._canonical_metric(item.driver_id)
            if metric not in {"revenue", "volume", "subscribers", "transactions"}:
                continue
            grouped[
                (
                    item.segment_id,
                    metric,
                    item.fiscal_year,
                    item.fiscal_period,
                    item.period_key,
                )
            ].append(item)
        totals: list[OperatingDriverObservation] = []
        for items in grouped.values():
            if any(item.is_total for item in items):
                continue
            aggregate = _aggregate_exhaustive_components(items)
            if aggregate is not None:
                totals.append(aggregate)
        return totals

    @staticmethod
    def _derived_pair_invalid_reason(
        output_metric: str,
        value: Decimal,
        inputs: tuple[OperatingDriverObservation, ...],
        observations: Iterable[OperatingDriverObservation],
        derived_history: Mapping[tuple[str, int, str], list[Decimal]],
    ) -> str | None:
        scope_reason = _scope_join_reason(*inputs[:2])
        if scope_reason:
            return scope_reason
        segment = inputs[0].segment_id
        year = inputs[0].fiscal_year
        references = [
            item.normalized_value
            for item in observations
            if item.segment_id == segment
            and item.fiscal_year == year
            and OperatingHistoryAssembler._canonical_metric(item.driver_id)
            == output_metric
            and item.origin != "derived"
            and item.normalized_value != 0
        ]
        references.extend(derived_history.get((segment, year, output_metric), ()))
        if (
            value != 0
            and references
            and any(
                value.copy_abs() > reference.copy_abs() * _EXTREME_DISCONTINUITY_FACTOR
                or value.copy_abs() * _EXTREME_DISCONTINUITY_FACTOR
                < reference.copy_abs()
                for reference in references
            )
        ):
            return "derived metric rejected: extreme order-of-magnitude discontinuity"
        return None

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
                        source_provenance=_source_references(*quarters.values()),
                        basis="; ".join(
                            item.basis for item in quarters.values() if item.basis
                        )
                        or None,
                        scope=_derived_scope(tuple(quarters.values())),
                        scope_evidence=_derived_scope_evidence(
                            tuple(quarters.values())
                        ),
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
