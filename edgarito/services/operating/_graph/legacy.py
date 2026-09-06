"""Compatibility conversion from the legacy operating forecast contracts.

The legacy forecast service is intentionally not used here.  This module only
translates its definitions and evidence into the provider-neutral economic
graph contracts.  Keeping the translation one-way makes it possible to adopt
the graph evaluator without changing the established operating service.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from edgarito.schemas.operating import (
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
    canonical_operating_segment_id,
)
from edgarito.schemas.operating_graph import (
    EconomicComponentRole,
    EconomicModel,
    EconomicNode,
    EconomicNodeType,
    EconomicObservation,
    EconomicProvenance,
    EconomicRelationship,
    EconomicRelationshipType,
    EconomicSourceEdge,
    EconomicUnitKind,
)
from edgarito.schemas.operating_normalization import normalize_operating_unit
from edgarito.services.operating._graph.validation import rate_unit_scale

_FORMULA_METRICS: dict[OperatingArchetype, tuple[str, ...]] = {
    OperatingArchetype.VOLUME_PRICE: ("volume", "price"),
    OperatingArchetype.SUBSCRIBERS_ARPU: ("subscribers", "arpu"),
    OperatingArchetype.CAPACITY_UTILIZATION_PRICE: (
        "capacity",
        "utilization",
        "price",
    ),
    OperatingArchetype.TRANSACTIONS_TAKE_RATE: ("transactions", "take_rate"),
    OperatingArchetype.BACKLOG_CONVERSION: ("backlog", "conversion_rate"),
    OperatingArchetype.STORE_COUNT_SALES_PER_STORE: (
        "store_count",
        "sales_per_store",
    ),
    OperatingArchetype.GENERIC_SEGMENT_GROWTH: ("growth",),
}

_METRIC_ALIASES = {
    "average_revenue_per_user": "arpu",
    "subscriber_count": "subscribers",
    "transaction_count": "transactions",
    "stores": "store_count",
    "sales_per_location": "sales_per_store",
    "conversion": "conversion_rate",
    "growth_rate": "growth",
    "segment_growth": "growth",
    "previous_revenue": "previous_revenue",
    "prior_revenue": "previous_revenue",
    "segment_revenue": "revenue",
}

_RATE_UNITS = frozenset({"%", "percent", "percentage", "percentage_points", "pp"})


@dataclass(frozen=True)
class LegacyEconomicGraphFragments:
    """The graph plus stable IDs useful to callers bridging old contracts."""

    model: EconomicModel
    revenue_root_by_segment: Mapping[str, str]
    formula_node_by_definition: Mapping[str, str]
    input_node_by_key: Mapping[tuple[str, str], str]

    @property
    def economic_model(self) -> EconomicModel:
        return self.model


def _text(value: object) -> str:
    return str(getattr(value, "value", value)).strip()


def _metric(value: str) -> str:
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    return _METRIC_ALIASES.get(normalized, normalized)


def _node_id(segment_id: str, metric: str) -> str:
    return f"operating:{segment_id}:input:{metric}"


def _segment_revenue_id(segment_id: str) -> str:
    return f"operating:{segment_id}:revenue"


def _formula_id(segment_id: str, driver_id: str, ordinal: int) -> str:
    # The ordinal is needed only for malformed-but-accepted legacy input with
    # repeated driver IDs.  Normal definitions retain the readable ID.
    suffix = f":{ordinal}" if ordinal else ""
    return f"operating:{segment_id}:formula:{driver_id}{suffix}"


def _formula_metrics(definition: OperatingDriverDefinition) -> tuple[str, ...]:
    try:
        return _FORMULA_METRICS[OperatingArchetype(definition.archetype)]
    except (KeyError, ValueError) as error:  # pragma: no cover - schema guards this
        raise ValueError(
            f"Unsupported operating archetype: {definition.archetype}"
        ) from error


def _definition_metric(definition: OperatingDriverDefinition, metric: str) -> str:
    wanted = _metric(metric)
    for declared in definition.input_metrics:
        if _metric(declared) == wanted:
            return declared
    return metric


def _fraction(value: Decimal, unit: str) -> Decimal:
    return value * rate_unit_scale(unit)


def _is_supported_rate_unit(unit: str) -> bool:
    try:
        rate_unit_scale(unit)
    except ValueError:
        return False
    return True


def _provenance(value: Any) -> EconomicProvenance | str | None:
    """Normalize all legacy provenance shapes into the graph provenance type."""

    if value is None:
        return None
    if isinstance(value, EconomicProvenance):
        return value
    if isinstance(value, str):
        return value
    data: Mapping[str, Any]
    if hasattr(value, "model_dump"):
        data = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        data = value
    else:
        data = {
            name: getattr(value, name, None)
            for name in (
                "provider",
                "source",
                "origin",
                "reference",
                "accession",
                "methodology",
                "document_name",
                "evidence_ids",
            )
        }
    source = data.get("source") or data.get("provider") or data.get("dataset")
    reference = data.get("reference") or data.get("accession") or data.get("series_id")
    methodology = data.get("methodology") or data.get("document_name")
    origin = data.get("origin")
    evidence_ids = data.get("evidence_ids") or ()
    if isinstance(evidence_ids, str):
        evidence_ids = (evidence_ids,)
    return EconomicProvenance(
        source=_optional_text(source),
        origin=_optional_text(origin),
        reference=_optional_text(reference),
        available_on=data.get("available_on") or data.get("filing_date"),
        evidence_ids=tuple(str(item) for item in evidence_ids),
        methodology=_optional_text(methodology),
    )


def _template_provenance(reference: str, methodology: str) -> EconomicProvenance:
    """Identify registry code, rather than a filing, as formula provenance."""

    return EconomicProvenance(
        source="legacy_operating_archetype",
        origin="deterministic_template",
        reference=reference,
        methodology=methodology,
    )


def _structure_provenance(
    value: Any,
    *,
    reference: str,
    methodology: str,
) -> EconomicProvenance:
    """Keep dated evidence, but never label an undated legacy template as evidence."""

    provenance = _provenance(value)
    if isinstance(provenance, EconomicProvenance) and (
        provenance.available_on is not None
        or provenance.origin == "deterministic_template"
    ):
        return provenance
    return _template_provenance(reference, methodology)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    return text or None


def _currency(unit: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit.upper()
    normalized, _ = normalize_operating_unit(unit)
    first = normalized.split("/", 1)[0]
    candidate = first.upper()
    if (
        len(candidate) == 3
        and candidate.isalpha()
        and candidate
        in {
            "USD",
            "EUR",
            "GBP",
            "JPY",
            "CNY",
            "CAD",
            "AUD",
            "CHF",
        }
    ):
        return candidate
    return None


def _is_rate(unit: str, metric: str) -> bool:
    normalized, _ = normalize_operating_unit(unit, metric)
    return (
        normalized in {"ratio", "percent", "percentage", "percentage_points", "pp"}
        or unit.strip().casefold() in _RATE_UNITS
        or metric in {"growth", "utilization", "take_rate", "conversion_rate"}
    )


def _is_monetary(unit: str, metric: str) -> bool:
    normalized, _ = normalize_operating_unit(unit, metric)
    first = normalized.split("/", 1)[0]
    return (
        first in {"currency", "monetary"}
        or _currency(unit) is not None
        or metric in {"price", "arpu", "sales_per_store", "backlog"}
    )


def _unit_shape(
    unit: str,
    metric: str,
    *,
    currency: str | None,
    denominator: str | None = None,
) -> tuple[str, EconomicUnitKind, str | None, str | None]:
    """Return canonical unit text, dimension, currency, and denominator."""

    normalized, _ = normalize_operating_unit(unit, metric)
    if _is_rate(unit, metric):
        return normalized, EconomicUnitKind.RATE, None, None
    if "/" in normalized or metric in {
        "price",
        "arpu",
        "sales_per_store",
    }:
        resolved_currency = currency or _currency(unit)
        resolved_currency = resolved_currency.upper() if resolved_currency else None
        denominator = denominator or (
            normalized.split("/", 1)[1] if "/" in normalized else None
        )
        denominator = denominator or {
            "price": "units",
            "arpu": "users",
            "sales_per_store": "locations",
        }.get(metric, "units")
        if resolved_currency:
            return (
                f"{resolved_currency}/{denominator}",
                EconomicUnitKind.MONETARY_PER_UNIT,
                resolved_currency,
                denominator,
            )
        return (
            f"currency/{denominator}",
            EconomicUnitKind.MONETARY_PER_UNIT,
            None,
            denominator,
        )
    if _is_monetary(unit, metric):
        resolved_currency = currency or _currency(unit)
        resolved_currency = resolved_currency.upper() if resolved_currency else None
        return (
            resolved_currency or "currency",
            EconomicUnitKind.MONETARY,
            resolved_currency,
            None,
        )
    return normalized or "units", EconomicUnitKind.COUNT, None, None


def _segment_scope(segment: OperatingSegment, company_id: str) -> tuple[str, str]:
    if segment.scope == "consolidated":
        return "consolidated", company_id
    return segment.scope, segment.segment_id


def _edge(node_id: str, *, sign: int = 1, lag: int = 0) -> EconomicSourceEdge:
    return EconomicSourceEdge(
        node_id=node_id,
        sign=Decimal(sign),
        weight=Decimal(1),
        fiscal_lag=lag,
    )


def _node_confidence(values: Iterable[str]) -> str:
    rank = {"low": 0, "medium": 1, "high": 2}
    candidates = tuple(str(value).casefold() for value in values if value)
    if not candidates:
        return "medium"
    return min(candidates, key=lambda value: rank.get(value, 1))


def _coerce_segments(
    segments: Iterable[OperatingSegment | Mapping[str, Any]],
    definitions: Sequence[OperatingDriverDefinition],
) -> tuple[OperatingSegment, ...]:
    by_id: dict[str, OperatingSegment] = {}
    for raw in segments:
        segment = (
            raw
            if isinstance(raw, OperatingSegment)
            else OperatingSegment.model_validate(raw)
        )
        by_id[segment.segment_id] = segment
    for definition in definitions:
        segment_id = (
            canonical_operating_segment_id(definition.segment_id)
            or definition.segment_id
        )
        if segment_id not in by_id:
            by_id[segment_id] = OperatingSegment(
                segment_id=segment_id,
                name=segment_id.replace("_", " ").title(),
                source="first_party_filing",
                confidence="medium",
            )
    return tuple(by_id.values())


def build_legacy_economic_graph(
    segments: Iterable[OperatingSegment | Mapping[str, Any]],
    definitions: Iterable[OperatingDriverDefinition | Mapping[str, Any]],
    observations: Iterable[OperatingDriverObservation | Mapping[str, Any]] = (),
    *,
    company_id: str = "company",
    revenue_root: str | None = None,
    business_roots: Iterable[str] | None = None,
    fiscal_period: str = "FY",
    historical_revenue: Mapping[Any, Any] | None = None,
) -> LegacyEconomicGraphFragments:
    """Build graph fragments for all seven legacy operating archetypes.

    Input observations are normalized once per segment/metric node.  Formula
    outputs are always collected by an explicit ``ADD`` relationship, including
    the one-definition case.  This mirrors the legacy service's additive
    definition behavior while retaining the graph's one-producer invariant.
    """

    normalized_definitions = tuple(
        item
        if isinstance(item, OperatingDriverDefinition)
        else OperatingDriverDefinition.model_validate(item)
        for item in definitions
    )
    normalized_definitions = tuple(
        sorted(
            normalized_definitions, key=lambda item: (item.segment_id, item.driver_id)
        )
    )
    normalized_segments = _coerce_segments(segments, normalized_definitions)
    normalized_observations = tuple(
        item
        if isinstance(item, OperatingDriverObservation)
        else OperatingDriverObservation.model_validate(item)
        for item in observations
    )
    if historical_revenue:
        historical_items: list[OperatingDriverObservation] = []
        segment_ids = tuple(segment.segment_id for segment in normalized_segments)
        for raw_key, raw_value in historical_revenue.items():
            if isinstance(raw_value, Mapping):
                segment_id = canonical_operating_segment_id(str(raw_key)) or str(
                    raw_key
                )
                values = raw_value
            else:
                if len(segment_ids) != 1:
                    raise ValueError(
                        "Year-keyed historical_revenue requires one operating segment"
                    )
                segment_id = segment_ids[0]
                values = {raw_key: raw_value}
            segment = next(
                (item for item in normalized_segments if item.segment_id == segment_id),
                None,
            )
            if segment is None:
                continue
            for raw_year, amount in values.items():
                if any(
                    item.segment_id == segment_id
                    and _metric(item.driver_id) == "revenue"
                    and item.fiscal_year == int(raw_year)
                    for item in normalized_observations
                ):
                    continue
                historical_items.append(
                    OperatingDriverObservation(
                        segment_id=segment_id,
                        driver_id="revenue",
                        fiscal_year=int(raw_year),
                        value=Decimal(str(amount)),
                        unit=segment.currency or "currency",
                        currency=segment.currency,
                        origin="reported",
                        confidence="medium",
                        method="legacy_historical_revenue",
                    )
                )
        historical_by_segment: dict[str, dict[int, Decimal]] = {}
        for item in historical_items:
            historical_by_segment.setdefault(item.segment_id, {})[item.fiscal_year] = (
                item.normalized_value
            )
        for definition in normalized_definitions:
            if definition.archetype != OperatingArchetype.GENERIC_SEGMENT_GROWTH:
                continue
            values = historical_by_segment.get(definition.segment_id, {})
            for previous_year in sorted(values):
                prior_year = previous_year - 1
                if prior_year not in values:
                    continue
                growth_year = previous_year + 1
                if any(
                    item.segment_id == definition.segment_id
                    and _metric(item.driver_id) == "growth"
                    and item.fiscal_year == growth_year
                    for item in normalized_observations
                ):
                    continue
                growth = values[previous_year] / values[prior_year] - Decimal(1)
                growth_metric = _definition_metric(definition, "growth")
                growth_unit = definition.units.get(growth_metric, "ratio")
                try:
                    growth /= rate_unit_scale(growth_unit)
                except ValueError:
                    pass
                historical_items.append(
                    OperatingDriverObservation(
                        segment_id=definition.segment_id,
                        driver_id=growth_metric,
                        fiscal_year=growth_year,
                        value=growth,
                        unit=growth_unit,
                        origin="derived",
                        confidence="medium",
                        method="generic_segment_growth_from_reported_revenue",
                    )
                )
        normalized_observations = (*normalized_observations, *historical_items)
    period = str(fiscal_period).strip().upper()
    if period not in {"FY", "FQ", "YTD", "LTM"}:
        raise ValueError("fiscal_period is not supported")

    definitions_by_segment: dict[str, list[OperatingDriverDefinition]] = {
        segment.segment_id: [] for segment in normalized_segments
    }
    for definition in normalized_definitions:
        if definition.output_metric == "revenue":
            definitions_by_segment.setdefault(definition.segment_id, []).append(
                definition
            )

    # Gather observations before creating nodes so shared inputs get one stable
    # canonical shape and one observation per fiscal cell.
    observation_by_key: dict[
        tuple[str, str, int, str, str | None], list[OperatingDriverObservation]
    ] = {}
    for observation in normalized_observations:
        observation_by_key.setdefault(
            (
                observation.segment_id,
                _metric(observation.driver_id),
                observation.fiscal_year,
                observation.fiscal_period,
                observation.period_key,
            ),
            [],
        ).append(observation)

    nodes: dict[str, EconomicNode] = {}
    relationships: list[EconomicRelationship] = []
    graph_observations: dict[tuple[Any, ...], EconomicObservation] = {}
    revenue_roots: dict[str, str] = {}
    formula_nodes: dict[str, str] = {}
    input_nodes: dict[tuple[str, str], str] = {}
    rate_units: dict[tuple[str, str], str] = {}
    dimensional_rate_nodes: set[str] = set()

    def ensure_input_node(
        segment: OperatingSegment,
        metric: str,
        definition: OperatingDriverDefinition | None = None,
    ) -> str:
        canonical_metric = _metric(metric)
        key = (segment.segment_id, canonical_metric)
        existing = input_nodes.get(key)
        if existing is not None:
            return existing
        matching = [
            item
            for item in normalized_observations
            if item.segment_id == segment.segment_id
            and _metric(item.driver_id) == canonical_metric
        ]
        declared_unit = (
            definition.units.get(_definition_metric(definition, metric), "units")
            if definition is not None
            else (matching[0].unit if matching else "units")
        )
        source_currency = next(
            (item.currency for item in matching if item.currency),
            segment.currency,
        )
        unit, kind, currency, denominator = _unit_shape(
            declared_unit,
            canonical_metric,
            currency=source_currency,
        )
        scope, scope_id = _segment_scope(segment, company_id)
        node_id = _node_id(segment.segment_id, canonical_metric)
        nodes[node_id] = EconomicNode(
            node_id=node_id,
            node_type=EconomicNodeType.INPUT,
            scope=scope,
            scope_id=scope_id,
            metric=canonical_metric,
            unit=unit,
            currency=currency,
            unit_kind=kind,
            denominator_unit=denominator,
            provenance=_structure_provenance(
                next((item.provenance or item.evidence for item in matching), None),
                reference=node_id,
                methodology="legacy operating input registry",
            ),
            confidence=_node_confidence([item.confidence for item in matching]),
            component_role=EconomicComponentRole.STANDARD,
            forecast_assumption_allowed=True,
        )
        input_nodes[key] = node_id
        return node_id

    def ensure_revenue_node(segment: OperatingSegment, *, has_formula: bool) -> str:
        segment_id = segment.segment_id
        existing = revenue_roots.get(segment_id)
        if existing is not None:
            return existing
        matching = [
            item
            for item in normalized_observations
            if item.segment_id == segment_id and _metric(item.driver_id) == "revenue"
        ]
        currency = next(
            (item.currency for item in matching if item.currency), segment.currency
        )
        unit = currency or "currency"
        scope, scope_id = _segment_scope(segment, company_id)
        node_id = _segment_revenue_id(segment_id)
        nodes[node_id] = EconomicNode(
            node_id=node_id,
            node_type=EconomicNodeType.AGGREGATE
            if has_formula
            else EconomicNodeType.INPUT,
            scope=scope,
            scope_id=scope_id,
            metric="revenue",
            unit=unit,
            currency=currency,
            unit_kind=EconomicUnitKind.MONETARY,
            provenance=_structure_provenance(
                segment.evidence or segment.source,
                reference=node_id,
                methodology="legacy operating revenue registry",
            ),
            confidence=segment.confidence,
            component_role=EconomicComponentRole.ADDITIVE,
            forecast_assumption_allowed=True,
        )
        revenue_roots[segment_id] = node_id
        return node_id

    for segment in normalized_segments:
        definitions_for_segment = definitions_by_segment.get(segment.segment_id, [])
        revenue_node_id = ensure_revenue_node(
            segment, has_formula=bool(definitions_for_segment)
        )
        formula_edges: list[EconomicSourceEdge] = []
        for ordinal, definition in enumerate(definitions_for_segment):
            archetype = OperatingArchetype(definition.archetype)
            formula_metrics = _formula_metrics(definition)
            for formula_metric in formula_metrics:
                if formula_metric in {
                    "growth",
                    "utilization",
                    "take_rate",
                    "conversion_rate",
                }:
                    declared_metric = _definition_metric(definition, formula_metric)
                    rate_units[(segment.segment_id, formula_metric)] = (
                        definition.units.get(declared_metric, "unit")
                    )
            formula_node_id = _formula_id(
                segment.segment_id, definition.driver_id, ordinal
            )
            formula_nodes[definition.driver_id] = formula_node_id
            # The output shape is determined by the archetype's monetary input.
            monetary_metric = next(
                (
                    metric
                    for metric in formula_metrics
                    if metric in {"price", "arpu", "sales_per_store", "backlog"}
                ),
                None,
            )
            source_observations = [
                observation
                for observation in normalized_observations
                if observation.segment_id == segment.segment_id
                and _metric(observation.driver_id)
                in {_metric(item) for item in definition.input_metrics}
            ]
            output_currency = next(
                (
                    item.currency
                    for item in source_observations
                    if item.currency
                    and (
                        monetary_metric is None
                        or _metric(item.driver_id) == monetary_metric
                    )
                ),
                segment.currency,
            )
            if archetype == OperatingArchetype.TRANSACTIONS_TAKE_RATE:
                transaction_observation = next(
                    (
                        item
                        for item in source_observations
                        if _metric(item.driver_id) == "transactions"
                    ),
                    None,
                )
                if transaction_observation is not None and _is_monetary(
                    transaction_observation.unit, "transactions"
                ):
                    output_currency = (
                        output_currency or transaction_observation.currency
                    )
            output_unit = output_currency or "currency"
            scope, scope_id = _segment_scope(segment, company_id)
            nodes[formula_node_id] = EconomicNode(
                node_id=formula_node_id,
                node_type=EconomicNodeType.DERIVED,
                scope=scope,
                scope_id=scope_id,
                metric=definition.driver_id,
                unit=output_unit,
                currency=output_currency,
                unit_kind=EconomicUnitKind.MONETARY,
                provenance=_structure_provenance(
                    definition.evidence or definition.source,
                    reference=formula_node_id,
                    methodology="legacy operating formula registry",
                ),
                confidence=_node_confidence(
                    (
                        definition.confidence,
                        segment.confidence,
                        *(item.confidence for item in source_observations),
                    )
                ),
                component_role=EconomicComponentRole.ADDITIVE,
                forecast_assumption_allowed=True,
            )
            sources: list[EconomicSourceEdge] = []
            for formula_metric in formula_metrics:
                declared_metric = _definition_metric(definition, formula_metric)
                source_id = ensure_input_node(segment, declared_metric, definition)
                if formula_metric == "growth":
                    # A generic growth formula uses the preceding aggregate
                    # revenue.  The base node is represented separately so
                    # historical revenue observations can seed the first year
                    # without mutating the aggregate's reported provenance.
                    base_metric = "previous_revenue"
                    base_id = ensure_input_node(segment, base_metric, definition)
                    base_node = nodes[base_id]
                    if base_node.unit_kind != EconomicUnitKind.MONETARY:
                        nodes[base_id] = base_node.model_copy(
                            update={
                                "unit": output_unit,
                                "unit_kind": EconomicUnitKind.MONETARY,
                                "currency": output_currency,
                                "denominator_unit": None,
                            }
                        )
                    source_id = base_id
                sources.append(_edge(source_id))
            if archetype == OperatingArchetype.GENERIC_SEGMENT_GROWTH:
                # The loop's source is the growth input; the first source is
                # the lagged aggregate base.
                growth_id = ensure_input_node(
                    segment,
                    _definition_metric(definition, "growth"),
                    definition,
                )
                sources = [sources[0], _edge(growth_id)]
            elif archetype == OperatingArchetype.CAPACITY_UTILIZATION_PRICE:
                capacity_id, utilization_id, price_id = (
                    edge.node_id for edge in sources
                )
                intermediate_id = f"{formula_node_id}:utilized_capacity"
                capacity_node = nodes[capacity_id]
                nodes[intermediate_id] = EconomicNode(
                    node_id=intermediate_id,
                    node_type=EconomicNodeType.DERIVED,
                    scope=scope,
                    scope_id=scope_id,
                    metric="utilized_capacity",
                    unit=capacity_node.unit,
                    currency=capacity_node.currency,
                    unit_kind=capacity_node.unit_kind,
                    provenance=_structure_provenance(
                        nodes[formula_node_id].provenance,
                        reference=intermediate_id,
                        methodology="legacy operating formula registry",
                    ),
                    confidence=nodes[formula_node_id].confidence,
                    forecast_assumption_allowed=True,
                )
                relationships.append(
                    EconomicRelationship(
                        target=intermediate_id,
                        relationship_type=EconomicRelationshipType.MULTIPLY,
                        sources=(_edge(capacity_id), _edge(utilization_id)),
                        relationship_id=f"{intermediate_id}:multiply",
                        provenance=_template_provenance(
                            f"{intermediate_id}:multiply",
                            "legacy archetype formula registry",
                        ),
                        confidence=definition.confidence,
                        fiscal_period=period,
                    )
                )
                sources = [_edge(intermediate_id), _edge(price_id)]
            elif archetype in {
                OperatingArchetype.SUBSCRIBERS_ARPU,
                OperatingArchetype.STORE_COUNT_SALES_PER_STORE,
            }:
                count_id, price_id = (edge.node_id for edge in sources)
                count_node = nodes[count_id]
                price_node = nodes[price_id]
                if price_node.unit_kind == EconomicUnitKind.MONETARY_PER_UNIT:
                    nodes[price_id] = price_node.model_copy(
                        update={
                            "unit": f"{output_currency or 'currency'}/{count_node.unit}",
                            "denominator_unit": count_node.unit,
                        }
                    )
            elif archetype == OperatingArchetype.TRANSACTIONS_TAKE_RATE:
                transaction_id, rate_id = (edge.node_id for edge in sources)
                transaction_node = nodes[transaction_id]
                rate_unit = rate_units.get(
                    (segment.segment_id, "take_rate"), nodes[rate_id].unit
                )
                if (
                    transaction_node.unit_kind == EconomicUnitKind.COUNT
                    and _is_supported_rate_unit(rate_unit)
                ):
                    # The core dimensional contract models a count multiplied
                    # by a monetary-per-transaction rate.  The legacy take-rate
                    # arithmetic is unchanged; only the graph dimension is
                    # made explicit for validation.
                    rate_node = nodes[rate_id]
                    nodes[rate_id] = rate_node.model_copy(
                        update={
                            "unit": f"{output_currency or 'currency'}/{transaction_node.unit}",
                            "unit_kind": EconomicUnitKind.MONETARY_PER_UNIT,
                            "currency": output_currency,
                            "denominator_unit": transaction_node.unit,
                        }
                    )
                    dimensional_rate_nodes.add(rate_id)
                sources = [_edge(transaction_id), _edge(rate_id)]
            relationships.append(
                EconomicRelationship(
                    target=formula_node_id,
                    relationship_type=(
                        EconomicRelationshipType.GROWTH
                        if archetype == OperatingArchetype.GENERIC_SEGMENT_GROWTH
                        else EconomicRelationshipType.MULTIPLY
                    ),
                    sources=tuple(sources),
                    relationship_id=f"{formula_node_id}:formula:{definition.formula_id}",
                    provenance=_template_provenance(
                        f"{formula_node_id}:formula:{definition.formula_id}",
                        "legacy archetype formula registry",
                    ),
                    confidence=definition.confidence,
                    fiscal_period=period,
                )
            )
            formula_edges.append(_edge(formula_node_id, sign=1))
        if formula_edges:
            relationships.append(
                EconomicRelationship(
                    target=revenue_node_id,
                    relationship_type=EconomicRelationshipType.ADD,
                    sources=tuple(formula_edges),
                    relationship_id=f"{revenue_node_id}:definitions",
                    provenance=_template_provenance(
                        f"{revenue_node_id}:definitions",
                        "legacy operating definition registry",
                    ),
                    confidence=segment.confidence,
                    fiscal_period=period,
                )
            )

    # Attach observations after all canonical nodes exist.  Revenue observations
    # stay on the aggregate root; generic growth also receives a historical-base
    # observation with the explicit derived-parameter origin used by the graph
    # evaluator for deterministic lag seeding.
    for observation in normalized_observations:
        segment = next(
            (
                item
                for item in normalized_segments
                if item.segment_id == observation.segment_id
            ),
            None,
        )
        if segment is None:
            continue
        canonical_metric = _metric(observation.driver_id)
        if canonical_metric == "revenue":
            target_id = revenue_roots.get(segment.segment_id)
        else:
            target_id = input_nodes.get((segment.segment_id, canonical_metric))
        if target_id is None or target_id not in nodes:
            continue
        node = nodes[target_id]
        value = observation.normalized_value
        if canonical_metric in {
            "growth",
            "utilization",
            "take_rate",
            "conversion_rate",
        } and target_id in dimensional_rate_nodes:
            value = _fraction(
                value,
                rate_units.get(
                    (segment.segment_id, canonical_metric), observation.unit
                ),
            )
        key = (
            target_id,
            observation.fiscal_year,
            observation.fiscal_period,
            value,
            node.unit,
            node.currency,
            observation.origin,
            str(_provenance(observation.provenance or observation.evidence)),
        )
        graph_origin = observation.origin
        graph_provenance = _provenance(observation.provenance or observation.evidence)
        if (
            canonical_metric == "revenue"
            and target_id in {item.target for item in relationships}
            and observation.origin
            in {"reported", "first_party_observation", "extracted_evidence"}
        ):
            # A legacy direct revenue fact is an authoritative historical seed
            # for a graph aggregate.  The graph evaluator reserves this origin
            # for exactly that kind of derived historical parameter; retain the
            # original origin in canonical provenance for the reverse adapter.
            graph_origin = "derived_historical_parameter"
            graph_provenance = EconomicProvenance(
                origin=observation.origin,
                source=(
                    graph_provenance.source
                    if isinstance(graph_provenance, EconomicProvenance)
                    else None
                ),
                reference=(
                    graph_provenance.reference
                    if isinstance(graph_provenance, EconomicProvenance)
                    else None
                ),
                methodology=(
                    graph_provenance.methodology
                    if isinstance(graph_provenance, EconomicProvenance)
                    else None
                ),
                available_on=(
                    graph_provenance.available_on
                    if isinstance(graph_provenance, EconomicProvenance)
                    else None
                ),
            )
        graph_observations[key] = EconomicObservation(
            node_id=target_id,
            fiscal_year=observation.fiscal_year,
            fiscal_period=observation.fiscal_period,
            value=value,
            unit=node.unit,
            currency=node.currency,
            origin=graph_origin,
            provenance=graph_provenance,
            available_on=getattr(graph_provenance, "available_on", None),
            scope=node.scope,
            scope_id=node.scope_id,
        )
        if canonical_metric == "revenue":
            for definition in definitions_by_segment.get(segment.segment_id, ()):
                if definition.archetype != OperatingArchetype.GENERIC_SEGMENT_GROWTH:
                    continue
                base_id = input_nodes.get((segment.segment_id, "previous_revenue"))
                if base_id is None:
                    continue
                base_node = nodes[base_id]
                base_key = (
                    base_id,
                    observation.fiscal_year + 1,
                    observation.fiscal_period,
                    value,
                    base_node.unit,
                    base_node.currency,
                    "derived_historical_parameter",
                    str(_provenance(observation.provenance or observation.evidence)),
                )
                graph_observations[base_key] = EconomicObservation(
                    node_id=base_id,
                    fiscal_year=observation.fiscal_year + 1,
                    fiscal_period=observation.fiscal_period,
                    value=value,
                    unit=base_node.unit,
                    currency=base_node.currency,
                    origin="derived_historical_parameter",
                    provenance=_provenance(
                        observation.provenance or observation.evidence
                    ),
                    available_on=getattr(
                        _provenance(observation.provenance or observation.evidence),
                        "available_on",
                        None,
                    ),
                    scope=base_node.scope,
                    scope_id=base_node.scope_id,
                )

    # The legacy generic-growth evaluator carries the selected prior aggregate
    # forward between years.  The graph contract evaluates all requested cells,
    # so materialize those prior values as historical-parameter observations
    # instead of creating an unbounded positive-lag self-cycle at the first
    # requested year.  The GROWTH relationship itself remains explicit.
    for segment in normalized_segments:
        generic_definitions = tuple(
            item
            for item in definitions_by_segment.get(segment.segment_id, ())
            if item.archetype == OperatingArchetype.GENERIC_SEGMENT_GROWTH
        )
        if not generic_definitions:
            continue
        base_id = input_nodes.get((segment.segment_id, "previous_revenue"))
        growth_id = input_nodes.get((segment.segment_id, "growth"))
        if base_id is None or growth_id is None:
            continue
        base_values = {
            item.fiscal_year: item.value
            for item in graph_observations.values()
            if item.node_id == base_id and item.origin == "derived_historical_parameter"
        }
        growth_values = {
            item.fiscal_year: item.value
            for item in graph_observations.values()
            if item.node_id == growth_id
        }
        for year in sorted(growth_values):
            base = base_values.get(year)
            if base is None:
                continue
            # Each generic definition sees the same selected previous segment
            # revenue, and the old service adds each resulting formula value.
            current = sum(
                (
                    base * (Decimal(1) + growth_values[year])
                    for _ in generic_definitions
                ),
                Decimal(0),
            )
            next_year = year + 1
            if next_year in base_values:
                continue
            base_values[next_year] = current
            node = nodes[base_id]
            graph_observations[
                (
                    base_id,
                    next_year,
                    period,
                    current,
                    node.unit,
                    node.currency,
                    "derived_historical_parameter",
                    "generic_segment_growth",
                )
            ] = EconomicObservation(
                node_id=base_id,
                fiscal_year=next_year,
                fiscal_period=period,
                value=current,
                unit=node.unit,
                currency=node.currency,
                origin="derived_historical_parameter",
                provenance=None,
                scope=node.scope,
                scope_id=node.scope_id,
            )

    true_business_roots = tuple(
        dict.fromkeys(
            business_roots
            if business_roots is not None
            else (revenue_roots[segment.segment_id] for segment in normalized_segments)
        )
    )
    true_business_roots = tuple(root for root in true_business_roots if root in nodes)
    company_only = tuple(
        root for root in true_business_roots if nodes[root].scope == "consolidated"
    )
    if revenue_root is None:
        if len(company_only) == 1:
            revenue_root = company_only[0]
        else:
            revenue_root = f"operating:{company_id}:revenue"
    if revenue_root not in nodes:
        if not true_business_roots:
            raise ValueError("Cannot infer a revenue root without graph business roots")
        first = nodes[true_business_roots[0]]
        nodes[revenue_root] = EconomicNode(
            node_id=revenue_root,
            node_type=EconomicNodeType.AGGREGATE,
            scope="consolidated",
            scope_id=company_id,
            metric="revenue",
            unit=first.unit,
            currency=first.currency,
            unit_kind=EconomicUnitKind.MONETARY,
            provenance=_template_provenance(
                revenue_root,
                "legacy operating company aggregate registry",
            ),
            confidence=_node_confidence(
                nodes[root].confidence for root in true_business_roots
            ),
            component_role=EconomicComponentRole.ADDITIVE,
            forecast_assumption_allowed=True,
        )
    if len(true_business_roots) > 1 and not any(
        relationship.target == revenue_root for relationship in relationships
    ):
        root = nodes[revenue_root]
        relationships.append(
            EconomicRelationship(
                target=revenue_root,
                relationship_type=EconomicRelationshipType.ADD,
                sources=tuple(_edge(item, sign=1) for item in true_business_roots),
                relationship_id=f"{revenue_root}:business-roots",
                provenance=_template_provenance(
                    f"{revenue_root}:business-roots",
                    "legacy business-root aggregation",
                ),
                confidence=root.confidence,
                fiscal_period=period,
            )
        )
    if revenue_root not in true_business_roots and len(true_business_roots) == 1:
        # A single segment is still a business root; an explicit company root
        # is allowed to be an identity aggregate without creating a fake segment.
        root = nodes[revenue_root]
        relationships.append(
            EconomicRelationship(
                target=revenue_root,
                relationship_type=EconomicRelationshipType.ADD,
                sources=(_edge(true_business_roots[0]),),
                relationship_id=f"{revenue_root}:business-root",
                provenance=_template_provenance(
                    f"{revenue_root}:business-root",
                    "legacy business-root aggregation",
                ),
                confidence=root.confidence,
                fiscal_period=period,
            )
        )

    model = EconomicModel(
        nodes=tuple(nodes.values()),
        relationships=tuple(relationships),
        observations=tuple(graph_observations.values()),
        revenue_root=revenue_root,
        business_roots=true_business_roots,
        fiscal_period=period,
    )
    return LegacyEconomicGraphFragments(
        model=model,
        revenue_root_by_segment=dict(revenue_roots),
        formula_node_by_definition=dict(formula_nodes),
        input_node_by_key=dict(input_nodes),
    )


def legacy_to_economic_model(*args, **kwargs) -> EconomicModel:
    """Return only the converted model (convenience compatibility API)."""

    return build_legacy_economic_graph(*args, **kwargs).model


def adapt_legacy_operating_inputs(*args, **kwargs) -> LegacyEconomicGraphFragments:
    return build_legacy_economic_graph(*args, **kwargs)


class LegacyEconomicGraphAdapter:
    """Small state-free adapter facade for dependency-injection callers."""

    def __init__(self, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs

    def adapt(self, *args, **kwargs) -> LegacyEconomicGraphFragments:
        return build_legacy_economic_graph(
            *(args or self._args), **(kwargs or self._kwargs)
        )

    build = adapt

    def __call__(self, *args, **kwargs) -> LegacyEconomicGraphFragments:
        return self.adapt(*args, **kwargs)


LegacyOperatingGraphAdapter = LegacyEconomicGraphAdapter
build_legacy_economic_model = legacy_to_economic_model


__all__ = [
    "LegacyEconomicGraphAdapter",
    "LegacyOperatingGraphAdapter",
    "LegacyEconomicGraphFragments",
    "adapt_legacy_operating_inputs",
    "build_legacy_economic_graph",
    "build_legacy_economic_model",
    "legacy_to_economic_model",
]
