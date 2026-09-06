"""Deterministic evaluation of provider-neutral economic operating graphs."""

from __future__ import annotations

import datetime as _datetime
from collections.abc import Iterable
from decimal import Decimal

from edgarito.schemas.operating_graph import (
    DependencyAudit,
    EconomicEvaluationResult,
    EconomicModel,
    EconomicNode,
    EconomicNodeType,
    EconomicObservation,
    EconomicRelationship,
    EconomicRelationshipType,
    EconomicUnitKind,
    EconomicValue,
    GraphDiagnostic,
    GraphDiagnostics,
    UnresolvedLeafRequirement,
)
from edgarito.services.operating._graph.point_in_time import (
    observation_as_of_issue,
    provenance_as_of_issue,
)
from edgarito.services.operating._graph.validation import (
    EconomicGraphValidator,
    rate_unit_scale,
)

_REPORTED_ORIGINS = {
    "reported",
    "first_party_observation",
    "extracted_evidence",
    "historical_reported",
}
_HISTORICAL_PARAMETER_ORIGIN = "derived_historical_parameter"


class EconomicGraphEvaluator:
    """Evaluate every requested node-period exactly once per run."""

    def __init__(self, validator: EconomicGraphValidator | None = None) -> None:
        self.validator = validator or EconomicGraphValidator()

    def evaluate(
        self,
        model: EconomicModel,
        target_years: int | Iterable[int],
        *,
        as_of: _datetime.date | _datetime.datetime | None = None,
        fiscal_period: str | None = None,
    ) -> EconomicEvaluationResult:
        report = self.validator.validate(model)
        years = _normalize_years(target_years)
        period = (fiscal_period or model.fiscal_period).strip().upper()
        if period not in {"FY", "FQ", "YTD", "LTM"}:
            raise ValueError("fiscal_period is not supported")
        as_of_date = _as_date(as_of)

        node_by_id = {node.node_id: node for node in model.nodes}
        producer_by_target = {
            relationship.target: relationship for relationship in model.relationships
        }
        observations_by_key: dict[tuple[str, int, str], list[EconomicObservation]] = {}
        for observation in model.observations:
            observations_by_key.setdefault(
                (
                    observation.node_id,
                    observation.fiscal_year,
                    observation.fiscal_period,
                ),
                [],
            ).append(observation)

        runtime_diagnostics: list[GraphDiagnostic] = [item for item in report.warnings]
        diagnostic_keys: set[tuple[str, str | None, int | None]] = {
            (item.code, item.node_id, item.fiscal_year) for item in runtime_diagnostics
        }

        def diagnostic(
            code: str,
            message: str,
            *,
            node_id: str | None = None,
            fiscal_year: int | None = None,
            severity: str = "warning",
        ) -> None:
            key = (code, node_id, fiscal_year)
            if key in diagnostic_keys:
                return
            diagnostic_keys.add(key)
            runtime_diagnostics.append(
                GraphDiagnostic(
                    code=code,
                    message=message,
                    severity=severity,
                    node_id=node_id,
                    fiscal_year=fiscal_year,
                )
            )

        memo: dict[tuple[str, int, str], EconomicValue] = {}
        active: set[tuple[str, int, str]] = set()

        def select_observation(
            node: EconomicNode,
            year: int,
            requested_period: str,
            *,
            origins: set[str] | frozenset[str] | None = None,
        ) -> EconomicObservation | None:
            candidates = observations_by_key.get(
                (node.node_id, year, requested_period), []
            )
            if not candidates:
                # A period mismatch is diagnostically different from a missing
                # leaf.  Do not make a Q1 fact look like an annual fact.
                period_candidates = [
                    observation
                    for (
                        node_id,
                        observation_year,
                        _period,
                    ), values in observations_by_key.items()
                    if node_id == node.node_id and observation_year == year
                    for observation in values
                ]
                if period_candidates:
                    diagnostic(
                        "period_mismatch",
                        f"No {requested_period} observation for {node.node_id}; "
                        "available observation has another fiscal period",
                        node_id=node.node_id,
                        fiscal_year=year,
                    )
                return None

            usable: list[EconomicObservation] = []
            for observation in candidates:
                if origins is not None and observation.origin not in origins:
                    continue
                if observation.unit != node.unit:
                    diagnostic(
                        "unit_mismatch",
                        f"Observation unit {observation.unit!r} does not match "
                        f"node unit {node.unit!r}",
                        node_id=node.node_id,
                        fiscal_year=year,
                    )
                    continue
                if observation.currency != node.currency:
                    diagnostic(
                        "currency_mismatch",
                        f"Observation currency does not match node {node.node_id}",
                        node_id=node.node_id,
                        fiscal_year=year,
                    )
                    continue
                availability_issue = observation_as_of_issue(
                    observation, as_of_date, frozenset(_REPORTED_ORIGINS)
                )
                if availability_issue is not None:
                    code, message = availability_issue
                    diagnostic(
                        code,
                        f"Observation for {node.node_id}: {message}",
                        node_id=node.node_id,
                        fiscal_year=year,
                    )
                    continue
                usable.append(observation)
            if not usable:
                return None
            # Latest available evidence wins; the provenance string provides a
            # stable tie breaker when two facts have the same availability date.
            return max(
                usable,
                key=lambda item: (
                    item.available_on or _datetime.date.min,
                    _provenance_label(item.provenance),
                ),
            )

        def observation_value(
            node: EconomicNode,
            observation: EconomicObservation,
            year: int,
            requested_period: str,
        ) -> EconomicValue:
            """Convert a selected observation into an auditable graph cell."""

            return EconomicValue(
                node_id=node.node_id,
                fiscal_year=year,
                fiscal_period=requested_period,
                value=observation.value,
                unit=node.unit,
                currency=node.currency,
                available=True,
                origin=observation.origin,
                provenance=observation.provenance or node.provenance,
                provenance_chain=_unique_text(
                    (
                        _provenance_label(observation.provenance),
                        _provenance_label(node.provenance),
                    )
                ),
                dependency_chain=(node.node_id,),
                leaf_nodes=(node.node_id,),
            )

        def observation_is_authoritative(
            node: EconomicNode,
            observation: EconomicObservation,
            relationship: EconomicRelationship | None,
        ) -> bool:
            if observation.origin == _HISTORICAL_PARAMETER_ORIGIN:
                return True
            if node.node_type == EconomicNodeType.INPUT:
                return True
            if node.node_type == EconomicNodeType.COMPONENT and relationship is None:
                return True
            if (
                node.node_type == EconomicNodeType.COMPONENT
                and relationship is not None
                and relationship.relationship_type == EconomicRelationshipType.RESIDUAL
                and observation.origin in _REPORTED_ORIGINS
            ):
                # An explicit component observation is the permitted future
                # exception to residual reconstruction.
                return True
            if not node.forecast_assumption_allowed:
                return False
            # A reported derived/aggregate value is a reconstruction target, not
            # a forecast override.  An explicit assumption remains an override.
            return observation.origin not in _REPORTED_ORIGINS

        def unavailable(
            node: EconomicNode,
            year: int,
            reason: str,
            *,
            children: Iterable[EconomicValue] = (),
        ) -> EconomicValue:
            child_values = tuple(children)
            leaves: list[str] = []
            chain: list[str] = [node.node_id]
            provenance_chain: list[str] = [_provenance_label(node.provenance)]
            for child in child_values:
                leaves.extend(child.leaf_nodes)
                chain.extend(child.dependency_chain)
                provenance_chain.extend(child.provenance_chain)
            if not child_values:
                leaves.append(node.node_id)
            return EconomicValue(
                node_id=node.node_id,
                fiscal_year=year,
                fiscal_period=period,
                unit=node.unit,
                currency=node.currency,
                available=False,
                origin="unavailable",
                provenance=node.provenance,
                provenance_chain=_unique_text(provenance_chain),
                dependency_chain=_unique_text(chain),
                leaf_nodes=_unique_text(leaves),
                unresolved_reasons=(reason,),
            )

        def evaluate_node(
            node_id: str, year: int, requested_period: str
        ) -> EconomicValue:
            key = (node_id, year, requested_period)
            if key in memo:
                return memo[key]
            node = node_by_id[node_id]
            if key in active:
                # This is defensive; zero-lag cycles are rejected before this
                # point.  A positive lag changes the key's fiscal year.
                value = unavailable(node, year, "cycle during evaluation")
                diagnostic(
                    "evaluation_cycle",
                    f"Cycle encountered while evaluating {node_id}",
                    node_id=node_id,
                    fiscal_year=year,
                    severity="error",
                )
                return value
            active.add(key)
            observation = select_observation(node, year, requested_period)
            relationship = producer_by_target.get(node_id)
            historical_parameter = relationship is not None and (
                relationship.historical_parameter_derivation
            )

            # A reasoner/manual observation is an explicit parameter override.
            # A reported target observation does not suppress a flagged
            # historical ratio when its reported numerator and denominator are
            # available; the derived cell must retain its non-reported origin.
            if (
                observation is not None
                and observation_is_authoritative(node, observation, relationship)
                and (
                    not historical_parameter
                    or observation.origin not in _REPORTED_ORIGINS
                )
            ):
                value = observation_value(node, observation, year, requested_period)
                memo[key] = value
                active.remove(key)
                return value

            if relationship is None:
                reason = (
                    "reported derived value is reconstruction target"
                    if observation is not None
                    else "missing required leaf"
                )
                value = unavailable(node, year, reason)
                memo[key] = value
                active.remove(key)
                return value

            structural_records = (
                ("target node", node.provenance, node.node_id),
                ("relationship", relationship.provenance, relationship.target),
                *(
                    ("source node", node_by_id[edge.node_id].provenance, edge.node_id)
                    for edge in relationship.sources
                    if edge.node_id in node_by_id
                ),
            )
            structural_issue = next(
                (
                    (label, record_id, issue)
                    for label, provenance, record_id in structural_records
                    if (issue := provenance_as_of_issue(provenance, as_of_date))
                    is not None
                ),
                None,
            )
            if structural_issue is not None:
                label, record_id, (code, message) = structural_issue
                value = unavailable(
                    node,
                    year,
                    f"{label} cannot execute point-in-time: {message}",
                )
                diagnostic(code, message, node_id=record_id, fiscal_year=year)
                memo[key] = value
                active.remove(key)
                return value

            if historical_parameter:
                # Historical parameter derivation is deliberately a direct
                # reported-evidence join.  Evaluating source nodes normally
                # would leak their unresolved forecast leaves into the
                # parameter requirement and could accidentally forecast the
                # ratio from a formula path.
                source_values: list[EconomicValue] = []
                for edge in relationship.sources:
                    source_node = node_by_id[edge.node_id]
                    source_year = year - edge.fiscal_lag
                    source_observation = select_observation(
                        source_node,
                        source_year,
                        requested_period,
                        origins=_REPORTED_ORIGINS,
                    )
                    if source_observation is None:
                        value = unavailable(
                            node,
                            year,
                            "missing reported parameter derivation sources",
                        )
                        memo[key] = value
                        active.remove(key)
                        return value
                    source_values.append(
                        observation_value(
                            source_node,
                            source_observation,
                            source_year,
                            requested_period,
                        )
                    )
                try:
                    result = _apply_relationship(
                        relationship, source_values, node_by_id=node_by_id
                    )
                except (ArithmeticError, ValueError, ZeroDivisionError) as exc:
                    value = unavailable(node, year, str(exc), children=source_values)
                    diagnostic(
                        "relationship_unavailable",
                        f"{node.node_id}: {exc}",
                        node_id=node.node_id,
                        fiscal_year=year,
                    )
                    memo[key] = value
                    active.remove(key)
                    return value
                value = EconomicValue(
                    node_id=node.node_id,
                    fiscal_year=year,
                    fiscal_period=requested_period,
                    value=result,
                    unit=node.unit,
                    currency=node.currency,
                    available=True,
                    origin=_HISTORICAL_PARAMETER_ORIGIN,
                    provenance=relationship.provenance or node.provenance,
                    provenance_chain=_unique_text(
                        (
                            _provenance_label(node.provenance),
                            _provenance_label(relationship.provenance),
                            *(item.provenance_chain for item in source_values),
                        )
                    ),
                    dependency_chain=_unique_text(
                        [node.node_id, *(item.dependency_chain for item in source_values)]
                    ),
                    leaf_nodes=_unique_text(
                        item.leaf_nodes for item in source_values
                    ),
                )
                memo[key] = value
                active.remove(key)
                return value

            if relationship.relationship_type == EconomicRelationshipType.RESIDUAL:
                # A residual is historical-only unless an explicit component
                # observation won above.  In particular, an absent total is
                # not an AI target: the residual component is the forecast
                # leaf, and the reported-total source is merely evidence.
                total_edge = relationship.sources[0]
                total_node = node_by_id[total_edge.node_id]
                total_year = year - total_edge.fiscal_lag
                total_observation = select_observation(
                    total_node,
                    total_year,
                    requested_period,
                    origins=_REPORTED_ORIGINS,
                )
                if total_observation is None:
                    value = unavailable(
                        node,
                        year,
                        "residual requires a reported historical total or explicit component observation",
                    )
                    memo[key] = value
                    active.remove(key)
                    return value

                child_values = [
                    observation_value(
                        total_node,
                        total_observation,
                        total_year,
                        requested_period,
                    )
                ]
                child_values.extend(
                    evaluate_node(
                        edge.node_id,
                        year - edge.fiscal_lag,
                        requested_period,
                    )
                    for edge in relationship.sources[1:]
                )
                if any(not child.is_available for child in child_values):
                    value = unavailable(
                        node,
                        year,
                        "residual requires reported historical total and known components",
                        children=child_values,
                    )
                    memo[key] = value
                    active.remove(key)
                    return value
                try:
                    result = _apply_relationship(
                        relationship, child_values, node_by_id=node_by_id
                    )
                except (ArithmeticError, ValueError, ZeroDivisionError) as exc:
                    value = unavailable(node, year, str(exc), children=child_values)
                    diagnostic(
                        "relationship_unavailable",
                        f"{node.node_id}: {exc}",
                        node_id=node.node_id,
                        fiscal_year=year,
                    )
                    memo[key] = value
                    active.remove(key)
                    return value
                value = EconomicValue(
                    node_id=node.node_id,
                    fiscal_year=year,
                    fiscal_period=requested_period,
                    value=result,
                    unit=node.unit,
                    currency=node.currency,
                    available=True,
                    origin="derived_historical_residual",
                    provenance=relationship.provenance or node.provenance,
                    provenance_chain=_unique_text(
                        (
                            _provenance_label(node.provenance),
                            _provenance_label(relationship.provenance),
                            *(item.provenance_chain for item in child_values),
                        )
                    ),
                    dependency_chain=_unique_text(
                        [node.node_id, *(item.dependency_chain for item in child_values)]
                    ),
                    leaf_nodes=_unique_text(
                        item.leaf_nodes for item in child_values
                    ),
                )
                memo[key] = value
                active.remove(key)
                return value

            child_values: list[EconomicValue] = []
            for edge in relationship.sources:
                source_year = year - edge.fiscal_lag
                child_values.append(
                    evaluate_node(edge.node_id, source_year, requested_period)
                )

            if any(not child.is_available for child in child_values):
                value = unavailable(
                    node,
                    year,
                    "missing required source",
                    children=child_values,
                )
                memo[key] = value
                active.remove(key)
                return value

            try:
                result = _apply_relationship(
                    relationship, child_values, node_by_id=node_by_id
                )
            except (ArithmeticError, ValueError, ZeroDivisionError) as exc:
                value = unavailable(node, year, str(exc), children=child_values)
                diagnostic(
                    "relationship_unavailable",
                    f"{node.node_id}: {exc}",
                    node_id=node.node_id,
                    fiscal_year=year,
                )
                memo[key] = value
                active.remove(key)
                return value

            dependency_chain = _unique_text(
                [node.node_id, *(item.dependency_chain for item in child_values)]
            )
            leaves = _unique_text(item.leaf_nodes for item in child_values)
            provenance_chain = _unique_text(
                (
                    _provenance_label(node.provenance),
                    _provenance_label(relationship.provenance),
                    *(item.provenance_chain for item in child_values),
                )
            )
            value = EconomicValue(
                node_id=node.node_id,
                fiscal_year=year,
                fiscal_period=requested_period,
                value=result,
                unit=node.unit,
                currency=node.currency,
                available=True,
                origin=f"relationship:{relationship.relationship_type.value}",
                provenance=relationship.provenance or node.provenance,
                provenance_chain=provenance_chain,
                dependency_chain=dependency_chain,
                leaf_nodes=leaves,
            )
            memo[key] = value
            active.remove(key)
            return value

        # Evaluating all nodes, rather than only the root, makes diagnostics
        # useful for partially constructed graphs and preserves shared drivers.
        cells: list[EconomicValue] = []
        for year in years:
            for node in model.nodes:
                cells.append(evaluate_node(node.node_id, year, period))

        values = {
            node.node_id: {
                year: memo[(node.node_id, year, period)].value for year in years
            }
            for node in model.nodes
        }
        audits = tuple(
            DependencyAudit(
                node_id=cell.node_id,
                fiscal_year=cell.fiscal_year,
                fiscal_period=cell.fiscal_period,
                available=cell.is_available,
                dependency_chain=cell.dependency_chain,
                leaf_nodes=cell.leaf_nodes,
                provenance_chain=cell.provenance_chain,
                unresolved_reasons=cell.unresolved_reasons,
            )
            for cell in cells
        )
        # ``memo`` also contains source cells reached through a fiscal lag even
        # when that source year is outside the requested output horizon.  The
        # requirement walk must see those cells or it would turn an available
        # lagged fact into a spurious forecast leaf.
        requirements = _requirements(cells, model, evaluated_cells=memo.values())
        diagnostics = _build_diagnostics(
            model,
            years,
            period,
            cells,
            requirements,
            observations_by_key,
            as_of_date,
            runtime_diagnostics,
        )
        return EconomicEvaluationResult(
            target_years=years,
            fiscal_period=period,
            as_of=as_of_date,
            values=values,
            cells=tuple(cells),
            dependency_audits=audits,
            unresolved_leaf_requirements=requirements,
            diagnostics=diagnostics,
        )

    run = evaluate


def _apply_relationship(
    relationship: EconomicRelationship,
    children: list[EconomicValue],
    *,
    node_by_id: dict[str, EconomicNode] | None = None,
) -> Decimal:
    edges = relationship.sources
    kind = relationship.relationship_type
    if kind in {EconomicRelationshipType.IDENTITY, EconomicRelationshipType.LAG}:
        return _require_value(children[0]) * edges[0].sign * edges[0].weight
    if kind in {
        EconomicRelationshipType.ADD,
        EconomicRelationshipType.WEIGHTED_SUM,
        EconomicRelationshipType.RESIDUAL,
    }:
        return sum(
            (
                _require_value(child) * edge.sign * edge.weight
                for edge, child in zip(edges, children, strict=True)
            ),
            Decimal(0),
        )
    if kind == EconomicRelationshipType.SUBTRACT:
        if all("sign" not in edge.model_fields_set for edge in edges):
            return _require_value(children[0]) - sum(
                (_require_value(child) for child in children[1:]), Decimal(0)
            )
        return sum(
            (
                _require_value(child) * edge.sign * edge.weight
                for edge, child in zip(edges, children, strict=True)
            ),
            Decimal(0),
        )
    if kind == EconomicRelationshipType.MULTIPLY:
        result = Decimal(1)
        for edge, child in zip(edges, children, strict=True):
            operand = _require_value(child) * edge.sign * edge.weight
            source = node_by_id.get(edge.node_id) if node_by_id is not None else None
            if source is not None and source.unit_kind == EconomicUnitKind.RATE:
                operand *= rate_unit_scale(source.unit)
            result *= operand
        return result
    if kind == EconomicRelationshipType.RATIO:
        numerator = _require_value(children[0]) * edges[0].sign * edges[0].weight
        denominator = _require_value(children[1]) * edges[1].sign * edges[1].weight
        if denominator == 0:
            raise ZeroDivisionError("ratio denominator is zero")
        return numerator / denominator
    if kind == EconomicRelationshipType.GROWTH:
        base = _require_value(children[0]) * edges[0].sign * edges[0].weight
        rate = _require_value(children[1]) * edges[1].sign * edges[1].weight
        source = node_by_id.get(edges[1].node_id) if node_by_id is not None else None
        if source is not None and source.unit_kind == EconomicUnitKind.RATE:
            rate *= rate_unit_scale(source.unit)
        return base * (Decimal(1) + rate)
    raise ValueError(f"Unsupported relationship type: {kind.value}")


def _require_value(cell: EconomicValue) -> Decimal:
    if not cell.is_available or cell.value is None:
        raise ValueError("source value is unavailable")
    return cell.value


def _requirements(
    cells: Iterable[EconomicValue],
    model: EconomicModel,
    *,
    evaluated_cells: Iterable[EconomicValue] | None = None,
) -> tuple[UnresolvedLeafRequirement, ...]:
    """Traverse unresolved requirements backward from declared graph roots.

    The evaluator intentionally materializes every node-period cell for
    diagnostics.  Requirements are a different surface: only a root's
    dependency paths are actionable.  Walking those paths here avoids both
    orphan requirements and the flattened dependency/relationship metadata
    produced by scanning every unavailable cell.
    """

    cells = tuple(cells)
    evaluated = tuple(evaluated_cells) if evaluated_cells is not None else cells
    by_key = {
        (cell.node_id, cell.fiscal_year, cell.fiscal_period): cell
        for cell in (*cells, *evaluated)
    }
    node_by_id = {node.node_id: node for node in model.nodes}
    relationship_by_target = {
        relationship.target: relationship for relationship in model.relationships
    }

    requirements: list[UnresolvedLeafRequirement] = []
    by_requirement: dict[
        tuple[str, int, str], UnresolvedLeafRequirement
    ] = {}

    def add_requirement(
        node_id: str,
        year: int,
        period: str,
        path: tuple[str, ...],
        relationship_ids: tuple[str, ...],
    ) -> None:
        node = node_by_id[node_id]
        cell = by_key.get((node_id, year, period))
        reason = (
            cell.unresolved_reasons[0]
            if cell is not None and cell.unresolved_reasons
            else "missing required leaf"
        )
        key = (node_id, year, period)
        requirement = UnresolvedLeafRequirement(
            node_id=node_id,
            fiscal_year=year,
            fiscal_period=period,
            reason=reason,
            path=path,
            scope=node.scope,
            scope_id=node.scope_id,
            metric=node.metric,
            unit=node.unit,
            currency=node.currency,
            materiality=node.materiality,
            required_by_relationship_ids=relationship_ids,
        )
        previous = by_requirement.get(key)
        if previous is None:
            by_requirement[key] = requirement
            requirements.append(requirement)
            return
        merged_ids = tuple(
            dict.fromkeys(
                (
                    *previous.required_by_relationship_ids,
                    *relationship_ids,
                )
            )
        )
        if merged_ids != previous.required_by_relationship_ids:
            replacement = previous.model_copy(
                update={"required_by_relationship_ids": merged_ids}
            )
            by_requirement[key] = replacement
            requirements[requirements.index(previous)] = replacement

    def walk(
        node_id: str,
        year: int,
        period: str,
        path: tuple[str, ...],
        relationship_ids: tuple[str, ...],
        active: frozenset[tuple[str, int, str]],
    ) -> None:
        node = node_by_id[node_id]
        key = (node_id, year, period)
        cell = by_key.get(key)
        if cell is not None and cell.is_available:
            return
        if key in active:
            # Validation rejects zero-lag cycles.  This guard keeps malformed
            # positive-lag inputs from recursing indefinitely if a caller uses
            # the non-raising validation report directly.
            add_requirement(node_id, year, period, path, relationship_ids)
            return

        relationship = relationship_by_target.get(node_id)
        current_path = (*path, node_id)
        current_relationship_ids = (
            (*relationship_ids, relationship.relationship_id)
            if relationship is not None and relationship.relationship_id is not None
            else relationship_ids
        )
        if relationship is None:
            add_requirement(
                node_id,
                year,
                period,
                current_path,
                current_relationship_ids,
            )
            return

        # These relationships describe historical reconstruction only.  When
        # their target is explicitly assumable, the target is the forecast
        # leaf; its evidence sources must not become independent AI targets.
        forecast_leaf = node.forecast_assumption_allowed and (
            (
                node.node_type == EconomicNodeType.INPUT
                and relationship.historical_parameter_derivation
            )
            or (
                node.node_type == EconomicNodeType.COMPONENT
                and relationship.relationship_type == EconomicRelationshipType.RESIDUAL
            )
        )
        if forecast_leaf:
            add_requirement(
                node_id,
                year,
                period,
                current_path,
                current_relationship_ids,
            )
            return

        for edge in relationship.sources:
            walk(
                edge.node_id,
                year - edge.fiscal_lag,
                period,
                current_path,
                current_relationship_ids,
                active | {key},
            )

    roots: list[str] = []
    if model.revenue_root:
        roots.append(model.revenue_root)
    # Business roots are explicit roots, not an invitation to scan every node.
    # Keeping them after the company root makes the company path canonical when
    # a business root is also a descendant of that company root.
    roots.extend(model.business_roots)
    if not roots:
        # Preserve standalone parameter/residual graph behavior for callers that
        # have not declared a company root, without reintroducing an all-node
        # orphan scan.
        roots.extend(
            node.node_id
            for node in model.nodes
            if (
                (
                    (relationship := relationship_by_target.get(node.node_id))
                    is not None
                    and node.node_type == EconomicNodeType.INPUT
                    and relationship.historical_parameter_derivation
                )
                or (
                    (relationship := relationship_by_target.get(node.node_id))
                    is not None
                    and node.node_type == EconomicNodeType.COMPONENT
                    and relationship.relationship_type
                    == EconomicRelationshipType.RESIDUAL
                )
            )
        )

    for root in dict.fromkeys(roots):
        if root not in node_by_id:
            continue
        for year in sorted(
            {
                cell.fiscal_year
                for cell in cells
                if cell.fiscal_period == model.fiscal_period
            }
        ):
            walk(root, year, model.fiscal_period, (), (), frozenset())
    return tuple(requirements)


def _build_diagnostics(
    model: EconomicModel,
    years: tuple[int, ...],
    period: str,
    cells: list[EconomicValue],
    requirements: tuple[UnresolvedLeafRequirement, ...],
    observations_by_key: dict[tuple[str, int, str], list[EconomicObservation]],
    as_of: _datetime.date | None,
    runtime_diagnostics: list[GraphDiagnostic],
) -> GraphDiagnostics:
    by_key = {
        (cell.node_id, cell.fiscal_year, cell.fiscal_period): cell for cell in cells
    }
    node_by_id = {node.node_id: node for node in model.nodes}

    component_nodes = [
        node for node in model.nodes if node.node_type == EconomicNodeType.COMPONENT
    ]
    aggregate_nodes = [
        node for node in model.nodes if node.node_type == EconomicNodeType.AGGREGATE
    ]
    component_by_node = {
        node.node_id: _coverage(node.node_id, years, period, by_key)
        for node in component_nodes
    }
    aggregate_by_node = {
        node.node_id: _coverage(node.node_id, years, period, by_key)
        for node in aggregate_nodes
    }
    component_coverage = _mean_coverage(component_by_node.values())
    aggregate_coverage = _mean_coverage(aggregate_by_node.values())

    root_id = model.revenue_root
    reported_years: list[int] = []
    reconstruction_errors: dict[int, Decimal] = {}
    if root_id and root_id in node_by_id:
        root = node_by_id[root_id]
        for year in years:
            observations = observations_by_key.get((root_id, year, period), [])
            reported = _available_reported_observation(observations, as_of, root)
            if reported is None:
                continue
            reported_years.append(year)
            cell = by_key.get((root_id, year, period))
            if cell is not None and cell.is_available and cell.value is not None:
                if reported.value == 0:
                    error = Decimal(0) if cell.value == 0 else Decimal(1)
                else:
                    error = abs(cell.value - reported.value) / abs(reported.value)
                reconstruction_errors[year] = error

    reconstructable_share = None
    if reported_years:
        reconstructable_share = Decimal(len(reconstruction_errors)) / Decimal(
            len(reported_years)
        )
    reconciliation_error = _mean_decimal(reconstruction_errors.values())

    if root_id and years:
        future_years = tuple(year for year in years if year not in reported_years)
        denominator = future_years or years
        available = sum(
            1
            for year in denominator
            if by_key.get(
                (root_id, year, period),
                EconomicValue(
                    node_id=root_id,
                    fiscal_year=year,
                    fiscal_period=period,
                    unit=node_by_id[root_id].unit,
                ),
            ).is_available
        )
        forecastable_share = Decimal(available) / Decimal(len(denominator))
    else:
        forecastable_share = None

    return GraphDiagnostics(
        component_coverage=component_coverage,
        aggregate_coverage=aggregate_coverage,
        component_coverage_by_node=component_by_node,
        aggregate_coverage_by_node=aggregate_by_node,
        unresolved_count=len(requirements),
        historical_reconstructable_share=reconstructable_share,
        reconciliation_error=reconciliation_error,
        reconciliation_error_by_year=reconstruction_errors,
        forecastable_share=forecastable_share,
        diagnostic_messages=tuple(runtime_diagnostics),
    )


def _coverage(
    node_id: str,
    years: tuple[int, ...],
    period: str,
    by_key: dict[tuple[str, int, str], EconomicValue],
) -> Decimal:
    if not years:
        return Decimal(0)
    return Decimal(
        sum(
            by_key.get((node_id, year, period)) is not None
            and by_key[(node_id, year, period)].is_available
            for year in years
        )
    ) / Decimal(len(years))


def _mean_coverage(values: Iterable[Decimal]) -> Decimal | None:
    values = tuple(values)
    return _mean_decimal(values)


def _mean_decimal(values: Iterable[Decimal]) -> Decimal | None:
    values = tuple(values)
    if not values:
        return None
    return sum(values, Decimal(0)) / Decimal(len(values))


def _available_reported_observation(
    observations: list[EconomicObservation],
    as_of: _datetime.date | None,
    node: EconomicNode,
) -> EconomicObservation | None:
    usable = [
        observation
        for observation in observations
        if observation.origin in _REPORTED_ORIGINS
        and observation.unit == node.unit
        and observation.currency == node.currency
        and observation_as_of_issue(
            observation, as_of, frozenset(_REPORTED_ORIGINS)
        )
        is None
    ]
    if not usable:
        return None
    return max(
        usable,
        key=lambda item: (
            item.available_on or _datetime.date.min,
            _provenance_label(item.provenance),
        ),
    )


def _normalize_years(target_years: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(target_years, int):
        target_years = (target_years,)
    years = tuple(sorted(dict.fromkeys(int(year) for year in target_years)))
    if not years:
        raise ValueError("At least one target year is required")
    return years


def _as_date(
    value: _datetime.date | _datetime.datetime | None,
) -> _datetime.date | None:
    if value is None:
        return None
    if isinstance(value, _datetime.datetime):
        return value.date()
    return value


def _provenance_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    source = getattr(value, "source", None)
    origin = getattr(value, "origin", None)
    reference = getattr(value, "reference", None)
    return ":".join(item for item in (source, origin, reference) if item)


def _unique_text(values: Iterable[object]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if isinstance(value, (tuple, list)):
            nested = value
        else:
            nested = (value,)
        for item in nested:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
    return tuple(result)


def evaluate_graph(
    model: EconomicModel,
    target_years: int | Iterable[int],
    *,
    as_of: _datetime.date | _datetime.datetime | None = None,
    fiscal_period: str | None = None,
) -> EconomicEvaluationResult:
    return EconomicGraphEvaluator().evaluate(
        model,
        target_years,
        as_of=as_of,
        fiscal_period=fiscal_period,
    )


def evaluate_model(
    model: EconomicModel,
    target_years: int | Iterable[int],
    *,
    as_of: _datetime.date | _datetime.datetime | None = None,
    fiscal_period: str | None = None,
) -> EconomicEvaluationResult:
    return evaluate_graph(
        model,
        target_years,
        as_of=as_of,
        fiscal_period=fiscal_period,
    )


evaluate_economic_graph = evaluate_graph
evaluate_economic_model = evaluate_model
OperatingGraphEvaluator = EconomicGraphEvaluator
EconomicDAGEvaluator = EconomicGraphEvaluator


__all__ = [
    "EconomicDAGEvaluator",
    "EconomicGraphEvaluator",
    "OperatingGraphEvaluator",
    "evaluate_economic_graph",
    "evaluate_economic_model",
    "evaluate_graph",
    "evaluate_model",
]
