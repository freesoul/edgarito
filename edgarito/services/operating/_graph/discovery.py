"""Evidence-only discovery of provider-neutral economic graph structure.

This module is deliberately a construction and validation seam.  It accepts
already extracted graph candidates; it does not retrieve documents, call an AI
provider, estimate parameters, or create forecasts.  Numeric values can enter
the result only as point-in-time observations supplied by the caller.
"""

from __future__ import annotations

import datetime as _datetime
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from edgarito.schemas.operating_graph import (
    EconomicModel,
    EconomicNode,
    EconomicNodeType,
    EconomicObservation,
    EconomicProvenance,
    EconomicRelationship,
    EconomicUnitKind,
    GraphDiagnostic,
)
from edgarito.services.operating._graph.point_in_time import provenance_as_of_issue
from edgarito.services.operating._graph.validation import EconomicGraphValidator

_FORBIDDEN_ORIGINS = frozenset(
    {
        "assumption",
        "derived_historical_parameter",
        "estimated",
        "forecast",
        "forecasted",
        "imputed",
        "inferred",
        "model_assumption",
        "synthetic",
    }
)

_OBSERVATION_ERROR_CODES = frozenset(
    {
        "unknown_observation_node",
        "observation_scope_mismatch",
        "observation_scope_id_mismatch",
        "observation_unit_mismatch",
        "observation_currency_mismatch",
    }
)


def _text(value: object) -> str:
    return str(value).strip()


def _date(value: _datetime.date | _datetime.datetime | None) -> _datetime.date | None:
    if value is None:
        return None
    return value.date() if isinstance(value, _datetime.datetime) else value


def _records(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping, BaseModel)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _has_provenance(value: object) -> bool:
    provenance = getattr(value, "provenance", None)
    if isinstance(provenance, str):
        return bool(provenance.strip())
    if isinstance(provenance, EconomicProvenance):
        return bool(
            provenance.source
            or provenance.origin
            or provenance.reference
            or provenance.evidence_ids
            or provenance.methodology
        )
    return provenance is not None and bool(str(provenance).strip())


def _record_id(value: object) -> str | None:
    if isinstance(value, EconomicNode):
        return value.node_id
    if isinstance(value, EconomicRelationship):
        return value.relationship_id or value.target
    if isinstance(value, EconomicObservation):
        return f"{value.node_id}:{value.fiscal_year}:{value.fiscal_period}"
    if isinstance(value, Mapping):
        for key in ("node_id", "stable_id", "id", "target", "node"):
            if value.get(key) is not None:
                return _text(value[key])
    return None


class EconomicModelDiscoveryRejection(BaseModel):
    """A candidate retained with a deterministic rejection reason."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_type: str
    record_id: str | None = None
    code: str
    reason: str
    record: EconomicNode | EconomicRelationship | EconomicObservation | None = None

    @property
    def message(self) -> str:
        return self.reason


class EconomicModelDiscoveryUnresolvedLeaf(BaseModel):
    """An unresolved structural input required by one or more graph paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    reason: str
    category: str = "input"
    required_by: tuple[str, ...] = ()

    @field_validator("node_id", "reason", "category")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = _text(value)
        if not normalized:
            raise ValueError("Discovery unresolved-leaf text cannot be blank")
        return normalized


class EconomicModelDiscoveryAudit(BaseModel):
    """Immutable counts and policy facts for one discovery run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: _datetime.date | None = None
    candidate_count: int = 0
    accepted_node_count: int = 0
    accepted_relationship_count: int = 0
    accepted_observation_count: int = 0
    rejected_record_count: int = 0
    future_record_count: int = 0
    unresolved_leaf_count: int = 0
    evidence_only: bool = True
    forecasts_added: int = 0
    coefficients_added: int = 0
    validation_error_count: int = 0

    @property
    def accepted_record_count(self) -> int:
        return (
            self.accepted_node_count
            + self.accepted_relationship_count
            + self.accepted_observation_count
        )


class EconomicModelDiscoveryResult(BaseModel):
    """Immutable accepted graph plus every discovery audit surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted_model: EconomicModel
    rejected_records: tuple[EconomicModelDiscoveryRejection, ...] = ()
    unresolved_leaves: tuple[EconomicModelDiscoveryUnresolvedLeaf, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    diagnostics: tuple[GraphDiagnostic, ...] = ()
    audit: EconomicModelDiscoveryAudit

    @property
    def model(self) -> EconomicModel:
        """Compatibility-friendly name for the accepted model."""

        return self.accepted_model

    @property
    def accepted(self) -> EconomicModel:
        return self.accepted_model

    @property
    def rejected(self) -> tuple[EconomicModelDiscoveryRejection, ...]:
        return self.rejected_records

    @property
    def rejections(self) -> tuple[EconomicModelDiscoveryRejection, ...]:
        return self.rejected_records

    @property
    def rejected_reasons(self) -> tuple[str, ...]:
        return tuple(item.reason for item in self.rejected_records)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_records)

    @property
    def unresolved_leaf_requirements(
        self,
    ) -> tuple[EconomicModelDiscoveryUnresolvedLeaf, ...]:
        return self.unresolved_leaves

    @property
    def audits(self) -> tuple[EconomicModelDiscoveryAudit, ...]:
        return (self.audit,)

    @property
    def valid(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)

    @property
    def available(self) -> bool:
        return bool(self.accepted_model.nodes)

    @property
    def diagnostic(self) -> tuple[GraphDiagnostic, ...]:
        return self.diagnostics

    @property
    def missing(self) -> tuple[str, ...]:
        return self.missing_evidence

    @property
    def audit_records(self) -> tuple[EconomicModelDiscoveryAudit, ...]:
        return (self.audit,)


# Short aliases keep the contract discoverable without creating divergent
# provider-specific result models.
EconomicDiscoveryRejection = EconomicModelDiscoveryRejection
EconomicDiscoveryUnresolvedLeaf = EconomicModelDiscoveryUnresolvedLeaf
EconomicDiscoveryAudit = EconomicModelDiscoveryAudit
EconomicDiscoveryResult = EconomicModelDiscoveryResult


class EconomicModelDiscoveryService:
    """Validate and assemble typed economic structure candidates.

    ``discover`` is the explicit seam for a future extractor.  The service has
    no retrieval dependencies and intentionally rejects forecast-like origins
    and non-unit numeric edge weights.
    """

    def __init__(self, validator: EconomicGraphValidator | None = None) -> None:
        self.validator = validator or EconomicGraphValidator()

    @staticmethod
    def construct_model(
        nodes: Iterable[EconomicNode],
        relationships: Iterable[EconomicRelationship] = (),
        observations: Iterable[EconomicObservation] = (),
        *,
        revenue_root: str | None = None,
        business_roots: Iterable[str] = (),
        fiscal_period: str = "FY",
    ) -> EconomicModel:
        """Construct only typed graph metadata; no values are inferred."""

        return EconomicModel(
            nodes=tuple(nodes),
            relationships=tuple(relationships),
            observations=tuple(observations),
            revenue_root=revenue_root,
            business_roots=tuple(business_roots),
            fiscal_period=fiscal_period,
        )

    build_model = construct_model

    def discover(
        self,
        nodes: EconomicModel | Iterable[EconomicNode] | Mapping[str, Any] | None = None,
        relationships: Iterable[EconomicRelationship] | None = None,
        observations: Iterable[EconomicObservation] | None = None,
        *,
        model: EconomicModel | Mapping[str, Any] | None = None,
        candidates: Mapping[str, Any] | None = None,
        as_of: _datetime.date | _datetime.datetime | None = None,
        revenue_root: str | None = None,
        business_roots: Iterable[str] = (),
        fiscal_period: str | None = None,
    ) -> EconomicModelDiscoveryResult:
        """Accept candidates, apply the as-of boundary, and validate the graph.

        Mapping inputs are accepted only as a convenience for future typed
        extraction adapters; every accepted record is validated into the
        provider-neutral graph contracts before it reaches the model.
        """

        source = model
        business_roots = tuple(business_roots)
        if source is None and isinstance(nodes, EconomicModel):
            source = nodes
            nodes = None
        if source is None and isinstance(nodes, Mapping):
            source = nodes if "nodes" in nodes else None
            if source is not None:
                nodes = None
        if candidates is not None:
            if source is not None:
                raise ValueError("Use either model or candidates, not both")
            source = candidates
        source_model = (
            source
            if isinstance(source, EconomicModel)
            else EconomicModel.model_validate(source)
            if source is not None
            else None
        )

        if source_model is not None:
            raw_nodes = source_model.nodes
            raw_relationships = source_model.relationships
            raw_observations = source_model.observations
            if revenue_root is None:
                revenue_root = source_model.revenue_root
            if not business_roots:
                business_roots = source_model.business_roots
            if fiscal_period is None:
                fiscal_period = source_model.fiscal_period
        else:
            payload = candidates or {}
            raw_nodes = nodes if nodes is not None else payload.get("nodes", ())
            raw_relationships = (
                relationships
                if relationships is not None
                else payload.get("relationships", ())
            )
            raw_observations = (
                observations
                if observations is not None
                else payload.get("observations", ())
            )
            if revenue_root is None:
                revenue_root = payload.get("revenue_root")
            if not business_roots:
                business_roots = payload.get("business_roots", ())
            if fiscal_period is None:
                fiscal_period = payload.get("fiscal_period", "FY")

        raw_nodes = _records(raw_nodes)
        raw_relationships = _records(raw_relationships)
        raw_observations = _records(raw_observations)
        business_roots = tuple(business_roots)
        cutoff = _date(as_of)
        rejected: list[EconomicModelDiscoveryRejection] = []
        future_count = 0

        accepted_nodes: list[EconomicNode] = []
        seen_node_ids: set[str] = set()
        for raw in _records(raw_nodes):
            node, rejection = self._candidate(raw, EconomicNode, "node")
            if rejection:
                rejected.append(rejection)
                continue
            assert node is not None
            if not _has_provenance(node):
                rejected.append(
                    self._rejection(
                        "node",
                        node,
                        "missing_provenance",
                        "Economic node candidates require provenance",
                    )
                )
            elif node.node_id in seen_node_ids:
                rejected.append(
                    self._rejection(
                        "node",
                        node,
                        "duplicate_node_id",
                        f"Duplicate economic node ID: {node.node_id}",
                    )
                )
            else:
                seen_node_ids.add(node.node_id)
                accepted_nodes.append(node)

        accepted_relationships: list[EconomicRelationship] = []
        seen_relationship_ids: set[str] = set()
        seen_targets: set[str] = set()
        accepted_nodes_by_id = {node.node_id: node for node in accepted_nodes}
        for raw in _records(raw_relationships):
            relationship, rejection = self._candidate(
                raw, EconomicRelationship, "relationship"
            )
            if rejection:
                rejected.append(rejection)
                continue
            assert relationship is not None
            if not _has_provenance(relationship):
                rejected.append(
                    self._rejection(
                        "relationship",
                        relationship,
                        "missing_provenance",
                        "Economic relationship candidates require provenance",
                    )
                )
                continue
            structural_records = (
                ("target node", accepted_nodes_by_id.get(relationship.target)),
                ("relationship", relationship),
                *(
                    ("source node", accepted_nodes_by_id.get(edge.node_id))
                    for edge in relationship.sources
                ),
            )
            structural_issue = next(
                (
                    (label, record, issue)
                    for label, record in structural_records
                    if record is not None
                    and (issue := provenance_as_of_issue(record.provenance, cutoff))
                    is not None
                ),
                None,
            )
            if structural_issue is not None:
                label, record, (code, message) = structural_issue
                rejected.append(
                    self._rejection(
                        "relationship",
                        relationship,
                        code,
                        f"{label} {record.node_id if isinstance(record, EconomicNode) else relationship.target}: {message}",
                    )
                )
                continue
            relation_issue = self._numeric_edge_issue(relationship)
            if relation_issue is not None:
                rejected.append(
                    self._rejection(
                        "relationship",
                        relationship,
                        "numeric_coefficient_forbidden",
                        relation_issue,
                    )
                )
                continue
            relation_id = relationship.relationship_id
            if relation_id and relation_id in seen_relationship_ids:
                rejected.append(
                    self._rejection(
                        "relationship",
                        relationship,
                        "duplicate_relationship_id",
                        f"Duplicate relationship ID: {relation_id}",
                    )
                )
                continue
            if relationship.target in seen_targets:
                rejected.append(
                    self._rejection(
                        "relationship",
                        relationship,
                        "duplicate_producer",
                        f"Node has more than one producer: {relationship.target}",
                    )
                )
                continue
            seen_targets.add(relationship.target)
            if relation_id:
                seen_relationship_ids.add(relation_id)
            accepted_relationships.append(relationship)

        accepted_observations: list[EconomicObservation] = []
        for raw in _records(raw_observations):
            observation, rejection = self._candidate(
                raw, EconomicObservation, "observation"
            )
            if rejection:
                rejected.append(rejection)
                continue
            assert observation is not None
            if not _has_provenance(observation):
                rejected.append(
                    self._rejection(
                        "observation",
                        observation,
                        "missing_provenance",
                        "Economic observation candidates require provenance",
                    )
                )
                continue
            if observation.available_on is None:
                rejected.append(
                    self._rejection(
                        "observation",
                        observation,
                        "missing_available_on",
                        "Economic observations require available_on for point-in-time safety",
                    )
                )
                continue
            if cutoff is not None and observation.available_on > cutoff:
                future_count += 1
                rejected.append(
                    self._rejection(
                        "observation",
                        observation,
                        "future_evidence",
                        f"Observation was not available as of {cutoff.isoformat()}",
                    )
                )
                continue
            if observation.origin in _FORBIDDEN_ORIGINS:
                rejected.append(
                    self._rejection(
                        "observation",
                        observation,
                        "non_evidence_origin",
                        f"Evidence-only discovery rejects observation origin {observation.origin!r}",
                    )
                )
                continue
            accepted_observations.append(observation)

        period = fiscal_period or "FY"
        accepted_relationships, accepted_observations = self._remove_unknown_refs(
            accepted_nodes,
            accepted_relationships,
            accepted_observations,
            rejected,
        )
        accepted_relationships, accepted_observations, graph_diagnostics = (
            self._remove_graph_invalid_records(
                accepted_nodes,
                accepted_relationships,
                accepted_observations,
                revenue_root,
                business_roots,
                period,
                rejected,
            )
        )
        accepted_model = self.construct_model(
            accepted_nodes,
            accepted_relationships,
            accepted_observations,
            revenue_root=revenue_root,
            business_roots=business_roots,
            fiscal_period=period,
        )
        report = self.validator.check(accepted_model)
        graph_diagnostics = tuple(report.diagnostics)
        unresolved = _unresolved_leaves(accepted_model)
        missing = _missing_evidence(accepted_model, unresolved)
        diagnostics = _rejection_diagnostics(rejected) + graph_diagnostics
        diagnostics = _unique_diagnostics(diagnostics)
        audit = EconomicModelDiscoveryAudit(
            as_of=cutoff,
            candidate_count=(
                len(_records(raw_nodes))
                + len(_records(raw_relationships))
                + len(_records(raw_observations))
            ),
            accepted_node_count=len(accepted_nodes),
            accepted_relationship_count=len(accepted_relationships),
            accepted_observation_count=len(accepted_observations),
            rejected_record_count=len(rejected),
            future_record_count=future_count,
            unresolved_leaf_count=len(unresolved),
            validation_error_count=sum(
                item.severity == "error" for item in graph_diagnostics
            ),
        )
        return EconomicModelDiscoveryResult(
            accepted_model=accepted_model,
            rejected_records=tuple(rejected),
            unresolved_leaves=unresolved,
            missing_evidence=missing,
            diagnostics=diagnostics,
            audit=audit,
        )

    run = discover
    retrieve = discover
    from_candidates = discover

    @staticmethod
    def _candidate(
        raw: Any, model_type: type[BaseModel], record_type: str
    ) -> tuple[Any | None, EconomicModelDiscoveryRejection | None]:
        try:
            value = (
                raw if isinstance(raw, model_type) else model_type.model_validate(raw)
            )
        except Exception as exc:
            return None, EconomicModelDiscoveryRejection(
                record_type=record_type,
                record_id=_record_id(raw),
                code="invalid_record",
                reason=f"Invalid {record_type} candidate: {exc}",
            )
        return value, None

    @staticmethod
    def _rejection(
        record_type: str,
        record: EconomicNode | EconomicRelationship | EconomicObservation,
        code: str,
        reason: str,
    ) -> EconomicModelDiscoveryRejection:
        return EconomicModelDiscoveryRejection(
            record_type=record_type,
            record_id=_record_id(record),
            code=code,
            reason=reason,
            record=record,
        )

    @staticmethod
    def _numeric_edge_issue(relationship: EconomicRelationship) -> str | None:
        for edge in relationship.sources:
            if edge.weight != 1:
                return (
                    "Economic discovery cannot accept a numeric edge weight; "
                    f"{edge.node_id} has weight {edge.weight}"
                )
            if abs(edge.sign) != 1:
                return (
                    "Economic discovery accepts only structural signs of +1 or -1; "
                    f"{edge.node_id} has sign {edge.sign}"
                )
        return None

    def _remove_unknown_refs(
        self,
        nodes: list[EconomicNode],
        relationships: list[EconomicRelationship],
        observations: list[EconomicObservation],
        rejected: list[EconomicModelDiscoveryRejection],
    ) -> tuple[list[EconomicRelationship], list[EconomicObservation]]:
        node_ids = {node.node_id for node in nodes}
        valid_relationships: list[EconomicRelationship] = []
        for relationship in relationships:
            references = (relationship.target,) + tuple(
                edge.node_id for edge in relationship.sources
            )
            if any(reference not in node_ids for reference in references):
                rejected.append(
                    self._rejection(
                        "relationship",
                        relationship,
                        "unknown_node_reference",
                        f"Relationship references unknown node(s): {', '.join(sorted(set(references) - node_ids))}",
                    )
                )
            else:
                valid_relationships.append(relationship)
        valid_observations: list[EconomicObservation] = []
        for observation in observations:
            if observation.node_id not in node_ids:
                rejected.append(
                    self._rejection(
                        "observation",
                        observation,
                        "unknown_observation_node",
                        f"Observation references unknown node: {observation.node_id}",
                    )
                )
            else:
                valid_observations.append(observation)
        return valid_relationships, valid_observations

    def _remove_graph_invalid_records(
        self,
        nodes: list[EconomicNode],
        relationships: list[EconomicRelationship],
        observations: list[EconomicObservation],
        revenue_root: str | None,
        business_roots: tuple[str, ...],
        fiscal_period: str,
        rejected: list[EconomicModelDiscoveryRejection],
    ) -> tuple[
        list[EconomicRelationship],
        list[EconomicObservation],
        tuple[GraphDiagnostic, ...],
    ]:
        # Remove relationship/observation records with a localized validator
        # error.  Re-checking after each pass keeps the returned model valid for
        # the common malformed-candidate cases while retaining diagnostics for
        # model-level errors such as an unknown requested root.
        current_relationships = list(relationships)
        current_observations = list(observations)
        final_report = None
        for _ in range(len(current_relationships) + len(current_observations) + 1):
            candidate = self.construct_model(
                nodes,
                current_relationships,
                current_observations,
                revenue_root=revenue_root,
                business_roots=business_roots,
                fiscal_period=fiscal_period,
            )
            report = self.validator.check(candidate)
            final_report = report
            errors = tuple(report.errors)
            bad_relationship_targets = {
                item.node_id
                for item in errors
                if item.node_id is not None
                and item.code not in _OBSERVATION_ERROR_CODES
            }
            cycle_targets: set[str] = set()
            for item in errors:
                if item.code == "zero_lag_cycle":
                    cycle_targets.update(
                        part.strip()
                        for part in item.message.split(":", 1)[-1].split("->")
                        if part.strip()
                    )
            bad_relationship_targets.update(cycle_targets)
            bad_observation_nodes = {
                item.node_id
                for item in errors
                if item.node_id is not None and item.code in _OBSERVATION_ERROR_CODES
            }
            changed = False
            kept_relationships: list[EconomicRelationship] = []
            for relationship in current_relationships:
                if relationship.target in bad_relationship_targets:
                    issues = tuple(
                        item
                        for item in errors
                        if item.node_id == relationship.target
                        or (
                            item.code == "zero_lag_cycle"
                            and relationship.target in cycle_targets
                        )
                    )
                    reason = "; ".join(item.message for item in issues)
                    rejected.append(
                        self._rejection(
                            "relationship",
                            relationship,
                            issues[0].code if issues else "graph_validation_error",
                            reason or "Relationship failed graph validation",
                        )
                    )
                    changed = True
                else:
                    kept_relationships.append(relationship)
            kept_observations: list[EconomicObservation] = []
            for observation in current_observations:
                if observation.node_id in bad_observation_nodes:
                    issues = tuple(
                        item
                        for item in errors
                        if item.node_id == observation.node_id
                        and item.code in _OBSERVATION_ERROR_CODES
                    )
                    rejected.append(
                        self._rejection(
                            "observation",
                            observation,
                            issues[0].code if issues else "graph_validation_error",
                            "; ".join(item.message for item in issues)
                            or "Observation failed graph validation",
                        )
                    )
                    changed = True
                else:
                    kept_observations.append(observation)
            current_relationships = kept_relationships
            current_observations = kept_observations
            if not changed:
                break
        return (
            current_relationships,
            current_observations,
            tuple(final_report.diagnostics) if final_report is not None else (),
        )


def _unresolved_leaves(
    model: EconomicModel,
) -> tuple[EconomicModelDiscoveryUnresolvedLeaf, ...]:
    producers = {
        relationship.target: relationship for relationship in model.relationships
    }
    observations = {observation.node_id for observation in model.observations}
    node_by_id = {node.node_id: node for node in model.nodes}
    paths: dict[str, list[str]] = {}

    def visit(node_id: str, path: tuple[str, ...], active: frozenset[str]) -> None:
        if node_id in active:
            return
        relationship = producers.get(node_id)
        if relationship is None:
            if node_id not in observations:
                paths.setdefault(node_id, []).append(" -> ".join(path))
            return
        for edge in relationship.sources:
            if edge.node_id in node_by_id:
                visit(edge.node_id, (*path, node_id), active | {node_id})

    roots = (model.revenue_root,) if model.revenue_root else tuple(node_by_id)
    for root in roots:
        if root in node_by_id:
            visit(root, (), frozenset())
    result: list[EconomicModelDiscoveryUnresolvedLeaf] = []
    growth_present = any("growth" in node.metric.casefold() for node in model.nodes)
    for node_id in sorted(paths):
        node = node_by_id[node_id]
        if node.unit_kind == EconomicUnitKind.MONETARY_PER_UNIT or any(
            marker in node.metric.casefold()
            for marker in ("yield", "per transaction", "per unit")
        ):
            category = "coefficient"
            reason = "missing disclosed coefficient evidence"
        elif "volume" in node.metric.casefold() and growth_present:
            category = "driver"
            reason = "missing absolute volume evidence; growth evidence does not establish a level"
        elif (
            node.node_type == EconomicNodeType.COMPONENT
            and node.forecast_assumption_allowed
        ):
            category = "component"
            reason = (
                "missing component evidence; forecast assumption remains unresolved"
            )
        else:
            category = "input"
            reason = "missing required leaf evidence"
        result.append(
            EconomicModelDiscoveryUnresolvedLeaf(
                node_id=node_id,
                reason=reason,
                category=category,
                required_by=tuple(dict.fromkeys(paths[node_id])),
            )
        )
    return tuple(result)


def _missing_evidence(
    model: EconomicModel,
    leaves: tuple[EconomicModelDiscoveryUnresolvedLeaf, ...],
) -> tuple[str, ...]:
    node_by_id = {node.node_id: node for node in model.nodes}
    observed = {observation.node_id for observation in model.observations}
    items = [f"{item.node_id}: {item.reason}" for item in leaves]
    for relationship in model.relationships:
        target = node_by_id.get(relationship.target)
        if target is not None and target.node_type == EconomicNodeType.COMPONENT:
            if relationship.target not in observed:
                items.append(
                    f"{relationship.target}: no historical component revenue observation"
                )
    if model.revenue_root and any(
        node.node_type == EconomicNodeType.COMPONENT and node.node_id not in observed
        for node in model.nodes
    ):
        items.append(
            f"{model.revenue_root}: total cannot be reconstructed or forecast without component and leaf evidence"
        )
    return tuple(dict.fromkeys(items))


def _rejection_diagnostics(
    rejected: Iterable[EconomicModelDiscoveryRejection],
) -> tuple[GraphDiagnostic, ...]:
    return tuple(
        GraphDiagnostic(
            code=item.code,
            message=item.reason,
            severity="warning",
            node_id=item.record_id,
        )
        for item in rejected
    )


def _unique_diagnostics(
    diagnostics: Iterable[GraphDiagnostic],
) -> tuple[GraphDiagnostic, ...]:
    result: list[GraphDiagnostic] = []
    seen: set[tuple[str, str, str | None, int | None]] = set()
    for item in diagnostics:
        key = (item.code, item.message, item.node_id, item.fiscal_year)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return tuple(result)


def construct_economic_model(
    nodes: Iterable[EconomicNode],
    relationships: Iterable[EconomicRelationship] = (),
    observations: Iterable[EconomicObservation] = (),
    **kwargs: Any,
) -> EconomicModel:
    """Explicit future-extraction construction seam."""

    return EconomicModelDiscoveryService.construct_model(
        nodes, relationships, observations, **kwargs
    )


def discover_economic_model(*args: Any, **kwargs: Any) -> EconomicModelDiscoveryResult:
    """Discover a model through the provider-neutral evidence-only service."""

    return EconomicModelDiscoveryService().discover(*args, **kwargs)


EconomicStructureDiscoveryService = EconomicModelDiscoveryService
EconomicGraphDiscoveryService = EconomicModelDiscoveryService


__all__ = [
    "EconomicDiscoveryAudit",
    "EconomicDiscoveryRejection",
    "EconomicDiscoveryResult",
    "EconomicDiscoveryUnresolvedLeaf",
    "EconomicGraphDiscoveryService",
    "EconomicModelDiscoveryAudit",
    "EconomicModelDiscoveryRejection",
    "EconomicModelDiscoveryResult",
    "EconomicModelDiscoveryService",
    "EconomicModelDiscoveryUnresolvedLeaf",
    "EconomicStructureDiscoveryService",
    "construct_economic_model",
    "discover_economic_model",
]
