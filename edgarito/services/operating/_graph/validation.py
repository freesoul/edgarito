"""Structural and dimensional validation for economic operating graphs."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal

from edgarito.schemas.operating_graph import (
    EconomicMateriality,
    EconomicModel,
    EconomicNode,
    EconomicNodeType,
    EconomicRelationship,
    EconomicRelationshipType,
    EconomicUnitKind,
    GraphDiagnostic,
)


class GraphValidationReport:
    """Collected graph errors and warnings."""

    __slots__ = ("diagnostics",)

    def __init__(self, diagnostics: Iterable[GraphDiagnostic] = ()) -> None:
        self.diagnostics = tuple(diagnostics)

    @property
    def errors(self) -> tuple[GraphDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "error")

    @property
    def warnings(self) -> tuple[GraphDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity != "error")

    @property
    def valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> "GraphValidationReport":
        if not self.valid:
            raise GraphValidationError(self)
        return self

    def __bool__(self) -> bool:
        return self.valid


class GraphValidationError(ValueError):
    """Raised when a graph cannot be deterministically evaluated."""

    def __init__(self, report: GraphValidationReport) -> None:
        self.report = report
        details = "; ".join(f"{item.code}: {item.message}" for item in report.errors)
        super().__init__(details or "Economic graph validation failed")


def _error(code: str, message: str, *, node_id: str | None = None) -> GraphDiagnostic:
    return GraphDiagnostic(
        code=code, message=message, severity="error", node_id=node_id
    )


def _warning(code: str, message: str, *, node_id: str | None = None) -> GraphDiagnostic:
    return GraphDiagnostic(
        code=code, message=message, severity="warning", node_id=node_id
    )


_PERCENT_RATE_UNITS = frozenset(
    {"%", "percent", "percentage", "percentage_point", "percentage_points", "pp"}
)
_FRACTION_RATE_UNITS = frozenset({"ratio", "fraction", "decimal"})
_BASIS_POINT_RATE_UNITS = frozenset(
    {"bp", "bps", "basis_point", "basis_points"}
)
_MONETARY_UNIT_TOKENS = frozenset(
    {"usd", "eur", "gbp", "jpy", "cny", "cad", "aud", "chf", "currency", "monetary"}
)


def rate_unit_scale(unit: str) -> Decimal:
    """Return the scale needed to turn a declared rate into a fraction."""

    normalized = str(unit).strip().casefold().replace("-", "_").replace(" ", "_")
    dimensional_parts = normalized.split("/")
    if len(dimensional_parts) == 2 and all(
        any(token in part.split("_") for token in _MONETARY_UNIT_TOKENS)
        for part in dimensional_parts
    ):
        return Decimal(1)
    if normalized in _PERCENT_RATE_UNITS:
        return Decimal("0.01")
    if normalized in _FRACTION_RATE_UNITS:
        return Decimal(1)
    if normalized in _BASIS_POINT_RATE_UNITS:
        return Decimal("0.0001")
    raise ValueError(
        "Rate unit must explicitly be percent, percentage_points, pp, ratio, "
        "fraction, decimal, or basis points"
    )


def _rate_unit_diagnostic(node: EconomicNode, target: EconomicNode) -> GraphDiagnostic:
    return _error(
        "rate_unit_ambiguous",
        f"Rate source {node.node_id} for {target.node_id} has unsupported or "
        f"ambiguous unit {node.unit!r}; declare percent or an explicit fraction unit",
        node_id=target.node_id,
    )


class EconomicGraphValidator:
    """Validate references, formulas, dimensions, and the zero-lag DAG."""

    def check(self, model: EconomicModel) -> GraphValidationReport:
        diagnostics: list[GraphDiagnostic] = []
        nodes = model.nodes
        node_by_id: dict[str, EconomicNode] = {}
        duplicate_nodes: set[str] = set()
        for node in nodes:
            if node.node_id in node_by_id:
                duplicate_nodes.add(node.node_id)
            else:
                node_by_id[node.node_id] = node
        for node_id in sorted(duplicate_nodes):
            diagnostics.append(
                _error("duplicate_node_id", f"Duplicate node ID: {node_id}")
            )

        relationship_by_target: dict[str, EconomicRelationship] = {}
        duplicate_relationship_ids: set[str] = set()
        relationship_ids: set[str] = set()
        zero_lag_edges: dict[str, set[str]] = {node_id: set() for node_id in node_by_id}

        if model.revenue_root and model.revenue_root not in node_by_id:
            diagnostics.append(
                _error(
                    "unknown_revenue_root",
                    f"Revenue root is unknown: {model.revenue_root}",
                )
            )
        for root in model.business_roots:
            if root not in node_by_id:
                diagnostics.append(
                    _error("unknown_business_root", f"Business root is unknown: {root}")
                )

        for observation in model.observations:
            node = node_by_id.get(observation.node_id)
            if node is None:
                diagnostics.append(
                    _error(
                        "unknown_observation_node",
                        f"Observation references unknown node: {observation.node_id}",
                    )
                )
                continue
            if observation.scope is not None and observation.scope != node.scope:
                diagnostics.append(
                    _error(
                        "observation_scope_mismatch",
                        f"Observation scope {observation.scope!r} does not match "
                        f"node scope {node.scope!r}",
                        node_id=node.node_id,
                    )
                )
            if (
                observation.scope_id is not None
                and observation.scope_id != node.scope_id
            ):
                diagnostics.append(
                    _error(
                        "observation_scope_id_mismatch",
                        f"Observation scope_id {observation.scope_id!r} does not match "
                        f"node scope_id {node.scope_id!r}",
                        node_id=node.node_id,
                    )
                )
            if observation.unit != node.unit:
                diagnostics.append(
                    _error(
                        "observation_unit_mismatch",
                        f"Observation unit {observation.unit!r} does not match "
                        f"node unit {node.unit!r}",
                        node_id=node.node_id,
                    )
                )
            if observation.currency != node.currency:
                diagnostics.append(
                    _error(
                        "observation_currency_mismatch",
                        "Observation currency does not match node currency",
                        node_id=node.node_id,
                    )
                )

        for relationship in model.relationships:
            if relationship.fiscal_period != model.fiscal_period:
                diagnostics.append(
                    _error(
                        "relationship_period_mismatch",
                        f"Relationship period {relationship.fiscal_period!r} does not "
                        f"match model period {model.fiscal_period!r}",
                        node_id=relationship.target,
                    )
                )
            target = node_by_id.get(relationship.target)
            if target is None:
                diagnostics.append(
                    _error(
                        "unknown_relationship_target",
                        f"Relationship targets unknown node: {relationship.target}",
                    )
                )
            elif relationship.target in relationship_by_target:
                diagnostics.append(
                    _error(
                        "duplicate_producer",
                        f"Node has more than one producer: {relationship.target}",
                        node_id=relationship.target,
                    )
                )
            else:
                relationship_by_target[relationship.target] = relationship

            if relationship.relationship_id:
                if relationship.relationship_id in relationship_ids:
                    duplicate_relationship_ids.add(relationship.relationship_id)
                relationship_ids.add(relationship.relationship_id)

            source_nodes: list[EconomicNode] = []
            for edge in relationship.sources:
                source = node_by_id.get(edge.node_id)
                if source is None:
                    diagnostics.append(
                        _error(
                            "unknown_source_node",
                            f"Relationship references unknown source node: {edge.node_id}",
                            node_id=relationship.target,
                        )
                    )
                    continue
                source_nodes.append(source)
                if edge.fiscal_lag == 0 and target is not None:
                    zero_lag_edges.setdefault(source.node_id, set()).add(target.node_id)

            if target is not None:
                diagnostics.extend(
                    self._validate_relationship_shape(
                        relationship, target, source_nodes, model
                    )
                )

        for relationship_id in sorted(duplicate_relationship_ids):
            diagnostics.append(
                _error(
                    "duplicate_relationship_id",
                    f"Duplicate relationship ID: {relationship_id}",
                )
            )

        for input_node in nodes:
            if (
                input_node.node_type == EconomicNodeType.INPUT
                and input_node.node_id in relationship_by_target
            ):
                producer = relationship_by_target[input_node.node_id]
                if (
                    producer.historical_parameter_derivation
                    and producer.relationship_type
                    == EconomicRelationshipType.RATIO
                    and input_node.forecast_assumption_allowed
                ):
                    continue
                diagnostics.append(
                    _error(
                        "input_has_producer",
                        "Input nodes cannot also have a derived producer",
                        node_id=input_node.node_id,
                    )
                )

        cycle = _find_cycle(zero_lag_edges)
        if cycle:
            diagnostics.append(
                _error(
                    "zero_lag_cycle",
                    "Explicit zero-lag cycle: " + " -> ".join(cycle),
                )
            )

        # Materiality is informational, but an unknown materiality is useful to
        # consumers as a warning rather than a reason to discard a graph.
        for node in nodes:
            if node.materiality == EconomicMateriality.UNKNOWN:
                diagnostics.append(
                    _warning(
                        "unknown_materiality",
                        "Node materiality is unknown",
                        node_id=node.node_id,
                    )
                )
        return GraphValidationReport(diagnostics)

    def validate(
        self, model: EconomicModel, *, raise_on_error: bool = True
    ) -> GraphValidationReport:
        report = self.check(model)
        if raise_on_error:
            report.raise_for_errors()
        return report

    def assert_valid(self, model: EconomicModel) -> EconomicModel:
        self.validate(model)
        return model

    def _validate_relationship_shape(
        self,
        relationship: EconomicRelationship,
        target: EconomicNode,
        source_nodes: list[EconomicNode],
        model: EconomicModel,
    ) -> list[GraphDiagnostic]:
        diagnostics: list[GraphDiagnostic] = []
        kind = relationship.relationship_type
        count = len(relationship.sources)

        if (
            kind
            in {
                EconomicRelationshipType.IDENTITY,
                EconomicRelationshipType.LAG,
            }
            and count != 1
        ):
            diagnostics.append(
                _error(
                    "relationship_arity",
                    f"{kind.value} requires exactly one source",
                    node_id=target.node_id,
                )
            )
        if (
            kind
            in {
                EconomicRelationshipType.MULTIPLY,
                EconomicRelationshipType.RATIO,
                EconomicRelationshipType.GROWTH,
            }
            and count != 2
        ):
            diagnostics.append(
                _error(
                    "relationship_arity",
                    f"{kind.value} requires exactly two sources",
                    node_id=target.node_id,
                )
            )
        if kind == EconomicRelationshipType.RESIDUAL and count < 2:
            diagnostics.append(
                _error(
                    "relationship_arity",
                    "residual requires a reported total and at least one component",
                    node_id=target.node_id,
                )
            )

        # All additive formulas require signs to be present in the input
        # contract.  The SourceEdge default remains convenient for identity and
        # multiplicative formulas, but it is never silently used for a sum.
        if kind in {
            EconomicRelationshipType.ADD,
            EconomicRelationshipType.WEIGHTED_SUM,
            EconomicRelationshipType.RESIDUAL,
        }:
            for edge in relationship.sources:
                if "sign" not in edge.model_fields_set:
                    diagnostics.append(
                        _error(
                            "missing_explicit_sign",
                            "Additive relationships require an explicit source sign",
                            node_id=target.node_id,
                        )
                    )

        if kind == EconomicRelationshipType.LAG:
            if relationship.sources and relationship.sources[0].fiscal_lag <= 0:
                diagnostics.append(
                    _error(
                        "lag_not_declared",
                        "lag relationships require a positive fiscal_lag",
                        node_id=target.node_id,
                    )
                )
        elif any(edge.fiscal_lag < 0 for edge in relationship.sources):
            diagnostics.append(
                _error(
                    "negative_fiscal_lag",
                    "Fiscal lag cannot be negative",
                    node_id=target.node_id,
                )
            )

        if len(source_nodes) != len(relationship.sources):
            return diagnostics

        if relationship.historical_parameter_derivation:
            if kind != EconomicRelationshipType.RATIO:
                diagnostics.append(
                    _error(
                        "historical_parameter_derivation_type",
                        "Historical parameter derivation requires a ratio relationship",
                        node_id=target.node_id,
                    )
                )
            if target.node_type != EconomicNodeType.INPUT:
                diagnostics.append(
                    _error(
                        "historical_parameter_derivation_target",
                        "Historical parameter derivation target must be an INPUT node",
                        node_id=target.node_id,
                    )
                )
            elif not target.forecast_assumption_allowed:
                diagnostics.append(
                    _error(
                        "historical_parameter_derivation_assumption",
                        "Historical parameter derivation target must allow forecast assumptions",
                        node_id=target.node_id,
                    )
                )

        for edge, source in zip(relationship.sources, source_nodes, strict=True):
            if source.component_role == "contra_revenue" and edge.sign >= 0:
                diagnostics.append(
                    _error(
                        "contra_revenue_sign",
                        "Contra-revenue sources require a negative explicit sign",
                        node_id=target.node_id,
                    )
                )
            if not _scopes_match(target, source, relationship, model):
                diagnostics.append(
                    _error(
                        "scope_mismatch",
                        f"Source {source.node_id} scope does not match target "
                        f"{target.node_id}",
                        node_id=target.node_id,
                    )
                )

        if kind in {
            EconomicRelationshipType.IDENTITY,
            EconomicRelationshipType.ADD,
            EconomicRelationshipType.SUBTRACT,
            EconomicRelationshipType.LAG,
            EconomicRelationshipType.WEIGHTED_SUM,
            EconomicRelationshipType.RESIDUAL,
        }:
            for source in source_nodes:
                if not _same_units(target, source):
                    diagnostics.append(
                        _error(
                            "unit_mismatch",
                            f"Source {source.node_id} unit/currency does not match "
                            f"target {target.node_id}",
                            node_id=target.node_id,
                        )
                    )

        if kind == EconomicRelationshipType.GROWTH and len(source_nodes) == 2:
            base, rate = source_nodes
            if not _same_units(target, base):
                diagnostics.append(
                    _error(
                        "growth_base_unit_mismatch",
                        "Growth base must have the target unit and currency",
                        node_id=target.node_id,
                    )
                )
            if rate.unit_kind != EconomicUnitKind.RATE:
                diagnostics.append(
                    _error(
                        "growth_rate_kind",
                        "Growth rate source must have rate unit kind",
                        node_id=target.node_id,
                    )
                )
            else:
                try:
                    rate_unit_scale(rate.unit)
                except ValueError:
                    diagnostics.append(_rate_unit_diagnostic(rate, target))

        if kind == EconomicRelationshipType.MULTIPLY and len(source_nodes) == 2:
            left, right = source_nodes
            pair = {left.unit_kind, right.unit_kind}
            for source in source_nodes:
                if source.unit_kind != EconomicUnitKind.RATE:
                    continue
                try:
                    rate_unit_scale(source.unit)
                except ValueError:
                    diagnostics.append(_rate_unit_diagnostic(source, target))
            expected = {
                frozenset(
                    {EconomicUnitKind.MONETARY_PER_UNIT, EconomicUnitKind.COUNT}
                ): EconomicUnitKind.MONETARY,
                frozenset(
                    {EconomicUnitKind.MONETARY, EconomicUnitKind.RATE}
                ): EconomicUnitKind.MONETARY,
                frozenset(
                    {EconomicUnitKind.COUNT, EconomicUnitKind.RATE}
                ): EconomicUnitKind.COUNT,
            }
            if expected.get(frozenset(pair)) != target.unit_kind:
                diagnostics.append(
                    _error(
                        "multiply_unit_kind",
                        "Multiply sources and target have incompatible unit kinds",
                        node_id=target.node_id,
                    )
                )
            if target.unit_kind == EconomicUnitKind.MONETARY:
                monetary = (
                    left if left.unit_kind == EconomicUnitKind.MONETARY else right
                )
                per_unit = (
                    left
                    if left.unit_kind == EconomicUnitKind.MONETARY_PER_UNIT
                    else right
                )
                if (
                    monetary.currency != target.currency
                    and per_unit.currency != target.currency
                ):
                    diagnostics.append(
                        _error(
                            "multiply_currency_mismatch",
                            "Multiply monetary currency does not match target",
                            node_id=target.node_id,
                        )
                    )
                if (
                    left.unit_kind == EconomicUnitKind.MONETARY_PER_UNIT
                    and right.unit_kind == EconomicUnitKind.COUNT
                    and left.denominator_unit != right.unit
                ) or (
                    right.unit_kind == EconomicUnitKind.MONETARY_PER_UNIT
                    and left.unit_kind == EconomicUnitKind.COUNT
                    and right.denominator_unit != left.unit
                ):
                    diagnostics.append(
                        _error(
                            "multiply_denominator_unit",
                            "Count source must match the monetary-per-unit denominator",
                            node_id=target.node_id,
                        )
                    )

        if kind == EconomicRelationshipType.RATIO and len(source_nodes) == 2:
            numerator, denominator = source_nodes
            valid_per_unit = (
                numerator.unit_kind == EconomicUnitKind.MONETARY
                and denominator.unit_kind == EconomicUnitKind.COUNT
                and target.unit_kind == EconomicUnitKind.MONETARY_PER_UNIT
            )
            valid_rate = (
                numerator.unit_kind == denominator.unit_kind
                and target.unit_kind == EconomicUnitKind.RATE
            )
            if not (valid_per_unit or valid_rate):
                diagnostics.append(
                    _error(
                        "ratio_unit_kind",
                        "Ratio sources and target have incompatible unit kinds",
                        node_id=target.node_id,
                    )
                )
            if valid_per_unit and target.denominator_unit != denominator.unit:
                diagnostics.append(
                    _error(
                        "ratio_denominator_unit",
                        "Ratio denominator unit must match target denominator_unit",
                        node_id=target.node_id,
                    )
                )
            if valid_per_unit and target.currency != numerator.currency:
                diagnostics.append(
                    _error(
                        "ratio_currency_mismatch",
                        "Monetary-per-unit ratio currency must match numerator",
                        node_id=target.node_id,
                    )
                )
            if valid_rate and target.currency is not None:
                diagnostics.append(
                    _error(
                        "ratio_rate_currency",
                        "Rate ratio targets must not declare a currency",
                        node_id=target.node_id,
                    )
                )
            if (
                valid_rate
                and numerator.currency != denominator.currency
                and numerator.currency is not None
                and denominator.currency is not None
            ):
                diagnostics.append(
                    _error(
                        "ratio_currency_mismatch",
                        "Ratio monetary currencies must match",
                        node_id=target.node_id,
                    )
                )

        return diagnostics


def _same_units(target: EconomicNode, source: EconomicNode) -> bool:
    return (
        target.unit == source.unit
        and target.unit_kind == source.unit_kind
        and target.currency == source.currency
        and target.denominator_unit == source.denominator_unit
    )


def _scopes_match(
    target: EconomicNode,
    source: EconomicNode,
    relationship: EconomicRelationship,
    model: EconomicModel,
) -> bool:
    if target.scope == source.scope and target.scope_id == source.scope_id:
        return True
    # A consolidated aggregate may explicitly collect declared business roots.
    # This is the only intentional scope boundary; all other formulas require
    # exact scope and scope_id equality.
    return (
        target.node_type == EconomicNodeType.AGGREGATE
        and target.scope == "consolidated"
        and source.node_id in model.business_roots
        and relationship.relationship_type
        in {EconomicRelationshipType.ADD, EconomicRelationshipType.WEIGHTED_SUM}
    )


def _find_cycle(edges: dict[str, set[str]]) -> tuple[str, ...] | None:
    visiting: set[str] = set()
    visited: set[str] = set()
    path: list[str] = []

    def visit(node_id: str) -> tuple[str, ...] | None:
        if node_id in visiting:
            try:
                start = path.index(node_id)
            except ValueError:
                start = 0
            return tuple((*path[start:], node_id))
        if node_id in visited:
            return None
        visiting.add(node_id)
        path.append(node_id)
        for target in sorted(edges.get(node_id, ())):
            cycle = visit(target)
            if cycle:
                return cycle
        path.pop()
        visiting.remove(node_id)
        visited.add(node_id)
        return None

    for node_id in sorted(edges):
        cycle = visit(node_id)
        if cycle:
            return cycle
    return None


def inspect_graph(model: EconomicModel) -> GraphValidationReport:
    """Return a report without raising for invalid graphs."""

    return EconomicGraphValidator().check(model)


def validate_graph(
    model: EconomicModel, *, raise_on_error: bool = True
) -> GraphValidationReport:
    """Validate a graph, raising ``GraphValidationError`` by default."""

    return EconomicGraphValidator().validate(model, raise_on_error=raise_on_error)


def validate_model(
    model: EconomicModel, *, raise_on_error: bool = True
) -> GraphValidationReport:
    return validate_graph(model, raise_on_error=raise_on_error)


validate_economic_graph = validate_graph
validate_economic_model = validate_model
OperatingGraphValidator = EconomicGraphValidator
EconomicDAGValidator = EconomicGraphValidator


__all__ = [
    "EconomicGraphValidator",
    "EconomicDAGValidator",
    "GraphValidationError",
    "GraphValidationReport",
    "OperatingGraphValidator",
    "inspect_graph",
    "validate_economic_graph",
    "validate_economic_model",
    "validate_graph",
    "validate_model",
]
