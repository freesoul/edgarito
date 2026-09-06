"""Deterministic, provider-neutral valuation factor graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping, Optional

from edgarito.services.valuation.factors.contracts import (
    CacheState,
    FactorEstimate,
    FactorKey,
    FactorMateriality,
    FactorPriority,
    FactorRange,
    FactorResolutionStatus,
    StopReason,
)


def _key(value: FactorKey | FactorGraphNode) -> FactorKey:
    return value.key if isinstance(value, FactorGraphNode) else value


@dataclass(frozen=True)
class FactorGraphNode:
    key: FactorKey
    status: FactorResolutionStatus = FactorResolutionStatus.UNRESOLVED
    estimate: Optional[FactorEstimate] = None
    depth: int = 0
    priority: FactorPriority = FactorPriority.NORMAL
    materiality: FactorMateriality = FactorMateriality.MEDIUM
    uncertainty: FactorRange | None = None
    stop_reason: Optional[StopReason] = None
    resolver: Optional[str] = None
    cache_state: Optional[CacheState] = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", FactorResolutionStatus(self.status))
        object.__setattr__(self, "priority", FactorPriority(self.priority))
        object.__setattr__(self, "materiality", FactorMateriality(self.materiality))
        if self.cache_state is not None:
            object.__setattr__(self, "cache_state", CacheState(self.cache_state))
        if self.stop_reason is not None:
            object.__setattr__(self, "stop_reason", StopReason(self.stop_reason))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def fingerprint(self) -> Optional[str]:
        return self.estimate.fingerprint if self.estimate is not None else None


@dataclass(frozen=True)
class FactorGraphEdge:
    """An edge is stored as dependency -> consumer."""

    dependency: FactorKey
    consumer: FactorKey
    role: str = "dependency"
    rationale: str = ""
    materiality: FactorMateriality = FactorMateriality.MEDIUM
    weight: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        object.__setattr__(self, "materiality", FactorMateriality(self.materiality))
        object.__setattr__(self, "weight", Decimal(str(self.weight)))

    @property
    def dependency_key(self) -> FactorKey:
        return self.dependency

    @property
    def consumer_key(self) -> FactorKey:
        return self.consumer


@dataclass(frozen=True)
class FactorGraphSnapshot:
    nodes: Mapping[str, FactorGraphNode]
    edges: tuple[FactorGraphEdge, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", MappingProxyType(dict(self.nodes)))

    def node(self, key: FactorKey) -> Optional[FactorGraphNode]:
        return self.nodes.get(key.digest)

    def ancestors(self, key: FactorKey) -> tuple[FactorKey, ...]:
        return _ancestors(self.nodes, self.edges, key)

    def descendants(self, key: FactorKey) -> tuple[FactorKey, ...]:
        return _descendants(self.nodes, self.edges, key)

    def dependency_paths(self, key: FactorKey) -> tuple[tuple[FactorKey, ...], ...]:
        return _dependency_paths(self.nodes, self.edges, key)

    def explain(self, key: FactorKey):
        from edgarito.services.valuation.factors.explain import explain

        return explain(self, key)


def _sorted_keys(keys):
    return tuple(sorted(keys, key=lambda item: (item.digest, item.semantic_id)))


def _adjacency(edges, *, incoming: bool):
    result: dict[str, list[FactorKey]] = {}
    for edge in edges:
        source, target = (
            (edge.consumer, edge.dependency)
            if incoming
            else (edge.dependency, edge.consumer)
        )
        result.setdefault(source.digest, []).append(target)
    return result


def _ancestors(nodes, edges, key):
    adjacency = _adjacency(edges, incoming=True)
    found: dict[str, FactorKey] = {}
    stack = list(adjacency.get(key.digest, ()))
    while stack:
        current = stack.pop()
        if current.digest in found:
            continue
        found[current.digest] = current
        stack.extend(adjacency.get(current.digest, ()))
    return _sorted_keys(found.values())


def _descendants(nodes, edges, key):
    adjacency = _adjacency(edges, incoming=False)
    found: dict[str, FactorKey] = {}
    stack = list(adjacency.get(key.digest, ()))
    while stack:
        current = stack.pop()
        if current.digest in found:
            continue
        found[current.digest] = current
        stack.extend(adjacency.get(current.digest, ()))
    return _sorted_keys(found.values())


def _dependency_paths(nodes, edges, key):
    adjacency = _adjacency(edges, incoming=True)

    def walk(current: FactorKey, prefix: tuple[FactorKey, ...]):
        children = _sorted_keys(adjacency.get(current.digest, ()))
        if not children:
            return (prefix,)
        paths = []
        for child in children:
            if child.digest in {item.digest for item in prefix}:
                continue
            paths.extend(walk(child, prefix + (child,)))
        return tuple(paths) or (prefix,)

    return walk(key, (key,))


class ValuationFactorGraph:
    """Mutable builder with canonical-key deduplication and pre-commit cycle checks."""

    def __init__(self) -> None:
        self._nodes: dict[str, FactorGraphNode] = {}
        self._edges: list[FactorGraphEdge] = []

    @property
    def nodes(self) -> Mapping[str, FactorGraphNode]:
        return MappingProxyType(self._nodes)

    @property
    def edges(self) -> tuple[FactorGraphEdge, ...]:
        return tuple(self._edges)

    def __len__(self) -> int:
        return len(self._nodes)

    def add_node(
        self,
        key: FactorKey | FactorGraphNode,
        **updates: Any,
    ) -> FactorGraphNode:
        node = key if isinstance(key, FactorGraphNode) else FactorGraphNode(key=key)
        existing = self._nodes.get(node.key.digest)
        if existing is not None:
            node = replace(existing, **updates) if updates else existing
        elif updates:
            node = replace(node, **updates)
        self._nodes[node.key.digest] = node
        return node

    def update_node(self, key: FactorKey, **updates: Any) -> FactorGraphNode:
        return self.add_node(key, **updates)

    def get_node(self, key: FactorKey) -> Optional[FactorGraphNode]:
        return self._nodes.get(key.digest)

    node = get_node

    def add_edge(
        self,
        dependency: FactorKey,
        consumer: FactorKey,
        *,
        role: str = "dependency",
        rationale: str = "",
        materiality: FactorMateriality = FactorMateriality.MEDIUM,
        weight: Decimal | int | str = Decimal("1"),
    ) -> FactorGraphEdge:
        dependency = _key(dependency)
        consumer = _key(consumer)
        materiality = FactorMateriality(materiality)
        edge = FactorGraphEdge(
            dependency=dependency,
            consumer=consumer,
            role=str(role),
            rationale=str(rationale),
            materiality=materiality,
            weight=Decimal(str(weight)),
        )
        if edge in self._edges:
            return edge
        # Adding dependency -> consumer is cyclic iff consumer already reaches
        # dependency. Perform this check before touching the edge list.
        if dependency.digest == consumer.digest or any(
            item.digest == dependency.digest
            for item in self.descendants(consumer)
        ):
            raise ValueError("factor graph cycle detected")
        if dependency.digest not in self._nodes:
            self.add_node(dependency)
        if consumer.digest not in self._nodes:
            self.add_node(consumer)
        self._edges.append(edge)
        return edge

    def ancestors(self, key: FactorKey) -> tuple[FactorKey, ...]:
        return _ancestors(self._nodes, self._edges, _key(key))

    def descendants(self, key: FactorKey) -> tuple[FactorKey, ...]:
        return _descendants(self._nodes, self._edges, _key(key))

    def dependency_paths(self, key: FactorKey) -> tuple[tuple[FactorKey, ...], ...]:
        return _dependency_paths(self._nodes, self._edges, _key(key))

    def explain(self, key: FactorKey):
        from edgarito.services.valuation.factors.explain import explain

        return explain(self, _key(key))

    def snapshot(self) -> FactorGraphSnapshot:
        return FactorGraphSnapshot(
            nodes=dict(self._nodes),
            edges=tuple(self._edges),
        )

    freeze = snapshot
    immutable_snapshot = snapshot


FactorGraph = ValuationFactorGraph
GraphNode = FactorGraphNode
GraphEdge = FactorGraphEdge


__all__ = [
    "FactorGraphEdge",
    "FactorGraph",
    "FactorGraphNode",
    "FactorGraphSnapshot",
    "GraphEdge",
    "GraphNode",
    "ValuationFactorGraph",
]
