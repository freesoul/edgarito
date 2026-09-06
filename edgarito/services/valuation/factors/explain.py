"""Stable graph explanations for audit output and focused tests."""

from __future__ import annotations

from dataclasses import dataclass

from edgarito.services.valuation.factors.graph import (
    FactorGraphEdge,
    FactorGraphNode,
    FactorGraphSnapshot,
    ValuationFactorGraph,
)


@dataclass(frozen=True)
class FactorExplanation:
    key: object
    node: FactorGraphNode | None
    paths: tuple[tuple[object, ...], ...]
    edges: tuple[FactorGraphEdge, ...]
    ancestors: tuple[object, ...] = ()
    descendants: tuple[object, ...] = ()

    @property
    def dependency_paths(self) -> tuple[tuple[object, ...], ...]:
        """The stable root-to-leaf paths included in the explanation."""

        return self.paths

    @property
    def text(self) -> str:
        status = self.node.status.value if self.node is not None else "unknown"
        lines = [f"{self.key.digest}: {status}"]
        for path in self.paths:
            lines.append(" -> ".join(item.digest for item in path))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.text


def explain(
    graph: ValuationFactorGraph | FactorGraphSnapshot, key
) -> FactorExplanation:
    node = (
        graph.get_node(key)
        if isinstance(graph, ValuationFactorGraph)
        else graph.node(key)
    )
    ancestors = graph.ancestors(key)
    descendants = graph.descendants(key)
    paths = graph.dependency_paths(key)
    edges = tuple(
        sorted(
            (
                edge
                for edge in graph.edges
                if edge.consumer.digest == key.digest
                or edge.dependency.digest in {item.digest for item in ancestors}
            ),
            key=lambda edge: (
                edge.consumer.digest,
                edge.dependency.digest,
                edge.role,
                edge.rationale,
            ),
        )
    )
    return FactorExplanation(
        key=key,
        node=node,
        paths=paths,
        edges=edges,
        ancestors=ancestors,
        descendants=descendants,
    )


def explain_paths(graph, key):
    return explain(graph, key).paths


__all__ = ["FactorExplanation", "explain", "explain_paths"]
