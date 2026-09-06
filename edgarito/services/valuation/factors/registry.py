"""Deterministic registries for factor resolvers and decomposers."""

from __future__ import annotations

from typing import Iterable, Optional


def _id(value) -> str:
    return str(getattr(value, "resolver_id", getattr(value, "decomposer_id", type(value).__name__)))


def _priority(value) -> int | float:
    """Return explicit registry priority, with a stable zero default."""

    priority = getattr(value, "priority", 0)
    if isinstance(priority, bool) or not isinstance(priority, (int, float)):
        raise TypeError(f"registry priority must be numeric, got {priority!r}")
    return priority


def _ordering_key(value):
    # Higher priorities are considered first. IDs only break ties.
    return (-_priority(value), _id(value))


class FactorResolverRegistry:
    def __init__(self, resolvers: Iterable[object] = ()):
        self._resolvers: list[object] = []
        for resolver in resolvers:
            self.register(resolver)

    @property
    def resolvers(self) -> tuple[object, ...]:
        return tuple(self._resolvers)

    def register(self, resolver: object) -> object:
        resolver_id = _id(resolver)
        self._resolvers = [item for item in self._resolvers if _id(item) != resolver_id]
        self._resolvers.append(resolver)
        self._resolvers.sort(key=_ordering_key)
        return resolver

    def candidates(self, request, context=None, *, derived: Optional[bool] = None):
        result = []
        for resolver in self._resolvers:
            is_derived = bool(getattr(resolver, "requires_dependencies", False))
            if derived is not None and is_derived != derived:
                continue
            can_resolve = getattr(resolver, "can_resolve", None)
            if can_resolve is None:
                result.append(resolver)
                continue
            try:
                allowed = can_resolve(request, context)
            except TypeError:
                allowed = can_resolve(request)
            if allowed:
                result.append(resolver)
        return tuple(result)

    def get(self, resolver_id: str):
        for resolver in self._resolvers:
            if _id(resolver) == resolver_id:
                return resolver
        return None

    resolver_for = get
    register_resolver = register


class FactorDecomposerRegistry:
    def __init__(self, decomposers: Iterable[object] = ()):
        self._decomposers: list[object] = []
        for decomposer in decomposers:
            self.register(decomposer)

    @property
    def decomposers(self) -> tuple[object, ...]:
        return tuple(self._decomposers)

    def register(self, decomposer: object) -> object:
        decomposer_id = _id(decomposer)
        self._decomposers = [
            item for item in self._decomposers if _id(item) != decomposer_id
        ]
        self._decomposers.append(decomposer)
        self._decomposers.sort(key=_ordering_key)
        return decomposer

    def candidates(self, request, context=None):
        result = []
        for decomposer in self._decomposers:
            can_decompose = getattr(decomposer, "can_decompose", None)
            if can_decompose is None:
                result.append(decomposer)
                continue
            try:
                allowed = can_decompose(request, context)
            except TypeError:
                allowed = can_decompose(request)
            if allowed:
                result.append(decomposer)
        return tuple(result)

    decomposer_for = candidates
    register_decomposer = register


class FactorRegistry:
    def __init__(self, resolvers=(), decomposers=()):
        self.resolvers = (
            resolvers
            if isinstance(resolvers, FactorResolverRegistry)
            else FactorResolverRegistry(resolvers)
        )
        self.decomposers = (
            decomposers
            if isinstance(decomposers, FactorDecomposerRegistry)
            else FactorDecomposerRegistry(decomposers)
        )


# Short public names are useful when the package is used as a small framework.
ResolverRegistry = FactorResolverRegistry
DecomposerRegistry = FactorDecomposerRegistry


__all__ = [
    "DecomposerRegistry",
    "FactorDecomposerRegistry",
    "FactorRegistry",
    "FactorResolverRegistry",
    "ResolverRegistry",
]
