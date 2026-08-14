"""Small helpers for explicit use-case dependency resolution.

The CLI facade passes its module as a compatibility context.  Use cases read
overrides from that context when one is supplied, but never copy those values
into their own module namespace.  This keeps direct imports independently
callable and avoids process-wide state leaking between commands.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def dependency(context: object | None, name: str, default: Any) -> Any:
    """Return a context override, falling back to the stable default.

    ``DependencySnapshot`` is the normal context passed from the CLI facade,
    but accepting mappings and simple namespaces keeps the focused use cases
    convenient to call directly in tests and by host applications.
    """

    if context is None:
        return default
    if isinstance(context, Mapping):
        return context.get(name, default)
    getter = getattr(context, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(context, name, default)


def call_with_context(callable_, *args: Any, context: object | None = None, **kwargs: Any):
    """Call a dependency-aware seam without breaking old narrow fakes."""

    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return callable_(*args, context=context, **kwargs)
    parameters = signature.parameters
    if "context" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return callable_(*args, context=context, **kwargs)
    if "dependencies" in parameters:
        return callable_(*args, dependencies=context, **kwargs)
    return callable_(*args, **kwargs)


@dataclass(frozen=True)
class DependencySnapshot:
    """An immutable per-invocation set of dependency overrides.

    Values themselves are intentionally not copied: collaborators such as
    service classes and test fakes are the dependencies being selected.  The
    mapping that binds their names is copied and frozen so an invocation never
    observes a later facade monkeypatch and no shared application module needs
    to be mutated.
    """

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            MappingProxyType(dict(self.values)),
        )

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)


@dataclass(frozen=True)
class ValuationDependencyContext:
    """Resolve valuation collaborators from an optional compatibility facade.

    Valuation used to be implemented in ``cli.__main__`` and therefore every
    imported service was a supported monkeypatch seam.  The focused use-case
    modules must not import the facade (that would create a cycle), so the
    application passes its module as ``source`` and this small typed wrapper
    performs all collaborator lookups in one place.
    """

    source: object | None = None

    def resolve(self, name: str, default: Any) -> Any:
        """Return a facade override, or the focused module's default."""

        return dependency(self.source, name, default)


__all__ = [
    "DependencySnapshot",
    "ValuationDependencyContext",
    "call_with_context",
    "dependency",
]
