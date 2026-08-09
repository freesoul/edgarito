"""Small helpers used to build detached, JSON-safe export snapshots."""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Mapping, Sequence, Set
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel


class _FrozenDict(dict):
    """A JSON-object-compatible dictionary that rejects all mutation."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("snapshot mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def snapshot(value: Any) -> Any:
    """Return a detached, deterministic representation of an arbitrary value.

    Valuation details deliberately remain generic because specialized adapters can
    return different Pydantic models.  This helper keeps those details useful for a
    later JSON export without retaining references to mutable domain objects.
    """

    if isinstance(value, BaseModel):
        return snapshot(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return snapshot(value.value)
    if isinstance(value, Mapping):
        items = tuple((snapshot(key), snapshot(item)) for key, item in value.items())
        for key, _ in items:
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise TypeError(
                    "snapshot mapping keys must be JSON-compatible primitives; "
                    f"got {type(key).__name__}"
                )
        return _FrozenDict(sorted(items, key=lambda item: repr(item[0])))
    if isinstance(value, Set):
        values = tuple(snapshot(item) for item in value)
        return tuple(sorted(values, key=repr))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(snapshot(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return snapshot(dataclasses.asdict(value))
    if isinstance(value, (str, int, float, bool, Decimal, datetime.date)):
        return value
    if value is None:
        return None

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return snapshot(model_dump(mode="python"))
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return snapshot(attributes)
    raise TypeError(f"Cannot snapshot unsupported value of type {type(value).__name__}")


def max_period_end(values: list[datetime.date | None]) -> datetime.date | None:
    dates = [value for value in values if value is not None]
    return max(dates) if dates else None
