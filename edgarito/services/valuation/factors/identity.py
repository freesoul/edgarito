"""Canonical identity helpers for valuation factors.

The identity layer deliberately contains only conservative normalizations.  In
particular, it does not attempt to infer that two products, markets, or
geographies are economically interchangeable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import unicodedata
from decimal import Decimal
from enum import Enum
from typing import Any

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def canonicalize_token(value: str, *, field: str = "value") -> str:
    """Return a stable snake-case token without semantic synonym inference."""

    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized:
        raise ValueError(f"{field} cannot be blank")
    normalized = _CAMEL_BOUNDARY.sub("_", normalized)
    normalized = _WHITESPACE.sub("_", normalized.casefold())
    normalized = _NON_WORD.sub("_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError(f"{field} cannot be blank")
    return normalized


def canonicalize_geography(value: str) -> str:
    """Canonicalize only unambiguous global and United States aliases."""

    token = canonicalize_token(value, field="geography")
    if token in {"global", "worldwide", "world_wide"}:
        return "global"
    if token in {
        "us",
        "usa",
        "u_s",
        "u_s_a",
        "united_states",
        "united_states_of_america",
    }:
        return "us"
    return token


# This list is intentionally finite.  Product names (including lithium
# carbonate and lithium hydroxide) are not units and never enter this map.
_UNIT_ALIASES = {
    "%": "percent",
    "pct": "percent",
    "percentage": "percent",
    "percentage_point": "percentage_point",
    "percentage_points": "percentage_point",
    "percent": "percent",
    "multiple": "multiple",
    "times": "multiple",
    "x": "multiple",
    "share": "share",
    "shares": "share",
    "kg": "kilogram",
    "kilogram": "kilogram",
    "kilograms": "kilogram",
    "lb": "pound",
    "lbs": "pound",
    "pound": "pound",
    "pounds": "pound",
    "t": "metric_tonne",
    # The unqualified short form is ambiguous; do not infer metric tonnes.
    "ton": "ton",
    "tonne": "metric_tonne",
    "tons": "ton",
    "tonnes": "metric_tonne",
    "metric_ton": "metric_tonne",
    "metric_tonne": "metric_tonne",
    "metric_tons": "metric_tonne",
    "metric_tonnes": "metric_tonne",
    "usd/t": "usd_per_metric_tonne",
    "usd/ton": "usd_per_ton",
    "usd/tons": "usd_per_ton",
    "usd/tonne": "usd_per_metric_tonne",
    "usd / t": "usd_per_metric_tonne",
    "usd / ton": "usd_per_ton",
    "usd / tons": "usd_per_ton",
    "usd / tonne": "usd_per_metric_tonne",
    "usd per ton": "usd_per_ton",
    "usd per tons": "usd_per_ton",
    "usd per tonne": "usd_per_metric_tonne",
    "usd_per_ton": "usd_per_ton",
    "usd_per_tons": "usd_per_ton",
    "usd_per_tonne": "usd_per_metric_tonne",
    "usd_per_metric_ton": "usd_per_metric_tonne",
    "usd_per_metric_tonne": "usd_per_metric_tonne",
}


def canonicalize_unit(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("unit must be a string")
    raw = unicodedata.normalize("NFKC", value).strip().casefold()
    if not raw:
        raise ValueError("unit cannot be blank")
    compact = _WHITESPACE.sub(" ", raw)
    direct = _UNIT_ALIASES.get(compact)
    if direct is not None:
        return direct
    return _UNIT_ALIASES.get(
        canonicalize_token(raw, field="unit"), canonicalize_token(raw, field="unit")
    )


def canonicalize_currency(value: str) -> str:
    normalized = value.strip().upper() if isinstance(value, str) else value
    if not isinstance(normalized, str) or not re.fullmatch(r"[A-Z]{3}", normalized):
        raise ValueError("currency must be a three-letter ISO code")
    return normalized


def canonical_json(value: Any) -> str:
    """Serialize identity material deterministically and without cache timing."""

    def encode(item: Any) -> Any:
        if isinstance(item, Enum):
            return encode(item.value)
        if isinstance(item, Decimal):
            if not item.is_finite():
                raise ValueError("non-finite decimals cannot be canonicalized")
            return format(item, "f")
        if isinstance(item, (dt.datetime, dt.date, dt.time)):
            return item.isoformat()
        if isinstance(item, dict):
            return {str(key): encode(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (set, frozenset)):
            encoded = [encode(part) for part in item]
            return sorted(encoded, key=lambda part: canonical_json(part))
        if isinstance(item, (list, tuple)):
            return [encode(part) for part in item]
        if hasattr(item, "model_dump"):
            return encode(item.model_dump(mode="python", by_alias=False))
        return item

    return json.dumps(
        encode(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "canonical_json",
    "canonicalize_currency",
    "canonicalize_geography",
    "canonicalize_token",
    "canonicalize_unit",
    "stable_digest",
]
