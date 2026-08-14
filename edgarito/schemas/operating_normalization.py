"""Pure normalization helpers for the operating evidence contracts.

The operating schema models use these helpers at their validation boundaries,
but this module does not define or eagerly import the contracts.  Operating
unit configuration remains a lazy dependency so importing this module cannot
eagerly load the operating configuration tables.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgarito.schemas.operating import OperatingArchetype


_MIN_FISCAL_YEAR = 1900
_MAX_FISCAL_YEAR = 2200
_CONFIDENCE_LEVELS = {"high", "medium", "low"}
_OPERATING_PERIODS = {"FY", "FQ", "YTD", "LTM"}


def _normalize_required_text(value: str, label: str) -> str:
    normalized = str(getattr(value, "value", value)).strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized


def canonical_operating_segment_id(value: str | None) -> str | None:
    """Return a filing-neutral segment identifier.

    Filings vary between labels such as ``"Platform"``, ``"platform
    segment"`` and ``"Platform Business"``.  Canonicalization removes only
    generic reporting words; it does not contain issuer or ticker aliases.
    """

    if value is None:
        return None
    normalized = _canonical_segment_text(value)
    if not normalized:
        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            unicodedata.normalize("NFKD", str(value)).casefold(),
        ).strip()
    if not normalized:
        raise ValueError("Operating segment identifier cannot be blank")
    return normalized.replace(" ", "_")


def canonical_operating_segment_identity(
    segment_id: str, name: str | None = None
) -> tuple[str, str]:
    """Normalize a segment's ID/name pair consistently across filings."""

    raw_display = name or segment_id
    display = _canonical_segment_display(raw_display) or str(raw_display).strip()
    canonical_id = canonical_operating_segment_id(display)
    if canonical_id is None:
        raise ValueError("Operating segment identifier cannot be blank")
    return canonical_id, display


def normalize_operating_fiscal_period(value: str | None) -> str:
    """Normalize supported evidence periods to FY, FQ, YTD, or LTM.

    Quarter labels are retained in ``period_key`` by the input coercion helper,
    while the period class remains deliberately small.  H1/H2 and other
    incompatible periods are rejected rather than silently mixed.
    """

    if value is None:
        return "FY"
    normalized = (
        str(getattr(value, "value", value))
        .strip()
        .casefold()
        .replace("-", " ")
        .replace("_", " ")
    )
    compact = re.sub(r"\s+", " ", normalized)
    if compact in {
        "fy",
        "annual",
        "annually",
        "full year",
        "fiscal year",
        "year",
    }:
        return "FY"
    if re.fullmatch(r"(?:19|20|21|22)\d{2}", compact):
        return "FY"
    if re.fullmatch(r"fy\s*\d{4}", compact) or re.fullmatch(
        r"fiscal year\s*\d{4}", compact
    ):
        return "FY"
    if compact in {"fq", "quarter", "quarterly", "fiscal quarter"}:
        return "FQ"
    if re.fullmatch(r"q[1-4]", compact):
        return "FQ"
    if re.fullmatch(r"q[1-4]\s+(?:19|20|21|22)\d{2}", compact):
        return "FQ"
    if compact in {
        "first quarter",
        "second quarter",
        "third quarter",
        "fourth quarter",
    }:
        return "FQ"
    if compact in {"ytd", "year to date", "year-to-date"}:
        return "YTD"
    if compact in {"ltm", "last twelve months", "trailing twelve months", "ttm"}:
        return "LTM"
    if re.search(r"three months? ended|three months? ending", compact):
        return "FQ"
    if re.search(r"(?:six|nine) months? ended|(?:six|nine) months? ending", compact):
        return "YTD"
    raise ValueError(
        "Operating evidence period must be compatible FY, FQ, or YTD; "
        f"received {value!r}"
    )


def operating_periods_compatible(
    left_period: str | None,
    right_period: str | None,
    left_period_key: str | None = None,
    right_period_key: str | None = None,
) -> bool:
    """Return whether two observations can participate in one derivation."""

    try:
        left = normalize_operating_fiscal_period(left_period)
        right = normalize_operating_fiscal_period(right_period)
    except ValueError:
        return False
    if left != right:
        return False
    if left == "FQ" and bool(left_period_key) != bool(right_period_key):
        # A bare quarter class does not identify the same quarter as Q1/Q2/etc.
        return False
    if left_period_key and right_period_key:
        return _period_key(left_period_key) == _period_key(right_period_key)
    return True


def normalize_operating_unit(
    unit: str, driver_id: str | None = None
) -> tuple[str, Decimal]:
    """Normalize common reporting scales without guessing from numeric size."""

    raw = _normalize_required_text(unit, "Operating observation unit")
    folded = (
        unicodedata.normalize("NFKC", raw)
        .casefold()
        .replace("$", "usd")
        .replace("€", "eur")
        .replace("£", "gbp")
    )
    folded = re.sub(r"\b(us dollars?|u\.s\. dollars?)\b", "usd", folded)
    scale = Decimal(1)
    for label, multiplier in sorted(
        ((item.label, item.multiplier) for item in _operating_unit_scale_aliases()),
        key=lambda item: -len(item[0]),
    ):
        if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", folded):
            scale *= multiplier
            folded = re.sub(
                rf"\b{re.escape(label)}\b", " ", folded, flags=re.IGNORECASE
            )
            break
    folded = folded.replace("per cent", "percent")
    folded = re.sub(r"\b(?:currency|monetary)\b", "currency", folded)
    folded = re.sub(r"\b(?:dollars?)\b", "usd", folded)
    folded = re.sub(r"\b(?:users?|subscribers?)\b", "users", folded)
    folded = re.sub(r"\b(?:vehicle|vehicles|car|cars|unit|units)\b", "units", folded)
    folded = re.sub(r"\b(?:stores?|locations?)\b", "locations", folded)
    folded = re.sub(r"\s+", " ", folded).strip()
    folded = re.sub(r"\s*/\s*", "/", folded)
    folded = re.sub(r"\s+per\s+", "/", folded)
    folded = folded.replace(" ", "_")
    if not folded:
        normalized_driver = _canonical_metric_name(driver_id or "")
        folded = "currency" if "revenue" in normalized_driver else "unit"
    return folded, scale


def _operating_unit_scale_aliases():
    """Load unit scale aliases lazily to avoid the schemas/config import cycle."""

    from edgarito.config.operating import OPERATING_UNITS

    return OPERATING_UNITS.scale_aliases


def operating_units_compatible(expected: str, actual: str) -> bool:
    """Return whether two declared operating units have the same dimension.

    Reporting scales are intentionally ignored because observations are already
    normalized to base units.  Ratios and explicit percentage units are the same
    rate dimension; currency symbols/codes are interchangeable with generic
    currency in a formula declaration.
    """

    if str(expected).strip().casefold() in {"unit", "unspecified"} or str(
        actual
    ).strip().casefold() in {"unit", "unspecified"}:
        return True
    expected_unit, _ = normalize_operating_unit(expected)
    actual_unit, _ = normalize_operating_unit(actual)
    if expected_unit == actual_unit:
        return True
    if _unit_dimension(expected_unit) == _unit_dimension(actual_unit):
        return True
    return False


def _unit_dimension(value: str) -> str:
    normalized = value.casefold()
    if normalized in {"ratio", "percent", "percentage", "bps", "bp"}:
        return "rate"
    normalized = re.sub(
        r"\b(?:usd|eur|gbp|jpy|cny|cad|aud|chf)\b", "currency", normalized
    )
    return normalized


def _canonical_segment_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().replace("&", " and ")
    text = re.sub(
        r"\b(?:reportable|operating|business|segment|division|group|unit)\b",
        " ",
        text,
    )
    text = re.sub(r"\band\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    tokens = set(text.split())
    # These are deliberately exact token aliases. Similar-looking labels such
    # as vehicle logistics or energy services must remain separate segments.
    if tokens <= {
        "automotive",
        "automobiles",
        "vehicle",
        "vehicles",
        "car",
        "cars",
        "revenue",
        "revenues",
        "sales",
    }:
        return "automotive"
    if tokens <= {
        "energy",
        "storage",
        "generation",
        "solar",
        "megapack",
        "powerwall",
        "deployment",
        "deployments",
        "revenue",
        "revenues",
        "sales",
    }:
        return "energy"
    return text


def _canonical_segment_display(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(
        r"\s+(?:reportable|operating)?\s*(?:segment|business|division|group|unit)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" -,:;")
    return re.sub(r"\s+", " ", text)


def _canonical_metric_name(value: str) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _period_key(value: str) -> str:
    return re.sub(r"\s+", "", str(value).strip().casefold())


def _coerce_operating_period_fields(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    data = dict(value)
    if "fiscal_period" not in data:
        for alias in ("period", "period_type", "period_basis"):
            if alias in data:
                data["fiscal_period"] = data.pop(alias)
                break
    raw_period = data.get("fiscal_period")
    if isinstance(raw_period, str) and re.fullmatch(
        r"q[1-4]", raw_period.strip(), re.I
    ):
        data.setdefault("period_key", raw_period.strip().upper())
    elif isinstance(raw_period, str) and re.fullmatch(
        r"q[1-4]\s+(?:19|20|21|22)\d{2}", raw_period.strip(), re.I
    ):
        data.setdefault("period_key", raw_period.strip()[:2].upper())
    elif isinstance(raw_period, str) and raw_period.strip().casefold() in {
        "first quarter",
        "second quarter",
        "third quarter",
        "fourth quarter",
    }:
        data.setdefault(
            "period_key",
            {
                "first quarter": "Q1",
                "second quarter": "Q2",
                "third quarter": "Q3",
                "fourth quarter": "Q4",
            }[raw_period.strip().casefold()],
        )
    elif isinstance(raw_period, str) and raw_period.strip().casefold() in {
        "quarter",
        "quarterly",
        "fiscal quarter",
    }:
        data.setdefault("fiscal_period", "FQ")
    return data


def _scale_value(value: Decimal | None, scale: Decimal) -> Decimal | None:
    return value * scale if value is not None else None


def _observation_scaled_value(value: Decimal | None, scale: Decimal) -> Decimal:
    if value is None:
        raise ValueError("Operating observation has no usable value")
    return value * scale


def _coerce_extracted_text_map(value: Any, label: str) -> dict[str, str]:
    """Normalize API-safe key/value pairs and legacy mapping fixtures.

    The model-facing schema uses a list of objects because arbitrary JSON
    object properties are not accepted by OpenAI Structured Outputs.  The
    domain model still exposes the more useful mapping shape to deterministic
    consumers, and legacy cached/fixture responses remain readable.
    """

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {
            _normalize_required_text(
                key, f"Extracted {label} key"
            ): _normalize_required_text(item, f"Extracted {label} value")
            for key, item in value.items()
        }
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Extracted {label} map must be an object or key/value list")
    result: dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"Extracted {label} entries must be objects")
        if set(item) != {"key", "value"}:
            raise ValueError(
                f"Extracted {label} entries must contain only key and value"
            )
        key = _normalize_required_text(item["key"], f"Extracted {label} key")
        if key in result:
            raise ValueError(f"Extracted {label} keys must be unique")
        result[key] = _normalize_required_text(
            item["value"], f"Extracted {label} value"
        )
    return result


def _coerce_operating_archetype(value: OperatingArchetype | str) -> OperatingArchetype:
    """Normalize the small vocabulary accepted at the extraction boundary."""

    from edgarito.schemas.operating import OperatingArchetype

    if isinstance(value, OperatingArchetype):
        return value
    normalized = (
        str(getattr(value, "value", value))
        .strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("×", "_times_")
    )
    aliases = {
        "volume_times_price": OperatingArchetype.VOLUME_PRICE,
        "volume_and_price": OperatingArchetype.VOLUME_PRICE,
        "subscriber_arpu": OperatingArchetype.SUBSCRIBERS_ARPU,
        "subscribers_and_arpu": OperatingArchetype.SUBSCRIBERS_ARPU,
        "capacity_utilization_and_price": OperatingArchetype.CAPACITY_UTILIZATION_PRICE,
        "capacity_times_utilization_times_price": OperatingArchetype.CAPACITY_UTILIZATION_PRICE,
        "transactions_and_take_rate": OperatingArchetype.TRANSACTIONS_TAKE_RATE,
        "backlog_and_conversion": OperatingArchetype.BACKLOG_CONVERSION,
        "store_count_and_sales_per_store": OperatingArchetype.STORE_COUNT_SALES_PER_STORE,
        "segment_growth": OperatingArchetype.GENERIC_SEGMENT_GROWTH,
    }
    aliased = aliases.get(normalized)
    if aliased is not None:
        return aliased
    try:
        return OperatingArchetype(normalized)
    except ValueError as error:
        raise ValueError(f"Unsupported operating archetype: {value}") from error


def _archetype_metrics(archetype: OperatingArchetype) -> tuple[str, ...]:
    """Return canonical input names for an extracted archetype mapping."""

    from edgarito.schemas.operating import OperatingArchetype

    return {
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
    }[archetype]


def _normalize_optional_text(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    return _normalize_required_text(value, label)


def _finite_decimal(value: Decimal | None, label: str) -> Decimal | None:
    if value is not None and not value.is_finite():
        raise ValueError(f"{label} must be finite")
    return value


def _non_negative_decimal(value: Decimal, label: str) -> Decimal:
    if not value.is_finite():
        raise ValueError(f"{label} must be finite")
    if value < 0:
        raise ValueError(f"{label} cannot be negative")
    return value


def _validate_year_sequence(years: tuple[int, ...], label: str) -> None:
    if not years:
        raise ValueError(f"{label} cannot be empty")
    if any(year < _MIN_FISCAL_YEAR or year > _MAX_FISCAL_YEAR for year in years):
        raise ValueError(
            f"{label} must be between {_MIN_FISCAL_YEAR} and {_MAX_FISCAL_YEAR}"
        )
    if tuple(sorted(years)) != years:
        raise ValueError(f"{label} must be in ascending order")
    if len(years) != len(set(years)):
        raise ValueError(f"{label} cannot contain duplicate years")


def _validate_subset(
    values: tuple[int, ...], universe: tuple[int, ...], label: str
) -> None:
    if tuple(sorted(values)) != values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be sorted and unique")
    if not set(values).issubset(universe):
        raise ValueError(f"{label} must be contained in fiscal_years")


def _validate_year_map(
    values: dict[int, str], years: tuple[int, ...], label: str
) -> None:
    if not set(values).issubset(years):
        raise ValueError(f"{label} contains a year outside fiscal_years")


def _validate_growth_consistency(
    revenue: tuple[Decimal, ...],
    growth: tuple[Decimal | None, ...],
    label: str,
) -> None:
    """Validate percentage-point growth where an absolute prior is available."""

    for index in range(1, len(revenue)):
        observed_growth = growth[index]
        if observed_growth is None:
            continue
        previous = revenue[index - 1]
        current = revenue[index]
        if previous == 0:
            if current != 0:
                raise ValueError(f"{label} cannot be derived from zero prior revenue")
            expected = Decimal(0)
        else:
            expected = (current / previous - Decimal(1)) * Decimal(100)
        scale = max(abs(expected), abs(observed_growth), Decimal(1))
        if abs(expected - observed_growth) > scale * Decimal("1e-18"):
            raise ValueError(
                f"{label} must match growth derived from absolute revenue values"
            )
