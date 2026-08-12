"""Deterministic formula registry for operating-revenue archetypes.

The registry deliberately contains arithmetic only.  It does not retrieve
evidence, infer missing values, or know anything about providers.  Those
responsibilities belong to the operating forecast service and discovery
layers respectively.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal

from edgarito.schemas.operating import OperatingArchetype

Formula = Callable[
    [Mapping[str, Decimal], Mapping[str, str] | None, Decimal | None], Decimal
]


def _metric_value(inputs: Mapping[str, Decimal], *names: str) -> Decimal:
    normalized = {
        str(key).strip().casefold().replace("-", "_").replace(" ", "_"): value
        for key, value in inputs.items()
    }
    for name in names:
        key = name.strip().casefold().replace("-", "_").replace(" ", "_")
        if key in normalized:
            return normalized[key]
    raise KeyError(f"Missing formula input: {names[0]}")


def _metric_unit(units: Mapping[str, str] | None, *names: str) -> str:
    if not units:
        return ""
    normalized = {
        str(key).strip().casefold().replace("-", "_").replace(" ", "_"): value
        for key, value in units.items()
    }
    for name in names:
        key = name.strip().casefold().replace("-", "_").replace(" ", "_")
        if key in normalized:
            return str(normalized[key]).strip().casefold()
    return ""


def _as_fraction(
    value: Decimal, units: Mapping[str, str] | None, *metric_names: str
) -> Decimal:
    """Convert explicitly percentage-denominated rates to a fraction.

    Ratio inputs such as ``Decimal("0.8")`` are left unchanged.  A unit that
    says percent/percentage-points is interpreted as ``80 == 80%``.  Keeping
    the conversion tied to the declared unit avoids guessing from a numeric
    value alone.
    """

    unit = _metric_unit(units, *metric_names).replace("_", " ")
    if "basis point" in unit or unit in {"bps", "bp"}:
        return value / Decimal(10_000)
    if "%" in unit or "percent" in unit or "percentage point" in unit:
        return value / Decimal(100)
    return value


def volume_price(
    inputs: Mapping[str, Decimal],
    units: Mapping[str, str] | None = None,
    previous_revenue: Decimal | None = None,
) -> Decimal:
    del previous_revenue
    return _metric_value(inputs, "volume") * _metric_value(inputs, "price")


def subscribers_arpu(
    inputs: Mapping[str, Decimal],
    units: Mapping[str, str] | None = None,
    previous_revenue: Decimal | None = None,
) -> Decimal:
    del previous_revenue
    return _metric_value(inputs, "subscribers", "subscriber_count") * _metric_value(
        inputs, "arpu", "average_revenue_per_user"
    )


def capacity_utilization_price(
    inputs: Mapping[str, Decimal],
    units: Mapping[str, str] | None = None,
    previous_revenue: Decimal | None = None,
) -> Decimal:
    del previous_revenue
    capacity = _metric_value(inputs, "capacity")
    utilization = _as_fraction(
        _metric_value(inputs, "utilization"), units, "utilization"
    )
    price = _metric_value(inputs, "price")
    return capacity * utilization * price


def transactions_take_rate(
    inputs: Mapping[str, Decimal],
    units: Mapping[str, str] | None = None,
    previous_revenue: Decimal | None = None,
) -> Decimal:
    del previous_revenue
    transactions = _metric_value(inputs, "transactions", "transaction_count")
    take_rate = _as_fraction(
        _metric_value(inputs, "take_rate"),
        units,
        "take_rate",
    )
    return transactions * take_rate


def backlog_conversion(
    inputs: Mapping[str, Decimal],
    units: Mapping[str, str] | None = None,
    previous_revenue: Decimal | None = None,
) -> Decimal:
    del previous_revenue
    backlog = _metric_value(inputs, "backlog")
    conversion_rate = _as_fraction(
        _metric_value(inputs, "conversion_rate", "conversion"),
        units,
        "conversion_rate",
        "conversion",
    )
    return backlog * conversion_rate


def store_count_sales_per_store(
    inputs: Mapping[str, Decimal],
    units: Mapping[str, str] | None = None,
    previous_revenue: Decimal | None = None,
) -> Decimal:
    del units, previous_revenue
    return _metric_value(inputs, "store_count", "stores") * _metric_value(
        inputs, "sales_per_store", "sales_per_location"
    )


def generic_segment_growth(
    inputs: Mapping[str, Decimal],
    units: Mapping[str, str] | None = None,
    previous_revenue: Decimal | None = None,
) -> Decimal:
    previous = previous_revenue
    if previous is None:
        try:
            previous = _metric_value(inputs, "previous_revenue", "prior_revenue")
        except KeyError as error:
            raise KeyError("Missing formula input: previous_revenue") from error
    growth = _as_fraction(
        _metric_value(inputs, "growth", "growth_rate"),
        units,
        "growth",
        "growth_rate",
    )
    return previous * (Decimal(1) + growth)


ARCHETYPE_FORMULAS: dict[OperatingArchetype, Formula] = {
    OperatingArchetype.VOLUME_PRICE: volume_price,
    OperatingArchetype.SUBSCRIBERS_ARPU: subscribers_arpu,
    OperatingArchetype.CAPACITY_UTILIZATION_PRICE: capacity_utilization_price,
    OperatingArchetype.TRANSACTIONS_TAKE_RATE: transactions_take_rate,
    OperatingArchetype.BACKLOG_CONVERSION: backlog_conversion,
    OperatingArchetype.STORE_COUNT_SALES_PER_STORE: store_count_sales_per_store,
    OperatingArchetype.GENERIC_SEGMENT_GROWTH: generic_segment_growth,
}


class ArchetypeFormulaRegistry:
    """Read-only-ish registry exposing the seven supported formulas."""

    def __init__(self, formulas: Mapping[OperatingArchetype, Formula] | None = None):
        self._formulas = dict(formulas or ARCHETYPE_FORMULAS)

    def __contains__(self, archetype: object) -> bool:
        try:
            return OperatingArchetype(archetype) in self._formulas
        except ValueError:
            return False

    def __getitem__(self, archetype: OperatingArchetype | str) -> Formula:
        return self.formula(archetype)

    def __iter__(self):
        return iter(self._formulas)

    def __len__(self) -> int:
        return len(self._formulas)

    def items(self):
        return self._formulas.items()

    def get(self, archetype: OperatingArchetype | str, default=None):
        try:
            return self.formula(archetype)
        except (KeyError, ValueError):
            return default

    def formula(self, archetype: OperatingArchetype | str) -> Formula:
        return self._formulas[OperatingArchetype(archetype)]

    def evaluate(
        self,
        archetype: OperatingArchetype | str,
        inputs: Mapping[str, Decimal],
        *,
        units: Mapping[str, str] | None = None,
        previous_revenue: Decimal | None = None,
    ) -> Decimal:
        value = self.formula(archetype)(inputs, units, previous_revenue)
        if not value.is_finite():
            raise ValueError("Operating formula result must be finite")
        return value


FORMULA_REGISTRY = ArchetypeFormulaRegistry()
OperatingFormulaRegistry = ArchetypeFormulaRegistry


__all__ = [
    "ARCHETYPE_FORMULAS",
    "FORMULA_REGISTRY",
    "ArchetypeFormulaRegistry",
    "Formula",
    "OperatingFormulaRegistry",
    "backlog_conversion",
    "capacity_utilization_price",
    "generic_segment_growth",
    "store_count_sales_per_store",
    "subscribers_arpu",
    "transactions_take_rate",
    "volume_price",
]
