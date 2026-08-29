"""Small provider-neutral effective-tax calculation shared by forecasts."""

from __future__ import annotations

from decimal import Decimal

PERCENT = Decimal(100)


def calculate_effective_tax_rate(
    pretax_income: Decimal,
    income_tax_expense: Decimal,
) -> Decimal | None:
    """Return a strict effective tax rate in percentage points.

    Pretax income must be positive, tax expense must not be negative, and the
    resulting percentage must be within 0--100 inclusive.
    """

    if pretax_income <= 0 or income_tax_expense < 0:
        return None
    rate = income_tax_expense / pretax_income * PERCENT
    return rate if Decimal(0) <= rate <= PERCENT else None


effective_tax_rate = calculate_effective_tax_rate


__all__ = ["PERCENT", "calculate_effective_tax_rate", "effective_tax_rate"]
