"""Compatibility import surface for provider-neutral tax calculations."""

from edgarito.services.financials.effective_tax import (
    PERCENT,
    calculate_effective_tax_rate,
    effective_tax_rate,
)

__all__ = ["PERCENT", "calculate_effective_tax_rate", "effective_tax_rate"]
