"""Individually testable deterministic forecast validation rules."""

from .compounding import CompoundingScaleRule
from .consistency import CrossMetricConsistencyRule
from .growth import FcffGrowthRule
from .horizon import HorizonIntegrityRule
from .margins import OperatingMarginRule, RevenueFcffMarginRule
from .reinvestment import ReinvestmentRule
from .terminal import (
    TerminalDiscontinuityRule,
    TerminalEconomicConsistencyRule,
    TerminalValueRule,
)
from .working_capital import WorkingCapitalRule

__all__ = [
    "CompoundingScaleRule",
    "CrossMetricConsistencyRule",
    "FcffGrowthRule",
    "HorizonIntegrityRule",
    "OperatingMarginRule",
    "ReinvestmentRule",
    "RevenueFcffMarginRule",
    "TerminalValueRule",
    "TerminalDiscontinuityRule",
    "TerminalEconomicConsistencyRule",
    "WorkingCapitalRule",
]
