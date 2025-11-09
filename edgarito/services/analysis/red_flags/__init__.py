"""
Red flags detection system - modular architecture.
"""
from edgarito.schemas.red_flags import RedFlag, RedFlagReport, RedFlagSeverity
from .base_analyzer import BaseAnalyzer
from .balance_sheet_analyzer import BalanceSheetAnalyzer
from .cash_flow_analyzer import CashFlowAnalyzer
from .profitability_analyzer import ProfitabilityAnalyzer
from .growth_analyzer import GrowthAnalyzer
from .valuation_analyzer import ValuationAnalyzer

__all__ = [
    "RedFlag",
    "RedFlagReport",
    "RedFlagSeverity",
    "BaseAnalyzer",
    "BalanceSheetAnalyzer",
    "CashFlowAnalyzer",
    "ProfitabilityAnalyzer",
    "GrowthAnalyzer",
    "ValuationAnalyzer",
]
