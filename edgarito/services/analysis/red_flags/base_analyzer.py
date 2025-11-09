"""
Base analyzer class with shared functionality for all red flag analyzers.
"""
from typing import Optional

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts
from edgarito.schemas.market_data import MarketData
from edgarito.services.financial.balance_sheet_reader import BalanceSheetReader
from edgarito.services.financial.income_statement_reader import IncomeStatementReader
from edgarito.services.financial.cash_flow_reader import CashFlowStatementReader
from edgarito.cli.settings import RedFlagsThresholds, SectorThresholdAdjustments, settings


class BaseAnalyzer:
    """Base class for red flag analyzers with shared state and utilities."""
    
    def __init__(
        self,
        facts: CompanyFacts,
        balance_sheet: BalanceSheetReader,
        income_statement: IncomeStatementReader,
        cash_flow: CashFlowStatementReader,
        thresholds: RedFlagsThresholds,
        sector: str,
        sector_profile: SectorThresholdAdjustments,
        market_cap: Optional[float] = None,
        market_data: Optional[MarketData] = None
    ):
        """
        Initialize base analyzer with shared resources.
        
        Args:
            facts: Company facts data from SEC
            balance_sheet: Balance sheet reader
            income_statement: Income statement reader
            cash_flow: Cash flow statement reader
            thresholds: Configured thresholds for red flag detection
            sector: Company sector (normalized lowercase)
            sector_profile: Sector-specific threshold adjustments
            market_cap: Current market capitalization (optional)
            market_data: Complete market data including cap, valuation metrics
        """
        self.facts = facts
        self.balance_sheet = balance_sheet
        self.income_statement = income_statement
        self.cash_flow = cash_flow
        self.thresholds = thresholds
        self.sector = sector
        self.sector_profile = sector_profile
        self.market_cap = market_cap
        self.market_data = market_data
    
    def _adjust_threshold(self, base_threshold: float, multiplier: float) -> float:
        """Apply sector-specific multiplier to a base threshold."""
        return base_threshold * multiplier
