"""
Red Flags Service - orchestrator for modular financial warning sign detection.

This slim service coordinates specialized analyzers for different financial aspects.
"""
from typing import Optional

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts
from edgarito.schemas.red_flags import RedFlag, RedFlagReport, RedFlagSeverity
from edgarito.schemas.market_data import MarketData
from edgarito.enums.granularity import Granularity
from edgarito.services.financial.balance_sheet_reader import BalanceSheetReader
from edgarito.services.financial.income_statement_reader import IncomeStatementReader
from edgarito.services.financial.cash_flow_reader import CashFlowStatementReader
from edgarito.cli.settings import RedFlagsThresholds

from .red_flags.balance_sheet_analyzer import BalanceSheetAnalyzer
from .red_flags.cash_flow_analyzer import CashFlowAnalyzer
from .red_flags.profitability_analyzer import ProfitabilityAnalyzer
from .red_flags.growth_analyzer import GrowthAnalyzer
from .red_flags.valuation_analyzer import ValuationAnalyzer



class RedFlagsService:
    """Service for detecting financial red flags - orchestrates specialized analyzers."""
    
    def __init__(
        self, 
        facts: CompanyFacts, 
        market_cap: Optional[float] = None,
        market_data: Optional[MarketData] = None,
        thresholds: Optional['RedFlagsThresholds'] = None
    ):
        """
        Initialize red flags service.
        
        Args:
            facts: Company facts data from SEC
            market_cap: Current market capitalization (optional, deprecated - use market_data)
            market_data: Complete market data including cap, valuation metrics, etc.
            thresholds: Configurable thresholds for red flag detection (optional, uses defaults if None)
        """
        self.facts = facts
        self.market_data = market_data
        # For backwards compatibility
        self.market_cap = market_data.market_cap if market_data else market_cap
        self.balance_sheet = BalanceSheetReader(facts)
        self.income_statement = IncomeStatementReader(facts)
        self.cash_flow = CashFlowStatementReader(facts)
        
        # Import here to avoid circular dependency
        if thresholds is None:
            from edgarito.cli.settings import RedFlagsThresholds
            thresholds = RedFlagsThresholds()
        self.thresholds = thresholds
        
        # Determine sector and apply adjustments
        self.sector = self._get_sector()
        self.sector_profile = self._get_sector_profile()
    
    def _get_sector(self) -> str:
        """Get the company's sector, normalized to lowercase."""
        if self.market_data and self.market_data.sector:
            return self.market_data.sector.lower()
        return "general"
    
    def _get_sector_profile(self):
        """Get sector-specific threshold adjustments."""
        from edgarito.cli.settings import settings, SectorThresholdAdjustments
        return settings.sector_profiles.get(self.sector, SectorThresholdAdjustments())
    
    def analyze(self) -> RedFlagReport:
        """
        Run complete red flags analysis using both quarterly and annual data where appropriate.
        
        Strategy:
        - Balance Sheet & Cash Flow: Use QUARTERLY for most recent liquidity/solvency
        - Profitability: Use QUARTERLY for current margins
        - Growth: Use ANNUAL for long-term CAGR (5+ years)
        - Valuation: Use QUARTERLY (most recent data)
        
        Returns:
            Complete red flag report
        """
        report = RedFlagReport(
            ticker=str(self.facts.cik),
            company_name=self.facts.entityName,
            market_data=self.market_data
        )
        
        # Instantiate all specialized analyzers with shared state
        balance_sheet_analyzer = BalanceSheetAnalyzer(
            facts=self.facts,
            balance_sheet=self.balance_sheet,
            income_statement=self.income_statement,
            cash_flow=self.cash_flow,
            thresholds=self.thresholds,
            sector=self.sector,
            sector_profile=self.sector_profile,
            market_cap=self.market_cap,
            market_data=self.market_data
        )
        
        cash_flow_analyzer = CashFlowAnalyzer(
            facts=self.facts,
            balance_sheet=self.balance_sheet,
            income_statement=self.income_statement,
            cash_flow=self.cash_flow,
            thresholds=self.thresholds,
            sector=self.sector,
            sector_profile=self.sector_profile,
            market_cap=self.market_cap,
            market_data=self.market_data
        )
        
        profitability_analyzer = ProfitabilityAnalyzer(
            facts=self.facts,
            balance_sheet=self.balance_sheet,
            income_statement=self.income_statement,
            cash_flow=self.cash_flow,
            thresholds=self.thresholds,
            sector=self.sector,
            sector_profile=self.sector_profile,
            market_cap=self.market_cap,
            market_data=self.market_data
        )
        
        growth_analyzer = GrowthAnalyzer(
            facts=self.facts,
            balance_sheet=self.balance_sheet,
            income_statement=self.income_statement,
            cash_flow=self.cash_flow,
            thresholds=self.thresholds,
            sector=self.sector,
            sector_profile=self.sector_profile,
            market_cap=self.market_cap,
            market_data=self.market_data
        )
        
        valuation_analyzer = ValuationAnalyzer(
            facts=self.facts,
            balance_sheet=self.balance_sheet,
            income_statement=self.income_statement,
            cash_flow=self.cash_flow,
            thresholds=self.thresholds,
            sector=self.sector,
            sector_profile=self.sector_profile,
            market_cap=self.market_cap,
            market_data=self.market_data
        )
        
        # Run all analyses with appropriate granularity
        report.balance_sheet_flags = balance_sheet_analyzer.analyze(Granularity.QUARTERLY)
        report.cash_flow_flags = cash_flow_analyzer.analyze(Granularity.QUARTERLY)
        report.profitability_flags = profitability_analyzer.analyze(Granularity.QUARTERLY)
        report.growth_flags = growth_analyzer.analyze(Granularity.ANNUAL)  # Long-term trends
        report.valuation_flags = valuation_analyzer.analyze(Granularity.QUARTERLY)
        
        # Count flags by severity
        for flag in report.all_flags:
            report.total_flags += 1
            if flag.severity == RedFlagSeverity.CRITICAL:
                report.critical_flags += 1
            elif flag.severity == RedFlagSeverity.WARNING:
                report.warning_flags += 1
            else:
                report.info_flags += 1
        
        return report
