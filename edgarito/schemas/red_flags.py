"""
Data models for red flag detection system.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from edgarito.schemas.market_data import MarketData


class RedFlagSeverity(Enum):
    """Severity level of a red flag"""
    CRITICAL = "🔴 CRITICAL"
    WARNING = "🟡 WARNING"
    INFO = "ℹ️  INFO"


@dataclass
class RedFlag:
    """A single red flag detection"""
    category: str
    severity: RedFlagSeverity
    title: str
    description: str
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    period: Optional[str] = None
    score_impact: float = 0.0
    
    def __str__(self) -> str:
        result = f"{self.severity.value} [{self.category}] {self.title}"
        if self.period:
            result += f" ({self.period})"
        result += f"\n  {self.description}"
        if self.current_value is not None and self.threshold is not None:
            result += f"\n  Current: {self.current_value:.2f} | Threshold: {self.threshold:.2f}"
        elif self.current_value is not None:
            result += f"\n  Current: {self.current_value:.2f}"
        return result


@dataclass
class RedFlagReport:
    """Complete red flags analysis report"""
    ticker: str
    company_name: str
    total_flags: int = 0
    critical_flags: int = 0
    warning_flags: int = 0
    info_flags: int = 0
    balance_sheet_flags: List[RedFlag] = field(default_factory=list)
    cash_flow_flags: List[RedFlag] = field(default_factory=list)
    profitability_flags: List[RedFlag] = field(default_factory=list)
    growth_flags: List[RedFlag] = field(default_factory=list)
    valuation_flags: List[RedFlag] = field(default_factory=list)
    market_data: Optional[MarketData] = None
    quality_score: float = 100.0
    data_availability_warnings: List[str] = field(default_factory=list)  # Track missing/insufficient data
    
    @property
    def all_flags(self) -> List[RedFlag]:
        """Get all flags sorted by severity"""
        all_flags = (
            self.balance_sheet_flags + 
            self.cash_flow_flags + 
            self.profitability_flags +
            self.growth_flags +
            self.valuation_flags
        )
        # Sort: CRITICAL first, then WARNING, then INFO
        severity_order = {
            RedFlagSeverity.CRITICAL: 0,
            RedFlagSeverity.WARNING: 1,
            RedFlagSeverity.INFO: 2
        }
        return sorted(all_flags, key=lambda f: severity_order[f.severity])
    
    def __str__(self) -> str:
        result = f"\n{'='*100}\n"
        result += f"RED FLAGS ANALYSIS: {self.ticker} - {self.company_name}\n"
        
        # Display sector and industry if available
        if self.market_data:
            sector_info = []
            if self.market_data.sector:
                sector_info.append(f"Sector: {self.market_data.sector}")
            if self.market_data.industry:
                sector_info.append(f"Industry: {self.market_data.industry}")
            if sector_info:
                result += f"{' | '.join(sector_info)}\n"
        
        result += f"{'='*100}\n\n"
        result += f"Summary: {self.total_flags} total flags "
        result += f"({self.critical_flags} critical, {self.warning_flags} warnings, {self.info_flags} info)\n"
        result += f"Quality Score: {self.quality_score:.1f}/100\n\n"
        
        # Display data availability warnings prominently if present
        if self.data_availability_warnings:
            result += f"\n{'='*100}\n"
            result += "⚠️  DATA AVAILABILITY WARNINGS\n"
            result += f"{'='*100}\n"
            for warning in self.data_availability_warnings:
                result += f"\n⚠️  {warning}\n"
            result += f"\n{'='*100}\n"
            
            # If there are warnings and no flags, explain why
            if self.total_flags == 0:
                result += "\n❌ Unable to perform comprehensive red flags analysis due to insufficient data.\n"
                result += "This may occur when:\n"
                result += "  • Company uses IFRS accounting (not US-GAAP)\n"
                result += "  • Company recently went public (limited historical data)\n"
                result += "  • SEC filing data is incomplete or unavailable\n"
                return result
        
        if self.total_flags == 0:
            result += "✅ No significant red flags detected!\n"
            return result
        
        # Group by category
        if self.balance_sheet_flags:
            result += f"\n{'='*100}\n"
            result += "🧾 BALANCE SHEET HEALTH\n"
            result += f"{'='*100}\n"
            for flag in self.balance_sheet_flags:
                result += f"\n{flag}\n"
        
        if self.cash_flow_flags:
            result += f"\n{'='*100}\n"
            result += "💸 CASH FLOW QUALITY\n"
            result += f"{'='*100}\n"
            for flag in self.cash_flow_flags:
                result += f"\n{flag}\n"
        
        if self.profitability_flags:
            result += f"\n{'='*100}\n"
            result += "📊 PROFITABILITY & INCOME QUALITY\n"
            result += f"{'='*100}\n"
            for flag in self.profitability_flags:
                result += f"\n{flag}\n"
        
        if self.growth_flags:
            result += f"\n{'='*100}\n"
            result += "🧯 GROWTH & SUSTAINABILITY\n"
            result += f"{'='*100}\n"
            for flag in self.growth_flags:
                result += f"\n{flag}\n"
        
        if self.valuation_flags:
            result += f"\n{'='*100}\n"
            result += "🧮 VALUATION CONCERNS\n"
            result += f"{'='*100}\n"
            for flag in self.valuation_flags:
                result += f"\n{flag}\n"
        
        return result
