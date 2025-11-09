"""
Growth and sustainability analyzer - revenue trends, expense discipline, R&D investment checks.
"""
from typing import List

from edgarito.enums.granularity import Granularity
from edgarito.schemas.red_flags import RedFlag, RedFlagSeverity
from .base_analyzer import BaseAnalyzer


class GrowthAnalyzer(BaseAnalyzer):
    """Analyzes growth and sustainability red flags."""
    
    def analyze(self, granularity: Granularity) -> List[RedFlag]:
        """Analyze growth and sustainability for red flags."""
        flags = []
        
        try:
            revenue = self.income_statement.get_revenue(granularity)
            
            if not revenue.values or len(revenue.values) < 5:
                return flags
            
            latest_period = revenue.periods[-1]
            period_str = f"{latest_period.year} {latest_period.fp.value}"
            
            # Run all growth checks
            flags.extend(self._check_revenue_cagr(revenue, granularity, period_str))
            flags.extend(self._check_sga_expenses(revenue, granularity, period_str))
            flags.extend(self._check_rd_investment(revenue, granularity, period_str))
            
        except ValueError:
            pass
        
        return flags
    
    def _check_revenue_cagr(self, revenue, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check revenue CAGR (severe decline = CRITICAL, stagnation = INFO)."""
        flags = []
        
        if len(revenue.values) >= 5:
            years = 5 if granularity == Granularity.ANNUAL else 5/4  # Adjust for quarterly
            cagr = ((revenue.values[-1] / revenue.values[-5]) ** (1/years) - 1) * 100
            
            if cagr < -20:  # Severe revenue decline (>20% CAGR)
                flags.append(RedFlag(
                    category="Growth",
                    severity=RedFlagSeverity.CRITICAL,
                    title="Severe Revenue Decline",
                    description="Business collapsing - catastrophic revenue contraction",
                    current_value=cagr,
                    threshold=-20.0,
                    period=f"{revenue.periods[-5].year}-{period_str.split()[0]}"
                ))
            elif cagr < self.thresholds.revenue_cagr_inflation:
                flags.append(RedFlag(
                    category="Growth",
                    severity=RedFlagSeverity.INFO,
                    title="Revenue Growth Below Inflation",
                    description="Stagnation - revenue barely keeping up with inflation",
                    current_value=cagr,
                    threshold=self.thresholds.revenue_cagr_inflation,
                    period=f"{revenue.periods[-5].year}-{period_str.split()[0]}"
                ))
        
        return flags
    
    def _check_sga_expenses(self, revenue, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check SG&A expenses with tiered severity."""
        flags = []
        
        try:
            sga = self.income_statement.get_selling_general_administrative_expense(granularity)
            
            if not sga.values or not revenue.values:
                return flags
            
            sga_pct = (sga.values[-1] / revenue.values[-1]) * 100 if revenue.values[-1] != 0 else 0
            
            # Check SG&A thresholds
            if sga_pct > self.thresholds.sga_percent_revenue_warning:
                flags.append(RedFlag(
                    category="Growth",
                    severity=RedFlagSeverity.WARNING,
                    title="High SG&A Expenses",
                    description=f"Bloated overhead - SG&A exceeds {self.thresholds.sga_percent_revenue_warning}% of revenue",
                    current_value=sga_pct,
                    threshold=self.thresholds.sga_percent_revenue_warning,
                    period=period_str
                ))
            elif sga_pct > self.thresholds.sga_percent_revenue_info:
                flags.append(RedFlag(
                    category="Growth",
                    severity=RedFlagSeverity.INFO,
                    title="Elevated SG&A Expenses",
                    description="SG&A expenses moderately high relative to revenue",
                    current_value=sga_pct,
                    threshold=self.thresholds.sga_percent_revenue_info,
                    period=period_str
                ))
            
            # Check if SG&A % is rising
            if len(sga.values) >= 3 and len(revenue.values) >= 3:
                sga_pct_prev = (sga.values[-3] / revenue.values[-3]) * 100 if revenue.values[-3] != 0 else 0
                
                if sga_pct > sga_pct_prev + self.thresholds.sga_increase_threshold:
                    flags.append(RedFlag(
                        category="Growth",
                        severity=RedFlagSeverity.INFO,
                        title="Rising SG&A as % of Revenue",
                        description="Poor cost discipline - overhead growing faster than revenue",
                        current_value=sga_pct,
                        threshold=sga_pct_prev,
                        period=f"{revenue.periods[-3].year}-{period_str.split()[0]}"
                    ))
        except ValueError:
            pass
        
        return flags
    
    def _check_rd_investment(self, revenue, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check for declining R&D while revenue is growing."""
        flags = []
        
        try:
            rd = self.income_statement.get_research_and_development_expense(granularity)
            
            if len(rd.values) >= 3 and len(revenue.values) >= 3:
                rd_growth = ((rd.values[-1] - rd.values[-3]) / rd.values[-3]) * 100 if rd.values[-3] != 0 else 0
                rev_growth = ((revenue.values[-1] - revenue.values[-3]) / revenue.values[-3]) * 100 if revenue.values[-3] != 0 else 0
                
                # If revenue is growing but R&D is declining
                if rev_growth > self.thresholds.revenue_growth_for_rd_check and rd_growth < self.thresholds.rd_decline_threshold:
                    flags.append(RedFlag(
                        category="Growth",
                        severity=RedFlagSeverity.WARNING,
                        title="Declining R&D Despite Revenue Growth",
                        description="Underinvesting in future - may hurt long-term competitiveness",
                        current_value=rd_growth,
                        threshold=0.0,
                        period=f"{revenue.periods[-3].year}-{period_str.split()[0]}"
                    ))
        except ValueError:
            pass
        
        return flags
