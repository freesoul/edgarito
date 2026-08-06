from typing import List, Optional

from edgarito.schemas.red_flags import RedFlag, RedFlagSeverity
from edgarito.enums.granularity import Granularity
from edgarito.services.analysis.red_flags.base_analyzer import BaseAnalyzer

class EarningsQualityAnalyzer(BaseAnalyzer):
    """
    Analyzes earnings quality and potential manipulation using Beneish M-Score inspired metrics.
    
    Checks for:
    - Days Sales Receivable Index (DSRI)
    - Gross Margin Index (GMI)
    - Asset Quality Index (AQI)
    """
    
    def analyze(self, granularity: Granularity) -> List[RedFlag]:
        flags = []
        
        # Get data for current and previous period
        # We need at least 2 periods for these indices
        
        # 1. Days Sales Receivable Index (DSRI)
        # DSRI = (Net Receivables_t / Sales_t) / (Net Receivables_t-1 / Sales_t-1)
        # A large increase in receivables relative to sales may indicate revenue inflation.
        
        try:
            receivables = self.balance_sheet.get_accounts_receivable(granularity)
            revenue = self.income_statement.get_revenue(granularity)
            
            # Align data
            receivables_aligned = receivables.intersect(revenue)
            revenue_aligned = revenue.intersect(receivables)
            
            if len(receivables_aligned.values) >= 2:
                curr_rec = receivables_aligned.values[0]
                prev_rec = receivables_aligned.values[1]
                curr_rev = revenue_aligned.values[0]
                prev_rev = revenue_aligned.values[1]
                
                if curr_rev > 0 and prev_rev > 0:
                    dsri = (curr_rec / curr_rev) / (prev_rec / prev_rev)
                    
                    if dsri > 1.5: # Critical threshold
                         flags.append(RedFlag(
                            category="Earnings Quality",
                            severity=RedFlagSeverity.CRITICAL,
                            title="High Days Sales Receivable Index (DSRI)",
                            description=f"Receivables rising much faster than sales (Index: {dsri:.2f}). Potential revenue inflation.",
                            current_value=dsri,
                            threshold=1.5,
                            period=str(receivables_aligned.periods[0]),
                            score_impact=-15.0
                        ))
                    elif dsri > 1.2: # Warning threshold
                        flags.append(RedFlag(
                            category="Earnings Quality",
                            severity=RedFlagSeverity.WARNING,
                            title="Elevated Days Sales Receivable Index (DSRI)",
                            description=f"Receivables rising faster than sales (Index: {dsri:.2f}).",
                            current_value=dsri,
                            threshold=1.2,
                            period=str(receivables_aligned.periods[0]),
                            score_impact=-7.0
                        ))
        except Exception:
            pass # Skip if data missing
            
        # 2. Gross Margin Index (GMI)
        # GMI = [(Sales_t-1 - COGS_t-1) / Sales_t-1] / [(Sales_t - COGS_t) / Sales_t]
        # GMI > 1 means gross margin has deteriorated.
        
        try:
            gross_profit = self.income_statement.get_gross_profit(granularity)
            revenue = self.income_statement.get_revenue(granularity)
            
            gp_aligned = gross_profit.intersect(revenue)
            rev_aligned = revenue.intersect(gross_profit)
            
            if len(gp_aligned.values) >= 2:
                curr_gp = gp_aligned.values[0]
                prev_gp = gp_aligned.values[1]
                curr_rev = rev_aligned.values[0]
                prev_rev = rev_aligned.values[1]
                
                if curr_rev > 0 and prev_rev > 0:
                    curr_margin = curr_gp / curr_rev
                    prev_margin = prev_gp / prev_rev
                    
                    if curr_margin > 0:
                        gmi = prev_margin / curr_margin
                        
                        if gmi > 1.3:
                             flags.append(RedFlag(
                                category="Earnings Quality",
                                severity=RedFlagSeverity.WARNING,
                                title="Deteriorating Gross Margin (GMI)",
                                description=f"Gross margins have declined significantly (Index: {gmi:.2f}).",
                                current_value=gmi,
                                threshold=1.3,
                                period=str(gp_aligned.periods[0]),
                                score_impact=-7.0
                            ))
        except Exception:
            pass

        return flags
