"""
Profitability and income quality analyzer - margins, returns, earnings quality checks.
"""
from typing import List

from edgarito.enums.granularity import Granularity
from edgarito.schemas.red_flags import RedFlag, RedFlagSeverity
from .base_analyzer import BaseAnalyzer


class ProfitabilityAnalyzer(BaseAnalyzer):
    """Analyzes profitability metrics and income quality for red flags."""
    
    def analyze(self, granularity: Granularity) -> List[RedFlag]:
        """Analyze profitability and income quality for red flags."""
        flags = []
        
        try:
            revenue = self.income_statement.get_revenue(granularity)
            gross_profit = self.income_statement.get_gross_profit(granularity)
            operating_income = self.income_statement.get_operating_income(granularity)
            net_income = self.income_statement.get_net_income(granularity)
            
            if not revenue.values or not net_income.values:
                return flags
            
            latest_period = revenue.periods[-1]
            period_str = f"{latest_period.year} {latest_period.fp.value}"
            
            # Run all profitability checks
            flags.extend(self._check_gross_margin(revenue, gross_profit, period_str))
            flags.extend(self._check_operating_margin(revenue, operating_income, period_str))
            flags.extend(self._check_net_margin(revenue, net_income, period_str))
            flags.extend(self._check_roe(net_income, granularity, period_str))
            flags.extend(self._check_revenue_vs_eps_growth(revenue, granularity, period_str))
            flags.extend(self._check_margin_volatility(revenue, gross_profit))
            
        except ValueError:
            pass
        
        return flags
    
    def _check_gross_margin(self, revenue, gross_profit, period_str: str) -> List[RedFlag]:
        """Check for declining gross margin."""
        flags = []
        
        if len(revenue.values) >= 3 and len(gross_profit.values) >= 3:
            gm_current = (gross_profit.values[-1] / revenue.values[-1]) * 100 if revenue.values[-1] != 0 else 0
            gm_prev = (gross_profit.values[-3] / revenue.values[-3]) * 100 if revenue.values[-3] != 0 else 0
            
            if gm_current < gm_prev - 2:  # 2% decline threshold
                flags.append(RedFlag(
                    category="Profitability",
                    severity=RedFlagSeverity.WARNING,
                    title="Declining Gross Margin",
                    description="Pricing pressure or cost inflation squeezing margins",
                    current_value=gm_current,
                    threshold=gm_prev,
                    period=f"{revenue.periods[-3].year}-{period_str.split()[0]}"
                ))
        
        return flags
    
    def _check_operating_margin(self, revenue, operating_income, period_str: str) -> List[RedFlag]:
        """Check for negative or declining operating margin."""
        flags = []
        
        if len(revenue.values) >= 3 and len(operating_income.values) >= 3:
            om_current = (operating_income.values[-1] / revenue.values[-1]) * 100 if revenue.values[-1] != 0 else 0
            om_prev = (operating_income.values[-3] / revenue.values[-3]) * 100 if revenue.values[-3] != 0 else 0
            
            # Check for negative operating margin (CRITICAL)
            if om_current < 0:
                flags.append(RedFlag(
                    category="Profitability",
                    severity=RedFlagSeverity.CRITICAL,
                    title="Negative Operating Margin",
                    description="Operating losses - core business not profitable",
                    current_value=om_current,
                    threshold=0.0,
                    period=period_str
                ))
            elif om_current < om_prev - 2:  # 2% decline threshold
                flags.append(RedFlag(
                    category="Profitability",
                    severity=RedFlagSeverity.WARNING,
                    title="Declining Operating Margin",
                    description="Poor cost control or increased competition",
                    current_value=om_current,
                    threshold=om_prev,
                    period=f"{revenue.periods[-3].year}-{period_str.split()[0]}"
                ))
        
        return flags
    
    def _check_net_margin(self, revenue, net_income, period_str: str) -> List[RedFlag]:
        """Check net margin (negative = CRITICAL, low = INFO)."""
        flags = []
        
        if not revenue.values or not net_income.values:
            return flags
        
        net_margin = (net_income.values[-1] / revenue.values[-1]) * 100 if revenue.values[-1] != 0 else 0
        
        if net_margin < 0:
            flags.append(RedFlag(
                category="Profitability",
                severity=RedFlagSeverity.CRITICAL,
                title="Negative Net Margin (Losses)",
                description="Company losing money - burning through capital",
                current_value=net_margin,
                threshold=0.0,
                period=period_str
            ))
        else:
            # Apply sector-specific threshold for low margins
            margin_threshold = self._adjust_threshold(
                self.thresholds.net_margin_percent,
                self.sector_profile.net_margin_multiplier
            )
            
            if 0 < net_margin < margin_threshold:
                flags.append(RedFlag(
                    category="Profitability",
                    severity=RedFlagSeverity.INFO,
                    title="Low Net Margin",
                    description=f"Thin margins for {self.sector} sector - vulnerable to downturns",
                    current_value=net_margin,
                    threshold=margin_threshold,
                    period=period_str
                ))
        
        return flags
    
    def _check_roe(self, net_income, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check ROE (negative = CRITICAL, low = INFO)."""
        flags = []
        
        try:
            equity = self.balance_sheet.get_stockholders_equity(granularity)
            
            if not net_income.values or not equity.values or len(equity.values) < 2:
                return flags
            
            avg_equity = (equity.values[-1] + equity.values[-2]) / 2
            if avg_equity > 0:
                roe = (net_income.values[-1] / avg_equity) * 100
                
                if roe < 0:
                    flags.append(RedFlag(
                        category="Profitability",
                        severity=RedFlagSeverity.CRITICAL,
                        title="Negative Return on Equity",
                        description="Destroying shareholder value - losses eating into equity",
                        current_value=roe,
                        threshold=0.0,
                        period=period_str
                    ))
                else:
                    # Apply sector-specific threshold for low ROE
                    roe_threshold = self._adjust_threshold(
                        self.thresholds.roe_percent,
                        self.sector_profile.roe_multiplier
                    )
                    
                    if 0 < roe < roe_threshold:
                        flags.append(RedFlag(
                            category="Profitability",
                            severity=RedFlagSeverity.INFO,
                            title="Low Return on Equity",
                            description=f"Poor capital efficiency for {self.sector} sector - not generating adequate returns",
                            current_value=roe,
                            threshold=roe_threshold,
                            period=period_str
                        ))
        except ValueError:
            pass
        
        return flags
    
    def _check_revenue_vs_eps_growth(self, revenue, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check for revenue growth without EPS growth."""
        flags = []
        
        try:
            eps = self.income_statement.get_earnings_per_share_diluted(granularity)
            
            if len(revenue.values) >= 3 and len(eps.values) >= 3:
                rev_growth = ((revenue.values[-1] - revenue.values[-3]) / revenue.values[-3]) * 100 if revenue.values[-3] != 0 else 0
                eps_growth = ((eps.values[-1] - eps.values[-3]) / eps.values[-3]) * 100 if eps.values[-3] != 0 else 0
                
                if rev_growth > 10 and eps_growth < rev_growth / 2:  # Revenue up >10%, EPS lags
                    flags.append(RedFlag(
                        category="Profitability",
                        severity=RedFlagSeverity.WARNING,
                        title="Revenue Growth Without EPS Growth",
                        description="Margin compression or share dilution eating into profits",
                        current_value=eps_growth,
                        threshold=rev_growth,
                        period=f"{revenue.periods[-3].year}-{period_str.split()[0]}"
                    ))
        except ValueError:
            pass
        
        return flags
    
    def _check_margin_volatility(self, revenue, gross_profit) -> List[RedFlag]:
        """Check for volatile gross margins."""
        flags = []
        
        if len(gross_profit.values) >= 4 and len(revenue.values) >= 4:
            margins = [(gross_profit.values[i] / revenue.values[i]) * 100 
                      for i in range(-4, 0) if revenue.values[i] != 0]
            
            if len(margins) == 4:
                # Calculate standard deviation
                mean_margin = sum(margins) / len(margins)
                variance = sum((m - mean_margin) ** 2 for m in margins) / len(margins)
                std_dev = variance ** 0.5
                
                if std_dev > self.thresholds.gross_margin_std_dev:
                    flags.append(RedFlag(
                        category="Profitability",
                        severity=RedFlagSeverity.INFO,
                        title="Highly Volatile Gross Margin",
                        description="Unstable business - margins fluctuate significantly",
                        current_value=std_dev,
                        threshold=self.thresholds.gross_margin_std_dev,
                        period=f"Last 4 periods"
                    ))
        
        return flags
