"""
Cash flow quality analyzer - OCF, FCF, dilution, and sustainability checks.
"""
from typing import List

from edgarito.enums.granularity import Granularity
from edgarito.schemas.red_flags import RedFlag, RedFlagSeverity
from .base_analyzer import BaseAnalyzer


class CashFlowAnalyzer(BaseAnalyzer):
    """Analyzes cash flow for quality, sustainability, and dilution red flags."""
    
    def analyze(self, granularity: Granularity) -> List[RedFlag]:
        """Analyze cash flow for red flags."""
        flags = []
        
        try:
            ocf = self.cash_flow.get_operating_cash_flow(granularity)
            net_income = self.income_statement.get_net_income(granularity)
            
            if not ocf.values or not net_income.values:
                return flags
            
            latest_period = ocf.periods[-1]
            period_str = f"{latest_period.year} {latest_period.fp.value}"
            
            # Run all cash flow checks
            flags.extend(self._check_negative_ocf(ocf, granularity, period_str))
            flags.extend(self._check_cash_conversion(ocf, net_income, granularity, period_str))
            flags.extend(self._check_negative_fcf(ocf, granularity, period_str))
            flags.extend(self._check_dividends_vs_fcf(ocf, granularity, period_str))
            flags.extend(self._check_capex_growth(ocf, granularity, period_str))
            flags.extend(self._check_stock_compensation(ocf, granularity, period_str))
            flags.extend(self._check_share_issuance(granularity))
            
        except ValueError:
            pass
        
        return flags
    
    def _check_negative_ocf(self, ocf, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check for negative operating cash flow."""
        flags = []
        
        if not ocf.values:
            return flags
        
        if granularity == Granularity.ANNUAL:
            # Annual negative OCF is very concerning
            if ocf.values[-1] < 0:
                flags.append(RedFlag(
                    category="Cash Flow",
                    severity=RedFlagSeverity.CRITICAL,
                    title="Negative Operating Cash Flow (Full Year)",
                    description="Burning cash from operations for entire year - unsustainable",
                    current_value=ocf.values[-1] / 1e9,
                    threshold=0.0,
                    period=period_str
                ))
        else:
            # Quarterly: only flag if negative for 2+ consecutive quarters
            if len(ocf.values) >= 2 and ocf.values[-1] < 0 and ocf.values[-2] < 0:
                flags.append(RedFlag(
                    category="Cash Flow",
                    severity=RedFlagSeverity.CRITICAL,
                    title="Negative Operating Cash Flow (Multiple Quarters)",
                    description="Burning cash from operations for 2+ quarters - concerning trend",
                    current_value=ocf.values[-1] / 1e9,
                    threshold=0.0,
                    period=f"Last 2 quarters"
                ))
        
        return flags
    
    def _check_cash_conversion(self, ocf, net_income, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check OCF/FCF vs Net Income with CapEx-aware logic."""
        flags = []
        
        try:
            capex = self.cash_flow.get_capital_expenditures(granularity)
            
            if len(ocf.values) >= 3 and len(net_income.values) >= 3 and capex.values and len(capex.values) >= 3:
                # Calculate Free Cash Flow (FCF = OCF + CapEx, since CapEx is negative)
                fcf_values = [ocf.values[i] + capex.values[i] for i in range(-3, 0)]
                
                # Check OCF vs Net Income
                ocf_below_count = sum(1 for i in range(-3, 0) 
                                     if ocf.values[i] < net_income.values[i])
                
                # Calculate average ratios
                avg_ocf = sum(ocf.values[-3:]) / 3
                avg_ni = sum(net_income.values[-3:]) / 3
                avg_fcf = sum(fcf_values) / 3
                
                ocf_ni_ratio = avg_ocf / avg_ni if avg_ni > 0 else 0
                fcf_ni_ratio = avg_fcf / avg_ni if avg_ni > 0 else 0
                
                # Apply sector-specific adjustments
                ocf_threshold = self._adjust_threshold(0.8, self.sector_profile.ocf_ni_ratio_multiplier)
                fcf_threshold = self._adjust_threshold(0.7, self.sector_profile.fcf_ni_ratio_multiplier)
                
                # Only flag if BOTH conditions are true:
                # 1. OCF significantly below Net Income (< sector-adjusted threshold)
                # 2. FCF is also weak (< sector-adjusted threshold) - indicating it's NOT just healthy CapEx
                if ocf_below_count >= 2 and ocf_ni_ratio < ocf_threshold and fcf_ni_ratio < fcf_threshold:
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.WARNING,
                        title="Weak Cash Conversion (OCF & FCF Below Net Income)",
                        description=f"Earnings quality concerns - profits not converting to cash even after CapEx (avg FCF {fcf_ni_ratio*100:.0f}% of Net Income, {self.sector} sector threshold: {fcf_threshold*100:.0f}%)",
                        current_value=fcf_ni_ratio * 100,
                        threshold=fcf_threshold * 100,
                        period=f"Last 3 periods"
                    ))
        except (ValueError, IndexError):
            # If we can't get CapEx data, fall back to simpler OCF check
            if len(ocf.values) >= 3 and len(net_income.values) >= 3:
                ocf_below_count = sum(1 for i in range(-3, 0) 
                                     if ocf.values[i] < net_income.values[i])
                
                avg_ocf = sum(ocf.values[-3:]) / 3
                avg_ni = sum(net_income.values[-3:]) / 3
                ocf_ni_ratio = avg_ocf / avg_ni if avg_ni > 0 else 0
                
                if ocf_below_count >= 2 and ocf_ni_ratio < 0.8:
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.WARNING,
                        title="OCF Consistently Below Net Income",
                        description="Earnings quality concerns - profits not converting to cash (avg OCF < 80% of Net Income)",
                        current_value=ocf_ni_ratio * 100,
                        threshold=80.0,
                        period=f"Last 3 periods"
                    ))
        
        return flags
    
    def _check_negative_fcf(self, ocf, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check for negative free cash flow for 2+ years."""
        flags = []
        
        try:
            capex = self.cash_flow.get_capital_expenditures(granularity)
            
            if capex.values and len(ocf.values) >= 2:
                # CapEx is usually negative, so FCF = OCF + CapEx (adding negative)
                fcf_values = [ocf.values[i] + capex.values[i] 
                             for i in range(min(len(ocf.values), len(capex.values)))]
                
                if len(fcf_values) >= 2 and fcf_values[-1] < 0 and fcf_values[-2] < 0:
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.CRITICAL,
                        title="Negative Free Cash Flow (2+ Years)",
                        description="Burning cash - cannot sustain operations without external funding",
                        current_value=fcf_values[-1] / 1e9,
                        period=period_str
                    ))
        except ValueError:
            pass
        
        return flags
    
    def _check_dividends_vs_fcf(self, ocf, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check if dividends exceed free cash flow."""
        flags = []
        
        try:
            capex = self.cash_flow.get_capital_expenditures(granularity)
            dividends = self.cash_flow.get_dividends_paid(granularity)
            
            if capex.values and dividends.values and ocf.values:
                fcf = ocf.values[-1] + capex.values[-1]  # CapEx is negative
                div = abs(dividends.values[-1])  # Dividends are negative in cash flow
                
                if fcf > 0 and div > fcf:
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.WARNING,
                        title="Dividends Exceed Free Cash Flow",
                        description="Unsustainable payout - paying more than generating",
                        current_value=div / fcf,
                        threshold=1.0,
                        period=period_str
                    ))
        except ValueError:
            pass
        
        return flags
    
    def _check_capex_growth(self, ocf, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check if CapEx is rising faster than OCF."""
        flags = []
        
        try:
            capex = self.cash_flow.get_capital_expenditures(granularity)
            
            if len(ocf.values) >= 3 and len(capex.values) >= 3:
                # CapEx is negative, so we use absolute values for growth
                ocf_growth = abs((ocf.values[-1] - ocf.values[-3]) / ocf.values[-3]) if ocf.values[-3] != 0 else 0
                capex_growth = abs((capex.values[-1] - capex.values[-3]) / capex.values[-3]) if capex.values[-3] != 0 else 0
                
                if capex_growth > ocf_growth * 1.5 and capex_growth > 0.2:  # 50% faster + 20% threshold
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.INFO,
                        title="CapEx Growing Faster than OCF",
                        description="Heavy reinvestment - monitor returns on capital",
                        current_value=capex_growth * 100,
                        threshold=ocf_growth * 100,
                        period=f"{ocf.periods[-3].year}-{period_str.split()[0]}"
                    ))
        except ValueError:
            pass
        
        return flags
    
    def _check_stock_compensation(self, ocf, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check if stock-based compensation is > threshold % of OCF."""
        flags = []
        
        try:
            sbc = self.cash_flow.get_stock_based_compensation(granularity)
            
            if sbc.values and ocf.values:
                sbc_val = sbc.values[-1]
                ocf_val = ocf.values[-1]
                
                if ocf_val > 0:
                    sbc_pct = sbc_val / ocf_val
                    if sbc_pct > (self.thresholds.stock_comp_percent_ocf / 100):
                        flags.append(RedFlag(
                            category="Cash Flow",
                            severity=RedFlagSeverity.WARNING,
                            title="High Stock-Based Compensation",
                            description="Dilution disguised as expense - impacts shareholder value",
                            current_value=sbc_pct * 100,
                            threshold=self.thresholds.stock_comp_percent_ocf,
                            period=period_str
                        ))
        except ValueError:
            pass
        
        return flags
    
    def _check_share_issuance(self, granularity: Granularity) -> List[RedFlag]:
        """Check for frequent share issuance."""
        flags = []
        
        try:
            stock_issued = self.cash_flow.get_proceeds_from_stock_issuance(granularity)
            
            if len(stock_issued.values) >= 3:
                # Check if issued stock in 2 of last 3 periods
                issuance_count = sum(1 for i in range(-3, 0) 
                                    if stock_issued.values[i] > 1e6)  # > $1M threshold
                
                if issuance_count >= 2:
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.INFO,
                        title="Frequent Share Issuance",
                        description="Funding growth via dilution, not profits",
                        current_value=stock_issued.values[-1] / 1e9,
                        period="Last 3 periods"
                    ))
        except ValueError:
            pass
        
        return flags
