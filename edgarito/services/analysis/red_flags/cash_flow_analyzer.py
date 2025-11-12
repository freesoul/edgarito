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
        """Check OCF/FCF vs Net Income with CapEx-aware logic using TTM data."""
        flags = []
        
        try:
            capex = self.cash_flow.get_capital_expenditures(granularity)
            
            # Use TTM (last 4 quarters) for more accurate cash conversion analysis
            if len(ocf.values) >= 4 and len(net_income.values) >= 4 and capex.values and len(capex.values) >= 4:
                # Calculate TTM values
                ttm_ocf = sum(ocf.values[-4:])
                ttm_ni = sum(net_income.values[-4:])
                ttm_capex = sum(capex.values[-4:])  # CapEx is negative
                ttm_fcf = ttm_ocf + ttm_capex  # FCF = OCF + CapEx (adding negative)
                
                # Calculate ratios
                ocf_ni_ratio = ttm_ocf / ttm_ni if ttm_ni > 0 else 0
                fcf_ni_ratio = ttm_fcf / ttm_ni if ttm_ni > 0 else 0
                
                # Apply sector-specific adjustments
                ocf_threshold = self._adjust_threshold(0.8, self.sector_profile.ocf_ni_ratio_multiplier)
                fcf_threshold = self._adjust_threshold(0.7, self.sector_profile.fcf_ni_ratio_multiplier)
                
                # Only flag if BOTH conditions are true:
                # 1. OCF significantly below Net Income (< sector-adjusted threshold)
                # 2. FCF is also weak (< sector-adjusted threshold) - indicating it's NOT just healthy CapEx
                if ocf_ni_ratio < ocf_threshold and fcf_ni_ratio < fcf_threshold:
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.WARNING,
                        title="Weak Cash Conversion (OCF & FCF Below Net Income)",
                        description=f"Earnings quality concerns - profits not converting to cash even after CapEx (TTM FCF {fcf_ni_ratio*100:.0f}% of Net Income, {self.sector} sector threshold: {fcf_threshold*100:.0f}%)",
                        current_value=fcf_ni_ratio * 100,
                        threshold=fcf_threshold * 100,
                        period=f"TTM"
                    ))
        except (ValueError, IndexError):
            # If we can't get CapEx data, fall back to simpler OCF check
            if len(ocf.values) >= 4 and len(net_income.values) >= 4:
                ttm_ocf = sum(ocf.values[-4:])
                ttm_ni = sum(net_income.values[-4:])
                ocf_ni_ratio = ttm_ocf / ttm_ni if ttm_ni > 0 else 0
                
                if ocf_ni_ratio < 0.8:
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.WARNING,
                        title="OCF Consistently Below Net Income",
                        description="Earnings quality concerns - profits not converting to cash (TTM OCF < 80% of Net Income)",
                        current_value=ocf_ni_ratio * 100,
                        threshold=80.0,
                        period=f"TTM"
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
        """Check if dividends exceed free cash flow using TTM data."""
        flags = []
        
        try:
            capex = self.cash_flow.get_capital_expenditures(granularity)
            dividends = self.cash_flow.get_dividends_paid(granularity)
            
            # Use TTM (sum of last 4 quarters) for accurate dividend payout analysis
            if capex.values and dividends.values and ocf.values:
                if len(ocf.values) >= 4 and len(capex.values) >= 4 and len(dividends.values) >= 4:
                    # Calculate TTM values
                    ttm_ocf = sum(ocf.values[-4:])
                    ttm_capex = sum(capex.values[-4:])  # CapEx is negative
                    ttm_fcf = ttm_ocf + ttm_capex
                    ttm_div = abs(sum(dividends.values[-4:]))  # Dividends are negative in cash flow
                    
                    if ttm_fcf > 0 and ttm_div > ttm_fcf:
                        flags.append(RedFlag(
                            category="Cash Flow",
                            severity=RedFlagSeverity.WARNING,
                            title="Dividends Exceed Free Cash Flow",
                            description="Unsustainable payout - paying more than generating",
                            current_value=ttm_div / ttm_fcf,
                            threshold=1.0,
                            period="TTM"
                        ))
        except ValueError:
            pass
        
        return flags
    
    def _check_capex_growth(self, ocf, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check if CapEx is rising faster than OCF using year-over-year comparison."""
        flags = []
        
        try:
            capex = self.cash_flow.get_capital_expenditures(granularity)
            
            # For quarterly data, compare TTM values year-over-year to avoid quarterly volatility
            if granularity == Granularity.QUARTERLY and len(ocf.values) >= 8 and len(capex.values) >= 8:
                # Current TTM (last 4 quarters)
                current_ocf = sum(ocf.values[-4:])
                current_capex = abs(sum(capex.values[-4:]))  # Use absolute value for growth calc
                
                # Prior year TTM (quarters 5-8 from end)
                prior_ocf = sum(ocf.values[-8:-4])
                prior_capex = abs(sum(capex.values[-8:-4]))
                
                # Calculate year-over-year growth
                ocf_growth = ((current_ocf - prior_ocf) / prior_ocf) if prior_ocf != 0 else 0
                capex_growth = ((current_capex - prior_capex) / prior_capex) if prior_capex != 0 else 0
                
                if capex_growth > ocf_growth * 1.5 and capex_growth > 0.2:  # 50% faster + 20% threshold
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.INFO,
                        title="CapEx Growing Faster than OCF",
                        description="Heavy reinvestment - monitor returns on capital",
                        current_value=capex_growth * 100,
                        threshold=ocf_growth * 100,
                        period=f"TTM YoY"
                    ))
            elif len(ocf.values) >= 3 and len(capex.values) >= 3:
                # Annual data: compare 2 years ago to current
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
        """
        Check for frequent/material share issuance vs buybacks.
        
        Net share activity = Stock Issuance + Stock Repurchases (repurchases are negative)
        
        Positive net = Dilution (issuing more shares than buying back)
        Negative net = Buybacks (repurchasing more shares than issuing)
        
        Only flag dilution if:
        - Net issuance > $50M per period
        - Occurred in 2 of last 3 periods
        
        Recognize buybacks as positive signal if:
        - Net repurchase > $100M per period
        - Consistent over 3 periods
        """
        flags = []
        
        try:
            stock_issued = self.cash_flow.get_proceeds_from_stock_issuance(granularity)
            
            # Try to get repurchase data
            try:
                stock_repurchased = self.cash_flow.get_stock_repurchases(granularity)
                has_repurchase_data = len(stock_repurchased.values) >= 3
            except ValueError:
                has_repurchase_data = False
                stock_repurchased = None
            
            if not stock_issued.values or len(stock_issued.values) < 3:
                return flags
            
            # Calculate net share activity for last 3 periods
            if has_repurchase_data:
                net_activities = []
                for i in range(-3, 0):
                    try:
                        issued = stock_issued.values[i]
                        repurchased = stock_repurchased.values[i]  # Already negative in SEC data
                        net = issued + repurchased  # Positive = dilution, Negative = net buybacks
                        net_activities.append(net)
                    except IndexError:
                        continue
                
                if not net_activities:
                    return flags
                
                # Check for material dilution (net issuance > $50M in 2 of last 3 periods)
                dilution_count = sum(1 for net in net_activities if net > 50e6)
                
                # Check for consistent buybacks (net repurchase > $100M in all 3 periods)
                buyback_count = sum(1 for net in net_activities if net < -100e6)
                
                if buyback_count >= 3:
                    # Positive signal - consistent share buybacks
                    total_net_buyback = sum(net_activities)
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.INFO,
                        title="Consistent Share Buybacks",
                        description="Returning capital to shareholders - reducing share count via repurchases",
                        current_value=abs(total_net_buyback) / 1e9,
                        period="Last 3 periods"
                    ))
                elif dilution_count >= 2:
                    # Negative signal - material dilution
                    latest_net = net_activities[-1]
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.INFO,
                        title="Frequent Share Issuance",
                        description="Net dilution - issuing more shares than repurchasing, funding growth via equity",
                        current_value=latest_net / 1e9,
                        period="Last 3 periods"
                    ))
            else:
                # Fallback: only have issuance data, no repurchase data
                # Check if issued stock in 2 of last 3 periods
                # Use higher threshold ($50M) to avoid flagging minor employee option exercises
                issuance_count = sum(1 for i in range(-3, 0) 
                                    if stock_issued.values[i] > 50e6)
                
                if issuance_count >= 2:
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.INFO,
                        title="Frequent Share Issuance",
                        description="Issuing shares regularly - may be funding growth via dilution (buyback data unavailable)",
                        current_value=stock_issued.values[-1] / 1e9,
                        period="Last 3 periods"
                    ))
        except Exception:
            pass  # Return empty flags on error
        
        return flags
