"""
Valuation metrics analyzer - P/S, P/E, P/B, PEG, EV/EBITDA, short interest, insider ownership.
"""
from typing import List

from edgarito.enums.granularity import Granularity
from edgarito.schemas.red_flags import RedFlag, RedFlagSeverity
from .base_analyzer import BaseAnalyzer


class ValuationAnalyzer(BaseAnalyzer):
    """Analyzes valuation and market perception red flags."""
    
    def analyze(self, granularity: Granularity) -> List[RedFlag]:
        """Analyze valuation metrics for red flags."""
        flags = []
        
        # All valuation checks require market cap
        if not self.market_cap or self.market_cap <= 0:
            return flags
        
        try:
            revenue = self.income_statement.get_revenue(granularity)
            
            if not revenue.values:
                return flags
            
            latest_period = revenue.periods[-1]
            period_str = f"{latest_period.year} {latest_period.fp.value}"
            
            # Run all valuation checks
            flags.extend(self._check_price_to_sales(revenue, period_str))
            flags.extend(self._check_pe_ratio(revenue, granularity, period_str))
            flags.extend(self._check_price_to_book(period_str))
            flags.extend(self._check_dividend_yield(period_str))
            
            # Yahoo Finance enhanced checks (if market_data available)
            if self.market_data:
                flags.extend(self._check_peg_ratio(period_str))
                flags.extend(self._check_ev_to_ebitda(period_str))
                flags.extend(self._check_short_interest(period_str))
                flags.extend(self._check_insider_ownership(period_str))
            
        except ValueError:
            pass
        
        return flags
    
    def _check_price_to_sales(self, revenue, period_str: str) -> List[RedFlag]:
        """Check P/S ratio for overvaluation using TTM revenue."""
        flags = []
        
        # P/S should use TTM (Trailing Twelve Months) revenue, not a single quarter
        # Calculate TTM by summing last 4 quarters if available
        ttm_revenue = None
        if len(revenue.values) >= 4:
            ttm_revenue = sum(revenue.values[-4:])
        elif len(revenue.values) >= 1:
            # Fallback: if we don't have 4 quarters, estimate TTM by multiplying latest quarter by 4
            # This is less accurate but better than using a single quarter directly
            ttm_revenue = revenue.values[-1] * 4
        
        if ttm_revenue and ttm_revenue > 0:
            price_to_sales = self.market_cap / ttm_revenue
            
            if price_to_sales > self.thresholds.price_to_sales:
                flags.append(RedFlag(
                    category="Valuation",
                    severity=RedFlagSeverity.WARNING,
                    title="High Price-to-Sales Ratio",
                    description=f"Overvalued - market cap significantly exceeds annual revenue (P/S > {self.thresholds.price_to_sales})",
                    current_value=price_to_sales,
                    threshold=self.thresholds.price_to_sales,
                    period=period_str
                ))
        
        return flags
    
    def _check_pe_ratio(self, revenue, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check P/E ratio for value traps using TTM net income."""
        flags = []
        
        try:
            net_income = self.income_statement.get_net_income(granularity)
            
            # P/E should use TTM net income, not a single quarter
            ttm_net_income = None
            if len(net_income.values) >= 4:
                ttm_net_income = sum(net_income.values[-4:])
            elif len(net_income.values) >= 1:
                ttm_net_income = net_income.values[-1] * 4
            
            if ttm_net_income and ttm_net_income > 0:
                pe_ratio = self.market_cap / ttm_net_income
                
                # Value trap: low P/E without growth
                if pe_ratio < self.thresholds.pe_ratio_low:
                    # Check revenue growth
                    if len(revenue.values) >= 3:
                        rev_growth = ((revenue.values[-1] - revenue.values[-3]) / revenue.values[-3]) * 100 if revenue.values[-3] != 0 else 0
                        
                        if rev_growth < self.thresholds.revenue_growth_for_rd_check:
                            flags.append(RedFlag(
                                category="Valuation",
                                severity=RedFlagSeverity.WARNING,
                                title="Low P/E with Weak Growth (Value Trap)",
                                description="Cheap for a reason - low valuation but no growth",
                                current_value=pe_ratio,
                                threshold=self.thresholds.pe_ratio_low,
                                period=period_str
                            ))
        except ValueError:
            pass
        
        return flags
    
    def _check_price_to_book(self, period_str: str) -> List[RedFlag]:
        """Check P/B ratio for overvaluation without returns."""
        flags = []
        
        try:
            equity = self.balance_sheet.get_stockholders_equity(Granularity.QUARTERLY)
            
            if equity.values[-1] > 0:
                price_to_book = self.market_cap / equity.values[-1]
                
                # High P/B without high ROE
                if price_to_book > self.thresholds.price_to_book:
                    # Check ROE using TTM net income
                    try:
                        net_income = self.income_statement.get_net_income(Granularity.QUARTERLY)
                        
                        # Calculate TTM net income for ROE
                        ttm_net_income = None
                        if len(net_income.values) >= 4:
                            ttm_net_income = sum(net_income.values[-4:])
                        elif len(net_income.values) >= 1:
                            ttm_net_income = net_income.values[-1] * 4
                        
                        if ttm_net_income and ttm_net_income > 0:
                            roe = (ttm_net_income / equity.values[-1]) * 100
                            
                            if roe < self.thresholds.roe_for_pb_check:
                                flags.append(RedFlag(
                                    category="Valuation",
                                    severity=RedFlagSeverity.INFO,
                                    title="High P/B Without High ROE",
                                    description=f"Overvalued - P/B > {self.thresholds.price_to_book} but ROE < {self.thresholds.roe_for_pb_check}%",
                                    current_value=price_to_book,
                                    threshold=self.thresholds.price_to_book,
                                    period=period_str
                                ))
                    except ValueError:
                        pass
        except ValueError:
            pass
        
        return flags
    
    def _check_dividend_yield(self, period_str: str) -> List[RedFlag]:
        """Check for unsustainably high dividend yield."""
        flags = []
        
        if not self.market_cap:
            return flags  # Need market cap to calculate yield
        
        try:
            dividends = self.cash_flow.get_dividends_paid(Granularity.ANNUAL)
            
            if dividends.values and dividends.values[-1] < 0:  # Dividends are negative in cash flow
                annual_dividend = abs(dividends.values[-1])
                dividend_yield = (annual_dividend / self.market_cap) * 100
                
                if dividend_yield > self.thresholds.dividend_yield:
                    flags.append(RedFlag(
                        category="Valuation",
                        severity=RedFlagSeverity.WARNING,
                        title="Unsustainably High Dividend Yield",
                        description=f"Potential dividend cut - yield > {self.thresholds.dividend_yield}% often unsustainable",
                        current_value=dividend_yield,
                        threshold=self.thresholds.dividend_yield,
                        period=period_str
                    ))
        except (ValueError, IndexError):
            pass
        
        return flags
    
    def _check_peg_ratio(self, period_str: str) -> List[RedFlag]:
        """Check PEG ratio from Yahoo Finance data."""
        flags = []
        
        if self.market_data and self.market_data.peg_ratio:
            if self.market_data.peg_ratio > self.thresholds.peg_ratio:
                flags.append(RedFlag(
                    category="Valuation",
                    severity=RedFlagSeverity.INFO,
                    title="High PEG Ratio",
                    description=f"Growth not justifying valuation - PEG > {self.thresholds.peg_ratio}",
                    current_value=self.market_data.peg_ratio,
                    threshold=self.thresholds.peg_ratio,
                    period=period_str
                ))
        
        return flags
    
    def _check_ev_to_ebitda(self, period_str: str) -> List[RedFlag]:
        """Check EV/EBITDA from Yahoo Finance data."""
        flags = []
        
        if self.market_data and self.market_data.ev_to_ebitda:
            if self.market_data.ev_to_ebitda > self.thresholds.ev_to_ebitda:
                flags.append(RedFlag(
                    category="Valuation",
                    severity=RedFlagSeverity.INFO,
                    title="High EV/EBITDA",
                    description=f"Enterprise value high relative to earnings - EV/EBITDA > {self.thresholds.ev_to_ebitda}",
                    current_value=self.market_data.ev_to_ebitda,
                    threshold=self.thresholds.ev_to_ebitda,
                    period=period_str
                ))
        
        return flags
    
    def _check_short_interest(self, period_str: str) -> List[RedFlag]:
        """Check short interest percentage from Yahoo Finance data."""
        flags = []
        
        if self.market_data and self.market_data.short_percent_float:
            if self.market_data.short_percent_float > self.thresholds.short_interest_percent:
                flags.append(RedFlag(
                    category="Valuation",
                    severity=RedFlagSeverity.WARNING,
                    title="High Short Interest",
                    description=f"Market skepticism - {self.market_data.short_percent_float:.1f}% of float shorted (> {self.thresholds.short_interest_percent}%)",
                    current_value=self.market_data.short_percent_float,
                    threshold=self.thresholds.short_interest_percent,
                    period=period_str
                ))
        
        return flags
    
    def _check_insider_ownership(self, period_str: str) -> List[RedFlag]:
        """Check insider ownership percentage from Yahoo Finance data."""
        flags = []
        
        if self.market_data and self.market_data.insider_ownership_percent is not None:
            if self.market_data.insider_ownership_percent < self.thresholds.insider_ownership_percent:
                flags.append(RedFlag(
                    category="Valuation",
                    severity=RedFlagSeverity.INFO,
                    title="Low Insider Ownership",
                    description=f"Management not aligned - insiders own < {self.thresholds.insider_ownership_percent}% of shares",
                    current_value=self.market_data.insider_ownership_percent,
                    threshold=self.thresholds.insider_ownership_percent,
                    period=period_str
                ))
        
        return flags
