"""
Balance sheet health analyzer - debt, liquidity, solvency checks.
"""
from typing import List

from edgarito.enums.granularity import Granularity
from edgarito.schemas.red_flags import RedFlag, RedFlagSeverity
from .base_analyzer import BaseAnalyzer


class BalanceSheetAnalyzer(BaseAnalyzer):
    """Analyzes balance sheet for debt, liquidity, and solvency red flags."""
    
    def analyze(self, granularity: Granularity) -> List[RedFlag]:
        """Analyze balance sheet for red flags."""
        flags = []
        
        # Get latest period data
        try:
            total_debt = self.balance_sheet.get_total_debt(granularity)
            equity = self.balance_sheet.get_stockholders_equity(granularity)
            total_assets = self.balance_sheet.get_total_assets(granularity)
            total_liabilities = self.balance_sheet.get_total_liabilities(granularity)
            current_assets = self.balance_sheet.get_current_assets(granularity)
            current_liabilities = self.balance_sheet.get_current_liabilities(granularity)
            
            if not total_debt.values or not equity.values:
                return flags
            
            latest_period = total_debt.periods[-1]
            period_str = f"{latest_period.year} {latest_period.fp.value}"
            
            # Run all balance sheet checks
            flags.extend(self._check_debt_to_equity(total_debt, equity, period_str))
            flags.extend(self._check_debt_vs_market_cap(total_debt, period_str))
            flags.extend(self._check_current_ratio(current_assets, current_liabilities, period_str))
            flags.extend(self._check_quick_ratio(current_liabilities, granularity, period_str))
            flags.extend(self._check_tangible_book_value(equity, granularity, period_str))
            flags.extend(self._check_interest_coverage(granularity, period_str))
            
        except Exception:
            pass  # Return empty flags on error
        
        return flags
    
    def _check_debt_to_equity(self, total_debt, equity, period_str: str) -> List[RedFlag]:
        """Check debt-to-equity ratio with tiered severity."""
        flags = []
        
        if not total_debt.values or not equity.values:
            return flags
        
        debt = total_debt.values[-1]
        eq = equity.values[-1]
        
        # Check for negative equity first (most critical)
        if eq <= 0:
            flags.append(RedFlag(
                category="Balance Sheet",
                severity=RedFlagSeverity.CRITICAL,
                title="Negative or Zero Stockholders' Equity",
                description="Technically insolvent - liabilities exceed assets, bankruptcy risk",
                current_value=eq / 1e9,
                period=period_str
            ))
        elif eq > 0:
            debt_to_equity = debt / eq
            # Apply sector-specific D/E threshold adjustments
            de_warning = self._adjust_threshold(
                self.thresholds.debt_to_equity_ratio_warning,
                self.sector_profile.debt_to_equity_multiplier
            )
            de_critical = self._adjust_threshold(
                self.thresholds.debt_to_equity_ratio_critical,
                self.sector_profile.debt_to_equity_multiplier
            )
            
            if debt_to_equity > de_critical:
                flags.append(RedFlag(
                    category="Balance Sheet",
                    severity=RedFlagSeverity.CRITICAL,
                    title="Extremely High Debt-to-Equity Ratio",
                    description=f"Severe overleveraging for {self.sector} sector - debt more than {de_critical:.1f}x equity",
                    current_value=debt_to_equity,
                    threshold=de_critical,
                    period=period_str
                ))
            elif debt_to_equity > de_warning:
                flags.append(RedFlag(
                    category="Balance Sheet",
                    severity=RedFlagSeverity.WARNING,
                    title="High Debt-to-Equity Ratio",
                    description=f"Capital structure too leveraged for {self.sector} sector - debt exceeds equity",
                    current_value=debt_to_equity,
                    threshold=de_warning,
                    period=period_str
                ))
        
        return flags
    
    def _check_debt_vs_market_cap(self, total_debt, period_str: str) -> List[RedFlag]:
        """Check if debt exceeds market capitalization."""
        flags = []
        
        if self.market_cap and total_debt.values:
            debt = total_debt.values[-1]
            if debt > self.market_cap:
                flags.append(RedFlag(
                    category="Balance Sheet",
                    severity=RedFlagSeverity.CRITICAL,
                    title="Debt exceeds Market Capitalization",
                    description=f"Total debt (${debt:,.0f}) exceeds company's market value (${self.market_cap:,.0f})",
                    current_value=debt,
                    threshold=self.market_cap,
                    period=period_str
                ))
        
        return flags
    
    def _check_current_ratio(self, current_assets, current_liabilities, period_str: str) -> List[RedFlag]:
        """Check current ratio with tiered severity."""
        flags = []
        
        if not current_assets.values or not current_liabilities.values:
            return flags
        
        curr_assets = current_assets.values[-1]
        curr_liab = current_liabilities.values[-1]
        
        if curr_liab > 0:
            current_ratio = curr_assets / curr_liab
            # Apply sector-specific adjustments
            cr_critical = self._adjust_threshold(
                self.thresholds.current_ratio_critical,
                self.sector_profile.current_ratio_multiplier
            )
            cr_warning = self._adjust_threshold(
                self.thresholds.current_ratio_warning,
                self.sector_profile.current_ratio_multiplier
            )
            
            if current_ratio < cr_critical:
                flags.append(RedFlag(
                    category="Balance Sheet",
                    severity=RedFlagSeverity.CRITICAL,
                    title="Critically Low Current Ratio",
                    description=f"Severe liquidity crisis for {self.sector} sector - current assets below current liabilities",
                    current_value=current_ratio,
                    threshold=cr_critical,
                    period=period_str
                ))
            elif current_ratio < cr_warning:
                flags.append(RedFlag(
                    category="Balance Sheet",
                    severity=RedFlagSeverity.WARNING,
                    title="Low Current Ratio",
                    description=f"Potential liquidity problems for {self.sector} sector - limited ability to meet short-term obligations",
                    current_value=current_ratio,
                    threshold=cr_warning,
                    period=period_str
                ))
        
        return flags
    
    def _check_quick_ratio(self, current_liabilities, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check quick ratio with tiered severity."""
        flags = []
        
        try:
            cash = self.balance_sheet.get_cash_and_equivalents(granularity)
            receivables = self.balance_sheet.get_accounts_receivable(granularity)
            
            if not cash.values or not receivables.values or not current_liabilities.values:
                return flags
            
            quick_assets = cash.values[-1] + receivables.values[-1]
            curr_liab = current_liabilities.values[-1]
            
            if curr_liab > 0:
                quick_ratio = quick_assets / curr_liab
                
                # Apply sector-specific thresholds
                qr_critical = self._adjust_threshold(
                    self.thresholds.quick_ratio_critical,
                    self.sector_profile.quick_ratio_multiplier
                )
                qr_warning = self._adjust_threshold(
                    self.thresholds.quick_ratio_warning,
                    self.sector_profile.quick_ratio_multiplier
                )
                
                if quick_ratio < qr_critical:
                    flags.append(RedFlag(
                        category="Balance Sheet",
                        severity=RedFlagSeverity.CRITICAL,
                        title="Critically Low Quick Ratio",
                        description=f"Severe liquidity for {self.sector} sector - cannot meet immediate obligations without selling inventory",
                        current_value=quick_ratio,
                        threshold=qr_critical,
                        period=period_str
                    ))
                elif quick_ratio < qr_warning:
                    flags.append(RedFlag(
                        category="Balance Sheet",
                        severity=RedFlagSeverity.WARNING,
                        title="Low Quick Ratio",
                        description=f"Cannot comfortably meet short-term obligations for {self.sector} sector without selling inventory",
                        current_value=quick_ratio,
                        threshold=qr_warning,
                        period=period_str
                    ))
        except ValueError:
            pass  # Data not available
        
        return flags
    
    def _check_tangible_book_value(self, equity, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check for negative tangible book value."""
        flags = []
        
        try:
            goodwill = self.balance_sheet.get_goodwill(granularity)
            intangibles = self.balance_sheet.get_intangible_assets(granularity)
            
            if not equity.values or not goodwill.values or not intangibles.values:
                return flags
            
            eq = equity.values[-1]
            gw = goodwill.values[-1]
            intang = intangibles.values[-1]
            tangible_book_value = eq - gw - intang
            
            if tangible_book_value < 0:
                flags.append(RedFlag(
                    category="Balance Sheet",
                    severity=RedFlagSeverity.CRITICAL,
                    title="Negative Tangible Book Value",
                    description="High intangibles/goodwill masking weak core assets - insolvency risk",
                    current_value=tangible_book_value / 1e9,
                    period=period_str
                ))
        except ValueError:
            pass
        
        return flags
    
    def _check_interest_coverage(self, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check interest coverage ratio with tiered severity."""
        flags = []
        
        try:
            ebit = self.income_statement.get_operating_income(granularity)
            interest = self.income_statement.get_interest_expense(granularity)
            
            if not ebit.values or not interest.values:
                return flags
            
            ebit_val = ebit.values[-1]
            interest_val = abs(interest.values[-1])  # Interest expense is typically negative
            
            if interest_val > 0:
                coverage = ebit_val / interest_val
                
                if coverage < self.thresholds.interest_coverage_critical:
                    flags.append(RedFlag(
                        category="Balance Sheet",
                        severity=RedFlagSeverity.CRITICAL,
                        title="Dangerously Low Interest Coverage",
                        description="Cannot cover interest payments with operating income - default risk",
                        current_value=coverage,
                        threshold=self.thresholds.interest_coverage_critical,
                        period=period_str
                    ))
                elif coverage < self.thresholds.interest_coverage_warning:
                    flags.append(RedFlag(
                        category="Balance Sheet",
                        severity=RedFlagSeverity.WARNING,
                        title="Low Interest Coverage",
                        description="Tight margin to cover interest - vulnerable to earnings decline",
                        current_value=coverage,
                        threshold=self.thresholds.interest_coverage_warning,
                        period=period_str
                    ))
        except ValueError:
            pass
        
        return flags
