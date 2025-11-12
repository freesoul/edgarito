"""
Profitability and income quality analyzer - margins, returns, earnings quality checks.
"""
from typing import List
import logging

from edgarito.enums.granularity import Granularity
from edgarito.schemas.red_flags import RedFlag, RedFlagSeverity
from edgarito.services.analysis.period_alignment import (
    align_series_for_ratio,
    align_series_for_growth,
    PeriodMismatchError,
    check_series_freshness
)
from .base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


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
        
        try:
            # Validate period alignment before calculating margins
            gp_values, rev_values, aligned_periods = align_series_for_ratio(
                numerator=gross_profit,
                denominator=revenue,
                context="gross margin calculation",
                require_recent_data=True,
                max_age_days=365  # Reject if latest data is > 1 year old
            )
            
            # Need at least 3 periods for trend analysis
            if len(aligned_periods) < 3:
                logger.info(f"Insufficient aligned periods for gross margin trend: {len(aligned_periods)}")
                return flags
            
            # Calculate margins on aligned data
            gm_current = (gp_values[-1] / rev_values[-1]) * 100 if rev_values[-1] != 0 else 0
            gm_prev = (gp_values[-3] / rev_values[-3]) * 100 if rev_values[-3] != 0 else 0
            
            if gm_current < gm_prev - 2:  # 2% decline threshold
                flags.append(RedFlag(
                    category="Profitability",
                    severity=RedFlagSeverity.WARNING,
                    title="Declining Gross Margin",
                    description="Pricing pressure or cost inflation squeezing margins",
                    current_value=gm_current,
                    threshold=gm_prev,
                    period=f"{aligned_periods[-3].year}-{aligned_periods[-1].year} {aligned_periods[-1].fp.value}"
                ))
                
        except PeriodMismatchError as e:
            logger.warning(f"Skipping gross margin check: {e}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error in gross margin check: {e}")
        
        return flags
    
    def _check_operating_margin(self, revenue, operating_income, period_str: str) -> List[RedFlag]:
        """Check for negative or declining operating margin."""
        flags = []
        
        try:
            # Validate period alignment
            oi_values, rev_values, aligned_periods = align_series_for_ratio(
                numerator=operating_income,
                denominator=revenue,
                context="operating margin calculation",
                require_recent_data=True,
                max_age_days=365
            )
            
            # Need at least 1 period for current margin, 3 for trend
            if len(aligned_periods) < 1:
                return flags
            
            # Calculate current margin
            om_current = (oi_values[-1] / rev_values[-1]) * 100 if rev_values[-1] != 0 else 0
            
            # Check for negative operating margin (CRITICAL)
            if om_current < 0:
                flags.append(RedFlag(
                    category="Profitability",
                    severity=RedFlagSeverity.CRITICAL,
                    title="Negative Operating Margin",
                    description="Operating losses - core business not profitable",
                    current_value=om_current,
                    threshold=0.0,
                    period=f"{aligned_periods[-1].year} {aligned_periods[-1].fp.value}"
                ))
            elif len(aligned_periods) >= 3:
                # Check for declining margin
                om_prev = (oi_values[-3] / rev_values[-3]) * 100 if rev_values[-3] != 0 else 0
                
                if om_current < om_prev - 2:  # 2% decline threshold
                    flags.append(RedFlag(
                        category="Profitability",
                        severity=RedFlagSeverity.WARNING,
                        title="Declining Operating Margin",
                        description="Poor cost control or increased competition",
                        current_value=om_current,
                        threshold=om_prev,
                        period=f"{aligned_periods[-3].year}-{aligned_periods[-1].year} {aligned_periods[-1].fp.value}"
                    ))
                    
        except PeriodMismatchError as e:
            logger.warning(f"Skipping operating margin check: {e}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error in operating margin check: {e}")
        
        return flags
    
    def _check_net_margin(self, revenue, net_income, period_str: str) -> List[RedFlag]:
        """Check net margin (negative = CRITICAL, low = INFO)."""
        flags = []
        
        try:
            # Validate period alignment
            ni_values, rev_values, aligned_periods = align_series_for_ratio(
                numerator=net_income,
                denominator=revenue,
                context="net margin calculation",
                require_recent_data=True,
                max_age_days=365
            )
            
            if len(aligned_periods) < 1:
                return flags
            
            net_margin = (ni_values[-1] / rev_values[-1]) * 100 if rev_values[-1] != 0 else 0
            period_display = f"{aligned_periods[-1].year} {aligned_periods[-1].fp.value}"
            
            if net_margin < 0:
                flags.append(RedFlag(
                    category="Profitability",
                    severity=RedFlagSeverity.CRITICAL,
                    title="Negative Net Margin (Losses)",
                    description="Company losing money - burning through capital",
                    current_value=net_margin,
                    threshold=0.0,
                    period=period_display
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
                        period=period_display
                    ))
                    
        except PeriodMismatchError as e:
            logger.warning(f"Skipping net margin check: {e}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error in net margin check: {e}")
        
        return flags
    
    def _check_roe(self, net_income, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check ROE (negative = CRITICAL, low = INFO)."""
        flags = []
        
        try:
            equity = self.balance_sheet.get_stockholders_equity(granularity)
            
            # Validate alignment between net income and equity
            ni_values, eq_values, aligned_periods = align_series_for_ratio(
                numerator=net_income,
                denominator=equity,
                context="ROE calculation",
                require_recent_data=True,
                max_age_days=365
            )
            
            if len(aligned_periods) < 2:
                # Need at least 2 periods to calculate average equity
                return flags
            
            # Use average equity (standard ROE calculation)
            avg_equity = (eq_values[-1] + eq_values[-2]) / 2
            if avg_equity > 0:
                roe = (ni_values[-1] / avg_equity) * 100
                period_display = f"{aligned_periods[-1].year} {aligned_periods[-1].fp.value}"
                
                if roe < 0:
                    flags.append(RedFlag(
                        category="Profitability",
                        severity=RedFlagSeverity.CRITICAL,
                        title="Negative Return on Equity",
                        description="Destroying shareholder value - losses eating into equity",
                        current_value=roe,
                        threshold=0.0,
                        period=period_display
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
                            period=period_display
                        ))
                        
        except PeriodMismatchError as e:
            logger.warning(f"Skipping ROE check: {e}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error in ROE check: {e}")
        
        return flags
    
    def _check_revenue_vs_eps_growth(self, revenue, granularity: Granularity, period_str: str) -> List[RedFlag]:
        """Check for revenue growth without EPS growth."""
        flags = []
        
        try:
            eps = self.income_statement.get_earnings_per_share_diluted(granularity)
            
            # Align revenue and EPS for growth calculation (need at least 3 periods)
            current_rev, prior_rev, current_periods_rev, prior_periods_rev = align_series_for_growth(
                series=revenue,
                context="revenue vs EPS growth check"
            )
            
            current_eps, prior_eps, current_periods_eps, prior_periods_eps = align_series_for_growth(
                series=eps,
                context="revenue vs EPS growth check"
            )
            
            # Ensure revenue and EPS are aligned to the same periods
            if (len(current_rev) >= 3 and len(prior_rev) >= 3 and 
                len(current_eps) >= 3 and len(prior_eps) >= 3):
                
                # Use the last 3 periods for growth calculation
                rev_growth = ((current_rev[-1] - prior_rev[-3]) / prior_rev[-3]) * 100 if prior_rev[-3] != 0 else 0
                eps_growth = ((current_eps[-1] - prior_eps[-3]) / prior_eps[-3]) * 100 if prior_eps[-3] != 0 else 0
                
                if rev_growth > 10 and eps_growth < rev_growth / 2:  # Revenue up >10%, EPS lags
                    flags.append(RedFlag(
                        category="Profitability",
                        severity=RedFlagSeverity.WARNING,
                        title="Revenue Growth Without EPS Growth",
                        description="Margin compression or share dilution eating into profits",
                        current_value=eps_growth,
                        threshold=rev_growth,
                        period=f"{current_periods_rev[-3].year}-{period_str.split()[0]}"
                    ))
                    
        except PeriodMismatchError as e:
            logger.warning(f"Skipping revenue vs EPS growth check: {e}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error in revenue vs EPS growth check: {e}")
        
        return flags
    
    def _check_margin_volatility(self, revenue, gross_profit) -> List[RedFlag]:
        """Check for volatile gross margins."""
        flags = []
        
        try:
            # Align gross profit and revenue for margin calculation (need 4 periods for volatility)
            gp_values, rev_values, aligned_periods = align_series_for_ratio(
                numerator=gross_profit,
                denominator=revenue,
                context="margin volatility calculation"
            )
            
            # Need at least 4 periods for volatility analysis
            if len(aligned_periods) < 4:
                logger.warning(f"Skipping margin volatility check: only {len(aligned_periods)} aligned periods (need 4)")
                return flags
            
            # Calculate margins for last 4 periods
            margins = [(gp_values[i] / rev_values[i]) * 100 
                      for i in range(-4, 0) if rev_values[i] != 0]
            
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
                    
        except PeriodMismatchError as e:
            logger.warning(f"Skipping margin volatility check: {e}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error in margin volatility check: {e}")
        
        return flags
