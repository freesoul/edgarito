"""
Red Flags Service - detects financial warning signs in company financials.

Analyzes balance sheet health, cash flow quality, and profitability metrics
to identify potential risks and concerns.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts
from edgarito.enums.granularity import Granularity
from edgarito.services.financial.balance_sheet_reader import BalanceSheetReader
from edgarito.services.financial.income_statement_reader import IncomeStatementReader
from edgarito.services.financial.cash_flow_reader import CashFlowStatementReader
from edgarito.schemas.market_data import MarketData
from edgarito.cli.settings import RedFlagsThresholds


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
        result += f"{'='*100}\n\n"
        result += f"Summary: {self.total_flags} total flags "
        result += f"({self.critical_flags} critical, {self.warning_flags} warnings, {self.info_flags} info)\n\n"
        
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
            result += "📈 PROFITABILITY & INCOME QUALITY\n"
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


class RedFlagsService:
    """Service for detecting financial red flags"""
    
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
    
    def analyze(self, granularity: Granularity = Granularity.ANNUAL) -> RedFlagReport:
        """
        Run complete red flags analysis.
        
        Args:
            granularity: ANNUAL or QUARTERLY analysis
        
        Returns:
            Complete red flag report
        """
        report = RedFlagReport(
            ticker=str(self.facts.cik),
            company_name=self.facts.entityName
        )
        
        # Run all analyses
        report.balance_sheet_flags = self._analyze_balance_sheet(granularity)
        report.cash_flow_flags = self._analyze_cash_flow(granularity)
        report.profitability_flags = self._analyze_profitability(granularity)
        report.growth_flags = self._analyze_growth(granularity)
        report.valuation_flags = self._analyze_valuation(granularity)
        
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
    
    # ========== BALANCE SHEET HEALTH ==========
    
    def _analyze_balance_sheet(self, granularity: Granularity) -> List[RedFlag]:
        """Analyze balance sheet for red flags"""
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
            
            # 1. Debt/Equity with tiered severity
            if total_debt.values and equity.values:
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
                    if debt_to_equity > self.thresholds.debt_to_equity_ratio_critical:
                        flags.append(RedFlag(
                            category="Balance Sheet",
                            severity=RedFlagSeverity.CRITICAL,
                            title="Extremely High Debt-to-Equity Ratio",
                            description="Severe overleveraging - debt more than 2x equity",
                            current_value=debt_to_equity,
                            threshold=self.thresholds.debt_to_equity_ratio_critical,
                            period=period_str
                        ))
                    elif debt_to_equity > self.thresholds.debt_to_equity_ratio_warning:
                        flags.append(RedFlag(
                            category="Balance Sheet",
                            severity=RedFlagSeverity.WARNING,
                            title="High Debt-to-Equity Ratio",
                            description="Capital structure too leveraged - debt exceeds equity",
                            current_value=debt_to_equity,
                            threshold=self.thresholds.debt_to_equity_ratio_warning,
                            period=period_str
                        ))
            
            # 2. Debt > Market Cap (if available)
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
            
            # 3. Current Ratio with tiered severity
            if current_assets.values and current_liabilities.values:
                curr_assets = current_assets.values[-1]
                curr_liab = current_liabilities.values[-1]
                if curr_liab > 0:
                    current_ratio = curr_assets / curr_liab
                    if current_ratio < self.thresholds.current_ratio_critical:
                        flags.append(RedFlag(
                            category="Balance Sheet",
                            severity=RedFlagSeverity.CRITICAL,
                            title="Critically Low Current Ratio",
                            description="Severe liquidity crisis - current assets below current liabilities",
                            current_value=current_ratio,
                            threshold=self.thresholds.current_ratio_critical,
                            period=period_str
                        ))
                    elif current_ratio < self.thresholds.current_ratio_warning:
                        flags.append(RedFlag(
                            category="Balance Sheet",
                            severity=RedFlagSeverity.WARNING,
                            title="Low Current Ratio",
                            description="Potential liquidity problems - limited ability to meet short-term obligations",
                            current_value=current_ratio,
                            threshold=self.thresholds.current_ratio_warning,
                            period=period_str
                        ))
            
            # 4. Quick Ratio with tiered severity
            try:
                cash = self.balance_sheet.get_cash_and_equivalents(granularity)
                receivables = self.balance_sheet.get_accounts_receivable(granularity)
                
                if cash.values and receivables.values and current_liabilities.values:
                    quick_assets = cash.values[-1] + receivables.values[-1]
                    curr_liab = current_liabilities.values[-1]
                    if curr_liab > 0:
                        quick_ratio = quick_assets / curr_liab
                        if quick_ratio < self.thresholds.quick_ratio_critical:
                            flags.append(RedFlag(
                                category="Balance Sheet",
                                severity=RedFlagSeverity.CRITICAL,
                                title="Critically Low Quick Ratio",
                                description="Severe liquidity - cannot meet immediate obligations without selling inventory",
                                current_value=quick_ratio,
                                threshold=self.thresholds.quick_ratio_critical,
                                period=period_str
                            ))
                        elif quick_ratio < self.thresholds.quick_ratio_warning:
                            flags.append(RedFlag(
                                category="Balance Sheet",
                                severity=RedFlagSeverity.WARNING,
                                title="Low Quick Ratio",
                                description="Cannot comfortably meet short-term obligations without selling inventory",
                                current_value=quick_ratio,
                                threshold=self.thresholds.quick_ratio_warning,
                                period=period_str
                            ))
            except ValueError:
                pass  # Data not available
            
            # 5. Negative Tangible Book Value
            try:
                goodwill = self.balance_sheet.get_goodwill(granularity)
                intangibles = self.balance_sheet.get_intangible_assets(granularity)
                
                if equity.values and goodwill.values and intangibles.values:
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
            
            # 6. Interest Coverage with tiered severity
            try:
                operating_income = self.income_statement.get_operating_income(granularity)
                interest_expense = self.income_statement.get_interest_expense(granularity)
                
                if operating_income.values and interest_expense.values:
                    ebit = operating_income.values[-1]
                    interest = interest_expense.values[-1]
                    
                    if interest > 0:
                        coverage = ebit / interest
                        if coverage < self.thresholds.interest_coverage_critical:
                            flags.append(RedFlag(
                                category="Balance Sheet",
                                severity=RedFlagSeverity.CRITICAL,
                                title="Critically Low Interest Coverage",
                                description="Imminent default risk - EBIT barely covers interest expense",
                                current_value=coverage,
                                threshold=self.thresholds.interest_coverage_critical,
                                period=period_str
                            ))
                        elif coverage < self.thresholds.interest_coverage_warning:
                            flags.append(RedFlag(
                                category="Balance Sheet",
                                severity=RedFlagSeverity.WARNING,
                                title="Low Interest Coverage",
                                description="Risk of default - insufficient EBIT cushion for interest payments",
                                current_value=coverage,
                                threshold=self.thresholds.interest_coverage_warning,
                                period=period_str
                            ))
            except ValueError:
                pass
            
            # 7. Rising Liabilities faster than Assets
            if len(total_assets.values) >= 3 and len(total_liabilities.values) >= 3:
                assets_growth = (total_assets.values[-1] - total_assets.values[-3]) / total_assets.values[-3]
                liab_growth = (total_liabilities.values[-1] - total_liabilities.values[-3]) / total_liabilities.values[-3]
                
                if liab_growth > assets_growth and liab_growth > 0.1:  # 10% threshold
                    flags.append(RedFlag(
                        category="Balance Sheet",
                        severity=RedFlagSeverity.WARNING,
                        title="Liabilities Growing Faster than Assets",
                        description="Deterioration of financial position over time",
                        current_value=liab_growth * 100,
                        threshold=assets_growth * 100,
                        period=f"{total_liabilities.periods[-3].year}-{latest_period.year}"
                    ))
        
        except ValueError as e:
            # Data not available for analysis
            pass
        
        return flags
    
    # ========== CASH FLOW RED FLAGS ==========
    
    def _analyze_cash_flow(self, granularity: Granularity) -> List[RedFlag]:
        """Analyze cash flow for red flags"""
        flags = []
        
        try:
            ocf = self.cash_flow.get_operating_cash_flow(granularity)
            net_income = self.income_statement.get_net_income(granularity)
            
            if not ocf.values or not net_income.values:
                return flags
            
            latest_period = ocf.periods[-1]
            period_str = f"{latest_period.year} {latest_period.fp.value}"
            
            # 0. Negative Operating Cash Flow (CRITICAL)
            if ocf.values and ocf.values[-1] < 0:
                flags.append(RedFlag(
                    category="Cash Flow",
                    severity=RedFlagSeverity.CRITICAL,
                    title="Negative Operating Cash Flow",
                    description="Burning cash from operations - unsustainable without financing",
                    current_value=ocf.values[-1] / 1e9,
                    threshold=0.0,
                    period=period_str
                ))
            
            # 1. Operating Cash Flow < Net Income consistently
            if len(ocf.values) >= 3 and len(net_income.values) >= 3:
                # Check last 3 periods
                ocf_below_count = sum(1 for i in range(-3, 0) 
                                     if ocf.values[i] < net_income.values[i])
                
                if ocf_below_count >= 2:
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.CRITICAL,
                        title="OCF Consistently Below Net Income",
                        description="Earnings quality issues - profits not converting to cash repeatedly",
                        current_value=ocf.values[-1] / net_income.values[-1] if net_income.values[-1] != 0 else 0,
                        threshold=1.0,
                        period=f"Last 3 periods"
                    ))
            
            # 2. Negative Free Cash Flow for 2+ years
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
            
            # 3. Dividends > Free Cash Flow
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
            
            # 4. CapEx rising faster than OCF
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
                            period=f"{ocf.periods[-3].year}-{latest_period.year}"
                        ))
            except ValueError:
                pass
            
            # 5. Stock-based compensation > 10% of OCF
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
            
            # 6. Frequent share issuance
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
        
        except ValueError as e:
            pass
        
        return flags
    
    # ========== PROFITABILITY / INCOME QUALITY ==========
    
    def _analyze_profitability(self, granularity: Granularity) -> List[RedFlag]:
        """Analyze profitability and income quality for red flags"""
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
            
            # 1. Declining gross margin
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
                        period=f"{revenue.periods[-3].year}-{latest_period.year}"
                    ))
            
            # 2. Declining operating margin
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
                        period=f"{revenue.periods[-3].year}-{latest_period.year}"
                    ))
            
            # 3. Net margin checks (negative = CRITICAL, low = INFO)
            if revenue.values and net_income.values:
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
                elif 0 < net_margin < self.thresholds.net_margin_percent:
                    flags.append(RedFlag(
                        category="Profitability",
                        severity=RedFlagSeverity.INFO,
                        title="Low Net Margin",
                        description="Thin margins - vulnerable to downturns",
                        current_value=net_margin,
                        threshold=3.0,
                        period=period_str
                    ))
            
            # 4. ROE checks (negative = CRITICAL, low = INFO)
            try:
                equity = self.balance_sheet.get_stockholders_equity(granularity)
                
                if net_income.values and equity.values and len(equity.values) >= 2:
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
                        elif 0 < roe < self.thresholds.roe_percent:
                            flags.append(RedFlag(
                                category="Profitability",
                                severity=RedFlagSeverity.INFO,
                                title="Low Return on Equity",
                                description="Poor capital efficiency - not generating adequate returns",
                                current_value=roe,
                                threshold=self.thresholds.roe_percent,
                                period=period_str
                            ))
            except ValueError:
                pass
            
            # 5. Revenue growth without EPS growth
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
                            period=f"{revenue.periods[-3].year}-{latest_period.year}"
                        ))
            except ValueError:
                pass
            
            # 6. Volatile margins
            if len(gross_profit.values) >= 4 and len(revenue.values) >= 4:
                margins = [(gross_profit.values[i] / revenue.values[i]) * 100 
                          for i in range(-4, 0) if revenue.values[i] != 0]
                
                if len(margins) == 4:
                    # Calculate standard deviation
                    mean_margin = sum(margins) / len(margins)
                    variance = sum((m - mean_margin) ** 2 for m in margins) / len(margins)
                    std_dev = variance ** 0.5
                    
                    if std_dev > 5:  # 5% std dev threshold
                        flags.append(RedFlag(
                            category="Profitability",
                            severity=RedFlagSeverity.INFO,
                            title="Highly Volatile Gross Margin",
                            description="Unstable business - margins fluctuate significantly",
                            current_value=std_dev,
                            threshold=5.0,
                            period=f"Last 4 periods"
                        ))
        
        except ValueError as e:
            pass
        
        return flags
    
    # ========== GROWTH & SUSTAINABILITY ==========
    
    def _analyze_growth(self, granularity: Granularity) -> List[RedFlag]:
        """Analyze growth and sustainability red flags"""
        flags = []
        
        try:
            revenue = self.income_statement.get_revenue(granularity)
            
            if not revenue.values or len(revenue.values) < 5:
                return flags
            
            latest_period = revenue.periods[-1]
            period_str = f"{latest_period.year} {latest_period.fp.value}"
            
            # 1. Revenue CAGR checks (severe decline = CRITICAL, stagnation = INFO)
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
                        period=f"{revenue.periods[-5].year}-{latest_period.year}"
                    ))
                elif cagr < self.thresholds.revenue_cagr_inflation:
                    flags.append(RedFlag(
                        category="Growth",
                        severity=RedFlagSeverity.INFO,
                        title="Revenue Growth Below Inflation",
                        description="Stagnation - revenue barely keeping up with inflation",
                        current_value=cagr,
                        threshold=self.thresholds.revenue_cagr_inflation,
                        period=f"{revenue.periods[-5].year}-{latest_period.year}"
                    ))
            
            # 2. High or rising SG&A % of revenue with tiered severity
            try:
                sga = self.income_statement.get_selling_general_administrative_expense(granularity)
                
                if sga.values and revenue.values:
                    sga_pct = (sga.values[-1] / revenue.values[-1]) * 100 if revenue.values[-1] != 0 else 0
                    
                    # Check SG&A thresholds
                    if sga_pct > self.thresholds.sga_percent_revenue_warning:
                        flags.append(RedFlag(
                            category="Growth",
                            severity=RedFlagSeverity.WARNING,
                            title="High SG&A Expenses",
                            description="Bloated overhead - SG&A exceeds 30% of revenue",
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
                                period=f"{revenue.periods[-3].year}-{latest_period.year}"
                            ))
            except ValueError:
                pass
            
            # 3. Declining R&D while revenue is growing
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
                            period=f"{revenue.periods[-3].year}-{latest_period.year}"
                        ))
            except ValueError:
                pass
            
        except ValueError:
            pass
        
        return flags
    
    # ========== VALUATION CONCERNS ==========
    
    def _analyze_valuation(self, granularity: Granularity) -> List[RedFlag]:
        """Analyze valuation metrics if market cap is available"""
        flags = []
        
        if not self.market_cap:
            return flags  # Need market cap for valuation metrics
        
        try:
            revenue = self.income_statement.get_revenue(granularity)
            net_income = self.income_statement.get_net_income(granularity)
            
            if not revenue.values or not net_income.values:
                return flags
            
            latest_period = revenue.periods[-1]
            period_str = f"{latest_period.year} {latest_period.fp.value}"
            
            # Get TTM (Trailing Twelve Months) values for annual metrics
            if granularity == Granularity.QUARTERLY and len(revenue.values) >= 4:
                ttm_revenue = sum(revenue.values[-4:])
                ttm_net_income = sum(net_income.values[-4:])
            else:
                ttm_revenue = revenue.values[-1]
                ttm_net_income = net_income.values[-1]
            
            # 1. P/S ratio > threshold
            price_to_sales = self.market_cap / ttm_revenue if ttm_revenue != 0 else 0
            
            if price_to_sales > self.thresholds.price_to_sales:
                flags.append(RedFlag(
                    category="Valuation",
                    severity=RedFlagSeverity.INFO,
                    title="High Price-to-Sales Ratio",
                    description="Needs massive growth to justify valuation - speculative pricing",
                    current_value=price_to_sales,
                    threshold=self.thresholds.price_to_sales,
                    period=period_str
                ))
            
            # 2. P/E ratio checks
            if ttm_net_income > 0:
                pe_ratio = self.market_cap / ttm_net_income
                
                # PE < threshold with low growth
                if pe_ratio < self.thresholds.pe_ratio_low:
                    # Check if revenue growth is low
                    if len(revenue.values) >= 3:
                        rev_growth = ((revenue.values[-1] - revenue.values[-3]) / revenue.values[-3]) * 100 if revenue.values[-3] != 0 else 0
                        
                        if rev_growth < self.thresholds.revenue_growth_for_rd_check:
                            flags.append(RedFlag(
                                category="Valuation",
                                severity=RedFlagSeverity.INFO,
                                title="Very Low P/E Without Growth Catalyst",
                                description="Might be a value trap - low valuation with stagnant growth",
                                current_value=pe_ratio,
                                threshold=self.thresholds.pe_ratio_low,
                                period=period_str
                            ))
            
            # 3. Price-to-Book ratio > 5 without high ROE
            try:
                equity = self.balance_sheet.get_stockholders_equity(granularity)
                
                if equity.values:
                    book_value = equity.values[-1]
                    if book_value > 0:
                        pb_ratio = self.market_cap / book_value
                        
                        # Calculate ROE
                        if len(equity.values) >= 2:
                            avg_equity = (equity.values[-1] + equity.values[-2]) / 2
                            roe = (ttm_net_income / avg_equity) * 100 if avg_equity > 0 else 0
                            
                            # High P/B without high ROE
                            if pb_ratio > self.thresholds.price_to_book and roe < self.thresholds.roe_for_pb_check:
                                flags.append(RedFlag(
                                    category="Valuation",
                                    severity=RedFlagSeverity.INFO,
                                    title="High Price-to-Book Without High ROE",
                                    description="Speculative pricing - valuation not supported by returns",
                                    current_value=pb_ratio,
                                    threshold=self.thresholds.price_to_book,
                                    period=period_str
                                ))
            except ValueError:
                pass
            
            # 4. Dividend yield > threshold (if dividends paid)
            try:
                dividends = self.cash_flow.get_dividends_paid(granularity)
                
                if dividends.values and len(dividends.values) >= 4:
                    # Calculate annual dividend
                    if granularity == Granularity.QUARTERLY:
                        annual_dividend = abs(sum(dividends.values[-4:]))
                    else:
                        annual_dividend = abs(dividends.values[-1])
                    
                    if annual_dividend > 0:
                        dividend_yield = (annual_dividend / self.market_cap) * 100
                        
                        if dividend_yield > self.thresholds.dividend_yield:
                            flags.append(RedFlag(
                                category="Valuation",
                                severity=RedFlagSeverity.WARNING,
                                title="Extremely High Dividend Yield",
                                description="Market pricing in dividend cut - yield too good to be true",
                                current_value=dividend_yield,
                                threshold=self.thresholds.dividend_yield,
                                period=period_str
                            ))
            except ValueError:
                pass
            
            # ========== Yahoo Finance Enhanced Checks ==========
            
            # 5. PEG Ratio > threshold (if available from Yahoo Finance)
            if self.market_data and self.market_data.peg_ratio:
                if self.market_data.peg_ratio > self.thresholds.peg_ratio:
                    flags.append(RedFlag(
                        category="Valuation",
                        severity=RedFlagSeverity.INFO,
                        title="High PEG Ratio",
                        description="Overpriced relative to growth - paying too much for earnings growth",
                        current_value=self.market_data.peg_ratio,
                        threshold=self.thresholds.peg_ratio,
                        period=period_str
                    ))
            
            # 6. EV/EBITDA > threshold (if available from Yahoo Finance)
            if self.market_data and self.market_data.ev_to_ebitda:
                if self.market_data.ev_to_ebitda > self.thresholds.ev_to_ebitda:
                    flags.append(RedFlag(
                        category="Valuation",
                        severity=RedFlagSeverity.INFO,
                        title="High Enterprise Value / EBITDA",
                        description="Expensive valuation - needs hyper-growth to justify EV multiple",
                        current_value=self.market_data.ev_to_ebitda,
                        threshold=self.thresholds.ev_to_ebitda,
                        period=period_str
                    ))
            
            # 7. High short interest > threshold (market skepticism)
            if self.market_data and self.market_data.short_percent_float:
                if self.market_data.short_percent_float > self.thresholds.short_interest_percent:
                    flags.append(RedFlag(
                        category="Valuation",
                        severity=RedFlagSeverity.WARNING,
                        title="High Short Interest",
                        description="Market skepticism - significant bearish positioning by traders",
                        current_value=self.market_data.short_percent_float,
                        threshold=self.thresholds.short_interest_percent,
                        period="Current"
                    ))
            
            # 8. Low insider ownership < threshold (lack of confidence)
            if self.market_data and self.market_data.insider_ownership_percent is not None:
                if self.market_data.insider_ownership_percent < self.thresholds.insider_ownership_percent:
                    flags.append(RedFlag(
                        category="Valuation",
                        severity=RedFlagSeverity.INFO,
                        title="Low Insider Ownership",
                        description="Lack of skin in the game - insiders own minimal stake",
                        current_value=self.market_data.insider_ownership_percent,
                        threshold=self.thresholds.insider_ownership_percent,
                        period="Current"
                    ))
            
        except ValueError:
            pass
        
        return flags
