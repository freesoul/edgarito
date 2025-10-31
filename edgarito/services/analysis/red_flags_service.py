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
    
    @property
    def all_flags(self) -> List[RedFlag]:
        """Get all flags sorted by severity"""
        all_flags = (
            self.balance_sheet_flags + 
            self.cash_flow_flags + 
            self.profitability_flags
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
        
        return result


class RedFlagsService:
    """Service for detecting financial red flags"""
    
    def __init__(self, facts: CompanyFacts):
        """
        Initialize red flags service.
        
        Args:
            facts: Company facts data from SEC
        """
        self.facts = facts
        self.balance_sheet = BalanceSheetReader(facts)
        self.income_statement = IncomeStatementReader(facts)
        self.cash_flow = CashFlowStatementReader(facts)
    
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
            
            # 1. Debt > Market Cap
            if self.market_cap and total_debt.values:
                debt = total_debt.values[-1]
                if debt > self.market_cap:
                    flags.append(RedFlag(
                        category="Balance Sheet",
                        severity=RedFlagSeverity.CRITICAL,
                        title="Debt exceeds Market Capitalization",
                        description="Total debt is higher than market cap - indicates high leverage risk",
                        current_value=debt / self.market_cap,
                        threshold=1.0,
                        period=period_str
                    ))
            
            # 2. Debt/Equity > 1.0
            if total_debt.values and equity.values:
                debt = total_debt.values[-1]
                eq = equity.values[-1]
                if eq > 0:
                    debt_to_equity = debt / eq
                    if debt_to_equity > 1.0:
                        flags.append(RedFlag(
                            category="Balance Sheet",
                            severity=RedFlagSeverity.WARNING,
                            title="High Debt-to-Equity Ratio",
                            description="Capital structure too leveraged - debt exceeds equity",
                            current_value=debt_to_equity,
                            threshold=1.0,
                            period=period_str
                        ))
            
            # 3. Current Ratio < 1.0
            if current_assets.values and current_liabilities.values:
                curr_assets = current_assets.values[-1]
                curr_liab = current_liabilities.values[-1]
                if curr_liab > 0:
                    current_ratio = curr_assets / curr_liab
                    if current_ratio < 1.0:
                        flags.append(RedFlag(
                            category="Balance Sheet",
                            severity=RedFlagSeverity.CRITICAL,
                            title="Low Current Ratio",
                            description="Potential liquidity problems - current assets < current liabilities",
                            current_value=current_ratio,
                            threshold=1.0,
                            period=period_str
                        ))
            
            # 4. Quick Ratio < 0.8
            try:
                cash = self.balance_sheet.get_cash_and_equivalents(granularity)
                receivables = self.balance_sheet.get_accounts_receivable(granularity)
                
                if cash.values and receivables.values and current_liabilities.values:
                    quick_assets = cash.values[-1] + receivables.values[-1]
                    curr_liab = current_liabilities.values[-1]
                    if curr_liab > 0:
                        quick_ratio = quick_assets / curr_liab
                        if quick_ratio < 0.8:
                            flags.append(RedFlag(
                                category="Balance Sheet",
                                severity=RedFlagSeverity.WARNING,
                                title="Low Quick Ratio",
                                description="Cannot meet short-term obligations without selling inventory",
                                current_value=quick_ratio,
                                threshold=0.8,
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
                            severity=RedFlagSeverity.WARNING,
                            title="Negative Tangible Book Value",
                            description="High intangibles/goodwill masking weak core assets",
                            current_value=tangible_book_value / 1e9,
                            period=period_str
                        ))
            except ValueError:
                pass
            
            # 6. Interest Coverage < 2x
            try:
                operating_income = self.income_statement.get_operating_income(granularity)
                interest_expense = self.income_statement.get_interest_expense(granularity)
                
                if operating_income.values and interest_expense.values:
                    ebit = operating_income.values[-1]
                    interest = interest_expense.values[-1]
                    
                    if interest > 0:
                        coverage = ebit / interest
                        if coverage < 2.0:
                            flags.append(RedFlag(
                                category="Balance Sheet",
                                severity=RedFlagSeverity.CRITICAL,
                                title="Low Interest Coverage",
                                description="Risk of default - EBIT < 2x interest expense",
                                current_value=coverage,
                                threshold=2.0,
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
            
            # 1. Operating Cash Flow < Net Income consistently
            if len(ocf.values) >= 3 and len(net_income.values) >= 3:
                # Check last 3 periods
                ocf_below_count = sum(1 for i in range(-3, 0) 
                                     if ocf.values[i] < net_income.values[i])
                
                if ocf_below_count >= 2:
                    flags.append(RedFlag(
                        category="Cash Flow",
                        severity=RedFlagSeverity.WARNING,
                        title="OCF Consistently Below Net Income",
                        description="Earnings quality issues - profits not converting to cash",
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
                        if sbc_pct > 0.10:
                            flags.append(RedFlag(
                                category="Cash Flow",
                                severity=RedFlagSeverity.WARNING,
                                title="High Stock-Based Compensation",
                                description="Dilution disguised as expense - impacts shareholder value",
                                current_value=sbc_pct * 100,
                                threshold=10.0,
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
                
                if om_current < om_prev - 2:  # 2% decline threshold
                    flags.append(RedFlag(
                        category="Profitability",
                        severity=RedFlagSeverity.WARNING,
                        title="Declining Operating Margin",
                        description="Poor cost control or increased competition",
                        current_value=om_current,
                        threshold=om_prev,
                        period=f"{revenue.periods[-3].year}-{latest_period.year}"
                    ))
            
            # 3. Net margin < 3%
            if revenue.values and net_income.values:
                net_margin = (net_income.values[-1] / revenue.values[-1]) * 100 if revenue.values[-1] != 0 else 0
                
                if 0 < net_margin < 3:
                    flags.append(RedFlag(
                        category="Profitability",
                        severity=RedFlagSeverity.INFO,
                        title="Low Net Margin",
                        description="Thin margins - vulnerable to downturns",
                        current_value=net_margin,
                        threshold=3.0,
                        period=period_str
                    ))
            
            # 4. ROE < 10%
            try:
                equity = self.balance_sheet.get_stockholders_equity(granularity)
                
                if net_income.values and equity.values and len(equity.values) >= 2:
                    avg_equity = (equity.values[-1] + equity.values[-2]) / 2
                    if avg_equity > 0:
                        roe = (net_income.values[-1] / avg_equity) * 100
                        
                        if 0 < roe < 10:
                            flags.append(RedFlag(
                                category="Profitability",
                                severity=RedFlagSeverity.INFO,
                                title="Low Return on Equity",
                                description="Poor capital efficiency - not generating adequate returns",
                                current_value=roe,
                                threshold=10.0,
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
