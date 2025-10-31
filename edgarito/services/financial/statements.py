"""
Unified Financial Statements Facade - coordinates all statement readers.

Provides a single entry point for loading and accessing all financial statement data:
- Balance Sheet
- Income Statement
- Cash Flow Statement
- Computed Metrics
"""
from typing import Optional
import json

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts
from edgarito.services.financial.balance_sheet_reader import BalanceSheetReader
from edgarito.services.financial.income_statement_reader import IncomeStatementReader
from edgarito.services.financial.cash_flow_reader import CashFlowStatementReader
from edgarito.services.financial.metrics import FinancialMetrics


class FinancialStatements:
    """
    Unified facade for accessing all financial statement data.
    
    Coordinates balance sheet, income statement, and cash flow readers,
    plus computed financial metrics.
    
    Usage:
        # Load from CompanyFacts object
        statements = FinancialStatements()
        statements.load_from_facts(facts)
        
        # Or load from JSON file
        statements = FinancialStatements()
        statements.load_from_json_file("cache/.../CIK0001652044.json")
        
        # Access individual statements
        revenue = statements.income_statement.get_revenue(Granularity.ANNUAL)
        assets = statements.balance_sheet.get_total_assets(Granularity.ANNUAL)
        ocf = statements.cash_flow.get_operating_cash_flow(Granularity.ANNUAL)
        
        # Compute metrics (ensure periods are aligned first)
        revenue = revenue.intersect(assets)
        assets = assets.intersect(revenue)
        asset_turnover = statements.metrics.asset_turnover(revenue, assets)
    """
    
    def __init__(self, facts: Optional[CompanyFacts] = None):
        """
        Initialize the financial statements facade.
        
        Args:
            facts: Optional CompanyFacts object to load immediately
        """
        self._balance_sheet = BalanceSheetReader()
        self._income_statement = IncomeStatementReader()
        self._cash_flow = CashFlowStatementReader()
        self._metrics = FinancialMetrics()
        
        if facts is not None:
            self.load_from_facts(facts)
    
    def load_from_facts(self, facts: CompanyFacts):
        """
        Load data from a CompanyFacts object.
        
        Updates all readers with the new data.
        
        Args:
            facts: CompanyFacts object from SEC EDGAR REST API
        """
        self._balance_sheet.load_from_facts(facts)
        self._income_statement.load_from_facts(facts)
        self._cash_flow.load_from_facts(facts)
    
    def load_from_json_file(self, path: str):
        """
        Load data from a JSON file containing CompanyFacts data.
        
        Updates all readers with the new data.
        
        Args:
            path: Path to JSON file with CompanyFacts data
        """
        with open(path, "r") as file:
            data_json = json.load(file)
        facts = CompanyFacts(**data_json)
        self.load_from_facts(facts)
    
    @property
    def balance_sheet(self) -> BalanceSheetReader:
        """Access balance sheet reader (assets, liabilities, equity)"""
        return self._balance_sheet
    
    @property
    def income_statement(self) -> IncomeStatementReader:
        """Access income statement reader (revenues, expenses, income)"""
        return self._income_statement
    
    @property
    def cash_flow(self) -> CashFlowStatementReader:
        """Access cash flow statement reader (operating, investing, financing)"""
        return self._cash_flow
    
    @property
    def metrics(self) -> FinancialMetrics:
        """Access financial metrics calculator (ratios, KPIs)"""
        return self._metrics


if __name__ == "__main__":
    # Example usage
    from edgarito.enums.granularity import Granularity
    
    # Load financial data
    statements = FinancialStatements()
    statements.load_from_json_file("cache/edgar_rest/api/xbrl/companyfacts/CIK0001652044.json")
    
    # Access balance sheet data
    assets = statements.balance_sheet.get_total_assets(Granularity.ANNUAL)
    equity = statements.balance_sheet.get_stockholders_equity(Granularity.ANNUAL)
    
    # Access income statement data
    revenue = statements.income_statement.get_revenue(Granularity.ANNUAL)
    net_income = statements.income_statement.get_net_income(Granularity.ANNUAL)
    
    # Access cash flow data
    operating_cf = statements.cash_flow.get_operating_cash_flow(Granularity.ANNUAL)
    capex = statements.cash_flow.get_capital_expenditures(Granularity.ANNUAL)
    
    # Compute metrics (align periods first)
    revenue_aligned = revenue.intersect(net_income)
    net_income_aligned = net_income.intersect(revenue)
    net_margin = statements.metrics.net_margin(net_income_aligned, revenue_aligned)
    
    assets_aligned = assets.intersect(equity)
    equity_aligned = equity.intersect(assets)
    equity_multiplier = statements.metrics.equity_multiplier(assets_aligned, equity_aligned)
    
    ocf_aligned = operating_cf.intersect(capex)
    capex_aligned = capex.intersect(operating_cf)
    free_cash_flow = statements.metrics.free_cash_flow(ocf_aligned, capex_aligned)
    
    print(f"Net Margin: {net_margin}")
    print(f"Equity Multiplier: {equity_multiplier}")
    print(f"Free Cash Flow: {free_cash_flow}")
