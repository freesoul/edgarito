"""
Financial Metrics - computes KPIs and ratios from raw financial statement data.

Provides comprehensive financial analysis including:
- Profitability ratios (margins, returns)
- Liquidity ratios (current, quick)
- Leverage ratios (debt-to-equity, debt-to-assets)
- Efficiency ratios (turnover, days outstanding)
- Valuation metrics (P/E, P/B, FCF)
"""
from typing import Optional

from edgarito.schemas.reader.measurements import UnivariateMeasurements


class FinancialMetrics:
    """
    Computes financial KPIs and ratios from raw financial data.
    
    All methods are static and accept UnivariateMeasurements as input.
    Ensure time periods are aligned using .intersect() before computing ratios.
    """
    
    # ========== PROFITABILITY RATIOS ==========
    
    @staticmethod
    def gross_profit(revenues: UnivariateMeasurements, costs: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Gross profit = Revenue - Cost of Revenue
        
        Args:
            revenues: Revenue time series
            costs: Cost of revenue time series
        
        Returns:
            Gross profit time series
        """
        return revenues - costs
    
    @staticmethod
    def gross_margin(revenues: UnivariateMeasurements, costs: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Gross margin = (Revenue - Cost of Revenue) / Revenue
        
        Measures profitability after direct costs.
        Higher is better (indicates pricing power and efficiency).
        
        Args:
            revenues: Revenue time series
            costs: Cost of revenue time series
        
        Returns:
            Gross margin time series (as decimal, e.g., 0.40 = 40%)
        """
        return (revenues - costs) / revenues
    
    @staticmethod
    def operating_margin(operating_income: UnivariateMeasurements, revenues: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Operating margin = Operating Income / Revenue
        
        Measures profitability after all operating expenses.
        Higher is better (indicates operational efficiency).
        
        Args:
            operating_income: Operating income time series (EBIT)
            revenues: Revenue time series
        
        Returns:
            Operating margin time series (as decimal)
        """
        return operating_income / revenues
    
    @staticmethod
    def net_margin(net_income: UnivariateMeasurements, revenues: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Net margin = Net Income / Revenue
        
        Measures bottom-line profitability.
        Higher is better (indicates overall profitability).
        
        Args:
            net_income: Net income time series
            revenues: Revenue time series
        
        Returns:
            Net margin time series (as decimal)
        """
        return net_income / revenues
    
    @staticmethod
    def return_on_assets(net_income: UnivariateMeasurements, total_assets: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        ROA = Net Income / Total Assets
        
        Measures how efficiently assets generate profit.
        Higher is better (indicates efficient asset utilization).
        
        Args:
            net_income: Net income time series
            total_assets: Total assets time series
        
        Returns:
            ROA time series (as decimal)
        """
        return net_income / total_assets
    
    @staticmethod
    def return_on_equity(net_income: UnivariateMeasurements, stockholders_equity: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        ROE = Net Income / Stockholders' Equity
        
        Measures return generated for shareholders.
        Higher is better (indicates efficient use of equity).
        
        Args:
            net_income: Net income time series
            stockholders_equity: Stockholders' equity time series
        
        Returns:
            ROE time series (as decimal)
        """
        return net_income / stockholders_equity
    
    # ========== LIQUIDITY RATIOS ==========
    
    @staticmethod
    def current_ratio(current_assets: UnivariateMeasurements, current_liabilities: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Current ratio = Current Assets / Current Liabilities
        
        Measures ability to pay short-term obligations.
        > 1.0 indicates sufficient liquidity.
        
        Args:
            current_assets: Current assets time series
            current_liabilities: Current liabilities time series
        
        Returns:
            Current ratio time series
        """
        return current_assets / current_liabilities
    
    @staticmethod
    def quick_ratio(
        current_assets: UnivariateMeasurements,
        inventory: UnivariateMeasurements,
        current_liabilities: UnivariateMeasurements
    ) -> UnivariateMeasurements:
        """
        Quick ratio = (Current Assets - Inventory) / Current Liabilities
        
        Measures ability to pay short-term obligations without selling inventory.
        > 1.0 indicates strong liquidity.
        
        Args:
            current_assets: Current assets time series
            inventory: Inventory time series
            current_liabilities: Current liabilities time series
        
        Returns:
            Quick ratio time series
        """
        return (current_assets - inventory) / current_liabilities
    
    @staticmethod
    def cash_ratio(
        cash_and_equivalents: UnivariateMeasurements,
        current_liabilities: UnivariateMeasurements
    ) -> UnivariateMeasurements:
        """
        Cash ratio = Cash and Cash Equivalents / Current Liabilities
        
        Measures ability to pay short-term obligations with cash only.
        Most conservative liquidity measure.
        
        Args:
            cash_and_equivalents: Cash and cash equivalents time series
            current_liabilities: Current liabilities time series
        
        Returns:
            Cash ratio time series
        """
        return cash_and_equivalents / current_liabilities
    
    # ========== LEVERAGE RATIOS ==========
    
    @staticmethod
    def debt_to_equity(total_debt: UnivariateMeasurements, stockholders_equity: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Debt-to-equity = Total Debt / Stockholders' Equity
        
        Measures financial leverage.
        Lower is better (indicates less financial risk).
        
        Args:
            total_debt: Total debt time series
            stockholders_equity: Stockholders' equity time series
        
        Returns:
            Debt-to-equity ratio time series
        """
        return total_debt / stockholders_equity
    
    @staticmethod
    def debt_to_assets(total_debt: UnivariateMeasurements, total_assets: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Debt-to-assets = Total Debt / Total Assets
        
        Measures proportion of assets financed by debt.
        Lower is better (indicates less financial risk).
        
        Args:
            total_debt: Total debt time series
            total_assets: Total assets time series
        
        Returns:
            Debt-to-assets ratio time series
        """
        return total_debt / total_assets
    
    @staticmethod
    def equity_multiplier(total_assets: UnivariateMeasurements, stockholders_equity: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Equity multiplier = Total Assets / Stockholders' Equity
        
        Measures financial leverage (part of DuPont analysis).
        Higher indicates more leverage.
        
        Args:
            total_assets: Total assets time series
            stockholders_equity: Stockholders' equity time series
        
        Returns:
            Equity multiplier time series
        """
        return total_assets / stockholders_equity
    
    @staticmethod
    def interest_coverage(operating_income: UnivariateMeasurements, interest_expense: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Interest coverage = Operating Income / Interest Expense
        
        Measures ability to pay interest obligations.
        Higher is better (> 2.5 is generally considered safe).
        
        Args:
            operating_income: Operating income time series (EBIT)
            interest_expense: Interest expense time series
        
        Returns:
            Interest coverage ratio time series
        """
        return operating_income / interest_expense
    
    # ========== EFFICIENCY RATIOS ==========
    
    @staticmethod
    def asset_turnover(revenues: UnivariateMeasurements, total_assets: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Asset turnover = Revenue / Total Assets
        
        Measures how efficiently assets generate revenue.
        Higher is better (indicates efficient asset utilization).
        
        Args:
            revenues: Revenue time series
            total_assets: Total assets time series
        
        Returns:
            Asset turnover ratio time series
        """
        return revenues / total_assets
    
    @staticmethod
    def inventory_turnover(cost_of_revenue: UnivariateMeasurements, inventory: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Inventory turnover = Cost of Revenue / Inventory
        
        Measures how quickly inventory is sold.
        Higher is better (indicates efficient inventory management).
        
        Args:
            cost_of_revenue: Cost of revenue time series
            inventory: Inventory time series
        
        Returns:
            Inventory turnover ratio time series
        """
        return cost_of_revenue / inventory
    
    @staticmethod
    def days_inventory_outstanding(cost_of_revenue: UnivariateMeasurements, inventory: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Days inventory outstanding = 365 / Inventory Turnover
        
        Measures average days to sell inventory.
        Lower is better (indicates faster inventory turnover).
        
        Args:
            cost_of_revenue: Cost of revenue time series
            inventory: Inventory time series
        
        Returns:
            Days inventory outstanding time series
        """
        turnover = cost_of_revenue / inventory
        return 365.0 / turnover
    
    @staticmethod
    def receivables_turnover(revenues: UnivariateMeasurements, accounts_receivable: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Receivables turnover = Revenue / Accounts Receivable
        
        Measures how quickly receivables are collected.
        Higher is better (indicates efficient collections).
        
        Args:
            revenues: Revenue time series
            accounts_receivable: Accounts receivable time series
        
        Returns:
            Receivables turnover ratio time series
        """
        return revenues / accounts_receivable
    
    @staticmethod
    def days_sales_outstanding(revenues: UnivariateMeasurements, accounts_receivable: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Days sales outstanding = 365 / Receivables Turnover
        
        Measures average days to collect receivables.
        Lower is better (indicates faster collections).
        
        Args:
            revenues: Revenue time series
            accounts_receivable: Accounts receivable time series
        
        Returns:
            Days sales outstanding time series
        """
        turnover = revenues / accounts_receivable
        return 365.0 / turnover
    
    # ========== CASH FLOW METRICS ==========
    
    @staticmethod
    def free_cash_flow(operating_cash_flow: UnivariateMeasurements, capital_expenditures: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Free cash flow = Operating Cash Flow - Capital Expenditures
        
        Measures cash available for distribution to investors.
        Positive FCF indicates cash generation.
        
        Note: CapEx values are typically negative, so this is effectively OCF + |CapEx|.
        
        Args:
            operating_cash_flow: Operating cash flow time series
            capital_expenditures: Capital expenditures time series (negative values)
        
        Returns:
            Free cash flow time series
        """
        return operating_cash_flow + capital_expenditures
    
    @staticmethod
    def operating_cash_flow_ratio(operating_cash_flow: UnivariateMeasurements, current_liabilities: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        OCF ratio = Operating Cash Flow / Current Liabilities
        
        Measures ability to pay current liabilities from operating cash.
        > 1.0 indicates strong cash generation.
        
        Args:
            operating_cash_flow: Operating cash flow time series
            current_liabilities: Current liabilities time series
        
        Returns:
            OCF ratio time series
        """
        return operating_cash_flow / current_liabilities
    
    @staticmethod
    def cash_flow_to_net_income(operating_cash_flow: UnivariateMeasurements, net_income: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Cash flow to net income = Operating Cash Flow / Net Income
        
        Measures quality of earnings.
        > 1.0 indicates strong cash conversion.
        
        Args:
            operating_cash_flow: Operating cash flow time series
            net_income: Net income time series
        
        Returns:
            Cash flow to net income ratio time series
        """
        return operating_cash_flow / net_income


if __name__ == "__main__":
    # Example usage with the new readers
    from edgarito.services.financial.income_statement_reader import IncomeStatementReader
    from edgarito.enums.granularity import Granularity

    reader = IncomeStatementReader()
    reader.load_from_json_file("cache/edgar_rest/api/xbrl/companyfacts/CIK0001652044.json")

    revenues = reader.get_revenue(Granularity.ANNUAL)
    costs = reader.get_cost_of_revenue(Granularity.ANNUAL)

    # Align periods
    revenues = revenues.intersect(costs)
    costs = costs.intersect(revenues)

    # Compute metrics
    gross_profit = FinancialMetrics.gross_profit(revenues, costs)
    gross_margin = FinancialMetrics.gross_margin(revenues, costs)
    
    print("Gross Profit:", gross_profit)
    print("Gross Margin:", gross_margin)
