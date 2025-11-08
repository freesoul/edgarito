"""Display and formatting utilities for CLI output."""

from typing import Optional
from edgarito.schemas.edgar_responses.company_facts import CompanyFacts
from edgarito.services.financial.statements import FinancialStatements
from edgarito.enums.granularity import Granularity
from edgarito.enums.edgar.period import FiscalPeriod


class FinancialFormatter:
    """Handles formatting and display of financial data."""

    @staticmethod
    def display_financials(facts: CompanyFacts, identifier: str):
        """Format and display financial statements for all periods."""
        statements = FinancialStatements(facts)
        
        print(f"\n{'='*100}")
        print(f"FINANCIAL STATEMENTS: {identifier.upper()} - {facts.entityName}")
        print(f"{'='*100}")
        
        # Display Annual Data
        FinancialFormatter.display_period_data(statements, Granularity.ANNUAL)
        
        # Display Quarterly Data
        FinancialFormatter.display_period_data(statements, Granularity.QUARTERLY)

    @staticmethod
    def display_period_data(statements: FinancialStatements, granularity: Granularity):
        """Display financial data for a specific granularity (ANNUAL or QUARTERLY)."""
        period_label = "ANNUAL" if granularity.name == "ANNUAL" else "QUARTERLY"
        
        print(f"\n{'-'*100}")
        print(f"{period_label} DATA")
        print(f"{'-'*100}\n")
        
        # Get all statement data
        try:
            assets = statements.balance_sheet.get_total_assets(granularity)
            revenue = statements.income_statement.get_revenue(granularity)
            operating_income = statements.income_statement.get_operating_income(granularity)
            net_income = statements.income_statement.get_net_income(granularity)
            ocf = statements.cash_flow.get_operating_cash_flow(granularity)
            
            # Get all unique periods, but prioritize periods with income statement data
            # We want to show periods where we have meaningful financial data, not just balance sheet
            all_periods = set()
            
            # Primary data sources (income statement is most important)
            if revenue: all_periods.update(revenue.periods)
            if operating_income: all_periods.update(operating_income.periods)
            if net_income: all_periods.update(net_income.periods)
            
            # Add balance sheet periods that also have some income/cash flow data
            if assets:
                income_periods = set()
                if revenue: income_periods.update(revenue.periods)
                if operating_income: income_periods.update(operating_income.periods)
                if net_income: income_periods.update(net_income.periods)
                if ocf: income_periods.update(ocf.periods)
                
                # Only add balance sheet periods if they have ANY income/cash flow data
                for period in assets.periods:
                    if period in income_periods:
                        all_periods.add(period)
            
            # Add OCF periods
            if ocf: all_periods.update(ocf.periods)
            
            if not all_periods:
                print(f"No {period_label.lower()} data available.\n")
                return
            
            # Sort periods chronologically
            sorted_periods = sorted(all_periods, key=lambda p: (p.year, p.fp.value if p.fp else 0))
            
            # Print header (wider period column for end dates)
            print(f"{'Period':<25} {'Assets':<18} {'Revenue':<18} {'Op. Income':<18} {'Net Income':<18} {'Op. Cash Flow':<18}")
            print(f"{'-'*123}")
            
            # Print each period
            for period in sorted_periods:
                # Build period string with fiscal year and actual end date
                if period.end_date:
                    period_str = f"FY{period.year}"
                    if period.fp and granularity.name == "QUARTERLY":
                        period_str += f" {period.fp.value} ({period.end_date.strftime('%b %Y')})"
                    else:
                        period_str += f" ({period.end_date.strftime('%b %Y')})"
                else:
                    # Fallback if no end_date available
                    period_str = f"{period.year}"
                    if period.fp and granularity.name == "QUARTERLY":
                        period_str += f" {period.fp.value}"
                
                assets_val = FinancialFormatter.format_value(assets, period) if assets else "N/A"
                revenue_val = FinancialFormatter.format_value(revenue, period) if revenue else "N/A"
                op_income_val = FinancialFormatter.format_value(operating_income, period) if operating_income else "N/A"
                net_income_val = FinancialFormatter.format_value(net_income, period) if net_income else "N/A"
                ocf_val = FinancialFormatter.format_value(ocf, period) if ocf else "N/A"
                
                print(f"{period_str:<25} {assets_val:<18} {revenue_val:<18} {op_income_val:<18} {net_income_val:<18} {ocf_val:<18}")
            
            # Try to compute and display key metrics
            FinancialFormatter.display_metrics(statements, granularity, sorted_periods)
            
        except Exception as e:
            print(f"Error displaying {period_label.lower()} data: {e}\n")

    @staticmethod
    def format_value(measurements, period) -> str:
        """Format a value in millions with proper unit."""
        try:
            value_dict = {p: v for p, v in zip(measurements.periods, measurements.values)}
            if period in value_dict:
                value = value_dict[period]
                # Convert to millions
                value_millions = value / 1_000_000
                return f"${value_millions:,.1f}M"
            
            # For balance sheet, check if there's an FY period when looking for Q4
            # Balance sheets are point-in-time, so FY period = Q4 period
            if period.fp == FiscalPeriod.Q4:
                fy_period = [p for p in measurements.periods if p.year == period.year and p.fp == FiscalPeriod.Year]
                if fy_period:
                    value = value_dict[fy_period[0]]
                    value_millions = value / 1_000_000
                    return f"${value_millions:,.1f}M"
            
            return "N/A"
        except:
            return "N/A"

    @staticmethod
    def display_metrics(statements: FinancialStatements, granularity: Granularity, sorted_periods):
        """Display computed financial metrics."""
        try:
            # Try to compute some key metrics
            gross_margin = statements.metrics.gross_margin(granularity)
            net_margin = statements.metrics.net_margin(granularity)
            roe = statements.metrics.return_on_equity(granularity)
            current_ratio = statements.metrics.current_ratio(granularity)
            
            if any([gross_margin, net_margin, roe, current_ratio]):
                print(f"\n{'Metrics':<15} {'Gross Margin':<15} {'Net Margin':<15} {'ROE':<15} {'Current Ratio':<15}")
                print(f"{'-'*75}")
                
                for period in sorted_periods:
                    period_str = f"{period.year}"
                    if period.fp and granularity.name == "QUARTERLY":
                        period_str += f" {period.fp.value}"
                    
                    gm = FinancialFormatter.format_ratio(gross_margin, period) if gross_margin else "N/A"
                    nm = FinancialFormatter.format_ratio(net_margin, period) if net_margin else "N/A"
                    roe_val = FinancialFormatter.format_ratio(roe, period) if roe else "N/A"
                    cr = FinancialFormatter.format_ratio(current_ratio, period, is_percentage=False) if current_ratio else "N/A"
                    
                    print(f"{period_str:<15} {gm:<15} {nm:<15} {roe_val:<15} {cr:<15}")
                
        except Exception as e:
            # Silently skip metrics if computation fails
            pass
        
        print()  # Add spacing after section

    @staticmethod
    def format_ratio(measurements, period, is_percentage: bool = True) -> str:
        """Format a ratio value."""
        try:
            value_dict = {p: v for p, v in zip(measurements.periods, measurements.values)}
            if period in value_dict:
                value = value_dict[period]
                if is_percentage:
                    return f"{value*100:.1f}%"
                else:
                    return f"{value:.2f}x"
            return "N/A"
        except:
            return "N/A"
