"""
Income Statement Reader - provides access to income statement accounts.

Retrieves revenues, expenses, and income metrics from US-GAAP concepts.
"""
import logging
from typing import Optional

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts
from edgarito.schemas.reader.measurements import UnivariateMeasurements
from edgarito.enums.granularity import Granularity
from edgarito.services.financial.base_reader import BaseStatementReader

logger = logging.getLogger(__name__)


class IncomeStatementReader(BaseStatementReader):
    """Reader for income statement accounts (revenues, expenses, income)"""
    
    def __init__(self, facts: Optional[CompanyFacts] = None):
        super().__init__(facts)
    
    # ========== REVENUES ==========
    
    def get_revenue(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Total revenue (also called sales or turnover).
        
        Companies may use different revenue concept names depending on their
        reporting period and accounting standards changes. This method tries
        multiple common revenue concepts to ensure comprehensive data coverage.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of revenue
        """
        # Try multiple revenue concepts in order of preference:
        # 1. RevenueFromContractWithCustomerExcludingAssessedTax (newer ASC 606 standard)
        # 2. Revenues (traditional/older filings)
        # 3. RevenueFromContractWithCustomerIncludingAssessedTax (alternative ASC 606)
        # 4. SalesRevenueNet (some companies use this)
        return self._get_concept_with_fallbacks(
            concepts=[
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet"
            ],
            granularity=granularity
        )
    
    def get_cost_of_revenue(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Cost of revenue (also called COGS - cost of goods sold).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of cost of revenue
        """
        return self._get_concept("CostOfRevenue", granularity)
    
    # ========== EXPENSES ==========
    
    def get_operating_expenses(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Total operating expenses.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of operating expenses
        """
        return self._get_concept("OperatingExpenses", granularity)
    
    def get_research_and_development_expense(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Research and development expenses.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of R&D expenses
        """
        return self._get_concept("ResearchAndDevelopmentExpense", granularity)
    
    def get_selling_general_administrative_expense(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Selling, general, and administrative expenses (SG&A).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of SG&A expenses
        """
        return self._get_concept("SellingGeneralAndAdministrativeExpense", granularity)
    
    def get_depreciation_and_amortization(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Depreciation and amortization expense.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of depreciation and amortization
        """
        return self._get_concept("DepreciationDepletionAndAmortization", granularity)
    
    def get_interest_expense(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Interest expense.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of interest expense
        """
        return self._get_concept("InterestExpense", granularity)
    
    def get_income_tax_expense(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Income tax expense.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of income tax expense
        """
        return self._get_concept("IncomeTaxExpenseBenefit", granularity)
    
    # ========== INCOME METRICS ==========
    
    def get_gross_profit(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Gross profit (revenue - cost of revenue).
        
        Strategy:
        1. Try direct GrossProfit concept
        2. Check if data is fresh (< 365 days)
        3. If stale or missing, calculate from Revenue - Costs
        
        If GrossProfit concept is not available or is stale, attempts to calculate it from:
        Revenue - DirectOperatingCosts (or other cost concepts)
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of gross profit
        """
        from datetime import datetime, timedelta
        from edgarito.services.analysis.period_alignment import check_series_freshness
        
        # Try to get direct GrossProfit concept
        try:
            gross_profit = self._get_concept("GrossProfit", granularity)
            
            # Check if data is fresh (< 365 days old)
            is_fresh = check_series_freshness(gross_profit, max_age_days=365, context="GrossProfit")
            if is_fresh:
                # Data is fresh, use it
                return gross_profit
            else:
                # Data is stale, fall through to calculation
                logger.info(f"GrossProfit concept exists but data is stale (>365 days), will calculate from components")
                
        except ValueError:
            # GrossProfit not available - try to calculate from components
            logger.info("GrossProfit concept not found, will calculate from Revenue - Costs")
        
        # Get revenue for calculation
        revenue = self.get_revenue(granularity)
        
        # Try different cost concepts in order of preference
        cost_concepts = [
            "DirectOperatingCosts",
            "CostOfRevenue",
            "CostOfGoodsAndServicesSold",
            "CostOfGoodsSold"
        ]
        
        for cost_concept in cost_concepts:
            try:
                costs = self._get_concept(cost_concept, granularity)
                
                # Calculate gross profit = revenue - costs
                # Find overlapping periods
                revenue_periods_set = set(revenue.periods)
                cost_periods_set = set(costs.periods)
                overlapping_periods = sorted(list(revenue_periods_set & cost_periods_set))
                
                if not overlapping_periods:
                    continue  # Try next cost concept
                
                # Build aligned values
                gross_profit_values = []
                gross_profit_periods = []
                
                for period in overlapping_periods:
                    # Find indices in both series
                    rev_idx = revenue.periods.index(period)
                    cost_idx = costs.periods.index(period)
                    
                    # Calculate gross profit
                    gp_value = revenue.values[rev_idx] - costs.values[cost_idx]
                    gross_profit_values.append(gp_value)
                    gross_profit_periods.append(period)
                
                # Return as UnivariateMeasurements
                logger.info(f"Successfully calculated GrossProfit from Revenue - {cost_concept} ({len(gross_profit_periods)} periods)")
                return UnivariateMeasurements(
                    concept=f"GrossProfit (calculated from Revenue - {cost_concept})",
                    granularity=granularity,
                    values=gross_profit_values,
                    periods=gross_profit_periods
                )
                
            except ValueError:
                continue  # Try next cost concept
        
        # If we get here, couldn't calculate gross profit
        raise ValueError(
            f"No USD measurements found for concept 'GrossProfit' and unable to calculate from components. "
            f"Tried: {', '.join(cost_concepts)}"
        )

    
    def get_operating_income(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Operating income (also called EBIT - earnings before interest and taxes).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of operating income
        """
        return self._get_concept("OperatingIncomeLoss", granularity)
    
    def get_income_before_tax(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Income before income taxes (also called pretax income or EBT).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of income before tax
        """
        return self._get_concept("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", granularity)
    
    def get_net_income(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Net income (bottom line profit).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of net income
        """
        return self._get_concept("NetIncomeLoss", granularity)
    
    def get_earnings_per_share_basic(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Basic earnings per share (EPS).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of basic EPS
        """
        return self._get_concept("EarningsPerShareBasic", granularity)
    
    def get_earnings_per_share_diluted(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Diluted earnings per share (EPS).
        
        EPS is often reported in 'pure' units (as a ratio) rather than USD.
        If the direct concept is not available in USD, this method:
        1. Checks for EPS in 'pure' units
        2. Calculates from NetIncomeLoss / WeightedAverageShares if needed
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of diluted EPS
        """
        # First try the standard method (checks USD units)
        try:
            return self._get_concept("EarningsPerShareDiluted", granularity)
        except ValueError as e:
            # Check if it's available in 'pure' units instead of USD
            if self._data.facts.us_gaap and "EarningsPerShareDiluted" in self._data.facts.us_gaap:
                fact = self._data.facts.us_gaap["EarningsPerShareDiluted"]
                if fact.units and fact.units.pure:
                    # EPS is in pure units - extract it
                    logger.info(f"Using EPS from 'pure' units (not USD) - {len(fact.units.pure)} periods")
                    
                    from edgarito.enums.edgar.core_filing_type import CoreFilingType
                    
                    # Use same filtering logic as _get_concept
                    filing_types = [CoreFilingType.FILING_10K, CoreFilingType.FILING_10Q] if granularity == Granularity.QUARTERLY else [CoreFilingType.FILING_10K]
                    
                    all_measurements = fact.units.pure
                    filtered_measurements = []
                    for filing_type in filing_types:
                        filtered = self._filter_measurements(all_measurements, filing_type)
                        filtered_measurements.extend(filtered)
                    
                    filtered_measurements = self._deduplicate_measurements(filtered_measurements)
                    
                    if not filtered_measurements:
                        raise ValueError(f"No measurements found for concept 'EarningsPerShareDiluted' in pure units with {granularity}")
                    
                    univariate = UnivariateMeasurements.from_measurements(
                        concept="EarningsPerShareDiluted",
                        granularity=granularity,
                        measurements=filtered_measurements
                    )
                    univariate.sort()
                    return univariate
            
            # EPS not available directly - try to calculate from components
            try:
                net_income = self.get_net_income(granularity)
                shares_diluted = self.get_weighted_average_shares_outstanding_diluted(granularity)
                
                # Find overlapping periods
                ni_periods_set = set(net_income.periods)
                shares_periods_set = set(shares_diluted.periods)
                overlapping_periods = sorted(list(ni_periods_set & shares_periods_set))
                
                if not overlapping_periods:
                    raise ValueError(
                        "Cannot calculate EPS: no overlapping periods between NetIncomeLoss and WeightedAverageShares"
                    )
                
                # Calculate EPS for overlapping periods
                eps_values = []
                eps_periods = []
                
                for period in overlapping_periods:
                    ni_idx = net_income.periods.index(period)
                    shares_idx = shares_diluted.periods.index(period)
                    
                    # EPS = Net Income / Shares
                    if shares_diluted.values[shares_idx] != 0:
                        eps_value = net_income.values[ni_idx] / shares_diluted.values[shares_idx]
                        eps_values.append(eps_value)
                        eps_periods.append(period)
                
                if not eps_values:
                    raise ValueError("Cannot calculate EPS: no valid periods with non-zero shares")
                
                return UnivariateMeasurements(
                    concept="EarningsPerShareDiluted (calculated from NetIncomeLoss / Shares)",
                    granularity=granularity,
                    values=eps_values,
                    periods=eps_periods
                )
                
            except ValueError as calc_error:
                # Re-raise the original error with additional context
                raise ValueError(
                    f"{str(e)}. Also failed to calculate from components: {str(calc_error)}"
                )
    
    def get_weighted_average_shares_outstanding_basic(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Weighted average shares outstanding (basic).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of weighted average shares (basic)
        """
        return self._get_concept("WeightedAverageNumberOfSharesOutstandingBasic", granularity)
    
    def get_weighted_average_shares_outstanding_diluted(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Weighted average shares outstanding (diluted).
        
        Share counts are typically reported in 'shares' units rather than USD.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of weighted average shares (diluted)
        """
        # Shares data is in 'shares' units, not USD - need special handling
        self._require_loaded()
        
        if not self._data.facts.us_gaap:
            raise ValueError("Company data does not have US-GAAP facts")
        
        facts_dict = self._data.facts.us_gaap
        concept = "WeightedAverageNumberOfDilutedSharesOutstanding"
        
        if concept not in facts_dict:
            raise ValueError(f"Concept '{concept}' not found in US-GAAP company data")
        
        fact = facts_dict[concept]
        
        # Check shares units (not USD)
        if not fact.units or not fact.units.shares:
            raise ValueError(f"No share measurements found for concept '{concept}'")
        
        # Use shares instead of USD
        all_measurements = fact.units.shares
        
        # Apply same filtering logic as _get_concept
        from edgarito.enums.edgar.core_filing_type import CoreFilingType
        
        filing_types = [CoreFilingType.FILING_10K, CoreFilingType.FILING_10Q] if granularity == Granularity.QUARTERLY else [CoreFilingType.FILING_10K]
        
        filtered_measurements = []
        for filing_type in filing_types:
            filtered = self._filter_measurements(all_measurements, filing_type)
            filtered_measurements.extend(filtered)
        
        filtered_measurements = self._deduplicate_measurements(filtered_measurements)
        
        univariate = UnivariateMeasurements.from_measurements(
            concept=concept,
            granularity=granularity,
            measurements=filtered_measurements
        )
        univariate.sort()
        
        return univariate
