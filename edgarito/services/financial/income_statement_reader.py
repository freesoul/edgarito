"""
Income Statement Reader - provides access to income statement accounts.

Retrieves revenues, expenses, and income metrics from US-GAAP concepts.
"""
from typing import Optional

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts
from edgarito.schemas.reader.measurements import UnivariateMeasurements
from edgarito.enums.granularity import Granularity
from edgarito.services.financial.base_reader import BaseStatementReader


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
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of gross profit
        """
        return self._get_concept("GrossProfit", granularity)
    
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
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of diluted EPS
        """
        return self._get_concept("EarningsPerShareDiluted", granularity)
    
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
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of weighted average shares (diluted)
        """
        return self._get_concept("WeightedAverageNumberOfDilutedSharesOutstanding", granularity)
