"""
Cash Flow Statement Reader - provides access to cash flow statement accounts.

Retrieves operating, investing, and financing activities from US-GAAP concepts.
"""
from typing import Optional

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts
from edgarito.schemas.reader.measurements import UnivariateMeasurements
from edgarito.enums.granularity import Granularity
from edgarito.services.financial.base_reader import BaseStatementReader


class CashFlowStatementReader(BaseStatementReader):
    """Reader for cash flow statement accounts (operating, investing, financing)"""
    
    def __init__(self, facts: Optional[CompanyFacts] = None):
        super().__init__(facts)
    
    # ========== OPERATING ACTIVITIES ==========
    
    def get_operating_cash_flow(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Net cash provided by (used in) operating activities.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of operating cash flow
        """
        return self._get_concept("NetCashProvidedByUsedInOperatingActivities", granularity)
    
    def get_depreciation_and_amortization(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Depreciation, depletion, and amortization (non-cash expense added back).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of depreciation and amortization
        """
        return self._get_concept("DepreciationDepletionAndAmortization", granularity)
    
    def get_stock_based_compensation(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Share-based compensation (non-cash expense added back).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of stock-based compensation
        """
        return self._get_concept("ShareBasedCompensation", granularity)
    
    def get_deferred_income_taxes(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Deferred income taxes.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of deferred income taxes
        """
        return self._get_concept("DeferredIncomeTaxExpenseBenefit", granularity)
    
    def get_changes_in_working_capital(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Changes in operating assets and liabilities (working capital).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of working capital changes
        """
        return self._get_concept("IncreaseDecreaseInOperatingCapital", granularity)
    
    # ========== INVESTING ACTIVITIES ==========
    
    def get_investing_cash_flow(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Net cash provided by (used in) investing activities.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of investing cash flow
        """
        return self._get_concept("NetCashProvidedByUsedInInvestingActivities", granularity)
    
    def get_capital_expenditures(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Payments to acquire property, plant, and equipment (CapEx).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of capital expenditures (negative values)
        """
        return self._get_concept("PaymentsToAcquirePropertyPlantAndEquipment", granularity)
    
    def get_acquisitions(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Payments to acquire businesses, net of cash acquired.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of acquisition payments (negative values)
        """
        return self._get_concept("PaymentsToAcquireBusinessesNetOfCashAcquired", granularity)
    
    def get_purchases_of_investments(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Payments to acquire investments (marketable securities, etc.).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of investment purchases (negative values)
        """
        return self._get_concept("PaymentsToAcquireInvestments", granularity)
    
    def get_proceeds_from_investments(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Proceeds from sale and maturity of investments.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of investment proceeds (positive values)
        """
        return self._get_concept("ProceedsFromSaleAndMaturityOfMarketableSecurities", granularity)
    
    # ========== FINANCING ACTIVITIES ==========
    
    def get_financing_cash_flow(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Net cash provided by (used in) financing activities.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of financing cash flow
        """
        return self._get_concept("NetCashProvidedByUsedInFinancingActivities", granularity)
    
    def get_proceeds_from_debt(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Proceeds from issuance of long-term debt.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of debt proceeds (positive values)
        """
        return self._get_concept("ProceedsFromIssuanceOfLongTermDebt", granularity)
    
    def get_repayments_of_debt(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Repayments of long-term debt.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of debt repayments (negative values)
        """
        return self._get_concept("RepaymentsOfLongTermDebt", granularity)
    
    def get_proceeds_from_stock_issuance(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Proceeds from issuance of common stock.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of stock issuance proceeds (positive values)
        """
        return self._get_concept("ProceedsFromIssuanceOfCommonStock", granularity)
    
    def get_stock_repurchases(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Payments for repurchase of common stock (buybacks).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of stock repurchases (negative values)
        """
        return self._get_concept("PaymentsForRepurchaseOfCommonStock", granularity)
    
    def get_dividends_paid(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Payments of dividends to shareholders.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of dividend payments (negative values)
        """
        return self._get_concept("PaymentsOfDividends", granularity)
    
    # ========== SUMMARY ==========
    
    def get_change_in_cash(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Net change in cash and cash equivalents during the period.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of cash changes
        """
        return self._get_concept("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect", granularity)
    
    def get_beginning_cash_balance(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Cash and cash equivalents at beginning of period.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of beginning cash balances
        """
        return self._get_concept("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", granularity)
