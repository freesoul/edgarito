"""
Balance Sheet Reader - provides access to balance sheet accounts.

Retrieves assets, liabilities, and equity accounts from US-GAAP concepts.
"""
from typing import Optional

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts
from edgarito.schemas.reader.measurements import UnivariateMeasurements
from edgarito.enums.granularity import Granularity
from edgarito.services.financial.base_reader import BaseStatementReader


class BalanceSheetReader(BaseStatementReader):
    """Reader for balance sheet accounts (assets, liabilities, equity)"""
    
    def __init__(self, facts: Optional[CompanyFacts] = None):
        super().__init__(facts)
    
    # ========== ASSETS ==========
    
    def get_total_assets(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Total assets.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of total assets
        """
        return self._get_concept("Assets", granularity, convert_fy_to_q4=False)
    
    def get_current_assets(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Current assets (cash, receivables, inventory, etc.).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of current assets
        """
        return self._get_concept("AssetsCurrent", granularity, convert_fy_to_q4=False)
    
    def get_cash_and_equivalents(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Cash and cash equivalents.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of cash and cash equivalents
        """
        return self._get_concept("CashAndCashEquivalentsAtCarryingValue", granularity, convert_fy_to_q4=False)
    
    def get_marketable_securities(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Marketable securities (short-term investments).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of marketable securities
        """
        return self._get_concept("MarketableSecuritiesCurrent", granularity, convert_fy_to_q4=False)
    
    def get_accounts_receivable(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Accounts receivable, net.
        
        Different companies may use different US-GAAP concepts for receivables:
        - AccountsReceivableNetCurrent: Most common, specific to trade receivables
        - ReceivablesNetCurrent: Broader category including all current receivables
        - AccountsReceivableNet: Sometimes used for net receivables
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of accounts receivable
        """
        return self._get_concept_with_fallbacks(
            concepts=["AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "AccountsReceivableNet"],
            granularity=granularity,
            convert_fy_to_q4=False
        )
    
    def get_inventory(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Inventory.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of inventory
        """
        return self._get_concept("InventoryNet", granularity, convert_fy_to_q4=False)
    
    def get_property_plant_equipment(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Property, plant, and equipment, net.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of PP&E
        """
        return self._get_concept("PropertyPlantAndEquipmentNet", granularity, convert_fy_to_q4=False)
    
    def get_goodwill(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Goodwill.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of goodwill
        """
        return self._get_concept("Goodwill", granularity, convert_fy_to_q4=False)
    
    def get_intangible_assets(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Intangible assets, net (excluding goodwill).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of intangible assets
        """
        return self._get_concept("IntangibleAssetsNetExcludingGoodwill", granularity, convert_fy_to_q4=False)
    
    # ========== LIABILITIES ==========
    
    def get_total_liabilities(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Total liabilities.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of total liabilities
        """
        return self._get_concept("Liabilities", granularity, convert_fy_to_q4=False)
    
    def get_current_liabilities(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Current liabilities (payables, accruals, short-term debt).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of current liabilities
        """
        return self._get_concept("LiabilitiesCurrent", granularity, convert_fy_to_q4=False)
    
    def get_accounts_payable(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Accounts payable.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of accounts payable
        """
        return self._get_concept("AccountsPayableCurrent", granularity, convert_fy_to_q4=False)
    
    def get_long_term_debt(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Long-term debt (excluding current portion).
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of long-term debt
        """
        return self._get_concept("LongTermDebtNoncurrent", granularity, convert_fy_to_q4=False)
    
    def get_total_debt(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Total debt (short-term + long-term).
        
        Tries to get comprehensive debt figure:
        1. LongTermDebtAndCapitalLeaseObligations (includes all debt)
        2. DebtCurrent + LongTermDebtNoncurrent (sum short + long term)
        3. LongTermDebt (fallback, may include current portion)
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of total debt
        """
        # Try comprehensive debt concept first
        try:
            return self._get_concept("LongTermDebtAndCapitalLeaseObligations", granularity, convert_fy_to_q4=False)
        except ValueError:
            pass
        
        # Try summing current + noncurrent debt
        try:
            current_debt = self._get_concept("DebtCurrent", granularity, convert_fy_to_q4=False)
            long_term_debt = self._get_concept("LongTermDebtNoncurrent", granularity, convert_fy_to_q4=False)
            
            # Sum the two series
            from edgarito.schemas.reader.measurements import UnivariateMeasurements
            
            # Create period-to-value dictionaries
            current_dict = {p: v for v, p in current_debt}
            lt_dict = {p: v for v, p in long_term_debt}
            
            # Find common periods and sum
            common_periods = set(current_dict.keys()) & set(lt_dict.keys())
            if common_periods:
                total_values = [current_dict[p] + lt_dict[p] for p in sorted(common_periods)]
                total_periods = sorted(common_periods)
                
                return UnivariateMeasurements(
                    concept="TotalDebt (computed)",
                    granularity=granularity,
                    values=total_values,
                    periods=total_periods
                )
        except ValueError:
            pass
        
        # Fallback to LongTermDebt (may include current portion)
        return self._get_concept("LongTermDebt", granularity, convert_fy_to_q4=False)
    
    # ========== EQUITY ==========
    
    def get_stockholders_equity(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Total stockholders' equity.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of stockholders' equity
        """
        return self._get_concept("StockholdersEquity", granularity, convert_fy_to_q4=False)
    
    def get_retained_earnings(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Retained earnings.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of retained earnings
        """
        return self._get_concept("RetainedEarningsAccumulatedDeficit", granularity, convert_fy_to_q4=False)
    
    def get_common_stock_shares_outstanding(self, granularity: Granularity) -> UnivariateMeasurements:
        """
        Common stock shares outstanding.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of shares outstanding
        """
        return self._get_concept("CommonStockSharesOutstanding", granularity, convert_fy_to_q4=False)
