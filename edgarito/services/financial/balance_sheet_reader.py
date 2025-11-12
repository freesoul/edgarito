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
        
        For IFRS companies that don't report "Liabilities" directly, 
        computes as EquityAndLiabilities - Equity.
        
        Args:
            granularity: ANNUAL or QUARTERLY
        
        Returns:
            Time series of total liabilities
        """
        try:
            return self._get_concept("Liabilities", granularity, convert_fy_to_q4=False)
        except ValueError as e:
            # If Liabilities not found, try computing from IFRS balance sheet equation
            if "not found" in str(e):
                try:
                    # Check if we're dealing with IFRS company
                    if not self._data.facts.us_gaap and self._data.facts.ifrs_full:
                        # IFRS: Total Liabilities = EquityAndLiabilities - Equity
                        # Access IFRS concepts directly
                        facts_dict = self._data.facts.ifrs_full
                        
                        if "EquityAndLiabilities" in facts_dict and "Equity" in facts_dict:
                            # Get the data using the IFRS concept names directly
                            from edgarito.services.financial.ifrs_mapping import get_gaap_concept
                            
                            # Temporarily get equity using the mapped concept
                            equity = self._get_concept("StockholdersEquity", granularity, convert_fy_to_q4=False)
                            
                            # For EquityAndLiabilities, we need to access it directly since it's an IFRS concept
                            # Use the parent _get_concept but with the IFRS name
                            from edgarito.enums.edgar.core_filing_type import CoreFilingType
                            from edgarito.schemas.edgar_responses.company_facts import Measurement
                            from edgarito.schemas.reader.measurements import UnivariateMeasurements
                            
                            # Get EquityAndLiabilities measurements
                            filing_types = [CoreFilingType.FILING_10K, CoreFilingType.FILING_20F] if granularity == Granularity.ANNUAL else [CoreFilingType.FILING_10K, CoreFilingType.FILING_10Q, CoreFilingType.FILING_20F]
                            
                            eur_data = getattr(facts_dict["EquityAndLiabilities"].units, 'EUR', None)
                            if eur_data:
                                eal_measurements = [Measurement(**m) if isinstance(m, dict) else m for m in eur_data]
                                filtered_eal = []
                                for filing_type in filing_types:
                                    filtered = self._filter_measurements(eal_measurements, filing_type)
                                    filtered_eal.extend(filtered)
                                
                                eal_univariate = UnivariateMeasurements.from_measurements(
                                    concept="EquityAndLiabilities",
                                    granularity=granularity,
                                    measurements=filtered_eal
                                )
                                eal_univariate.sort()
                                self._deduplicate_periods(eal_univariate)
                                
                                # Compute liabilities
                                liabilities_values = []
                                liabilities_periods = []
                                
                                equity_dict = {p: v for v, p in equity}
                                
                                for value, period in eal_univariate:
                                    if period in equity_dict:
                                        liabilities_value = value - equity_dict[period]
                                        liabilities_values.append(liabilities_value)
                                        liabilities_periods.append(period)
                                
                                result = UnivariateMeasurements(
                                    concept="Liabilities (computed)",
                                    granularity=granularity,
                                    values=liabilities_values,
                                    periods=liabilities_periods
                                )
                                result.sort()
                                return result
                except Exception as inner_e:
                    # If that also fails, re-raise the original error
                    raise e
            raise e
    
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
