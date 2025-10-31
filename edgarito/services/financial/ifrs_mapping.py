"""
IFRS to US-GAAP Concept Mapping

Maps IFRS (International Financial Reporting Standards) concepts to their
US-GAAP equivalents to enable analysis of foreign companies.
"""
from typing import Dict, Optional


# IFRS-Full to US-GAAP concept mapping
IFRS_TO_GAAP_MAPPING: Dict[str, str] = {
    # ========== ASSETS ==========
    "Assets": "Assets",
    "CurrentAssets": "AssetsCurrent",
    "NoncurrentAssets": "AssetsNoncurrent",
    "CashAndCashEquivalents": "CashAndCashEquivalentsAtCarryingValue",
    "TradeAndOtherCurrentReceivables": "AccountsReceivableNetCurrent",
    "TradeReceivables": "AccountsReceivableNetCurrent",
    "Inventories": "InventoryNet",
    "PropertyPlantAndEquipment": "PropertyPlantAndEquipmentNet",
    "Goodwill": "Goodwill",
    "IntangibleAssetsOtherThanGoodwill": "IntangibleAssetsNetExcludingGoodwill",
    "CurrentFinancialAssets": "MarketableSecuritiesCurrent",
    "OtherCurrentAssets": "OtherAssetsCurrent",
    
    # ========== LIABILITIES ==========
    # Note: IFRS doesn't always have a direct "Liabilities" line item
    # Some companies report EquityAndLiabilities and Equity separately
    "Liabilities": "Liabilities",  # When available
    "LiabilitiesCurrent": "LiabilitiesCurrent",  # Map to itself when available
    "EquityAndLiabilities": "LiabilitiesAndStockholdersEquity",
    "CurrentLiabilities": "LiabilitiesCurrent",
    "NoncurrentLiabilities": "LiabilitiesNoncurrent",
    "TradeAndOtherCurrentPayables": "AccountsPayableCurrent",
    "BorrowingsAndOtherFinancialLiabilities": "LongTermDebt",
    "LongTermBorrowings": "LongTermDebtNoncurrent",
    "FinancialLiabilities": "LongTermDebt",
    "LeaseLiabilities": "LongTermDebt",  # Close approximation
    
    # ========== EQUITY ==========
    "Equity": "StockholdersEquity",
    "EquityAttributableToOwnersOfParent": "StockholdersEquity",
    "RetainedEarnings": "RetainedEarningsAccumulatedDeficit",
    "IssuedCapital": "CommonStockValue",
    "ShareCapital": "CommonStockValue",
    
    # ========== INCOME STATEMENT ==========
    "Revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "CostOfSales": "CostOfRevenue",
    "GrossProfit": "GrossProfit",
    "OperatingExpenses": "OperatingExpenses",
    "ProfitLossFromOperatingActivities": "OperatingIncomeLoss",
    "ProfitLossBeforeTax": "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "ProfitLoss": "NetIncomeLoss",
    "ProfitLossAttributableToOwnersOfParent": "NetIncomeLoss",
    "InterestExpense": "InterestExpense",
    "IncomeTaxExpenseContinuingOperations": "IncomeTaxExpenseBenefit",
    "DepreciationAndAmortisationExpense": "DepreciationDepletionAndAmortization",
    
    # ========== CASH FLOW ==========
    "CashFlowsFromUsedInOperatingActivities": "NetCashProvidedByUsedInOperatingActivities",
    "CashFlowsFromUsedInInvestingActivities": "NetCashProvidedByUsedInInvestingActivities",
    "CashFlowsFromUsedInFinancingActivities": "NetCashProvidedByUsedInFinancingActivities",
    "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities": "PaymentsToAcquirePropertyPlantAndEquipment",
    "PurchaseOfIntangibleAssetsClassifiedAsInvestingActivities": "PaymentsToAcquireIntangibleAssets",
    "PaymentsToAcquireBusinessesNetOfCashAcquired": "PaymentsToAcquireBusinessesNetOfCashAcquired",
    "ProceedsFromIssuingShares": "ProceedsFromIssuanceOfCommonStock",
    "PaymentsForRepurchaseOfEquityInstruments": "PaymentsForRepurchaseOfCommonStock",
    "DividendsPaid": "PaymentsOfDividends",
    "DividendsPaidToEquityHoldersOfParentClassifiedAsFinancingActivities": "PaymentsOfDividends",
    "ProceedsFromBorrowings": "ProceedsFromIssuanceOfLongTermDebt",
    "RepaymentsOfBorrowings": "RepaymentsOfLongTermDebt",
    "IncreaseDecreaseInCashAndCashEquivalents": "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect",
    
    # ========== ADDITIONAL MAPPINGS ==========
    "ExpenseFromSharebasedPaymentTransactionsInWhichGoodsOrServicesReceivedDidNotQualifyForRecognitionAsAssets": "ShareBasedCompensation",
    "IncreaseDecreaseInTradeAccountReceivable": "IncreaseDecreaseInAccountsReceivable",
    "IncreaseDecreaseInInventories": "IncreaseDecreaseInInventories",
    "IncreaseDecreaseInTradeAccountPayable": "IncreaseDecreaseInAccountsPayable",
}


def get_gaap_concept(ifrs_concept: str) -> Optional[str]:
    """
    Get the US-GAAP equivalent for an IFRS concept.
    
    Args:
        ifrs_concept: IFRS-Full concept name
    
    Returns:
        US-GAAP concept name, or None if no mapping exists
    """
    return IFRS_TO_GAAP_MAPPING.get(ifrs_concept)


def get_ifrs_concept(gaap_concept: str) -> Optional[str]:
    """
    Get the IFRS equivalent for a US-GAAP concept (reverse mapping).
    
    Args:
        gaap_concept: US-GAAP concept name
    
    Returns:
        IFRS-Full concept name, or None if no mapping exists
    """
    # Create reverse mapping
    reverse_map = {v: k for k, v in IFRS_TO_GAAP_MAPPING.items()}
    return reverse_map.get(gaap_concept)


def list_available_mappings() -> Dict[str, str]:
    """
    Get all available IFRS to US-GAAP mappings.
    
    Returns:
        Dictionary of IFRS concept -> US-GAAP concept
    """
    return IFRS_TO_GAAP_MAPPING.copy()
