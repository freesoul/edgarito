from enum import Enum


class CoreFilingType(Enum):
    """
    Essential SEC filing types for investment analysis and financial data extraction.
    
    This is a curated subset focusing on the most important filings for investors.
    For the complete list of all SEC form types, see enums.edgar.reference.filing_type
    """
    
    # Annual Reports
    FILING_10K = '10-K'           # Annual report (US companies)
    FILING_10K_AMEND = '10-K/A'   # Amendment to 10-K
    FILING_20F = '20-F'           # Annual report (foreign companies)
    
    # Quarterly Reports  
    FILING_10Q = '10-Q'           # Quarterly report
    FILING_10Q_AMEND = '10-Q/A'   # Amendment to 10-Q
    FILING_6K = '6-K'             # Current report (foreign companies)
    
    # Current Events
    FILING_8K = '8-K'             # Material events and corporate changes
    FILING_8K_AMEND = '8-K/A'     # Amendment to 8-K
    
    # Proxy Statements
    FILING_DEF14A = 'DEF 14A'     # Definitive proxy statement
    FILING_DEFA14A = 'DEFA14A'    # Additional proxy materials
    
    # Registration & IPOs
    FILING_S1 = 'S-1'             # Registration statement (IPOs)
    FILING_S3 = 'S-3'             # Registration statement (shelf offerings)
    FILING_S4 = 'S-4'             # Registration statement (M&A)
    
    # Ownership & Insider Trading
    FILING_SC13D = 'SC 13D'       # Beneficial ownership (>5% stake)
    FILING_SC13G = 'SC 13G'       # Beneficial ownership (passive)
    FILING_13F = '13F-HR'         # Institutional holdings (>$100M AUM)
    FILING_3 = '3'                # Initial statement of beneficial ownership
    FILING_4 = '4'                # Changes in beneficial ownership (insider trading)
    FILING_5 = '5'                # Annual statement of beneficial ownership
    
    @classmethod
    def is_annual_report(cls, form: str) -> bool:
        """Check if a form type is an annual report"""
        return form in (cls.FILING_10K.value, cls.FILING_10K_AMEND.value, cls.FILING_20F.value)
    
    @classmethod
    def is_quarterly_report(cls, form: str) -> bool:
        """Check if a form type is a quarterly report"""
        return form in (cls.FILING_10Q.value, cls.FILING_10Q_AMEND.value, cls.FILING_6K.value)
    
    @classmethod
    def is_periodic_report(cls, form: str) -> bool:
        """Check if a form type is a periodic financial report (annual or quarterly)"""
        return cls.is_annual_report(form) or cls.is_quarterly_report(form)
    
    @classmethod
    def is_insider_filing(cls, form: str) -> bool:
        """Check if a form type relates to insider trading or beneficial ownership"""
        return form in (cls.FILING_3.value, cls.FILING_4.value, cls.FILING_5.value,
                       cls.FILING_SC13D.value, cls.FILING_SC13G.value)
    
    @classmethod
    def from_string(cls, form_str: str) -> 'CoreFilingType':
        """
        Safely convert a string to CoreFilingType enum.
        
        Args:
            form_str: The form type as a string (e.g., '10-K', '10-Q')
            
        Returns:
            CoreFilingType enum member
            
        Raises:
            ValueError: If the form type is not in the core filing types
        """
        # Normalize the input
        form_str = form_str.strip()
        
        # Try exact match first
        for filing_type in cls:
            if filing_type.value == form_str:
                return filing_type
        
        # Try case-insensitive match
        for filing_type in cls:
            if filing_type.value.lower() == form_str.lower():
                return filing_type
                
        raise ValueError(
            f"Form type '{form_str}' is not in CoreFilingType. "
            f"If this is a valid SEC form, it may be in enums.edgar.reference.filing_type. "
            f"Available core types: {[ft.value for ft in cls]}"
        )
    
    @classmethod
    def try_from_string(cls, form_str: str) -> 'CoreFilingType | None':
        """
        Safely convert a string to CoreFilingType enum, returning None if not found.
        
        Args:
            form_str: The form type as a string
            
        Returns:
            CoreFilingType enum member or None if not found
        """
        try:
            return cls.from_string(form_str)
        except ValueError:
            return None
