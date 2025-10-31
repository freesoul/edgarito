"""
6-K Data Combiner - Merges SEC CompanyFacts API with 6-K filing data.

For foreign companies filing 20-F (annual reports), quarterly data is NOT available
in the CompanyFacts API because the SEC only indexes XBRL data from 10-K/10-Q/20-F filings.
Foreign companies file 6-K for quarterly earnings, which contain HTML press releases
with structured financial tables.

This combiner:
1. Takes CompanyFacts API data (annual from XBRL)
2. Takes parsed 6-K quarterly data (from HTML press releases)
3. Merges them into a unified interface

Priority: 6-K parsed data > CompanyFacts API data

The combiner does NOT download or parse - it only combines already-extracted data.
"""

from typing import Optional, List, Dict, Any
from decimal import Decimal

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts


class QuarterlyDataPoint:
    """Represents a single quarterly financial data point."""
    
    def __init__(
        self,
        filing_date: str,
        period_end_date: str,
        fiscal_year: int,
        fiscal_period: str,
        accession_number: str,
        metric_name: str,
        value: Decimal,
        currency: str = "EUR"
    ):
        self.filing_date = filing_date
        self.period_end_date = period_end_date
        self.fiscal_year = fiscal_year
        self.fiscal_period = fiscal_period  # Q1, Q2, Q3, Q4
        self.accession_number = accession_number
        self.metric_name = metric_name
        self.value = value
        self.currency = currency
    
    def __repr__(self):
        return f"<DataPoint {self.metric_name} {self.fiscal_period} {self.fiscal_year}: {self.currency}{self.value:,}M>"


class SixKCombiner:
    """
    Unified interface for merging financial data from multiple sources.
    
    Combines:
    - SEC CompanyFacts API (annual data, structured XBRL)
    - 6-K parsed quarterly data (from HTML press releases)
    
    Usage:
        combiner = SixKCombiner()
        
        # Add data from sources
        combiner.add_company_facts(facts)
        combiner.add_6k_quarterly_data(quarterly_data_points)
        
        # Get merged data
        quarterly_revenue = combiner.get_quarterly_metric("net_revenues")
    """
    
    def __init__(self):
        """Initialize the 6-K combiner with empty data stores."""
        # Data stores
        self.company_facts: Optional[CompanyFacts] = None
        self.quarterly_6k_data: Dict[str, List[QuarterlyDataPoint]] = {}
    
    def add_company_facts(self, facts: CompanyFacts):
        """
        Add CompanyFacts data from SEC API.
        
        Args:
            facts: CompanyFacts object containing annual XBRL data
        """
        self.company_facts = facts
    
    def add_6k_quarterly_data(self, data_points: List[QuarterlyDataPoint]):
        """
        Add quarterly data points parsed from 6-K filings.
        
        Args:
            data_points: List of QuarterlyDataPoint objects
        """
        for dp in data_points:
            if dp.metric_name not in self.quarterly_6k_data:
                self.quarterly_6k_data[dp.metric_name] = []
            self.quarterly_6k_data[dp.metric_name].append(dp)
        
        # Sort all data points by period
        for metric_name in self.quarterly_6k_data:
            self.quarterly_6k_data[metric_name].sort(
                key=lambda dp: (dp.fiscal_year, dp.fiscal_period),
                reverse=True
            )
    
    def get_quarterly_metric(
        self,
        metric_name: str,
        limit: Optional[int] = None
    ) -> List[QuarterlyDataPoint]:
        """
        Get quarterly data for a specific metric.
        
        Args:
            metric_name: Metric name (e.g., "net_revenues", "ebit", "net_profit")
            limit: Optional limit on number of data points to return (most recent)
        
        Returns:
            List of QuarterlyDataPoint objects, sorted by period (most recent first)
        """
        data = self.quarterly_6k_data.get(metric_name, [])
        
        if limit:
            return data[:limit]
        
        return data
    
    def get_available_metrics(self) -> List[str]:
        """Get list of metric names available in 6-K quarterly data."""
        return list(self.quarterly_6k_data.keys())
    
    def has_quarterly_data(self) -> bool:
        """Check if any quarterly data from 6-K filings is available."""
        return len(self.quarterly_6k_data) > 0
