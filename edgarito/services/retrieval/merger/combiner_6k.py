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

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from decimal import Decimal
import datetime

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts

if TYPE_CHECKING:
    from edgarito.enums.edgar.period import FiscalPeriod
    from edgarito.schemas.edgar_responses.company_facts import Measurement


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
    
    This combiner injects 6-K data directly into CompanyFacts as synthetic Measurement
    objects, making the integration transparent to downstream code.
    
    Usage:
        combiner = SixKCombiner(facts)
        combiner.add_6k_quarterly_data(quarterly_data_points)
        # facts now contains merged quarterly data from 6-K filings
    """
    
    def __init__(self, facts: CompanyFacts):
        """Initialize the 6-K combiner with CompanyFacts to merge into.
        
        Args:
            facts: CompanyFacts object to inject 6-K data into
        """
        self.company_facts = facts
        self.quarterly_6k_data: Dict[str, List[QuarterlyDataPoint]] = {}
    
    def add_6k_quarterly_data(self, data_points: List[QuarterlyDataPoint]):
        """
        Add quarterly data points parsed from 6-K filings and merge into CompanyFacts.
        
        Creates synthetic Measurement objects and injects them into the CompanyFacts
        structure, making the 6-K data transparent to downstream code.
        
        Args:
            data_points: List of QuarterlyDataPoint objects
        """
        from edgarito.schemas.edgar_responses.company_facts import Measurement, Fact, FactUnits
        from edgarito.enums.edgar.period import FiscalPeriod
        import datetime
        
        for dp in data_points:
            # Store for tracking
            if dp.metric_name not in self.quarterly_6k_data:
                self.quarterly_6k_data[dp.metric_name] = []
            self.quarterly_6k_data[dp.metric_name].append(dp)
            
            # Map metric name to fact name
            fact_name = self._map_metric_to_fact_name(dp.metric_name)
            if not fact_name:
                continue
            
            # Parse fiscal period
            fiscal_period_enum = self._parse_fiscal_period(dp.fiscal_period)
            if not fiscal_period_enum:
                continue
            
            # Parse dates
            filing_date = datetime.date.fromisoformat(dp.filing_date) if isinstance(dp.filing_date, str) else dp.filing_date
            period_end_date = datetime.date.fromisoformat(dp.period_end_date) if isinstance(dp.period_end_date, str) else dp.period_end_date
            
            # Convert value (values are in millions, need actual value)
            value_actual = int(dp.value * 1_000_000)
            
            # Create frame string to indicate this is an individual quarter (not YTD cumulative)
            # Format: CY{year}{Q1-Q4}I where I = instant/individual quarter
            frame_str = f"CY{dp.fiscal_year}{fiscal_period_enum.value}I"
            
            # Calculate start date (beginning of quarter) - approximate as 3 months before end
            # This is approximate but good enough for filtering purposes
            start_date = period_end_date - datetime.timedelta(days=90)
            
            # Create synthetic Measurement
            measurement = Measurement(
                end=period_end_date,
                val=value_actual,
                accn=dp.accession_number,
                fy=dp.fiscal_year,
                fp=fiscal_period_enum,
                form="6-K",
                filed=filing_date,
                frame=frame_str,
                start=start_date
            )
            
            # Inject into CompanyFacts (using currency from data point)
            self._inject_measurement(fact_name, measurement, dp.currency)
        
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
    
    def _map_metric_to_fact_name(self, metric_name: str) -> Optional[str]:
        """Map 6-K parsed metric names to IFRS/US-GAAP fact names."""
        mapping = {
            'net_revenues': 'Revenue',  # IFRS uses 'Revenue' singular
            'revenue': 'Revenue',
            'ebit': 'ProfitLossFromOperatingActivities',  # IFRS equivalent of EBIT
            'net_profit': 'ProfitLossAttributableToOwnersOfParent',  # IFRS net income concept
            'net_income': 'ProfitLossAttributableToOwnersOfParent',
            'ebitda': 'EBITDA',  # May not exist in standard IFRS, but we'll try
            'eps': 'BasicEarningsLossPerShare',  # IFRS name for EPS
        }
        return mapping.get(metric_name.lower())
    
    def _parse_fiscal_period(self, fiscal_period_str: str) -> Optional['FiscalPeriod']:
        """Parse fiscal period string (Q1, Q2, Q3, Q4) to FiscalPeriod enum."""
        from edgarito.enums.edgar.period import FiscalPeriod
        mapping = {
            'Q1': FiscalPeriod.Q1,
            'Q2': FiscalPeriod.Q2,
            'Q3': FiscalPeriod.Q3,
            'Q4': FiscalPeriod.Q4,
        }
        return mapping.get(fiscal_period_str.upper())
    
    def _inject_measurement(self, fact_name: str, measurement: 'Measurement', currency: str = "USD") -> None:
        """Inject a synthetic measurement into CompanyFacts structure."""
        from edgarito.schemas.edgar_responses.company_facts import Fact, FactUnits
        
        # Determine which taxonomy to use (prefer ifrs-full for foreign companies)
        if self.company_facts.facts.ifrs_full is not None:
            taxonomy = self.company_facts.facts.ifrs_full
        elif self.company_facts.facts.us_gaap is not None:
            taxonomy = self.company_facts.facts.us_gaap
        else:
            # Create us-gaap if neither exists
            self.company_facts.facts.us_gaap = {}
            taxonomy = self.company_facts.facts.us_gaap
        
        # Get or create the fact
        if fact_name not in taxonomy:
            taxonomy[fact_name] = Fact(
                label=fact_name,
                description=f"6-K parsed data for {fact_name}",
                units=FactUnits()
            )
        
        # Get the fact and ensure the currency list exists
        fact = taxonomy[fact_name]
        
        # Get or create the currency list (EUR, USD, etc.)
        currency_list = getattr(fact.units, currency, None)
        if currency_list is None:
            setattr(fact.units, currency, [])
            currency_list = getattr(fact.units, currency)
        
        # Add measurement (check for duplicates by accession number)
        # Handle both dict (API data) and Measurement objects (injected data)
        from datetime import date
        
        def get_accn(m):
            return m['accn'] if isinstance(m, dict) else m.accn
        
        def get_fp(m):
            return m['fp'] if isinstance(m, dict) else m.fp
        
        def get_end(m):
            end = m['end'] if isinstance(m, dict) else m.end
            # Convert string dates to date objects for consistent comparison
            if isinstance(end, str):
                return date.fromisoformat(end)
            return end
        
        if not any(get_accn(m) == measurement.accn and get_fp(m) == measurement.fp for m in currency_list):
            currency_list.append(measurement)
            # Sort by end date
            currency_list.sort(key=lambda m: get_end(m))
