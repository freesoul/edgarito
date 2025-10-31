"""
Base class for financial statement readers.

All readers inherit from this to ensure consistent interface and access to company facts data.
"""
import json
from typing import Optional

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts, Measurement
from edgarito.schemas.reader.measurements import UnivariateMeasurements
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.edgar.core_filing_type import CoreFilingType
from edgarito.enums.granularity import Granularity
from edgarito.services.financial.ifrs_mapping import get_ifrs_concept


class BaseStatementReader:
    """Base class for all financial statement readers"""
    
    def __init__(self, facts: Optional[CompanyFacts] = None):
        """
        Initialize the reader with company facts data.
        
        Args:
            facts: CompanyFacts object from SEC EDGAR REST API
        """
        self._data = facts
    
    def load_from_facts(self, facts: CompanyFacts):
        """Load data from a CompanyFacts object"""
        self._data = facts
    
    def load_from_json_file(self, path: str):
        """Load data from a JSON file containing CompanyFacts data"""
        with open(path, "r") as file:
            data_json = json.load(file)
        self._data = CompanyFacts(**data_json)
    
    def _require_loaded(self):
        """Ensure data has been loaded before accessing it"""
        if self._data is None:
            raise ValueError("No data loaded. Call load_from_facts() or load_from_json_file() first.")
    
    def _get_concept(
        self,
        concept: str,
        granularity: Granularity,
        filing_types: Optional[list[CoreFilingType]] = None,
        convert_fy_to_q4: bool = True
    ) -> UnivariateMeasurements:
        """
        Internal method to extract a GAAP concept as time series.
        
        Supports both US-GAAP and IFRS companies. For IFRS companies, automatically
        maps US-GAAP concept names to their IFRS equivalents.
        
        Args:
            concept: US-GAAP concept name (e.g., "Assets", "Revenue")
            granularity: ANNUAL or QUARTERLY
            filing_types: List of filing types to include (defaults to 10-K/20-F for annual, 10-K/10-Q/20-F for quarterly)
            convert_fy_to_q4: Whether to convert FY periods to Q4 for quarterly data (default True).
                             Set to False for stock/point-in-time data like balance sheet.
        
        Returns:
            UnivariateMeasurements with the time series data
        """
        self._require_loaded()
        
        # Determine if company uses US-GAAP or IFRS
        use_ifrs = False
        if not self._data.facts.us_gaap:
            if self._data.facts.ifrs_full:
                use_ifrs = True
                # Map US-GAAP concept to IFRS equivalent
                ifrs_concept = get_ifrs_concept(concept)
                if not ifrs_concept:
                    raise ValueError(f"Concept '{concept}' has no IFRS mapping")
                concept = ifrs_concept
            else:
                raise ValueError(f"Company data has neither US-GAAP nor IFRS standards")
        
        # Get the appropriate facts dictionary
        facts_dict = self._data.facts.ifrs_full if use_ifrs else self._data.facts.us_gaap
        
        if concept not in facts_dict:
            standard = "IFRS" if use_ifrs else "US-GAAP"
            raise ValueError(f"Concept '{concept}' not found in {standard} company data")
        
        # Default filing types based on granularity
        if filing_types is None:
            if granularity == Granularity.ANNUAL:
                # Support both US and foreign companies
                filing_types = [CoreFilingType.FILING_10K, CoreFilingType.FILING_20F]
            elif granularity == Granularity.QUARTERLY:
                filing_types = [CoreFilingType.FILING_10K, CoreFilingType.FILING_10Q, CoreFilingType.FILING_20F]
            else:
                raise ValueError(f"Unsupported granularity: {granularity}")
        
        # Get measurements - try USD first, then EUR for IFRS companies
        all_measurements = facts_dict[concept].units.USD
        if all_measurements is None and use_ifrs:
            # Try EUR for IFRS companies (using getattr since EUR is in extra fields)
            eur_data = getattr(facts_dict[concept].units, 'EUR', None)
            if eur_data is not None:
                # EUR data comes as list of dicts, need to convert to Measurement objects
                from edgarito.schemas.edgar_responses.company_facts import Measurement
                all_measurements = [Measurement(**m) if isinstance(m, dict) else m for m in eur_data]
        
        if all_measurements is None:
            currency = "USD/EUR" if use_ifrs else "USD"
            raise ValueError(f"No {currency} measurements found for concept '{concept}'")
        
        filtered_measurements = []
        for filing_type in filing_types:
            filtered = self._filter_measurements(all_measurements, filing_type)
            filtered_measurements.extend(filtered)
        
        # Create univariate measurements
        univariate = UnivariateMeasurements.from_measurements(
            concept=concept,
            granularity=granularity,
            measurements=filtered_measurements
        )
        univariate.sort()
        
        # Deduplicate periods (keep the most recent filing for each period)
        self._deduplicate_periods(univariate)
        
        # Convert YTD cumulative to individual quarters (for Q2 and Q3)
        # This must happen BEFORE FY→Q4 conversion
        if granularity == Granularity.QUARTERLY and convert_fy_to_q4:
            self._convert_ytd_to_quarterly(univariate)
        
        # Convert FY to Q4 for quarterly data (only for flow statements, not balance sheet)
        if granularity == Granularity.QUARTERLY and convert_fy_to_q4:
            self._convert_fy_to_q4(univariate)
        
        return univariate
    
    def _filter_measurements(
        self,
        measurements: list[Measurement],
        filing_type: CoreFilingType
    ) -> list[Measurement]:
        """Filter measurements by filing type and data quality"""
        filtered = []
        
        for measurement in measurements:
            # Match filing type
            if measurement.form != filing_type.value:
                continue
            
            # Ensure we have fiscal year and period (SEC API sometimes returns null)
            if measurement.fy is None or measurement.fp is None:
                continue
            
            # Ensure we have calendar year
            if measurement.calendar_year is None:
                continue
            
            # Note: We don't filter by frame here because YTD cumulative values
            # (Q2, Q3 for cash flow/income) don't have a frame, but we need them
            # for YTD-to-quarterly conversion
            
            filtered.append(measurement)
        
        return filtered
    
    def _deduplicate_periods(self, univariate: UnivariateMeasurements):
        """
        Remove duplicate periods, keeping only the first occurrence.
        
        After sorting, periods are in chronological order, so the first occurrence
        is the earliest data point for that period. In cases where there are
        multiple filings for the same period (amendments, etc.), we keep the first
        to maintain consistency.
        """
        seen_periods = set()
        deduplicated_values = []
        deduplicated_periods = []
        
        for value, period in zip(univariate.values, univariate.periods):
            period_key = (period.year, period.fp)
            if period_key not in seen_periods:
                seen_periods.add(period_key)
                deduplicated_values.append(value)
                deduplicated_periods.append(period)
        
        univariate.values = deduplicated_values
        univariate.periods = deduplicated_periods
    
    def _convert_ytd_to_quarterly(self, univariate: UnivariateMeasurements):
        """
        Convert year-to-date (cumulative) quarterly values to individual quarters.
        
        For cash flow and income statement, SEC reports cumulative values:
        - Q1: Q1 value
        - Q2: Q1+Q2 (YTD)
        - Q3: Q1+Q2+Q3 (YTD)
        
        This method converts to individual quarters:
        - Q1: stays as is
        - Q2: Q2_ytd - Q1
        - Q3: Q3_ytd - Q2_ytd
        """
        # Build a dictionary of (year, fp) -> (index, value)
        period_map = {}
        for i, (value, period) in enumerate(zip(univariate.values, univariate.periods)):
            key = (period.year, period.fp)
            period_map[key] = (i, value)
        
        converted_values = list(univariate.values)
        
        for i, period in enumerate(univariate.periods):
            year = period.year
            
            # Convert Q2 YTD to individual Q2
            if period.fp == FiscalPeriod.Q2:
                q1_key = (year, FiscalPeriod.Q1)
                if q1_key in period_map:
                    q1_value = period_map[q1_key][1]
                    q2_ytd = univariate.values[i]
                    converted_values[i] = q2_ytd - q1_value
            
            # Convert Q3 YTD to individual Q3
            elif period.fp == FiscalPeriod.Q3:
                q2_key = (year, FiscalPeriod.Q2)
                if q2_key in period_map:
                    # Q2 might still be YTD at this point, so use original values
                    q2_ytd = univariate.values[period_map[q2_key][0]]
                    q3_ytd = univariate.values[i]
                    converted_values[i] = q3_ytd - q2_ytd
        
        univariate.values = converted_values
    
    def _convert_fy_to_q4(self, univariate: UnivariateMeasurements):
        """
        Convert full-year (FY) values to Q4 values by subtracting Q1+Q2+Q3.
        Only converts FY periods where we have all three quarters available.
        """
        # Build a dictionary of (year, fp) -> (index, value)
        period_map = {}
        for i, (value, period) in enumerate(zip(univariate.values, univariate.periods)):
            key = (period.year, period.fp)
            period_map[key] = (i, value)
        
        # Convert FY to Q4 by subtraction (only when all quarters exist)
        converted_values = list(univariate.values)
        converted_periods = list(univariate.periods)
        
        for i, period in enumerate(univariate.periods):
            if period.fp == FiscalPeriod.Year:
                # Check if we have Q1, Q2, Q3 for this year
                q1_key = (period.year, FiscalPeriod.Q1)
                q2_key = (period.year, FiscalPeriod.Q2)
                q3_key = (period.year, FiscalPeriod.Q3)
                
                if q1_key in period_map and q2_key in period_map and q3_key in period_map:
                    # We have all quarters, compute Q4 = FY - (Q1 + Q2 + Q3)
                    q1_value = period_map[q1_key][1]
                    q2_value = period_map[q2_key][1]
                    q3_value = period_map[q3_key][1]
                    fy_value = converted_values[i]
                    
                    q4_value = fy_value - (q1_value + q2_value + q3_value)
                    converted_values[i] = q4_value
                    converted_periods[i].fp = FiscalPeriod.Q4
                # else: leave as FY (we'll filter it out below)
        
        # Remove FY periods that weren't converted (incomplete data)
        filtered_values = []
        filtered_periods = []
        for value, period in zip(converted_values, converted_periods):
            if period.fp != FiscalPeriod.Year:
                filtered_values.append(value)
                filtered_periods.append(period)
        
        univariate.values = filtered_values
        univariate.periods = filtered_periods
