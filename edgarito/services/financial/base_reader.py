"""
Base class for financial statement readers.

All readers inherit from this to ensure consistent interface and access to company facts data.

CONCEPT NAMING:
===============
This library uses US-GAAP (Generally Accepted Accounting Principles) concept names.
All concept names are official US-GAAP taxonomy terms from the SEC EDGAR system.

Examples:
- Revenue: "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"
- Assets: "Assets", "AssetsCurrent"
- Net Income: "NetIncomeLoss"

IFRS (International Financial Reporting Standards) is not currently supported.
"""
import json
import logging
from typing import Optional
from datetime import datetime

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts, Measurement
from edgarito.schemas.reader.measurements import UnivariateMeasurements
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.edgar.core_filing_type import CoreFilingType
from edgarito.enums.granularity import Granularity

logger = logging.getLogger(__name__)


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
        Extract a single US-GAAP concept as a time series.
        
        Args:
            concept: US-GAAP concept name (e.g., "Assets", "NetIncomeLoss")
            granularity: ANNUAL or QUARTERLY
            filing_types: List of filing types (defaults to 10-K for annual, 10-K/10-Q for quarterly)
            convert_fy_to_q4: Whether to calculate Q4 from FY data (True for income/cash flow, False for balance sheet)
        
        Returns:
            UnivariateMeasurements with the time series data
        
        Note:
            For concepts that may have multiple names (like Revenue), use _get_concept_with_fallbacks instead.
            That method tries multiple concept names and merges the results intelligently.
        """
        self._require_loaded()
        
        # Ensure company has US-GAAP data
        if not self._data.facts.us_gaap:
            raise ValueError("Company data does not have US-GAAP facts. Only US-GAAP is currently supported.")
        
        facts_dict = self._data.facts.us_gaap
        
        if concept not in facts_dict:
            raise ValueError(f"Concept '{concept}' not found in US-GAAP company data")
        
        # Default filing types based on granularity
        if filing_types is None:
            if granularity == Granularity.ANNUAL:
                filing_types = [CoreFilingType.FILING_10K]
            elif granularity == Granularity.QUARTERLY:
                filing_types = [CoreFilingType.FILING_10K, CoreFilingType.FILING_10Q]
            else:
                raise ValueError(f"Unsupported granularity: {granularity}")
        
        # Get USD measurements
        all_measurements = facts_dict[concept].units.USD
        
        if all_measurements is None:
            raise ValueError(f"No USD measurements found for concept '{concept}'")
        
        filtered_measurements = []
        for filing_type in filing_types:
            filtered = self._filter_measurements(all_measurements, filing_type)
            logger.debug(f"Filtered {len(filtered)} measurements for filing_type={filing_type.value} (concept={concept})")
            filtered_measurements.extend(filtered)
        
        # Deduplicate at measurement level BEFORE creating UnivariateMeasurements
        # This handles:
        # - Multiple filings of same period (amendments, prior year comparatives in 10-K)
        # - YTD cumulative vs individual quarters (prefers shorter duration)
        # - Prefers most recent filing when durations are equal
        filtered_measurements = self._deduplicate_measurements(filtered_measurements)
        
        # Create univariate measurements
        univariate = UnivariateMeasurements.from_measurements(
            concept=concept,
            granularity=granularity,
            measurements=filtered_measurements
        )
        univariate.sort()
        
        # Convert YTD cumulative to individual quarters (for Q2 and Q3)
        # This must happen BEFORE FY→Q4 conversion
        # NOW ROBUST: Uses duration_days from MeasurementPeriod to detect YTD vs individual
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
                logger.debug(f"Skipping measurement without calendar_year: end={measurement.end}, start={measurement.start}, form={measurement.form}")
                continue
            
            # Filter out measurements with duration mismatch for their period type
            # Some companies report periods with wrong duration or wrong FP label
            # 
            # Expected durations:
            # - FY: 350-380 days (full year, allowing for leap years and 52/53 week years)
            # - Q1/Q2/Q3: 80-100 days (individual quarter)
            # - Balance sheet (no start date): always valid
            #
            # Examples of bad data:
            # - TSLA: 90-day period labeled as "FP: Year" (should be Q1)
            # - AMZN: 364-day period labeled as "Q1" (TTM from prior year, not current Q1)
            
            if measurement.start is not None:
                duration_days = (measurement.end - measurement.start).days
                
                if measurement.fp.value == 'FY':
                    # Full year should be roughly 350-380 days
                    if duration_days < 350 or duration_days > 380:
                        logger.debug(
                            f"Skipping FY measurement with invalid duration {duration_days} days: "
                            f"end={measurement.end}, val=${measurement.val:,.0f}"
                        )
                        continue
                
                elif measurement.fp.value in ['Q1', 'Q2', 'Q3']:
                    # Individual quarters should be 80-100 days (accounting for month variations)
                    # We allow YTD periods (180-280 days) to pass here - they'll be converted later
                    # But we reject TTM periods (350+ days) mislabeled as quarters
                    if duration_days < 80 or duration_days > 350:
                        logger.debug(
                            f"Skipping {measurement.fp.value} measurement with invalid duration {duration_days} days: "
                            f"end={measurement.end}, val=${measurement.val:,.0f}"
                        )
                        continue
            
            # Note: We don't filter by frame here because YTD cumulative values
            # (Q2, Q3 for cash flow/income) don't have a frame, but we need them
            # for YTD-to-quarterly conversion
            
            filtered.append(measurement)
        
        return filtered
    
    def _deduplicate_measurements(self, measurements: list[Measurement]) -> list[Measurement]:
        """
        Deduplicate measurements for the same fiscal period, handling multiple data quality issues.
        
        This method is CRITICAL for handling real-world SEC EDGAR data quality issues:
        
        1. **YTD vs Individual Quarters**: Companies report both cumulative and individual values
           - Individual quarter: Q2 with 90 days (Apr-Jun only)
           - YTD cumulative: Q2 with 181 days (Jan-Jun cumulative)
           → Prefers individual quarters (shorter duration) for accurate Q4 calculation
        
        2. **Repeated Filings**: Same period appears in multiple filings
           - 10-Q filed in original year (e.g., Q1 2023 filed Feb 2023)
           - 10-Q re-filed next year with amendments (Q1 2023 re-filed Feb 2024)
           → Prefers most recent filing (latest filed date) to get corrected/restated data
        
        3. **Prior Year Comparatives**: 10-K includes multiple fiscal years
           - 2025 10-K includes FY2025 AND FY2024 data
           - 2024 10-K includes FY2024 AND FY2023 data
           → When same period in multiple 10-Ks, picks most recent filing
        
        4. **Amendments**: Companies file amendments (10-K/A, 10-Q/A) with corrected data
           - Original filing with incorrect values
           - Amended filing with corrected values
           → Prefers most recent filing to get corrections
        
        Deduplication Strategy:
        - Groups by (fiscal_year, fiscal_period) - the logical period identifier
        - Sorts by (duration ASC, filed_date DESC):
          * Shortest duration first → Individual quarters over YTD
          * Most recent filing second → Latest corrections/amendments/restatements
        
        CRITICAL: Uses same fiscal year derivation logic as measurements.py from_measurements()
        to ensure consistency across the pipeline.
        """
        # Detect fiscal year end month from FY periods
        fy_end_month = None
        for m in measurements:
            if m.fp and m.fp.value == 'FY' and m.start:
                # FY period - the end month is the fiscal year end
                fy_end_month = m.end.month
                break
        
        if fy_end_month is None:
            # Fallback: Try to infer from the data
            # Most companies have fiscal year end in Dec (month 12)
            # Some have Jan (1), Feb (2), etc.
            # Without FY data, we default to calendar year (Dec)
            fy_end_month = 12
        
        def get_fiscal_year(m: Measurement) -> int:
            """
            Derive fiscal year from end date (matches measurements.py logic).
            
            CRITICAL: This MUST match the logic in measurements.py from_measurements()
            to avoid duplicates slipping through deduplication.
            """
            if fy_end_month <= 2:
                # Jan/Feb fiscal year end
                if m.end.month <= 2:
                    return m.end.year
                else:
                    return m.end.year + 1
            else:
                # Normal fiscal year (fiscal year = end date year)
                return m.end.year
        
        # Group by period key (derived fiscal year, fiscal period)
        period_groups = {}
        for m in measurements:
            fy = get_fiscal_year(m)
            key = (fy, m.fp)
            if key not in period_groups:
                period_groups[key] = []
            period_groups[key].append(m)
        
        # For each period, pick the best measurement
        deduplicated = []
        for key, group in period_groups.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Multiple measurements for same period - prefer shorter duration
                # Duration = end - start (or 0 if no start date)
                def get_duration(m: Measurement) -> int:
                    if m.start is None:
                        # Balance sheet items don't have start, treat as zero duration
                        return 0
                    return (m.end - m.start).days
                
                # Sort by duration (shortest first), then by filed date (newest first)
                sorted_group = sorted(group, key=lambda m: (get_duration(m), -m.filed.toordinal()))
                best = sorted_group[0]
                deduplicated.append(best)
                
                if len(group) > 1:
                    logger.debug(
                        f"Deduplicated {len(group)} measurements for FY:{key[0]} {key[1].value if key[1] else '?'}: "
                        f"chose duration={get_duration(best)} days, value=${best.val:,.0f}"
                    )
        
        return deduplicated
    
    def _deduplicate_periods(self, univariate: UnivariateMeasurements):
        """
        Remove duplicate periods at the UnivariateMeasurements level (safety net).
        
        This should rarely trigger since _deduplicate_measurements already handles duplicates
        at the Measurement level. However, this acts as a safety check in case:
        - The fiscal year derivation logic differs between methods
        - New code paths bypass _deduplicate_measurements
        
        When duplicates exist, keeps the first occurrence (which is the result from
        _deduplicate_measurements after sorting).
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
        
        ROBUST: Now uses duration_days to detect YTD vs individual:
        - YTD Q2: ~180 days (Jan-Jun)
        - Individual Q2: ~90 days (Apr-Jun)
        """
        # Build a dictionary of (year, fp) -> (index, value, period)
        period_map = {}
        for i, (value, period) in enumerate(zip(univariate.values, univariate.periods)):
            key = (period.year, period.fp)
            period_map[key] = (i, value, period)
        
        converted_values = list(univariate.values)
        
        for i, period in enumerate(univariate.periods):
            year = period.year
            
            # Skip if no duration info (balance sheet items)
            if period.duration_days is None or period.duration_days == 0:
                continue
            
            # Skip if already individual quarter (duration < 120 days)
            # Individual quarters are typically 89-92 days
            # YTD are 180-272 days
            if period.duration_days < 120:
                continue
            
            # Convert Q2 YTD to individual Q2
            if period.fp == FiscalPeriod.Q2:
                q1_key = (year, FiscalPeriod.Q1)
                if q1_key in period_map:
                    q1_value = period_map[q1_key][1]
                    q2_ytd = univariate.values[i]
                    converted_values[i] = q2_ytd - q1_value
                    logger.debug(
                        f"YTD→Q: FY{year} Q2: {q2_ytd:,.0f} - {q1_value:,.0f} = {converted_values[i]:,.0f} (duration: {period.duration_days} days)"
                    )
            
            # Convert Q3 YTD to individual Q3
            elif period.fp == FiscalPeriod.Q3:
                q2_key = (year, FiscalPeriod.Q2)
                if q2_key in period_map:
                    # Get Q2 value from converted_values (might have been converted already)
                    q2_idx = period_map[q2_key][0]
                    q2_ytd_or_individual = converted_values[q2_idx]
                    q3_ytd = univariate.values[i]
                    
                    # If Q2 was also YTD (180 days), use it as Q2_ytd
                    # If Q2 was individual (90 days), we need Q1+Q2
                    q2_period = period_map[q2_key][2]
                    if q2_period.duration_days and q2_period.duration_days < 120:
                        # Q2 is individual, need to add Q1
                        q1_key = (year, FiscalPeriod.Q1)
                        if q1_key in period_map:
                            q1_value = period_map[q1_key][1]
                            q2_ytd_value = q1_value + q2_ytd_or_individual
                            converted_values[i] = q3_ytd - q2_ytd_value
                        else:
                            # Can't convert without Q1
                            continue
                    else:
                        # Q2 is YTD, use it directly
                        converted_values[i] = q3_ytd - q2_ytd_or_individual
                    
                    logger.debug(
                        f"YTD→Q: FY{year} Q3: {q3_ytd:,.0f} - ... = {converted_values[i]:,.0f} (duration: {period.duration_days} days)"
                    )
        
        univariate.values = converted_values
    
    def _convert_fy_to_q4(self, univariate: UnivariateMeasurements):
        """
        Convert full-year (FY) periods to Q4 by calculating: Q4 = FY - (Q1 + Q2 + Q3).
        
        Why needed: Companies report Q1, Q2, Q3, and FY, but NOT Q4 separately.
        
        Example (NVDA fiscal 2025):
        - Q1 = $26,044M, Q2 = $30,040M, Q3 = $35,082M, FY = $130,497M
        - Q4 = $130,497M - ($26,044M + $30,040M + $35,082M) = $39,331M
        
        Validation: Skips conversion only if mathematically impossible:
        - Q4 < 0 (negative quarter when FY is positive)
        - Q4 > FY (one quarter exceeding full year)
        
        These cases indicate Q1/Q2/Q3 are from different accounting standard than FY.
        
        Only converts when all three quarters are available for that fiscal year.
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
                    quarters_sum = q1_value + q2_value + q3_value
                    
                    # Sanity check: Only skip if mathematically impossible
                    # The logic must handle both positive and negative values:
                    # 
                    # For POSITIVE FY (profit):
                    #   - Q4 < 0: Can't have negative quarter when FY is positive
                    #   - Q4 > FY: One quarter can't exceed full year
                    # 
                    # For NEGATIVE FY (loss):
                    #   - Q4 > 0 is VALID: e.g., FY=-$2,722M, Q1+Q2+Q3=-$3,000M → Q4=$278M
                    #   - Q4 < FY: One quarter loss can't exceed full year loss
                    # 
                    # Robust check using absolute values:
                    #   Skip if |Q4| > |FY| (one quarter magnitude exceeds full year magnitude)
                    if abs(q4_value) > abs(fy_value):
                        logger.warning(
                            f"Skipping FY→Q4 conversion for year {period.year}: "
                            f"Q4 would be ${q4_value:,.0f} (FY=${fy_value:,.0f}, Q1+Q2+Q3=${quarters_sum:,.0f}). "
                            f"This likely indicates Q1/Q2/Q3 are from a different accounting standard than FY."
                        )
                        # Leave as FY, will be filtered out
                        continue
                    
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
    
    def _get_concept_with_fallbacks(
        self,
        concepts: list[str],
        granularity: Granularity,
        filing_types: Optional[list[CoreFilingType]] = None,
        convert_fy_to_q4: bool = True
    ) -> UnivariateMeasurements:
        """
        Extract a concept trying multiple US-GAAP names, intelligently merging results.
        
        Use this when a concept may be reported under different names across time periods.
        
        Example: Revenue can be reported as:
        - "RevenueFromContractWithCustomerExcludingAssessedTax" (ASC 606 standard, newer)
        - "Revenues" (traditional, older filings)
        - "SalesRevenueNet" (some companies use this)
        
        How it works:
        1. Each concept is processed separately to avoid mixing YTD data
        2. For each fiscal year, selects ONE concept (the one with most quarters)
        3. Ensures Q1/Q2/Q3/FY all come from the same concept for accurate Q4 calculation
        
        Args:
            concepts: US-GAAP concept names in order of preference (newest first)
            granularity: ANNUAL or QUARTERLY
            filing_types: Filing types to include (defaults based on granularity)
            convert_fy_to_q4: Whether to calculate Q4 from FY data
        
        Returns:
            UnivariateMeasurements with merged data from available concepts
        
        Raises:
            ValueError: If none of the concepts are found with USD measurements
        """
        self._require_loaded()
        
        # Ensure company has US-GAAP data
        if not self._data.facts.us_gaap:
            raise ValueError("Company data does not have US-GAAP facts. Only US-GAAP is currently supported.")
        
        facts_dict = self._data.facts.us_gaap
        
        # Default filing types based on granularity
        if filing_types is None:
            if granularity == Granularity.ANNUAL:
                filing_types = [CoreFilingType.FILING_10K]
            elif granularity == Granularity.QUARTERLY:
                filing_types = [CoreFilingType.FILING_10K, CoreFilingType.FILING_10Q]
            else:
                raise ValueError(f"Unsupported granularity: {granularity}")
        
        # Process each concept separately to avoid YTD conversion issues
        all_univariates = []
        found_concepts = []
        
        for concept in concepts:
            if concept not in facts_dict:
                continue
            
            fact_data = facts_dict[concept].units.USD
            if not fact_data:
                continue
            
            found_concepts.append(concept)
            logger.debug(f"Found {len(fact_data)} measurements for concept '{concept}'")
            
            # Filter measurements by filing type
            filtered_measurements = []
            for filing_type in filing_types:
                filtered = self._filter_measurements(fact_data, filing_type)
                filtered_measurements.extend(filtered)
            
            if not filtered_measurements:
                continue
            
            # Deduplicate at measurement level (prefer shorter duration = individual quarters over YTD)
            filtered_measurements = self._deduplicate_measurements(filtered_measurements)
            
            # Create univariate measurements for this concept
            univariate = UnivariateMeasurements.from_measurements(
                concept=concept,
                granularity=granularity,
                measurements=filtered_measurements
            )
            univariate.sort()
            
            # No need for separate deduplicate_periods since we did it at measurement level
            # Note: We DON'T convert FY to Q4 here - we do it after merging all concepts
            # to ensure Q1+Q2+Q3+FY come from the same concept
            
            all_univariates.append(univariate)
        
        if not all_univariates:
            raise ValueError(
                f"None of the concepts {concepts} found with USD measurements in US-GAAP data"
            )
        
        logger.info(
            f"Using concepts {found_concepts} with data from {sum(len(u.values) for u in all_univariates)} periods"
        )
        
        # Merge all univariates
        # For quarterly data with FY→Q4 conversion, ensure each fiscal year uses ONE concept only
        # This prevents mixing Q1/Q2/Q3 from new standard with FY from old standard
        merged_values = []
        merged_periods = []
        seen_periods = set()
        
        if granularity == Granularity.QUARTERLY and convert_fy_to_q4:
            # Strategy: For each fiscal year, use only the concept that has BOTH quarters AND annual
            # This ensures Q4 calculation (FY - Q1 - Q2 - Q3) uses consistent data
            logger.debug(f"Applying per-FY concept selection for Q4 calculation")
            
            # First pass: catalog what each concept has for each FY
            fy_concept_data = {}  # fy -> {concept_idx: {'quarters': set(), 'has_fy': bool}}
            for concept_idx, univariate in enumerate(all_univariates):
                for value, period in zip(univariate.values, univariate.periods):
                    fy = period.year
                    if fy not in fy_concept_data:
                        fy_concept_data[fy] = {}
                    if concept_idx not in fy_concept_data[fy]:
                        fy_concept_data[fy][concept_idx] = {'quarters': set(), 'has_fy': False}
                    
                    if period.fp in [FiscalPeriod.Q1, FiscalPeriod.Q2, FiscalPeriod.Q3, FiscalPeriod.Q4]:
                        fy_concept_data[fy][concept_idx]['quarters'].add(period.fp)
                    elif period.fp == FiscalPeriod.Year:
                        fy_concept_data[fy][concept_idx]['has_fy'] = True
            
            # Second pass: for each FY with FY data, pick which FY to use for Q4 calculation
            # Prefer FY from concept that has most quarters (same source = consistent)
            fy_best_concept_for_q4 = {}  # fy -> concept_idx (only for FYs that will be converted to Q4)
            for fy, concepts in fy_concept_data.items():
                # Find concepts that have FY data
                concepts_with_fy = [idx for idx, data in concepts.items() if data['has_fy']]
                
                if concepts_with_fy:
                    # Pick concept with most quarters that also has FY
                    # This ensures Q4 = FY - (Q1+Q2+Q3) uses consistent data
                    best_idx = max(concepts_with_fy, key=lambda idx: len(concepts[idx]['quarters']))
                    fy_best_concept_for_q4[fy] = best_idx
                    concept_name = found_concepts[best_idx]
                    quarters_str = ','.join([fp.value for fp in sorted(concepts[best_idx]['quarters'])])
                    logger.debug(f"FY {fy}: Will use FY from concept {best_idx} ({concept_name}) for Q4 calculation - Quarters:{quarters_str}")
            
            # Third pass: merge ALL quarters from all concepts
            # But only include FY if it's from the selected concept (to ensure consistent Q4 calculation)
            # When same period exists in multiple concepts, pick from the concept with most data
            # (likely the newer/primary accounting standard)
            
            # First, collect all periods with their source concept
            period_candidates = {}  # period_key -> [(concept_idx, value, period), ...]
            for concept_idx, univariate in enumerate(all_univariates):
                for value, period in zip(univariate.values, univariate.periods):
                    fy = period.year
                    period_key = (period.year, period.fp)
                    
                    if period_key not in period_candidates:
                        period_candidates[period_key] = []
                    period_candidates[period_key].append((concept_idx, value, period))
            
            # Then, pick the best candidate for each period
            for period_key, candidates in period_candidates.items():
                year, fp = period_key
                
                # For quarterly periods (Q1/Q2/Q3/Q4): pick from concept with most total quarters
                if fp in [FiscalPeriod.Q1, FiscalPeriod.Q2, FiscalPeriod.Q3, FiscalPeriod.Q4]:
                    if len(candidates) == 1:
                        concept_idx, value, period = candidates[0]
                    else:
                        # Multiple concepts have this quarter - pick from concept with most data
                        # This favors the newer/primary accounting standard
                        best_candidate = max(
                            candidates,
                            key=lambda c: len(all_univariates[c[0]].values)
                        )
                        concept_idx, value, period = best_candidate
                        logger.debug(
                            f"Period {year} {fp.value}: Multiple concepts available, "
                            f"chose concept {concept_idx} ({found_concepts[concept_idx]}) "
                            f"with {len(all_univariates[concept_idx].values)} total periods"
                        )
                    
                    seen_periods.add(period_key)
                    merged_values.append(value)
                    merged_periods.append(period)
                
                # For FY periods: only include if from selected concept for Q4 calculation
                elif fp == FiscalPeriod.Year:
                    if year in fy_best_concept_for_q4:
                        # Find the candidate from the selected concept
                        for concept_idx, value, period in candidates:
                            if concept_idx == fy_best_concept_for_q4[year]:
                                seen_periods.add(period_key)
                                merged_values.append(value)
                                merged_periods.append(period)
                                break  # Found the right concept, stop looking
        else:
            # For annual data or when not converting FY to Q4, simple merge
            for univariate in all_univariates:
                for value, period in zip(univariate.values, univariate.periods):
                    period_key = (period.year, period.fp)
                    if period_key not in seen_periods:
                        seen_periods.add(period_key)
                        merged_values.append(value)
                        merged_periods.append(period)
        
        # Create final merged result
        concept_name = " or ".join(found_concepts)
        result = UnivariateMeasurements(
            concept=concept_name,
            granularity=granularity,
            values=merged_values,
            periods=merged_periods
        )
        result.sort()
        
        # Now convert FY to Q4 on the merged result
        if granularity == Granularity.QUARTERLY and convert_fy_to_q4:
            self._convert_fy_to_q4(result)
        
        return result
