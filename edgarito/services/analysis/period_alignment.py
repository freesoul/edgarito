"""
Period Alignment Utilities - Prevent misaligned time series comparisons.

This module provides validation and alignment checking for financial time series data
to prevent bugs where data from different time periods is incorrectly combined or compared.

Common issues this prevents:
1. Comparing old gross profit (2021) with new revenue (2025)
2. Calculating ratios when denominators/numerators are from different periods
3. Computing growth rates when time series have gaps or inconsistent periods
"""
import logging
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

from edgarito.schemas.reader.measurements import UnivariateMeasurements, MeasurementPeriod

logger = logging.getLogger(__name__)


class PeriodMismatchError(Exception):
    """Raised when time series periods are critically misaligned."""
    pass


class PeriodMismatchWarning(UserWarning):
    """Warning for non-critical period alignment issues."""
    pass


def check_series_freshness(
    series: UnivariateMeasurements,
    max_age_days: int = 365,
    context: str = ""
) -> bool:
    """
    Check if the most recent data point in a time series is fresh enough for analysis.
    
    Args:
        series: The time series to check
        max_age_days: Maximum age in days for the latest data point (default: 1 year)
        context: Description for logging (e.g., "GrossProfit for margin calculation")
    
    Returns:
        True if data is fresh, False if stale
    
    Raises:
        Warning if data is stale
    """
    if not series.periods or not series.values:
        logger.warning(f"Empty time series for {context or series.concept}")
        return False
    
    latest_period = series.periods[-1]
    
    # If we have an end_date, use it; otherwise, estimate from fiscal period
    if latest_period.end_date:
        age_days = (datetime.now().date() - latest_period.end_date).days
    else:
        # Estimate: assume Q4 ends at year-end, quarters are ~90 days apart
        # This is approximate but catches egregiously stale data
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Rough estimate of how old the latest period is
        year_diff = current_year - latest_period.year
        age_days = year_diff * 365
    
    if age_days > max_age_days:
        msg = f"Stale data for {context or series.concept}: latest period is {latest_period.year} {latest_period.fp.value if latest_period.fp else ''} ({age_days} days old, max: {max_age_days})"
        logger.warning(msg)
        return False
    
    return True


def check_period_overlap(
    series1: UnivariateMeasurements,
    series2: UnivariateMeasurements,
    min_overlap: int = 3,
    require_same_latest: bool = True,
    context: str = ""
) -> Tuple[bool, List[MeasurementPeriod]]:
    """
    Check if two time series have sufficient overlapping periods for comparison.
    
    Args:
        series1: First time series
        series2: Second time series
        min_overlap: Minimum number of overlapping periods required
        require_same_latest: If True, require both series to have the same latest period
        context: Description for logging (e.g., "revenue vs gross_profit")
    
    Returns:
        Tuple of (is_aligned, overlapping_periods)
        
    Raises:
        PeriodMismatchError if alignment is critically bad (no overlap)
        Warning if alignment is suboptimal but usable
    """
    if not series1.periods or not series2.periods:
        raise PeriodMismatchError(f"One or both series are empty in {context}")
    
    # Convert to sets for intersection
    periods1_set = set(series1.periods)
    periods2_set = set(series2.periods)
    
    overlapping = sorted(list(periods1_set & periods2_set))
    
    # Critical error: no overlap at all
    if not overlapping:
        raise PeriodMismatchError(
            f"No overlapping periods between {series1.concept} and {series2.concept} in {context}. "
            f"{series1.concept}: {series1.periods[-1].year} {series1.periods[-1].fp.value if series1.periods[-1].fp else ''}, "
            f"{series2.concept}: {series2.periods[-1].year} {series2.periods[-1].fp.value if series2.periods[-1].fp else ''}"
        )
    
    # Check if we have enough overlap
    if len(overlapping) < min_overlap:
        logger.warning(
            f"Insufficient overlap between {series1.concept} and {series2.concept} in {context}: "
            f"only {len(overlapping)} periods overlap (need {min_overlap})"
        )
        return False, overlapping
    
    # Check if latest periods match
    if require_same_latest:
        latest1 = series1.periods[-1]
        latest2 = series2.periods[-1]
        
        if latest1 != latest2:
            logger.warning(
                f"Latest periods don't match in {context}: "
                f"{series1.concept} ends at {latest1.year} {latest1.fp.value if latest1.fp else ''}, "
                f"{series2.concept} ends at {latest2.year} {latest2.fp.value if latest2.fp else ''}"
            )
            return False, overlapping
    
    return True, overlapping


def align_series_for_ratio(
    numerator: UnivariateMeasurements,
    denominator: UnivariateMeasurements,
    context: str = "",
    require_recent_data: bool = True,
    max_age_days: int = 365
) -> Tuple[List[float], List[float], List[MeasurementPeriod]]:
    """
    Align two time series for ratio calculation (e.g., margin = profit / revenue).
    
    Returns only the periods where both series have data, properly aligned.
    
    Args:
        numerator: Numerator time series (e.g., gross profit)
        denominator: Denominator time series (e.g., revenue)
        context: Description for error messages
        require_recent_data: If True, reject stale data
        max_age_days: Maximum age for latest data point
    
    Returns:
        Tuple of (aligned_numerator_values, aligned_denominator_values, aligned_periods)
    
    Raises:
        PeriodMismatchError if series cannot be aligned properly
    """
    # Check freshness
    if require_recent_data:
        num_fresh = check_series_freshness(numerator, max_age_days, f"{context} numerator ({numerator.concept})")
        den_fresh = check_series_freshness(denominator, max_age_days, f"{context} denominator ({denominator.concept})")
        
        if not num_fresh or not den_fresh:
            raise PeriodMismatchError(
                f"Stale data in {context}: cannot calculate ratio with old data. "
                f"{numerator.concept} latest: {numerator.periods[-1].year if numerator.periods else 'N/A'}, "
                f"{denominator.concept} latest: {denominator.periods[-1].year if denominator.periods else 'N/A'}"
            )
    
    # Check period overlap
    is_aligned, overlapping = check_period_overlap(
        numerator, denominator,
        min_overlap=1,  # Need at least 1 period for ratio
        require_same_latest=True,
        context=context
    )
    
    if not is_aligned or not overlapping:
        raise PeriodMismatchError(f"Cannot align {numerator.concept} and {denominator.concept} for {context}")
    
    # Build aligned values
    # Create lookup maps for fast access
    num_map = {period: value for period, value in zip(numerator.periods, numerator.values)}
    den_map = {period: value for period, value in zip(denominator.periods, denominator.values)}
    
    aligned_num_values = []
    aligned_den_values = []
    aligned_periods = []
    
    for period in overlapping:
        if period in num_map and period in den_map:
            aligned_num_values.append(num_map[period])
            aligned_den_values.append(den_map[period])
            aligned_periods.append(period)
    
    if not aligned_periods:
        raise PeriodMismatchError(f"No aligned periods found for {context}")
    
    return aligned_num_values, aligned_den_values, aligned_periods


def align_series_for_growth(
    series: UnivariateMeasurements,
    lookback_periods: int = 1,
    context: str = ""
) -> Tuple[List[float], List[float], List[MeasurementPeriod], List[MeasurementPeriod]]:
    """
    Align a time series for growth rate calculation (comparing current vs prior periods).
    
    Args:
        series: Time series to analyze
        lookback_periods: Number of periods to look back (1 for QoQ, 4 for YoY quarterly)
        context: Description for error messages
    
    Returns:
        Tuple of (current_values, prior_values, current_periods, prior_periods)
    
    Raises:
        PeriodMismatchError if insufficient data for growth calculation
    """
    if len(series.values) < lookback_periods + 1:
        raise PeriodMismatchError(
            f"Insufficient data for growth calculation in {context}: "
            f"need at least {lookback_periods + 1} periods, have {len(series.values)}"
        )
    
    # Check for gaps in the time series
    for i in range(len(series.periods) - 1):
        expected_gap = 1 if series.granularity.name == "QUARTERLY" else 1  # Quarters or years
        actual_gap = series.periods[i + 1] - series.periods[i]
        
        if actual_gap != expected_gap and actual_gap != lookback_periods:
            logger.warning(
                f"Gap detected in {context} time series: "
                f"{series.periods[i].year} {series.periods[i].fp.value if series.periods[i].fp else ''} -> "
                f"{series.periods[i+1].year} {series.periods[i+1].fp.value if series.periods[i+1].fp else ''} "
                f"(gap: {actual_gap} periods)"
            )
    
    # Build aligned current and prior values
    current_values = series.values[lookback_periods:]
    prior_values = series.values[:-lookback_periods]
    current_periods = series.periods[lookback_periods:]
    prior_periods = series.periods[:-lookback_periods]
    
    return current_values, prior_values, current_periods, prior_periods


def validate_series_alignment(*series: UnivariateMeasurements, context: str = "", min_overlap: int = 3) -> bool:
    """
    Validate that multiple time series are properly aligned for analysis.
    
    Args:
        *series: Variable number of time series to check
        context: Description for logging
        min_overlap: Minimum overlapping periods required
    
    Returns:
        True if all series are aligned, False otherwise
    
    Raises:
        PeriodMismatchError if alignment is critically bad
    """
    if len(series) < 2:
        return True
    
    # Check all pairs
    for i in range(len(series) - 1):
        is_aligned, _ = check_period_overlap(
            series[i], series[i + 1],
            min_overlap=min_overlap,
            require_same_latest=True,
            context=f"{context} ({series[i].concept} vs {series[i+1].concept})"
        )
        if not is_aligned:
            return False
    
    return True
