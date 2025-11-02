from typing import List, Optional, Tuple, Iterator

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity

from edgarito.schemas.edgar_responses.company_facts import Measurement


class MeasurementPeriod:
    year: int
    fp: FiscalPeriod
    frame: Optional[str]

    def __init__(self, year: int, fp: FiscalPeriod, frame: Optional[str] = None):
        self.year = year
        self.fp = fp
        self.frame = frame

    def __eq__(self, other: "MeasurementPeriod"):
        return self.year == other.year and self.fp == other.fp

    def __hash__(self):
        """Make MeasurementPeriod hashable for use in sets and dicts"""
        return hash((self.year, self.fp))

    def __gt__(self, other: "MeasurementPeriod"):
        return self.year > other.year or (self.year == other.year and self.fp > other.fp)

    def __ge__(self, other: "MeasurementPeriod"):
        return self.year > other.year or (self.year == other.year and self.fp >= other.fp)

    def __lt__(self, other: "MeasurementPeriod"):
        return self.year < other.year or (self.year == other.year and self.fp < other.fp)

    def __le__(self, other: "MeasurementPeriod"):
        return self.year < other.year or (self.year == other.year and self.fp <= other.fp)

    def __sub__(self, other: "MeasurementPeriod") -> int:
        """gives the number of periods between self and other"""
        return (self.year - other.year) * 4 + (self.fp - other.fp)


class UnivariateMeasurements:
    concept: str
    granularity: Granularity
    periods: List[MeasurementPeriod]
    values: List[float]

    def __init__(self, concept: str, granularity: Granularity, values: List[float], periods: List[MeasurementPeriod], validate_no_gaps: bool = False):
        self.concept = concept
        self.granularity = granularity
        self.periods = periods
        self.values = values

        if validate_no_gaps:
            self._validate_no_missing_periods()

    def __str__(self):
        ret = f"{self.concept}:\n"
        for value, period in zip(self.values, self.periods):
            ret += f"\t{period.year} {period.fp}\t{value}\n"
        return ret

    def _validate_no_missing_periods(self):
        """
        Validates that there are no missing periods in the time series.
        Note: This is disabled by default because companies sometimes skip filing periods
        (e.g., missing a year or quarter), which is valid but would fail this check.
        """
        expected_num_quarters_per_row = 1 if self.granularity == Granularity.QUARTERLY else 4
        for i in range(1, len(self.periods)):
            if self.periods[i] - self.periods[i - 1] != expected_num_quarters_per_row:
                raise ValueError(f"Missing period between {self.periods[i - 1].__dict__} and {self.periods[i].__dict__}")

    @staticmethod
    def from_measurements(concept: str, granularity: Granularity, measurements: List[Measurement]) -> "UnivariateMeasurements":
        concept = concept
        values = []
        periods = []
        for measurement in measurements:
            values.append(measurement.val)
            periods.append(MeasurementPeriod(year=measurement.calendar_year, fp=measurement.fp, frame=measurement.frame))

        return UnivariateMeasurements(concept=concept, granularity=granularity, values=values, periods=periods)

    def sort(self):
        sorted_pairs = sorted(zip(self.values, self.periods), key=lambda x: x[1])
        if sorted_pairs:
            self.values, self.periods = map(list, zip(*sorted_pairs))
        else:
            self.values, self.periods = [], []

    def __iter__(self) -> Iterator[Tuple[float, MeasurementPeriod]]:
        for value, period in zip(self.values, self.periods):
            yield value, period

    def __getitem__(self, index) -> Tuple[float, MeasurementPeriod]:
        return self.values[index], self.periods[index]

    def __len__(self):
        return len(self.values)

    def __contains__(self, period: MeasurementPeriod):
        return period in self.periods

    def __sub__(self, other: "UnivariateMeasurements") -> "UnivariateMeasurements":
        """
        Returns a UnivariateMeasurements containing the difference between the measurements.
        """
        if len(self) != len(other):
            raise ValueError("Measurements must have the same length")
        return UnivariateMeasurements(
            concept=self.concept,
            granularity=self.granularity,
            periods=self.periods,
            values=[value - other_value for value, other_value in zip(self.values, other.values)],
        )

    def __add__(self, other: "UnivariateMeasurements") -> "UnivariateMeasurements":
        """
        Returns a UnivariateMeasurements containing the sum of the measurements.
        """
        if len(self) != len(other):
            raise ValueError("Measurements must have the same length")
        return UnivariateMeasurements(
            concept=self.concept,
            granularity=self.granularity,
            periods=self.periods,
            values=[value + other_value for value, other_value in zip(self.values, other.values)],
        )

    def __truediv__(self, other: "UnivariateMeasurements") -> "UnivariateMeasurements":
        """
        Returns a UnivariateMeasurements containing the ratio of the measurements.
        """
        if len(self) != len(other):
            raise ValueError("Measurements must have the same length")
        return UnivariateMeasurements(
            concept=f"{self.concept}/{other.concept}",
            granularity=self.granularity,
            periods=self.periods,
            values=[value / other_value if other_value != 0 else 0.0 for value, other_value in zip(self.values, other.values)],
        )

    def __mul__(self, other: "UnivariateMeasurements") -> "UnivariateMeasurements":
        """
        Returns a UnivariateMeasurements containing the product of the measurements.
        """
        if len(self) != len(other):
            raise ValueError("Measurements must have the same length")
        return UnivariateMeasurements(
            concept=f"{self.concept}*{other.concept}",
            granularity=self.granularity,
            periods=self.periods,
            values=[value * other_value for value, other_value in zip(self.values, other.values)],
        )

    @property
    def min_period(self) -> MeasurementPeriod:
        return self.periods[0]

    @property
    def max_period(self) -> MeasurementPeriod:
        return self.periods[-1]

    def limit_periods(self, min_period: Optional[MeasurementPeriod] = None, max_period: Optional[MeasurementPeriod] = None):
        """
        Returns a UnivariateMeasurements containing only the measurements within the specified period.
        """
        values: List[float] = []
        periods: List[MeasurementPeriod] = []
        for value, period in self:
            if (min_period is None or period >= min_period) and (max_period is None or period <= max_period):
                values.append(value)
                periods.append(period)
        return UnivariateMeasurements(concept=self.concept, granularity=self.granularity, values=values, periods=periods)

    def intersect(self, other: "UnivariateMeasurements") -> "UnivariateMeasurements":
        """
        Returns a UnivariateMeasurements containing only the periods that exist in both time series.
        This ensures proper alignment for arithmetic operations.
        """
        values: List[float] = []
        periods: List[MeasurementPeriod] = []
        
        # Create a set of other's periods for fast lookup
        other_periods_set = set(other.periods)
        
        # Keep only periods that exist in both
        for value, period in self:
            if period in other_periods_set:
                values.append(value)
                periods.append(period)
        
        return UnivariateMeasurements(concept=self.concept, granularity=self.granularity, values=values, periods=periods)

    def align(self, other: "UnivariateMeasurements", strategy: str = "intersection") -> Tuple["UnivariateMeasurements", "UnivariateMeasurements"]:
        """
        Align two time series to have matching periods using the specified strategy.
        
        Args:
            other: Another UnivariateMeasurements to align with
            strategy: Alignment strategy to use:
                - "intersection" (default): Keep only periods present in both series
                - "forward_fill": Use all periods from self, fill missing values from other with last known value
                - "union_fill": Use all unique periods from both, forward-fill missing values
        
        Returns:
            Tuple of two aligned UnivariateMeasurements (self_aligned, other_aligned)
        
        Examples:
            >>> revenue = UnivariateMeasurements(...)  # Has periods 2020, 2021, 2022, 2023
            >>> cogs = UnivariateMeasurements(...)     # Has periods 2020, 2022
            >>> 
            >>> # Intersection strategy (default) - only keeps common periods
            >>> rev_aligned, cogs_aligned = revenue.align(cogs)
            >>> # Result: Both have periods 2020, 2022
            >>> 
            >>> # Forward fill strategy - uses self's periods, fills other
            >>> rev_aligned, cogs_aligned = revenue.align(cogs, strategy="forward_fill")
            >>> # Result: Both have periods 2020, 2021, 2022, 2023
            >>> # COGS for 2021 = COGS from 2020, COGS for 2023 = COGS from 2022
        """
        if strategy == "intersection":
            # Keep only periods present in both
            return self.intersect(other), other.intersect(self)
        
        elif strategy == "forward_fill":
            # Use self's periods, forward-fill other's missing values
            other_dict = {p: v for p, v in zip(other.periods, other.values)}
            
            aligned_self_values = []
            aligned_other_values = []
            aligned_periods = []
            
            last_other_value = None
            for value, period in self:
                aligned_self_values.append(value)
                aligned_periods.append(period)
                
                if period in other_dict:
                    last_other_value = other_dict[period]
                    aligned_other_values.append(last_other_value)
                elif last_other_value is not None:
                    # Forward fill with last known value
                    aligned_other_values.append(last_other_value)
                else:
                    # No previous value available, skip this period
                    aligned_self_values.pop()
                    aligned_periods.pop()
            
            return (
                UnivariateMeasurements(self.concept, self.granularity, aligned_self_values, aligned_periods),
                UnivariateMeasurements(other.concept, other.granularity, aligned_other_values, aligned_periods)
            )
        
        elif strategy == "union_fill":
            # Use all unique periods from both, forward-fill missing values
            all_periods = sorted(set(self.periods + other.periods))
            
            self_dict = {p: v for p, v in zip(self.periods, self.values)}
            other_dict = {p: v for p, v in zip(other.periods, other.values)}
            
            aligned_self_values = []
            aligned_other_values = []
            aligned_periods = []
            
            last_self_value = None
            last_other_value = None
            
            for period in all_periods:
                has_self = period in self_dict
                has_other = period in other_dict
                
                if has_self:
                    last_self_value = self_dict[period]
                if has_other:
                    last_other_value = other_dict[period]
                
                # Only include period if both have values (current or forward-filled)
                if last_self_value is not None and last_other_value is not None:
                    aligned_self_values.append(last_self_value)
                    aligned_other_values.append(last_other_value)
                    aligned_periods.append(period)
            
            return (
                UnivariateMeasurements(self.concept, self.granularity, aligned_self_values, aligned_periods),
                UnivariateMeasurements(other.concept, other.granularity, aligned_other_values, aligned_periods)
            )
        
        else:
            raise ValueError(f"Unknown alignment strategy: {strategy}. Use 'intersection', 'forward_fill', or 'union_fill'")
