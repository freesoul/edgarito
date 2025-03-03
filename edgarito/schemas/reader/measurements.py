from typing import List, Optional, Tuple, Iterator

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity

from edgarito.schemas.edgar_responses.company_facts import Measurement


class MeasurementPeriod:
    year: int
    fp: FiscalPeriod

    def __init__(self, year: int, fp: FiscalPeriod):
        self.year = year
        self.fp = fp

    def __eq__(self, other: "MeasurementPeriod"):
        return self.year == other.year and self.fp == other.fp

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

    def __init__(self, concept: str, granularity: Granularity, values: List[float], periods: List[MeasurementPeriod]):
        self.concept = concept
        self.granularity = granularity
        self.periods = periods
        self.values = values

        self._validate_no_missing_periods()

    def __str__(self):
        ret = f"{self.concept}:\n"
        for value, period in zip(self.values, self.periods):
            ret += f"\t{period.year} {period.fp}\t{value}\n"
        return ret

    def _validate_no_missing_periods(self):
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
            periods.append(MeasurementPeriod(year=measurement.calendar_year, fp=measurement.fp))

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
        Returns a UnivariateMeasurements containing the intersection of the measurement by period.
        """
        return self.limit_periods(min_period=other.min_period, max_period=other.max_period)
