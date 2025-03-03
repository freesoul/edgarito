from typing import List, Optional, Tuple, Iterator

from edgarito.enums.edgar.period import FiscalPeriod

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


class UnivariateMeasurements:
    concept: str
    values: List[float]
    periods: List[MeasurementPeriod]

    def __init__(self, concept: str, values: List[float], periods: List[MeasurementPeriod]):
        self.concept = concept
        self.values = values
        self.periods = periods

    @staticmethod
    def from_measurements(concept: str, measurements: List[Measurement]) -> "UnivariateMeasurements":
        concept = concept
        values = []
        periods = []
        # form_types = set()
        for measurement in measurements:
            values.append(measurement.val)
            periods.append(MeasurementPeriod(year=measurement.calendar_year, fp=measurement.fp))
            # form_types.add(measurement.form)
        # if len(form_types) > 1:
        #     raise ValueError(f"Measurements for concept {concept} have different form types: {form_types}")

        return UnivariateMeasurements(concept=concept, values=values, periods=periods)

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
        return UnivariateMeasurements(concept=self.concept, values=values, periods=periods)

    def intersect(self, other: "UnivariateMeasurements") -> "UnivariateMeasurements":
        """
        Returns a UnivariateMeasurements containing the intersection of the measurement by period.
        """
        return self.limit_periods(min_period=other.min_period, max_period=other.max_period)
