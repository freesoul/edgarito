import json
from collections import OrderedDict
from typing import List, Optional

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts, Measurement
from edgarito.schemas.reader.measurements import UnivariateMeasurements

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.edgar.filing_type import FilingType
from edgarito.enums.granularity import Granularity


class FinancialStatementReader:

    def load_from_facts(self, facts: CompanyFacts):
        self._data = facts

    def load_from_json_file(self, path: str):
        with open(path, "r") as file:
            data_json = json.load(file)
        self._data = CompanyFacts(**data_json)

    def get_revenue(self, granularity: Granularity) -> UnivariateMeasurements:
        return self.get_usd_concept("RevenueFromContractWithCustomerExcludingAssessedTax", granularity)

    def get_cost_of_revenue(self, granularity: Granularity) -> UnivariateMeasurements:
        return self.get_usd_concept("CostOfRevenue", granularity)

    def get_research_and_development_expense(self, granularity: Granularity) -> UnivariateMeasurements:
        return self.get_usd_concept("ResearchAndDevelopmentExpense", granularity)

    def get_general_and_administrative_expense(self, granularity: Granularity) -> UnivariateMeasurements:
        return self.get_usd_concept("GeneralAndAdministrativeExpense", granularity)

    def get_usd_concept(self, concept: str, granularity: Granularity) -> UnivariateMeasurements:
        if concept not in self._data.facts.us_gaap:
            raise ValueError(f"Concept {concept} not found in the data")

        if granularity == Granularity.ANNUAL:
            measurements_filtered_filings = self.filter_multivariate_measurements(
                self._data.facts.us_gaap[concept].units.USD, filing_type=FilingType.FILING_10K
            )
        elif granularity == Granularity.QUARTERLY:
            measurements_filtered_filings = self.filter_multivariate_measurements(
                self._data.facts.us_gaap[concept].units.USD, filing_type=FilingType.FILING_10K
            )
            measurements_filtered_filings += self.filter_multivariate_measurements(
                self._data.facts.us_gaap[concept].units.USD, filing_type=FilingType.FILING_10Q
            )
        else:
            raise ValueError(f"Granularity {granularity} not supported")

        univariate = UnivariateMeasurements.from_measurements(concept=concept, granularity=granularity, measurements=measurements_filtered_filings)
        univariate.sort()

        if granularity == Granularity.QUARTERLY:
            self._fy_to_q4(univariate)

        return univariate

    def _fy_to_q4(self, univariate: UnivariateMeasurements):
        """
        Change FY by Q4 by substracting each FY by its previous Q3 value.
        This also removes the incomplete FY and previous quarters.
        """
        for i, period in enumerate(univariate.periods):
            if i < 3:
                continue
            if period.fp == FiscalPeriod.Year:
                univariate.values[i] -= univariate.values[i - 1] + univariate.values[i - 2] + univariate.values[i - 3]
                univariate.periods[i].fp = FiscalPeriod.Q4

        # remove everything before the first "fp" which is FY, including it
        for i, period in enumerate(univariate.periods):
            if i == 3:
                break
            if period.fp == FiscalPeriod.Year:
                univariate.values = univariate.values[i + 1 :]
                univariate.periods = univariate.periods[i + 1 :]
                break

    def filter_multivariate_measurements(
        self,
        measurements: List[Measurement],
        filing_type: Optional[FilingType],
    ) -> List[Measurement]:

        filtered_measurements = []

        for measurement in measurements:

            if filing_type is not None and measurement.form != filing_type.value:
                continue

            if measurement.frame is None:
                continue

            if measurement.calendar_year is None:
                continue

            filtered_measurements.append(measurement)

        return filtered_measurements


if __name__ == "__main__":
    reader = FinancialStatementReader()
    reader.load_from_json_file("cache/edgar_rest/api/xbrl/companyfacts/CIK0001652044.json")

    revenues = reader.get_cost_of_revenue(Granularity.ANNUAL)
    print(revenues)

    costs = reader.get_revenue(Granularity.ANNUAL)
    print(costs)

    expenses_admin = reader.get_general_and_administrative_expense(Granularity.ANNUAL)
    print(expenses_admin)

    expenses_rd = reader.get_research_and_development_expense(Granularity.ANNUAL)
    print(expenses_rd)
