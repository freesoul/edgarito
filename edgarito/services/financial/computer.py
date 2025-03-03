from edgarito.schemas.reader.measurements import UnivariateMeasurements


class Computer:

    def gross_profit(revenues: UnivariateMeasurements, costs: UnivariateMeasurements) -> UnivariateMeasurements:
        """
        Compute the gross profit as the difference between revenues and costs.
        """
        return revenues - costs


if __name__ == "__main__":

    from edgarito.services.financial.reader import FinancialStatementReader, Granularity

    reader = FinancialStatementReader()
    reader.load_from_json_file("cache/edgar_rest/api/xbrl/companyfacts/CIK0001652044.json")

    revenues = reader.get_revenue(Granularity.ANNUAL)
    costs = reader.get_cost_of_revenue(Granularity.ANNUAL)

    revenues = revenues.intersect(costs)
    costs = costs.intersect(revenues)

    gross_profit = Computer.gross_profit(revenues, costs)
