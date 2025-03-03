import enum


class FiscalPeriod(enum.Enum):
    """
    Constant value in the SEC EDGAR database
    """

    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    Year = "FY"

    def __gt__(self, other: "FiscalPeriod"):
        return FISCAL_PERIOD_PRIORITY[self] > FISCAL_PERIOD_PRIORITY[other]

    def __ge__(self, other: "FiscalPeriod"):
        return FISCAL_PERIOD_PRIORITY[self] >= FISCAL_PERIOD_PRIORITY[other]

    def __lt__(self, other: "FiscalPeriod"):
        return FISCAL_PERIOD_PRIORITY[self] < FISCAL_PERIOD_PRIORITY[other]

    def __le__(self, other: "FiscalPeriod"):
        return FISCAL_PERIOD_PRIORITY[self] <= FISCAL_PERIOD_PRIORITY[other]


FISCAL_PERIOD_PRIORITY = {FiscalPeriod.Q1: 1, FiscalPeriod.Q2: 2, FiscalPeriod.Q3: 3, FiscalPeriod.Q4: 4, FiscalPeriod.Year: 5}
