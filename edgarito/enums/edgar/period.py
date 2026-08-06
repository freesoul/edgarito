import enum


class FiscalPeriod(enum.Enum):
    """
    Constant value in the SEC EDGAR database
    """

    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    FY = "FY"
    Year = "FY"  # Backwards-compatible alias.

    def __gt__(self, other: "FiscalPeriod"):
        return FISCAL_PERIOD_PRIORITY[self] > FISCAL_PERIOD_PRIORITY[other]

    def __ge__(self, other: "FiscalPeriod"):
        return FISCAL_PERIOD_PRIORITY[self] >= FISCAL_PERIOD_PRIORITY[other]

    def __lt__(self, other: "FiscalPeriod"):
        return FISCAL_PERIOD_PRIORITY[self] < FISCAL_PERIOD_PRIORITY[other]

    def __le__(self, other: "FiscalPeriod"):
        return FISCAL_PERIOD_PRIORITY[self] <= FISCAL_PERIOD_PRIORITY[other]

    def __sub__(self, other: "FiscalPeriod") -> int:
        return FISCAL_PERIOD_PRIORITY[self] - FISCAL_PERIOD_PRIORITY[other]

    def __add__(self, other: "FiscalPeriod") -> "FiscalPeriod":
        return FiscalPeriod(
            FISCAL_PERIOD_PRIORITY[self] + FISCAL_PERIOD_PRIORITY[other]
        )


FISCAL_PERIOD_PRIORITY = {
    FiscalPeriod.Q1: 1,
    FiscalPeriod.Q2: 2,
    FiscalPeriod.Q3: 3,
    FiscalPeriod.Q4: 4,
    FiscalPeriod.Year: 5,
}
