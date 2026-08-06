import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity


class FinancialStatement(str, Enum):
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"


class FinancialConcept(str, Enum):
    REVENUE = "revenue"
    OPERATING_INCOME = "operating_income"
    NET_INCOME = "net_income"
    TOTAL_ASSETS = "total_assets"
    TOTAL_LIABILITIES = "total_liabilities"
    STOCKHOLDERS_EQUITY = "stockholders_equity"
    CASH_AND_EQUIVALENTS = "cash_and_equivalents"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    CAPITAL_EXPENDITURES = "capital_expenditures"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


class FinancialObservation(BaseModel):
    """One provider-neutral reported or derived financial value."""

    concept: FinancialConcept
    statement: FinancialStatement
    value: Decimal
    unit: str
    granularity: Granularity
    fiscal_year: int
    fiscal_period: FiscalPeriod
    period_start: Optional[datetime.date] = None
    period_end: datetime.date

    provider: str
    taxonomy: str
    source_concept: str
    accession_number: Optional[str] = None
    form: Optional[str] = None
    filed: Optional[datetime.date] = None

    is_derived: bool = False
    derivation: Optional[str] = None

    @property
    def period_key(self) -> tuple[int, FiscalPeriod]:
        return self.fiscal_year, self.fiscal_period


class NormalizedCompanyFinancials(BaseModel):
    provider: str
    company_id: str
    company_name: str
    ticker: Optional[str] = None
    observations: list[FinancialObservation] = Field(default_factory=list)

    def filtered(
        self,
        granularity: Optional[Granularity] = None,
        concepts: Optional[set[FinancialConcept]] = None,
    ) -> list[FinancialObservation]:
        return [
            observation
            for observation in self.observations
            if (granularity is None or observation.granularity == granularity)
            and (not concepts or observation.concept in concepts)
        ]
