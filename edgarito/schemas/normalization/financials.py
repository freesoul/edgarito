import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.identifiers import SecurityIdentifiers


class FinancialStatement(str, Enum):
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"


class ObservationDerivationKind(str, Enum):
    PERIOD_RECONSTRUCTION = "period_reconstruction"
    COMPONENT_AGGREGATION = "component_aggregation"
    CONCEPT_FALLBACK = "concept_fallback"


class FinancialConcept(str, Enum):
    REVENUE = "revenue"
    OPERATING_INCOME = "operating_income"
    PRETAX_INCOME = "pretax_income"
    INCOME_TAX_EXPENSE = "income_tax_expense"
    NET_INCOME = "net_income"
    NET_INCOME_COMMON = "net_income_common"
    INTEREST_EXPENSE = "interest_expense"
    TOTAL_ASSETS = "total_assets"
    CURRENT_ASSETS = "current_assets"
    ACCOUNTS_RECEIVABLE = "accounts_receivable"
    INVENTORY = "inventory"
    PREPAID_AND_OTHER_CURRENT_ASSETS = "prepaid_and_other_current_assets"
    TOTAL_LIABILITIES = "total_liabilities"
    CURRENT_LIABILITIES = "current_liabilities"
    ACCOUNTS_PAYABLE = "accounts_payable"
    ACCRUED_LIABILITIES = "accrued_liabilities"
    DEFERRED_REVENUE_CURRENT = "deferred_revenue_current"
    SHORT_TERM_DEBT = "short_term_debt"
    LONG_TERM_DEBT_CURRENT = "long_term_debt_current"
    LONG_TERM_DEBT_NONCURRENT = "long_term_debt_noncurrent"
    STOCKHOLDERS_EQUITY = "stockholders_equity"
    COMMON_EQUITY = "common_equity"
    CASH_AND_EQUIVALENTS = "cash_and_equivalents"
    SHORT_TERM_INVESTMENTS = "short_term_investments"
    NONCURRENT_INVESTMENTS = "noncurrent_investments"
    GOODWILL = "goodwill"
    INTANGIBLE_ASSETS_NET = "intangible_assets_net"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    DEPRECIATION_AND_AMORTIZATION = "depreciation_and_amortization"
    CAPITAL_EXPENDITURES = "capital_expenditures"
    DIVIDENDS_PAID = "dividends_paid"
    DEBT_ISSUANCE = "debt_issuance"
    DEBT_REPAYMENT = "debt_repayment"
    DIVIDENDS_PER_SHARE = "dividends_per_share"
    SHARES_OUTSTANDING = "shares_outstanding"
    WEIGHTED_AVERAGE_BASIC_SHARES = "weighted_average_basic_shares"
    WEIGHTED_AVERAGE_DILUTED_SHARES = "weighted_average_diluted_shares"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()

    @property
    def statement(self) -> FinancialStatement:
        return CONCEPT_STATEMENTS[self]


CONCEPT_STATEMENTS: dict[FinancialConcept, FinancialStatement] = {
    FinancialConcept.REVENUE: FinancialStatement.INCOME_STATEMENT,
    FinancialConcept.OPERATING_INCOME: FinancialStatement.INCOME_STATEMENT,
    FinancialConcept.PRETAX_INCOME: FinancialStatement.INCOME_STATEMENT,
    FinancialConcept.INCOME_TAX_EXPENSE: FinancialStatement.INCOME_STATEMENT,
    FinancialConcept.NET_INCOME: FinancialStatement.INCOME_STATEMENT,
    FinancialConcept.NET_INCOME_COMMON: FinancialStatement.INCOME_STATEMENT,
    FinancialConcept.INTEREST_EXPENSE: FinancialStatement.INCOME_STATEMENT,
    FinancialConcept.TOTAL_ASSETS: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.CURRENT_ASSETS: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.ACCOUNTS_RECEIVABLE: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.INVENTORY: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: (
        FinancialStatement.BALANCE_SHEET
    ),
    FinancialConcept.TOTAL_LIABILITIES: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.CURRENT_LIABILITIES: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.ACCOUNTS_PAYABLE: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.ACCRUED_LIABILITIES: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.DEFERRED_REVENUE_CURRENT: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.SHORT_TERM_DEBT: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.LONG_TERM_DEBT_CURRENT: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.LONG_TERM_DEBT_NONCURRENT: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.STOCKHOLDERS_EQUITY: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.COMMON_EQUITY: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.CASH_AND_EQUIVALENTS: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.SHORT_TERM_INVESTMENTS: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.NONCURRENT_INVESTMENTS: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.GOODWILL: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.INTANGIBLE_ASSETS_NET: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.OPERATING_CASH_FLOW: FinancialStatement.CASH_FLOW,
    FinancialConcept.DEPRECIATION_AND_AMORTIZATION: FinancialStatement.CASH_FLOW,
    FinancialConcept.CAPITAL_EXPENDITURES: FinancialStatement.CASH_FLOW,
    FinancialConcept.DIVIDENDS_PAID: FinancialStatement.CASH_FLOW,
    FinancialConcept.DEBT_ISSUANCE: FinancialStatement.CASH_FLOW,
    FinancialConcept.DEBT_REPAYMENT: FinancialStatement.CASH_FLOW,
    FinancialConcept.DIVIDENDS_PER_SHARE: FinancialStatement.CASH_FLOW,
    FinancialConcept.SHARES_OUTSTANDING: FinancialStatement.BALANCE_SHEET,
    FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES: (
        FinancialStatement.INCOME_STATEMENT
    ),
    FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES: (
        FinancialStatement.INCOME_STATEMENT
    ),
}


class FinancialObservation(BaseModel):
    """One provider-neutral reported or period-reconstructed financial value."""

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

    derivation_kind: Optional[ObservationDerivationKind] = None
    derivation: Optional[str] = None

    @model_validator(mode="after")
    def validate_statement(self) -> "FinancialObservation":
        if self.statement != self.concept.statement:
            raise ValueError(
                f"{self.concept.value} belongs to "
                f"{self.concept.statement.value}, not {self.statement.value}"
            )
        return self

    @property
    def period_key(self) -> tuple[int, FiscalPeriod]:
        return self.fiscal_year, self.fiscal_period

    @property
    def is_derived(self) -> bool:
        """Retain the existing convenience API for reconstructed periods."""
        return self.derivation_kind is not None


class NormalizedCompanyFinancials(BaseModel):
    provider: str
    company_id: str
    company_name: str
    ticker: Optional[str] = None
    identifiers: Optional[SecurityIdentifiers] = None
    retrieved_at: Optional[datetime.datetime] = None
    observations: list[FinancialObservation] = Field(default_factory=list)

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(
        cls, value: Optional[datetime.datetime]
    ) -> Optional[datetime.datetime]:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("retrieved_at must include a timezone")
        return value

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
