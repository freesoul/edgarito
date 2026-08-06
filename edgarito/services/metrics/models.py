import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import FinancialConcept


class FinancialMetric(str, Enum):
    REVENUE_GROWTH = "revenue_growth"
    OPERATING_MARGIN = "operating_margin"
    NET_MARGIN = "net_margin"
    FREE_CASH_FLOW = "free_cash_flow"
    FREE_CASH_FLOW_MARGIN = "free_cash_flow_margin"
    RETURN_ON_ASSETS = "return_on_assets"
    RETURN_ON_EQUITY = "return_on_equity"
    LIABILITIES_TO_ASSETS = "liabilities_to_assets"
    CASH_TO_LIABILITIES = "cash_to_liabilities"
    OPERATING_CASH_FLOW_TO_NET_INCOME = "operating_cash_flow_to_net_income"

    @property
    def label(self) -> str:
        return {
            FinancialMetric.REVENUE_GROWTH: "Revenue Growth",
            FinancialMetric.OPERATING_MARGIN: "Operating Margin",
            FinancialMetric.NET_MARGIN: "Net Margin",
            FinancialMetric.FREE_CASH_FLOW: "Free Cash Flow",
            FinancialMetric.FREE_CASH_FLOW_MARGIN: "Free Cash Flow Margin",
            FinancialMetric.RETURN_ON_ASSETS: "Return on Assets",
            FinancialMetric.RETURN_ON_EQUITY: "Return on Equity",
            FinancialMetric.LIABILITIES_TO_ASSETS: "Liabilities / Assets",
            FinancialMetric.CASH_TO_LIABILITIES: "Cash / Liabilities",
            FinancialMetric.OPERATING_CASH_FLOW_TO_NET_INCOME: (
                "Operating Cash Flow / Net Income"
            ),
        }[self]


class MetricObservation(BaseModel):
    metric: FinancialMetric
    value: Decimal
    unit: str
    granularity: Granularity
    fiscal_year: int
    fiscal_period: FiscalPeriod
    period_end: datetime.date
    provider: str
    formula: str
    input_concepts: tuple[FinancialConcept, ...]

    @property
    def period_key(self) -> tuple[int, FiscalPeriod]:
        return self.fiscal_year, self.fiscal_period


class CompanyMetrics(BaseModel):
    provider: str
    company_id: str
    company_name: str
    ticker: str | None = None
    observations: list[MetricObservation] = Field(default_factory=list)

    def filtered(
        self,
        granularity: Granularity | None = None,
        metrics: set[FinancialMetric] | None = None,
    ) -> list[MetricObservation]:
        return [
            observation
            for observation in self.observations
            if (granularity is None or observation.granularity == granularity)
            and (metrics is None or observation.metric in metrics)
        ]
