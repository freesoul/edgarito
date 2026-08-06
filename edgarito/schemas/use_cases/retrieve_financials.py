from typing import Optional

from pydantic import BaseModel, model_validator

from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import FinancialConcept


class RetrieveFinancialsRequest(BaseModel):
    ticker: Optional[str] = None
    cik: Optional[int] = None
    granularity: Optional[Granularity] = Granularity.ANNUAL
    concepts: Optional[set[FinancialConcept]] = None
    use_cache: bool = True
    make_cache: bool = True

    @model_validator(mode="after")
    def require_one_identifier(self):
        if (self.ticker is None) == (self.cik is None):
            raise ValueError("Provide exactly one of ticker or cik")
        return self
