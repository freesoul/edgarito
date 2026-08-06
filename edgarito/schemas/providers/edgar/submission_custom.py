import datetime
from typing import Optional

from pydantic import BaseModel

from edgarito.enums.edgar.core_filing_type import CoreFilingType


class TransposedFiling(BaseModel):
    """
    Created for transposing! not from original responses.
    """

    accessionNumber: str
    filingDate: datetime.date
    acceptanceDateTime: datetime.datetime
    act: str
    form: str
    fileNumber: str
    filmNumber: str
    items: str
    core_type: Optional[str]
    size: int
    isXBRL: bool
    isInlineXBRL: bool
    primaryDocument: str
    primaryDocDescription: str
    reportDate: Optional[datetime.date]

    @property
    def parsed_type(self) -> Optional[CoreFilingType]:
        """
        Try to parse the form string into a CoreFilingType enum.
        Returns None if the form type is not in the core filing types.
        """
        return CoreFilingType.try_from_string(self.form)
