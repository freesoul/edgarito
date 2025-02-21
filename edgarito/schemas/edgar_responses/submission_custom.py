import datetime
from typing import Optional

from pydantic import BaseModel


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
