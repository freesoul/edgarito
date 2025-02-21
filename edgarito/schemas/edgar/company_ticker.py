from pydantic import BaseModel


class CompanyTicker(BaseModel):
    cik_str: int
    ticker: str
    title: str
