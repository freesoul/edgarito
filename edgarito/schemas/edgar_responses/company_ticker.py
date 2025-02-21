from pydantic import BaseModel


class CompanyTickerResponse(BaseModel):
    cik_str: int
    ticker: str
    title: str
