from pydantic import BaseModel


class CompanyTickerResponse(BaseModel):
    cik_str: int  # This is how they name it in the response
    ticker: str
    title: str

    @property
    def cik(self):
        return self.cik_str
