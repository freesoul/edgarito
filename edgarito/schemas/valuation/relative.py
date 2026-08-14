"""Provider-neutral inputs and outputs for relative implied valuation."""

import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgarito.schemas.valuation.selection import (
    MultipleConfidence,
    RelativeValuationBasis,
)


class RelativeNumeratorBasis(str, Enum):
    ENTERPRISE_VALUE = "enterprise_value"
    EQUITY_VALUE = "equity_value"


class ForwardValuationMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    basis: RelativeValuationBasis
    amount: Decimal = Field(gt=0)
    label: str
    target_date: datetime.date
    currency: str
    numerator_basis: RelativeNumeratorBasis

    @model_validator(mode="after")
    def validate_basis(self) -> "ForwardValuationMetric":
        enterprise_bases = {
            RelativeValuationBasis.EV_TO_REVENUE,
            RelativeValuationBasis.EV_TO_EBIT,
            RelativeValuationBasis.EV_TO_EBITDA,
            RelativeValuationBasis.EV_TO_FCF,
        }
        equity_bases = {
            RelativeValuationBasis.PE,
            RelativeValuationBasis.PRICE_TO_BOOK,
            RelativeValuationBasis.PRICE_TO_TANGIBLE_BOOK,
            RelativeValuationBasis.PRICE_TO_AFFO,
            RelativeValuationBasis.PRICE_TO_NAV,
        }
        if self.basis not in enterprise_bases | equity_bases:
            raise ValueError(f"Unsupported implied valuation basis: {self.basis.value}")
        enterprise = self.basis in enterprise_bases
        if enterprise != (
            self.numerator_basis == RelativeNumeratorBasis.ENTERPRISE_VALUE
        ):
            raise ValueError(
                "Relative numerator type does not match the multiple basis"
            )
        return self


class RelativeCapitalBridge(BaseModel):
    model_config = ConfigDict(frozen=True)

    net_debt: Decimal
    non_operating_assets: Decimal = Decimal(0)


class ProviderNeutralRelativeCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    multiple: Decimal = Field(gt=0)
    target_date_numerator_value: Decimal
    target_date_equity_value: Decimal
    target_date_value_per_share: Decimal
    present_value_per_share: Decimal


class ProviderNeutralRelativeValuation(BaseModel):
    model_config = ConfigDict(frozen=True)

    valuation_date: datetime.date
    target_date: datetime.date
    horizon_years: Decimal = Field(gt=0)
    currency: str
    metric: ForwardValuationMetric
    diluted_shares: Decimal = Field(gt=0)
    discount_rate: Decimal
    lower_case: ProviderNeutralRelativeCase
    point_case: ProviderNeutralRelativeCase
    upper_case: ProviderNeutralRelativeCase
    confidence: MultipleConfidence
    current_price: Decimal | None = None
    current_price_implied_multiple: Decimal | None = None
    warnings: tuple[str, ...] = ()
