import datetime
import re
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgarito.schemas.valuation.assumptions import (
    AssumptionOrigin,
    AssumptionProvenance,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReferenceDatasetRelease(BaseModel):
    """Immutable identity and integrity contract for a reference dataset release."""

    model_config = ConfigDict(frozen=True)

    dataset: str = Field(min_length=1)
    version: str = Field(min_length=1)
    published_on: datetime.date
    source_url: str = Field(min_length=1)
    expected_sha256: str
    region: Optional[str] = None

    @field_validator("dataset", "version", "region")
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Release text fields cannot be blank")
        return normalized

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith("https://"):
            raise ValueError("Reference dataset URLs must use HTTPS")
        return normalized

    @field_validator("expected_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        return normalized


class ReferenceDatasetMetadata(BaseModel):
    """Provenance retained with every parsed reference-data snapshot."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    version: str = Field(min_length=1)
    published_on: datetime.date
    retrieved_at: datetime.datetime
    source_url: str = Field(min_length=1)
    sha256: str
    region: Optional[str] = None

    @field_validator("provider", "dataset", "version", "region")
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Reference metadata fields cannot be blank")
        return normalized

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime.datetime) -> datetime.datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return normalized

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith("https://"):
            raise ValueError("Reference dataset URLs must use HTTPS")
        return normalized

    def assumption_provenance(self) -> AssumptionProvenance:
        return AssumptionProvenance(
            origin=AssumptionOrigin.REFERENCE_DATASET,
            provider=self.provider,
            dataset=self.dataset,
            version=self.version,
            observed_on=self.published_on,
            retrieved_at=self.retrieved_at,
        )


class CountryRiskPremium(BaseModel):
    model_config = ConfigDict(frozen=True)

    country: str = Field(min_length=1)
    rating: Optional[str] = None
    adjusted_default_spread: Decimal
    country_risk_premium: Decimal
    equity_risk_premium: Decimal
    corporate_tax_rate: Decimal
    sovereign_cds_spread: Optional[Decimal] = None
    cds_equity_risk_premium: Optional[Decimal] = None

    @field_validator("country", "rating")
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Country risk text fields cannot be blank")
        return normalized

    @field_validator(
        "adjusted_default_spread",
        "country_risk_premium",
        "equity_risk_premium",
        "corporate_tax_rate",
        "sovereign_cds_spread",
        "cds_equity_risk_premium",
    )
    @classmethod
    def require_finite(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("Country risk values must be finite")
        return value


class CountryRiskPremiumSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: ReferenceDatasetMetadata
    countries: tuple[CountryRiskPremium, ...]

    @model_validator(mode="after")
    def validate_countries(self) -> "CountryRiskPremiumSnapshot":
        if not self.countries:
            raise ValueError("Country risk snapshots cannot be empty")
        keys = [row.country.casefold() for row in self.countries]
        if len(keys) != len(set(keys)):
            raise ValueError("Country risk snapshots cannot repeat a country")
        return self

    def find_country(self, country: str) -> Optional[CountryRiskPremium]:
        key = country.strip().casefold()
        return next(
            (row for row in self.countries if row.country.casefold() == key), None
        )


class IndustryBeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    industry: str = Field(min_length=1)
    number_of_firms: int = Field(ge=1)
    levered_beta: Decimal
    debt_to_equity: Decimal
    effective_tax_rate: Decimal
    unlevered_beta: Decimal
    cash_to_firm_value: Decimal
    cash_adjusted_unlevered_beta: Decimal
    hilo_risk: Optional[Decimal] = None
    equity_standard_deviation: Optional[Decimal] = None
    operating_income_standard_deviation: Optional[Decimal] = None

    @field_validator("industry")
    @classmethod
    def normalize_industry(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Industry cannot be blank")
        return normalized

    @field_validator(
        "levered_beta",
        "debt_to_equity",
        "effective_tax_rate",
        "unlevered_beta",
        "cash_to_firm_value",
        "cash_adjusted_unlevered_beta",
        "hilo_risk",
        "equity_standard_deviation",
        "operating_income_standard_deviation",
    )
    @classmethod
    def require_finite(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("Industry beta values must be finite")
        return value


class IndustryBetaSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: ReferenceDatasetMetadata
    industries: tuple[IndustryBeta, ...]

    @model_validator(mode="after")
    def validate_industries(self) -> "IndustryBetaSnapshot":
        if not self.industries:
            raise ValueError("Industry beta snapshots cannot be empty")
        keys = [row.industry.casefold() for row in self.industries]
        if len(keys) != len(set(keys)):
            raise ValueError("Industry beta snapshots cannot repeat an industry")
        return self

    def find_industry(self, industry: str) -> Optional[IndustryBeta]:
        key = industry.strip().casefold()
        return next(
            (row for row in self.industries if row.industry.casefold() == key), None
        )


__all__ = [
    "CountryRiskPremium",
    "CountryRiskPremiumSnapshot",
    "IndustryBeta",
    "IndustryBetaSnapshot",
    "ReferenceDatasetMetadata",
    "ReferenceDatasetRelease",
]
