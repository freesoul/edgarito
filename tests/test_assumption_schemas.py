import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from edgarito.schemas import ValuationAssumptionSet as PublicValuationAssumptionSet
from edgarito.schemas.valuation import (
    AssumptionOrigin,
    AssumptionProvenance,
    AssumptionUnit,
    ValuationAssumption,
    ValuationAssumptionKind,
    ValuationAssumptionSet,
    ValuationScenario,
)

UTC = datetime.timezone.utc
VALUATION_DATE = datetime.date(2026, 8, 6)


def test_valuation_assumptions_are_available_from_the_public_schema_api():
    assert PublicValuationAssumptionSet is ValuationAssumptionSet


def _explicit_assumption(
    kind: ValuationAssumptionKind,
    value: str,
    *,
    unit: AssumptionUnit = AssumptionUnit.PERCENTAGE_POINTS,
    forecast_year: int | None = None,
    currency: str | None = "usd",
) -> ValuationAssumption:
    return ValuationAssumption(
        kind=kind,
        value=Decimal(value),
        unit=unit,
        selected_on=VALUATION_DATE,
        forecast_year=forecast_year,
        currency=currency,
        provenance=AssumptionProvenance(origin=AssumptionOrigin.EXPLICIT),
    )


def test_assumption_set_retains_scenario_scope_and_versioned_provenance():
    risk_free_rate = ValuationAssumption(
        kind=ValuationAssumptionKind.RISK_FREE_RATE,
        value=Decimal("4.12"),
        unit=AssumptionUnit.PERCENTAGE_POINTS,
        selected_on=VALUATION_DATE,
        currency="usd",
        country="us",
        provenance=AssumptionProvenance(
            origin=AssumptionOrigin.MARKET_OBSERVATION,
            provider="treasury",
            series_id="DGS10",
            observed_on=VALUATION_DATE,
            retrieved_at=datetime.datetime(2026, 8, 6, 18, tzinfo=UTC),
        ),
    )
    equity_risk_premium = ValuationAssumption(
        kind=ValuationAssumptionKind.EQUITY_RISK_PREMIUM,
        value=Decimal("4.33"),
        unit=AssumptionUnit.PERCENTAGE_POINTS,
        selected_on=VALUATION_DATE,
        currency="usd",
        provenance=AssumptionProvenance(
            origin=AssumptionOrigin.REFERENCE_DATASET,
            provider="damodaran",
            dataset="country-risk-premiums",
            version="2026-01-05",
            observed_on=datetime.date(2026, 1, 5),
        ),
    )
    assumptions = ValuationAssumptionSet(
        valuation_date=VALUATION_DATE,
        currency="usd",
        scenario=ValuationScenario.BASE,
        name=" Base case ",
        assumptions=(risk_free_rate, equity_risk_premium),
    )

    assert assumptions.currency == "USD"
    assert assumptions.name == "Base case"
    assert assumptions.require(ValuationAssumptionKind.RISK_FREE_RATE).value == (
        Decimal("4.12")
    )
    assert (
        assumptions.require(
            ValuationAssumptionKind.EQUITY_RISK_PREMIUM
        ).provenance.version
        == "2026-01-05"
    )
    assert (
        ValuationAssumptionSet.model_validate_json(assumptions.model_dump_json())
        == assumptions
    )


def test_forecast_years_allow_auditable_assumption_paths():
    assumptions = ValuationAssumptionSet(
        valuation_date=VALUATION_DATE,
        currency="USD",
        assumptions=(
            _explicit_assumption(
                ValuationAssumptionKind.REVENUE_GROWTH, "8", forecast_year=1
            ),
            _explicit_assumption(
                ValuationAssumptionKind.REVENUE_GROWTH, "6", forecast_year=2
            ),
        ),
    )

    assert assumptions.require(
        ValuationAssumptionKind.REVENUE_GROWTH, forecast_year=2
    ).value == Decimal("6")
    with pytest.raises(ValueError, match="Missing terminal_growth"):
        assumptions.require(ValuationAssumptionKind.TERMINAL_GROWTH)


def test_assumptions_reject_wrong_units_future_sources_and_currency_mismatch():
    with pytest.raises(ValidationError, match="must use the multiple unit"):
        _explicit_assumption(
            ValuationAssumptionKind.LEVERED_BETA,
            "1.1",
            unit=AssumptionUnit.PERCENTAGE_POINTS,
        )

    with pytest.raises(ValidationError, match="future observation"):
        ValuationAssumption(
            kind=ValuationAssumptionKind.RISK_FREE_RATE,
            value=Decimal("4"),
            unit=AssumptionUnit.PERCENTAGE_POINTS,
            selected_on=VALUATION_DATE,
            provenance=AssumptionProvenance(
                origin=AssumptionOrigin.MARKET_OBSERVATION,
                provider="treasury",
                observed_on=datetime.date(2026, 8, 7),
            ),
        )

    with pytest.raises(ValidationError, match="must match the valuation currency"):
        ValuationAssumptionSet(
            valuation_date=VALUATION_DATE,
            currency="USD",
            assumptions=(
                _explicit_assumption(
                    ValuationAssumptionKind.TERMINAL_GROWTH,
                    "2",
                    currency="EUR",
                ),
            ),
        )


def test_assumption_set_rejects_duplicate_keys_and_unversioned_reference_data():
    assumption = _explicit_assumption(ValuationAssumptionKind.TERMINAL_GROWTH, "2")
    with pytest.raises(ValidationError, match="keys must be unique"):
        ValuationAssumptionSet(
            valuation_date=VALUATION_DATE,
            currency="USD",
            assumptions=(assumption, assumption),
        )

    with pytest.raises(ValidationError, match="Reference assumptions require"):
        AssumptionProvenance(
            origin=AssumptionOrigin.REFERENCE_DATASET,
            provider="damodaran",
            dataset="country-risk-premiums",
            observed_on=datetime.date(2026, 1, 5),
        )
