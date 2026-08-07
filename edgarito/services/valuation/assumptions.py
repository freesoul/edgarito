import datetime
import re
from dataclasses import dataclass
from decimal import Decimal
from statistics import median
from typing import Optional

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.market import ReferenceMarketSeries, SecurityMarketData
from edgarito.schemas.normalization.classification import (
    NormalizedCompanyClassification,
)
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.valuation.assumptions import (
    AssumptionOrigin,
    AssumptionProvenance,
    AssumptionUnit,
    ValuationAssumption,
    ValuationAssumptionKind,
    ValuationAssumptionSet,
)
from edgarito.schemas.valuation.reference import (
    CountryRiskPremium,
    CountryRiskPremiumSnapshot,
    IndustryBeta,
    IndustryBetaSnapshot,
)
from edgarito.services.valuation.discounting import DiscountRateService
from edgarito.services.valuation.models import FcffDcfCapitalBridge


@dataclass(frozen=True)
class ResolvedDcfAssumptions:
    wacc: Decimal
    wacc_source: str
    perpetual_growth_rate: Optional[Decimal]
    perpetual_growth_source: Optional[str]
    assumption_set: ValuationAssumptionSet


class ValuationAssumptionResolver:
    """Resolve FCFF DCF assumptions with explicit values taking precedence."""

    _INDUSTRY_ALIASES = {
        "aerospacedefense": "aerospacedefense",
        "automanufacturers": "autotruck",
        # Damodaran's US industry-beta dataset does not publish a separate
        # luxury-goods row. Apparel is its closest available luxury proxy.
        "luxurygoods": "apparel",
        "apparelluxurygoods": "apparel",
        "consumerelectronics": "electronicsconsumeroffice",
        "softwareinfrastructure": "softwaresystemapplication",
        "softwareapplication": "softwaresystemapplication",
        "semiconductors": "semiconductor",
        "biotechnology": "drugsbiotechnology",
        "drugmanufacturersgeneral": "drugspharmaceutical",
        "reitdiversified": "reits",
    }

    def resolve(
        self,
        *,
        financials: NormalizedCompanyFinancials,
        capital_bridge: FcffDcfCapitalBridge,
        discount_configuration,
        terminal_configuration,
        terminal_is_perpetuity: bool,
        valuation_date: datetime.date,
        wacc_override: Optional[Decimal] = None,
        terminal_growth_override: Optional[Decimal] = None,
        classification: Optional[NormalizedCompanyClassification] = None,
        market_data: Optional[SecurityMarketData] = None,
        risk_free_series: Optional[ReferenceMarketSeries] = None,
        inflation_series: Optional[ReferenceMarketSeries] = None,
        country_snapshot: Optional[CountryRiskPremiumSnapshot] = None,
        industry_snapshot: Optional[IndustryBetaSnapshot] = None,
        company_beta: Optional[Decimal] = None,
    ) -> ResolvedDcfAssumptions:
        selected_on = valuation_date
        if (
            market_data is not None
            and market_data.latest_price is not None
            and market_data.latest_price.observed_on > valuation_date
        ):
            raise ValueError(
                "Automatic assumptions cannot use market data after valuation_date"
            )
        currency = capital_bridge.unit.upper()
        company_id = financials.company_id
        assumptions: list[ValuationAssumption] = []

        if wacc_override is not None:
            wacc = wacc_override
            wacc_source = "explicit CLI override"
            assumptions.append(
                self._explicit_assumption(
                    ValuationAssumptionKind.WACC,
                    wacc,
                    selected_on,
                    currency,
                    company_id,
                    provider="cli",
                )
            )
        elif discount_configuration.wacc is not None:
            wacc = discount_configuration.wacc
            wacc_source = "explicit valuation profile"
            assumptions.append(
                self._explicit_assumption(
                    ValuationAssumptionKind.WACC,
                    wacc,
                    selected_on,
                    currency,
                    company_id,
                    provider="valuation-profile",
                )
            )
        else:
            wacc, derived = self._derive_wacc(
                financials=financials,
                capital_bridge=capital_bridge,
                configuration=discount_configuration,
                selected_on=selected_on,
                currency=currency,
                classification=classification,
                market_data=market_data,
                risk_free_series=risk_free_series,
                country_snapshot=country_snapshot,
                industry_snapshot=industry_snapshot,
                company_beta=company_beta,
            )
            assumptions.extend(derived)
            providers = {
                item.provenance.provider for item in derived if item.provenance.provider
            }
            if market_data is not None:
                providers.add(market_data.provider)
            wacc_source = "automatic: " + ", ".join(sorted(providers))

        perpetual_growth = None
        perpetual_growth_source = None
        if terminal_is_perpetuity:
            configured_growth = terminal_configuration.perpetual_growth_rate
            if terminal_growth_override is not None:
                perpetual_growth = terminal_growth_override
                perpetual_growth_source = "explicit CLI override"
                growth_assumption = self._explicit_assumption(
                    ValuationAssumptionKind.TERMINAL_GROWTH,
                    perpetual_growth,
                    selected_on,
                    currency,
                    company_id,
                    provider="cli",
                )
            elif configured_growth is not None:
                perpetual_growth = configured_growth
                perpetual_growth_source = "explicit valuation profile"
                growth_assumption = self._explicit_assumption(
                    ValuationAssumptionKind.TERMINAL_GROWTH,
                    perpetual_growth,
                    selected_on,
                    currency,
                    company_id,
                    provider="valuation-profile",
                )
            else:
                perpetual_growth, growth_assumption = self._derive_terminal_growth(
                    wacc=wacc,
                    selected_on=selected_on,
                    currency=currency,
                    company_id=company_id,
                    inflation_series=inflation_series,
                    risk_free_series=risk_free_series,
                )
                perpetual_growth_source = (
                    growth_assumption.provenance.methodology or "automatic"
                )
            assumptions.append(growth_assumption)

        assumption_set = ValuationAssumptionSet(
            valuation_date=selected_on,
            currency=currency,
            name="resolved-fcff-dcf",
            assumptions=tuple(assumptions),
        )
        return ResolvedDcfAssumptions(
            wacc=wacc,
            wacc_source=wacc_source,
            perpetual_growth_rate=perpetual_growth,
            perpetual_growth_source=perpetual_growth_source,
            assumption_set=assumption_set,
        )

    def _derive_wacc(
        self,
        *,
        financials,
        capital_bridge,
        configuration,
        selected_on,
        currency,
        classification,
        market_data,
        risk_free_series,
        country_snapshot,
        industry_snapshot,
        company_beta,
    ) -> tuple[Decimal, list[ValuationAssumption]]:
        company_id = financials.company_id
        country_name = classification.country if classification else None
        industry_name = classification.industry if classification else None
        country_row = self._country_row(country_snapshot, country_name)
        industry_row = self._industry_row(industry_snapshot, industry_name)
        assumptions: list[ValuationAssumption] = []

        risk_free, risk_free_assumption = self._configured_or_market_rate(
            configuration.risk_free_rate,
            ValuationAssumptionKind.RISK_FREE_RATE,
            risk_free_series,
            selected_on,
            currency,
            company_id,
        )
        assumptions.append(risk_free_assumption)

        tax_rate = configuration.normalized_tax_rate
        if tax_rate is not None:
            tax_assumption = self._explicit_assumption(
                ValuationAssumptionKind.NORMALIZED_TAX_RATE,
                tax_rate,
                selected_on,
                currency,
                company_id,
                provider="valuation-profile",
            )
        else:
            tax_rate, observed_on = self._historical_tax_rate(financials)
            if tax_rate is not None:
                tax_assumption = self._historical_assumption(
                    ValuationAssumptionKind.NORMALIZED_TAX_RATE,
                    tax_rate,
                    selected_on,
                    currency,
                    company_id,
                    observed_on,
                    "Median effective tax rate from up to three profitable annual periods",
                )
            elif country_row is not None and country_snapshot is not None:
                tax_rate = country_row.corporate_tax_rate
                tax_assumption = self._reference_assumption(
                    ValuationAssumptionKind.NORMALIZED_TAX_RATE,
                    tax_rate,
                    selected_on,
                    currency,
                    company_id,
                    country_snapshot,
                    country=country_name,
                    methodology="Country corporate tax rate fallback",
                )
            else:
                raise ValueError(
                    "Automatic WACC could not resolve a normalized tax rate; set "
                    "valuation.discount_rates.normalized_tax_rate in the profile"
                )
        assumptions.append(tax_assumption)

        market_equity = configuration.market_value_equity
        if market_equity is None:
            if market_data is None or market_data.latest_price is None:
                raise ValueError(
                    "Automatic WACC requires a latest share price; provide Yahoo "
                    "market data or valuation.discount_rates.market_value_equity"
                )
            if market_data.currency != currency:
                raise ValueError(
                    "Market price and financial statements use different currencies"
                )
            market_equity = (
                market_data.latest_price.close * capital_bridge.diluted_shares
            )
        market_debt = configuration.market_value_debt
        if market_debt is None:
            if capital_bridge.gross_debt is None:
                raise ValueError(
                    "Automatic WACC requires gross debt; set "
                    "valuation.discount_rates.market_value_debt or provide gross debt and cash"
                )
            market_debt = capital_bridge.gross_debt

        beta = configuration.levered_beta
        if beta is not None:
            beta_assumption = self._explicit_assumption(
                ValuationAssumptionKind.LEVERED_BETA,
                beta,
                selected_on,
                currency,
                company_id,
                provider="valuation-profile",
                unit=AssumptionUnit.MULTIPLE,
            )
        elif company_beta is not None:
            if not company_beta.is_finite() or company_beta <= 0:
                raise ValueError("Yahoo company beta must be finite and positive")
            beta = company_beta
            beta_assumption = ValuationAssumption(
                kind=ValuationAssumptionKind.LEVERED_BETA,
                value=beta,
                unit=AssumptionUnit.MULTIPLE,
                selected_on=selected_on,
                currency=currency,
                company_id=company_id,
                provenance=AssumptionProvenance(
                    origin=AssumptionOrigin.MARKET_OBSERVATION,
                    provider="yahoo",
                    observed_on=selected_on,
                    methodology="Yahoo company-level levered beta",
                ),
            )
        else:
            unlevered_beta = configuration.unlevered_beta
            if unlevered_beta is not None:
                unlevered_assumption = self._explicit_assumption(
                    ValuationAssumptionKind.UNLEVERED_BETA,
                    unlevered_beta,
                    selected_on,
                    currency,
                    company_id,
                    provider="valuation-profile",
                    unit=AssumptionUnit.MULTIPLE,
                )
            elif industry_row is not None and industry_snapshot is not None:
                unlevered_beta = industry_row.unlevered_beta
                unlevered_assumption = self._reference_assumption(
                    ValuationAssumptionKind.UNLEVERED_BETA,
                    unlevered_beta,
                    selected_on,
                    currency,
                    company_id,
                    industry_snapshot,
                    industry=industry_row.industry,
                    methodology=f"Damodaran industry match for {industry_name}",
                    unit=AssumptionUnit.MULTIPLE,
                )
            else:
                raise ValueError(
                    f"Automatic WACC could not match industry {industry_name!r} to "
                    "the Damodaran beta dataset; set levered_beta or unlevered_beta "
                    "in the profile"
                )
            assumptions.append(unlevered_assumption)
            beta = DiscountRateService.lever_beta(
                unlevered_beta, market_debt, market_equity, tax_rate
            )
            beta_assumption = ValuationAssumption(
                kind=ValuationAssumptionKind.LEVERED_BETA,
                value=beta,
                unit=AssumptionUnit.MULTIPLE,
                selected_on=selected_on,
                currency=currency,
                company_id=company_id,
                provenance=AssumptionProvenance(
                    origin=AssumptionOrigin.DERIVED,
                    provider="edgarito",
                    methodology=(
                        "Hamada relevering using latest book debt as a market-debt "
                        "proxy and FX-aligned market capitalization"
                    ),
                ),
            )
        assumptions.append(beta_assumption)

        equity_premium = configuration.equity_risk_premium
        country_premium = configuration.country_risk_premium
        country_premium_methodology = "Country equity risk premium"
        if equity_premium is None:
            if country_row is None or country_snapshot is None:
                raise ValueError(
                    f"Automatic WACC could not match country {country_name!r} to "
                    "the Damodaran country-risk dataset; set equity_risk_premium "
                    "and country_risk_premium in the profile"
                )
            equity_premium = (
                country_row.equity_risk_premium - country_row.country_risk_premium
            )
            if country_premium is None:
                if self._is_mature_market_base(country_name):
                    country_premium = Decimal(0)
                    country_premium_methodology = (
                        "No incremental country premium for the mature-market "
                        "base used to estimate ERP"
                    )
                else:
                    country_premium, country_premium_methodology = (
                        self._market_country_premium(country_row, equity_premium)
                    )
            erp_assumption = self._reference_assumption(
                ValuationAssumptionKind.EQUITY_RISK_PREMIUM,
                equity_premium,
                selected_on,
                currency,
                company_id,
                country_snapshot,
                country=country_name,
                methodology="Mature-market ERP = total country ERP minus country risk premium",
            )
        else:
            erp_assumption = self._explicit_assumption(
                ValuationAssumptionKind.EQUITY_RISK_PREMIUM,
                equity_premium,
                selected_on,
                currency,
                company_id,
                provider="valuation-profile",
            )
        assumptions.append(erp_assumption)
        if country_premium is None:
            country_premium = Decimal(0)
        assumptions.append(
            (
                self._reference_assumption(
                    ValuationAssumptionKind.COUNTRY_RISK_PREMIUM,
                    country_premium,
                    selected_on,
                    currency,
                    company_id,
                    country_snapshot,
                    country=country_name,
                    methodology=country_premium_methodology,
                )
                if configuration.country_risk_premium is None
                and country_snapshot is not None
                and country_row is not None
                else self._explicit_assumption(
                    ValuationAssumptionKind.COUNTRY_RISK_PREMIUM,
                    country_premium,
                    selected_on,
                    currency,
                    company_id,
                    provider="valuation-profile",
                )
            )
        )

        cost_of_equity = configuration.cost_of_equity
        if cost_of_equity is None:
            cost_of_equity = DiscountRateService.cost_of_equity(
                risk_free, beta, equity_premium, country_premium
            ).cost_of_equity
            cost_equity_assumption = ValuationAssumption(
                kind=ValuationAssumptionKind.COST_OF_EQUITY,
                value=cost_of_equity,
                unit=AssumptionUnit.PERCENTAGE_POINTS,
                selected_on=selected_on,
                currency=currency,
                company_id=company_id,
                provenance=AssumptionProvenance(
                    origin=AssumptionOrigin.DERIVED,
                    provider="edgarito",
                    methodology="CAPM plus country risk premium",
                ),
            )
        else:
            cost_equity_assumption = self._explicit_assumption(
                ValuationAssumptionKind.COST_OF_EQUITY,
                cost_of_equity,
                selected_on,
                currency,
                company_id,
                provider="valuation-profile",
            )
        assumptions.append(cost_equity_assumption)

        debt_cost = configuration.pretax_cost_of_debt
        if debt_cost is not None:
            debt_cost_assumption = self._explicit_assumption(
                ValuationAssumptionKind.PRETAX_COST_OF_DEBT,
                debt_cost,
                selected_on,
                currency,
                company_id,
                provider="valuation-profile",
            )
        else:
            debt_cost, debt_date = self._historical_cost_of_debt(
                financials,
                capital_bridge.gross_debt,
                capital_bridge.fiscal_year,
                currency,
            )
            if debt_cost is not None:
                debt_cost_assumption = self._historical_assumption(
                    ValuationAssumptionKind.PRETAX_COST_OF_DEBT,
                    debt_cost,
                    selected_on,
                    currency,
                    company_id,
                    debt_date,
                    "Latest annual interest expense divided by latest gross debt",
                )
            else:
                sovereign_spread = (
                    country_row.adjusted_default_spread
                    if country_row is not None
                    else Decimal(0)
                )
                debt_cost = risk_free + sovereign_spread
                debt_cost_assumption = ValuationAssumption(
                    kind=ValuationAssumptionKind.PRETAX_COST_OF_DEBT,
                    value=debt_cost,
                    unit=AssumptionUnit.PERCENTAGE_POINTS,
                    selected_on=selected_on,
                    currency=currency,
                    company_id=company_id,
                    provenance=AssumptionProvenance(
                        origin=AssumptionOrigin.DERIVED,
                        provider="edgarito",
                        methodology=(
                            "Long-term risk-free rate plus Damodaran sovereign default "
                            "spread proxy; override for issuer-specific credit risk"
                        ),
                    ),
                )
        assumptions.append(debt_cost_assumption)

        wacc_result = DiscountRateService.wacc(
            cost_of_equity,
            debt_cost,
            tax_rate,
            market_equity,
            market_debt,
        )
        assumptions.append(
            ValuationAssumption(
                kind=ValuationAssumptionKind.WACC,
                value=wacc_result.wacc,
                unit=AssumptionUnit.PERCENTAGE_POINTS,
                selected_on=selected_on,
                currency=currency,
                company_id=company_id,
                provenance=AssumptionProvenance(
                    origin=AssumptionOrigin.DERIVED,
                    provider="edgarito",
                    methodology=(
                        f"Market-value WACC; equity={market_equity}, debt={market_debt}, "
                        f"equity weight={wacc_result.equity_weight}"
                    ),
                ),
            )
        )
        return wacc_result.wacc, assumptions

    @staticmethod
    def _market_country_premium(
        country: CountryRiskPremium,
        mature_market_premium: Decimal,
    ) -> tuple[Decimal, str]:
        """Prefer a market-observed sovereign-CDS premium when available."""
        cds_total = country.cds_equity_risk_premium
        if cds_total is not None and cds_total >= mature_market_premium:
            return (
                cds_total - mature_market_premium,
                "Sovereign-CDS total ERP minus mature-market ERP",
            )
        return country.country_risk_premium, "Rating-based country equity risk premium"

    @staticmethod
    def _is_mature_market_base(country_name: str | None) -> bool:
        normalized = re.sub(r"[^a-z0-9]", "", (country_name or "").casefold())
        return normalized in {"unitedstates", "unitedstatesofamerica", "usa", "us"}

    def _derive_terminal_growth(
        self,
        *,
        wacc,
        selected_on,
        currency,
        company_id,
        inflation_series,
        risk_free_series,
    ) -> tuple[Decimal, ValuationAssumption]:
        ceiling = wacc - Decimal("0.5")
        if ceiling <= 0:
            raise ValueError(
                "Automatic terminal growth requires WACC above 0.5%; provide "
                "--terminal-growth for this case"
            )
        if inflation_series is not None:
            observations = sorted(
                inflation_series.observations,
                key=lambda item: item.period_end,
                reverse=True,
            )[:60]
            raw = Decimal(median([item.value for item in observations]))
            growth = min(max(raw, Decimal(0)), Decimal("3"), ceiling)
            latest = max(observations, key=lambda item: item.period_end)
            methodology = (
                f"Median of the latest {len(observations)} {inflation_series.name} "
                "observations, bounded to 0%-3% and at least 0.5pp below WACC"
            )
            provenance = AssumptionProvenance(
                origin=AssumptionOrigin.DERIVED,
                provider=inflation_series.provider,
                dataset=inflation_series.name,
                series_id=inflation_series.series_id,
                version=inflation_series.source_version,
                observed_on=latest.period_end,
                retrieved_at=inflation_series.retrieved_at,
                methodology=methodology,
            )
        elif risk_free_series is not None:
            latest = risk_free_series.latest_observation
            growth = min(
                max(latest.value - Decimal(1), Decimal(0)),
                Decimal("2.5"),
                ceiling,
            )
            methodology = (
                "Long-term sovereign yield minus 1pp, bounded to 0%-2.5% and "
                "at least 0.5pp below WACC; inflation series unavailable"
            )
            provenance = AssumptionProvenance(
                origin=AssumptionOrigin.DERIVED,
                provider=risk_free_series.provider,
                dataset=risk_free_series.name,
                series_id=risk_free_series.series_id,
                version=risk_free_series.source_version,
                observed_on=latest.period_end,
                retrieved_at=risk_free_series.retrieved_at,
                methodology=methodology,
            )
        else:
            raise ValueError(
                "Automatic terminal growth requires an inflation or long-term "
                "sovereign-yield series; provide --terminal-growth or set it in the profile"
            )
        return growth, ValuationAssumption(
            kind=ValuationAssumptionKind.TERMINAL_GROWTH,
            value=growth,
            unit=AssumptionUnit.PERCENTAGE_POINTS,
            selected_on=selected_on,
            currency=currency,
            company_id=company_id,
            provenance=provenance,
        )

    @staticmethod
    def _configured_or_market_rate(
        configured,
        kind,
        series,
        selected_on,
        currency,
        company_id,
    ):
        if configured is not None:
            return configured, ValuationAssumptionResolver._explicit_assumption(
                kind,
                configured,
                selected_on,
                currency,
                company_id,
                provider="valuation-profile",
            )
        if series is None:
            raise ValueError(
                "Automatic WACC requires a long-term risk-free rate; set "
                "valuation.discount_rates.risk_free_rate in the profile"
            )
        latest = series.latest_observation
        return latest.value, ValuationAssumption(
            kind=kind,
            value=latest.value,
            unit=AssumptionUnit.PERCENTAGE_POINTS,
            selected_on=selected_on,
            currency=currency,
            company_id=company_id,
            provenance=AssumptionProvenance(
                origin=AssumptionOrigin.MARKET_OBSERVATION,
                provider=series.provider,
                dataset=series.name,
                series_id=series.series_id,
                version=series.source_version,
                observed_on=latest.period_end,
                retrieved_at=series.retrieved_at,
                methodology="Latest available 10-year sovereign yield",
            ),
        )

    @staticmethod
    def _historical_tax_rate(financials):
        annual = ValuationAssumptionResolver._annual_by_year(financials)
        values = []
        dates = []
        for year in sorted(annual, reverse=True):
            pretax = annual[year].get(FinancialConcept.PRETAX_INCOME)
            tax = annual[year].get(FinancialConcept.INCOME_TAX_EXPENSE)
            if pretax and tax and pretax.value > 0 and pretax.unit == tax.unit:
                rate = abs(tax.value) / pretax.value * Decimal(100)
                if Decimal(0) <= rate <= Decimal(100):
                    values.append(rate)
                    dates.append(max(pretax.period_end, tax.period_end))
            if len(values) == 3:
                break
        return (Decimal(median(values)), max(dates)) if values else (None, None)

    @staticmethod
    def _historical_cost_of_debt(financials, gross_debt, fiscal_year, currency):
        if gross_debt is None or gross_debt <= 0:
            return None, None
        interest = [
            item
            for item in financials.observations
            if item.granularity == Granularity.ANNUAL
            and item.fiscal_period == FiscalPeriod.FY
            and item.concept == FinancialConcept.INTEREST_EXPENSE
            and item.fiscal_year == fiscal_year
            and item.unit == currency
            and item.value != 0
        ]
        if not interest:
            return None, None
        latest = max(interest, key=lambda item: item.period_end)
        rate = abs(latest.value) / gross_debt * Decimal(100)
        if rate > Decimal(30):
            return None, None
        return rate, latest.period_end

    @staticmethod
    def _annual_by_year(financials):
        result: dict[int, dict[FinancialConcept, FinancialObservation]] = {}
        for item in financials.observations:
            if (
                item.granularity == Granularity.ANNUAL
                and item.fiscal_period == FiscalPeriod.FY
            ):
                result.setdefault(item.fiscal_year, {}).setdefault(item.concept, item)
        return result

    @classmethod
    def _industry_row(cls, snapshot, name) -> Optional[IndustryBeta]:
        if snapshot is None or not name:
            return None
        target = cls._normalized_label(name)
        target = cls._INDUSTRY_ALIASES.get(target, target)
        rows = {
            cls._normalized_label(item.industry): item for item in snapshot.industries
        }
        if target in rows:
            return rows[target]
        matches = [item for key, item in rows.items() if target in key or key in target]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _country_row(snapshot, name) -> Optional[CountryRiskPremium]:
        if snapshot is None or not name:
            return None
        aliases = {
            "usa": "unitedstates",
            "unitedstatesofamerica": "unitedstates",
            "uk": "unitedkingdom",
        }
        target = ValuationAssumptionResolver._normalized_label(name)
        target = aliases.get(target, target)
        return next(
            (
                item
                for item in snapshot.countries
                if ValuationAssumptionResolver._normalized_label(item.country) == target
            ),
            None,
        )

    @staticmethod
    def _normalized_label(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold())

    @staticmethod
    def _explicit_assumption(
        kind,
        value,
        selected_on,
        currency,
        company_id,
        *,
        provider,
        unit=AssumptionUnit.PERCENTAGE_POINTS,
    ):
        return ValuationAssumption(
            kind=kind,
            value=value,
            unit=unit,
            selected_on=selected_on,
            currency=currency,
            company_id=company_id,
            provenance=AssumptionProvenance(
                origin=AssumptionOrigin.EXPLICIT,
                provider=provider,
                methodology="User-supplied assumption",
            ),
        )

    @staticmethod
    def _historical_assumption(
        kind,
        value,
        selected_on,
        currency,
        company_id,
        observed_on,
        methodology,
    ):
        return ValuationAssumption(
            kind=kind,
            value=value,
            unit=AssumptionUnit.PERCENTAGE_POINTS,
            selected_on=selected_on,
            currency=currency,
            company_id=company_id,
            provenance=AssumptionProvenance(
                origin=AssumptionOrigin.HISTORICAL_METRIC,
                provider="company-financials",
                observed_on=observed_on,
                methodology=methodology,
            ),
        )

    @staticmethod
    def _reference_assumption(
        kind,
        value,
        selected_on,
        currency,
        company_id,
        snapshot,
        *,
        country=None,
        industry=None,
        methodology,
        unit=AssumptionUnit.PERCENTAGE_POINTS,
    ):
        metadata = snapshot.metadata
        return ValuationAssumption(
            kind=kind,
            value=value,
            unit=unit,
            selected_on=selected_on,
            currency=currency,
            country=country,
            industry=industry,
            company_id=company_id,
            provenance=AssumptionProvenance(
                origin=AssumptionOrigin.REFERENCE_DATASET,
                provider=metadata.provider,
                dataset=metadata.dataset,
                version=metadata.version,
                observed_on=metadata.published_on,
                retrieved_at=metadata.retrieved_at,
                methodology=methodology,
            ),
        )


__all__ = ["ResolvedDcfAssumptions", "ValuationAssumptionResolver"]
