"""Canonical FCFF observation adapter for complete operating economics years."""

from __future__ import annotations

import calendar
import datetime
from decimal import Decimal

from edgarito.schemas.forecasting import (
    FcffForecast,
    FcffForecastDcfStub,
    FcffForecastObservation,
    ForecastValue,
)
from edgarito.schemas.operating import (
    CompanyOperatingEconomicsForecast,
    operating_units_compatible,
)

from .contracts import _HUNDRED, _UNAVAILABLE


class DriverBasedCanonicalFcffAdapter:
    """Map complete operating-economics years to canonical FCFF observations."""

    def observations(
        self,
        economics: CompanyOperatingEconomicsForecast,
        template: FcffForecast | None = None,
        *,
        base_period_end: datetime.date | None = None,
        fiscal_year_end: datetime.date | None = None,
        base_revenue: Decimal | None = None,
    ) -> tuple[FcffForecastObservation, ...]:
        if not isinstance(economics, CompanyOperatingEconomicsForecast):
            economics = CompanyOperatingEconomicsForecast.model_validate(economics)
        if template is not None and not isinstance(template, FcffForecast):
            template = FcffForecast.model_validate(template)
        if template is not None:
            self._validate_compatibility(economics, template)
        elif base_period_end is None and fiscal_year_end is None:
            raise ValueError(
                "Standalone canonical FCFF observation mapping requires an "
                "explicit base_period_end or fiscal_year_end anchor"
            )
        existing = {
            item.fiscal_year: item
            for item in (template.observations if template else ())
        }
        result = []
        for index, year in enumerate(economics.fiscal_years):
            revenue = economics.consolidated_revenue[index]
            ebit = economics.consolidated_ebit[index]
            nopat = economics.nopat[index]
            da = economics.depreciation_and_amortization[index]
            capex = economics.capital_expenditures[index]
            owc = economics.operating_working_capital[index]
            delta = economics.change_in_operating_working_capital[index]
            fcff = economics.fcff[index]
            tax_rate = economics.tax_rate[index]
            if revenue is None or revenue <= 0 or any(
                value is None
                for value in (ebit, nopat, tax_rate, da, capex, owc, delta, fcff)
            ):
                continue
            source = existing.get(year)
            derived_period_end = self._period_end(
                year,
                template=template,
                base_period_end=base_period_end,
                fiscal_year_end=fiscal_year_end,
            )
            period_end = (
                source.period_end
                if source is not None and source.period_end == derived_period_end
                else derived_period_end
            )
            previous_revenue = (
                economics.consolidated_revenue[index - 1]
                if index > 0
                else template.ytd_anchor.latest_annual_revenue
                if template is not None
                and template.ytd_anchor is not None
                else template.base_revenue
                if template is not None
                else base_revenue
            )
            if previous_revenue is None or previous_revenue <= 0:
                raise ValueError(
                    "Canonical FCFF observation mapping requires a truthful "
                    "base revenue anchor for first-year revenue growth"
                )
            revenue_growth = (
                (revenue - previous_revenue) / previous_revenue * _HUNDRED
                if previous_revenue not in (None, Decimal(0))
                else Decimal(0)
            )
            operating_margin = ebit / revenue * _HUNDRED
            da_ratio = da / revenue * _HUNDRED
            capex_ratio = capex / revenue * _HUNDRED
            owc_ratio = owc / revenue * _HUNDRED
            values = {
                "revenue_growth": revenue_growth,
                "revenue": revenue,
                "operating_margin": operating_margin,
                "operating_income": ebit,
                "tax_rate": tax_rate,
                "nopat": nopat,
                "depreciation_and_amortization": da,
                "depreciation_to_revenue": da_ratio,
                "capital_expenditures": capex,
                "capex_to_revenue": capex_ratio,
                "operating_working_capital": owc,
                "operating_working_capital_to_revenue": owc_ratio,
                "change_in_operating_working_capital": delta,
                "fcff": fcff,
            }
            output_unit = (
                source.unit
                if source is not None
                else template.unit
                if template is not None
                else economics.unit
            )
            result.append(
                FcffForecastObservation(
                    forecast_year=source.forecast_year if source else index + 1,
                    fiscal_year=year,
                    period_end=period_end,
                    revenue_growth=revenue_growth,
                    revenue=revenue,
                    operating_margin=operating_margin,
                    operating_income=ebit,
                    tax_rate=tax_rate,
                    nopat=nopat,
                    depreciation_to_revenue=da_ratio,
                    depreciation_and_amortization=da,
                    capex_to_revenue=capex_ratio,
                    capital_expenditures=capex,
                    operating_working_capital_to_revenue=owc_ratio,
                    operating_working_capital=owc,
                    change_in_operating_working_capital=delta,
                    fcff=fcff,
                    unit=output_unit,
                    cell_audits=self._audits(economics, year, values),
                )
            )
        return tuple(result)

    def adapt(
        self,
        economics: CompanyOperatingEconomicsForecast,
        template: FcffForecast | None = None,
        *,
        canonical_forecast: FcffForecast | None = None,
        canonical_template: FcffForecast | None = None,
        base_period_end: datetime.date | None = None,
        fiscal_year_end: datetime.date | None = None,
        base_revenue: Decimal | None = None,
    ):
        template = template or canonical_forecast or canonical_template
        if template is not None and not isinstance(template, FcffForecast):
            template = FcffForecast.model_validate(template)
        mapped = self.observations(
            economics,
            template,
            base_period_end=base_period_end,
            fiscal_year_end=fiscal_year_end,
            base_revenue=base_revenue,
        )
        if template is None:
            return mapped
        updates = {"observations": list(mapped)}
        if template.ytd_anchor is not None:
            updates["dcf_stub"] = self._remaining_stub(template, mapped)
        return FcffForecast.model_validate(
            template.model_copy(update=updates).model_dump(mode="python")
        )

    map = observations
    adapt_observations = observations
    to_canonical_forecast = adapt

    @staticmethod
    def _period_end(
        fiscal_year: int,
        *,
        template: FcffForecast | None,
        base_period_end: datetime.date | None,
        fiscal_year_end: datetime.date | None,
    ) -> datetime.date:
        anchor = (
            template.ytd_anchor.fiscal_year_end
            if template is not None and template.ytd_anchor is not None
            else fiscal_year_end
            if fiscal_year_end is not None
            else template.fiscal_year_end
            if template is not None and template.fiscal_year_end is not None
            else template.base_period_end
            if template is not None
            else base_period_end
        )
        if anchor is None:
            raise ValueError(
                "Canonical FCFF observation mapping requires a period-end anchor"
            )
        try:
            return anchor.replace(year=fiscal_year)
        except ValueError:
            return anchor.replace(
                year=fiscal_year,
                day=calendar.monthrange(fiscal_year, anchor.month)[1],
            )

    @staticmethod
    def _validate_compatibility(
        economics: CompanyOperatingEconomicsForecast,
        template: FcffForecast,
    ) -> None:
        if not operating_units_compatible(template.unit, economics.unit):
            raise ValueError(
                "Canonical FCFF adapter template and economics units are incompatible"
            )
        # The operating contract uses the literal ``company`` only as an
        # intentionally generic fixture/alias.  Any two concrete issuer IDs
        # must match; the returned template remains the metadata authority for
        # the generic-alias case.
        if (
            template.company_id != economics.company_id
            and template.company_id != "company"
            and economics.company_id != "company"
        ):
            raise ValueError(
                "Canonical FCFF adapter template and economics company_id values "
                "are incompatible"
            )
        template_currency = self_currency_code(template.unit)
        economics_currency = self_currency_code(economics.unit)
        if (
            template_currency is not None
            and economics_currency is not None
            and template_currency != economics_currency
        ):
            raise ValueError(
                "Canonical FCFF adapter template and economics units/currencies are incompatible"
            )
        years = tuple(economics.fiscal_years)
        if len(years) != template.parameters.forecast_years:
            raise ValueError(
                "Canonical FCFF adapter template and economics horizons are incompatible"
            )
        if template.observations:
            template_years = tuple(item.fiscal_year for item in template.observations)
            if template_years != years:
                raise ValueError(
                    "Canonical FCFF adapter template observations do not match the economics horizon"
                )
        else:
            first_year = (
                template.current_fiscal_year
                if template.current_fiscal_year is not None
                else template.base_fiscal_year + 1
            )
            if years != tuple(first_year + index for index in range(len(years))):
                raise ValueError(
                    "Canonical FCFF adapter economics years do not match the template horizon"
                )

    @staticmethod
    def _remaining_stub(template: FcffForecast, mapped) -> FcffForecastDcfStub | None:
        anchor = template.ytd_anchor
        if anchor is None or not mapped or mapped[0].fiscal_year != anchor.fiscal_year:
            return None
        first = mapped[0]
        if first.period_end <= anchor.ytd_period_end:
            return None
        actual_ytd_nopat = (
            anchor.actual_operating_income * (Decimal(1) - anchor.actual_tax_rate / _HUNDRED)
            if anchor.actual_tax_rate is not None
            # Keep the fallback identical to FcffDcfService's read-only
            # validation: when actual tax is unavailable, use the resolved
            # first-year anchor rate rather than reconstructing NOPAT from a
            # potentially differently signed tax expense.
            else anchor.actual_operating_income
            * (Decimal(1) - anchor.tax_rate / _HUNDRED)
        )
        fcff = (
            first.nopat
            - actual_ytd_nopat
            + first.depreciation_and_amortization
            - anchor.actual_depreciation_and_amortization
            - first.capital_expenditures
            + anchor.actual_capital_expenditures
            - first.operating_working_capital
            + anchor.actual_operating_working_capital
        )
        old_stub = template.dcf_stub
        return FcffForecastDcfStub(
            forecast_year=first.forecast_year,
            fiscal_year=first.fiscal_year,
            period_start=anchor.ytd_period_end,
            period_end=first.period_end,
            unit=first.unit,
            annual_nopat=first.nopat,
            actual_ytd_nopat=actual_ytd_nopat,
            annual_depreciation_and_amortization=first.depreciation_and_amortization,
            actual_ytd_depreciation_and_amortization=anchor.actual_depreciation_and_amortization,
            annual_capital_expenditures=first.capital_expenditures,
            actual_ytd_capital_expenditures=anchor.actual_capital_expenditures,
            fiscal_year_end_operating_working_capital=first.operating_working_capital,
            actual_ytd_operating_working_capital=anchor.actual_operating_working_capital,
            fcff=fcff,
            formula=old_stub.formula if old_stub is not None else FcffForecastDcfStub.model_fields["formula"].default,
        )

    @staticmethod
    def _audits(economics, year, values):
        result = {}
        for field, value in values.items():
            if value is None:
                continue
            source_field = {
                "operating_margin": "ebit",
                "operating_income": "ebit",
                "tax_rate": "tax_rate",
                "nopat": "nopat",
                "depreciation_and_amortization": "depreciation_and_amortization",
                "capital_expenditures": "capital_expenditures",
                "operating_working_capital": "operating_working_capital",
                "change_in_operating_working_capital": "change_in_operating_working_capital",
                "fcff": "fcff",
                "depreciation_to_revenue": "depreciation_and_amortization",
                "capex_to_revenue": "capital_expenditures",
                "operating_working_capital_to_revenue": "operating_working_capital",
            }.get(field)
            if source_field is None:
                source = economics.source_by_year.get(year, _UNAVAILABLE)
                method = economics.method_by_year.get(year, "economics_stage_output")
                confidence = economics.confidence_by_year.get(year, "low")
            else:
                source = getattr(
                    economics, f"{source_field}_source_by_year", {}
                ).get(year, _UNAVAILABLE)
                method = getattr(
                    economics, f"{source_field}_method_by_year", {}
                ).get(year, "economics_stage_output")
                confidence = getattr(
                    economics, f"{source_field}_confidence_by_year", {}
                ).get(year, "low")
            provenance = getattr(
                economics,
                f"{source_field or field}_provenance_by_year",
                {},
            ).get(year)
            if provenance is not None:
                method = f"{method}; provenance={provenance}"
            if field in {
                "depreciation_to_revenue",
                "capex_to_revenue",
                "operating_working_capital_to_revenue",
            }:
                numerator = {
                    "depreciation_to_revenue": values["depreciation_and_amortization"],
                    "capex_to_revenue": values["capital_expenditures"],
                    "operating_working_capital_to_revenue": values["operating_working_capital"],
                }[field]
                method = (
                    f"{method}; ratio_numerator={numerator}; "
                    f"ratio_denominator={values['revenue']}; ratio={value}"
                )
            result[field] = ForecastValue(
                value=value,
                source=source,
                method=method,
                confidence=confidence,
            )
        return result


__all__ = ["DriverBasedCanonicalFcffAdapter"]


def self_currency_code(unit: str | None) -> str | None:
    folded = (unit or "").casefold()
    for code in ("usd", "eur", "gbp", "jpy", "cny", "cad", "aud", "chf"):
        if code in folded:
            return code
    return None
