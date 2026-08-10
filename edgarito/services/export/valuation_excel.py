"""Render an FCFF forecast and its DCF result as an editable Excel model."""

from __future__ import annotations

import datetime
import math
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import xlsxwriter
from xlsxwriter.utility import xl_col_to_name

from edgarito.schemas.valuation.relative import ProviderNeutralRelativeValuation
from edgarito.services.forecasting.models import FcffForecast, ForecastSeedType
from edgarito.services.valuation.discounting import (
    PresentValueService,
    TerminalValueService,
)
from edgarito.services.valuation.models import (
    CashFlowTiming,
    ComparableImpliedValuation,
    ComparableMultiplesReport,
    FcffDcfResult,
    MultipleStatus,
    RelativeValuationBasis,
    TerminalMetric,
    TerminalValueMethod,
)

DiscountTimingBasis = Literal["forecast_year", "calendar"]


class ValuationExcelRenderer:
    """Write a linked, parameterized FCFF DCF workbook.

    The renderer intentionally does not recalculate a valuation in Python.  It
    copies the already-resolved forecast and DCF result into formula cached
    values while putting the model's operating and valuation assumptions in
    yellow input cells.  Excel recalculates the formulas when the workbook is
    opened or an input is changed.
    """

    _SUMMARY = "Valuation Summary"
    _INPUTS = "FCFF Inputs"
    _FORECAST = "FCFF Forecast"
    _DCF = "DCF Calculation"
    _YELLOW = "#FFFF00"

    _DRIVERS = (
        ("revenue_growth", "Revenue Growth", "% (percentage points)"),
        ("operating_margin", "Operating Margin", "% (percentage points)"),
        ("tax_rate", "Tax Rate", "% (percentage points)"),
        (
            "depreciation_to_revenue",
            "D&A / Revenue",
            "% (percentage points)",
        ),
        ("capex_to_revenue", "Capex / Revenue", "% (percentage points)"),
        (
            "operating_working_capital_to_revenue",
            "Operating NWC / Revenue",
            "% (percentage points)",
        ),
    )
    _DRIVER_LABELS = {name: label for name, label, _ in _DRIVERS}

    def render(
        self,
        forecast: FcffForecast,
        result: FcffDcfResult,
        output: str | Path,
        *,
        report: Any | None = None,
        canonical_report: Any | None = None,
        relative: ComparableImpliedValuation
        | ProviderNeutralRelativeValuation
        | None = None,
        relative_result: ComparableImpliedValuation
        | ProviderNeutralRelativeValuation
        | None = None,
        peer_report: ComparableMultiplesReport | None = None,
        discount_timing_basis: DiscountTimingBasis | str | None = None,
    ) -> Path:
        """Render ``forecast`` and ``result`` to ``output``.

        ``report`` and ``canonical_report`` are accepted as optional context
        for callers that already have a canonical analysis snapshot.  The FCFF
        forecast and DCF result remain the source of all calculation values.
        """
        if not isinstance(forecast, FcffForecast):
            raise TypeError("forecast must be an FcffForecast")
        if not isinstance(result, FcffDcfResult):
            raise TypeError("result must be an FcffDcfResult")
        if not forecast.observations:
            raise ValueError("FCFF valuation export requires forecast observations")
        if report is not None and canonical_report is not None:
            raise TypeError("report and canonical_report are mutually exclusive")
        if relative is not None and relative_result is not None:
            raise TypeError("relative and relative_result are mutually exclusive")
        relative_result = relative if relative is not None else relative_result

        self._validate_export_inputs(forecast, result)
        timing_basis = self._validate_result_alignment(
            forecast,
            result,
            discount_timing_basis,
        )
        self._validate_forecast_observations(forecast)

        output_path = Path(output).expanduser()
        if output_path.exists() and output_path.is_dir():
            raise ValueError(f"Excel output path is a directory: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        context = canonical_report if canonical_report is not None else report
        workbook = xlsxwriter.Workbook(str(output_path))
        try:
            workbook.set_properties(
                {
                    "title": "Edgarito FCFF DCF valuation export",
                    "subject": "Parameterized FCFF discounted cash-flow valuation",
                    "author": "Edgarito",
                }
            )
            workbook.set_calc_mode("auto")
            formats = self._formats(workbook)
            summary_worksheet = workbook.add_worksheet(self._SUMMARY)
            input_refs = self._write_inputs(workbook, forecast, result, formats)
            forecast_refs = self._write_forecast(
                workbook, forecast, input_refs, formats
            )
            dcf_refs = self._write_dcf(
                workbook,
                forecast,
                result,
                input_refs,
                forecast_refs,
                formats,
                timing_basis,
            )
            self._write_summary(
                forecast,
                result,
                dcf_refs,
                context,
                formats,
                worksheet=summary_worksheet,
            )
            if relative_result is not None or peer_report is not None:
                self._write_relative_valuation(
                    relative_result,
                    peer_report,
                    formats,
                    workbook=workbook,
                )
        finally:
            workbook.close()
        return output_path

    @classmethod
    def _validate_export_inputs(
        cls,
        forecast: FcffForecast,
        result: FcffDcfResult,
    ) -> None:
        """Reject inputs that cannot be represented by the workbook formulas.

        The valuation services validate these values during normal execution, but
        an export can also receive model instances assembled by callers.  Keeping
        this check before the output path is created prevents a partial workbook
        when such an instance contains an invalid assumption.
        """
        seed_type = getattr(forecast.seed_type, "value", forecast.seed_type)
        if seed_type == ForecastSeedType.YTD_PLUS_FORECAST.value:
            if forecast.ytd_anchor is None:
                raise ValueError(
                    "Excel valuation export requires YTD anchor metadata for "
                    "ForecastSeedType.YTD_PLUS_FORECAST; forecast the YTD anchor "
                    "with FcffForecastService before exporting"
                )
            cls._validate_ytd_anchor(forecast)

        parameter = result.parameters
        cls._require_finite(parameter.wacc, "WACC")
        if parameter.wacc <= Decimal("-100"):
            raise ValueError("WACC must be greater than -100% for Excel export")
        if parameter.terminal_method == TerminalValueMethod.PERPETUITY_GROWTH:
            if parameter.perpetual_growth_rate is None:
                raise ValueError(
                    "Perpetuity-growth Excel export requires a finite terminal "
                    "growth rate; the input is blank or invalid"
                )
            cls._require_finite(parameter.perpetual_growth_rate, "terminal growth")
            if parameter.wacc <= parameter.perpetual_growth_rate:
                raise ValueError(
                    "WACC must exceed terminal growth for perpetuity-growth Excel "
                    "export"
                )
        elif parameter.terminal_method == TerminalValueMethod.EXIT_MULTIPLE:
            if parameter.exit_multiple is None:
                raise ValueError(
                    "Exit-multiple Excel export requires a finite exit multiple; "
                    "the input is blank or invalid"
                )
            cls._require_finite(parameter.exit_multiple, "exit multiple")
            if parameter.exit_multiple < 0:
                raise ValueError("Exit multiple must be nonnegative for Excel export")
        else:
            raise ValueError(
                "Excel valuation export requires a supported terminal method"
            )

        bridge = result.capital_bridge
        for field in ("net_debt", "non_operating_assets", "diluted_shares"):
            cls._require_finite(getattr(bridge, field), field.replace("_", " "))
        if bridge.non_operating_assets < 0:
            raise ValueError(
                "Non-operating assets must be nonnegative for Excel export"
            )
        if bridge.diluted_shares <= 0:
            raise ValueError(
                "Diluted shares must be positive for Excel export; zero or negative "
                "shares would make per-share value invalid"
            )

        result_values = (
            ("enterprise value", result.enterprise_value),
            ("equity value", result.equity_value),
            ("value per share", result.value_per_share),
            (
                "explicit forecast PV",
                result.explicit_forecast_present_value.total_present_value,
            ),
            ("terminal value", result.terminal_value.terminal_value),
            ("terminal PV", result.terminal_present_value.present_value),
        )
        for label, value in result_values:
            cls._require_finite(value, label)
        if result.terminal_value.terminal_metric is not None:
            cls._require_finite(
                result.terminal_value.terminal_metric,
                "terminal metric",
            )

        for cash_flow in result.explicit_forecast_present_value.cash_flows:
            for field in (
                "amount",
                "period",
                "discount_rate",
                "discount_factor",
                "present_value",
            ):
                cls._require_finite(
                    getattr(cash_flow, field),
                    f"explicit cash flow {field}",
                )
        for field in (
            "amount",
            "period",
            "discount_rate",
            "discount_factor",
            "present_value",
        ):
            cls._require_finite(
                getattr(result.terminal_present_value, field),
                f"terminal cash flow {field}",
            )

    @staticmethod
    def _require_finite(value: Any, label: str) -> None:
        if not isinstance(value, (Decimal, int, float)) or not math.isfinite(
            float(value)
        ):
            raise ValueError(f"Excel valuation export requires finite {label}")

    @classmethod
    def _validate_ytd_anchor(cls, forecast: FcffForecast) -> None:
        anchor = forecast.ytd_anchor
        if anchor is None:
            return
        if not forecast.observations:
            raise ValueError("YTD anchor metadata requires forecast observations")
        first = forecast.observations[0]
        if anchor.fiscal_year != first.fiscal_year:
            raise ValueError(
                "YTD anchor fiscal year must match the first forecast observation"
            )
        if anchor.fiscal_year_end != first.period_end:
            raise ValueError(
                "YTD anchor fiscal year end must match the first forecast period end"
            )
        if anchor.ytd_period_end >= anchor.fiscal_year_end:
            raise ValueError("YTD anchor period must precede its fiscal year end")

        numeric_fields = (
            "actual_revenue",
            "actual_operating_income",
            "actual_pretax_income",
            "actual_income_tax_expense",
            "actual_depreciation_and_amortization",
            "actual_capital_expenditures",
            "actual_operating_working_capital",
            "latest_annual_revenue",
            "revenue_growth",
            "operating_margin",
            "tax_rate",
            "depreciation_to_revenue",
            "capex_to_revenue",
            "operating_working_capital_to_revenue",
        )
        for field in numeric_fields:
            cls._require_finite(getattr(anchor, field), f"YTD anchor {field}")
        if anchor.actual_tax_rate is not None:
            cls._require_finite(anchor.actual_tax_rate, "YTD anchor actual tax rate")
            if not Decimal(0) <= anchor.actual_tax_rate <= Decimal(100):
                raise ValueError(
                    "YTD anchor actual tax rate must be between 0% and 100%"
                )
        if anchor.actual_revenue <= 0 or anchor.latest_annual_revenue <= 0:
            raise ValueError("YTD anchor revenues must be positive")
        if anchor.revenue_anchor is not None:
            cls._require_finite(anchor.revenue_anchor, "YTD anchor revenue anchor")
            if anchor.revenue_anchor < anchor.actual_revenue:
                raise ValueError(
                    "YTD anchor revenue anchor cannot be below actual YTD revenue"
                )
        if not Decimal(0) <= anchor.tax_rate <= Decimal(100):
            raise ValueError("YTD anchor tax rate must be between 0% and 100%")
        if not Decimal(0) <= anchor.depreciation_to_revenue <= Decimal(500):
            raise ValueError("YTD anchor D&A / revenue must be between 0% and 500%")
        if not Decimal(0) <= anchor.capex_to_revenue <= Decimal(500):
            raise ValueError("YTD anchor capex / revenue must be between 0% and 500%")
        if not Decimal("-100") < anchor.revenue_growth <= Decimal("1000"):
            raise ValueError(
                "YTD anchor revenue growth must be greater than -100% and at most 1000%"
            )
        if abs(anchor.operating_margin) > Decimal(500):
            raise ValueError(
                "YTD anchor operating margin must be between -500% and 500%"
            )
        if abs(anchor.operating_working_capital_to_revenue) > Decimal(500):
            raise ValueError(
                "YTD anchor operating NWC / revenue must be between -500% and 500%"
            )

    def export(
        self,
        forecast: FcffForecast,
        result: FcffDcfResult,
        output: str | Path,
        *,
        report: Any | None = None,
        canonical_report: Any | None = None,
        relative: ComparableImpliedValuation
        | ProviderNeutralRelativeValuation
        | None = None,
        relative_result: ComparableImpliedValuation
        | ProviderNeutralRelativeValuation
        | None = None,
        peer_report: ComparableMultiplesReport | None = None,
        discount_timing_basis: DiscountTimingBasis | str | None = None,
    ) -> Path:
        """Alias for :meth:`render` for export-service style callers."""
        return self.render(
            forecast,
            result,
            output,
            report=report,
            canonical_report=canonical_report,
            relative=relative,
            relative_result=relative_result,
            peer_report=peer_report,
            discount_timing_basis=discount_timing_basis,
        )

    def __call__(
        self,
        forecast: FcffForecast,
        result: FcffDcfResult,
        output: str | Path,
        *,
        report: Any | None = None,
        canonical_report: Any | None = None,
        relative: ComparableImpliedValuation
        | ProviderNeutralRelativeValuation
        | None = None,
        relative_result: ComparableImpliedValuation
        | ProviderNeutralRelativeValuation
        | None = None,
        peer_report: ComparableMultiplesReport | None = None,
        discount_timing_basis: DiscountTimingBasis | str | None = None,
    ) -> Path:
        return self.render(
            forecast,
            result,
            output,
            report=report,
            canonical_report=canonical_report,
            relative=relative,
            relative_result=relative_result,
            peer_report=peer_report,
            discount_timing_basis=discount_timing_basis,
        )

    @classmethod
    def _formats(cls, workbook: xlsxwriter.Workbook) -> dict[str, Any]:
        return {
            "title": workbook.add_format(
                {"bold": True, "font_size": 14, "font_color": "#1F4E78"}
            ),
            "section": workbook.add_format(
                {
                    "bold": True,
                    "bg_color": "#D9EAF7",
                    "font_color": "#1F1F1F",
                    "border": 1,
                }
            ),
            "header": workbook.add_format(
                {
                    "bold": True,
                    "bg_color": "#1F4E78",
                    "font_color": "#FFFFFF",
                    "border": 1,
                    "text_wrap": True,
                }
            ),
            "label": workbook.add_format({"bold": True}),
            "text": workbook.add_format({}),
            "source": workbook.add_format({"font_color": "#666666", "text_wrap": True}),
            "number": workbook.add_format(
                {"num_format": "0.############################"}
            ),
            "percent": workbook.add_format({"num_format": "0.00"}),
            "date": workbook.add_format({"num_format": "yyyy-mm-dd"}),
            "datetime": workbook.add_format({"num_format": "yyyy-mm-dd hh:mm:ss"}),
            "formula": workbook.add_format(
                {
                    "font_color": "#0000FF",
                    "num_format": "0.############################",
                }
            ),
            "formula_percent": workbook.add_format(
                {"font_color": "#0000FF", "num_format": "0.00"}
            ),
            "formula_date": workbook.add_format(
                {"font_color": "#0000FF", "num_format": "yyyy-mm-dd"}
            ),
            "input": workbook.add_format(
                {
                    "bg_color": cls._YELLOW,
                    "border": 1,
                    "num_format": "0.############################",
                }
            ),
            "input_percent": workbook.add_format(
                {"bg_color": cls._YELLOW, "border": 1, "num_format": "0.00"}
            ),
            "input_text": workbook.add_format({"bg_color": cls._YELLOW, "border": 1}),
            "input_date": workbook.add_format(
                {"bg_color": cls._YELLOW, "border": 1, "num_format": "yyyy-mm-dd"}
            ),
            "warning": workbook.add_format(
                {"font_color": "#9C0006", "text_wrap": True}
            ),
        }

    @classmethod
    def _validate_forecast_observations(cls, forecast: FcffForecast) -> None:
        """Ensure formula caches agree with the workbook's forecast formulas."""
        is_ytd = getattr(forecast.seed_type, "value", forecast.seed_type) == (
            ForecastSeedType.YTD_PLUS_FORECAST.value
        )
        ytd_anchor = forecast.ytd_anchor if is_ytd else None
        if is_ytd:
            cls._validate_ytd_anchor(forecast)
        required_base_numeric_fields = (
            "base_revenue",
            "base_operating_income",
            "base_tax_rate",
            "base_nopat",
            "base_depreciation_and_amortization",
            "base_capital_expenditures",
            "base_operating_working_capital",
        )
        for field in required_base_numeric_fields:
            value = getattr(forecast, field)
            if value is None or not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"Forecast {field} must be finite for Excel export")

        if forecast.base_fcff is not None and (
            not isinstance(forecast.base_fcff, Decimal)
            or not forecast.base_fcff.is_finite()
        ):
            raise ValueError("Forecast base_fcff must be finite for Excel export")

        previous_revenue = forecast.base_revenue
        previous_working_capital = forecast.base_operating_working_capital
        expected_years = range(1, len(forecast.observations) + 1)
        for expected_year, observation in zip(
            expected_years, forecast.observations, strict=True
        ):
            if observation.forecast_year != expected_year:
                raise ValueError(
                    "FCFF forecast years must be consecutive and start at one; "
                    f"observation FY{observation.fiscal_year} has forecast year "
                    f"{observation.forecast_year}, expected {expected_year}"
                )

            driver_fields = tuple(name for name, _, _ in cls._DRIVERS)
            numeric_fields = driver_fields + (
                "revenue",
                "operating_income",
                "nopat",
                "depreciation_and_amortization",
                "capital_expenditures",
                "operating_working_capital",
                "change_in_operating_working_capital",
                "fcff",
            )
            for field in numeric_fields:
                value = getattr(observation, field)
                if not isinstance(value, Decimal) or not value.is_finite():
                    raise ValueError(
                        f"Forecast observation FY{observation.fiscal_year} {field} "
                        "must be finite for Excel export"
                    )
            cls._validate_driver_ranges(observation)

            if expected_year == 1 and ytd_anchor is not None:
                growth_driver = ytd_anchor.revenue_growth
                operating_margin_driver = ytd_anchor.operating_margin
                tax_rate_driver = ytd_anchor.tax_rate
                depreciation_ratio_driver = ytd_anchor.depreciation_to_revenue
                capex_ratio_driver = ytd_anchor.capex_to_revenue
                revenue_anchor = ytd_anchor.revenue_anchor
                expected_revenue = revenue_anchor or max(
                    ytd_anchor.actual_revenue,
                    ytd_anchor.latest_annual_revenue
                    * (Decimal(1) + growth_driver / Decimal(100)),
                )
                remaining_revenue = expected_revenue - ytd_anchor.actual_revenue
                expected_operating_income = (
                    ytd_anchor.actual_operating_income
                    + remaining_revenue * operating_margin_driver / Decimal(100)
                )
                actual_tax_rate = ytd_anchor.actual_tax_rate
                if actual_tax_rate is None:
                    actual_tax_rate = tax_rate_driver
                actual_nopat = ytd_anchor.actual_operating_income * (
                    Decimal(1) - actual_tax_rate / Decimal(100)
                )
                projected_nopat = (
                    remaining_revenue
                    * operating_margin_driver
                    / Decimal(100)
                    * (Decimal(1) - tax_rate_driver / Decimal(100))
                )
                expected_nopat = actual_nopat + projected_nopat
                expected_depreciation = (
                    ytd_anchor.actual_depreciation_and_amortization
                    + remaining_revenue * depreciation_ratio_driver / Decimal(100)
                )
                expected_capex = (
                    ytd_anchor.actual_capital_expenditures
                    + remaining_revenue * capex_ratio_driver / Decimal(100)
                )
                expected_revenue_growth = (
                    expected_revenue / ytd_anchor.latest_annual_revenue - Decimal(1)
                ) * Decimal(100)
                expected_operating_margin = (
                    expected_operating_income / expected_revenue * Decimal(100)
                    if expected_revenue != 0
                    else operating_margin_driver
                )
                expected_tax_rate = (
                    (Decimal(1) - expected_nopat / expected_operating_income)
                    * Decimal(100)
                    if expected_operating_income > 0
                    else tax_rate_driver
                )
                expected_depreciation_ratio = (
                    expected_depreciation / expected_revenue * Decimal(100)
                )
                expected_capex_ratio = expected_capex / expected_revenue * Decimal(100)
            else:
                growth_driver = observation.revenue_growth
                operating_margin_driver = observation.operating_margin
                tax_rate_driver = observation.tax_rate
                depreciation_ratio_driver = observation.depreciation_to_revenue
                capex_ratio_driver = observation.capex_to_revenue
                expected_revenue = previous_revenue * (
                    Decimal(1) + growth_driver / Decimal(100)
                )
                expected_operating_income = (
                    expected_revenue * operating_margin_driver / Decimal(100)
                )
                expected_nopat = expected_operating_income * (
                    Decimal(1) - tax_rate_driver / Decimal(100)
                )
                expected_depreciation = (
                    expected_revenue * depreciation_ratio_driver / Decimal(100)
                )
                expected_capex = expected_revenue * capex_ratio_driver / Decimal(100)
                expected_revenue_growth = observation.revenue_growth
                expected_operating_margin = observation.operating_margin
                expected_tax_rate = observation.tax_rate
                expected_depreciation_ratio = observation.depreciation_to_revenue
                expected_capex_ratio = observation.capex_to_revenue
            expected_working_capital = (
                expected_revenue
                * observation.operating_working_capital_to_revenue
                / Decimal(100)
            )
            expected_change_in_working_capital = (
                expected_working_capital - previous_working_capital
            )
            expected_fcff = (
                expected_nopat
                + expected_depreciation
                - expected_capex
                - expected_change_in_working_capital
            )
            expected_values = {
                "revenue": expected_revenue,
                "operating_income": expected_operating_income,
                "nopat": expected_nopat,
                "depreciation_and_amortization": expected_depreciation,
                "capital_expenditures": expected_capex,
                "operating_working_capital": expected_working_capital,
                "change_in_operating_working_capital": (
                    expected_change_in_working_capital
                ),
                "fcff": expected_fcff,
                "revenue_growth": expected_revenue_growth,
                "operating_margin": expected_operating_margin,
                "tax_rate": expected_tax_rate,
                "depreciation_to_revenue": expected_depreciation_ratio,
                "capex_to_revenue": expected_capex_ratio,
            }
            for field, expected in expected_values.items():
                cls._assert_close(
                    field,
                    getattr(observation, field),
                    expected,
                    context=f"forecast observation FY{observation.fiscal_year}",
                )
            previous_revenue = expected_revenue
            previous_working_capital = expected_working_capital

    @staticmethod
    def _validate_driver_ranges(observation) -> None:
        checks = (
            (
                "revenue_growth",
                observation.revenue_growth > Decimal("-100")
                and observation.revenue_growth <= Decimal("1000"),
                "must be greater than -100% and at most 1000%",
            ),
            (
                "operating_margin",
                abs(observation.operating_margin) <= Decimal("500"),
                "must be between -500% and 500%",
            ),
            (
                "tax_rate",
                Decimal(0) <= observation.tax_rate <= Decimal(100),
                "must be between 0% and 100%",
            ),
            (
                "depreciation_to_revenue",
                Decimal(0) <= observation.depreciation_to_revenue <= Decimal(500),
                "must be between 0% and 500%",
            ),
            (
                "capex_to_revenue",
                Decimal(0) <= observation.capex_to_revenue <= Decimal(500),
                "must be between 0% and 500%",
            ),
            (
                "operating_working_capital_to_revenue",
                abs(observation.operating_working_capital_to_revenue) <= Decimal(500),
                "must be between -500% and 500%",
            ),
        )
        for field, valid, rule in checks:
            if not valid:
                raise ValueError(
                    f"Forecast observation FY{observation.fiscal_year} {field} "
                    f"{rule} for Excel export"
                )

    @classmethod
    def _validate_result_alignment(
        cls,
        forecast: FcffForecast,
        result: FcffDcfResult,
        requested_basis: DiscountTimingBasis | str | None,
    ) -> DiscountTimingBasis:
        if result.unit != forecast.unit:
            raise ValueError("Forecast and DCF result must use one currency")
        explicit_result = result.explicit_forecast_present_value
        if explicit_result.unit != forecast.unit:
            raise ValueError(
                "Forecast and explicit DCF cash flows must use one currency"
            )
        if len(explicit_result.cash_flows) != len(forecast.observations):
            raise ValueError(
                "Forecast/result alignment failed: forecast has "
                f"{len(forecast.observations)} observations but DCF result has "
                f"{len(explicit_result.cash_flows)} explicit cash flows"
            )

        basis = cls._resolve_timing_basis(forecast, result, requested_basis)
        timing_offset = cls._timing_offset(result)
        for index, (observation, cash_flow) in enumerate(
            zip(forecast.observations, explicit_result.cash_flows, strict=True),
            start=1,
        ):
            cls._assert_close(
                "cash-flow amount",
                cash_flow.amount,
                observation.fcff,
                context=f"forecast/result period {index}",
            )
            expected_period = cls._explicit_period(
                observation,
                result,
                basis,
                timing_offset,
            )
            cls._assert_close(
                "discount period",
                cash_flow.period,
                expected_period,
                context=f"forecast/result period {index}",
            )
            expected_discount_factor = PresentValueService.discount_factor(
                result.parameters.wacc,
                expected_period,
            )
            cls._assert_close(
                "discount rate",
                cash_flow.discount_rate,
                result.parameters.wacc,
                context=f"forecast/result period {index}",
            )
            cls._assert_close(
                "discount factor",
                cash_flow.discount_factor,
                expected_discount_factor,
                context=f"forecast/result period {index}",
            )
            cls._assert_close(
                "present value",
                cash_flow.present_value,
                cash_flow.amount * expected_discount_factor,
                context=f"forecast/result period {index}",
            )

        if result.terminal_value.method != result.parameters.terminal_method:
            raise ValueError(
                "Forecast/result alignment failed: terminal method does not match "
                "the DCF parameters"
            )
        final = forecast.observations[-1]
        if result.parameters.terminal_method == TerminalValueMethod.EXIT_MULTIPLE:
            expected_metric = cls._terminal_metric(final, result.parameters.exit_metric)
            if result.terminal_value.terminal_metric is None:
                raise ValueError(
                    "Forecast/result alignment failed: exit-multiple result is "
                    "missing its terminal metric"
                )
            cls._assert_close(
                "terminal metric",
                result.terminal_value.terminal_metric,
                expected_metric,
                context="terminal result",
            )
            assert result.parameters.exit_multiple is not None
            expected_terminal_value = TerminalValueService.exit_multiple(
                expected_metric,
                result.parameters.exit_multiple,
            ).terminal_value
        else:
            if result.parameters.perpetual_growth_rate is None:
                raise ValueError(
                    "Forecast/result alignment failed: perpetuity-growth result is "
                    "missing perpetual growth"
                )
            expected_terminal_value = TerminalValueService.perpetuity_growth(
                final.fcff,
                result.parameters.wacc,
                result.parameters.perpetual_growth_rate,
            ).terminal_value
        cls._assert_close(
            "terminal value",
            result.terminal_value.terminal_value,
            expected_terminal_value,
            context="terminal result",
        )
        cls._assert_close(
            "terminal cash-flow amount",
            result.terminal_present_value.amount,
            result.terminal_value.terminal_value,
            context="terminal result",
        )
        expected_terminal_period = cls._terminal_period(forecast, result, basis)
        cls._assert_close(
            "terminal discount period",
            result.terminal_present_value.period,
            expected_terminal_period,
            context="terminal result",
        )
        expected_terminal_discount_factor = PresentValueService.discount_factor(
            result.parameters.wacc,
            expected_terminal_period,
        )
        cls._assert_close(
            "terminal discount rate",
            result.terminal_present_value.discount_rate,
            result.parameters.wacc,
            context="terminal result",
        )
        cls._assert_close(
            "terminal discount factor",
            result.terminal_present_value.discount_factor,
            expected_terminal_discount_factor,
            context="terminal result",
        )
        cls._assert_close(
            "terminal present value",
            result.terminal_present_value.present_value,
            result.terminal_value.terminal_value * expected_terminal_discount_factor,
            context="terminal result",
        )
        return basis

    @classmethod
    def _resolve_timing_basis(
        cls,
        forecast: FcffForecast,
        result: FcffDcfResult,
        requested_basis: DiscountTimingBasis | str | None,
    ) -> DiscountTimingBasis:
        normalized = None
        if requested_basis is not None:
            normalized = str(requested_basis).strip().lower()
            normalized = {
                "forecast_years": "forecast_year",
                "forecast-years": "forecast_year",
                "forecast": "forecast_year",
                "calendar_dates": "calendar",
                "calendar_date": "calendar",
                "calendar-days": "calendar",
            }.get(normalized, normalized)
            if normalized not in {"forecast_year", "calendar"}:
                raise ValueError(
                    "discount_timing_basis must be 'forecast_year' or 'calendar'"
                )

        timing_offset = cls._timing_offset(result)
        forecast_matches = all(
            cls._decimal_close(
                cash_flow.period,
                Decimal(observation.forecast_year) - timing_offset,
            )
            for observation, cash_flow in zip(
                forecast.observations,
                result.explicit_forecast_present_value.cash_flows,
                strict=True,
            )
        )
        calendar_matches = all(
            cls._decimal_close(
                cash_flow.period,
                cls._year_fraction(result.valuation_date, observation.period_end)
                - timing_offset,
            )
            for observation, cash_flow in zip(
                forecast.observations,
                result.explicit_forecast_present_value.cash_flows,
                strict=True,
            )
        )
        if normalized is not None:
            if (normalized == "forecast_year" and not forecast_matches) or (
                normalized == "calendar" and not calendar_matches
            ):
                raise ValueError(
                    "Forecast/result discount periods do not match the requested "
                    f"{normalized} timing basis"
                )
            return normalized  # type: ignore[return-value]
        if forecast_matches and not calendar_matches:
            return "forecast_year"
        if calendar_matches and not forecast_matches:
            return "calendar"
        if forecast_matches and calendar_matches:
            # A valuation date equal to the base date can make both bases have
            # identical values. The service's no-date path is forecast-year based;
            # callers with an explicit date can pass ``discount_timing_basis``.
            return (
                "forecast_year"
                if result.valuation_date == forecast.base_period_end
                else "calendar"
            )
        raise ValueError(
            "Forecast/result discount periods match neither forecast-year nor "
            "calendar-date timing"
        )

    @classmethod
    def _explicit_period(
        cls,
        observation,
        result: FcffDcfResult,
        basis: DiscountTimingBasis,
        timing_offset: Decimal,
    ) -> Decimal:
        if basis == "forecast_year":
            return Decimal(observation.forecast_year) - timing_offset
        return cls._year_fraction(result.valuation_date, observation.period_end) - (
            timing_offset
        )

    @classmethod
    def _terminal_period(
        cls,
        forecast: FcffForecast,
        result: FcffDcfResult,
        basis: DiscountTimingBasis,
    ) -> Decimal:
        if basis == "forecast_year":
            return Decimal(forecast.observations[-1].forecast_year)
        return cls._year_fraction(
            result.valuation_date,
            forecast.observations[-1].period_end,
        )

    @staticmethod
    def _timing_offset(result: FcffDcfResult) -> Decimal:
        return (
            Decimal("0.5")
            if result.parameters.cash_flow_timing == CashFlowTiming.MID_YEAR
            else Decimal(0)
        )

    @staticmethod
    def _year_fraction(start: datetime.date, end: datetime.date) -> Decimal:
        return Decimal((end - start).days) / Decimal(365)

    @staticmethod
    def _terminal_metric(observation, metric: TerminalMetric) -> Decimal:
        if metric == TerminalMetric.EBITDA:
            return (
                observation.operating_income + observation.depreciation_and_amortization
            )
        if metric == TerminalMetric.EBIT:
            return observation.operating_income
        if metric == TerminalMetric.FCFF:
            return observation.fcff
        return observation.revenue

    @staticmethod
    def _decimal_close(left: Decimal, right: Decimal) -> bool:
        scale = max(abs(left), abs(right), Decimal(1))
        return abs(left - right) <= scale * Decimal("1e-24")

    @classmethod
    def _assert_close(
        cls,
        field: str,
        supplied: Decimal,
        expected: Decimal,
        *,
        context: str,
    ) -> None:
        if not cls._decimal_close(supplied, expected):
            raise ValueError(
                f"{context} {field} differs from the workbook formula: supplied "
                f"{supplied}, expected {expected}"
            )

    def _write_inputs(
        self,
        workbook: xlsxwriter.Workbook,
        forecast: FcffForecast,
        result: FcffDcfResult,
        formats: dict[str, Any],
    ) -> dict[str, Any]:
        worksheet = workbook.add_worksheet(self._INPUTS)
        worksheet.freeze_panes(5, 2)
        worksheet.write("A1", "FCFF Inputs", formats["title"])
        worksheet.write(
            "A3",
            "Yellow cells are user-editable. Rates are percentage points (8 means 8%). "
            "Blank or invalid numeric inputs make guarded formulas return #N/A.",
            formats["source"],
        )
        headers = (
            "Forecast Year",
            "Period End",
            "Revenue Growth",
            "Operating Margin",
            "Tax Rate",
            "D&A / Revenue",
            "Capex / Revenue",
            "Operating NWC / Revenue",
            "Source / provenance",
        )
        for column, header in enumerate(headers):
            worksheet.write(4, column, header, formats["header"])

        driver_columns = {
            name: 2 + index for index, (name, _, _) in enumerate(self._DRIVERS)
        }
        driver_sources = {
            **getattr(result, "forecast_assumption_sources", {}),
            **{
                driver.value: source.value
                for driver, source in forecast.assumption_sources.items()
            },
        }
        forecast_rows: list[int] = []
        anchor = forecast.ytd_anchor

        def driver_input_value(index: int, name: str, observation) -> Decimal:
            if index == 0 and anchor is not None and hasattr(anchor, name):
                return getattr(anchor, name)
            path = getattr(forecast.parameters, name)
            if path is None:
                return getattr(observation, name)
            return path[index] if index < len(path) else path[-1]

        for index, observation in enumerate(forecast.observations):
            row = 5 + index
            forecast_rows.append(row)
            worksheet.write_number(row, 0, observation.forecast_year, formats["number"])
            self._write_date(worksheet, row, 1, observation.period_end, formats["date"])
            for driver_index, (name, _, _) in enumerate(self._DRIVERS):
                value = driver_input_value(index, name, observation)
                worksheet.write_number(
                    row,
                    2 + driver_index,
                    self._number(value),
                    formats["input_percent"],
                )
            notes = driver_sources
            worksheet.write_string(
                row,
                8,
                "; ".join(
                    f"{label}: {notes.get(name, 'forecast output')}"
                    for name, label, _ in self._DRIVERS
                ),
                formats["source"],
            )

        anchor_rows: dict[str, int] = {}
        assumptions_start = 5 + len(forecast.observations) + 2
        if anchor is not None:
            anchor_start = assumptions_start
            worksheet.write(
                anchor_start - 1,
                0,
                "YTD+forecast anchor (editable)",
                formats["section"],
            )
            for column, header in enumerate(
                ("Anchor input", "Value (editable)", "Units", "Source / provenance")
            ):
                worksheet.write(anchor_start, column, header, formats["header"])

            def anchor_input_row(
                key: str,
                label: str,
                value: Any,
                units: str,
                value_kind: str = "number",
            ) -> None:
                row = anchor_start + 1 + len(anchor_rows)
                anchor_rows[key] = row
                worksheet.write(row, 0, label, formats["label"])
                if value_kind == "date":
                    self._write_date(worksheet, row, 1, value, formats["input_date"])
                elif value_kind == "text":
                    worksheet.write_string(row, 1, str(value), formats["input_text"])
                elif value is None:
                    worksheet.write_blank(
                        row,
                        1,
                        None,
                        formats["input_percent" if units.startswith("%") else "input"],
                    )
                else:
                    worksheet.write_number(
                        row,
                        1,
                        self._number(value),
                        formats["input_percent" if units.startswith("%") else "input"],
                    )
                worksheet.write(row, 2, units, formats["text"])
                worksheet.write(
                    row, 3, "FcffForecastService YTD anchor", formats["source"]
                )

            anchor_input_row(
                "fiscal_year", "Forecast Fiscal Year", anchor.fiscal_year, "year"
            )
            anchor_input_row(
                "ytd_period_end",
                "Actual YTD Period End",
                anchor.ytd_period_end,
                "date",
                "date",
            )
            anchor_input_row(
                "fiscal_year_end",
                "Fiscal Year End",
                anchor.fiscal_year_end,
                "date",
                "date",
            )
            anchor_input_row(
                "actual_quarters",
                "Actual Quarters Included",
                anchor.actual_quarters,
                "quarters",
            )
            anchor_input_row(
                "actual_revenue",
                "Actual YTD Revenue",
                anchor.actual_revenue,
                forecast.unit,
            )
            anchor_input_row(
                "actual_operating_income",
                "Actual YTD Operating Income",
                anchor.actual_operating_income,
                forecast.unit,
            )
            anchor_input_row(
                "actual_tax_rate",
                "Actual YTD Effective Tax Rate",
                anchor.actual_tax_rate,
                "% (percentage points)",
            )
            anchor_input_row(
                "actual_pretax_income",
                "Actual YTD Pretax Income",
                anchor.actual_pretax_income,
                forecast.unit,
            )
            anchor_input_row(
                "actual_income_tax_expense",
                "Actual YTD Income Tax Expense",
                anchor.actual_income_tax_expense,
                forecast.unit,
            )
            anchor_input_row(
                "actual_depreciation_and_amortization",
                "Actual YTD D&A",
                anchor.actual_depreciation_and_amortization,
                forecast.unit,
            )
            anchor_input_row(
                "actual_capital_expenditures",
                "Actual YTD Capital Expenditures",
                anchor.actual_capital_expenditures,
                forecast.unit,
            )
            anchor_input_row(
                "actual_operating_working_capital",
                "Actual YTD Operating Working Capital",
                anchor.actual_operating_working_capital,
                forecast.unit,
            )
            anchor_input_row(
                "latest_annual_revenue",
                "Latest Annual Revenue",
                anchor.latest_annual_revenue,
                forecast.unit,
            )
            anchor_input_row(
                "revenue_anchor",
                "Revenue Anchor (optional)",
                anchor.revenue_anchor,
                forecast.unit,
            )
            assumptions_start = anchor_start + len(anchor_rows) + 3

        worksheet.write(assumptions_start - 1, 0, "DCF assumptions", formats["section"])
        for column, header in enumerate(
            ("Input", "Value (editable)", "Units", "Source / provenance")
        ):
            worksheet.write(assumptions_start, column, header, formats["header"])

        parameter = result.parameters
        bridge = result.capital_bridge
        assumption_rows: dict[str, int] = {}

        def input_row(
            key: str,
            label: str,
            value: Any,
            units: str,
            source: str,
            value_kind: str = "number",
        ) -> None:
            row = assumptions_start + 1 + len(assumption_rows)
            assumption_rows[key] = row
            worksheet.write(row, 0, label, formats["label"])
            if value_kind == "date":
                if value is None:
                    worksheet.write_blank(row, 1, None, formats["input_date"])
                else:
                    self._write_date(worksheet, row, 1, value, formats["input_date"])
            elif value_kind == "text":
                if value is None:
                    worksheet.write_blank(row, 1, None, formats["input_text"])
                else:
                    worksheet.write_string(row, 1, str(value), formats["input_text"])
            elif value is None:
                worksheet.write_blank(
                    row,
                    1,
                    None,
                    formats["input_percent" if units.startswith("%") else "input"],
                )
            else:
                worksheet.write_number(
                    row,
                    1,
                    self._number(value),
                    formats["input_percent" if units.startswith("%") else "input"],
                )
            worksheet.write(row, 2, units, formats["text"])
            worksheet.write(row, 3, source, formats["source"])

        input_row(
            "valuation_date",
            "Valuation Date",
            result.valuation_date,
            "date",
            "valuation run date",
            "date",
        )
        input_row(
            "wacc",
            "WACC",
            parameter.wacc,
            "% (percentage points)",
            parameter.wacc_source,
            "number",
        )
        input_row(
            "cash_flow_timing",
            "Cash-flow Timing",
            parameter.cash_flow_timing.value,
            "enum",
            "valuation profile / CLI parameter",
            "text",
        )
        input_row(
            "terminal_method",
            "Terminal Method",
            parameter.terminal_method.value,
            "enum",
            "valuation profile / CLI parameter",
            "text",
        )
        input_row(
            "perpetual_growth_rate",
            "Perpetual Growth",
            parameter.perpetual_growth_rate,
            "% (percentage points)",
            parameter.perpetual_growth_source or "inactive for exit multiple",
            "number",
        )
        input_row(
            "exit_multiple",
            "Exit Multiple",
            parameter.exit_multiple,
            "multiple",
            "valuation profile / CLI parameter",
            "number",
        )
        input_row(
            "exit_metric",
            "Exit Metric",
            parameter.exit_metric.value,
            "enum",
            "valuation profile / CLI parameter",
            "text",
        )
        input_row(
            "net_debt",
            "Net Debt",
            bridge.net_debt,
            bridge.unit,
            bridge.net_debt_source,
            "number",
        )
        input_row(
            "non_operating_assets",
            "Non-operating Assets",
            bridge.non_operating_assets,
            bridge.unit,
            bridge.non_operating_assets_source,
            "number",
        )
        input_row(
            "diluted_shares",
            "Diluted Shares",
            bridge.diluted_shares,
            "shares",
            bridge.shares_source,
            "number",
        )

        enum_validations = {
            "cash_flow_timing": (
                [item.value for item in CashFlowTiming],
                "Cash-flow timing",
            ),
            "terminal_method": (
                [item.value for item in TerminalValueMethod],
                "terminal method",
            ),
            "exit_metric": (
                [item.value for item in TerminalMetric],
                "exit metric",
            ),
        }
        for key, (values, label) in enum_validations.items():
            row = assumption_rows[key]
            worksheet.data_validation(
                row,
                1,
                row,
                1,
                {
                    "validate": "list",
                    "source": values,
                    "ignore_blank": False,
                    "input_title": f"Select {label}",
                    "input_message": f"Choose a valid {label}.",
                    "error_title": f"Invalid {label}",
                    "error_message": f"Select a value from the {label} dropdown.",
                    "error_type": "stop",
                },
            )

        first_forecast_row = forecast_rows[0]
        last_forecast_row = forecast_rows[-1]
        driver_validation_rules = {
            "revenue_growth": (
                "AND(ISNUMBER({cell}),{cell}>-100,{cell}<=1000)",
                "Revenue growth must be greater than -100% and at most 1000%.",
            ),
            "operating_margin": (
                "AND(ISNUMBER({cell}),{cell}>=-500,{cell}<=500)",
                "Operating margin must be between -500% and 500%.",
            ),
            "tax_rate": (
                "AND(ISNUMBER({cell}),{cell}>=0,{cell}<=100)",
                "Tax rate must be between 0% and 100%.",
            ),
            "depreciation_to_revenue": (
                "AND(ISNUMBER({cell}),{cell}>=0,{cell}<=500)",
                "D&A / Revenue must be between 0% and 500%.",
            ),
            "capex_to_revenue": (
                "AND(ISNUMBER({cell}),{cell}>=0,{cell}<=500)",
                "Capex / Revenue must be between 0% and 500%.",
            ),
            "operating_working_capital_to_revenue": (
                "AND(ISNUMBER({cell}),{cell}>=-500,{cell}<=500)",
                "Operating NWC / Revenue must be between -500% and 500%.",
            ),
        }
        for index, (name, _, _) in enumerate(self._DRIVERS):
            column = 2 + index
            rule, message = driver_validation_rules[name]
            cell = f"{xl_col_to_name(column)}{first_forecast_row + 1}"
            worksheet.data_validation(
                first_forecast_row,
                column,
                last_forecast_row,
                column,
                {
                    "validate": "custom",
                    "value": f"={rule.format(cell=cell)}",
                    "ignore_blank": False,
                    "input_title": f"Valid {self._DRIVER_LABELS[name]}",
                    "input_message": message,
                    "error_title": f"Invalid {self._DRIVER_LABELS[name]}",
                    "error_message": message,
                    "error_type": "stop",
                },
            )

        def local_ref(key: str) -> str:
            return f"$B${assumption_rows[key] + 1}"

        numeric_validations = {
            "wacc": {
                "validate": "decimal",
                "criteria": ">",
                "value": -100,
                "ignore_blank": False,
                "error_message": "WACC must be greater than -100%.",
            },
            "perpetual_growth_rate": {
                "validate": "custom",
                "value": (
                    f'=OR({local_ref("perpetual_growth_rate")}="",'
                    f"ISNUMBER({local_ref('perpetual_growth_rate')}))"
                ),
                "ignore_blank": False,
                "error_message": "Terminal growth must be numeric or blank when inactive.",
            },
            "exit_multiple": {
                "validate": "custom",
                "value": (
                    f'=OR({local_ref("exit_multiple")}="",AND('
                    f"ISNUMBER({local_ref('exit_multiple')})"
                    f",{local_ref('exit_multiple')}>=0))"
                ),
                "ignore_blank": False,
                "error_message": "Exit multiple must be numeric and nonnegative.",
            },
            "net_debt": {
                "validate": "custom",
                "value": f"=ISNUMBER({local_ref('net_debt')})",
                "ignore_blank": False,
                "error_message": "Net debt must be numeric.",
            },
            "non_operating_assets": {
                "validate": "decimal",
                "criteria": ">=",
                "value": 0,
                "ignore_blank": False,
                "error_message": "Non-operating assets must be nonnegative.",
            },
            "diluted_shares": {
                "validate": "decimal",
                "criteria": ">",
                "value": 0,
                "ignore_blank": False,
                "error_message": "Diluted shares must be positive.",
            },
        }
        for key, validation in numeric_validations.items():
            row = assumption_rows[key]
            worksheet.data_validation(
                row,
                1,
                row,
                1,
                {
                    **validation,
                    "input_title": f"Valid {key.replace('_', ' ')}",
                    "input_message": validation["error_message"],
                    "error_title": "Invalid numeric input",
                    "error_type": "stop",
                },
            )

        if anchor is not None:
            actual_revenue_ref = self._ref(
                self._INPUTS, anchor_rows["actual_revenue"], 1
            )
            revenue_anchor_row = anchor_rows["revenue_anchor"]
            worksheet.data_validation(
                revenue_anchor_row,
                1,
                revenue_anchor_row,
                1,
                {
                    "validate": "custom",
                    "value": (
                        f'=OR({self._ref(self._INPUTS, revenue_anchor_row, 1)}="",'
                        f"AND(ISNUMBER({self._ref(self._INPUTS, revenue_anchor_row, 1)}),"
                        f"{self._ref(self._INPUTS, revenue_anchor_row, 1)}>="
                        f"{actual_revenue_ref},"
                        f"{self._ref(self._INPUTS, revenue_anchor_row, 1)}>0))"
                    ),
                    "ignore_blank": False,
                    "input_title": "Valid revenue anchor",
                    "input_message": (
                        "Optional FY revenue anchor must be numeric, positive, and "
                        "at least actual YTD revenue."
                    ),
                    "error_title": "Invalid revenue anchor",
                    "error_message": (
                        "Revenue anchor must be blank or at least actual YTD revenue."
                    ),
                    "error_type": "stop",
                },
            )
            quarters_row = anchor_rows["actual_quarters"]
            worksheet.data_validation(
                quarters_row,
                1,
                quarters_row,
                1,
                {
                    "validate": "integer",
                    "criteria": "between",
                    "minimum": 1,
                    "maximum": 3,
                    "ignore_blank": False,
                    "input_title": "Valid actual quarters",
                    "input_message": "Actual quarters must be between 1 and 3.",
                    "error_title": "Invalid actual quarters",
                    "error_message": "Actual quarters must be between 1 and 3.",
                    "error_type": "stop",
                },
            )

        worksheet.set_column("A:A", 25)
        worksheet.set_column("B:B", 18)
        worksheet.set_column("C:H", 18)
        worksheet.set_column("I:I", 66)
        worksheet.set_column("J:J", 3)
        worksheet.autofilter(4, 0, 4 + len(forecast.observations), 8)

        return {
            "forecast_rows": forecast_rows,
            "driver_columns": driver_columns,
            "assumption_rows": assumption_rows,
            "anchor_rows": anchor_rows,
        }

    def _write_forecast(
        self,
        workbook: xlsxwriter.Workbook,
        forecast: FcffForecast,
        input_refs: dict[str, Any],
        formats: dict[str, Any],
    ) -> dict[str, Any]:
        worksheet = workbook.add_worksheet(self._FORECAST)
        worksheet.freeze_panes(3, 2)
        worksheet.write("A1", "FCFF Forecast", formats["title"])
        worksheet.write(
            "A2",
            "Blue cells are formulas linked to the yellow FCFF Inputs cells; base values are static forecast seed values.",
            formats["source"],
        )
        forecast_columns = {
            index: 2 + index for index in range(len(forecast.observations))
        }
        headers = ["Line Item", f"Base FY{forecast.base_fiscal_year}"] + [
            f"FY{item.fiscal_year}E" for item in forecast.observations
        ]
        for column, header in enumerate(headers):
            worksheet.write(2, column, header, formats["header"])

        rows = {
            "period_end": 3,
            "revenue_growth": 4,
            "revenue": 5,
            "operating_margin": 6,
            "operating_income": 7,
            "tax_rate": 8,
            "nopat": 9,
            "depreciation_to_revenue": 10,
            "depreciation_and_amortization": 11,
            "capex_to_revenue": 12,
            "capital_expenditures": 13,
            "operating_working_capital_to_revenue": 14,
            "operating_working_capital": 15,
            "change_in_operating_working_capital": 16,
            "fcff": 17,
        }
        labels = {
            "period_end": "Period End",
            "revenue_growth": "Revenue Growth (%)",
            "revenue": "Revenue",
            "operating_margin": "Operating Margin (%)",
            "operating_income": "Operating Income",
            "tax_rate": "Tax Rate (%)",
            "nopat": "NOPAT",
            "depreciation_to_revenue": "D&A / Revenue (%)",
            "depreciation_and_amortization": "D&A",
            "capex_to_revenue": "Capex / Revenue (%)",
            "capital_expenditures": "Capital Expenditures",
            "operating_working_capital_to_revenue": "Operating NWC / Revenue (%)",
            "operating_working_capital": "Operating Working Capital",
            "change_in_operating_working_capital": "Change in Operating Working Capital",
            "fcff": "FCFF",
        }
        for key, row in rows.items():
            worksheet.write(row, 0, labels[key], formats["label"])

        base_values = {
            "period_end": forecast.base_period_end,
            "revenue_growth": None,
            "revenue": forecast.base_revenue,
            "operating_margin": self._ratio(
                forecast.base_operating_income, forecast.base_revenue
            ),
            "operating_income": forecast.base_operating_income,
            "tax_rate": forecast.base_tax_rate,
            "nopat": forecast.base_nopat,
            "depreciation_to_revenue": self._ratio(
                forecast.base_depreciation_and_amortization, forecast.base_revenue
            ),
            "depreciation_and_amortization": forecast.base_depreciation_and_amortization,
            "capex_to_revenue": self._ratio(
                forecast.base_capital_expenditures, forecast.base_revenue
            ),
            "capital_expenditures": forecast.base_capital_expenditures,
            "operating_working_capital_to_revenue": self._ratio(
                forecast.base_operating_working_capital, forecast.base_revenue
            ),
            "operating_working_capital": forecast.base_operating_working_capital,
            "change_in_operating_working_capital": None,
            "fcff": forecast.base_fcff,
        }
        for key, row in rows.items():
            value = base_values[key]
            if key == "period_end":
                self._write_date(worksheet, row, 1, value, formats["date"])
            elif value is None:
                worksheet.write_blank(row, 1, None, formats["number"])
            else:
                worksheet.write_number(
                    row,
                    1,
                    self._number(value),
                    formats["percent" if key in self._percent_rows else "number"],
                )

        driver_columns = input_refs["driver_columns"]
        input_rows = input_refs["forecast_rows"]
        anchor_rows = input_refs["anchor_rows"]
        anchor = forecast.ytd_anchor
        input_sheet = self._INPUTS

        def anchor_ref(key: str) -> str:
            return self._ref(input_sheet, anchor_rows[key], 1)

        for index, observation in enumerate(forecast.observations):
            column = 2 + index
            previous_column = column - 1
            previous_letter = xl_col_to_name(previous_column)
            current_letter = xl_col_to_name(column)
            input_row = input_rows[index]
            is_first_ytd = index == 0 and anchor is not None

            def driver_ref(name: str, *, row=input_row) -> str:
                return self._ref(input_sheet, row, driver_columns[name])

            self._write_date(
                worksheet,
                rows["period_end"],
                column,
                observation.period_end,
                formats["date"],
            )
            for name, _, _ in self._DRIVERS:
                driver_row = rows[name]
                input_driver_ref = driver_ref(name)
                if is_first_ytd:
                    output_formulas = {
                        "revenue_growth": (
                            f"=IF(AND(ISNUMBER({current_letter}{rows['revenue'] + 1}),"
                            f"ISNUMBER({anchor_ref('latest_annual_revenue')}),"
                            f"{anchor_ref('latest_annual_revenue')}<>0),"
                            f"({current_letter}{rows['revenue'] + 1}/"
                            f"{anchor_ref('latest_annual_revenue')}-1)*100,NA())"
                        ),
                        "operating_margin": (
                            f"=IF(AND(ISNUMBER({current_letter}{rows['operating_income'] + 1}),"
                            f"ISNUMBER({current_letter}{rows['revenue'] + 1}),"
                            f"{current_letter}{rows['revenue'] + 1}<>0),"
                            f"{current_letter}{rows['operating_income'] + 1}/"
                            f"{current_letter}{rows['revenue'] + 1}*100,NA())"
                        ),
                        "tax_rate": (
                            f"=IF(ISNUMBER({current_letter}{rows['operating_income'] + 1}),"
                            f"IF({current_letter}{rows['operating_income'] + 1}>0,"
                            f"(1-{current_letter}{rows['nopat'] + 1}/"
                            f"{current_letter}{rows['operating_income'] + 1})*100,"
                            f"{input_driver_ref}),NA())"
                        ),
                        "depreciation_to_revenue": (
                            f"=IF(AND(ISNUMBER({current_letter}{rows['depreciation_and_amortization'] + 1}),"
                            f"ISNUMBER({current_letter}{rows['revenue'] + 1}),"
                            f"{current_letter}{rows['revenue'] + 1}<>0),"
                            f"{current_letter}{rows['depreciation_and_amortization'] + 1}/"
                            f"{current_letter}{rows['revenue'] + 1}*100,NA())"
                        ),
                        "capex_to_revenue": (
                            f"=IF(AND(ISNUMBER({current_letter}{rows['capital_expenditures'] + 1}),"
                            f"ISNUMBER({current_letter}{rows['revenue'] + 1}),"
                            f"{current_letter}{rows['revenue'] + 1}<>0),"
                            f"{current_letter}{rows['capital_expenditures'] + 1}/"
                            f"{current_letter}{rows['revenue'] + 1}*100,NA())"
                        ),
                    }
                    formula = output_formulas.get(
                        name,
                        f"=IF({self._invalid_driver_formula(name, input_driver_ref)},"
                        f"NA(),{input_driver_ref})",
                    )
                else:
                    formula = (
                        f"=IF({self._invalid_driver_formula(name, input_driver_ref)},"
                        f"NA(),{input_driver_ref})"
                    )
                self._write_formula(
                    worksheet,
                    driver_row,
                    column,
                    formula,
                    getattr(observation, name),
                    formats["formula_percent"],
                )

            actual_tax_formula = (
                (
                    f"IF(ISNUMBER({anchor_ref('actual_tax_rate')}),"
                    f"{anchor_ref('actual_tax_rate')},"
                    f"IF(AND(ISNUMBER({anchor_ref('actual_pretax_income')}),"
                    f"{anchor_ref('actual_pretax_income')}>0,"
                    f"ISNUMBER({anchor_ref('actual_income_tax_expense')}),"
                    f"{anchor_ref('actual_income_tax_expense')}>=0,"
                    f"{anchor_ref('actual_income_tax_expense')}<="
                    f"{anchor_ref('actual_pretax_income')}),"
                    f"{anchor_ref('actual_income_tax_expense')}/"
                    f"{anchor_ref('actual_pretax_income')}*100,"
                    f"{driver_ref('tax_rate')}))"
                )
                if is_first_ytd
                else None
            )

            formulas = {
                "revenue": (
                    (
                        f"=IF(ISNUMBER({anchor_ref('revenue_anchor')}),"
                        f"IF(AND({anchor_ref('revenue_anchor')}>0,"
                        f"{anchor_ref('revenue_anchor')}>={anchor_ref('actual_revenue')}),"
                        f"{anchor_ref('revenue_anchor')},NA()),"
                        f"IF(AND(ISNUMBER({anchor_ref('actual_revenue')}),"
                        f"{anchor_ref('actual_revenue')}>0,"
                        f"ISNUMBER({anchor_ref('latest_annual_revenue')}),"
                        f"{anchor_ref('latest_annual_revenue')}>0,"
                        f"ISNUMBER({driver_ref('revenue_growth')}),"
                        f"{driver_ref('revenue_growth')}>-100,"
                        f"{driver_ref('revenue_growth')}<=1000),"
                        f"MAX({anchor_ref('actual_revenue')},"
                        f"{anchor_ref('latest_annual_revenue')}*(1+"
                        f"{driver_ref('revenue_growth')}/100)),NA()))"
                    )
                    if is_first_ytd
                    else (
                        f"={previous_letter}{rows['revenue'] + 1}*"
                        f"(1+{driver_ref('revenue_growth')}/100)"
                    )
                ),
                "operating_income": (
                    (
                        f"=IF(OR(NOT(ISNUMBER({current_letter}{rows['revenue'] + 1})),"
                        f"NOT(ISNUMBER({anchor_ref('actual_revenue')})),"
                        f"NOT(ISNUMBER({anchor_ref('actual_operating_income')})),"
                        f"NOT(ISNUMBER({driver_ref('operating_margin')}))),NA(),"
                        f"{anchor_ref('actual_operating_income')}+({current_letter}"
                        f"{rows['revenue'] + 1}-{anchor_ref('actual_revenue')})*"
                        f"{driver_ref('operating_margin')}/100)"
                    )
                    if is_first_ytd
                    else (
                        f"={current_letter}{rows['revenue'] + 1}*"
                        f"{driver_ref('operating_margin')}/100"
                    )
                ),
                "nopat": (
                    (
                        f"=IF(OR(NOT(ISNUMBER({current_letter}{rows['revenue'] + 1})),"
                        f"NOT(ISNUMBER({anchor_ref('actual_revenue')})),"
                        f"NOT(ISNUMBER({anchor_ref('actual_operating_income')})),"
                        f"NOT(ISNUMBER({driver_ref('operating_margin')})),"
                        f"NOT(ISNUMBER({driver_ref('tax_rate')}))),NA(),"
                        f"{anchor_ref('actual_operating_income')}*(1-"
                        f"{actual_tax_formula}/100)+({current_letter}{rows['revenue'] + 1}-"
                        f"{anchor_ref('actual_revenue')})*{driver_ref('operating_margin')}/100*"
                        f"(1-{driver_ref('tax_rate')}/100))"
                    )
                    if is_first_ytd
                    else (
                        f"={current_letter}{rows['operating_income'] + 1}*"
                        f"(1-{driver_ref('tax_rate')}/100)"
                    )
                ),
                "depreciation_and_amortization": (
                    (
                        f"=IF(OR(NOT(ISNUMBER({current_letter}{rows['revenue'] + 1})),"
                        f"NOT(ISNUMBER({anchor_ref('actual_revenue')})),"
                        f"NOT(ISNUMBER({anchor_ref('actual_depreciation_and_amortization')})),"
                        f"NOT(ISNUMBER({driver_ref('depreciation_to_revenue')}))),NA(),"
                        f"{anchor_ref('actual_depreciation_and_amortization')}+({current_letter}"
                        f"{rows['revenue'] + 1}-{anchor_ref('actual_revenue')})*"
                        f"{driver_ref('depreciation_to_revenue')}/100)"
                    )
                    if is_first_ytd
                    else (
                        f"={current_letter}{rows['revenue'] + 1}*"
                        f"{driver_ref('depreciation_to_revenue')}/100"
                    )
                ),
                "capital_expenditures": (
                    (
                        f"=IF(OR(NOT(ISNUMBER({current_letter}{rows['revenue'] + 1})),"
                        f"NOT(ISNUMBER({anchor_ref('actual_revenue')})),"
                        f"NOT(ISNUMBER({anchor_ref('actual_capital_expenditures')})),"
                        f"NOT(ISNUMBER({driver_ref('capex_to_revenue')}))),NA(),"
                        f"{anchor_ref('actual_capital_expenditures')}+({current_letter}"
                        f"{rows['revenue'] + 1}-{anchor_ref('actual_revenue')})*"
                        f"{driver_ref('capex_to_revenue')}/100)"
                    )
                    if is_first_ytd
                    else (
                        f"={current_letter}{rows['revenue'] + 1}*"
                        f"{driver_ref('capex_to_revenue')}/100"
                    )
                ),
                "operating_working_capital": (
                    f"={current_letter}{rows['revenue'] + 1}*"
                    f"{driver_ref('operating_working_capital_to_revenue')}/100"
                ),
                "change_in_operating_working_capital": (
                    f"={current_letter}{rows['operating_working_capital'] + 1}-"
                    f"{previous_letter}{rows['operating_working_capital'] + 1}"
                ),
                "fcff": (
                    f"={current_letter}{rows['nopat'] + 1}+"
                    f"{current_letter}{rows['depreciation_and_amortization'] + 1}-"
                    f"{current_letter}{rows['capital_expenditures'] + 1}-"
                    f"{current_letter}{rows['change_in_operating_working_capital'] + 1}"
                ),
            }
            values = {
                "revenue": observation.revenue,
                "operating_income": observation.operating_income,
                "nopat": observation.nopat,
                "depreciation_and_amortization": observation.depreciation_and_amortization,
                "capital_expenditures": observation.capital_expenditures,
                "operating_working_capital": observation.operating_working_capital,
                "change_in_operating_working_capital": observation.change_in_operating_working_capital,
                "fcff": observation.fcff,
            }
            for key, formula in formulas.items():
                self._write_formula(
                    worksheet,
                    rows[key],
                    column,
                    formula,
                    values[key],
                    formats["formula"],
                )

        worksheet.set_column("A:A", 36)
        worksheet.set_column(1, 1 + len(forecast.observations), 18)
        return {"rows": rows, "columns": forecast_columns}

    @staticmethod
    def _invalid_driver_formula(name: str, reference: str) -> str:
        if name == "revenue_growth":
            return f"OR(NOT(ISNUMBER({reference})),{reference}<=-100,{reference}>1000)"
        if name in {"operating_margin", "operating_working_capital_to_revenue"}:
            return f"OR(NOT(ISNUMBER({reference})),ABS({reference})>500)"
        if name == "tax_rate":
            return f"OR(NOT(ISNUMBER({reference})),{reference}<0,{reference}>100)"
        return f"OR(NOT(ISNUMBER({reference})),{reference}<0,{reference}>500)"

    def _write_dcf(
        self,
        workbook: xlsxwriter.Workbook,
        forecast: FcffForecast,
        result: FcffDcfResult,
        input_refs: dict[str, Any],
        forecast_refs: dict[str, Any],
        formats: dict[str, Any],
        timing_basis: DiscountTimingBasis,
    ) -> dict[str, Any]:
        worksheet = workbook.add_worksheet(self._DCF)
        worksheet.freeze_panes(3, 1)
        worksheet.write("A1", "DCF Calculation", formats["title"])
        worksheet.write(
            "A2",
            "Blue cells are Excel formulas. Cached values are the values from the Python FCFF DCF result until Excel recalculates; discount timing uses "
            f"{timing_basis.replace('_', '-')} periods. Invalid inputs return #N/A.",
            formats["source"],
        )
        forecast_count = len(forecast.observations)
        terminal_column = 1 + forecast_count
        headers = (
            ["Metric"]
            + [f"FY{item.fiscal_year}E" for item in forecast.observations]
            + ["Terminal"]
        )
        for column, header in enumerate(headers):
            worksheet.write(2, column, header, formats["header"])

        rows = {
            "period_end": 3,
            "fcff": 4,
            "discount_period": 5,
            "discount_factor": 6,
            "pv_fcff": 7,
            "terminal_metric": 8,
            "terminal_value": 9,
            "pv_terminal_value": 10,
        }
        labels = {
            "period_end": "Period End",
            "fcff": "FCFF",
            "discount_period": "Discount Period",
            "discount_factor": "Discount Factor",
            "pv_fcff": "PV of FCFF",
            "terminal_metric": "Terminal Metric",
            "terminal_value": "Terminal Value",
            "pv_terminal_value": "PV of Terminal Value",
        }
        for key, row in rows.items():
            worksheet.write(row, 0, labels[key], formats["label"])

        input_sheet = self._INPUTS
        input_rows = input_refs["assumption_rows"]

        def input_ref(key: str) -> str:
            return self._ref(input_sheet, input_rows[key], 1)

        forecast_rows = forecast_refs["rows"]
        explicit_flows = result.explicit_forecast_present_value.cash_flows

        for index, (observation, cash_flow) in enumerate(
            zip(forecast.observations, explicit_flows, strict=True)
        ):
            column = 1 + index
            column_letter = xl_col_to_name(column)
            forecast_column = 2 + index
            self._write_date(
                worksheet,
                rows["period_end"],
                column,
                observation.period_end,
                formats["date"],
            )
            self._write_formula(
                worksheet,
                rows["fcff"],
                column,
                f"={self._ref(self._FORECAST, forecast_rows['fcff'], forecast_column)}",
                observation.fcff,
                formats["formula"],
            )
            if timing_basis == "forecast_year":
                period_formula = (
                    f"={observation.forecast_year}-"
                    f'IF({input_ref("cash_flow_timing")}="mid_year",0.5,0)'
                )
            else:
                period_formula = (
                    f"=({column_letter}{rows['period_end'] + 1}-"
                    f"{input_ref('valuation_date')})/365"
                    f'-IF({input_ref("cash_flow_timing")}="mid_year",0.5,0)'
                )
            self._write_formula(
                worksheet,
                rows["discount_period"],
                column,
                period_formula,
                cash_flow.period,
                formats["formula"],
            )
            factor_formula = (
                f"=IF(OR(NOT(ISNUMBER({input_ref('wacc')})), "
                f"{input_ref('wacc')}<=-100),NA(),"
                f"1/(1+{input_ref('wacc')}/100)^{column_letter}"
                f"{rows['discount_period'] + 1})"
            )
            self._write_formula(
                worksheet,
                rows["discount_factor"],
                column,
                factor_formula,
                cash_flow.discount_factor,
                formats["formula"],
            )
            self._write_formula(
                worksheet,
                rows["pv_fcff"],
                column,
                f"={column_letter}{rows['fcff'] + 1}*{column_letter}{rows['discount_factor'] + 1}",
                cash_flow.present_value,
                formats["formula"],
            )

        terminal_letter = xl_col_to_name(terminal_column)

        def final_forecast_ref(row: str) -> str:
            return self._ref(self._FORECAST, forecast_rows[row], 1 + forecast_count)

        self._write_date(
            worksheet,
            rows["period_end"],
            terminal_column,
            forecast.observations[-1].period_end,
            formats["date"],
        )
        terminal_metric_formula = (
            f'=IF({input_ref("terminal_method")}="exit_multiple",'
            f'IF({input_ref("exit_metric")}="ebitda",'
            f"{final_forecast_ref('operating_income')}+{final_forecast_ref('depreciation_and_amortization')},"
            f'IF({input_ref("exit_metric")}="ebit",{final_forecast_ref("operating_income")},'
            f'IF({input_ref("exit_metric")}="fcff",{final_forecast_ref("fcff")},'
            f'IF({input_ref("exit_metric")}="revenue",{final_forecast_ref("revenue")},NA()))),"")'
        )
        self._write_formula(
            worksheet,
            rows["terminal_metric"],
            terminal_column,
            terminal_metric_formula,
            (
                result.terminal_value.terminal_metric
                if result.parameters.terminal_method
                == TerminalValueMethod.EXIT_MULTIPLE
                else None
            ),
            formats["formula"],
        )
        terminal_value_formula = (
            f'=IF({input_ref("terminal_method")}="perpetuity_growth",'
            f"IF(OR(NOT(ISNUMBER({input_ref('wacc')})), "
            f"NOT(ISNUMBER({input_ref('perpetual_growth_rate')})), "
            f"{input_ref('wacc')}<={input_ref('perpetual_growth_rate')}, "
            f"NOT(ISNUMBER({final_forecast_ref('fcff')}))),NA(),"
            f"{final_forecast_ref('fcff')}*(1+{input_ref('perpetual_growth_rate')}/100)/"
            f"(({input_ref('wacc')}-{input_ref('perpetual_growth_rate')})/100)),"
            f'IF({input_ref("terminal_method")}="exit_multiple",'
            f"IF(OR(NOT(ISNUMBER({input_ref('exit_multiple')})), "
            f"{input_ref('exit_multiple')}<0, "
            f"NOT(ISNUMBER({terminal_letter}{rows['terminal_metric'] + 1}))),NA(),"
            f"{terminal_letter}{rows['terminal_metric'] + 1}*"
            f"{input_ref('exit_multiple')}),NA()))"
        )
        self._write_formula(
            worksheet,
            rows["terminal_value"],
            terminal_column,
            terminal_value_formula,
            result.terminal_value.terminal_value,
            formats["formula"],
        )
        if timing_basis == "forecast_year":
            terminal_period_formula = f"={forecast.observations[-1].forecast_year}"
        else:
            terminal_period_formula = (
                f"=({terminal_letter}{rows['period_end'] + 1}-"
                f"{input_ref('valuation_date')})/365"
            )
        self._write_formula(
            worksheet,
            rows["discount_period"],
            terminal_column,
            terminal_period_formula,
            result.terminal_present_value.period,
            formats["formula"],
        )
        self._write_formula(
            worksheet,
            rows["discount_factor"],
            terminal_column,
            (
                f"=IF(OR(NOT(ISNUMBER({input_ref('wacc')})), "
                f"{input_ref('wacc')}<=-100),NA(),"
                f"1/(1+{input_ref('wacc')}/100)^{terminal_letter}"
                f"{rows['discount_period'] + 1})"
            ),
            result.terminal_present_value.discount_factor,
            formats["formula"],
        )
        worksheet.write_blank(rows["fcff"], terminal_column, None, formats["number"])
        worksheet.write_blank(rows["pv_fcff"], terminal_column, None, formats["number"])
        self._write_formula(
            worksheet,
            rows["pv_terminal_value"],
            terminal_column,
            f"={terminal_letter}{rows['terminal_value'] + 1}*{terminal_letter}{rows['discount_factor'] + 1}",
            result.terminal_present_value.present_value,
            formats["formula"],
        )

        output_rows = {
            "explicit_forecast_pv": 14,
            "terminal_pv": 15,
            "enterprise_value": 16,
            "net_debt": 17,
            "non_operating_assets": 18,
            "equity_value": 19,
            "diluted_shares": 20,
            "value_per_share": 21,
            "terminal_value_percentage": 22,
        }
        output_labels = {
            "explicit_forecast_pv": "Explicit Forecast PV",
            "terminal_pv": "Terminal PV",
            "enterprise_value": "Enterprise Value",
            "net_debt": "Net Debt",
            "non_operating_assets": "Non-operating Assets",
            "equity_value": "Equity Value",
            "diluted_shares": "Diluted Shares",
            "value_per_share": "Value / Share",
            "terminal_value_percentage": "Terminal Value % of EV",
        }
        for key, row in output_rows.items():
            worksheet.write(row, 0, output_labels[key], formats["label"])

        forecast_pv_start = f"B{rows['pv_fcff'] + 1}"
        forecast_pv_end = f"{xl_col_to_name(forecast_count)}{rows['pv_fcff'] + 1}"
        output_formulas = {
            "explicit_forecast_pv": f"=SUM({forecast_pv_start}:{forecast_pv_end})",
            "terminal_pv": f"={terminal_letter}{rows['pv_terminal_value'] + 1}",
            "enterprise_value": "=IF(AND(ISNUMBER(B15),ISNUMBER(B16)),B15+B16,NA())",
            "net_debt": f"=IF(ISNUMBER({input_ref('net_debt')}),{input_ref('net_debt')},NA())",
            "non_operating_assets": (
                f"=IF(AND(ISNUMBER({input_ref('non_operating_assets')}),"
                f"{input_ref('non_operating_assets')}>=0),"
                f"{input_ref('non_operating_assets')},NA())"
            ),
            "equity_value": (
                "=IF(AND(ISNUMBER(B17),ISNUMBER(B18),ISNUMBER(B19)),B17-B18+B19,NA())"
            ),
            "diluted_shares": (
                f"=IF(AND(ISNUMBER({input_ref('diluted_shares')}),"
                f"{input_ref('diluted_shares')}>0),"
                f"{input_ref('diluted_shares')},NA())"
            ),
            "value_per_share": (
                "=IF(OR(NOT(ISNUMBER(B20)),NOT(ISNUMBER(B21)),B21<=0),NA(),B20/B21)"
            ),
            "terminal_value_percentage": (
                "=IF(OR(NOT(ISNUMBER(B17)),NOT(ISNUMBER(B16))),NA(),"
                'IF(B17=0,"",B16/B17*100))'
            ),
        }
        output_values = {
            "explicit_forecast_pv": result.explicit_forecast_present_value.total_present_value,
            "terminal_pv": result.terminal_present_value.present_value,
            "enterprise_value": result.enterprise_value,
            "net_debt": result.capital_bridge.net_debt,
            "non_operating_assets": result.capital_bridge.non_operating_assets,
            "equity_value": result.equity_value,
            "diluted_shares": result.capital_bridge.diluted_shares,
            "value_per_share": result.value_per_share,
            "terminal_value_percentage": result.terminal_value_percentage,
        }
        for key, formula in output_formulas.items():
            self._write_formula(
                worksheet,
                output_rows[key],
                1,
                formula,
                output_values[key],
                formats[
                    "formula_percent"
                    if key == "terminal_value_percentage"
                    else "formula"
                ],
            )

        worksheet.set_column("A:A", 32)
        worksheet.set_column(1, terminal_column, 18)
        return {"output_rows": output_rows}

    def _write_relative_valuation(
        self,
        relative: ComparableImpliedValuation | ProviderNeutralRelativeValuation | None,
        peer_report: ComparableMultiplesReport | None,
        formats: dict[str, Any],
        *,
        workbook: xlsxwriter.Workbook,
    ) -> None:
        """Write the relative outputs attached to a valuation CLI run.

        Relative valuation is resolved from provider observations and peer
        evidence, so this sheet deliberately preserves the resolved values and
        provenance rather than pretending that the peer-selection process is an
        editable Excel formula model.
        """
        worksheet = workbook.add_worksheet("Relative Valuation")
        worksheet.freeze_panes(4, 0)
        worksheet.write("A1", "Relative Valuation", formats["title"])
        worksheet.write(
            "A2",
            "Static peer evidence and resolved target-date cases from the valuation run.",
            formats["source"],
        )

        row = 3

        def section_row(label: str) -> None:
            nonlocal row
            worksheet.write(row, 0, label, formats["section"])
            row += 1

        def table_header(headers: tuple[str, ...]) -> None:
            nonlocal row
            for column, header in enumerate(headers):
                worksheet.write(row, column, header, formats["header"])
            row += 1

        def cell(column: int, value: Any) -> None:
            if value is None:
                worksheet.write_blank(row, column, None, formats["number"])
            elif isinstance(value, datetime.datetime):
                worksheet.write_datetime(row, column, value, formats["datetime"])
            elif isinstance(value, datetime.date):
                worksheet.write_datetime(
                    row,
                    column,
                    datetime.datetime.combine(value, datetime.time()),
                    formats["date"],
                )
            elif isinstance(value, Decimal):
                worksheet.write_number(
                    row, column, self._number(value), formats["number"]
                )
            elif isinstance(value, Enum):
                worksheet.write_string(row, column, str(value.value), formats["text"])
            elif isinstance(value, (int, float)):
                worksheet.write_number(row, column, value, formats["number"])
            else:
                worksheet.write(row, column, value, formats["text"])

        if peer_report is not None:
            section_row("LTM Peer Multiples")
            table_header(
                (
                    "Basis",
                    "Target",
                    "Peer Median",
                    "Peer Minimum",
                    "Peer Maximum",
                    "N",
                    "Forward Peer Median",
                    "Forward Peer Minimum",
                    "Forward Peer Maximum",
                    "Forward N",
                    "Unit",
                    "Target Status",
                )
            )
            target_multiples = {
                item.basis: item for item in peer_report.target.multiples
            }
            summaries = {item.basis: item for item in peer_report.summaries}
            forward_summaries = {
                item.basis: item for item in peer_report.forward_summaries
            }
            bases = list(
                dict.fromkeys(
                    (
                        RelativeValuationBasis.PE,
                        RelativeValuationBasis.EV_TO_EBITDA,
                        RelativeValuationBasis.EV_TO_FCF,
                        *target_multiples,
                        *summaries,
                        *forward_summaries,
                    )
                )
            )
            for basis in bases:
                target_multiple = target_multiples.get(basis)
                summary = summaries.get(basis)
                forward_summary = forward_summaries.get(basis)
                values = (
                    target_multiple.value
                    if target_multiple is not None
                    and target_multiple.status == MultipleStatus.COMPUTED
                    else None
                )
                cell(0, basis.label)
                cell(1, values)
                cell(2, summary.median if summary is not None else None)
                cell(3, summary.minimum if summary is not None else None)
                cell(4, summary.maximum if summary is not None else None)
                cell(5, summary.sample_size if summary is not None else None)
                cell(
                    6,
                    forward_summary.median if forward_summary is not None else None,
                )
                cell(
                    7,
                    forward_summary.minimum if forward_summary is not None else None,
                )
                cell(
                    8,
                    forward_summary.maximum if forward_summary is not None else None,
                )
                cell(
                    9,
                    forward_summary.sample_size
                    if forward_summary is not None
                    else None,
                )
                cell(
                    10,
                    target_multiple.unit if target_multiple is not None else None,
                )
                cell(
                    11,
                    target_multiple.status.value
                    if target_multiple is not None
                    else None,
                )
                row += 1

        if relative is not None:
            row += 1
            section_row("Resolved Relative Valuation (Composite / DCF Diagnostic)")
            table_header(("Input", "Value", "Units / description"))
            if isinstance(relative, ComparableImpliedValuation):
                basis_label = relative.basis.label
                metric_amount = relative.forecast_metric
                metric_label = relative.forecast_metric_label
            else:
                basis_label = relative.metric.basis.label
                metric_amount = relative.metric.amount
                metric_label = relative.metric.label
            input_rows = (
                ("Basis", basis_label, "relative basis"),
                ("Forecast Metric", metric_amount, relative.currency),
                ("Forecast Metric Label", metric_label, "target date"),
                ("Target Date", relative.target_date, "date"),
                ("Horizon", relative.horizon_years, "years"),
                ("Discount Rate", relative.discount_rate, "% points"),
            )
            if isinstance(relative, ComparableImpliedValuation):
                input_rows = (
                    *input_rows,
                    (
                        "Projected Diluted Shares",
                        relative.projected_diluted_shares,
                        "shares",
                    ),
                    (
                        "Projected Net Debt",
                        relative.projected_net_debt,
                        relative.currency,
                    ),
                    (
                        "DCF-implied Multiple (diagnostic)",
                        relative.resolved_multiple.fundamental_anchor,
                        "multiple",
                    ),
                    (
                        "Peer Median Multiple",
                        relative.resolved_multiple.peer_anchor,
                        "multiple",
                    ),
                    (
                        "Historical Median Multiple",
                        relative.resolved_multiple.historical_anchor,
                        "multiple",
                    ),
                )
            else:
                input_rows = (
                    *input_rows,
                    ("Diluted Shares", relative.diluted_shares, "shares"),
                    (
                        "Numerator Basis",
                        relative.metric.numerator_basis,
                        "enterprise or equity value",
                    ),
                )
            for label, value, units in input_rows:
                cell(0, label)
                cell(1, value)
                cell(2, units)
                row += 1

            row += 1
            table_header(
                (
                    "Case",
                    "Multiple",
                    "Target-date Numerator / EV",
                    "Target-date Equity Value",
                    "Target-date Value / Share",
                    (
                        "DCF PV Diagnostic / Share"
                        if isinstance(relative, ComparableImpliedValuation)
                        else "Present Value / Share"
                    ),
                )
            )
            cases = (
                ("Lower", relative.lower_case),
                ("Resolved", relative.point_case),
                ("Upper", relative.upper_case),
            )
            for label, case in cases:
                cell(0, case.label or label)
                cell(1, case.multiple)
                if isinstance(relative, ComparableImpliedValuation):
                    cell(2, case.implied_enterprise_value)
                    cell(3, case.implied_equity_value)
                    cell(4, case.implied_value_per_share)
                else:
                    cell(2, case.target_date_numerator_value)
                    cell(3, case.target_date_equity_value)
                    cell(4, case.target_date_value_per_share)
                cell(5, case.present_value_per_share)
                row += 1

            if isinstance(relative, ComparableImpliedValuation):
                independent_case_groups = (
                    (
                        "Independent Peer Valuation",
                        (
                            relative.pure_peer_lower_case,
                            relative.pure_peer_point_case,
                            relative.pure_peer_upper_case,
                        ),
                    ),
                    (
                        "Historical Multiple Valuation",
                        (
                            relative.historical_lower_case,
                            relative.historical_point_case,
                            relative.historical_upper_case,
                        ),
                    ),
                )

                for title, case_group in independent_case_groups:
                    if any(case is None for case in case_group):
                        continue
                    row += 2
                    section_row(title)
                    table_header(
                        (
                            "Case",
                            "Multiple",
                            "Target-date Enterprise Value",
                            "Target-date Equity Value",
                            "Target-date Value / Share",
                        )
                    )
                    for case in case_group:
                        assert case is not None
                        cell(0, case.label)
                        cell(1, case.multiple)
                        cell(2, case.implied_enterprise_value)
                        cell(3, case.implied_equity_value)
                        cell(4, case.target_date_value_per_share)
                        row += 1

            if relative.current_price is not None:
                row += 1
                cell(0, "Current Price")
                cell(1, relative.current_price)
                cell(2, relative.currency)
                row += 1
            if (
                isinstance(relative, ComparableImpliedValuation)
                and relative.analyst_target_price is not None
            ):
                cell(0, "Analyst Target Price")
                cell(1, relative.analyst_target_price)
                cell(2, relative.currency)
                row += 1

            if relative.warnings:
                row += 1
                section_row("Warnings")
                for warning in relative.warnings:
                    cell(0, "Warning")
                    cell(1, warning)
                    row += 1

        worksheet.set_column("A:A", 32)
        worksheet.set_column("B:J", 22)
        worksheet.set_column("K:K", 22)
        worksheet.set_column("L:L", 18)

    def _write_summary(
        self,
        forecast: FcffForecast,
        result: FcffDcfResult,
        dcf_refs: dict[str, Any],
        context: Any | None,
        formats: dict[str, Any],
        *,
        worksheet,
    ) -> None:
        worksheet.freeze_panes(3, 1)
        worksheet.write("A1", "Valuation Summary", formats["title"])
        worksheet.write(
            "A2",
            "FCFF DCF valuation export. Formula outputs link to DCF Calculation; static values preserve the Python run and provenance.",
            formats["source"],
        )
        worksheet.write_row(
            3,
            0,
            ("Item", "Value", "Units / description", "Source / provenance"),
            formats["header"],
        )
        rows = dcf_refs["output_rows"]

        def dcf_ref(key: str) -> str:
            return self._ref(self._DCF, rows[key], 1)

        summary_rows = [
            ("Company", result.company_name, "", "FCFF DCF result"),
            ("Ticker", result.ticker or "", "", "FCFF DCF result"),
            ("Provider", result.provider, "", "FCFF forecast"),
            ("Model", "FCFF DCF", "model", "valuation run"),
            ("Valuation Date", result.valuation_date, "date", "valuation run"),
            (
                "Forecast Seed",
                result.forecast_seed_type,
                "seed type",
                result.forecast_seed_methodology,
            ),
            (
                "Forecast Horizon",
                len(forecast.observations),
                "annual periods",
                "actual forecast supplied to renderer",
            ),
            ("Base Period End", forecast.base_period_end, "date", "forecast seed"),
            ("Base Revenue", forecast.base_revenue, forecast.unit, "forecast seed"),
            ("Base FCFF", forecast.base_fcff, forecast.unit, "forecast seed"),
            (
                "Enterprise Value",
                dcf_ref("enterprise_value"),
                forecast.unit,
                "linked to DCF Calculation",
            ),
            (
                "Equity Value",
                dcf_ref("equity_value"),
                forecast.unit,
                "linked to DCF Calculation",
            ),
            (
                "Value / Share",
                dcf_ref("value_per_share"),
                f"{forecast.unit} / share",
                "linked to DCF Calculation",
            ),
            (
                "Explicit Forecast PV",
                dcf_ref("explicit_forecast_pv"),
                forecast.unit,
                "linked to DCF Calculation",
            ),
            (
                "Terminal PV",
                dcf_ref("terminal_pv"),
                forecast.unit,
                "linked to DCF Calculation",
            ),
            (
                "Terminal Value % of EV",
                dcf_ref("terminal_value_percentage"),
                "%",
                "linked to DCF Calculation",
            ),
            (
                "Forecast Availability",
                result.observation_availability_mode or "",
                "observation mode",
                "forecast snapshot",
            ),
            (
                "Snapshot Retrieved At",
                result.financial_snapshot_retrieved_at,
                "datetime",
                "forecast snapshot",
            ),
        ]
        row = 4
        for label, value, units, source in summary_rows:
            worksheet.write(row, 0, label, formats["label"])
            if isinstance(value, str) and value.startswith("'"):
                self._write_formula(
                    worksheet,
                    row,
                    1,
                    f"={value}",
                    self._summary_cached_value(label, result),
                    formats[
                        "formula_percent"
                        if label == "Terminal Value % of EV"
                        else "formula"
                    ],
                )
            elif isinstance(value, datetime.datetime):
                worksheet.write_string(row, 1, value.isoformat(), formats["text"])
            elif isinstance(value, datetime.date):
                self._write_date(worksheet, row, 1, value, formats["date"])
            elif isinstance(value, Decimal):
                worksheet.write_number(row, 1, self._number(value), formats["number"])
            elif isinstance(value, (int, float)):
                worksheet.write_number(row, 1, value, formats["number"])
            else:
                worksheet.write(row, 1, value, formats["text"])
            worksheet.write(row, 2, units, formats["text"])
            worksheet.write(row, 3, source, formats["source"])
            row += 1

        if context is not None:
            worksheet.write(row + 1, 0, "Canonical Context", formats["section"])
            worksheet.write(
                row + 2,
                0,
                "Context type",
                formats["label"],
            )
            worksheet.write(row + 2, 1, type(context).__name__, formats["text"])
            worksheet.write(
                row + 2,
                3,
                "Optional context supplied by caller; calculations use FCFF inputs/result.",
                formats["source"],
            )
            row += 3

        if result.share_repurchases is not None:
            repurchases = result.share_repurchases
            worksheet.write(
                row + 1,
                0,
                "Share Repurchase Result (static valuation-run output)",
                formats["section"],
            )
            row += 2
            static_buyback_rows = (
                ("Source", repurchases.source, "", repurchases.source),
                (
                    "Starting Shares",
                    repurchases.starting_shares,
                    "shares",
                    "static result",
                ),
                ("Ending Shares", repurchases.ending_shares, "shares", "static result"),
                (
                    "Shares Repurchased",
                    repurchases.shares_repurchased,
                    "shares",
                    "static result",
                ),
                (
                    "Total Cash Spent",
                    repurchases.total_cash_spent,
                    forecast.unit,
                    "static result",
                ),
                (
                    "PV of Cash Spent",
                    repurchases.present_value_cash_spent,
                    forecast.unit,
                    "static result",
                ),
                (
                    "Pre-repurchase Equity Value",
                    repurchases.pre_repurchase_equity_value,
                    forecast.unit,
                    "static result",
                ),
                (
                    "Residual Equity Value",
                    repurchases.residual_equity_value,
                    forecast.unit,
                    "static result",
                ),
                (
                    "Pre-repurchase Value / Share",
                    repurchases.pre_repurchase_value_per_share,
                    f"{forecast.unit} / share",
                    "static result",
                ),
                (
                    "Value / Remaining Share",
                    repurchases.value_per_remaining_share,
                    f"{forecast.unit} / share",
                    "static result",
                ),
                (
                    "Accretion / (Dilution)",
                    repurchases.accretion_percentage,
                    "% (percentage points)",
                    "static result",
                ),
            )
            for label, value, units, source in static_buyback_rows:
                worksheet.write(row, 0, label, formats["label"])
                if isinstance(value, Decimal):
                    worksheet.write_number(
                        row, 1, self._number(value), formats["number"]
                    )
                else:
                    worksheet.write(row, 1, value, formats["text"])
                worksheet.write(row, 2, units, formats["text"])
                worksheet.write(row, 3, source, formats["source"])
                row += 1
            worksheet.write(row, 0, "Repurchase periods", formats["label"])
            row += 1
            for period in repurchases.periods:
                worksheet.write(
                    row,
                    0,
                    f"FY{period.fiscal_year} cash spent",
                    formats["text"],
                )
                worksheet.write_number(
                    row, 1, self._number(period.cash_spent), formats["number"]
                )
                worksheet.write(row, 2, forecast.unit, formats["text"])
                worksheet.write(
                    row,
                    3,
                    "static result; no unsupported formulas",
                    formats["source"],
                )
                row += 1

        warnings = (
            "Excel formula guards return #N/A when WACC, terminal growth or "
            "multiple, shares, or FCFF drivers are blank or invalid; correct the "
            "yellow input cells before relying on the valuation.",
            *result.warnings,
        )
        if warnings:
            worksheet.write(row + 1, 0, "Warnings", formats["section"])
            for index, warning in enumerate(warnings, start=row + 2):
                worksheet.write(index, 0, "Warning", formats["label"])
                worksheet.write(index, 1, warning, formats["warning"])

        worksheet.set_column("A:A", 34)
        worksheet.set_column("B:B", 30)
        worksheet.set_column("C:C", 24)
        worksheet.set_column("D:D", 68)

    @property
    def _percent_rows(self) -> frozenset[str]:
        return frozenset(
            {
                "revenue_growth",
                "operating_margin",
                "tax_rate",
                "depreciation_to_revenue",
                "capex_to_revenue",
                "operating_working_capital_to_revenue",
            }
        )

    @staticmethod
    def _ref(sheet: str, row: int, column: int) -> str:
        return f"'{sheet}'!${xl_col_to_name(column)}${row + 1}"

    @staticmethod
    def _ratio(numerator: Decimal, denominator: Decimal) -> Decimal | None:
        if denominator == 0:
            return None
        return numerator / denominator * Decimal(100)

    @staticmethod
    def _number(value: Decimal | int | float) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Excel valuation exports require finite numeric values")
        return number

    @classmethod
    def _write_formula(
        cls,
        worksheet,
        row: int,
        column: int,
        formula: str,
        cached_value: Decimal | int | float | str | None,
        cell_format,
    ) -> None:
        if isinstance(cached_value, Decimal):
            cached_value = cls._number(cached_value)
        elif isinstance(cached_value, (int, float)):
            cached_value = cls._number(cached_value)
        elif cached_value is None:
            cached_value = ""
        worksheet.write_formula(row, column, formula, cell_format, cached_value)

    @staticmethod
    def _write_datetime(worksheet, row, column, value, cell_format) -> None:
        worksheet.write_datetime(row, column, value, cell_format)

    @classmethod
    def _write_date(cls, worksheet, row, column, value, cell_format) -> None:
        cls._write_datetime(
            worksheet,
            row,
            column,
            datetime.datetime.combine(value, datetime.time()),
            cell_format,
        )

    @staticmethod
    def _summary_cached_value(label: str, result: FcffDcfResult):
        return {
            "Enterprise Value": result.enterprise_value,
            "Equity Value": result.equity_value,
            "Value / Share": result.value_per_share,
            "Explicit Forecast PV": result.explicit_forecast_present_value.total_present_value,
            "Terminal PV": result.terminal_present_value.present_value,
            "Terminal Value % of EV": result.terminal_value_percentage,
        }[label]


ValuationExcelExportService = ValuationExcelRenderer


def render_valuation_excel(
    forecast: FcffForecast,
    result: FcffDcfResult,
    output: str | Path,
    *,
    report: Any | None = None,
    canonical_report: Any | None = None,
    relative: ComparableImpliedValuation
    | ProviderNeutralRelativeValuation
    | None = None,
    relative_result: ComparableImpliedValuation
    | ProviderNeutralRelativeValuation
    | None = None,
    peer_report: ComparableMultiplesReport | None = None,
    discount_timing_basis: DiscountTimingBasis | str | None = None,
) -> Path:
    """Render an FCFF forecast and DCF result as an Excel workbook."""
    return ValuationExcelRenderer().render(
        forecast,
        result,
        output,
        report=report,
        canonical_report=canonical_report,
        relative=relative,
        relative_result=relative_result,
        peer_report=peer_report,
        discount_timing_basis=discount_timing_basis,
    )


__all__ = [
    "ValuationExcelExportService",
    "ValuationExcelRenderer",
    "render_valuation_excel",
]
