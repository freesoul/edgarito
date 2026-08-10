import datetime
import json
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest

from edgarito.cli.__main__ import main
from edgarito.cli.parser import build_parser
from edgarito.cli.presentation.dcf import FcffDcfConsolePresenter
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.export import ValuationExcelRenderer
from edgarito.services.forecasting import (
    FcffForecast,
    FcffForecastDriver,
    FcffForecastObservation,
    FcffForecastParameters,
    FcffForecastYtdAnchor,
    ForecastAssumptionSource,
    ForecastSeedType,
    ForecastValue,
)
from edgarito.services.forecasting.models import FcffForecastDcfStub
from edgarito.services.valuation import (
    CashFlowTiming,
    CompanyTradingMultiples,
    ComparableMultiplesReport,
    FcffDcfCapitalBridge,
    FcffDcfCapitalBridgeResolver,
    FcffDcfParameters,
    FcffDcfResult,
    FcffDcfService,
    LtmFundamentals,
    MultipleStatus,
    PeerMultipleSummary,
    PeerSelectionParameters,
    PeerUniverse,
    RelativeValuationBasis,
    ShareRepurchaseParameters,
    TerminalMetric,
    TerminalValueMethod,
    TradingMultiple,
)

ROOT = Path(__file__).parents[1]
XML_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def test_fcff_dcf_discounts_forecast_terminal_value_and_equity_bridge():
    result = FcffDcfService().value(
        _forecast(),
        FcffDcfParameters(
            wacc="10",
            perpetual_growth_rate="2",
        ),
        _capital_bridge(),
    )

    assert result.explicit_forecast_present_value.total_present_value.quantize(
        Decimal("0.001")
    ) == Decimal("181.818")
    assert result.terminal_value.terminal_value == Decimal("1402.5")
    assert result.enterprise_value.quantize(Decimal("0.001")) == Decimal("1340.909")
    assert result.equity_value.quantize(Decimal("0.001")) == Decimal("1240.909")
    assert result.value_per_share.quantize(Decimal("0.001")) == Decimal("124.091")
    assert result.terminal_value_percentage is not None
    assert result.terminal_value_percentage > Decimal(75)
    assert "highly sensitive" in result.warnings[0]
    assert result.forecast_cell_audits[2026]["fcff"].value == Decimal("100")
    assert (
        result.forecast_cell_audits[2026]["fcff"].source
        == "unknown/legacy/inconsistent"
    )
    assert result.forecast_cell_audits[2026]["fcff"].confidence == "low"
    assert FcffDcfResult.model_validate_json(result.model_dump_json()) == result


def test_fcff_dcf_verbose_output_includes_economic_model_and_cell_provenance():
    forecast = _consistent_forecast()
    forecast.assumption_sources = {
        driver: ForecastAssumptionSource.EXPLICIT for driver in FcffForecastDriver
    }
    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )

    output = FcffDcfConsolePresenter().render(result, verbose=True)

    assert "ECONOMIC FCFF MODEL" in output
    assert output.index("ECONOMIC FCFF MODEL") < output.index("INTRINSIC VALUATION")
    assert f"Revenue ({result.unit}" in output
    assert f"FCFF ({result.unit}" in output
    assert "ECONOMIC CELL PROVENANCE" in output
    assert "revenue: source=" in output
    assert "revenue: value=" not in output
    assert "source=explicit" in output
    assert "confidence=high" in output


def test_fcff_dcf_regenerates_authoritative_cell_audits_from_current_observations():
    forecast = _consistent_forecast()
    forecast.assumption_sources = {
        driver: ForecastAssumptionSource.EXPLICIT for driver in FcffForecastDriver
    }
    forecast.observations[0] = forecast.observations[0].model_copy(
        update={
            "cell_audits": {
                "fcff": ForecastValue(
                    value=Decimal("-999"),
                    source="stale",
                    method="stale metadata",
                    confidence="low",
                )
            }
        }
    )

    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )

    audit = result.forecast_cell_audits[2026]["fcff"]
    assert audit.value == Decimal("109.45")
    assert audit.source.startswith("derived[")
    assert audit.source != "stale"


def test_fcff_dcf_rejects_inconsistent_forecast_with_audit_metadata():
    forecast = _forecast()
    forecast.observations[0] = forecast.observations[0].model_copy(
        update={
            "cell_audits": {
                "fcff": ForecastValue(
                    value=Decimal("100"),
                    source="explicit",
                    method="test",
                    confidence="high",
                )
            }
        }
    )

    with pytest.raises(ValueError, match="economic identities are inconsistent"):
        FcffDcfService().value(
            forecast,
            FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
            _capital_bridge(),
        )


def test_fcff_dcf_adds_non_operating_investments_to_equity_value():
    plain = FcffDcfService().value(
        _forecast(),
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )
    bridge = _capital_bridge().model_copy(
        update={
            "non_operating_assets": Decimal("75"),
            "non_operating_assets_source": "marketable securities",
        }
    )
    adjusted = FcffDcfService().value(
        _forecast(),
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        bridge,
    )

    assert adjusted.equity_value - plain.equity_value == Decimal("75")
    assert adjusted.value_per_share - plain.value_per_share == Decimal("7.5")


def test_mid_year_timing_changes_explicit_cash_flows_but_not_terminal_timing():
    service = FcffDcfService()
    end_period = service.value(
        _forecast(),
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )
    mid_year = service.value(
        _forecast(),
        FcffDcfParameters(
            wacc="10",
            perpetual_growth_rate="2",
            cash_flow_timing=CashFlowTiming.MID_YEAR,
        ),
        _capital_bridge(),
    )

    assert mid_year.explicit_forecast_present_value.cash_flows[0].period == Decimal(
        "0.5"
    )
    assert mid_year.explicit_forecast_present_value.total_present_value > (
        end_period.explicit_forecast_present_value.total_present_value
    )
    assert mid_year.terminal_present_value.period == Decimal(2)
    assert mid_year.terminal_present_value.present_value == (
        end_period.terminal_present_value.present_value
    )
    assert any("mid-year timing" in warning for warning in mid_year.warnings)


def test_current_valuation_date_uses_calendar_stub_periods():
    valuation_date = datetime.date(2026, 7, 1)
    result = FcffDcfService().value(
        _forecast(),
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
        valuation_date=valuation_date,
    )

    first = result.explicit_forecast_present_value.cash_flows[0]
    assert result.valuation_date == valuation_date
    assert first.period == Decimal(183) / Decimal(365)
    assert result.terminal_present_value.period == Decimal(548) / Decimal(365)
    assert any("Capital bridge is dated 2025-12-31" in item for item in result.warnings)


def test_ytd_dcf_discounts_remaining_stub_before_later_full_year_observations():
    forecast = _ytd_forecast()
    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _ytd_capital_bridge(),
        valuation_date=datetime.date(2026, 8, 10),
    )

    first, second = result.explicit_forecast_present_value.cash_flows
    assert first.label == "FY2026E FCFF remaining stub"
    assert first.amount == Decimal("8.45")
    assert first.period == Decimal(143) / Decimal(365)
    assert second.amount == forecast.observations[1].fcff
    assert second.period == Decimal(508) / Decimal(365)
    assert result.terminal_value.final_cash_flow == forecast.observations[-1].fcff
    assert result.terminal_present_value.period == Decimal(508) / Decimal(365)


def test_ytd_dcf_rejects_a_missing_remaining_stub_instead_of_using_full_year_fcff():
    forecast = _ytd_forecast().model_copy(update={"dcf_stub": None})

    with pytest.raises(ValueError, match="requires complete remaining stub"):
        FcffDcfService().value(
            forecast,
            FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
            _ytd_capital_bridge(),
            valuation_date=datetime.date(2026, 8, 10),
        )


def test_recent_quarterly_bridge_is_not_reported_as_stale():
    result = FcffDcfService().value(
        _forecast(),
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
        valuation_date=datetime.date(2026, 4, 15),
    )

    assert not any("Capital bridge is dated" in item for item in result.warnings)


def test_current_mid_year_timing_rejects_an_already_elapsed_cash_flow_date():
    with pytest.raises(ValueError, match="Mid-year cash-flow timing"):
        FcffDcfService().value(
            _forecast(),
            FcffDcfParameters(
                wacc="10",
                perpetual_growth_rate="2",
                cash_flow_timing=CashFlowTiming.MID_YEAR,
            ),
            _capital_bridge(),
            valuation_date=datetime.date(2026, 8, 1),
        )


def test_perpetuity_growth_warns_when_explicit_fcff_has_not_converged():
    service = FcffDcfService()
    abrupt = service.value(
        _forecast(),
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )
    converged_forecast = _forecast()
    converged_forecast.observations[-1].fcff = Decimal("102")
    converged = service.value(
        converged_forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )

    assert any(
        "Final explicit FCFF growth (10.0%)" in warning
        and "sensitive to --years" in warning
        for warning in abrupt.warnings
    )
    assert not any(
        "terminal transition is abrupt" in warning for warning in converged.warnings
    )


def test_exit_multiple_supports_explicit_terminal_metrics():
    result = FcffDcfService().value(
        _forecast(),
        FcffDcfParameters(
            wacc="10",
            terminal_method=TerminalValueMethod.EXIT_MULTIPLE,
            exit_multiple="8",
            exit_metric=TerminalMetric.EBITDA,
        ),
        _capital_bridge(),
    )

    assert result.terminal_value.method == TerminalValueMethod.EXIT_MULTIPLE
    assert result.terminal_value.terminal_metric == Decimal("170")
    assert result.terminal_value.terminal_value == Decimal("1360")
    assert any("market-relative scenario" in item for item in result.warnings)


@pytest.mark.parametrize(
    ("terminal_method", "expected_terminal_formula"),
    (
        (TerminalValueMethod.PERPETUITY_GROWTH, "perpetuity_growth"),
        (TerminalValueMethod.EXIT_MULTIPLE, "exit_multiple"),
    ),
)
def test_valuation_excel_export_contains_linked_formulas_yellow_inputs_and_cached_values(
    tmp_path, terminal_method, expected_terminal_formula
):
    parameters = (
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2")
        if terminal_method == TerminalValueMethod.PERPETUITY_GROWTH
        else FcffDcfParameters(
            wacc="10",
            terminal_method=terminal_method,
            exit_multiple="8",
            exit_metric=TerminalMetric.EBITDA,
        )
    )
    forecast = _consistent_forecast()
    result = FcffDcfService().value(forecast, parameters, _capital_bridge())
    output = tmp_path / "nested" / f"valuation-{terminal_method.value}.xlsx"

    ValuationExcelRenderer().render(forecast, result, output)

    with ZipFile(output) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        names = [
            sheet.attrib["name"]
            for sheet in workbook.findall("main:sheets/main:sheet", XML_NS)
        ]
        assert names == [
            "Valuation Summary",
            "FCFF Inputs",
            "FCFF Forecast",
            "DCF Calculation",
        ]
        calc_pr = workbook.find("main:calcPr", XML_NS)
        assert calc_pr is not None
        assert (
            calc_pr.attrib.get("calcMode") == "auto"
            or calc_pr.attrib.get("fullCalcOnLoad") == "1"
            or calc_pr.attrib.get("forceFullCalc") == "1"
        )

        styles = ElementTree.fromstring(archive.read("xl/styles.xml"))
        fills = styles.find("main:fills", XML_NS)
        cell_xfs = styles.find("main:cellXfs", XML_NS)
        assert fills is not None and cell_xfs is not None

        inputs = ElementTree.fromstring(archive.read("xl/worksheets/sheet2.xml"))
        input_cell = inputs.find(".//main:c[@r='C6']", XML_NS)
        assert input_cell is not None
        input_style = cell_xfs[int(input_cell.attrib["s"])]
        fill = fills[int(input_style.attrib["fillId"])]
        foreground = fill.find("main:patternFill/main:fgColor", XML_NS)
        assert foreground is not None
        assert foreground.attrib.get("rgb", "").endswith("FFFF00")
        validations = inputs.findall(".//main:dataValidation", XML_NS)
        validation_refs = {item.attrib["sqref"] for item in validations}
        assert {
            "B13",
            "B14",
            "B17",
            "B12",
            "B15",
            "B16",
            "B18",
            "B19",
            "B20",
            "C6:C7",
            "D6:D7",
            "E6:E7",
            "F6:F7",
            "G6:G7",
            "H6:H7",
        }.issubset(validation_refs)

        forecast_sheet = ElementTree.fromstring(
            archive.read("xl/worksheets/sheet3.xml")
        )
        forecast_formula = forecast_sheet.find(".//main:c[@r='C6']/main:f", XML_NS)
        assert forecast_formula is not None
        assert "FCFF Inputs" in forecast_formula.text

        dcf_sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet4.xml"))
        formula_nodes = dcf_sheet.findall(".//main:f", XML_NS)
        formulas = [node.text or "" for node in formula_nodes]
        assert any("FCFF Forecast" in formula for formula in formulas)
        terminal_formula = dcf_sheet.find(".//main:c[@r='D10']/main:f", XML_NS)
        assert terminal_formula is not None
        assert "IF" in terminal_formula.text
        assert expected_terminal_formula in terminal_formula.text
        assert "NA()" in terminal_formula.text
        assert "ISNUMBER" in terminal_formula.text
        terminal_metric_formula = dcf_sheet.find(".//main:c[@r='D9']/main:f", XML_NS)
        assert terminal_metric_formula is not None
        assert '"exit_multiple"' in terminal_metric_formula.text

        terminal_cell = dcf_sheet.find(".//main:c[@r='D10']/main:v", XML_NS)
        terminal_metric_cell = dcf_sheet.find(".//main:c[@r='D9']/main:v", XML_NS)
        summary_value = ElementTree.fromstring(
            archive.read("xl/worksheets/sheet1.xml")
        ).find(".//main:c[@r='B15']/main:v", XML_NS)
        assert terminal_cell is not None
        if terminal_method == TerminalValueMethod.PERPETUITY_GROWTH:
            assert terminal_metric_cell is None or terminal_metric_cell.text in (
                None,
                "",
            )
        assert summary_value is not None
        assert Decimal(terminal_cell.text) == result.terminal_value.terminal_value
        assert float(summary_value.text) == pytest.approx(
            float(result.enterprise_value)
        )


def test_valuation_excel_export_contains_economic_cell_audit_section(tmp_path):
    forecast = _consistent_forecast()
    forecast.assumption_sources = {
        driver: ForecastAssumptionSource.EXPLICIT for driver in FcffForecastDriver
    }
    forecast.observations[0] = forecast.observations[0].model_copy(
        update={
            "cell_audits": {
                "fcff": ForecastValue(
                    value=forecast.observations[0].fcff,
                    source="stale-observation",
                    method="stale observation metadata",
                    confidence="low",
                )
            }
        }
    )
    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )
    output = tmp_path / "economic-audit.xlsx"

    ValuationExcelRenderer().render(forecast, result, output)

    with ZipFile(output) as archive:
        shared_strings = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = "".join(shared_strings.itertext())
        assert "Economic FCFF Cell Audit" in strings
        assert "derived[" in strings
        assert "stale-observation" not in strings


def test_valuation_excel_export_includes_requested_relative_multiple_evidence(tmp_path):
    forecast = _consistent_forecast()
    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )
    target = CompanyTradingMultiples(
        provider="test",
        market_provider="test-market",
        company_id="EX",
        company_name="Example Co",
        ticker="EX",
        price_date=datetime.date(2026, 7, 1),
        price=Decimal("100"),
        currency="USD",
        market_capitalization=Decimal("1000"),
        enterprise_value=Decimal("1100"),
        fundamentals=LtmFundamentals(
            period_start=datetime.date(2025, 1, 1),
            period_end=datetime.date(2025, 12, 31),
            currency="USD",
            revenue=Decimal("500"),
            ebitda=Decimal("100"),
            net_income=Decimal("50"),
            free_cash_flow=Decimal("75"),
            shares=Decimal("10"),
        ),
        multiples=[
            TradingMultiple(
                basis=basis,
                status=MultipleStatus.COMPUTED,
                value=value,
                numerator=Decimal("1000"),
                denominator=denominator,
            )
            for basis, value, denominator in (
                (RelativeValuationBasis.PE, Decimal("20"), Decimal("50")),
                (RelativeValuationBasis.EV_TO_EBITDA, Decimal("11"), Decimal("100")),
                (RelativeValuationBasis.EV_TO_FCF, Decimal("14.6667"), Decimal("75")),
            )
        ],
    )
    universe = PeerUniverse(
        target_ticker="EX",
        target_company_id="EX",
        parameters=PeerSelectionParameters(max_peers=3, preferred_minimum=1),
        selected_tickers=("PEER",),
    )
    peer_report = ComparableMultiplesReport(
        universe=universe,
        target=target,
        summaries=[
            PeerMultipleSummary(
                basis=basis,
                median=value,
                minimum=value - Decimal("1"),
                maximum=value + Decimal("1"),
                sample_size=3,
            )
            for basis, value in (
                (RelativeValuationBasis.PE, Decimal("18")),
                (RelativeValuationBasis.EV_TO_EBITDA, Decimal("10")),
                (RelativeValuationBasis.EV_TO_FCF, Decimal("12")),
            )
        ],
    )
    output = tmp_path / "relative-valuations.xlsx"

    ValuationExcelRenderer().render(forecast, result, output, peer_report=peer_report)

    with ZipFile(output) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        names = [
            sheet.attrib["name"]
            for sheet in workbook.findall("main:sheets/main:sheet", XML_NS)
        ]
        assert names[-1] == "Relative Valuation"
        shared_strings = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = "".join(shared_strings.itertext())
        assert "P/E (PER)" in strings
        assert "EV/EBITDA" in strings
        assert "EV/FCF" in strings


def test_valuation_excel_export_allows_missing_base_fcff_and_writes_blank_cells(
    tmp_path,
):
    forecast = _consistent_forecast().model_copy(update={"base_fcff": None})
    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )
    output = tmp_path / "missing-base-fcff.xlsx"

    ValuationExcelRenderer().render(forecast, result, output)

    with ZipFile(output) as archive:
        for worksheet_name, cell_ref in (
            ("xl/worksheets/sheet1.xml", "B14"),
            ("xl/worksheets/sheet3.xml", "B18"),
        ):
            worksheet = ElementTree.fromstring(archive.read(worksheet_name))
            cell = worksheet.find(f".//main:c[@r='{cell_ref}']", XML_NS)
            assert cell is not None
            assert cell.find("main:f", XML_NS) is None
            assert cell.find("main:v", XML_NS) is None


def test_valuation_excel_export_rejects_ytd_plus_forecast_before_writing(tmp_path):
    forecast = _consistent_forecast()
    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )
    forecast.seed_type = ForecastSeedType.YTD_PLUS_FORECAST
    output = tmp_path / "unsupported-ytd.xlsx"

    with pytest.raises(ValueError, match="ForecastSeedType.YTD_PLUS_FORECAST"):
        ValuationExcelRenderer().render(forecast, result, output)

    assert not output.exists()


def test_valuation_excel_export_reproduces_ytd_anchor_formulas_and_inputs(tmp_path):
    forecast = _consistent_forecast()
    forecast.ytd_anchor = FcffForecastYtdAnchor(
        fiscal_year=2026,
        ytd_period_end=datetime.date(2026, 6, 30),
        fiscal_year_end=datetime.date(2026, 12, 31),
        actual_quarters=2,
        actual_revenue=Decimal("500"),
        actual_operating_income=Decimal("60"),
        actual_pretax_income=Decimal("40"),
        actual_income_tax_expense=Decimal("10"),
        actual_tax_rate=Decimal("25"),
        actual_depreciation_and_amortization=Decimal("8"),
        actual_capital_expenditures=Decimal("12"),
        actual_operating_working_capital=Decimal("40"),
        latest_annual_revenue=Decimal("1000"),
        revenue_growth=Decimal("10"),
        operating_margin=Decimal("10"),
        tax_rate=Decimal("20"),
        depreciation_to_revenue=Decimal("2"),
        capex_to_revenue=Decimal("3"),
        operating_working_capital_to_revenue=Decimal("10"),
    )
    forecast.observations = [
        forecast.observations[0].model_copy(
            update={
                "revenue_growth": Decimal("10"),
                "revenue": Decimal("1100"),
                "operating_margin": Decimal(120) / Decimal(1100) * 100,
                "operating_income": Decimal("120"),
                "tax_rate": Decimal("22.5"),
                "nopat": Decimal("93"),
                "depreciation_to_revenue": Decimal(20) / Decimal(1100) * 100,
                "depreciation_and_amortization": Decimal("20"),
                "capex_to_revenue": Decimal(30) / Decimal(1100) * 100,
                "capital_expenditures": Decimal("30"),
                "operating_working_capital": Decimal("110"),
                "change_in_operating_working_capital": Decimal("10"),
                "fcff": Decimal("73"),
            }
        ),
        forecast.observations[1].model_copy(
            update={
                "revenue": Decimal("1155"),
                "operating_income": Decimal("173.25"),
                "nopat": Decimal("138.6"),
                "depreciation_and_amortization": Decimal("23.1"),
                "capital_expenditures": Decimal("34.65"),
                "operating_working_capital": Decimal("115.5"),
                "change_in_operating_working_capital": Decimal("5.5"),
                "fcff": Decimal("121.55"),
            }
        ),
    ]
    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )
    forecast.seed_type = ForecastSeedType.YTD_PLUS_FORECAST
    output = tmp_path / "ytd-valuation.xlsx"

    ValuationExcelRenderer().render(forecast, result, output)

    with ZipFile(output) as archive:
        shared_strings = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        assert "YTD+forecast anchor (editable)" in "".join(shared_strings.itertext())
        forecast_sheet = ElementTree.fromstring(
            archive.read("xl/worksheets/sheet3.xml")
        )
        revenue_formula = forecast_sheet.find(".//main:c[@r='C6']/main:f", XML_NS)
        operating_income_formula = forecast_sheet.find(
            ".//main:c[@r='C8']/main:f", XML_NS
        )
        nopat_formula = forecast_sheet.find(".//main:c[@r='C10']/main:f", XML_NS)
        depreciation_formula = forecast_sheet.find(".//main:c[@r='C12']/main:f", XML_NS)
        capex_formula = forecast_sheet.find(".//main:c[@r='C14']/main:f", XML_NS)
        assert revenue_formula is not None and "MAX" in revenue_formula.text
        assert revenue_formula.text.count("FCFF Inputs") >= 3
        assert operating_income_formula is not None
        assert "actual YTD" not in operating_income_formula.text
        assert "FCFF Inputs" in operating_income_formula.text
        assert nopat_formula is not None and "IF(ISNUMBER" in nopat_formula.text
        assert (
            depreciation_formula is not None
            and "FCFF Inputs" in depreciation_formula.text
        )
        assert capex_formula is not None and "FCFF Inputs" in capex_formula.text


@pytest.mark.parametrize(
    ("parameter_updates", "message"),
    (
        ({"perpetual_growth_rate": None}, "terminal growth"),
        ({"perpetual_growth_rate": Decimal("10")}, "WACC must exceed"),
        ({"wacc": Decimal("1")}, "WACC must exceed"),
    ),
)
def test_valuation_excel_export_rejects_invalid_terminal_inputs(
    tmp_path, parameter_updates, message
):
    forecast = _consistent_forecast()
    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )
    invalid_parameters = result.parameters.model_construct(
        **{**result.parameters.model_dump(), **parameter_updates}
    )
    invalid_result = result.model_copy(update={"parameters": invalid_parameters})
    output = tmp_path / "invalid-inputs.xlsx"

    with pytest.raises(ValueError, match=message):
        ValuationExcelRenderer().render(forecast, invalid_result, output)

    assert not output.exists()


def test_cli_rejects_both_exit_multiple_excel_export_before_writing(tmp_path, capsys):
    output = tmp_path / "both-exit-multiple.xlsx"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "valuation",
                "--ticker",
                "EX",
                "--model",
                "both",
                "--terminal-method",
                "exit_multiple",
                "--excel-output",
                str(output),
            ]
        )

    assert exc_info.value.code == 2
    assert "requires a perpetuity-growth terminal method" in capsys.readouterr().err
    assert not output.exists()


def test_valuation_parser_accepts_excel_output_path(tmp_path):
    args = build_parser().parse_args(
        ["valuation", "--ticker", "EX", "--excel-output", str(tmp_path / "model.xlsx")]
    )

    assert args.excel_output == tmp_path / "model.xlsx"


def test_valuation_excel_export_rejects_formula_cache_divergence(tmp_path):
    forecast = _consistent_forecast()
    forecast.observations[0].revenue += Decimal("1")
    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )

    with pytest.raises(ValueError, match="revenue differs from the workbook formula"):
        ValuationExcelRenderer().render(forecast, result, tmp_path / "invalid.xlsx")


def test_valuation_excel_export_rejects_forecast_result_truncation(tmp_path):
    forecast = _forecast()
    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )
    forecast.observations.pop()

    with pytest.raises(ValueError, match="observations but DCF result has"):
        ValuationExcelRenderer().render(forecast, result, tmp_path / "misaligned.xlsx")


def test_valuation_excel_export_preserves_discount_timing_basis(tmp_path):
    parameters = FcffDcfParameters(wacc="10", perpetual_growth_rate="2")
    forecast = _consistent_forecast()
    service = FcffDcfService()
    forecast_year_result = service.value(forecast, parameters, _capital_bridge())
    calendar_result = service.value(
        forecast,
        parameters,
        _capital_bridge(),
        valuation_date=datetime.date(2026, 7, 1),
    )
    forecast_year_output = tmp_path / "forecast-year.xlsx"
    calendar_output = tmp_path / "calendar.xlsx"
    renderer = ValuationExcelRenderer()
    renderer.render(forecast, forecast_year_result, forecast_year_output)
    renderer.render(forecast, calendar_result, calendar_output)

    def discount_period_formula(path):
        with ZipFile(path) as archive:
            sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet4.xml"))
        return sheet.find(".//main:c[@r='B6']/main:f", XML_NS).text

    forecast_year_formula = discount_period_formula(forecast_year_output)
    calendar_formula = discount_period_formula(calendar_output)
    assert forecast_year_formula.startswith("1-IF(")
    assert "/365" in calendar_formula


def test_fair_value_buybacks_account_for_cash_and_shares_without_fake_accretion():
    result = FcffDcfService().value(
        _forecast(),
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
        share_repurchase_parameters=ShareRepurchaseParameters(
            annual_cash_amounts=(Decimal("100"), Decimal("100")),
            source="test plan",
        ),
    )

    repurchases = result.share_repurchases
    assert repurchases is not None
    assert repurchases.total_cash_spent == Decimal("200")
    assert repurchases.discount_rate == Decimal("10")
    assert repurchases.discount_rate_source == "WACC fallback"
    assert repurchases.purchase_price_source.startswith("model-implied")
    assert repurchases.ending_shares < repurchases.starting_shares
    assert repurchases.residual_equity_value < result.equity_value
    assert repurchases.value_per_remaining_share.quantize(Decimal("0.000001")) == (
        result.value_per_share.quantize(Decimal("0.000001"))
    )
    assert repurchases.accretion_percentage.copy_abs() < Decimal("1e-24")


def test_buybacks_below_intrinsic_value_are_accretive_after_cash_spent():
    result = FcffDcfService().value(
        _forecast(),
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
        share_repurchase_parameters=ShareRepurchaseParameters(
            annual_cash_amounts=(Decimal("100"), Decimal("100")),
            initial_purchase_price=Decimal("100"),
            price_growth_rate=Decimal("10"),
            discount_rate=Decimal("10"),
        ),
    )

    repurchases = result.share_repurchases
    assert repurchases is not None
    assert repurchases.value_per_remaining_share > result.value_per_share
    assert repurchases.accretion_percentage > 0
    assert any("accretive" in warning for warning in result.warnings)


def test_buyback_schedule_cannot_exceed_explicit_forecast_horizon():
    with pytest.raises(ValueError, match="exceeds the explicit forecast horizon"):
        FcffDcfService().value(
            _forecast(),
            FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
            _capital_bridge(),
            share_repurchase_parameters=ShareRepurchaseParameters(
                annual_cash_amounts=(Decimal("1"), Decimal("1"), Decimal("1")),
            ),
        )


def test_fcff_dcf_rejects_invalid_forecast_or_terminal_economics():
    forecast = _forecast()
    forecast.observations[-1].fcff = Decimal("-1")
    with pytest.raises(ValueError, match="cannot be negative"):
        FcffDcfService().value(
            forecast,
            FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
            _capital_bridge(),
        )
    with pytest.raises(ValueError, match="must exceed"):
        FcffDcfParameters(wacc="2", perpetual_growth_rate="2")


def test_capital_bridge_resolves_normalized_net_debt_and_diluted_shares():
    financials = _financials_with_bridge()
    result = FcffDcfCapitalBridgeResolver().resolve(
        financials,
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
    )

    assert result.gross_debt == Decimal("100")
    assert result.cash_and_equivalents == Decimal("25")
    assert result.net_debt == Decimal("75")
    assert result.diluted_shares == Decimal("9")
    assert "current shares outstanding" in result.shares_source


def test_capital_bridge_prefers_latest_coherent_quarterly_balance_over_annual():
    financials = _financials_with_bridge()
    financials.observations = [
        observation.model_copy(
            update={
                "fiscal_year": 2026,
                "period_end": datetime.date(2026, 12, 31),
            }
        )
        for observation in financials.observations
    ]
    financials.observations.extend(
        _bridge_period(
            fiscal_year=2026,
            fiscal_period=FiscalPeriod.Q3,
            period_end=datetime.date(2026, 9, 30),
            values={
                FinancialConcept.SHORT_TERM_DEBT: "3",
                FinancialConcept.LONG_TERM_DEBT_CURRENT: "7",
                FinancialConcept.LONG_TERM_DEBT_NONCURRENT: "50",
                FinancialConcept.CASH_AND_EQUIVALENTS: "11",
                FinancialConcept.SHARES_OUTSTANDING: "8",
            },
        )
    )

    result = FcffDcfCapitalBridgeResolver().resolve(
        financials,
        fiscal_year=2026,
        period_end=datetime.date(2026, 12, 31),
        unit="USD",
    )

    assert result.period_end == datetime.date(2026, 9, 30)
    assert result.gross_debt == Decimal("60")
    assert result.cash_and_equivalents == Decimal("11")
    assert result.diluted_shares == Decimal("8")
    assert not any("latest annual" in warning for warning in result.warnings)


def test_capital_bridge_duplicate_concept_date_uses_latest_filed_observation():
    financials = _financials_with_bridge()
    latest_cash = _bridge_observation(
        FinancialConcept.CASH_AND_EQUIVALENTS,
        "40",
        Granularity.ANNUAL,
        2025,
        FiscalPeriod.FY,
        datetime.date(2025, 12, 31),
        filed=datetime.date(2026, 2, 1),
    )
    financials.observations = [latest_cash, *financials.observations]

    result = FcffDcfCapitalBridgeResolver().resolve(
        financials,
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
        valuation_date=datetime.date(2026, 2, 2),
    )

    assert result.cash_and_equivalents == Decimal("40")
    assert result.net_debt == Decimal("60")


def test_capital_bridge_prefers_quarterly_current_shares_to_annual_shares():
    financials = _financials_with_bridge()
    financials.observations.extend(
        _bridge_period(
            fiscal_year=2026,
            fiscal_period=FiscalPeriod.Q1,
            period_end=datetime.date(2026, 3, 31),
            values={
                FinancialConcept.SHORT_TERM_DEBT: "10",
                FinancialConcept.LONG_TERM_DEBT_CURRENT: "20",
                FinancialConcept.LONG_TERM_DEBT_NONCURRENT: "70",
                FinancialConcept.CASH_AND_EQUIVALENTS: "25",
                FinancialConcept.SHARES_OUTSTANDING: "8",
                FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES: "12",
            },
        )
    )

    result = FcffDcfCapitalBridgeResolver().resolve(
        financials,
        fiscal_year=2026,
        period_end=datetime.date(2026, 3, 31),
        unit="USD",
    )

    assert result.diluted_shares == Decimal("8")
    assert "current shares outstanding" in result.shares_source
    assert result.shares_date == datetime.date(2026, 3, 31)


def test_capital_bridge_accepts_reported_debt_when_short_term_line_is_absent():
    financials = _financials_with_bridge()
    financials.observations = [
        observation
        for observation in financials.observations
        if observation.concept != FinancialConcept.SHORT_TERM_DEBT
    ]

    result = FcffDcfCapitalBridgeResolver().resolve(
        financials,
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
    )

    assert result.gross_debt == Decimal("90")
    assert result.net_debt == Decimal("65")


def test_capital_bridge_missing_data_can_be_supplied_explicitly():
    financials = NormalizedCompanyFinancials(
        provider="test",
        company_id="1",
        company_name="Example",
        ticker="EX",
    )
    resolver = FcffDcfCapitalBridgeResolver()
    with pytest.raises(ValueError, match="--net-debt"):
        resolver.resolve(
            financials,
            fiscal_year=2025,
            period_end=datetime.date(2025, 12, 31),
            unit="USD",
        )

    supplied = resolver.resolve(
        financials,
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
        net_debt=Decimal("-20"),
        diluted_shares=Decimal("10"),
    )
    assert supplied.net_debt == Decimal("-20")
    assert supplied.gross_debt is None
    assert supplied.shares_source == "explicit profile or CLI override"

    supplied_components = resolver.resolve(
        financials,
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
        gross_debt=Decimal("100"),
        cash_and_equivalents=Decimal("25"),
        diluted_shares=Decimal("10"),
    )
    assert supplied_components.net_debt == Decimal("75")
    assert supplied_components.gross_debt == Decimal("100")
    assert supplied_components.net_debt_source.startswith("explicit gross debt")


def test_yahoo_bridge_uses_aggregate_current_debt_without_current_portion():
    financials = _financials_with_bridge()
    financials.provider = "yahoo"
    financials.observations = [
        item
        for item in financials.observations
        if item.concept != FinancialConcept.LONG_TERM_DEBT_CURRENT
    ]

    result = FcffDcfCapitalBridgeResolver().resolve(
        financials,
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
    )

    assert result.gross_debt == Decimal("80")
    assert result.net_debt == Decimal("55")
    assert "aggregate CurrentDebt" in result.net_debt_source


def test_cli_runs_fcff_dcf_from_profile_and_cached_financials(tmp_path, capsys):
    _cache_aapl(tmp_path)
    profile = tmp_path / "dcf.json"
    profile.write_text(
        json.dumps(
            {
                "name": "dcf-test",
                "valuation": {
                    "discount_rates": {"wacc": "8"},
                    "terminal_value": {"perpetual_growth_rate": "2"},
                },
            }
        ),
        encoding="utf-8",
    )

    arguments = [
        "valuation",
        "--ticker",
        "AAPL",
        "--years",
        "2",
        "--profile",
        str(profile),
        "--cache-dir",
        str(tmp_path),
        "--user-agent",
        "Edgarito Tests (tests@example.com)",
        "--excel-output",
        str(tmp_path / "AAPL.xlsx"),
    ]
    exit_code = main(arguments)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Model: FCFF DCF" in output
    assert "WACC:                    8.00%" in output
    assert "FY2026E FCFF" in output
    assert "FY2027E FCFF" in output
    assert "INTRINSIC VALUATION" in output
    assert "EV → EQUITY BRIDGE" in output
    assert "Intrinsic value/share" in output
    assert "Exported valuation Excel workbook" in output
    assert (tmp_path / "AAPL.xlsx").exists()
    assert "Net debt source:" not in output

    assert main([*arguments, "--audit"]) == 0
    audit_output = capsys.readouterr().out
    assert "ASSUMPTION AND PROVENANCE AUDIT" in audit_output
    assert "Net debt source: gross debt - cash and equivalents" in audit_output


def _forecast() -> FcffForecast:
    observations = [
        _forecast_observation(1, 2026, "100", "140", "20"),
        _forecast_observation(2, 2027, "110", "150", "20"),
    ]
    return FcffForecast(
        provider="test",
        company_id="1",
        company_name="Example",
        ticker="EX",
        base_fiscal_year=2025,
        base_period_end=datetime.date(2025, 12, 31),
        base_revenue=Decimal("900"),
        base_operating_income=Decimal("130"),
        base_tax_rate=Decimal("20"),
        base_nopat=Decimal("104"),
        base_depreciation_and_amortization=Decimal("20"),
        base_capital_expenditures=Decimal("30"),
        base_operating_working_capital=Decimal("100"),
        base_fcff=Decimal("90"),
        unit="USD",
        parameters=FcffForecastParameters(
            forecast_years=2,
            revenue_growth="5",
            operating_margin="15",
            tax_rate="20",
            depreciation_to_revenue="2",
            capex_to_revenue="3",
            operating_working_capital_to_revenue="10",
        ),
        historical_fiscal_years=(2024, 2025),
        assumption_sources={},
        observations=observations,
    )


def _consistent_forecast() -> FcffForecast:
    forecast = _forecast()
    forecast.observations = [
        observation.model_copy(
            update={
                "revenue": revenue,
                "operating_income": operating_income,
                "nopat": nopat,
                "depreciation_and_amortization": depreciation,
                "capital_expenditures": capex,
                "operating_working_capital": working_capital,
                "change_in_operating_working_capital": change_working_capital,
                "fcff": fcff,
            }
        )
        for observation, revenue, operating_income, nopat, depreciation, capex, working_capital, change_working_capital, fcff in (
            (
                forecast.observations[0],
                Decimal("945"),
                Decimal("141.75"),
                Decimal("113.4"),
                Decimal("18.9"),
                Decimal("28.35"),
                Decimal("94.5"),
                Decimal("-5.5"),
                Decimal("109.45"),
            ),
            (
                forecast.observations[1],
                Decimal("992.25"),
                Decimal("148.8375"),
                Decimal("119.07"),
                Decimal("19.845"),
                Decimal("29.7675"),
                Decimal("99.225"),
                Decimal("4.725"),
                Decimal("104.4225"),
            ),
        )
    ]
    return forecast


def _ytd_forecast() -> FcffForecast:
    forecast = _consistent_forecast()
    anchor = FcffForecastYtdAnchor(
        fiscal_year=2026,
        ytd_period_end=datetime.date(2026, 6, 30),
        fiscal_year_end=datetime.date(2026, 12, 31),
        actual_quarters=2,
        actual_revenue=Decimal("500"),
        actual_operating_income=Decimal("60"),
        actual_pretax_income=Decimal("40"),
        actual_income_tax_expense=Decimal("10"),
        actual_tax_rate=Decimal("25"),
        actual_depreciation_and_amortization=Decimal("8"),
        actual_capital_expenditures=Decimal("12"),
        actual_operating_working_capital=Decimal("40"),
        latest_annual_revenue=Decimal("1000"),
        revenue_growth=Decimal("5"),
        operating_margin=Decimal("15"),
        tax_rate=Decimal("20"),
        depreciation_to_revenue=Decimal("2"),
        capex_to_revenue=Decimal("3"),
        operating_working_capital_to_revenue=Decimal("10"),
    )
    stub = FcffForecastDcfStub(
        fiscal_year=2026,
        period_start=datetime.date(2026, 6, 30),
        period_end=datetime.date(2026, 12, 31),
        unit="USD",
        annual_nopat=Decimal("113.4"),
        actual_ytd_nopat=Decimal("45"),
        annual_depreciation_and_amortization=Decimal("18.9"),
        actual_ytd_depreciation_and_amortization=Decimal("8"),
        annual_capital_expenditures=Decimal("28.35"),
        actual_ytd_capital_expenditures=Decimal("12"),
        fiscal_year_end_operating_working_capital=Decimal("94.5"),
        actual_ytd_operating_working_capital=Decimal("40"),
        fcff=Decimal("8.45"),
    )
    return forecast.model_copy(
        update={
            "seed_type": ForecastSeedType.YTD_PLUS_FORECAST,
            "seed_period_end": datetime.date(2026, 6, 30),
            "base_period_end": datetime.date(2026, 6, 30),
            "current_fiscal_year": 2026,
            "actual_quarters": 2,
            "ytd_anchor": anchor,
            "dcf_stub": stub,
        }
    )


def _ytd_capital_bridge() -> FcffDcfCapitalBridge:
    return FcffDcfCapitalBridge(
        fiscal_year=2026,
        period_end=datetime.date(2026, 6, 30),
        unit="USD",
        net_debt=Decimal("100"),
        diluted_shares=Decimal("10"),
        net_debt_source="test",
        shares_source="test",
    )


def _forecast_observation(
    year: int,
    fiscal_year: int,
    fcff: str,
    operating_income: str,
    depreciation: str,
) -> FcffForecastObservation:
    return FcffForecastObservation(
        forecast_year=year,
        fiscal_year=fiscal_year,
        period_end=datetime.date(fiscal_year, 12, 31),
        revenue_growth=Decimal("5"),
        revenue=Decimal("1000") + Decimal(100 * year),
        operating_margin=Decimal("15"),
        operating_income=Decimal(operating_income),
        tax_rate=Decimal("20"),
        nopat=Decimal(operating_income) * Decimal("0.8"),
        depreciation_to_revenue=Decimal("2"),
        depreciation_and_amortization=Decimal(depreciation),
        capex_to_revenue=Decimal("3"),
        capital_expenditures=Decimal("30"),
        operating_working_capital_to_revenue=Decimal("10"),
        operating_working_capital=Decimal("110"),
        change_in_operating_working_capital=Decimal("10"),
        fcff=Decimal(fcff),
        unit="USD",
    )


def _capital_bridge() -> FcffDcfCapitalBridge:
    return FcffDcfCapitalBridge(
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
        net_debt=Decimal("100"),
        diluted_shares=Decimal("10"),
        net_debt_source="test",
        shares_source="test",
    )


def _financials_with_bridge() -> NormalizedCompanyFinancials:
    values = {
        FinancialConcept.SHORT_TERM_DEBT: "10",
        FinancialConcept.LONG_TERM_DEBT_CURRENT: "20",
        FinancialConcept.LONG_TERM_DEBT_NONCURRENT: "70",
        FinancialConcept.CASH_AND_EQUIVALENTS: "25",
        FinancialConcept.SHARES_OUTSTANDING: "9",
        FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES: "10",
    }
    return NormalizedCompanyFinancials(
        provider="test",
        company_id="1",
        company_name="Example",
        ticker="EX",
        observations=[
            FinancialObservation(
                concept=concept,
                statement=concept.statement,
                value=Decimal(value),
                unit=("shares" if "shares" in concept.value else "USD"),
                granularity=Granularity.ANNUAL,
                fiscal_year=2025,
                fiscal_period=FiscalPeriod.FY,
                period_end=datetime.date(2025, 12, 31),
                provider="test",
                taxonomy="test",
                source_concept=concept.value,
            )
            for concept, value in values.items()
        ],
    )


def _bridge_period(
    *,
    fiscal_year: int,
    fiscal_period: FiscalPeriod,
    period_end: datetime.date,
    values: dict[FinancialConcept, str],
) -> list[FinancialObservation]:
    granularity = (
        Granularity.ANNUAL
        if fiscal_period == FiscalPeriod.FY
        else Granularity.QUARTERLY
    )
    return [
        _bridge_observation(
            concept,
            value,
            granularity,
            fiscal_year,
            fiscal_period,
            period_end,
        )
        for concept, value in values.items()
    ]


def _bridge_observation(
    concept: FinancialConcept,
    value: str,
    granularity: Granularity,
    fiscal_year: int,
    fiscal_period: FiscalPeriod,
    period_end: datetime.date,
    *,
    filed: datetime.date | None = None,
) -> FinancialObservation:
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(value),
        unit="shares" if "shares" in concept.value else "USD",
        granularity=granularity,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        period_end=period_end,
        provider="test",
        taxonomy="test",
        source_concept=concept.value,
        filed=filed,
    )


def _cache_aapl(cache_dir: Path) -> None:
    ticker_path = (
        cache_dir
        / "providers"
        / "edgar"
        / "www.sec.gov"
        / "files"
        / "company_tickers.json"
    )
    facts_path = (
        cache_dir
        / "providers"
        / "edgar"
        / "data.sec.gov"
        / "api"
        / "xbrl"
        / "companyfacts"
        / "CIK0000320193.json"
    )
    ticker_path.parent.mkdir(parents=True)
    facts_path.parent.mkdir(parents=True)
    ticker_path.write_text(
        json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}),
        encoding="utf-8",
    )
    fixture = ROOT / "tests" / "fixtures" / "aapl_facts.json"
    facts_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
