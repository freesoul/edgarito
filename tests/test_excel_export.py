import datetime
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest

import edgarito.cli.__main__ as cli_main
from edgarito.cli import main
from edgarito.cli.parser import build_parser
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
    ObservationDerivationKind,
)
from edgarito.services.export import (
    CompanyAnalysisReportService,
    ExcelRenderer,
    FinancialDataExportService,
    MetricsExportService,
)
from edgarito.services.metrics import (
    CompanyMetrics,
    FinancialMetric,
    MetricObservation,
)

XML_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _source() -> NormalizedCompanyFinancials:
    return NormalizedCompanyFinancials(
        provider="sec",
        company_id="320193",
        company_name="Apple Inc.",
        ticker="AAPL",
        retrieved_at=datetime.datetime(2025, 2, 2, 12, 0, tzinfo=datetime.timezone.utc),
        observations=[
            FinancialObservation(
                concept=FinancialConcept.REVENUE,
                statement=FinancialConcept.REVENUE.statement,
                value=Decimal("123.45"),
                unit="USD",
                granularity=Granularity.ANNUAL,
                fiscal_year=2024,
                fiscal_period=FiscalPeriod.FY,
                period_start=datetime.date(2024, 1, 1),
                period_end=datetime.date(2024, 12, 31),
                provider="sec",
                taxonomy="us-gaap",
                source_concept="Revenues",
                accession_number="0000000000-25-000001",
                form="10-K",
                filed=datetime.date(2025, 2, 1),
                derivation_kind=ObservationDerivationKind.PERIOD_RECONSTRUCTION,
                derivation="test derivation",
            )
        ],
    )


def _report():
    financials = _source()
    metrics = CompanyMetrics(
        provider="sec",
        company_id=financials.company_id,
        company_name=financials.company_name,
        ticker=financials.ticker,
        observations=[
            MetricObservation(
                metric=FinancialMetric.OPERATING_MARGIN,
                value=Decimal("12.5"),
                unit="%",
                granularity=Granularity.ANNUAL,
                fiscal_year=2024,
                fiscal_period=FiscalPeriod.FY,
                period_end=datetime.date(2024, 12, 31),
                provider="sec",
                formula="100 × operating income / revenue",
                input_concepts=(
                    FinancialConcept.OPERATING_INCOME,
                    FinancialConcept.REVENUE,
                ),
            )
        ],
    )
    return CompanyAnalysisReportService().compose(
        financials=FinancialDataExportService().export(financials),
        metrics=MetricsExportService().export(metrics),
    )


def _cell_value(cell, shared_strings):
    value = cell.find("main:v", XML_NS)
    if value is None:
        return None
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value.text)]
    return value.text


def _worksheet_cells(archive, filename):
    strings = []
    if "xl/sharedStrings.xml" in archive.namelist():
        shared = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = [
            "".join(item.itertext()) for item in shared.findall("main:si", XML_NS)
        ]
    root = ElementTree.fromstring(archive.read(filename))
    return {
        cell.attrib["r"]: _cell_value(cell, strings)
        for cell in root.findall(".//main:c", XML_NS)
    }


def test_excel_renderer_writes_canonical_data_with_sheets_headers_and_typed_values(
    tmp_path,
):
    output = tmp_path / "nested" / "AAPL.xlsx"

    ExcelRenderer().render(_report(), output)

    with ZipFile(output) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        names = [
            sheet.attrib["name"]
            for sheet in workbook.findall("main:sheets/main:sheet", XML_NS)
        ]
        assert names == ["Financial Data", "Metrics"]

        financial_cells = _worksheet_cells(archive, "xl/worksheets/sheet1.xml")
        metrics_cells = _worksheet_cells(archive, "xl/worksheets/sheet2.xml")

        assert financial_cells["A1"] == "Concept"
        assert financial_cells["C1"] == "Value"
        assert financial_cells["I1"] == "Period End"
        assert financial_cells["L1"] == "Source Concept"
        assert financial_cells["Q1"] == "Derivation"
        assert financial_cells["A2"] == "revenue"
        assert financial_cells["C2"] == "123.45"
        assert financial_cells["I2"] is not None
        assert financial_cells["L2"] == "Revenues"
        assert financial_cells["Q2"] == "test derivation"

        assert metrics_cells["A1"] == "Metric"
        assert metrics_cells["I1"] == "Formula"
        assert metrics_cells["J1"] == "Input Concepts"
        assert metrics_cells["A2"] == "operating_margin"
        assert metrics_cells["B2"] == "12.5"
        assert metrics_cells["I2"] == "100 × operating income / revenue"
        assert metrics_cells["J2"] == "operating_income, revenue"


def test_excel_renderer_rejects_decimal_outside_excel_numeric_range(tmp_path):
    report = _report()
    observation = report.financial_data.observations[0].model_copy(
        update={"value": Decimal("1e1000")}
    )
    financial_data = report.financial_data.model_copy(
        update={"observations": (observation,)}
    )
    report = report.model_copy(update={"financial_data": financial_data})

    with pytest.raises(ValueError, match="outside Excel numeric range"):
        ExcelRenderer().render(report, tmp_path / "AAPL.xlsx")


def test_export_parser_and_orchestration_retrieve_all_concepts_from_one_snapshot(
    monkeypatch, tmp_path, capsys
):
    args = build_parser().parse_args(
        [
            "export",
            "--ticker",
            "AAPL",
            "--period",
            "all",
            "--provider",
            "sec",
            "--refresh",
            "--output",
            str(tmp_path / "AAPL.xlsx"),
        ]
    )
    assert args.command == "export"
    assert args.period == "all"
    assert args.output == tmp_path / "AAPL.xlsx"

    source = _source()
    calls = {}

    async def fake_retrieve(args, granularity, concepts):
        calls["retrieve"] = (granularity, concepts)
        return source

    class FakeMetricsService:
        def calculate(self, financials, *, granularity, metrics=None):
            assert financials is source
            calls["metrics"] = (granularity, metrics)
            return CompanyMetrics(
                provider=financials.provider,
                company_id=financials.company_id,
                company_name=financials.company_name,
                ticker=financials.ticker,
            )

    class FakeRenderer:
        def render(self, report, output):
            calls["render"] = (report, output)
            return Path(output)

    monkeypatch.setattr(cli_main, "_retrieve_financials", fake_retrieve)
    monkeypatch.setattr(cli_main, "FinancialMetricsService", FakeMetricsService)
    monkeypatch.setattr(cli_main, "ExcelRenderer", FakeRenderer)

    assert (
        main(
            [
                "export",
                "--ticker",
                "AAPL",
                "--period",
                "all",
                "--output",
                str(tmp_path / "AAPL.xlsx"),
            ]
        )
        == 0
    )

    assert calls["retrieve"] == (None, None)
    assert calls["metrics"] == (None, None)
    report, output = calls["render"]
    assert report.financial_data is not None
    assert report.metrics is not None
    assert report.financial_data.observations[0].value == Decimal("123.45")
    assert output == tmp_path / "AAPL.xlsx"
    assert f"Exported Excel workbook to {output}" in capsys.readouterr().out
