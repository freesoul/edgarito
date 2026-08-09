"""Render canonical analysis snapshots as deterministic Excel workbooks."""

from __future__ import annotations

import datetime
import math
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import xlsxwriter

from .models import (
    CompanyAnalysisReport,
    FinancialDataExport,
    MetricsExport,
)
from .services import CompanyAnalysisReportService


class ExcelRenderer:
    """Write financial data and calculated metrics to an XLSX workbook.

    The renderer accepts canonical export snapshots rather than presentation-layer
    strings.  Existing output files are intentionally replaced, and missing parent
    directories are created.
    """

    _FINANCIAL_HEADERS = (
        "Concept",
        "Statement",
        "Value",
        "Unit",
        "Granularity",
        "Fiscal Year",
        "Fiscal Period",
        "Period Start",
        "Period End",
        "Provider",
        "Taxonomy",
        "Source Concept",
        "Accession Number",
        "Form",
        "Filed",
        "Derivation Kind",
        "Derivation",
    )
    _METRIC_HEADERS = (
        "Metric",
        "Value",
        "Unit",
        "Granularity",
        "Fiscal Year",
        "Fiscal Period",
        "Period End",
        "Provider",
        "Formula",
        "Input Concepts",
    )

    def render(
        self,
        source: CompanyAnalysisReport | FinancialDataExport,
        output: str | Path,
        *,
        metrics: MetricsExport | None = None,
    ) -> Path:
        """Render a report, or compose one from canonical financials and metrics."""
        report = self._report(source, metrics)
        output_path = Path(output).expanduser()
        if output_path.exists() and output_path.is_dir():
            raise ValueError(f"Excel output path is a directory: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        workbook = xlsxwriter.Workbook(str(output_path))
        try:
            workbook.set_properties(
                {
                    "title": "Edgarito financial export",
                    "subject": "Normalized financial data and calculated metrics",
                    "author": "Edgarito",
                }
            )
            formats = self._formats(workbook)
            self._write_financial_data(workbook, report.financial_data, formats)
            self._write_metrics(workbook, report.metrics, formats)
        finally:
            workbook.close()
        return output_path

    def export(
        self,
        source: CompanyAnalysisReport | FinancialDataExport,
        output: str | Path,
        *,
        metrics: MetricsExport | None = None,
    ) -> Path:
        """Alias for :meth:`render` for export-service style callers."""
        return self.render(source, output, metrics=metrics)

    def __call__(
        self,
        source: CompanyAnalysisReport | FinancialDataExport,
        output: str | Path,
        *,
        metrics: MetricsExport | None = None,
    ) -> Path:
        return self.render(source, output, metrics=metrics)

    @staticmethod
    def _report(
        source: CompanyAnalysisReport | FinancialDataExport,
        metrics: MetricsExport | None,
    ) -> CompanyAnalysisReport:
        if isinstance(source, CompanyAnalysisReport):
            if metrics is not None:
                raise TypeError("metrics cannot be supplied with a company report")
            return source
        if isinstance(source, FinancialDataExport):
            if metrics is None:
                raise ValueError("metrics are required with a financial data export")
            return CompanyAnalysisReportService().compose(
                financials=source,
                metrics=metrics,
            )
        raise TypeError("source must be CompanyAnalysisReport or FinancialDataExport")

    @staticmethod
    def _formats(workbook: xlsxwriter.Workbook) -> dict[str, Any]:
        return {
            "header": workbook.add_format(
                {
                    "bold": True,
                    "bg_color": "#1F4E78",
                    "font_color": "#FFFFFF",
                    "border": 1,
                }
            ),
            "date": workbook.add_format({"num_format": "yyyy-mm-dd"}),
            "datetime": workbook.add_format({"num_format": "yyyy-mm-dd hh:mm:ss"}),
            "number": workbook.add_format(
                {"num_format": "0.############################"}
            ),
        }

    def _write_financial_data(
        self,
        workbook: xlsxwriter.Workbook,
        export: FinancialDataExport | None,
        formats: dict[str, Any],
    ) -> None:
        worksheet = workbook.add_worksheet("Financial Data")
        self._write_table_header(worksheet, self._FINANCIAL_HEADERS, formats["header"])
        rows = () if export is None else export.observations
        for row, observation in enumerate(rows, start=1):
            values = (
                observation.concept,
                observation.statement,
                observation.value,
                observation.unit,
                observation.granularity,
                observation.fiscal_year,
                observation.fiscal_period,
                observation.period_start,
                observation.period_end,
                observation.provider,
                observation.taxonomy,
                observation.source_concept,
                observation.accession_number,
                observation.form,
                observation.filed,
                observation.derivation_kind,
                observation.derivation,
            )
            self._write_values(worksheet, row, values, formats)
        self._finish_table(worksheet, len(self._FINANCIAL_HEADERS), len(rows))

    def _write_metrics(
        self,
        workbook: xlsxwriter.Workbook,
        export: MetricsExport | None,
        formats: dict[str, Any],
    ) -> None:
        worksheet = workbook.add_worksheet("Metrics")
        self._write_table_header(worksheet, self._METRIC_HEADERS, formats["header"])
        rows = () if export is None else export.observations
        for row, observation in enumerate(rows, start=1):
            values = (
                observation.metric,
                observation.value,
                observation.unit,
                observation.granularity,
                observation.fiscal_year,
                observation.fiscal_period,
                observation.period_end,
                observation.provider,
                observation.formula,
                ", ".join(concept.value for concept in observation.input_concepts),
            )
            self._write_values(worksheet, row, values, formats)
        self._finish_table(worksheet, len(self._METRIC_HEADERS), len(rows))

    @staticmethod
    def _write_table_header(worksheet, headers, header_format) -> None:
        for column, header in enumerate(headers):
            worksheet.write_string(0, column, header, header_format)

    @staticmethod
    def _finish_table(worksheet, column_count: int, row_count: int) -> None:
        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, row_count, column_count - 1)
        widths = {
            0: 30,
            1: 20,
            2: 18,
            3: 16,
            4: 14,
            5: 12,
            6: 14,
            7: 14,
            8: 16,
            9: 16,
            10: 18,
            11: 48,
            12: 24,
            13: 14,
            14: 14,
            15: 24,
            16: 52,
        }
        for column, width in widths.items():
            if column < column_count:
                worksheet.set_column(column, column, width)

    @staticmethod
    def _write_values(worksheet, row: int, values, formats: dict[str, Any]) -> None:
        for column, value in enumerate(values):
            if value is None:
                worksheet.write_blank(row, column, None)
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
                if not value.is_finite():
                    raise ValueError("Excel exports require finite Decimal values")
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(
                        f"Decimal value {value} is outside Excel numeric range"
                    )
                worksheet.write_number(row, column, number, formats["number"])
            elif isinstance(value, Enum):
                worksheet.write_string(row, column, str(value.value))
            elif isinstance(value, str):
                worksheet.write_string(row, column, value)
            else:
                worksheet.write(row, column, value)


ExcelExportService = ExcelRenderer


def render_excel(
    report: CompanyAnalysisReport,
    output: str | Path,
) -> Path:
    """Render a canonical company analysis report as an Excel workbook."""
    return ExcelRenderer().render(report, output)


__all__ = ["ExcelExportService", "ExcelRenderer", "render_excel"]
