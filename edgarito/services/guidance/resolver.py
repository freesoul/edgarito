from __future__ import annotations

import datetime
from dataclasses import dataclass

from edgarito.schemas.guidance.management import (
    GuidanceStatus,
    ManagementGuidance,
)


@dataclass(frozen=True)
class ResolvedManagementGuidance:
    records: tuple[ManagementGuidance, ...]
    warnings: tuple[str, ...] = ()


class ManagementGuidanceResolver:
    """Resolve a conservative current view from validated guidance history."""

    def resolve(
        self,
        records: list[ManagementGuidance] | tuple[ManagementGuidance, ...],
        *,
        as_of: datetime.date,
    ) -> ResolvedManagementGuidance:
        eligible = [
            record
            for record in records
            if record.evidence_verified and record.filing_date <= as_of
        ]
        if not eligible:
            return ResolvedManagementGuidance(())

        latest_date = max(record.filing_date for record in eligible)
        latest = [record for record in eligible if record.filing_date == latest_date]
        withdrawn_keys = {
            self._key(record)
            for record in latest
            if record.status == GuidanceStatus.WITHDRAWN
        }
        current: dict[tuple, ManagementGuidance] = {}
        warnings: list[str] = []
        for record in sorted(latest, key=self._source_strength, reverse=True):
            key = self._key(record)
            if key in withdrawn_keys or record.status == GuidanceStatus.WITHDRAWN:
                continue
            previous = current.get(key)
            if previous is None:
                current[key] = record
                continue
            if self._values(previous) != self._values(record):
                warnings.append(
                    f"Conflicting {record.metric.value} guidance for "
                    f"{record.period_label}; retained {previous.source_document} "
                    "without averaging"
                )
        return ResolvedManagementGuidance(
            records=tuple(current.values()),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _key(record: ManagementGuidance) -> tuple:
        return (
            record.metric,
            record.fiscal_year,
            record.fiscal_quarter,
            record.period_type,
            record.scope,
            record.segment_name,
            record.basis,
        )

    @staticmethod
    def _values(record: ManagementGuidance) -> tuple:
        return record.point, record.low, record.high, record.currency, record.unit

    @staticmethod
    def _source_strength(record: ManagementGuidance) -> int:
        document_type = record.source_document_type.upper()
        filename = record.source_document.casefold()
        if document_type == "EX-99.1" and "presentation" not in filename:
            return 3
        if document_type.startswith("EX-99"):
            return 2
        return 1
