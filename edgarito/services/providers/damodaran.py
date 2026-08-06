import datetime
import re
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from typing import Optional

import aiohttp

from edgarito.schemas.valuation.reference import (
    CountryRiskPremium,
    CountryRiskPremiumSnapshot,
    IndustryBeta,
    IndustryBetaSnapshot,
    ReferenceDatasetMetadata,
    ReferenceDatasetRelease,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.providers._reference import CachedTextProvider, RetrievedText

COUNTRY_RISK_PREMIUMS_2026 = ReferenceDatasetRelease(
    dataset="country-risk-premiums",
    version="2026-01-05",
    published_on=datetime.date(2026, 1, 5),
    source_url=(
        "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html"
    ),
    expected_sha256="ea30d57fc0858072c520930e2e5b8a5e5affdce81f7f2e1f9054efab79959dbc",
    region="global",
)

US_INDUSTRY_BETAS_2026 = ReferenceDatasetRelease(
    dataset="industry-betas",
    version="2026-01",
    published_on=datetime.date(2026, 1, 9),
    source_url=(
        "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/Betas.html"
    ),
    expected_sha256="10993bab71504e2449ccdd9e3d67fa4da983bed380b2f4486e5200a7b258616b",
    region="US",
)


class DamodaranClient(CachedTextProvider):
    """Retrieve checksum-verified, explicitly versioned Damodaran snapshots."""

    def __init__(
        self,
        cache: FileSystemCache,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(cache, session)

    async def get_country_risk_premiums(
        self,
        release: ReferenceDatasetRelease = COUNTRY_RISK_PREMIUMS_2026,
        *,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> CountryRiskPremiumSnapshot:
        self._require_dataset(release, "country-risk-premiums")
        payload = await self._retrieve_release(release, use_cache, make_cache)
        rows = _HtmlTableParser.parse(payload.content)
        header, data_rows = _find_table(rows, "Country")
        columns = _column_indexes(header)
        required = {
            "country",
            "moodysrating",
            "adjdefaultspread",
            "countryriskpremium",
            "equityriskpremium",
            "corporatetaxrate",
        }
        if not required.issubset(columns):
            raise RuntimeError("Damodaran country-risk table has unexpected columns")
        countries = []
        for row in data_rows:
            if len(row) != len(header) or not row[columns["country"]]:
                continue
            countries.append(
                CountryRiskPremium(
                    country=row[columns["country"]],
                    rating=_optional_text(row[columns["moodysrating"]]),
                    adjusted_default_spread=_required_decimal(
                        row[columns["adjdefaultspread"]]
                    ),
                    country_risk_premium=_required_decimal(
                        row[columns["countryriskpremium"]]
                    ),
                    equity_risk_premium=_required_decimal(
                        row[columns["equityriskpremium"]]
                    ),
                    corporate_tax_rate=_required_decimal(
                        row[columns["corporatetaxrate"]]
                    ),
                    sovereign_cds_spread=_optional_decimal(
                        _cell(
                            row,
                            columns,
                            "sovereignncds",
                            "sovereigncds",
                            "sovereigncdsspread",
                        )
                    ),
                    cds_equity_risk_premium=_optional_decimal(
                        _cell(row, columns, "erpbasedonsovereigncdss")
                    ),
                )
            )
        return CountryRiskPremiumSnapshot(
            metadata=self._metadata(release, payload), countries=tuple(countries)
        )

    async def get_industry_betas(
        self,
        release: ReferenceDatasetRelease = US_INDUSTRY_BETAS_2026,
        *,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> IndustryBetaSnapshot:
        self._require_dataset(release, "industry-betas")
        payload = await self._retrieve_release(release, use_cache, make_cache)
        rows = _HtmlTableParser.parse(payload.content)
        header, data_rows = _find_table(rows, "Industry Name")
        columns = _column_indexes(header)
        required = {
            "industryname",
            "numberoffirms",
            "beta",
            "deratio",
            "effectivetaxrate",
            "unleveredbeta",
            "cashfirmvalue",
            "unleveredbetacorrectedforcash",
        }
        if not required.issubset(columns):
            raise RuntimeError("Damodaran industry-beta table has unexpected columns")
        industries = []
        for row in data_rows:
            if len(row) != len(header) or not row[columns["industryname"]]:
                continue
            industries.append(
                IndustryBeta(
                    industry=row[columns["industryname"]],
                    number_of_firms=int(row[columns["numberoffirms"]].replace(",", "")),
                    levered_beta=_required_decimal(row[columns["beta"]]),
                    debt_to_equity=_required_decimal(row[columns["deratio"]]),
                    effective_tax_rate=_required_decimal(
                        row[columns["effectivetaxrate"]]
                    ),
                    unlevered_beta=_required_decimal(row[columns["unleveredbeta"]]),
                    cash_to_firm_value=_required_decimal(row[columns["cashfirmvalue"]]),
                    cash_adjusted_unlevered_beta=_required_decimal(
                        row[columns["unleveredbetacorrectedforcash"]]
                    ),
                    hilo_risk=_optional_decimal(_cell(row, columns, "hilorisk")),
                    equity_standard_deviation=_optional_decimal(
                        _cell(row, columns, "standarddeviationofequity")
                    ),
                    operating_income_standard_deviation=_optional_decimal(
                        _cell(
                            row,
                            columns,
                            "standarddeviationinoperatingincomelast10years",
                        )
                    ),
                )
            )
        return IndustryBetaSnapshot(
            metadata=self._metadata(release, payload), industries=tuple(industries)
        )

    async def _retrieve_release(
        self,
        release: ReferenceDatasetRelease,
        use_cache: bool,
        make_cache: bool,
    ) -> RetrievedText:
        safe_dataset = re.sub(r"[^a-z0-9_-]", "_", release.dataset.casefold())
        safe_version = re.sub(r"[^a-z0-9_.-]", "_", release.version.casefold())
        return await self._retrieve_text(
            provider="Damodaran",
            url=release.source_url,
            cache_path=f"providers/damodaran/{safe_dataset}/{safe_version}.json",
            expected_sha256=release.expected_sha256,
            use_cache=use_cache,
            make_cache=make_cache,
        )

    @staticmethod
    def _require_dataset(release: ReferenceDatasetRelease, expected: str) -> None:
        if release.dataset != expected:
            raise ValueError(
                f"Expected Damodaran dataset {expected!r}, got {release.dataset!r}"
            )

    @staticmethod
    def _metadata(
        release: ReferenceDatasetRelease, payload: RetrievedText
    ) -> ReferenceDatasetMetadata:
        return ReferenceDatasetMetadata(
            provider="damodaran",
            dataset=release.dataset,
            version=release.version,
            published_on=release.published_on,
            retrieved_at=payload.retrieved_at,
            source_url=release.source_url,
            sha256=payload.sha256,
            region=release.region,
        )


class _HtmlTableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: Optional[list[str]] = None
        self._cell: Optional[list[str]] = None

    @classmethod
    def parse(cls, content: str) -> list[list[str]]:
        parser = cls()
        parser.feed(content)
        return parser.rows

    def handle_starttag(self, tag: str, attrs) -> None:
        normalized = tag.casefold()
        if normalized == "tr":
            self._row = []
        elif normalized in {"td", "th"} and self._row is not None:
            self._cell = []
        elif normalized == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in {"td", "th"} and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif normalized == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None


def _find_table(
    rows: list[list[str]], first_column: str
) -> tuple[list[str], list[list[str]]]:
    for index, row in enumerate(rows):
        if row and row[0].strip().casefold() == first_column.casefold():
            return row, rows[index + 1 :]
    raise RuntimeError(f"Damodaran table header {first_column!r} was not found")


def _column_indexes(header: list[str]) -> dict[str, int]:
    return {_normalize_heading(value): index for index, value in enumerate(header)}


def _normalize_heading(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _cell(row: list[str], columns: dict[str, int], *names: str) -> str:
    for name in names:
        index = columns.get(name)
        if index is not None and index < len(row):
            return row[index]
    return ""


def _required_decimal(value: str) -> Decimal:
    parsed = _optional_decimal(value)
    if parsed is None:
        raise RuntimeError(f"Damodaran required value is missing: {value!r}")
    return parsed


def _optional_decimal(value: str) -> Optional[Decimal]:
    normalized = value.strip()
    if not normalized or normalized.casefold() in {"na", "n/a", "nm", "-"}:
        return None
    try:
        return Decimal(normalized.removesuffix("%").replace(",", ""))
    except InvalidOperation as exc:
        raise RuntimeError(f"Damodaran returned an invalid number: {value!r}") from exc


def _optional_text(value: str) -> Optional[str]:
    normalized = value.strip()
    return (
        None if not normalized or normalized.casefold() in {"na", "n/a"} else normalized
    )


__all__ = [
    "COUNTRY_RISK_PREMIUMS_2026",
    "DamodaranClient",
    "US_INDUSTRY_BETAS_2026",
]
