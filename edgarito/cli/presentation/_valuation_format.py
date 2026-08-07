from __future__ import annotations

import re
from decimal import Decimal

SECTION_WIDTH = 80


def section(title: str) -> list[str]:
    rule = "=" * SECTION_WIDTH
    return [rule, title.upper(), rule]


def subsection(title: str) -> list[str]:
    return [title.upper(), "-" * len(title)]


def format_currency(value: Decimal, currency: str) -> str:
    return f"{value:,.2f} {currency}"


def format_percent(value: Decimal | None, *, signed: bool = False) -> str:
    if value is None:
        return "unavailable"
    sign = "+" if signed else ""
    return f"{value:{sign},.1f}%"


def format_ratio_percent(value: Decimal | None, *, signed: bool = False) -> str:
    if value is None:
        return "unavailable"
    sign = "+" if signed else ""
    return f"{value:{sign},.1%}"


def format_multiple(value: Decimal | None) -> str:
    return f"{value:,.2f}x" if value is not None else "unavailable"


def warning_severity(message: str) -> str:
    normalized = message.casefold()
    if any(
        marker in normalized
        for marker in (
            "enterprise value does not cover",
            "no current or diluted share count",
            "requires market cap",
            "valuation unavailable",
            "valuation skipped",
            "decision analysis unavailable",
            "decision analysis skipped",
        )
    ):
        return "HIGH"
    if any(
        marker in normalized
        for marker in (
            "conservative 45-day",
            "reconstructed from ltm",
            "held flat",
            "methodology",
        )
    ):
        return "INFO"
    if any(
        marker in normalized
        for marker in (
            "quarterly capital-bridge data were unavailable",
            "estimated filing-availability",
            "historical snapshots",
            "annual balance-sheet values",
        )
    ):
        return "LOW"
    if any(
        marker in normalized
        for marker in (
            "before the",
            "only ",
            "insufficient",
            "low confidence",
            "low precision",
            "fallback",
            "capped",
            "unavailable",
            "stale",
        )
    ):
        return "MED"
    return "MED"


def short_warning(message: str) -> str:
    normalized = message.casefold()
    replacements = (
        (
            "quarterly capital-bridge data were unavailable",
            "Quarterly capital-bridge data unavailable; using annual values.",
        ),
        (
            "capital bridge is dated",
            "Capital bridge predates the valuation date.",
        ),
        (
            "yahoo statements do not expose filing dates",
            "Historical snapshots use estimated filing-availability lags.",
        ),
        (
            "historical premium observations are reconstructed",
            "Historical premiums use reconstructed LTM multiples.",
        ),
        (
            "premium mean reversion is estimated from only",
            "Premium mean-reversion estimate uses a small sample.",
        ),
        (
            "historical peer-premium variation or sample size is insufficient",
            "AR(1) persistence falls back to the configured prior.",
        ),
        (
            "projected net debt and diluted shares are held flat",
            "Projected net debt and diluted shares are held flat.",
        ),
    )
    for marker, replacement in replacements:
        if marker in normalized:
            return replacement
    compact = re.sub(r"\s+", " ", message).strip()
    if not compact.endswith((".", "!", "?")):
        compact += "."
    return compact


def unique_warnings(messages: list[str] | tuple[str, ...]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for message in messages:
        normalized = re.sub(r"\s+", " ", message).strip()
        key = normalized.casefold().rstrip(".")
        if key and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique
