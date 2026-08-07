from __future__ import annotations

import re

from edgarito.schemas.identifiers import SecurityIdentifiers

_LEGAL_SUFFIXES = {
    "a",
    "ab",
    "adr",
    "adrs",
    "ads",
    "ag",
    "b",
    "bv",
    "cdr",
    "cedear",
    "cedears",
    "class",
    "common",
    "corporation",
    "corp",
    "depository",
    "depositary",
    "gdr",
    "inc",
    "kgaa",
    "limited",
    "llc",
    "llp",
    "ltd",
    "nv",
    "ordinary",
    "ord",
    "oyj",
    "plc",
    "receipt",
    "receipts",
    "registered",
    "registry",
    "sa",
    "sarl",
    "se",
    "share",
    "shares",
    "sponsored",
    "spad",
    "spa",
    "stock",
    "the",
    "unsponsored",
}

_LEGAL_TOKEN_SEQUENCES = (
    ("a", "g"),
    ("a", "s"),
    ("l", "t", "d"),
    ("n", "v"),
    ("p", "l", "c"),
    ("s", "a"),
    ("s", "e"),
)


def issuer_identity_keys(
    *,
    company_id: str | None = None,
    company_name: str | None = None,
    ticker: str | None = None,
    identifiers: SecurityIdentifiers | None = None,
) -> frozenset[str]:
    """Return stable and conservative keys for one issuer's listed securities."""

    keys: set[str] = set()
    if identifiers is not None and identifiers.cik is not None:
        keys.add(f"cik:{int(identifiers.cik)}")

    normalized_company_id = _normalized(company_id)
    normalized_ticker = _normalized(ticker)
    if normalized_company_id and normalized_company_id != normalized_ticker:
        keys.add(f"company:{normalized_company_id}")
    if normalized_company_id and normalized_company_id.isdigit():
        keys.add(f"cik:{int(normalized_company_id)}")

    name_key = normalize_issuer_name(company_name, ticker=ticker)
    if name_key:
        keys.add(f"name:{name_key}")

    ticker_root = _ticker_root(ticker)
    if ticker_root:
        keys.add(f"ticker-root:{ticker_root}")
    return frozenset(keys)


def normalize_issuer_name(name: str | None, *, ticker: str | None = None) -> str:
    """Normalize legal-entity and listing suffixes without relying on ticker text."""

    if not name:
        return ""
    tokens = re.findall(r"[a-z0-9]+", name.casefold())
    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        matched = False
        for sequence in _LEGAL_TOKEN_SEQUENCES:
            if tuple(tokens[index : index + len(sequence)]) == sequence:
                index += len(sequence)
                matched = True
                break
        if matched:
            continue
        token = tokens[index]
        if token not in _LEGAL_SUFFIXES:
            normalized.append(token)
        index += 1

    ticker_root = _ticker_root(ticker)
    if ticker_root and len(normalized) > 1 and normalized[-1] == ticker_root:
        normalized.pop()
    return "".join(normalized)


def _ticker_root(ticker: str | None) -> str:
    if not ticker:
        return ""
    value = _normalized(ticker)
    return re.split(r"[.\-]", value, maxsplit=1)[0]


def _normalized(value: str | None) -> str:
    return str(value).strip().casefold() if value else ""


__all__ = ["issuer_identity_keys", "normalize_issuer_name"]
