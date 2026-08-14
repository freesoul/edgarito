"""Shared helpers for versioned, packaged JSON data configuration."""

from __future__ import annotations

import hashlib
import json
import sysconfig
from pathlib import Path
from typing import Any


def resolve_config_path(
    relative_path: str | Path,
    explicit_path: str | Path | None,
    *,
    description: str,
) -> Path:
    """Resolve a checked-out or installed data file without a silent fallback."""

    relative = Path(relative_path)
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"{description} not found: {candidate}")
        return candidate

    source_checkout = Path(__file__).resolve().parents[2] / relative
    installed_data = Path(sysconfig.get_path("data")) / relative
    for candidate in (source_checkout, installed_data):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{description} is unavailable; checked {source_checkout} and {installed_data}"
    )


def read_config_payload(
    relative_path: str | Path,
    explicit_path: str | Path | None,
    *,
    description: str,
) -> tuple[dict[str, Any], Path, str, str]:
    """Read one JSON object and return its payload, source, text, and digest."""

    path = resolve_config_path(
        relative_path,
        explicit_path,
        description=description,
    )
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read {description}: {path}") from exc
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {description} {path}: "
            f"line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return payload, path, content, digest
