"""Typed loaders for guidance and SEC document-selection data tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from edgarito.config._loader import read_config_payload
from edgarito.config.operating import CompiledPattern


class _GuidanceConfigModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GuidanceWeightedPatternConfiguration(_GuidanceConfigModel):
    pattern: str = Field(min_length=1)
    weight: int = Field(ge=0)


class GuidanceKeywordPatternConfiguration(_GuidanceConfigModel):
    keyword: str = Field(min_length=1)
    pattern: str = Field(min_length=1)


class GuidanceUnitScaleConfiguration(_GuidanceConfigModel):
    unit: str = Field(min_length=1)
    multiplier: Decimal


class GuidanceDocumentsConfiguration(_GuidanceConfigModel):
    schema_version: Literal[1]
    guidance_terms: tuple[str, ...]
    operating_terms: tuple[str, ...]
    current_report_forms: tuple[str, ...]
    periodic_report_forms: tuple[str, ...]
    exhibit_pattern: str = Field(min_length=1)
    guidance_context_patterns: tuple[GuidanceWeightedPatternConfiguration, ...]
    operating_context_patterns: tuple[GuidanceWeightedPatternConfiguration, ...]
    guidance_audit_patterns: tuple[GuidanceKeywordPatternConfiguration, ...]


class GuidanceExtractionConfiguration(_GuidanceConfigModel):
    schema_version: Literal[1]
    unit_scales: tuple[GuidanceUnitScaleConfiguration, ...]
    currencies: tuple[str, ...]
    number_pattern: str = Field(min_length=1)
    third_party_terms: tuple[str, ...]
    forward_terms: tuple[str, ...]
    result_terms: tuple[str, ...]


@dataclass(frozen=True)
class GuidanceDocumentsView:
    """Immutable document terms, forms, and compiled ranking patterns."""

    schema_version: int
    fingerprint: str
    source_path: Path
    guidance_terms: tuple[str, ...]
    operating_terms: tuple[str, ...]
    current_report_forms: tuple[str, ...]
    periodic_report_forms: tuple[str, ...]
    exhibit_pattern: CompiledPattern
    guidance_context_patterns: tuple[CompiledPattern, ...]
    operating_context_patterns: tuple[CompiledPattern, ...]
    guidance_audit_patterns: tuple[CompiledPattern, ...]

    @property
    def cache_version(self) -> str:
        return f"guidance-documents-v{self.schema_version}:{self.fingerprint}"


@dataclass(frozen=True)
class GuidanceExtractionView:
    """Immutable guidance scaling, currency, and evidence-term tables."""

    schema_version: int
    fingerprint: str
    source_path: Path
    unit_scales: Mapping[str, Decimal]
    currencies: frozenset[str]
    number_pattern: CompiledPattern
    third_party_terms: tuple[str, ...]
    forward_terms: tuple[str, ...]
    result_terms: tuple[str, ...]

    @property
    def cache_version(self) -> str:
        return f"guidance-extraction-v{self.schema_version}:{self.fingerprint}"


def _compile_pattern(
    pattern: str,
    *,
    weight=None,
    keyword=None,
    flags: int = re.IGNORECASE,
) -> CompiledPattern:
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"Invalid guidance configuration regex {pattern!r}") from exc
    return CompiledPattern(
        pattern=pattern,
        regex=regex,
        weight=weight,
        keyword=keyword,
    )


class GuidanceDocumentsLoader:
    """Load packaged guidance/document-selection tables."""

    PATH = Path("configs/guidance/documents.json")
    DESCRIPTION = "Guidance document configuration"

    @classmethod
    def load(cls, path: str | Path | None = None) -> GuidanceDocumentsConfiguration:
        return cls._read(path)[0]

    @classmethod
    def compiled(cls, path: str | Path | None = None) -> GuidanceDocumentsView:
        if path is None:
            return cls._default_compiled()
        return cls._compile(*cls._read(path))

    @classmethod
    def default_path(cls) -> Path:
        return cls._read(None)[1]

    @classmethod
    def _read(
        cls, path: str | Path | None
    ) -> tuple[GuidanceDocumentsConfiguration, Path, str, str]:
        payload, source, content, fingerprint = read_config_payload(
            cls.PATH,
            path,
            description=cls.DESCRIPTION,
        )
        try:
            model = GuidanceDocumentsConfiguration.model_validate(payload)
        except ValueError as exc:
            raise ValueError(f"Invalid guidance documents {source}: {exc}") from exc
        return model, source, content, fingerprint

    @classmethod
    def _compile(
        cls,
        model: GuidanceDocumentsConfiguration,
        source: Path,
        _content: str,
        fingerprint: str,
    ) -> GuidanceDocumentsView:
        def patterns(items):
            return tuple(
                _compile_pattern(
                    item.pattern,
                    weight=getattr(item, "weight", None),
                    keyword=getattr(item, "keyword", None),
                )
                for item in items
            )

        return GuidanceDocumentsView(
            schema_version=model.schema_version,
            fingerprint=fingerprint,
            source_path=source,
            guidance_terms=model.guidance_terms,
            operating_terms=model.operating_terms,
            current_report_forms=model.current_report_forms,
            periodic_report_forms=model.periodic_report_forms,
            exhibit_pattern=_compile_pattern(model.exhibit_pattern),
            guidance_context_patterns=patterns(model.guidance_context_patterns),
            operating_context_patterns=patterns(model.operating_context_patterns),
            guidance_audit_patterns=patterns(model.guidance_audit_patterns),
        )

    @classmethod
    def _default_compiled(cls) -> GuidanceDocumentsView:
        return cls._compile(*cls._read(None))


class GuidanceExtractionLoader:
    """Load packaged guidance scaling and evidence-validation tables."""

    PATH = Path("configs/guidance/extraction.json")
    DESCRIPTION = "Guidance extraction configuration"

    @classmethod
    def load(cls, path: str | Path | None = None) -> GuidanceExtractionConfiguration:
        return cls._read(path)[0]

    @classmethod
    def compiled(cls, path: str | Path | None = None) -> GuidanceExtractionView:
        if path is None:
            return cls._default_compiled()
        return cls._compile(*cls._read(path))

    @classmethod
    def default_path(cls) -> Path:
        return cls._read(None)[1]

    @classmethod
    def _read(
        cls, path: str | Path | None
    ) -> tuple[GuidanceExtractionConfiguration, Path, str, str]:
        payload, source, content, fingerprint = read_config_payload(
            cls.PATH,
            path,
            description=cls.DESCRIPTION,
        )
        try:
            model = GuidanceExtractionConfiguration.model_validate(payload)
        except ValueError as exc:
            raise ValueError(f"Invalid guidance extraction {source}: {exc}") from exc
        return model, source, content, fingerprint

    @classmethod
    def _compile(
        cls,
        model: GuidanceExtractionConfiguration,
        source: Path,
        _content: str,
        fingerprint: str,
    ) -> GuidanceExtractionView:
        return GuidanceExtractionView(
            schema_version=model.schema_version,
            fingerprint=fingerprint,
            source_path=source,
            unit_scales=MappingProxyType(
                {item.unit: item.multiplier for item in model.unit_scales}
            ),
            currencies=frozenset(model.currencies),
            number_pattern=_compile_pattern(model.number_pattern, flags=0),
            third_party_terms=model.third_party_terms,
            forward_terms=model.forward_terms,
            result_terms=model.result_terms,
        )

    @classmethod
    def _default_compiled(cls) -> GuidanceExtractionView:
        return cls._compile(*cls._read(None))


GUIDANCE_DOCUMENTS = GuidanceDocumentsLoader.compiled()
GUIDANCE_EXTRACTION = GuidanceExtractionLoader.compiled()
DEFAULT_GUIDANCE_DOCUMENTS_PATH = GuidanceDocumentsLoader.PATH
DEFAULT_GUIDANCE_EXTRACTION_PATH = GuidanceExtractionLoader.PATH


__all__ = [
    "GuidanceDocumentsConfiguration",
    "GuidanceDocumentsLoader",
    "GuidanceDocumentsView",
    "DEFAULT_GUIDANCE_DOCUMENTS_PATH",
    "DEFAULT_GUIDANCE_EXTRACTION_PATH",
    "GuidanceExtractionConfiguration",
    "GuidanceExtractionLoader",
    "GuidanceExtractionView",
    "GuidanceKeywordPatternConfiguration",
    "GuidanceUnitScaleConfiguration",
    "GuidanceWeightedPatternConfiguration",
    "GUIDANCE_DOCUMENTS",
    "GUIDANCE_EXTRACTION",
]
