"""Typed loader for provider-sector aliases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict

from edgarito.config._loader import read_config_payload
from edgarito.schemas.normalization.classification import Sector


class ClassificationConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    sector_aliases: dict[str, str]


@dataclass(frozen=True)
class ClassificationView:
    """Immutable source-sector alias lookup."""

    schema_version: int
    fingerprint: str
    source_path: Path
    sector_aliases: Mapping[str, Sector]

    @property
    def cache_version(self) -> str:
        return f"classification-sectors-v{self.schema_version}:{self.fingerprint}"


class ClassificationConfigurationLoader:
    """Load the packaged classification alias table."""

    PATH = Path("configs/classification/sectors.json")
    DESCRIPTION = "Classification sector configuration"

    @classmethod
    def load(cls, path: str | Path | None = None) -> ClassificationConfiguration:
        return cls._read(path)[0]

    @classmethod
    def compiled(cls, path: str | Path | None = None) -> ClassificationView:
        if path is None:
            return cls._default_compiled()
        return cls._compile(*cls._read(path))

    @classmethod
    def default_path(cls) -> Path:
        return cls._read(None)[1]

    @classmethod
    def _read(
        cls, path: str | Path | None
    ) -> tuple[ClassificationConfiguration, Path, str, str]:
        payload, source, content, fingerprint = read_config_payload(
            cls.PATH,
            path,
            description=cls.DESCRIPTION,
        )
        try:
            model = ClassificationConfiguration.model_validate(payload)
        except ValueError as exc:
            raise ValueError(f"Invalid classification sectors {source}: {exc}") from exc
        return model, source, content, fingerprint

    @classmethod
    def _compile(
        cls,
        model: ClassificationConfiguration,
        source: Path,
        _content: str,
        fingerprint: str,
    ) -> ClassificationView:
        try:
            aliases = {
                key: Sector(value) for key, value in model.sector_aliases.items()
            }
        except ValueError as exc:
            raise ValueError(
                f"Invalid classification sector alias in {source}: {exc}"
            ) from exc
        return ClassificationView(
            schema_version=model.schema_version,
            fingerprint=fingerprint,
            source_path=source,
            sector_aliases=MappingProxyType(aliases),
        )

    @classmethod
    def _default_compiled(cls) -> ClassificationView:
        return cls._compile(*cls._read(None))


ClassificationLoader = ClassificationConfigurationLoader
CLASSIFICATION = ClassificationConfigurationLoader.compiled()
DEFAULT_CLASSIFICATION_SECTORS_PATH = ClassificationConfigurationLoader.PATH


__all__ = [
    "CLASSIFICATION",
    "DEFAULT_CLASSIFICATION_SECTORS_PATH",
    "ClassificationConfiguration",
    "ClassificationConfigurationLoader",
    "ClassificationLoader",
    "ClassificationView",
]
