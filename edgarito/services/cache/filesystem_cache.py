import logging
from typing import Union, Optional
import pathlib

import urllib.parse


class FileSystemCache:
    def __init__(self, root_directory: Union[str, pathlib.Path]):
        self._logger = logging.getLogger(__class__.__name__)
        self._root = pathlib.Path(root_directory).absolute()
        self._root.mkdir(parents=True, exist_ok=True)
        self._logger.debug(f"Using cache directory: {self._root}")

    def read(self, path_in_cache: str) -> Optional[str]:
        file_path = self._root / path_in_cache
        if not file_path.is_file():
            self._logger.info(f"Not found in cache: {file_path}")
            return None
        self._logger.debug(f"Reading from cache: {file_path}")
        with file_path.open("r", encoding="utf-8") as file:
            return file.read()

    def save(self, path_in_cache: str, content: str) -> None:
        file_path = self._root / path_in_cache
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger.info(f"Saving to cache: {file_path}")
        with file_path.open("w", encoding="utf-8") as file:
            file.write(content)

    def remove(self, path_in_cache: str):
        file_path = self._root / path_in_cache
        self._logger.info(f"Removing from cache: {file_path}")
        if file_path.is_file():
            file_path.unlink()

    @staticmethod
    def path_from_url(url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        return f"{parsed.netloc}/{parsed.path.lstrip('/')}"
