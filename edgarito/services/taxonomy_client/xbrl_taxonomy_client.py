import asyncio
from typing import Optional, Set
import pathlib

import aiohttp
import xml.etree.ElementTree as ET

from edgarito.services.cache.filesystem_cache import FileSystemCache


class TaxonomyClient:
    """
    If xsd dynamic linking would be needed in the future, then use xmlschema lib instead and override xmlschema.fetch
    """

    def __init__(self, cache: FileSystemCache):
        self._tree: Optional[ET.Element] = None
        self._cache = cache

    async def load(self, taxonomy_url: str, use_cache: bool = True, make_cache: bool = True) -> None:

        file_path = f"xbrl-taxonomy/{FileSystemCache.path_from_url(taxonomy_url)}"

        if use_cache:
            cached_data = self._cache.read(file_path)
            if cached_data is not None:
                self._tree = ET.fromstring(cached_data)
                return

        async with aiohttp.ClientSession() as session:
            async with session.get(taxonomy_url) as response:
                response.raise_for_status()
                taxonomy_data = await response.text()
                self._tree = ET.fromstring(taxonomy_data)

        if make_cache:
            self._cache.save(file_path, taxonomy_data)

    def get_gaap_keys(self) -> Set[str]:
        self._require_loaded()
        xs_ns = "{http://www.w3.org/2001/XMLSchema}"
        return {elem.attrib["name"] for elem in self._tree.iter(f"{xs_ns}element") if "name" in elem.attrib}

    def _require_loaded(self) -> None:
        if self._tree is None:
            raise ValueError("No taxonomy loaded")


if __name__ == "__main__":
    import asyncio

    async def main():
        cache = FileSystemCache(root_directory=pathlib.Path("./cache"))
        client = TaxonomyClient(cache=cache)
        await client.load("https://xbrl.fasb.org/us-gaap/2025/elts/us-gaap-2025.xsd")
        print(client.get_gaap_keys())

    asyncio.run(main())
