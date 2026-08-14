import json
from collections.abc import Callable, Mapping


class FakeResponse:
    def __init__(self, data=None, *, content=None, status=200, headers=None):
        self._content = content
        self._data = data
        self.status = status
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def text(self, **kwargs):
        if self._content is not None:
            return self._content
        return json.dumps(self._data)


class FakeSession:
    def __init__(self, responses, *, key: Callable | None = None, sequence_keys=()):
        self.responses = responses
        self.key = key
        self.sequence_keys = set(sequence_keys)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if isinstance(self.responses, Mapping):
            response_key = self.key(url, kwargs) if self.key else url
            response = self.responses[response_key]
        else:
            response = self.responses.pop(0)
        if isinstance(response, list) and (
            not isinstance(self.responses, Mapping)
            or response_key in self.sequence_keys
        ):
            response = response.pop(0)
        if isinstance(response, FakeResponse):
            return response
        return FakeResponse(response)
