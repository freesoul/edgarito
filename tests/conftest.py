import pytest


@pytest.fixture(autouse=True)
def _disable_real_openai_for_tests(monkeypatch):
    """The normal test suite must never make paid external model requests."""
    monkeypatch.setattr("edgarito.cli.__main__.OPENAI_API_KEY", None)
