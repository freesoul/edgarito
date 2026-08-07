import os
import subprocess
import sys


def _settings_probe(tmp_path, env):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from edgarito.settings import (OPENAI_API_KEY, OPENAI_MODEL, "
                "OPENAI_REASONING_EFFORT); "
                "print(bool(OPENAI_API_KEY), OPENAI_MODEL, OPENAI_REASONING_EFFORT)"
            ),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_lowercase_project_openai_key_is_loaded_with_defaults(tmp_path):
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_MODEL", None)
    env.pop("OPENAI_REASONING_EFFORT", None)
    env["openai_secret_api_key"] = "test-only-sentinel"

    assert _settings_probe(tmp_path, env) == "True gpt-5.6-luna low"


def test_missing_openai_keys_cleanly_disable_configuration(tmp_path):
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("openai_secret_api_key", None)

    assert _settings_probe(tmp_path, env).startswith("False ")
