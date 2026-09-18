"""Version consistency: the app self-reports the package version (Wave 5, D3).

The drift this pins: the FastAPI app hardcoded version="1.0.0" while
backend/pyproject.toml declared 1.5.0. The app now derives its version from
the package — installed metadata when available, the pyproject.toml read
otherwise (same source file either way) — and this test locks the FastAPI
app, the helper, and pyproject together so they cannot drift again.
"""

import tomllib
from pathlib import Path

import server

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_version() -> str:
    with open(PYPROJECT_PATH, "rb") as fh:
        parsed = tomllib.load(fh)
    version = parsed["project"]["version"]
    assert isinstance(version, str)
    return version


def test_fastapi_app_version_matches_pyproject():
    assert server.app.version == _pyproject_version()


def test_app_version_helper_matches_pyproject():
    assert server._app_version() == _pyproject_version()
