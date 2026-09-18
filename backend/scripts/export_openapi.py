#!/usr/bin/env python3
"""Export the OpenAPI schema for SDK generation (Wave 5, D2).

Importing ``server`` runs AxiomConfig's import-time validation, which
requires POSTGRES_URL and the API-key env vars to be PRESENT — values are
never dialed at import, the same posture as tests/conftest.py and the CI
dummy-env blocks. This script seals harmless dummies for anything unset so
it can run in any environment (CI export job, local dev) without leaking
or requiring real credentials:

    python scripts/export_openapi.py [output.json]

Writes OpenAPI 3.1 JSON (default: backend/openapi.json, git-ignored). CI
pins the client generator separately — inline-pin precedent (ruff/mypy
jobs) — and uploads the generated client as an artifact.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Presence-only validators (conftest.py pattern): dummies, never real creds.
DUMMY_ENV = {
    "ANTHROPIC_API_KEY": "ci-dummy-anthropic-key",
    "OPENAI_API_KEY": "ci-dummy-openai-key",
    "POSTGRES_URL": "postgresql+psycopg://dummy:dummy@localhost:5432/dummy",
    "API_KEY": "ci-dummy-api-key",
}


def main(argv: list[str]) -> int:
    for key, value in DUMMY_ENV.items():
        os.environ.setdefault(key, value)

    sys.path.insert(0, str(BACKEND_DIR))  # CWD-independent `import server`
    from server import app

    spec = app.openapi()
    out_path = Path(argv[0]) if argv else BACKEND_DIR / "openapi.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(f"Exported {len(spec.get('paths', {}))} paths "
          f"(app version {spec['info']['version']}) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
