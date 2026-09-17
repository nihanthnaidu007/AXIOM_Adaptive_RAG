"""Programmatic alembic wiring for application startup.

The alembic migration chain is the single source of truth for the database
schema; startup applies ``upgrade head`` before any store touches the DB.

Alembic runs synchronously in its own thread (via ``asyncio.to_thread`` at the
call site): env.py builds an async engine and drives it with ``asyncio.run``,
which requires a loop-free thread.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent


def migration_database_url() -> str:
    """Resolve the migration target URL from app config, normalized to asyncpg.

    asyncpg is the only postgres driver installed (no psycopg), so plain and
    psycopg URLs are rewritten to the asyncpg driver for env.py.
    """
    from axiom.config import get_config

    cfg = get_config()
    url = cfg.postgres_url or (
        f"postgresql+psycopg://{cfg.postgres_user}:{cfg.postgres_password}"
        f"@{cfg.postgres_host}:{cfg.postgres_port}/{cfg.postgres_db}"
    )
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    return url


def upgrade_to_head() -> None:
    """Run ``alembic upgrade head`` against the configured database."""
    from alembic.config import Config

    from alembic import command

    alembic_cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))

    # env.py resolves its target from the environment; keep app config
    # authoritative when DATABASE_URL is not pinned there already.
    os.environ.setdefault("DATABASE_URL", migration_database_url())

    command.upgrade(alembic_cfg, "head")
