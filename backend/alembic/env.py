"""Alembic migration environment for AXIOM.

Runs migrations over the same database the application uses. Because the
runtime URL is an asyncpg URL (and psycopg is not a dependency), online
migrations go through SQLAlchemy's async engine with ``run_sync`` — a sync
``engine_from_config`` over an asyncpg URL crashes with MissingGreenlet.
Sync URLs (e.g. ``postgresql+psycopg``) still work via the plain engine.
"""

import asyncio
import os
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

load_dotenv()


def _resolve_database_url() -> str:
    """Target URL for migrations, normalized to the asyncpg driver.

    DATABASE_URL (standard alembic env var) wins; POSTGRES_URL — which the app
    accepts as psycopg-flavored — is rewritten to asyncpg since psycopg is not
    a dependency. Env.py runs inside a loop-free thread at startup and as a
    plain process under the alembic CLI, so asyncio.run is safe in both.
    """
    url = (
        os.environ.get("DATABASE_URL")
        or os.environ.get("POSTGRES_URL")
        or "postgresql+asyncpg://axiom:axiom@localhost:5432/axiom_rag"
    )
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    return url


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

config.set_main_option("sqlalchemy.url", _resolve_database_url())

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The schema is owned by the migrations in alembic/versions (single source of
# truth); autogenerate is not wired to models on purpose — new schema changes
# are written as explicit migration scripts.
target_metadata = None


def _run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode over an async or sync engine."""
    engine_config = config.get_section(config.config_ini_section, {})
    url = engine_config.get("sqlalchemy.url", "")

    if url.startswith("postgresql+asyncpg"):
        connectable = async_engine_from_config(
            engine_config,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        async def _run_async() -> None:
            async with connectable.connect() as connection:
                await connection.run_sync(_run_migrations)
            await connectable.dispose()

        asyncio.run(_run_async())
        return

    connectable = engine_from_config(
        engine_config,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        _run_migrations(connection)
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
