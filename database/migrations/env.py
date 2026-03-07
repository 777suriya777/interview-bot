"""
Alembic async migration environment.

Supports both:
  - Online mode: run migrations against a live database
  - Offline mode: generate SQL scripts without a live connection

Usage:
    # Generate a new migration after changing models.py
    alembic -c database/alembic.ini revision --autogenerate -m "add column xyz"

    # Apply all pending migrations
    alembic -c database/alembic.ini upgrade head

    # Roll back one migration
    alembic -c database/alembic.ini downgrade -1
"""
import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ── Import models so Alembic can detect schema changes ─────────────────
# This import populates Base.metadata with all table definitions.
import sys
from pathlib import Path

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database.models import Base  # noqa: E402

# ── Alembic Config ─────────────────────────────────────────────────────

config = context.config

# Read logging config from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Use models' metadata for autogenerate support
target_metadata = Base.metadata


def get_database_url() -> str:
    """
    Read DATABASE_URL from environment, converting to asyncpg scheme.
    Env variable takes precedence over alembic.ini sqlalchemy.url.
    """
    url = os.environ.get("DATABASE_URL", config.get_main_option("sqlalchemy.url", ""))
    if not url:
        raise RuntimeError("DATABASE_URL is not set in environment or alembic.ini")
    # Ensure asyncpg scheme
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


# ── Offline Mode ────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """
    Run migrations without a live database connection.
    Generates SQL that can be reviewed and applied manually.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Compare types so Alembic detects column type changes
        compare_type=True,
        # Compare server defaults (e.g. DEFAULT NOW())
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online Mode ─────────────────────────────────────────────────────────

def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations within a sync context."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_database_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # No pool for migration scripts
    )

    async with connectable.connect() as connection:
        # run_sync wraps the sync migration runner inside the async connection
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


# ── Entry Point ─────────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
