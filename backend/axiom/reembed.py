"""AXIOM re-embed tool — make the vector store dimension-honest after a
provider/width switch (W3, spec D2).

Run from ``backend/`` with the project venv:

    python -m axiom.reembed          # plan mode — prints the plan, changes nothing, exit 2
    python -m axiom.reembed --yes    # destructive — truncate + rebuild, exit 0 on success

Why this exists: switching embedding provider (or width) makes every stored
vector incomparable with newly embedded ones — the provider-switch contract is
a FULL re-index. Original upload bytes are not persisted (documents are
processed in temp files and discarded), so this tool cannot re-embed the
corpus by itself; it resets the store to the configured width and prints the
sources that must be re-uploaded. It never silently truncates: without
``--yes`` it only prints the plan and exits 2.

What ``--yes`` does, in order:
  1. Lists the indexed sources and ingested-document records about to be
     dropped (from chunk_embeddings / ingested_documents).
  2. Truncates chunk_embeddings and deletes ingested_documents records
     (they would otherwise describe content that is no longer indexed).
  3. Drops the ivfflat index, rebuilds the embedding column at the configured
     width (EMBEDDING_DIMENSIONS cloud / LOCAL_EMBEDDING_DIMENSIONS local),
     and recreates the index.
  4. Best-effort clears the Redis semantic cache (versioned keys already
     isolate old entries; this frees the space).
  5. Prints the re-upload list. BM25 is rebuilt from scratch on the next
     process start / ingest — no action needed.
"""

import argparse
import asyncio
import logging

from sqlalchemy import text

from axiom.config import get_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("axiom.reembed")


def _build_dsn() -> str:
    cfg = get_config()
    return (
        f"postgresql+asyncpg://{cfg.postgres_user}:{cfg.postgres_password}"
        f"@{cfg.postgres_host}:{cfg.postgres_port}/{cfg.postgres_db}"
    )


async def _snapshot(conn) -> tuple[int, list[str], list[str]]:
    """Return (row_count, indexed_sources, ingested_filenames)."""
    count = (await conn.execute(text("SELECT COUNT(*) FROM chunk_embeddings"))).scalar() or 0
    sources = [
        row[0]
        for row in (
            await conn.execute(text("SELECT DISTINCT source FROM chunk_embeddings ORDER BY source"))
        ).fetchall()
    ]
    try:
        filenames = [
            row[0]
            for row in (
                await conn.execute(
                    text("SELECT filename FROM ingested_documents ORDER BY indexed_at")
                )
            ).fetchall()
        ]
    except Exception:
        filenames = []  # table may not exist on very old databases
    return int(count), sources, filenames


async def _run(assume_yes: bool) -> int:
    cfg = get_config()
    target_dims = cfg.effective_embedding_dimensions
    print(
        f"Embedding identity: provider={cfg.embedding_provider} "
        f"model={cfg.effective_embedding_model} width={target_dims}"
    )

    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(_build_dsn())
    try:
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT atttypmod FROM pg_attribute "
                        "WHERE attrelid = 'chunk_embeddings'::regclass "
                        "AND attname = 'embedding' AND attisdropped = false"
                    )
                )
            ).fetchone()
            current_width = int(row[0]) if row and row[0] is not None and row[0] >= 0 else None
            count, sources, filenames = await _snapshot(conn)

        print(f"Current column width: vector({current_width}) — target: vector({target_dims})")
        print(f"Indexed rows: {count} across {len(sources)} source(s)")
        for source in sources:
            print(f"  - {source}")
        if not assume_yes:
            print(
                "Plan (not executed): truncate chunk_embeddings, drop ingested_documents "
                "records, rebuild the column at the target width, recreate the ivfflat "
                "index, clear the Redis cache, then re-upload the sources above."
            )
            print("Re-run with --yes to execute.")
            return 2

        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE chunk_embeddings"))
            await conn.execute(text("DELETE FROM ingested_documents"))
            await conn.execute(text("DROP INDEX IF EXISTS chunk_embeddings_vec_idx"))
            await conn.execute(
                text(
                    f"ALTER TABLE chunk_embeddings "
                    f"ALTER COLUMN embedding TYPE vector({target_dims})"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX chunk_embeddings_vec_idx ON chunk_embeddings "
                    "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
                )
            )

        try:
            from axiom.cache.semantic_cache import semantic_cache

            if await semantic_cache.connect():
                cleared = await semantic_cache.clear()
                print(f"Redis semantic cache cleared: {cleared} entr(y/ies)")
        except Exception as exc:  # cache clear is best-effort; versioned keys already isolate
            logger.warning("Semantic cache clear skipped: %s", exc)

        print(f"Done. chunk_embeddings is now vector({target_dims}).")
        if sources:
            print("Re-upload these sources to rebuild the index:")
            for source in sources:
                print(f"  - {source}")
        elif filenames:
            print("Previously ingested (metadata only): " + ", ".join(filenames))
        return 0
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the AXIOM vector store for the configured embedding width."
    )
    parser.add_argument(
        "--yes", action="store_true", help="Execute the destructive truncate + rebuild."
    )
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args.yes)))
    except Exception as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
