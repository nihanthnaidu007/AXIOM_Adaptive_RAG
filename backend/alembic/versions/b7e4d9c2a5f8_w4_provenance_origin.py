"""w4_provenance_origin: document origin + chunk page-span provenance

Revision ID: b7e4d9c2a5f8
Revises: 9d3b7c4e2f61
Create Date: 2026-09-18

Wave 4 (D5/D3): carry provenance from ingest through retrieval and citations.

- ingested_documents gains origin_type ("upload" | "s3" | "crawl"), origin_uri
  (stable source key: S3 URI / URL / filename), fetched_at, content_hash, and
  the background-ingestion lifecycle columns (status, error_reason,
  parse_confidence). The status lifecycle was previously in-memory only;
  persisting it lets the authenticated polling endpoint survive restarts.
- chunk_embeddings gains page_start/page_end/origin_type: page spans survive
  the page-join at chunk time so the SSE sources frame and citations can
  render page-level provenance.

No embedding-dimension change: vector width stays governed by provider
config (the W3 migration owns that seam); these are plain columns.
"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "b7e4d9c2a5f8"
down_revision: Union[str, Sequence[str], None] = "9d3b7c4e2f61"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_DOC_COLUMNS = [
    ("origin_type", "TEXT NOT NULL DEFAULT 'upload'"),
    ("origin_uri", "TEXT"),
    ("fetched_at", "TIMESTAMPTZ"),
    ("content_hash", "TEXT"),
    ("status", "TEXT NOT NULL DEFAULT 'completed'"),
    ("error_reason", "TEXT"),
    ("parse_confidence", "DOUBLE PRECISION"),
]

_CHUNK_COLUMNS = [
    ("page_start", "INTEGER"),
    ("page_end", "INTEGER"),
    ("origin_type", "TEXT"),
]


def _existing_columns(conn, table: str) -> set:
    rows = conn.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = :t"
        ),
        {"t": table},
    )
    return {r[0] for r in rows}


def upgrade() -> None:
    conn = op.get_bind()
    doc_cols = _existing_columns(conn, "ingested_documents")
    for name, ddl in _DOC_COLUMNS:
        if name not in doc_cols:
            conn.execute(text(f"ALTER TABLE ingested_documents ADD COLUMN {name} {ddl}"))

    chunk_cols = _existing_columns(conn, "chunk_embeddings")
    for name, ddl in _CHUNK_COLUMNS:
        if name not in chunk_cols:
            conn.execute(text(f"ALTER TABLE chunk_embeddings ADD COLUMN {name} {ddl}"))


def downgrade() -> None:
    conn = op.get_bind()
    for name, _ddl in reversed(_CHUNK_COLUMNS):
        conn.execute(text(f"ALTER TABLE chunk_embeddings DROP COLUMN IF EXISTS {name}"))
    for name, _ddl in reversed(_DOC_COLUMNS):
        conn.execute(text(f"ALTER TABLE ingested_documents DROP COLUMN IF EXISTS {name}"))
