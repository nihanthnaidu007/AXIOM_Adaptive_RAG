"""embedding_dimensions: provider-aware chunk_embeddings width

Revision ID: 9d3b7c4e2f61
Revises: f2a9c4e7b1d8
Create Date: 2026-09-17

Wave 3 (D2): make stored embedding width follow the configured embedding
provider instead of the historical hardcoded vector(1536).

- Fresh databases: no-op here — server startup creates chunk_embeddings at
  the configured width (EMBEDDING_DIMENSIONS cloud / LOCAL_EMBEDDING_DIMENSIONS
  local).
- Existing databases already at the target width: idempotent no-op.
- Existing databases at a different width: the ivfflat index is dropped and
  the column is rebuilt — but ONLY when the table is empty. A width change
  makes every stored vector incomparable with newly embedded ones (the
  provider-switch contract is a full re-index), yet this migration refuses to
  destroy rows silently: it raises with the exact remediation commands
  (``python -m axiom.reembed --yes``, then re-run ``alembic upgrade head``).

The target dimension is resolved from the process environment at migration
time (mirroring axiom.config resolution; the migration stays self-contained
and does not import application code, so it keeps working on hosts where only
alembic and the driver are installed).
"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "9d3b7c4e2f61"
down_revision: Union[str, Sequence[str], None] = "f2a9c4e7b1d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INDEX_NAME = "chunk_embeddings_vec_idx"


def _resolve_target_dimension() -> int:
    """Mirror axiom.config dimension resolution, env-only (frozen logic)."""
    import os

    if os.environ.get("EMBEDDING_PROVIDER", "cloud").strip().lower() == "local":
        raw = os.environ.get("LOCAL_EMBEDDING_DIMENSIONS", "768")
    else:
        raw = os.environ.get("EMBEDDING_DIMENSIONS", "1536")
    try:
        dim = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Embedding dimension env var is not an integer: {raw!r}") from exc
    if dim <= 0:
        raise RuntimeError(f"Embedding dimension must be positive, got {dim}")
    return dim


def _current_dimension(conn) -> "int | None":
    """Return the vector typmod (width) of chunk_embeddings.embedding, or
    None when the table/column does not exist (fresh database)."""
    row = conn.execute(
        text(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'chunk_embeddings'::regclass "
            "AND attname = 'embedding' AND attisdropped = false"
        )
    ).fetchone()
    if row is None or row[0] is None or row[0] < 0:
        return None
    return int(row[0])


def upgrade() -> None:
    target = _resolve_target_dimension()
    conn = op.get_bind()
    current = _current_dimension(conn)
    if current is None or current == target:
        return  # fresh DB (startup DDL owns creation) or already correct

    count = conn.execute(text("SELECT COUNT(*) FROM chunk_embeddings")).scalar() or 0
    if count:
        raise RuntimeError(
            f"chunk_embeddings.embedding is vector({current}) but the configured "
            f"embedding width is {target}, and the table holds {count} rows. "
            f"Changing width invalidates every stored vector (full re-index "
            f"required on provider switch). Run `python -m axiom.reembed --yes` "
            f"to truncate + rebuild the column, re-upload the listed sources, "
            f"then re-run `alembic upgrade head`."
        )

    conn.execute(text(f"DROP INDEX IF EXISTS {_INDEX_NAME}"))
    conn.execute(text(f"ALTER TABLE chunk_embeddings ALTER COLUMN embedding TYPE vector({target})"))


def downgrade() -> None:
    conn = op.get_bind()
    current = _current_dimension(conn)
    if current is None or current == 1536:
        return
    count = conn.execute(text("SELECT COUNT(*) FROM chunk_embeddings")).scalar() or 0
    if count:
        raise RuntimeError(
            f"chunk_embeddings.embedding is vector({current}) with {count} rows; "
            f"downgrade would require vector(1536). Run `python -m axiom.reembed --yes` "
            f"first, then re-run the downgrade."
        )
    conn.execute(text(f"DROP INDEX IF EXISTS {_INDEX_NAME}"))
    conn.execute(text("ALTER TABLE chunk_embeddings ALTER COLUMN embedding TYPE vector(1536)"))
