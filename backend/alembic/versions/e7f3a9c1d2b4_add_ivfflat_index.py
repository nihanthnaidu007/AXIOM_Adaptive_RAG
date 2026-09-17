"""add_ivfflat_index

Revision ID: e7f3a9c1d2b4
Revises: 22496c2e6b17
Create Date: 2026-09-17

Brings migration-built schemas to parity with the runtime DDL in
axiom/retrieval/vector_store.py: the initial migration created chunk_embeddings
but not the ivfflat ANN index, so a fresh `alembic upgrade head` database
differed from one where the app created the table itself.

Idempotent DDL on purpose: databases created by the legacy startup path
(before alembic was wired in) carry the index already and must upgrade
in place without error.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e7f3a9c1d2b4'
down_revision: Union[str, Sequence[str], None] = '22496c2e6b17'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS chunk_embeddings_vec_idx "
        "ON chunk_embeddings "
        "USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunk_embeddings_vec_idx")
