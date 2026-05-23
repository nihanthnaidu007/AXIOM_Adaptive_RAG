"""initial_schema

Revision ID: 22496c2e6b17
Revises: 
Create Date: 2026-05-17 18:37:38.244766

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '22496c2e6b17'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("""
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            id SERIAL PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            embedding vector(1536) NOT NULL,
            token_count INTEGER,
            bm25_score FLOAT,
            ingested_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_traces (
            session_id TEXT PRIMARY KEY,
            trace_data JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS ingested_documents (
            doc_id TEXT PRIMARY KEY,
            filename TEXT,
            chunk_count INTEGER,
            file_size_bytes INTEGER,
            indexed_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            job_id TEXT PRIMARY KEY,
            status TEXT,
            progress INTEGER,
            total INTEGER,
            aggregate JSONB,
            results JSONB,
            started_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        )
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE IF EXISTS eval_runs")
    op.execute("DROP TABLE IF EXISTS ingested_documents")
    op.execute("DROP TABLE IF EXISTS pipeline_traces")
    op.execute("DROP TABLE IF EXISTS chunk_embeddings")
    op.execute("DROP EXTENSION IF EXISTS vector")
