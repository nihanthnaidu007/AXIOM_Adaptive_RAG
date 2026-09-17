"""query_feedback: persist thumbs-up/down feedback per trace

Revision ID: f2a9c4e7b1d8
Revises: c4d8e2f6a9b1
Create Date: 2026-09-17

Wave 2 feedback loop (persistence + visibility only — no automatic strategy
re-tuning this wave). One row per user rating on a query trace:

- ``trace_id``: the pipeline_traces.session_id the feedback applies to, with
  an FK so feedback cannot outlive its trace.
- ``rating``: +1 (thumbs-up) or -1 (thumbs-down) — the only signal the tuning
  follow-up consumes.
- ``comment``: optional free-text context from the user.
- ``query_snippet``: the query text captured at submit time, so the summary
  endpoint can show what was rated without joining pipeline_traces.

Idempotent DDL on purpose, matching the house style of the eval_runs
migration: databases provisioned mid-wave may already carry the table.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f2a9c4e7b1d8'
down_revision: Union[str, Sequence[str], None] = 'c4d8e2f6a9b1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS query_feedback (
            id SERIAL PRIMARY KEY,
            trace_id TEXT NOT NULL REFERENCES pipeline_traces(session_id) ON DELETE CASCADE,
            rating SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
            comment TEXT,
            query_snippet TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_query_feedback_trace_id ON query_feedback (trace_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_query_feedback_created_at ON query_feedback (created_at)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS query_feedback")
