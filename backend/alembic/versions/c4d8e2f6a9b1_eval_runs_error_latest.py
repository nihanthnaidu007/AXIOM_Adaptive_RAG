"""eval_runs: add error and latest columns for live job status

Revision ID: c4d8e2f6a9b1
Revises: e7f3a9c1d2b4
Create Date: 2026-09-17

Wave 2 moves eval-job state out of the in-process ``_eval_jobs`` dict into
the ``eval_runs`` table (already created by the initial migration) so job
progress is visible from every worker. Two columns the live status payload
needs were missing:

- ``error``: failure message when a job's status becomes 'failed'.
- ``latest``: JSONB snapshot of the most recent per-query result, for
  progress polling without re-reading the whole ``results`` array.

Idempotent DDL on purpose: databases provisioned before this migration may
already carry the columns from the app's previous ad-hoc writes.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c4d8e2f6a9b1'
down_revision: Union[str, Sequence[str], None] = 'e7f3a9c1d2b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS error TEXT")
    op.execute("ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS latest JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE eval_runs DROP COLUMN IF EXISTS latest")
    op.execute("ALTER TABLE eval_runs DROP COLUMN IF EXISTS error")
