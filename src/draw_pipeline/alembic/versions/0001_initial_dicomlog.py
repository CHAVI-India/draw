"""initial dicomlog queue table

Regenerated to match the current ORM (``draw_pipeline.dao.table.DicomLog``):
``model`` is a plain string column (was a stale hardcoded ``TSGyne/TSPrime`` enum),
and ``status`` includes ``FAILED``.

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dicomlog",
        sa.Column(
            "id",
            sa.BigInteger()
            .with_variant(mysql.BIGINT(), "mysql")
            .with_variant(sa.BIGINT(), "postgresql")
            .with_variant(sa.INTEGER(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("series_name", sa.String(length=256), nullable=False),
        sa.Column("input_path", sa.String(length=1024), nullable=False),
        sa.Column("output_path", sa.String(length=1024), nullable=True),
        sa.Column(
            "status",
            sa.Enum("INIT", "STARTED", "PREDICTED", "SENT", "FAILED", name="status"),
            server_default="INIT",
            nullable=False,
        ),
        sa.Column("model", sa.String(length=256), nullable=False),
        sa.Column(
            "created_on",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        # Lease bookkeeping for crash recovery (visibility-timeout pattern).
        sa.Column("claimed_at", sa.DateTime(), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("series_name"),
    )
    op.create_index(op.f("ix_dicomlog_status"), "dicomlog", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_dicomlog_status"), table_name="dicomlog")
    op.drop_table("dicomlog")
