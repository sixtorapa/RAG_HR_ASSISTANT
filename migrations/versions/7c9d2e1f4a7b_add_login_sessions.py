"""add login sessions

Revision ID: 7c9d2e1f4a7b
Revises: 31b456694013
Create Date: 2026-01-05
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "7c9d2e1f4a7b"
down_revision = "31b456694013"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "login_session",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("user.id"), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("duration_sec", sa.Integer(), nullable=True),
        sa.Column("n_questions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_activity_at", sa.DateTime(), nullable=True),
    )

    op.create_index("ix_login_session_user_id", "login_session", ["user_id"], unique=False)
    op.create_index("ix_login_session_started_at", "login_session", ["started_at"], unique=False)


def downgrade():
    op.drop_index("ix_login_session_started_at", table_name="login_session")
    op.drop_index("ix_login_session_user_id", table_name="login_session")
    op.drop_table("login_session")
