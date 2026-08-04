"""drop the project table and its foreign key

La aplicación sirve una sola base de conocimiento. De las ocho columnas de
`project`, cinco eran una copia de variables de entorno (nombre, rutas del
corpus y del índice, modelo), `status` valía siempre "READY", `id` solo existía
para esta clave foránea, y `cost` se acumulaba en una columna que ninguna
pantalla mostraba. Lo único con estado real —la instrucción de sistema y el
contexto SQL— es configuración y vive en config.py.

El problema no era el código de más: era tener dos fuentes de verdad. Cambiar
UP_VECTOR_STORE_PATH no movía un proyecto ya creado, porque la ruta estaba
congelada en la fila; fue uno de los tres fallos que solo aparecieron al
desplegar en Lambda.

⚠️ Esta migración es DESTRUCTIVA: borra la tabla `project` y la columna
`chat_session.project_id`. Las conversaciones y los mensajes no se tocan. El
downgrade recrea la estructura, pero no puede devolver los datos borrados.

Revision ID: b7f1c3d95e20
Revises: e0e58b03301b
Create Date: 2026-08-04
"""

from alembic import op
import sqlalchemy as sa


revision = "b7f1c3d95e20"
down_revision = "e0e58b03301b"
branch_labels = None
depends_on = None


def upgrade():
    # batch_alter_table: SQLite no admite DROP COLUMN ni DROP CONSTRAINT
    # directamente, así que Alembic recrea la tabla por debajo. En PostgreSQL
    # se traduce a un ALTER TABLE normal.
    with op.batch_alter_table("chat_session", schema=None) as batch_op:
        batch_op.drop_column("project_id")

    op.drop_table("project")


def downgrade():
    op.create_table(
        "project",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("document_path", sa.String(length=255), nullable=False),
        sa.Column("vector_store_path", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("model_name", sa.String(length=50), nullable=True),
        sa.Column("settings", sa.JSON(), nullable=True),
        sa.Column("cost", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        sa.UniqueConstraint("vector_store_path"),
    )
    op.create_index(op.f("ix_project_name"), "project", ["name"], unique=True)

    # nullable=True a propósito: las sesiones existentes no tienen proyecto al
    # que apuntar, así que exigir un valor haría fallar el downgrade.
    with op.batch_alter_table("chat_session", schema=None) as batch_op:
        batch_op.add_column(sa.Column("project_id", sa.String(length=36), nullable=True))
