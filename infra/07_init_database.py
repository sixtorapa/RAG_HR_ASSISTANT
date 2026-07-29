"""
Paso 7 — crear el esquema en RDS y conectar la Lambda a él.

Se ejecuta DESDE FUERA, no desde la Lambda, y eso es deliberado: así la función
nunca necesita permisos para modificar el esquema. Solo lee y escribe filas. Es
lo que hace innecesario ejecutar startup.sh dentro de Lambda.

⚠️ Contra una base de datos VACÍA no se puede hacer `flask db upgrade`. Las
migraciones de este proyecto son ALTER TABLE que asumen que las tablas ya
existen, así que la primera falla con:

    relation "chat_session" does not exist

Hay que crear el esquema de cero con db.create_all() y luego MARCAR las
migraciones como aplicadas (stamp head). Es exactamente la bifurcación que ya
hacía startup.sh.

Uso:  python infra/07_init_database.py
"""

import os
import sys

from _comun import (
    NOMBRE_FUNCION,
    RAIZ_REPO,
    cliente,
    exigir_credenciales,
)

sys.path.insert(0, str(RAIZ_REPO))
os.chdir(RAIZ_REPO)

if not os.environ.get("DATABASE_URL"):
    sys.exit("✗ No hay DATABASE_URL en el .env. ¿Terminó de crearse la RDS (paso 6)?")

print(f"  destino: {os.environ['DATABASE_URL'].split('@')[-1]}")

from flask_migrate import stamp, upgrade  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app import create_app, db  # noqa: E402
from config import Config  # noqa: E402

app = create_app(Config)

with app.app_context():
    tablas = set(db.inspect(db.engine).get_table_names())
    print(f"  tablas al empezar: {sorted(tablas) or '(ninguna)'}")

    if "user" in tablas:
        print("  BD existente → aplicando migraciones pendientes")
        upgrade()
    else:
        print("  BD vacía → creando esquema de cero")
        db.session.execute(text("DROP TABLE IF EXISTS alembic_version"))
        db.session.commit()
        db.create_all()
        stamp()   # marca las migraciones como ya aplicadas
        print("  esquema creado y migraciones marcadas (stamp head)")

    print(f"  tablas ahora: {', '.join(sorted(db.inspect(db.engine).get_table_names()))}")

    from app.models import User
    if not User.query.filter_by(username="admin").first():
        u = User(username="admin", role="admin", is_active=True)
        u.set_password("admin1234")
        db.session.add(u)
        db.session.commit()
        print("  usuario admin creado")
    else:
        print("  usuario admin ya existía")

# ── Conectar la Lambda a la BD ───────────────────────────────────────────────
# Aquí se añaden las variables con secretos, que el paso 4 dejó fuera a propósito.
exigir_credenciales()
lam = cliente("lambda")
env = lam.get_function_configuration(FunctionName=NOMBRE_FUNCION)["Environment"]["Variables"]
env.update({
    "DATABASE_URL": os.environ["DATABASE_URL"],
    # Los embeddings siguen en OpenAI por decisión consciente: cambiarlos
    # invalidaría el índice vectorial y las métricas de evaluación ya medidas.
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    "SECRET_KEY": os.environ.get("SECRET_KEY", "cambiar-en-produccion"),
    # Tope de coste: Bedrock es pago por uso SIN techo automático, y las alarmas
    # de presupuesto de AWS avisan pero no cortan. El contador vive en Postgres,
    # no en memoria: cada contenedor tiene la suya, y N contenedores en paralelo
    # multiplicarían un contador en RAM.
    "DAILY_QUESTION_LIMIT": os.environ.get("DAILY_QUESTION_LIMIT", "15"),
})
lam.update_function_configuration(
    FunctionName=NOMBRE_FUNCION, Environment={"Variables": env}
)
lam.get_waiter("function_updated_v2").wait(FunctionName=NOMBRE_FUNCION)
print("\n  ✓ Lambda conectada a la base de datos")

# ⚠️ Project.vector_store_path es una COLUMNA de la BD, no un ajuste. Cambiar la
# variable de entorno NO mueve un proyecto ya creado: hay que actualizar la fila.
print("\n  RECUERDA: si el proyecto ya existe en la BD, actualiza su")
print("  vector_store_path a /tmp/vector_store/info — es una columna, no una")
print("  variable de entorno.")
