"""
Paso 4 — crear la función Lambda a partir de la imagen de ECR.

"Función Lambda" es el nombre que le da AWS a un DESPLIEGUE. No se crea código
aquí: el código ya está en la imagen. Lo que se registra es la ficha de cómo
montarla — qué imagen, cuánta máquina, cuánto puede tardar, con qué permisos.

Los dos números tienen motivo:

    Timeout=300   El defecto son 3 s. Una consulta RAG los pasa de largo.

    MemorySize    En Lambda la CPU va LIGADA a la memoria. Se piden 3008 MB no
                  porque hagan falta 3 GB —medido: usa 308 MB— sino porque eso
                  da más procesador y acorta el arranque en frío.

Contenedor y no zip: el paquete zip topa en 250 MB descomprimido y chromadb con
sus dependencias no cabe. Una imagen llega a 10 GB.

Uso:  python infra/04_create_lambda.py
"""

import time

from _comun import (
    NOMBRE_FUNCION,
    NOMBRE_ROL,
    REGION,
    cliente,
    exigir_credenciales,
    uri_ecr,
)

exigir_credenciales()
lam = cliente("lambda")
URI = f"{uri_ecr()}:latest"
ROL = cliente("iam").get_role(RoleName=NOMBRE_ROL)["Role"]["Arn"]

CONFIG = dict(
    FunctionName=NOMBRE_FUNCION,
    Role=ROL,
    Code={"ImageUri": URI},
    PackageType="Image",
    MemorySize=3008,
    Timeout=300,
    Architectures=["x86_64"],
    Environment={"Variables": {
        "LLM_PROVIDER": "bedrock",
        # El vector store se copia a /tmp en el arranque (ver lambda_handler.py):
        # Chroma abre su SQLite en lectura-escritura aunque solo consultes.
        "UP_VECTOR_STORE_PATH": "/var/task/vector_store/info",
        # La BD analítica sí puede leerse desde la imagen: sql_tool solo admite
        # SELECT/WITH, así que nunca escribe.
        "HR_DB_URI": "sqlite:////var/task/hr_data.db",
        # DATABASE_URL y OPENAI_API_KEY se añaden después (pasos 6 y 7): llevan
        # secretos y no se escriben aquí.
    }},
    Description="RAG HR Assistant — Flask sobre Lambda via apig-wsgi, LLM en Bedrock",
)

for intento in range(6):
    try:
        r = lam.create_function(**CONFIG)
        print(f"  función creada: {r['FunctionArn']}")
        break
    except lam.exceptions.ResourceConflictException:
        lam.update_function_code(FunctionName=NOMBRE_FUNCION, ImageUri=URI)
        print("  la función ya existía → código actualizado")
        break
    except lam.exceptions.InvalidParameterValueException as e:
        # Un rol IAM recién creado tarda unos segundos en propagarse. Pero este
        # error también sale con un manifiesto OCI, así que se muestra el motivo
        # en vez de reintentar a ciegas.
        if "manifest" in str(e).lower():
            raise SystemExit(
                "  ✗ Lambda ha rechazado el manifiesto de la imagen.\n"
                "    Reconstruye con oci-mediatypes=false (ver 03_push_to_ecr.py)."
            )
        print(f"  esperando propagación del rol... ({intento + 1}/6)")
        time.sleep(10)
else:
    raise SystemExit("  ✗ no se pudo crear la función")

print("  esperando a que esté activa...")
lam.get_waiter("function_active_v2").wait(FunctionName=NOMBRE_FUNCION)
c = lam.get_function_configuration(FunctionName=NOMBRE_FUNCION)
print(f"  estado: {c['State']} · {c['MemorySize']} MB · timeout {c['Timeout']} s · {REGION}")
