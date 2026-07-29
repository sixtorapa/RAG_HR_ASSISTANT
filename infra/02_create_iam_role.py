"""
Paso 2 — el rol de ejecución de la Lambda.

Una Lambda no lleva usuario ni contraseña. Asume un ROL, y AWS le inyecta
credenciales temporales que caducan solas. No hay nada que guardar ni que rotar.

El rol tiene dos mitades:
    - Quién puede asumirlo  -> solo el servicio Lambda
    - Qué puede hacer       -> invocar DOS modelos concretos, y nada más

Ese "nada más" es la decisión que importa: ni bedrock:*, ni Resource "*". Si
alguien compromete la función, lo máximo que consigue es hacerle preguntas a
Claude. El usuario IAM del desarrollador sí tiene permisos amplios; el servicio no.

Uso:  python infra/02_create_iam_role.py
"""

import json

from _comun import MODELOS, NOMBRE_ROL, REGION, cliente, cuenta, exigir_credenciales

exigir_credenciales()
iam = cliente("iam")

# ── Quién puede asumir el rol ────────────────────────────────────────────────
CONFIANZA = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
}

try:
    r = iam.create_role(
        RoleName=NOMBRE_ROL,
        AssumeRolePolicyDocument=json.dumps(CONFIANZA),
        Description="Rol de ejecucion del RAG HR Assistant en Lambda",
    )
    print(f"  rol creado: {r['Role']['Arn']}")
except iam.exceptions.EntityAlreadyExistsException:
    print(f"  rol ya existía: {iam.get_role(RoleName=NOMBRE_ROL)['Role']['Arn']}")

# ── Permiso 1: escribir logs en CloudWatch ───────────────────────────────────
# Lo mínimo imprescindible. Sin esto la función corre pero no puedes depurarla.
iam.attach_role_policy(
    RoleName=NOMBRE_ROL,
    PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
)
print("  + AWSLambdaBasicExecutionRole (logs)")

# ── Permiso 2: invocar Bedrock, acotado ──────────────────────────────────────
ID_CUENTA = cuenta()
recursos = [
    f"arn:aws:bedrock:{REGION}:{ID_CUENTA}:inference-profile/{m}" for m in MODELOS
]
# Los inference profiles enrutan a los foundation models subyacentes, así que
# hace falta permiso sobre ambos. Se acota igualmente a modelos de Anthropic.
recursos.append("arn:aws:bedrock:*::foundation-model/anthropic.claude-*")

POLITICA = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
        "Resource": recursos,
    }],
}

iam.put_role_policy(
    RoleName=NOMBRE_ROL,
    PolicyName="InvokeClaudeOnly",
    PolicyDocument=json.dumps(POLITICA),
)
print("  + InvokeClaudeOnly — solo InvokeModel, solo estos modelos:")
for m in MODELOS:
    print(f"      {m}")
