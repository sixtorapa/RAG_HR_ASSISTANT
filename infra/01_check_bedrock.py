"""
Paso 1 — comprobar que las credenciales valen y que Bedrock responde.

No crea nada. Es la primera comprobación porque todo lo demás depende de ella:
si la cuenta no puede invocar el modelo, no tiene sentido construir imágenes ni
funciones encima.

Cuatro comprobaciones EN ORDEN, para que un fallo diga dónde está el problema y
no solo que lo hay:
    1. ¿La clave es válida?          -> si no, el problema son las credenciales
    2. ¿Qué modelos hay?             -> si no, el problema es la región
    3. ¿Cuáles son invocables?       -> te da el ID exacto que hay que usar
    4. Una llamada real              -> la prueba de fuego

Uso:  python infra/01_check_bedrock.py
"""

import sys

from botocore.exceptions import ClientError, NoCredentialsError

from _comun import MODELOS, REGION, cliente, exigir_credenciales


def titulo(n, texto):
    print(f"\n{'=' * 62}\n  PASO {n} — {texto}\n{'=' * 62}")


exigir_credenciales()

# ── 1. ¿Quién soy con esta clave? ────────────────────────────────────────────
titulo(1, "identidad")
try:
    ident = cliente("sts").get_caller_identity()
    print(f"  Cuenta : {ident['Account']}")
    print(f"  ARN    : {ident['Arn']}")
except NoCredentialsError:
    sys.exit("  ✗ No hay credenciales en el entorno.")
except ClientError as e:
    sys.exit(f"  ✗ Credenciales rechazadas: {e.response['Error']['Code']}")

# ── 2. Modelos de Anthropic en la región ─────────────────────────────────────
titulo(2, f"modelos de Anthropic en {REGION}")
bedrock = cliente("bedrock")
try:
    for m in bedrock.list_foundation_models(byProvider="anthropic")["modelSummaries"]:
        tipos = ",".join(m.get("inferenceTypesSupported", []))
        # ON_DEMAND = invocable con este ID tal cual.
        # Solo INFERENCE_PROFILE = hay que usar el ID con prefijo del paso 3.
        print(f"  {m['modelId']:<50} {tipos}")
except ClientError as e:
    print(f"  ✗ {e.response['Error']['Code']}: {e.response['Error']['Message']}")

# ── 3. Los IDs que de verdad se pueden invocar ───────────────────────────────
titulo(3, "inference profiles — ESTOS son los IDs invocables")
try:
    for p in bedrock.list_inference_profiles()["inferenceProfileSummaries"]:
        if "anthropic" in p["inferenceProfileId"].lower():
            print(f"  {p['inferenceProfileId']:<55} {p['status']}")
except ClientError as e:
    print(f"  ✗ {e.response['Error']['Code']}: {e.response['Error']['Message']}")

# ── 4. Invocación real ───────────────────────────────────────────────────────
titulo(4, "invocación real")
modelo = MODELOS[0]
print(f"  Modelo: {modelo}")
try:
    r = cliente("bedrock-runtime").converse(
        modelId=modelo,
        messages=[{"role": "user", "content": [{"text": "Responde solo: OK"}]}],
        inferenceConfig={"maxTokens": 20, "temperature": 0},
    )
    print(f"  ✓ RESPUESTA: {r['output']['message']['content'][0]['text']}")
    print(f"  ✓ Tokens   : {r['usage']}")
except ClientError as e:
    code = e.response["Error"]["Code"]
    print(f"  ✗ {code}: {e.response['Error']['Message']}")
    if code == "AccessDeniedException":
        print("    → falta bedrock:InvokeModel, o el 'use case details' de Anthropic")
    elif code == "ValidationException":
        print("    → el modelo exige el ID del inference profile (prefijo eu.)")
    sys.exit(1)
