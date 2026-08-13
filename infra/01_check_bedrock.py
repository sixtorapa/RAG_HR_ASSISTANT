"""
Step 1 — check the credentials work and that Bedrock answers.

Creates nothing. It comes first because everything else depends on it: if the
account cannot invoke the model, there is no point building images or functions
on top.

Four checks IN ORDER, so a failure says where the problem is rather than only
that there is one:
    1. Is the key valid?        -> if not, the problem is the credentials
    2. Which models are there?  -> if not, the problem is the region
    3. Which are invocable?     -> gives the exact ID to use
    4. A real call              -> the acid test

Usage:  python infra/01_check_bedrock.py
"""

import sys

from botocore.exceptions import ClientError, NoCredentialsError

from _common import MODELS, REGION, client, require_credentials


def heading(n, text):
    print(f"\n{'=' * 62}\n  STEP {n} — {text}\n{'=' * 62}")


require_credentials()

# ── 1. Who am I with this key? ───────────────────────────────────────────────
heading(1, "identidad")
try:
    ident = client("sts").get_caller_identity()
    print(f"  Cuenta : {ident['Account']}")
    print(f"  ARN    : {ident['Arn']}")
except NoCredentialsError:
    sys.exit("  ✗ No credentials in the environment.")
except ClientError as e:
    sys.exit(f"  ✗ Credenciales rechazadas: {e.response['Error']['Code']}")

# ── 2. Anthropic models available in the region ──────────────────────────────
heading(2, f"modelos de Anthropic en {REGION}")
bedrock = client("bedrock")
try:
    for m in bedrock.list_foundation_models(byProvider="anthropic")["modelSummaries"]:
        tipos = ",".join(m.get("inferenceTypesSupported", []))
        # ON_DEMAND = invocable with this ID as-is.
        # INFERENCE_PROFILE only = the prefixed ID from step 3 is required.
        print(f"  {m['modelId']:<50} {tipos}")
except ClientError as e:
    print(f"  ✗ {e.response['Error']['Code']}: {e.response['Error']['Message']}")

# ── 3. The IDs that can actually be invoked ──────────────────────────────────
heading(3, "inference profiles — ESTOS son los IDs invocables")
try:
    for p in bedrock.list_inference_profiles()["inferenceProfileSummaries"]:
        if "anthropic" in p["inferenceProfileId"].lower():
            print(f"  {p['inferenceProfileId']:<55} {p['status']}")
except ClientError as e:
    print(f"  ✗ {e.response['Error']['Code']}: {e.response['Error']['Message']}")

# ── 4. A real invocation ─────────────────────────────────────────────────────
heading(4, "real invocation")
modelo = MODELS[0]
print(f"  Modelo: {modelo}")
try:
    r = client("bedrock-runtime").converse(
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
        print("    → missing bedrock:InvokeModel, or Anthropic's 'use case details'")
    elif code == "ValidationException":
        print("    → the model requires the inference profile ID (eu. prefix)")
    sys.exit(1)
