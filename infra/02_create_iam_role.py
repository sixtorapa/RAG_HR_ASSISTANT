"""
Step 2 — the Lambda execution role.

A Lambda carries no username or password. It assumes a ROLE, and AWS injects
temporary credentials that expire on their own. There is nothing to store and
nothing to rotate.

The role has two halves:
    - Who may assume it  -> only the Lambda service
    - What it may do     -> invoke TWO specific models, and nothing else

That "nothing else" is the decision that matters: no bedrock:*, no Resource "*".
If someone compromises the function, the most they get is to ask Claude
questions. The developer's IAM user does have broad permissions; the service
does not.

Usage:  python infra/02_create_iam_role.py
"""

import json

from _common import MODELS, ROLE_NAME, REGION, client, account, require_credentials

require_credentials()
iam = client("iam")

# ── Who may assume the role ──────────────────────────────────────────────────
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
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(CONFIANZA),
        Description="Rol de ejecucion del RAG HR Assistant en Lambda",
    )
    print(f"  rol creado: {r['Role']['Arn']}")
except iam.exceptions.EntityAlreadyExistsException:
    print(f"  role already existed: {iam.get_role(RoleName=ROLE_NAME)['Role']['Arn']}")

# ── Permission 1: write logs to CloudWatch ───────────────────────────────────
# The bare minimum. Without it the function runs but cannot be debugged.
iam.attach_role_policy(
    RoleName=ROLE_NAME,
    PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
)
print("  + AWSLambdaBasicExecutionRole (logs)")

# ── Permiso 2: invocar Bedrock, acotado ──────────────────────────────────────
ID_CUENTA = account()
recursos = [
    f"arn:aws:bedrock:{REGION}:{ID_CUENTA}:inference-profile/{m}" for m in MODELS
]
# Inference profiles route to the underlying foundation models, so permission
# on both is required. Still scoped to Anthropic models only.
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
    RoleName=ROLE_NAME,
    PolicyName="InvokeClaudeOnly",
    PolicyDocument=json.dumps(POLITICA),
)
print("  + InvokeClaudeOnly — solo InvokeModel, solo estos modelos:")
for m in MODELS:
    print(f"      {m}")
