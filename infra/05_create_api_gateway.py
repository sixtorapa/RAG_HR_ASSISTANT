"""
Step 5 — the public door.

Up to here the Lambda exists and answers, but it has NO web address: it can only
be invoked from inside AWS or from code. This gives it a URL.

Why API Gateway and not a Lambda Function URL:
    The Function URL, which has no 29 s cut-off, was tried first. It returned a
    permanent 403 despite AuthType NONE and a policy with Principal "*". Likely
    cause: AWS blocks public Lambda URLs by default on new accounts. API Gateway
    is a different service and is not affected.

⚠️ API Gateway cuts the request at 29 SECONDS. Measured on this system: an /ask
query takes around 27 s with the chain already cached, and the first one of each
container goes over and returns 503. The function itself does finish — a 49 s
run was measured — so it is a limit of the door, not of Lambda.

Usage:  python infra/05_create_api_gateway.py
"""

from _common import (
    API_NAME,
    FUNCTION_NAME,
    REGION,
    client,
    account,
    require_credentials,
)

require_credentials()
lam = client("lambda")
api = client("apigatewayv2")

arn = lam.get_function_configuration(FunctionName=FUNCTION_NAME)["FunctionArn"]

existentes = [a for a in api.get_apis()["Items"] if a["Name"] == API_NAME]
if existentes:
    a = existentes[0]
    print(f"  API already existed: {a['ApiId']}")
else:
    # Target= creates the integration, the $default route and the stage in one go.
    # $default sends EVERYTHING to the Lambda, which is what we want: the real
    # routing is done by Flask, not by API Gateway.
    a = api.create_api(Name=API_NAME, ProtocolType="HTTP", Target=arn)
    print(f"  API creada: {a['ApiId']}")

# Without this permission API Gateway cannot invoke the function and everything 500s.
try:
    lam.add_permission(
        FunctionName=FUNCTION_NAME,
        StatementId="APIGatewayInvoke",
        Action="lambda:InvokeFunction",
        Principal="apigateway.amazonaws.com",
        SourceArn=f"arn:aws:execute-api:{REGION}:{account()}:{a['ApiId']}/*/*",
    )
    print("  permission for API Gateway added")
except lam.exceptions.ResourceConflictException:
    print("  permission already existed")

endpoint = a.get("ApiEndpoint") or api.get_api(ApiId=a["ApiId"])["ApiEndpoint"]
print(f"\n  ENDPOINT: {endpoint}")
print(f"  prueba:   curl {endpoint}/health")
