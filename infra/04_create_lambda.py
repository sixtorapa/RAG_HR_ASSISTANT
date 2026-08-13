"""
Step 4 — create the Lambda function from the ECR image.

"Lambda function" is what AWS calls a DEPLOYMENT. No code is created here: the
code is already in the image. What gets registered is the recipe for running it
— which image, how much machine, how long it may take, with which permissions.

Both numbers have a reason:

    Timeout=300   The default is 3 s. A RAG query blows past that.

    MemorySize    In Lambda, CPU is TIED to memory. 3008 MB is requested not
                  because 3 GB are needed — measured usage is 308 MB — but
                  because it buys more processor and shortens the cold start.

A container rather than a zip: the zip package caps at 250 MB uncompressed and
chromadb with its dependencies does not fit. An image can reach 10 GB.

Usage:  python infra/04_create_lambda.py
"""

import time

from _common import (
    FUNCTION_NAME,
    ROLE_NAME,
    REGION,
    client,
    require_credentials,
    ecr_uri,
)

require_credentials()
lam = client("lambda")
URI = f"{ecr_uri()}:latest"
ROL = client("iam").get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

CONFIG = dict(
    FunctionName=FUNCTION_NAME,
    Role=ROL,
    Code={"ImageUri": URI},
    PackageType="Image",
    MemorySize=3008,
    Timeout=300,
    Architectures=["x86_64"],
    Environment={"Variables": {
        "LLM_PROVIDER": "bedrock",
        # The vector store is copied to /tmp at start-up (see lambda_handler.py):
        # Chroma opens its SQLite read-write even for queries.
        "UP_VECTOR_STORE_PATH": "/var/task/vector_store/info",
        # The analytics DB can be read from the image: sql_tool only accepts
        # SELECT/WITH, so it never writes.
        "HR_DB_URI": "sqlite:////var/task/hr_data.db",
        # DATABASE_URL and OPENAI_API_KEY are added later (steps 6 and 7): they carry
        # secrets and are not written here.
    }},
    Description="RAG HR Assistant — Flask sobre Lambda via apig-wsgi, LLM en Bedrock",
)

for attempt in range(6):
    try:
        r = lam.create_function(**CONFIG)
        print(f"  function created: {r['FunctionArn']}")
        break
    except lam.exceptions.ResourceConflictException:
        lam.update_function_code(FunctionName=FUNCTION_NAME, ImageUri=URI)
        print("  the function already existed → code updated")
        break
    except lam.exceptions.InvalidParameterValueException as e:
        # A freshly created IAM role takes a few seconds to propagate. But this
        # error also appears with an OCI manifest, so the reason is shown
        # instead of retrying blindly.
        if "manifest" in str(e).lower():
            raise SystemExit(
                "  ✗ Lambda rejected the image manifest.\n"
                "    Rebuild with oci-mediatypes=false (see 03_push_to_ecr.py)."
            )
        print(f"  waiting for the role to propagate... ({attempt + 1}/6)")
        time.sleep(10)
else:
    raise SystemExit("  ✗ could not create the function")

print("  waiting for it to become active...")
lam.get_waiter("function_active_v2").wait(FunctionName=FUNCTION_NAME)
c = lam.get_function_configuration(FunctionName=FUNCTION_NAME)
print(f"  estado: {c['State']} · {c['MemorySize']} MB · timeout {c['Timeout']} s · {REGION}")
