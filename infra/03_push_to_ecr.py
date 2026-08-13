"""
Step 3 — create the ECR repository and push the image.

ECR is the AWS image store. The image is on your machine and Lambda cannot pull
it from there, so it has to be left somewhere Lambda can reach.

Two things that cost time and are worth writing down:

1. The AWS CLI is NOT required. The ECR access token is obtained with boto3 and
   piped into `docker login` through stdin.

2. Lambda REJECTS OCI manifests. Docker produces them by default, and it fails
   with "The image manifest, config or layer media type is not supported".
   `--provenance=false` is NOT enough: `oci-mediatypes=false` must be forced.
   That is why the image is built and pushed with `docker buildx` in one step,
   rather than `docker build` + `docker push`.

Usage:  python infra/03_push_to_ecr.py
"""

import base64
import subprocess
import sys

from _common import ECR_REPO_NAME, REPO_ROOT, client, require_credentials, ecr_uri

require_credentials()
ecr = client("ecr")

# ── El repositorio ───────────────────────────────────────────────────────────
try:
    ecr.create_repository(
        repositoryName=ECR_REPO_NAME,
        imageScanningConfiguration={"scanOnPush": True},  # vulnerability scanning
        imageTagMutability="MUTABLE",
    )
    print("  repositorio creado")
except ecr.exceptions.RepositoryAlreadyExistsException:
    print("  repository already existed")

URI = ecr_uri()
print(f"  URI: {URI}")

# ── Autenticar docker contra ECR ─────────────────────────────────────────────
tok = ecr.get_authorization_token()["authorizationData"][0]
usuario, clave = base64.b64decode(tok["authorizationToken"]).decode().split(":", 1)

login = subprocess.run(
    ["docker", "login", "-u", usuario, "--password-stdin", tok["proxyEndpoint"]],
    input=clave, text=True, capture_output=True,
)
if login.returncode != 0:
    sys.exit(f"  ✗ docker login failed: {login.stderr.strip()}")
print("  docker autenticado")

# ── Build and push with a Docker v2 manifest ─────────────────────────────────
print("  construyendo y subiendo (unos minutos)...")
build = subprocess.run(
    [
        "docker", "buildx", "build",
        "--provenance=false", "--sbom=false",
        # oci-mediatypes=false is the part Lambda needs
        "--output", "type=image,oci-mediatypes=false,push=true",
        "-f", "Dockerfile.lambda",
        "-t", f"{URI}:latest",
        ".",
    ],
    cwd=REPO_ROOT,
)
if build.returncode != 0:
    sys.exit("  ✗ the build failed")

print(f"  ✓ image at {URI}:latest")
