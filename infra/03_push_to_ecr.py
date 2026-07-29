"""
Paso 3 — crear el repositorio de ECR y subir la imagen.

ECR es el almacén de imágenes de AWS. La imagen está en tu máquina; Lambda no
puede cogerla de ahí, así que hay que dejarla en un sitio del que pueda tirar.

Dos cosas que costaron tiempo y conviene que estén escritas:

1. NO hace falta el AWS CLI. El token de acceso a ECR se saca con boto3 y se le
   pasa a `docker login` por stdin.

2. Lambda RECHAZA los manifiestos OCI. Docker los genera por defecto, y falla con
   "The image manifest, config or layer media type is not supported".
   `--provenance=false` NO basta: hay que forzar `oci-mediatypes=false`. Por eso
   se construye y se sube con `docker buildx` en un solo paso, en vez de
   `docker build` + `docker push`.

Uso:  python infra/03_push_to_ecr.py
"""

import base64
import subprocess
import sys

from _comun import NOMBRE_REPO_ECR, RAIZ_REPO, cliente, exigir_credenciales, uri_ecr

exigir_credenciales()
ecr = cliente("ecr")

# ── El repositorio ───────────────────────────────────────────────────────────
try:
    ecr.create_repository(
        repositoryName=NOMBRE_REPO_ECR,
        imageScanningConfiguration={"scanOnPush": True},  # análisis de vulnerabilidades
        imageTagMutability="MUTABLE",
    )
    print("  repositorio creado")
except ecr.exceptions.RepositoryAlreadyExistsException:
    print("  repositorio ya existía")

URI = uri_ecr()
print(f"  URI: {URI}")

# ── Autenticar docker contra ECR ─────────────────────────────────────────────
tok = ecr.get_authorization_token()["authorizationData"][0]
usuario, clave = base64.b64decode(tok["authorizationToken"]).decode().split(":", 1)

login = subprocess.run(
    ["docker", "login", "-u", usuario, "--password-stdin", tok["proxyEndpoint"]],
    input=clave, text=True, capture_output=True,
)
if login.returncode != 0:
    sys.exit(f"  ✗ docker login falló: {login.stderr.strip()}")
print("  docker autenticado")

# ── Construir y subir con manifiesto Docker v2 ───────────────────────────────
print("  construyendo y subiendo (unos minutos)...")
build = subprocess.run(
    [
        "docker", "buildx", "build",
        "--provenance=false", "--sbom=false",
        # oci-mediatypes=false es la parte que Lambda necesita
        "--output", "type=image,oci-mediatypes=false,push=true",
        "-f", "Dockerfile.lambda",
        "-t", f"{URI}:latest",
        ".",
    ],
    cwd=RAIZ_REPO,
)
if build.returncode != 0:
    sys.exit("  ✗ el build falló")

print(f"  ✓ imagen en {URI}:latest")
