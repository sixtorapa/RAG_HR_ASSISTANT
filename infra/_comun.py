"""
Utilidades compartidas por los scripts de infraestructura.

Centraliza tres cosas que si no se repetirían en los siete: cargar el .env de la
raíz del repo, resolver la región, y averiguar el número de cuenta por STS en vez
de escribirlo a mano (que además lo dejaría publicado en el repo).
"""

import os
import pathlib
import sys

import boto3
from dotenv import load_dotenv

RAIZ_REPO = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(RAIZ_REPO / ".env")

REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")

# Nombres de los recursos. Un único sitio donde cambiarlos.
NOMBRE_FUNCION = "hr-assistant"
NOMBRE_REPO_ECR = "hr-assistant"
NOMBRE_ROL = "hr-assistant-lambda-role"
NOMBRE_API = "hr-assistant-api"
ID_INSTANCIA_RDS = "hr-assistant-db"
NOMBRE_SG_RDS = "hr-assistant-db-sg"

# Inference profiles europeos. El prefijo "eu." NO es decorativo: estos modelos
# solo se ofrecen como INFERENCE_PROFILE, así que "anthropic.claude-..." a secas
# devuelve ValidationException. Y se elige "eu." sobre "global." a propósito: el
# perfil europeo mantiene la inferencia dentro de la UE, que en un sistema con
# datos de personal es residencia del dato, no una preferencia.
MODELOS = [
    "eu.anthropic.claude-sonnet-4-6",
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
]


def cuenta() -> str:
    """Número de cuenta, resuelto en tiempo de ejecución (nunca hardcodeado)."""
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def cliente(servicio):
    return boto3.client(servicio, region_name=REGION)


def exigir_credenciales() -> None:
    """Falla pronto y con un mensaje útil si el .env no está puesto."""
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        sys.exit(
            "✗ Faltan credenciales. Añade al .env de la raíz del repo:\n"
            "    AWS_ACCESS_KEY_ID=...\n"
            "    AWS_SECRET_ACCESS_KEY=...\n"
            "    AWS_DEFAULT_REGION=eu-west-1"
        )


def uri_ecr() -> str:
    return f"{cuenta()}.dkr.ecr.{REGION}.amazonaws.com/{NOMBRE_REPO_ECR}"
