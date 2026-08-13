"""
Shared helpers for the infrastructure scripts.

Centralises three things that would otherwise be repeated across all seven:
loading the repo-root .env, resolving the region, and looking the account number
up through STS instead of hardcoding it (which would also publish it in the repo).
"""

import os
import pathlib
import sys

import boto3
from dotenv import load_dotenv

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")

# Resource names. One place to change them.
FUNCTION_NAME = "hr-assistant"
ECR_REPO_NAME = "hr-assistant"
ROLE_NAME = "hr-assistant-lambda-role"
API_NAME = "hr-assistant-api"
RDS_INSTANCE_ID = "hr-assistant-db"
RDS_SG_NAME = "hr-assistant-db-sg"

# European inference profiles. The "eu." prefix is NOT decorative: these models
# are only offered as INFERENCE_PROFILE, so a bare "anthropic.claude-..." returns
# ValidationException. And "eu." is chosen over "global." deliberately: the European
# profile keeps inference inside the EU, which in a system holding employee data is
# data residency, not a preference.
MODELS = [
    "eu.anthropic.claude-sonnet-4-6",
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
]


def account() -> str:
    """Account number, resolved at runtime and never hardcoded."""
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def client(service):
    return boto3.client(service, region_name=REGION)


def require_credentials() -> None:
    """Fail early, with a useful message, when the .env is missing."""
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        sys.exit(
            "✗ Missing credentials. Add these to the repo-root .env:\n"
            "    AWS_ACCESS_KEY_ID=...\n"
            "    AWS_SECRET_ACCESS_KEY=...\n"
            "    AWS_DEFAULT_REGION=eu-west-1"
        )


def ecr_uri() -> str:
    return f"{account()}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO_NAME}"
