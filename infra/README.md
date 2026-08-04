# infra/ — how the AWS deployment was built

These are the scripts that created the AWS side of this project, in the order they
were run. They are **imperative**: each one calls the AWS API to create a resource.
They were run once, from a laptop, on 29 July 2026.

Nothing here runs as part of the application. This folder exists so that what lives
in AWS is written down somewhere, rather than only existing in the console.

## Honest limitation

This is **not** infrastructure as code. There is no state file, no plan, no diff before
applying, and nothing detects drift if someone changes a setting by hand. Re-running a
script mostly no-ops because each one checks whether the resource already exists, but
that is a convention here, not a guarantee the tooling gives you.

Terraform or CloudFormation is what this folder should eventually become. The
translation is mechanical — every resource and every parameter is already spelled out
below — but it has not been done.

## Order

| | Script | Creates |
|---|---|---|
| 1 | `01_check_bedrock.py` | Nothing. Verifies credentials and lists invocable model IDs |
| 2 | `02_create_iam_role.py` | The Lambda execution role, scoped to two Claude models |
| 3 | `03_push_to_ecr.py` | The ECR repository, and pushes the image |
| 4 | `04_create_lambda.py` | The Lambda function from that image |
| 5 | `05_create_api_gateway.py` | The public HTTP endpoint |
| 6 | `06_create_rds.py` | The PostgreSQL instance and its security group |
| 7 | `07_init_database.py` | Schema and admin user, run from outside the Lambda |

## Requirements

```bash
pip install boto3 python-dotenv
```

Credentials are read from a `.env` at the repository root:

```env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=eu-west-1
```

The account ID is never hardcoded — every script resolves it through STS.

## Decisions worth knowing before you run any of this

**The Lambda lives outside the VPC.** Putting it inside to reach RDS privately would cost
it internet access, and it needs that to reach Bedrock and OpenAI. Restoring it means a
NAT Gateway at roughly €32/month. The trade is that RDS accepts connections from
anywhere, protected by credentials and TLS. For a demo on synthetic data that is
acceptable; for real employee data it would not be.

**The execution role can only invoke two specific models.** Not `bedrock:*`, not
`Resource: "*"`. The developer's own IAM user has broad permissions; the service does not.

**Migrations run from outside.** `07_init_database.py` is executed from a laptop against
RDS, so the function never needs permission to alter the schema. It reads and writes
rows, nothing more.

**Lambda rejects OCI image manifests.** The image must be pushed with Docker v2 schema 2.
`--provenance=false` alone is not enough; see `03_push_to_ecr.py`.

**Memory is over-allocated on purpose.** The function uses about 308 MB of 3008. In
Lambda, CPU scales with memory, and lowering it lengthens the cold start.

## Known ceiling

A warm `/ask` takes 17.7 s against API Gateway's 29 s limit. The first query on a cold
container still exceeds it and returns 503: building the chain — opening Chroma, loading
the BM25 index, assembling the ensemble — does not fit in the window.

The function itself completes well past 29 s, so this is a gateway limit rather than a
Lambda one. Two levers are untried: the function runs Claude Sonnet because MODEL_NAME is
unset and defaults to gpt-4o, and setting it to gpt-4o-mini maps to Haiku; and streaming
would remove the ceiling entirely, since the connection opens on the first token.

A Lambda Function URL, which has no 29 s ceiling, returned 403 despite `AuthType: NONE`
and a `Principal: "*"` policy — AWS blocks public function URLs by default on new
accounts.

## Tearing it down

Nothing here is free forever. RDS is the one that costs real money outside the free tier.

```bash
aws rds delete-db-instance --db-instance-identifier hr-assistant-db --skip-final-snapshot
aws lambda delete-function --function-name hr-assistant
aws ecr delete-repository --repository-name hr-assistant --force
aws apigatewayv2 delete-api --api-id <id>
aws iam delete-role --role-name hr-assistant-lambda-role
```
