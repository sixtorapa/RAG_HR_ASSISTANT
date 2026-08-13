"""
Step 6 — the database.

Why it is needed: in Lambda the disk is read-only except for /tmp, which is also
per-container and ephemeral. Users, sessions and messages have to persist and be
shared across containers, so SQLite does not serve. A managed database is
required.

⚠️ SECURITY DECISION — read before running this.

This script opens port 5432 to 0.0.0.0/0. That is not an oversight, it is the
consequence of another decision:

    The Lambda lives OUTSIDE the VPC, to keep its internet egress to Bedrock and
    OpenAI. Moving it inside would remove that, and restoring it requires a NAT
    Gateway (~€32/month). Being outside, its IPs are dynamic and there is no
    range to authorise.

What protects the database is credentials and TLS. For a demo on synthetic data
that is acceptable. For real employee data it would NOT be: that would call for
VPC + NAT, or private endpoints.

Cost: db.t4g.micro is free-tier eligible for the first year; after that about
€12/month. Remember to delete it when it is no longer in use.

Usage:  python infra/06_create_rds.py --si
"""

import secrets
import string
import sys

from _common import (
    RDS_INSTANCE_ID,
    RDS_SG_NAME,
    REPO_ROOT,
    client,
    require_credentials,
)

if "--si" not in sys.argv:
    sys.exit(
        "Este script crea una base de datos que CUESTA DINERO y queda accesible\n"
        "desde internet. Lee la cabecera del fichero y vuelve con:\n"
        "    python infra/06_create_rds.py --si"
    )

require_credentials()
ec2 = client("ec2")
rds = client("rds")

# ── Default VPC ──────────────────────────────────────────────────────────────
vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
print(f"  default VPC: {vpc}")

# ── Security group ───────────────────────────────────────────────────────────
try:
    sg = ec2.create_security_group(
        GroupName=RDS_SG_NAME,
        Description="Postgres del RAG HR Assistant",
        VpcId=vpc,
    )["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=sg,
        IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 5432, "ToPort": 5432,
            "IpRanges": [{
                "CidrIp": "0.0.0.0/0",
                "Description": "Lambda fuera de VPC: IP dinamica, no acotable",
            }],
        }],
    )
    print(f"  security group created: {sg}")
except Exception as e:
    if "already exists" in str(e) or "Duplicate" in str(e):
        sg = ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [RDS_SG_NAME]}]
        )["SecurityGroups"][0]["GroupId"]
        print(f"  security group already existed: {sg}")
    else:
        raise

# ── La instancia ─────────────────────────────────────────────────────────────
contrasena = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(24))

try:
    rds.create_db_instance(
        DBInstanceIdentifier=RDS_INSTANCE_ID,
        DBName="hrassistant",
        Engine="postgres",
        DBInstanceClass="db.t4g.micro",
        AllocatedStorage=20,
        StorageType="gp3",
        MasterUsername="hradmin",
        MasterUserPassword=contrasena,
        VpcSecurityGroupIds=[sg],
        PubliclyAccessible=True,
        BackupRetentionPeriod=0,      # demo environment: no backups
        MultiAZ=False,
        StorageEncrypted=True,
    )
    with open(REPO_ROOT / ".env", "a") as f:
        f.write(f"\n# RDS Postgres\nRDS_PASSWORD={contrasena}\n")
    print("  instance being created... ready in 8-10 min")
    print("  password appended to .env (which is gitignored)")
except rds.exceptions.DBInstanceAlreadyExistsFault:
    print("  the instance already existed; left untouched")
    sys.exit(0)

print("\n  waiting for it to become available...")
rds.get_waiter("db_instance_available").wait(DBInstanceIdentifier=RDS_INSTANCE_ID)
d = rds.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE_ID)["DBInstances"][0]
host, puerto = d["Endpoint"]["Address"], d["Endpoint"]["Port"]

url = f"postgresql+psycopg2://hradmin:{contrasena}@{host}:{puerto}/hrassistant?sslmode=require"
with open(REPO_ROOT / ".env", "a") as f:
    f.write(f"DATABASE_URL={url}\n")

print(f"  ✓ disponible: {host}")
print("  ✓ DATABASE_URL written to .env (sslmode=require)")
print("\n  siguiente: python infra/07_init_database.py")
