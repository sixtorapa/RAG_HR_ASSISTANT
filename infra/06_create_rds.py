"""
Paso 6 — la base de datos.

Por qué hace falta: en Lambda el disco es de solo lectura salvo /tmp, que además
es por contenedor y efímero. Usuarios, sesiones y mensajes tienen que durar y ser
compartidos entre contenedores, así que SQLite no sirve. Hace falta una base de
datos gestionada.

⚠️ DECISIÓN DE SEGURIDAD, léela antes de ejecutar esto.

Este script abre el puerto 5432 a 0.0.0.0/0. No es un descuido, es la
consecuencia de otra decisión:

    La Lambda vive FUERA de la VPC, para conservar salida a internet hacia
    Bedrock y OpenAI. Meterla dentro se la quitaría, y recuperarla exige un NAT
    Gateway (~32 €/mes). Al estar fuera, sus IPs son dinámicas y no hay rango
    que autorizar.

Lo que protege la base son las credenciales y TLS. Para una demo con datos
sintéticos es asumible. Para datos reales de empleados NO lo sería: ahí tocaría
VPC + NAT, o endpoints privados.

Coste: db.t4g.micro es elegible para capa gratuita el primer año; después son
unos 12 €/mes. Acuérdate de borrarla si dejas de usarla.

Uso:  python infra/06_create_rds.py --si
"""

import secrets
import string
import sys

from _comun import (
    ID_INSTANCIA_RDS,
    NOMBRE_SG_RDS,
    RAIZ_REPO,
    cliente,
    exigir_credenciales,
)

if "--si" not in sys.argv:
    sys.exit(
        "Este script crea una base de datos que CUESTA DINERO y queda accesible\n"
        "desde internet. Lee la cabecera del fichero y vuelve con:\n"
        "    python infra/06_create_rds.py --si"
    )

exigir_credenciales()
ec2 = cliente("ec2")
rds = cliente("rds")

# ── VPC por defecto ──────────────────────────────────────────────────────────
vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
print(f"  VPC por defecto: {vpc}")

# ── Grupo de seguridad ───────────────────────────────────────────────────────
try:
    sg = ec2.create_security_group(
        GroupName=NOMBRE_SG_RDS,
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
    print(f"  grupo de seguridad creado: {sg}")
except Exception as e:
    if "already exists" in str(e) or "Duplicate" in str(e):
        sg = ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [NOMBRE_SG_RDS]}]
        )["SecurityGroups"][0]["GroupId"]
        print(f"  grupo de seguridad ya existía: {sg}")
    else:
        raise

# ── La instancia ─────────────────────────────────────────────────────────────
contrasena = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(24))

try:
    rds.create_db_instance(
        DBInstanceIdentifier=ID_INSTANCIA_RDS,
        DBName="hrassistant",
        Engine="postgres",
        DBInstanceClass="db.t4g.micro",
        AllocatedStorage=20,
        StorageType="gp3",
        MasterUsername="hradmin",
        MasterUserPassword=contrasena,
        VpcSecurityGroupIds=[sg],
        PubliclyAccessible=True,
        BackupRetentionPeriod=0,      # entorno de demo: sin copias de seguridad
        MultiAZ=False,
        StorageEncrypted=True,
    )
    with open(RAIZ_REPO / ".env", "a") as f:
        f.write(f"\n# RDS Postgres\nRDS_PASSWORD={contrasena}\n")
    print("  instancia creándose... estará lista en 8-10 min")
    print("  contraseña añadida al .env (que está en .gitignore)")
except rds.exceptions.DBInstanceAlreadyExistsFault:
    print("  la instancia ya existía; no se toca")
    sys.exit(0)

print("\n  esperando a que esté disponible...")
rds.get_waiter("db_instance_available").wait(DBInstanceIdentifier=ID_INSTANCIA_RDS)
d = rds.describe_db_instances(DBInstanceIdentifier=ID_INSTANCIA_RDS)["DBInstances"][0]
host, puerto = d["Endpoint"]["Address"], d["Endpoint"]["Port"]

url = f"postgresql+psycopg2://hradmin:{contrasena}@{host}:{puerto}/hrassistant?sslmode=require"
with open(RAIZ_REPO / ".env", "a") as f:
    f.write(f"DATABASE_URL={url}\n")

print(f"  ✓ disponible: {host}")
print("  ✓ DATABASE_URL escrita en el .env (sslmode=require)")
print("\n  siguiente: python infra/07_init_database.py")
