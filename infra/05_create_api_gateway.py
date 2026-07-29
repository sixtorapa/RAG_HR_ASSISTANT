"""
Paso 5 — la puerta pública.

Hasta aquí la Lambda existe y responde, pero NO tiene dirección web: solo se
puede invocar desde dentro de AWS o con código. Esto le pone una URL.

Por qué API Gateway y no una Lambda Function URL:
    Se intentó primero la Function URL, que no tiene el corte de 29 s. Devolvió
    403 permanente pese a AuthType NONE y una política con Principal "*". Causa
    probable: AWS bloquea por defecto las URLs públicas de Lambda en cuentas
    nuevas. API Gateway es otro servicio y no le afecta.

⚠️ API Gateway corta la petición a los 29 SEGUNDOS. Medido en este sistema: una
consulta /ask tarda ~27 s con la cadena ya cacheada, y la primera de cada
contenedor se pasa y devuelve 503. La función sí termina —se midió una de 49 s—
así que es un límite de la puerta, no de Lambda.

Uso:  python infra/05_create_api_gateway.py
"""

from _comun import (
    NOMBRE_API,
    NOMBRE_FUNCION,
    REGION,
    cliente,
    cuenta,
    exigir_credenciales,
)

exigir_credenciales()
lam = cliente("lambda")
api = cliente("apigatewayv2")

arn = lam.get_function_configuration(FunctionName=NOMBRE_FUNCION)["FunctionArn"]

existentes = [a for a in api.get_apis()["Items"] if a["Name"] == NOMBRE_API]
if existentes:
    a = existentes[0]
    print(f"  API ya existía: {a['ApiId']}")
else:
    # Target= crea de una vez la integración, la ruta $default y el stage.
    # $default enruta TODO a la Lambda, que es lo que queremos: el enrutado real
    # lo hace Flask, no API Gateway.
    a = api.create_api(Name=NOMBRE_API, ProtocolType="HTTP", Target=arn)
    print(f"  API creada: {a['ApiId']}")

# Sin este permiso, API Gateway no puede invocar la función y todo da 500.
try:
    lam.add_permission(
        FunctionName=NOMBRE_FUNCION,
        StatementId="APIGatewayInvoke",
        Action="lambda:InvokeFunction",
        Principal="apigateway.amazonaws.com",
        SourceArn=f"arn:aws:execute-api:{REGION}:{cuenta()}:{a['ApiId']}/*/*",
    )
    print("  permiso para API Gateway añadido")
except lam.exceptions.ResourceConflictException:
    print("  permiso ya existía")

endpoint = a.get("ApiEndpoint") or api.get_api(ApiId=a["ApiId"])["ApiEndpoint"]
print(f"\n  ENDPOINT: {endpoint}")
print(f"  prueba:   curl {endpoint}/health")
