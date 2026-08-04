"""
test_guardrails.py — los controles de seguridad, que hasta ahora se afirmaban
sin poder demostrarse.

Dos guardarraíles, dos propiedades que hay que poder enseñar:

  RBAC fail-closed: una lista de departamentos vacía tiene que DENEGAR, no
  abrir. Es el fallo clásico y silencioso — tratar "vacío" como "sin
  restricción" pasa los tests que no existen y filtra en producción.

  DLP de entrada: IBAN, tarjeta y DNI/NIE se validan por dígito de control, no
  por parecido. Un checksum no tiene falsos positivos por azar, y eso es
  comprobable: los números inventados con el formato correcto deben pasar sin
  ser detectados.
"""

import pytest

from app.rag_logic.pii_guard import (
    contains_sensitive_data,
    find_sensitive_entities,
    mask,
)
from app.rag_logic.qa_chain import _build_security_filter, _combine_filters


# ══════════════════════════════════════════════════════════════════════════
# RBAC — control de acceso por departamento
# ══════════════════════════════════════════════════════════════════════════

class TestRBACFailClosed:

    def test_lista_vacia_deniega_todo(self):
        """
        La propiedad central. `[]` no significa "sin restricción", significa
        "sin acceso": se inyecta un departamento que ningún chunk puede tener,
        de modo que la consulta devuelve cero resultados.
        """
        f = _build_security_filter([])
        assert f == {"department": "__no_access__"}

    def test_lista_vacia_no_produce_filtro_nulo(self):
        """
        El fallo que se está evitando: si devolviera None, el retriever
        buscaría sobre TODO el corpus y el usuario sin permisos lo vería todo.
        """
        assert _build_security_filter([]) is not None

    def test_none_es_sin_restriccion_solo_para_admin(self):
        # None lo produce User.get_allowed_departments() únicamente si role=admin.
        assert _build_security_filter(None) is None

    def test_lista_con_departamentos_restringe_a_esos(self):
        f = _build_security_filter(["hr", "it"])
        assert f == {"department": {"$in": ["hr", "it"]}}

    def test_los_departamentos_se_ordenan(self):
        """
        Mismo permiso escrito en otro orden = mismo filtro. Importa porque el
        alcance de seguridad forma parte de la clave de caché de cadenas.
        """
        assert _build_security_filter(["it", "hr"]) == _build_security_filter(["hr", "it"])

    def test_se_deduplican_y_se_normalizan(self):
        f = _build_security_filter(["HR", "hr", " it "])
        assert f == {"department": {"$in": ["hr", "it"]}}

    def test_una_lista_de_cadenas_vacias_tambien_deniega(self):
        """Caso borde: ["", "  "] se queda sin valores válidos -> deny, no allow."""
        assert _build_security_filter(["", "   "]) == {"department": "__no_access__"}


class TestFiltroSiempreEnAnd:

    def test_el_filtro_de_seguridad_se_combina_con_el_funcional(self):
        seguridad = _build_security_filter(["hr"])
        funcional = {"relative_path_norm": "hr/politica.pdf"}
        combinado = _combine_filters(funcional, seguridad)
        assert combinado == {"$and": [funcional, seguridad]}

    def test_un_filtro_solo_no_se_envuelve(self):
        assert _combine_filters(None, {"department": "hr"}) == {"department": "hr"}

    def test_sin_filtros_devuelve_none(self):
        assert _combine_filters(None, None) is None

    def test_el_de_seguridad_no_puede_perderse_al_combinar(self):
        """
        Un documento fuera del alcance no puede salir aunque el filtro
        funcional lo pida explícitamente: los dos van en AND.
        """
        seguridad = _build_security_filter([])
        combinado = _combine_filters({"relative_path_norm": "finanzas/secreto.pdf"}, seguridad)
        assert seguridad in combinado["$and"]


# ══════════════════════════════════════════════════════════════════════════
# DLP — detección de identificadores por dígito de control
# ══════════════════════════════════════════════════════════════════════════

class TestIBAN:

    # IBANs con checksum mod-97 válido
    @pytest.mark.parametrize("iban", [
        "ES9121000418450200051332",
        "ES91 2100 0418 4502 0005 1332",
        "GB82WEST12345698765432",
        "DE89370400440532013000",
    ])
    def test_detecta_iban_valido(self, iban):
        hallazgos = find_sensitive_entities(f"mi cuenta es {iban}, gracias")
        assert "IBAN" in {h.kind for h in hallazgos}

    @pytest.mark.parametrize("falso", [
        "ES9121000418450200051333",   # mismo IBAN con el último dígito cambiado
        "ES0000000000000000000000",
        "XX1234567890123456789012",
    ])
    def test_no_detecta_un_iban_con_checksum_invalido(self, falso):
        """
        Lo que hace valioso al checksum: un número con la PINTA de un IBAN
        pero mal calculado no dispara. Sin él, cualquier cadena alfanumérica
        larga sería un falso positivo.
        """
        hallazgos = find_sensitive_entities(f"referencia {falso}")
        assert "IBAN" not in {h.kind for h in hallazgos}


class TestTarjeta:

    @pytest.mark.parametrize("tarjeta", [
        "4539578763621486",          # Visa de prueba, Luhn válido
        "4539 5787 6362 1486",
        "5500005555555559",
    ])
    def test_detecta_tarjeta_valida(self, tarjeta):
        hallazgos = find_sensitive_entities(f"pagué con la {tarjeta}")
        assert "CARD" in {h.kind for h in hallazgos}

    def test_no_detecta_un_numero_largo_cualquiera(self):
        hallazgos = find_sensitive_entities("el pedido 1234567812345678 llegó tarde")
        assert "CARD" not in {h.kind for h in hallazgos}


class TestDNIyNIE:

    @pytest.mark.parametrize("doc", ["12345678Z", "00000000T", "X1234567L"])
    def test_detecta_documento_valido(self, doc):
        hallazgos = find_sensitive_entities(f"mi documento es {doc}")
        assert "DNI_NIE" in {h.kind for h in hallazgos}

    @pytest.mark.parametrize("doc", ["12345678A", "00000000Z", "X1234567Z"])
    def test_no_detecta_letra_de_control_incorrecta(self, doc):
        hallazgos = find_sensitive_entities(f"referencia {doc}")
        assert "DNI_NIE" not in {h.kind for h in hallazgos}


class TestPoliticaDelGuardarrail:

    def test_texto_limpio_no_dispara_nada(self):
        assert find_sensitive_entities("¿cuántos días de vacaciones tengo?") == []
        assert not contains_sensitive_data("¿cuál es la política de teletrabajo?")

    def test_texto_vacio_no_revienta(self):
        assert find_sensitive_entities("") == []
        assert find_sensitive_entities(None) == []

    def test_el_valor_detectado_nunca_se_registra_completo(self):
        """
        Los hallazgos van al log. Guardar el valor entero convertiría el
        detector de fugas en la fuga.
        """
        iban = "ES9121000418450200051332"
        hallazgos = find_sensitive_entities(f"cuenta {iban}")
        assert hallazgos
        for h in hallazgos:
            assert iban not in h.masked
            assert "*" in h.masked

    def test_la_mascara_conserva_los_extremos(self):
        assert mask("ES9121000418450200051332").startswith("ES")
        assert mask("ES9121000418450200051332").endswith("32")

    def test_varias_entidades_en_el_mismo_texto(self):
        texto = "soy 12345678Z y mi cuenta ES9121000418450200051332"
        tipos = {h.kind for h in find_sensitive_entities(texto)}
        assert {"DNI_NIE", "IBAN"} <= tipos
