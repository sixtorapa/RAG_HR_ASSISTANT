"""
test_router.py — el componente que decide, que hasta ahora no tenía red.

El router es el único punto del sistema donde se toma una decisión, y dos de
sus tres caminos ejecutan una herramienta SIN pasar por el LLM. Un fallo aquí
no lo corrige nadie aguas abajo: por eso las heurísticas necesitan pruebas más
que cualquier otra parte.

Se cubren tres cosas:
  1. Que la coincidencia sea por PALABRA y no por subcadena (el defecto real).
  2. Que cada uno de los tres caminos se tome cuando toca.
  3. Que las heurísticas sigan acertando en las frases para las que existen —
     al endurecer la coincidencia es fácil romper lo que sí funcionaba.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.rag_logic.agent_router import (
    AgentRouter,
    _contains_whole_word,
    _is_greeting,
    _is_thanks,
    _looks_like_docs_intent,
    _looks_like_excel_intent,
    _looks_like_sql_intent,
    _norm,
)


# ── Normalización ────────────────────────────────────────────────────────────

class TestNormalizacion:

    def test_quita_acentos(self):
        assert _norm("Nómina") == "nomina"
        assert _norm("ANTIGÜEDAD") == "antiguedad"
        assert _norm("  Política  ") == "politica"

    def test_una_pregunta_con_acentos_casa_con_la_clave_sin_ellos(self):
        # La lista de señales trae "nomina" y "nómina"; con normalización
        # bastaría una, pero lo que importa es que las dos formas funcionen.
        assert _looks_like_sql_intent("¿cuál es la nómina de enero?")
        assert _looks_like_sql_intent("cual es la nomina de enero?")


# ── El defecto: subcadena vs palabra ─────────────────────────────────────────

class TestCoincidenciaPorPalabra:
    """
    `key in text` casa dentro de otras palabras. Estos son los casos reales.

    Nota: el ejemplo "tabla" ⊂ "estable" que circulaba en la documentación es
    FALSO — "estable" no contiene "tabla". Los de aquí sí están verificados.
    """

    @pytest.mark.parametrize("key,frase", [
        ("suma", "quiero consumar la operación"),
        ("file", "un filete poco hecho"),
        ("hoja", "una plancha de hojalata"),
        ("total", "estoy totalmente de acuerdo"),
        ("pay", "lo pagué por paypal"),
    ])
    def test_no_casa_dentro_de_otra_palabra(self, key, frase):
        assert key in _norm(frase), "el caso dejó de ser un ejemplo de subcadena"
        assert not _contains_whole_word(_norm(frase), key)

    @pytest.mark.parametrize("key,frase", [
        ("suma", "dame la suma del archivo"),
        ("file", "abre el file de gastos"),
        ("hoja", "la hoja de gastos"),
        ("salary", "what is the average salary?"),
    ])
    def test_si_casa_como_palabra_suelta(self, key, frase):
        assert _contains_whole_word(_norm(frase), key)

    def test_claves_de_varias_palabras_toleran_espacios(self):
        assert _contains_whole_word(_norm("dame la hoja de calculo"), "hoja de calculo")
        assert _contains_whole_word(_norm("dame la hoja  de   calculo"), "hoja de calculo")

    def test_claves_con_extension_de_fichero(self):
        assert _contains_whole_word(_norm("abre ventas.xlsx"), ".xlsx")
        assert _contains_whole_word(_norm("el informe .xls de marzo"), ".xls")

    def test_contrapartida_asumida_no_casan_los_plurales(self):
        """
        Exigir palabra completa pierde variantes morfológicas. Se documenta
        como decisión: un falso negativo va al LLM, que decide bien; un falso
        positivo va a la herramienta equivocada sin red.
        """
        assert not _contains_whole_word(_norm("los empleado"), "empleados")
        # Por eso la lista de señales enumera la forma que interesa:
        assert _looks_like_sql_intent("cuantos empleados hay")


# ── El bug que motivó todo esto ──────────────────────────────────────────────

class TestFalsoPositivoDeExcel:

    def test_la_frase_del_diagnostico_ya_no_dispara_excel(self):
        # Antes: "file" ⊂ "filete" y "suma" ⊂ "consumar" -> se forzaba Excel,
        # y el despachador ni siquiera sabía ejecutarlo.
        assert not _looks_like_excel_intent(
            "quiero consumar la operación con un filete estable"
        )

    def test_excel_de_verdad_sigue_detectandose(self):
        assert _looks_like_excel_intent("abre el excel de nóminas")
        assert _looks_like_excel_intent("dame la suma del archivo de gastos")
        assert _looks_like_excel_intent("cuántas cells tiene el fichero")

    @pytest.mark.parametrize("frase", [
        "¿cuál es el total de ventas?",        # calc sin fichero
        "¿cuántas cells tiene la tabla?",     # weak sin fichero
    ])
    def test_señales_debiles_sin_fichero_no_bastan(self, frase):
        assert not _looks_like_excel_intent(frase)


# ── Las tres heurísticas ─────────────────────────────────────────────────────

class TestHeuristicas:

    @pytest.mark.parametrize("frase", [
        "what is the average salary in engineering?",
        "cuántos empleados hay activos",
        "dame el headcount por departamento",
        "quién cobra más",
    ])
    def test_intencion_sql(self, frase):
        assert _looks_like_sql_intent(frase)

    @pytest.mark.parametrize("frase", [
        "cuál es la política de teletrabajo",
        "what does the handbook say about onboarding?",
        "cuántos días de vacaciones me corresponden",
    ])
    def test_intencion_documental(self, frase):
        assert _looks_like_docs_intent(frase)

    def test_una_pregunta_de_documentos_no_parece_sql(self):
        assert not _looks_like_sql_intent("cuál es la política de teletrabajo")

    @pytest.mark.parametrize("frase", ["hola", "Buenos días", "hey", "hello there"])
    def test_saludos(self, frase):
        assert _is_greeting(frase)

    @pytest.mark.parametrize("frase", ["gracias", "thanks a lot", "perfecto", "vale"])
    def test_agradecimientos(self, frase):
        assert _is_thanks(frase)

    def test_un_saludo_dentro_de_una_pregunta_no_cuenta(self):
        # Los saludos se anclan al INICIO: "hola" en medio no debe cortar el flujo.
        assert not _is_greeting("necesito saber si hola es un saludo formal")


# ── Los tres caminos, sobre el router completo ───────────────────────────────

@pytest.fixture
def router():
    """
    AgentRouter con el LLM mockeado, para ejercer solo la lógica de decisión.

    `router_chain` se sustituye por un mock DESPUÉS de construir el router:
    en el constructor es `prompt | llm_with_tools`, o sea un Runnable real de
    LangChain, y sobre un Runnable no se puede afirmar "no te han llamado".
    Sustituirlo es lo que permite comprobar la propiedad que importa de los dos
    primeros caminos: que el LLM no llega a invocarse.
    """
    with patch("app.rag_logic.agent_router.get_llm") as mock_llm:
        llm = MagicMock()
        llm.bind_tools.return_value = MagicMock()
        mock_llm.return_value = llm
        r = AgentRouter(model_name="gpt-4o-mini", tools=[])
        r.router_chain = MagicMock()
        yield r


class TestLosTresCaminos:

    @pytest.mark.parametrize("frase", ["hola", "gracias", "", "   ", "???", "¿qué puedes hacer?"])
    def test_camino_1_responde_sin_llamar_al_llm(self, router, frase):
        out = router.route(frase, [])
        assert not getattr(out, "tool_calls", None)
        assert "ROUTE: DIRECT" in out.content
        router.router_chain.invoke.assert_not_called()

    def test_camino_2_fuerza_sql_sin_llm(self, router):
        out = router.route("cuántos empleados activos hay", [])
        assert [c["name"] for c in out.tool_calls] == ["query_hr_database"]
        router.router_chain.invoke.assert_not_called()

    def test_camino_2_encadena_sql_y_docs_si_hay_las_dos_intenciones(self, router):
        out = router.route(
            "cuánto cobra un senior y qué dice la política salarial", []
        )
        assert [c["name"] for c in out.tool_calls] == [
            "query_hr_database", "chat_with_documents",
        ]

    def test_camino_2_fuerza_excel_sin_llm(self, router):
        out = router.route("dame la suma del archivo de gastos", [])
        assert [c["name"] for c in out.tool_calls] == ["analista_de_excel"]
        router.router_chain.invoke.assert_not_called()

    def test_camino_3_lo_ambiguo_llega_al_llm(self, router):
        router.router_chain.invoke.return_value = MagicMock(
            tool_calls=[{"name": "chat_with_documents", "args": {}}]
        )
        router.route("cuál es la política de teletrabajo", [])
        router.router_chain.invoke.assert_called_once()

    def test_smalltalk_corto_no_llega_al_llm(self, router):
        # <= 4 palabras y sin señal de SQL/Excel -> respuesta directa.
        out = router.route("y eso qué es", [])
        assert not getattr(out, "tool_calls", None)
        router.router_chain.invoke.assert_not_called()

    def test_una_pregunta_corta_de_sql_si_se_enruta(self, router):
        # Corta, pero con señal clara: no debe caer en el smalltalk.
        out = router.route("headcount por departamento", [])
        assert [c["name"] for c in out.tool_calls] == ["query_hr_database"]
