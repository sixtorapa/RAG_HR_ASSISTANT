"""
test_router.py — the component that decides.

The router is the only point in the system where a decision is made, and two of
its three paths run a tool WITHOUT going through the LLM. A mistake here is not
corrected by anything downstream, which is why the heuristics need tests more
than any other part.

Three things are covered:
  1. That matching is by WHOLE WORD and not by substring.
  2. That each of the three paths is taken when it should be.
  3. That the heuristics still fire on the phrases they exist for — tightening
     the matching makes it easy to break what already worked.
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


# ── Normalisation ────────────────────────────────────────────────────────────

class TestNormalisation:

    def test_strips_accents(self):
        assert _norm("Nómina") == "nomina"
        assert _norm("ANTIGÜEDAD") == "antiguedad"
        assert _norm("  Política  ") == "politica"

    def test_an_accented_question_matches_the_unaccented_key(self):
        # The signal list carries both "nomina" and "nómina"; with normalisation
        # one would do, but what matters is that both forms work.
        assert _looks_like_sql_intent("¿cuál es la nómina de enero?")
        assert _looks_like_sql_intent("cual es la nomina de enero?")


# ── El defecto: subcadena vs palabra ─────────────────────────────────────────

class TestWholeWordMatching:
    """
    `key in text` matches inside other words. These are the verified cases.

    Note: the "tabla" ⊂ "estable" example that circulated in the notes is FALSE —
    "estable" does not contain "tabla". The ones here are verified.
    """

    @pytest.mark.parametrize("key,phrase", [
        ("suma", "quiero consumar la operación"),
        ("file", "un filete poco hecho"),
        ("hoja", "una plancha de hojalata"),
        ("total", "estoy totalmente de acuerdo"),
        ("pay", "lo pagué por paypal"),
    ])
    def test_does_not_match_inside_another_word(self, key, phrase):
        assert key in _norm(phrase), "el caso dejó de ser un ejemplo de subcadena"
        assert not _contains_whole_word(_norm(phrase), key)

    @pytest.mark.parametrize("key,phrase", [
        ("suma", "dame la suma del archivo"),
        ("file", "abre el file de gastos"),
        ("hoja", "la hoja de gastos"),
        ("salary", "what is the average salary?"),
    ])
    def test_matches_as_a_standalone_word(self, key, phrase):
        assert _contains_whole_word(_norm(phrase), key)

    def test_multi_word_keys_tolerate_spacing(self):
        assert _contains_whole_word(_norm("dame la hoja de calculo"), "hoja de calculo")
        assert _contains_whole_word(_norm("dame la hoja  de   calculo"), "hoja de calculo")

    def test_keys_with_a_file_extension(self):
        assert _contains_whole_word(_norm("abre ventas.xlsx"), ".xlsx")
        assert _contains_whole_word(_norm("el informe .xls de marzo"), ".xls")

    def test_accepted_tradeoff_plurals_no_longer_match(self):
        """
        Requiring whole words loses morphological variants. Documented as a
        decision: a false negative reaches the LLM, which decides well; a false
        positivo va a la herramienta equivocada sin red.
        """
        assert not _contains_whole_word(_norm("los empleado"), "empleados")
        # Which is why the signal list spells out the form that matters:
        assert _looks_like_sql_intent("cuantos empleados hay")


# ── Substring false positives ────────────────────────────────────────────────

class TestExcelFalsePositive:

    def test_the_phrase_of_the_diagnosis_not_fires_excel(self):
        # "file" ⊂ "filete" and "suma" ⊂ "consumar" would force Excel, and the
        # dispatcher did not even know how to run it.
        assert not _looks_like_excel_intent(
            "quiero consumar la operación con un filete estable"
        )

    def test_a_genuine_excel_question_is_still_detected(self):
        assert _looks_like_excel_intent("abre el excel de nóminas")
        assert _looks_like_excel_intent("dame la suma del archivo de gastos")
        assert _looks_like_excel_intent("cuántas cells tiene el fichero")

    @pytest.mark.parametrize("phrase", [
        "¿cuál es el total de ventas?",        # calc signal, no file
        "¿cuántas cells tiene la tabla?",     # weak signal, no file
    ])
    def test_weak_signals_without_a_file_do_not_suffice(self, phrase):
        assert not _looks_like_excel_intent(phrase)


# ── The three heuristics ─────────────────────────────────────────────────────

class TestHeuristics:

    @pytest.mark.parametrize("phrase", [
        "what is the average salary in engineering?",
        "cuántos empleados hay activos",
        "dame el headcount por departamento",
        "quién cobra más",
    ])
    def test_sql_intent(self, phrase):
        assert _looks_like_sql_intent(phrase)

    @pytest.mark.parametrize("phrase", [
        "cuál es la política de teletrabajo",
        "what does the handbook say about onboarding?",
        "cuántos días de vacaciones me corresponden",
    ])
    def test_document_intent(self, phrase):
        assert _looks_like_docs_intent(phrase)

    def test_a_document_question_does_not_look_like_sql(self):
        assert not _looks_like_sql_intent("cuál es la política de teletrabajo")

    @pytest.mark.parametrize("phrase", ["hola", "Buenos días", "hey", "hello there"])
    def test_greetings(self, phrase):
        assert _is_greeting(phrase)

    @pytest.mark.parametrize("phrase", ["gracias", "thanks a lot", "perfecto", "vale"])
    def test_acknowledgements(self, phrase):
        assert _is_thanks(phrase)

    def test_a_greeting_inside_a_question_does_not_count(self):
        # Greetings are anchored to the START: "hola" mid-sentence must not cut the flow.
        assert not _is_greeting("necesito saber si hola es un saludo formal")


# ── The three paths, against the full router ─────────────────────────────────

@pytest.fixture
def router():
    """
    AgentRouter with the LLM mocked, to exercise only the decision logic.

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


class TestTheThreePaths:

    @pytest.mark.parametrize("phrase", ["hola", "gracias", "", "   ", "???", "¿qué puedes hacer?"])
    def test_path_1_answers_without_calling_the_llm(self, router, phrase):
        out = router.route(phrase, [])
        assert not getattr(out, "tool_calls", None)
        assert "ROUTE: DIRECT" in out.content
        router.router_chain.invoke.assert_not_called()

    def test_path_2_forces_sql_without_the_llm(self, router):
        out = router.route("cuántos empleados activos hay", [])
        assert [c["name"] for c in out.tool_calls] == ["query_hr_database"]
        router.router_chain.invoke.assert_not_called()

    def test_path_2_chains_sql_and_docs_when_both_intents_are_present(self, router):
        out = router.route(
            "cuánto cobra un senior y qué dice la política salarial", []
        )
        assert [c["name"] for c in out.tool_calls] == [
            "query_hr_database", "chat_with_documents",
        ]

    def test_path_2_forces_excel_without_the_llm(self, router):
        out = router.route("dame la suma del archivo de gastos", [])
        assert [c["name"] for c in out.tool_calls] == ["excel_analyst"]
        router.router_chain.invoke.assert_not_called()

    def test_path_3_sends_the_ambiguous_case_to_the_llm(self, router):
        router.router_chain.invoke.return_value = MagicMock(
            tool_calls=[{"name": "chat_with_documents", "args": {}}]
        )
        router.route("cuál es la política de teletrabajo", [])
        router.router_chain.invoke.assert_called_once()

    def test_short_smalltalk_does_not_reach_the_llm(self, router):
        # <= 4 words and no SQL/Excel signal -> direct answer.
        out = router.route("y eso qué es", [])
        assert not getattr(out, "tool_calls", None)
        router.router_chain.invoke.assert_not_called()

    def test_a_short_sql_question_still_routes(self, router):
        # Short, but with a clear signal: must not fall into smalltalk.
        out = router.route("headcount por departamento", [])
        assert [c["name"] for c in out.tool_calls] == ["query_hr_database"]
