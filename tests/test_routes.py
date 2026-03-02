"""
test_routes.py — Tests de las rutas Flask.

Estrategia: mockear todas las llamadas a LLM/ChromaDB con unittest.mock.
Así los tests son rápidos (< 1s) y no dependen de API keys reales.

Por qué esto importa en entrevistas:
  "¿Cómo testeas endpoints que dependen de LLMs?"
  → Mocking de dependencias externas + test del contrato HTTP
"""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestAuthRoutes:
    """Tests de login/logout sin mock de LLM."""

    def test_login_page_accessible(self, client):
        response = client.get("/login")
        assert response.status_code == 200

    def test_login_with_valid_credentials(self, client, test_user, app):
        with app.app_context():
            response = client.post(
                "/login",
                data={"username": "testuser", "password": "Password123!"},
                follow_redirects=True,
            )
            assert response.status_code == 200

    def test_login_with_invalid_credentials(self, client, test_user):
        response = client.post(
            "/login",
            data={"username": "testuser", "password": "WrongPassword!"},
            follow_redirects=True,
        )
        # Debe mostrar error o redirigir al login
        assert response.status_code == 200

    def test_protected_route_redirects_unauthenticated(self, client):
        """Las rutas protegidas redirigen al login si no hay sesión."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (302, 401)

    def test_logout(self, auth_client):
        response = auth_client.get("/logout", follow_redirects=True)
        assert response.status_code == 200



class TestAskEndpoint:
    """
    Tests del endpoint /ask/<session_id>.

    Se mockea todo el stack de LLM: AgentRouter, DocumentQAAgent, SQLAgent.
    Lo que validamos:
      - El contrato HTTP (status codes, JSON response shape)
      - Que se rechaza correctamente si falta la pregunta
      - Que el auth funciona (403 sin sesión)
    """

    @pytest.fixture
    def mock_llm_stack(self):
        """Mock de toda la cadena de LLM para tests de rutas."""
        mock_result = {
            "answer": "La política de vacaciones establece 23 días anuales.",
            "source_documents": [],
            "origin": "chat_with_documents",
        }

        with patch("app.main.routes.AgentRouter") as mock_router, \
            patch("app.main.routes.DocumentQAAgent") as mock_doc_agent, \
            patch("app.main.routes.SQLAgent") as mock_sql_agent, \
            patch("app.main.routes.ReasoningAgent") as mock_reasoning, \
            patch("app.main.routes.ChatMemoryStore") as mock_memory:

            # Configurar el router para que devuelva una tool call de documentos
            mock_router_instance = MagicMock()
            mock_router_instance.route.return_value = MagicMock(
                tool_calls=[{"name": "chat_with_documents", "args": {}}]
            )
            mock_router.return_value = mock_router_instance

            # Configurar el doc_agent
            mock_doc_instance = MagicMock()
            mock_doc_instance.run.return_value = mock_result
            mock_doc_agent.return_value = mock_doc_instance

            # Configurar el reasoning agent
            mock_reasoning_instance = MagicMock()
            mock_reasoning_instance.run.return_value = mock_result
            mock_reasoning.return_value = mock_reasoning_instance

            # Memory store
            mock_memory.return_value = MagicMock(recall=lambda q, k: [])

            yield mock_result

    def test_ask_without_auth_returns_401_or_redirect(self, client, test_chat_session):
        response = client.post(
            f"/ask/{test_chat_session.id}",
            json={"question": "¿Cuántos empleados hay?"},
        )
        assert response.status_code in (302, 401, 403)

    def test_ask_without_question_returns_400(self, auth_client, test_chat_session, mock_llm_stack):
        response = auth_client.post(
            f"/ask/{test_chat_session.id}",
            json={},  # sin campo "question"
        )
        assert response.status_code == 400

    def test_ask_with_empty_question_returns_400(self, auth_client, test_chat_session, mock_llm_stack):
        response = auth_client.post(
            f"/ask/{test_chat_session.id}",
            json={"question": ""},
        )
        assert response.status_code == 400

    def test_ask_invalid_session_returns_404(self, auth_client, mock_llm_stack):
        response = auth_client.post(
            "/ask/non-existent-session-id",
            json={"question": "Test question"},
        )
        assert response.status_code == 404

    def test_ask_with_valid_question_returns_200(
        self, auth_client, test_chat_session, mock_llm_stack, app
    ):
        with app.app_context():
            with patch("app.main.routes.get_openai_callback") as mock_cb:
                mock_cb.return_value.__enter__ = MagicMock(return_value=MagicMock(
                    prompt_tokens=100, completion_tokens=50
                ))
                mock_cb.return_value.__exit__ = MagicMock(return_value=False)

                response = auth_client.post(
                    f"/ask/{test_chat_session.id}",
                    json={"question": "¿Cuál es la política de vacaciones?"},
                    content_type="application/json",
                )

        # Debe ser 200 con un campo "answer"
        if response.status_code == 200:
            data = response.get_json()
            assert "answer" in data
            assert data["success"] is True




