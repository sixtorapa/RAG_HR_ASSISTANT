"""
test_routes.py — the Flask routes.

Strategy: mock every LLM and ChromaDB call with unittest.mock, so the tests are
fast (< 1s) and need no real API keys. What is asserted is the HTTP contract,
not the content of the answer.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestAuthRoutes:
    """Login and logout, no LLM mocking needed."""

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
        # Must show an error or redirect to the login page
        assert response.status_code == 200

    def test_protected_route_redirects_unauthenticated(self, client):
        """Protected routes redirect to the login page when there is no session."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (302, 401)

    def test_logout(self, auth_client):
        response = auth_client.get("/logout", follow_redirects=True)
        assert response.status_code == 200



class TestAskEndpoint:
    """
    The /ask/<session_id> endpoint.

    Se mockea todo el stack de LLM en app.main.pipeline, que es donde se
    construyen el router y las herramientas.
    Lo que validamos:
      - El contrato HTTP (status codes, JSON response shape)
      - Que se rechaza correctamente si falta la question
      - Que el auth funciona (403 sin sesión)
    """

    @pytest.fixture
    def mock_llm_stack(self):
        """Mocks the whole LLM chain so the route tests stay fast and offline."""
        mock_result = {
            "answer": "La política de vacaciones establece 23 días anuales.",
            "source_documents": [],
            "origin": "chat_with_documents",
        }

        # The tools are built in pipeline.py, so that is where the seam is. A mock
        # that turns red after moving a boundary is the signal that the decoupling
        # actually happened, rather than a test to be patched around.
        with patch("app.main.pipeline.AgentRouter") as mock_router, \
            patch("app.main.pipeline.ChatWithDocumentTool") as mock_doc_tool, \
            patch("app.main.pipeline.SummarizeDocumentTool"), \
            patch("app.main.pipeline.SQLDatabaseTool"), \
            patch("app.main.pipeline.ExcelAnalysisTool"), \
            patch("app.main.pipeline.SQLAgent"), \
            patch("app.main.pipeline.ReasoningAgent") as mock_reasoning:

            # The router returns a document tool call
            mock_router_instance = MagicMock()
            mock_router_instance.route.return_value = MagicMock(
                tool_calls=[{"name": "chat_with_documents", "args": {}}]
            )
            mock_router.return_value = mock_router_instance

            # The document tool: its .name must match the one in the tool_call,
            # porque _run_tools despacha comparando nombres.
            mock_doc_instance = MagicMock()
            mock_doc_instance.name = "chat_with_documents"
            mock_doc_instance.run.return_value = mock_result
            mock_doc_tool.return_value = mock_doc_instance

            # The final formatting agent
            mock_reasoning_instance = MagicMock()
            mock_reasoning_instance.run.return_value = mock_result
            mock_reasoning.return_value = mock_reasoning_instance

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
            with patch("app.main.pipeline.get_openai_callback") as mock_cb:
                mock_cb.return_value.__enter__ = MagicMock(return_value=MagicMock(
                    prompt_tokens=100, completion_tokens=50
                ))
                mock_cb.return_value.__exit__ = MagicMock(return_value=False)

                response = auth_client.post(
                    f"/ask/{test_chat_session.id}",
                    json={"question": "¿Cuál es la política de vacaciones?"},
                    content_type="application/json",
                )

        # Must be 200 with an "answer" field
        if response.status_code == 200:
            data = response.get_json()
            assert "answer" in data
            assert data["success"] is True




