# app/rag_logic/tools.py

import re
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, List, Type
from langchain.callbacks.manager import CallbackManagerForToolRun

from app.rag_logic.qa_chain import get_conversational_qa_chain
from app.rag_logic.summarizer import resumir_documentos_proyecto
from app.rag_logic.excel_tool import ExcelAnalysisTool


# ── Input Schemas ─────────────────────────────────────────────────────────────

class ChatDocInput(BaseModel):
    question: str = Field(description="The user's specific question.")
    chat_history: Optional[List] = Field(default=[], description="The conversation history.")


class SummarizeInput(BaseModel):
    doc_name_hint: Optional[str] = Field(
        default="",
        description=(
            "Name or hint of the document to summarise (e.g. 'employee_handbook'). "
            "If empty, summarises the entire knowledge base."
        ),
    )


# ── Tool 1: Chat with Documents ───────────────────────────────────────────────

class ChatWithDocumentTool(BaseTool):
    name: str = "chat_with_documents"
    description: str = (
        "Use this tool to answer questions about internal documents (PDFs, XLSX, PPTX), "
        "both simple and complex, by reasoning exclusively over the retrieved context (with sources). "
        "Also use it to summarise a specific document when the user requests it."
    )

    args_schema: Type[BaseModel] = ChatDocInput

    project_id: str
    vector_store_path: str
    model_name: str
    project_settings: dict = {}

    def _run(
        self,
        question: str,
        chat_history: list = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ):
        try:
            if chat_history is None:
                chat_history = []

            # Detect optional path filter in the question: (Search in context: PATH)
            search_kwargs = {}
            match = re.search(r"\(Search in context: (.*?)\)", question)

            if match:
                target_path = match.group(1).strip()
                print(f"🎯 PATH FILTER DETECTED: Restricting search to '{target_path}'")
                search_kwargs = {"python_path_filter": target_path}

            settings = dict(self.project_settings or {})
            settings["last_user_question"] = question

            chain = get_conversational_qa_chain(
                self.project_id,
                self.vector_store_path,
                self.model_name,
                settings,
                search_kwargs_override=search_kwargs,
            )

            payload = {"question": question, "chat_history": chat_history}
            if run_manager:
                response = chain.invoke(payload, config={"callbacks": run_manager.get_child()})
            else:
                response = chain.invoke(payload)

            return response

        except Exception as e:
            print(f"❌ Error in ChatWithDocumentTool: {e}")
            return {"answer": f"Internal error in the document chat tool: {str(e)}", "source_documents": []}


# ── Tool 2: Summarise Document ────────────────────────────────────────────────

class SummarizeDocumentTool(BaseTool):
    name: str = "summarise_document"
    description: str = (
        "Use this tool when the user asks for a summary or synthesis of a document or the knowledge base. "
        "Generates a structured summary using the same retrieval pipeline as 'chat_with_documents'. "
        "If the user mentions a specific document, pass it in doc_name_hint to scope the summary."
    )

    args_schema: Type[BaseModel] = SummarizeInput

    project_id: str
    vector_store_path: str
    model_name: str
    project_settings: dict = {}

    def _run(
        self,
        doc_name_hint: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ):
        try:
            hint = (doc_name_hint or "").strip()

            if hint:
                question = (
                    f"Provide a DETAILED and structured summary of the document '{hint}'.\n"
                    "Requirements:\n"
                    "- Organise by sections (what it is, main topics, key findings, data/metrics, conclusions, recommendations).\n"
                    "- Include figures, dates and percentages when present.\n"
                    "- Do not invent anything.\n"
                    "- Cite sources at the end (filename + page/slide) where possible.\n"
                )
            else:
                question = (
                    "Provide a DETAILED and structured summary of the knowledge base documents.\n"
                    "Requirements:\n"
                    "- Organise by topics/sections.\n"
                    "- Include figures, dates and percentages when present.\n"
                    "- Do not invent anything.\n"
                    "- Cite sources at the end (filename + page/slide) where possible.\n"
                )

            settings = dict(self.project_settings or {})
            settings["last_user_question"] = question

            chain = get_conversational_qa_chain(
                self.project_id,
                self.vector_store_path,
                self.model_name,
                settings,
                search_kwargs_override={},
            )

            payload = {"question": question, "chat_history": []}
            if run_manager:
                response = chain.invoke(payload, config={"callbacks": run_manager.get_child()})
            else:
                response = chain.invoke(payload)

            return response

        except Exception as e:
            print(f"❌ Error in SummarizeDocumentTool: {e}")
            return {"answer": f"Error generating summary: {str(e)}", "source_documents": []}
