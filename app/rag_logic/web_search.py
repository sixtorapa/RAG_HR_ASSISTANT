# app/rag_logic/web_search.py

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(description="The specific search query for the web.")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Use this tool for GENERAL KNOWLEDGE questions (capitals, history, science), "
        "current news, weather, or any public information NOT specific to the company. "
        "If the question is 'who is', 'what is', 'where is' about something public, use this tool."
    )
    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str):
        search = DuckDuckGoSearchRun()
        try:
            results = search.invoke(query)
            return {"answer": results, "source_documents": []}
        except Exception as e:
            return {"answer": f"Error searching the web: {str(e)}", "source_documents": []}
