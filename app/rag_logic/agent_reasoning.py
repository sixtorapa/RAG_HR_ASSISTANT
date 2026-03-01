# app/rag_logic/agent_reasoning.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

AgentResult = Dict[str, Any]


def _merge_source_docs(agent_results: List[AgentResult]) -> List[Any]:
    all_docs = []
    for r in agent_results:
        docs = r.get("source_documents") or []
        all_docs.extend(docs)
    return all_docs


def _build_contributions_summary(agent_results: List[AgentResult], max_chars_per_block: int = 8000) -> str:
    blocks = []
    for idx, r in enumerate(agent_results, start=1):
        origin = r.get("origin") or r.get("agent_name") or f"Agent {idx}"

        # Prefer raw SQL output if available, always truncated to avoid token overflow
        text = (r.get("sql_raw_output") or r.get("answer") or "").strip()
        if not text:
            continue

        if len(text) > max_chars_per_block:
            text = (
                text[:max_chars_per_block]
                + "\n... [OUTPUT TRUNCATED FOR ANALYSIS: download Excel/CSV to see full data]"
            )

        blocks.append(f"--- INFO FROM {origin.upper()} ---\n{text}")

    if not blocks:
        return "No useful response was received from the intermediate agents."
    return "\n\n".join(blocks)


class ReasoningAgent:
    """
    FINAL REASONING AGENT.
    Formats, cleans numbers and structures the final response for the user.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        extra_style: str = "",
        callbacks: Optional[list] = None,
    ) -> None:
        self.callbacks = callbacks or []
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature, callbacks=self.callbacks)
        self.extra_style = extra_style

    def run(self, user_question: str, agent_results: List[AgentResult]) -> AgentResult:
        contributions_text = _build_contributions_summary(agent_results)
        merged_sources = _merge_source_docs(agent_results)

        system_prompt = (
            "You are a SENIOR DATA ANALYST expert in executive communication.\n"
            "Your mission is to take raw or technical data and present it in a CLEAR and well-structured way.\n\n"
            "FORMATTING RULES (MANDATORY):\n"
            "1) USE MARKDOWN: **bold** for key figures, names and important concepts.\n"
            "2) TABLES: if there are lists or series (top rankings, trends, comparisons), present them as a Markdown table.\n"
            "3) NUMBER FORMATTING:\n"
            "   - use comma as thousands separator (13,394,244)\n"
            "   - avoid unnecessary decimals in large figures\n"
            "   - add € symbol for euro amounts\n"
            "4) STRUCTURE:\n"
            "   - start with a direct sentence that answers the question\n"
            "   - include table/series if applicable\n"
            "   - end with an 'Insights' section with bullet points\n"
            "5) LANGUAGE: English, professional tone.\n"
        )

        if self.extra_style:
            system_prompt += "\nAdditional instructions:\n" + self.extra_style.strip()

        user_prompt = (
            f"User question:\n{user_question}\n\n"
            f"Raw data received:\n{contributions_text}\n\n"
            "Task:\n"
            "- Generate the final formatted response.\n"
            "- Clean up the numbers.\n"
            "- Use a table if there is a list or series.\n"
            "- Do not invent data not present in the raw input.\n"
            "- If the raw data contains the word 'Error' or the symbol '⚠️':\n"
            "  - Do not generate tables.\n"
            "  - Do not invent or complete missing information.\n"
            "  - Simply explain briefly and clearly that the analysis cannot be performed "
            "with that data and what needs to be corrected or provided.\n"
        )

        final_msg = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        )

        return {
            "answer": (final_msg.content or "").strip(),
            "source_documents": merged_sources,
        }
