# app/rag_logic/agent_router.py

from typing import List, Optional, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool


class AgentRouter:
    """
    AGENT ORCHESTRATOR — HR Knowledge Base Assistant

    Receives the user question + conversation history and decides which
    tool(s) to call (or answers directly for trivial exchanges).

    Tools available:
        - chat_with_documents   : RAG over internal HR docs (PDFs, XLSX, PPTX)
        - query_hr_database     : SQL queries over structured HR data (headcount,
                                  salaries, departments, performance scores)
        - summarise_document    : Full-document summarisation via RAG
        - web_search            : (optional) live web lookup
    """

    def __init__(
        self,
        model_name: str,
        tools: List[BaseTool],
        doc_path: Optional[str] = None,   # kept for compatibility
        temperature: float = 0.0,
        extra_system_context: str = "",
    ) -> None:
        self.model_name = model_name
        self.tools = tools
        self.doc_path = doc_path
        self.temperature = temperature

        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        self.llm_with_tools = llm.bind_tools(self.tools)

        system_prompt = f"""You are a ROUTING ORCHESTRATOR for an internal HR & Knowledge Base assistant.
Your ONLY job is to decide which tool(s) to call based on the user's question.

{extra_system_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTING GUIDE  (follow this order of priority)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A) DOCUMENTS  →  use 'chat_with_documents'
   Trigger when the user asks about:
   - HR policies, employee handbook, code of conduct
   - Job descriptions, competency frameworks
   - Onboarding / offboarding procedures
   - Benefits guides, training materials, compensation policy
   - Recruitment process, interview stages, referral programme
   - Leave policy, probation, remote work rules
   - Any reference to a specific file: "according to the handbook", "in the PDF", etc.

B) STRUCTURED HR DATA  →  use 'query_hr_database'
   Trigger when the user asks about NUMBERS or AGGREGATIONS:
   - Headcount by department / location / role / level
   - Salary, average compensation, highest/lowest paid employee
   - Performance scores, ratings, top performers, attrition rates
   - Year-over-year comparisons, trends, rankings, job postings
   - "Who has the highest...", "How many employees...", "What is the average..."
   Use this even if the user doesn't say "database" or "SQL" explicitly.

C) SUMMARISE  →  use 'summarise_document'
   Trigger when the user asks for:
   - A summary of a specific document: "summarise the handbook", "give me an overview of..."
   - A general overview of the knowledge base: "what documents do we have?"

D) WEB SEARCH  →  use 'web_search'  (ONLY when tool is available)
   Trigger ONLY for questions requiring live external information:
   - Current labour law updates, industry benchmarks
   - General knowledge questions not related to company data
   - Market salary data not in our DB

E) HYBRID MODE  (SQL → DOCS)
   Use ONLY when the user explicitly asks for both numbers AND policy context, e.g.:
   "How does our average salary compare to the pay equity policy?"
   - 1st tool_call: 'query_hr_database'
   - 2nd tool_call: 'chat_with_documents'
   - Order is ALWAYS: SQL first, then DOCS.

F) DIRECT RESPONSE  (very restricted)
   Use ONLY for:
   - Greetings / farewells: "hello", "bye", "thanks"
   - Meta questions: "what can you do?", "who are you?"
   - One-word confirmations: "ok", "got it"
   In ALL other cases, call a tool.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOLDEN RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- When in doubt between DIRECT and a tool → choose the tool.
- When in doubt between SQL and DOCS → default to DOCS.
- Questions about specific people's salaries, scores or roles → always SQL.
- Questions about rules, processes or policies → always DOCS.
- Never invent data; always retrieve it.

MANDATORY FIRST LINE (always):
ROUTE: <route> — <reason in 8-15 words>
  <route> must be one of: DIRECT | DOCS | SQL | SUMMARISE | WEB | HYBRID(SQL→DOCS)
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        self.router_chain = prompt | self.llm_with_tools

    def route(
        self,
        user_input: str,
        chat_history: List[Any],
        callbacks: Optional[List[Any]] = None,
    ):
        """
        Run the router and return the LLM output (with optional tool_calls).

        Args:
            user_input:   The user's question.
            chat_history: List of HumanMessage / AIMessage objects.
            callbacks:    LangChain callbacks (e.g. LangSmith tracer).
        """
        payload = {"input": user_input, "chat_history": chat_history}

        if callbacks:
            return self.router_chain.invoke(payload, config={"callbacks": callbacks})
        return self.router_chain.invoke(payload)
