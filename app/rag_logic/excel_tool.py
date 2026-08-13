# app/rag_logic/excel_tool.py
import os
import pandas as pd
import concurrent.futures
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from .llm_factory import get_llm
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import ClassVar, List, Optional

class ExcelQueryInput(BaseModel):
    query: str = Field(description="The question about figures, calculations or data in the Excel file.")
    file_name_hint: str = Field(description="Approximate name of the Excel file (e.g. 'Billing', 'Plan').", default="")
    

class ExcelAnalysisTool(BaseTool):
    name: str = "excel_analyst"
    description: str = "Use this MANDATORILY for questions about totals, costs, budgets, dates or tabular data held in Excel files."
    args_schema: type[BaseModel] = ExcelQueryInput
    
    doc_path: str
    model_name: str

    # Department access guardrail (same concept as in qa_chain.py):
    # None = unrestricted (admin); a list (including empty) = restricted to those
    # departments, i.e. those subfolders of knowledge_base/. Without it this tool
    # could read any .xlsx in the project without passing the RAG filter.
    allowed_departments: Optional[List[str]] = None

    # SAFETY LIMITS
    MAX_WAIT_TIME_SECONDS: ClassVar[int] = 60  # Max wait before aborting
    MAX_ITERATIONS: ClassVar[int] = 5          # Max reasoning steps for the agent


    def _find_excel_files(self):
        """Find every .xlsx under the project path, honouring the department guardrail."""
        allowed_norm = (
            {d.strip().lower() for d in self.allowed_departments}
            if self.allowed_departments is not None else None
        )
        excel_files = []
        for root, dirs, files in os.walk(self.doc_path):
            department = os.path.basename(root).strip().lower()
            if allowed_norm is not None and department not in allowed_norm:
                continue
            for file in files:
                if file.endswith(".xlsx") and not file.startswith("~$"):
                    excel_files.append(os.path.join(root, file))
        return excel_files

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame to avoid junk tokens (empty Unnamed columns).
        This sharply reduces 429 (Too Many Requests) errors.
        """
        # 1. Drop all-NaN columns
        df = df.dropna(axis=1, how='all')
        # 2. Drop all-NaN rows
        df = df.dropna(axis=0, how='all')
        
        # 3. Clean column names (extra spaces and stray "Unnamed" labels)
        df.columns = [str(c).strip() if 'Unnamed' not in str(c) else '' for c in df.columns]
        
        return df

    def _run_agent_with_timeout(self, agent, query):
        """Run the agent on a separate thread, with a timeout."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(agent.invoke, query)
            try:
                # Wait up to the configured timeout
                result = future.result(timeout=self.MAX_WAIT_TIME_SECONDS)
                return result
            except concurrent.futures.TimeoutError:
                raise TimeoutError("The analysis exceeded the time limit.")

    def _run(self, query: str, file_name_hint: str = ""):
        try:
            files = self._find_excel_files()
            if not files:
                return "No Excel files were found in this project."

            # File selection
            target_file = files[0]  # first by default
            if file_name_hint:
                for f in files:
                    if file_name_hint.lower() in f.lower():
                        target_file = f
                        break
            
            # Load the DataFrame
            try:
                # Read normally and clean afterwards, which is more robust than
                # guessing whether row 1 is a header
                df = pd.read_excel(target_file, engine='openpyxl')
                df = self._preprocess_dataframe(df)
            except Exception as e:
                return {"answer": f"Error reading the Excel file {target_file}: {str(e)}", "source_documents": []}

            print(f"📊 Excel agent: analysing '{os.path.basename(target_file)}' (cols: {len(df.columns)}, rows: {len(df)})")

            # Build the pandas agent WITH LIMITS
            llm = get_llm(self.model_name, 0)
            
            agent = create_pandas_dataframe_agent(
                llm, 
                df, 
                verbose=True,
                allow_dangerous_code=True,
                agent_type="openai-tools",
                handle_parsing_errors=True,
                max_iterations=self.MAX_ITERATIONS,  # <--- cut-off 1: bounded reasoning steps
                early_stopping_method="force"        # <--- force a stop when the limit is reached
            )

            # Execute WITH TIMEOUT
            try:
                response = self._run_agent_with_timeout(agent, query)
                output_text = response['output']
            except TimeoutError:
                return {
                    "answer": (
                        "⏱️ **Timed out.**\n"
                        "The Excel file is too complex, or the question requires processing too much data.\n"
                        "Please be more specific (e.g. 'give me the total in cell J10' rather than 'analyse the file')."
                    ),
                    "source_documents": []
                }
            except Exception as e:
                # Hitting the iteration cap sometimes raises instead of returning
                if "Agent stopped" in str(e) or "iteration limit" in str(e):
                    return {
                        "answer": "🛑 **Analysis stopped.** The agent tried several times without finding an answer. Try rephrasing the question.",
                        "source_documents": []
                    }
                raise e  # re-raise anything else for the general handler
            
            # Format the successful answer
            return {
                "answer": f"📊 **Analysis of {os.path.basename(target_file)}:**\n\n{output_text}",
                "source_documents": [] 
            }

        except Exception as e:
            print(f"❌ Critical error in the Excel tool: {e}")
            return {"answer": f"Could not complete the Excel analysis. Error: {str(e)}", "source_documents": []}