# app/rag_logic/excel_tool.py
import os
import pandas as pd
import concurrent.futures
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import ClassVar

class ExcelQueryInput(BaseModel):
    query: str = Field(description="La pregunta sobre cifras, cálculos o datos del Excel.")
    file_name_hint: str = Field(description="Nombre aproximado del archivo Excel (ej: 'Facturacion', 'Plan').", default="")
    

class ExcelAnalysisTool(BaseTool):
    name: str = "analista_de_excel"
    description: str = "Úsala OBLIGATORIAMENTE para preguntas de 'cuánto suma', 'costes', 'presupuestos', 'fechas' o datos tabulares en archivos Excel."
    args_schema: type[BaseModel] = ExcelQueryInput
    
    doc_path: str
    model_name: str
    
    # CONFIGURACIÓN DE SEGURIDAD
    MAX_WAIT_TIME_SECONDS: ClassVar[int] = 60  # Tiempo máximo de espera antes de abortar
    MAX_ITERATIONS: ClassVar[int] = 5          # Número máximo de pasos de pensamiento del agente


    def _find_excel_files(self):
        """Busca todos los .xlsx en la ruta del proyecto"""
        excel_files = []
        for root, dirs, files in os.walk(self.doc_path):
            for file in files:
                if file.endswith(".xlsx") and not file.startswith("~$"):
                    excel_files.append(os.path.join(root, file))
        return excel_files

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia el DataFrame para evitar tokens basura (columnas Unnamed vacías).
        Esto reduce drásticamente los errores 429 (Too Many Requests).
        """
        # 1. Eliminar columnas que sean todo NaN
        df = df.dropna(axis=1, how='all')
        # 2. Eliminar filas que sean todo NaN
        df = df.dropna(axis=0, how='all')
        
        # 3. Limpiar nombres de columnas (quitar espacios extra y Unnamed raros)
        df.columns = [str(c).strip() if 'Unnamed' not in str(c) else '' for c in df.columns]
        
        return df

    def _run_agent_with_timeout(self, agent, query):
        """Ejecuta el agente en un hilo separado con timeout."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(agent.invoke, query)
            try:
                # Esperamos X segundos
                result = future.result(timeout=self.MAX_WAIT_TIME_SECONDS)
                return result
            except concurrent.futures.TimeoutError:
                raise TimeoutError("El análisis excedió el tiempo límite.")

    def _run(self, query: str, file_name_hint: str = ""):
        try:
            files = self._find_excel_files()
            if not files:
                return "No se encontraron archivos Excel en este proyecto."

            # Selección inteligente del archivo
            target_file = files[0] # Por defecto el primero
            if file_name_hint:
                for f in files:
                    if file_name_hint.lower() in f.lower():
                        target_file = f
                        break
            
            # Cargar DataFrame
            try:
                # Usamos header=None si detectamos que la fila 1 está vacía, o dejamos que pandas infiera
                # Para mayor robustez, leemos normal y luego limpiamos
                df = pd.read_excel(target_file, engine='openpyxl')
                df = self._preprocess_dataframe(df)
            except Exception as e:
                return {"answer": f"Error leyendo el archivo Excel {target_file}: {str(e)}", "source_documents": []}

            print(f"📊 Excel Agent: Analizando '{os.path.basename(target_file)}' (Cols: {len(df.columns)}, Rows: {len(df)})")

            # Crear Agente Pandas con LÍMITES
            llm = ChatOpenAI(model_name=self.model_name, temperature=0)
            
            agent = create_pandas_dataframe_agent(
                llm, 
                df, 
                verbose=True,
                allow_dangerous_code=True,
                agent_type="openai-tools",
                handle_parsing_errors=True,
                max_iterations=self.MAX_ITERATIONS,  # <--- CORTE 1: Máximo 4 intentos de pensamiento
                early_stopping_method="force"        # <--- Forzar parada si llega al límite
            )

            # Ejecutar con TIMEOUT
            try:
                response = self._run_agent_with_timeout(agent, query)
                output_text = response['output']
            except TimeoutError:
                return {
                    "answer": (
                        "⏱️ **Tiempo de espera agotado.**\n"
                        "El archivo Excel es muy complejo o la pregunta requiere procesar demasiados datos.\n"
                        "Por favor, intenta ser más específico (ej: 'dime el total de la celda J10' en lugar de 'analiza el archivo')."
                    ),
                    "source_documents": []
                }
            except Exception as e:
                # Si el agente falla por iteraciones máximas, a veces lanza excepción
                if "Agent stopped" in str(e) or "iteration limit" in str(e):
                    return {
                        "answer": "🛑 **Análisis detenido.** El agente intentó buscar la respuesta varias veces sin éxito. Intenta reformular la pregunta.",
                        "source_documents": []
                    }
                raise e # Relanzar otros errores para el catch general
            
            # Formatear respuesta exitosa
            return {
                "answer": f"📊 **Análisis de {os.path.basename(target_file)}:**\n\n{output_text}",
                "source_documents": [] 
            }

        except Exception as e:
            print(f"❌ Error crítico en Excel Tool: {e}")
            return {"answer": f"No pude completar el análisis del Excel. Error: {str(e)}", "source_documents": []}