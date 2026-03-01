# 🏗️ ARQUITECTURA DE PROMPTS - DIAGRAMA CONCEPTUAL


┌─────────────────────────────────────────────────────────────────────────────┐
│                         USUARIO HACE PREGUNTA                               │
│                    (Chat HTML en navegador)                                 │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │                                 │
                    │   Flask Route: /ask/<proj_id>   │
                    │   (routes.py, línea ~120)       │
                    │                                 │
                    └────────────────┬────────────────┘
                                     │
          ┌──────────────────────────▼──────────────────────────┐
          │                                                      │
          │    ROUTER CHAIN (router.py)                         │
          │    ┌────────────────────────────────────────────┐   │
          │    │ Prompt MEJORADO:                           │   │
          │    │ ✓ Reglas explícitas para decisión          │   │
          │    │ ✓ Análisis de contexto conversación        │   │
          │    │ ✓ Palabras clave indicadoras               │   │
          │    │                                            │   │
          │    │ Decide: ¿Chat o Resumen?                  │   │
          │    └────────────────────────────────────────────┘   │
          │                                                      │
          └──────────────────────────┬───────────────────────────┘
                                     │
               ┌─────────────────────┴─────────────────────┐
               │                                           │
      ┌────────▼─────────┐                    ┌───────────▼──────────┐
      │  CHAT MODE       │                    │  RESUMEN MODE        │
      │  chat_con_..     │                    │  resumir_documento.. │
      └────────┬─────────┘                    └───────────┬──────────┘
               │                                          │
      ┌─-───────▼──────────────────────────┐     ┌────────▼──────────────┐
      │ QA Chain (qa_chain.py)             │     │ Summarizer (summar..) │
      │ ┌──────────────────────────────┐   │     │ ┌────────────────────┐ │
      │ │ 1. RECUPERACIÓN HÍBRIDA      │   │     │ │ Map-Reduce Resume  │ │
      │ │ • Vectorial (20 docs)        │   │     │ │ • Procesa todos    │ │
      │ │ • BM25 (20 docs)             │   │     │ │ • Genera resumen   │ │
      │ │ • Ensemble (combinado)       │   │     │ └────────────────────┘ │
      │ │                              │   │     │                        │
      │ │ 2. RERANKING                 │   │     │ LLM: ChatOpenAI        │
      │ │ • FlashrankRerank            │   │     │ Temp: 0.2              │
      │ │ • Top k_value (default: 5)   │   │     │                        │
      │ │                              │   │     │ Output: Resumen claro  │
      │ │ 3. CONTEXTO ARMADO           │   │     └────────────────────────┘
      │ │ • Best k_value docs          │   │
      │ │ + Chat history               │   │
      │ │                              │   │
      │ │ 4. PROMPT DINÁMICO           │   │
      │ │ ┌──────────────────────────┐ │   │
      │ │ │ BASE INSTRUCTION         │ │   │
      │ │ │ (de template o custom)   │ │   │
      │ │ │                          │ │   │
      │ │ │ + REGLAS DE CITAS        │ │   │
      │ │ │ (5 reglas explícitas)    │ │   │
      │ │ │                          │ │   │
      │ │ │ + CONTEXTO               │ │   │
      │ │ │ (docs recuperados)       │ │   │
      │ │ │                          │ │   │
      │ │ │ + PREGUNTA               │ │   │
      │ │ │ (del usuario)            │ │   │
      │ │ └──────────────────────────┘ │   │
      │ │                              │   │
      │ │ 5. LLM GENERA RESPUESTA      │   │
      │ │ ChatOpenAI (temp=0.2)        │   │
      │ │ ✓ Respuesta citada           │   │
      │ │ ✓ Estructurada               │   │
      │ │ ✓ Profesional                │   │
      │ └──────────────────────────────┘   │
      │                                    │
      └────────────────┬───────────────────┘
                       │
          ┌────────────▼────────────┐
          │ Respuesta + Fuentes     │
          │ (guardar en DB)         │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │ Devolver a Usuario      │
          │ (JSON + visualizar)     │
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │ Usuario ve respuesta    │
          │ + citas desplegables    │
          └─────────────────────────┘


## 2️⃣ SISTEMA DE TEMPLATES


┌────────────────────────────────────────────────────────────┐
│          TEMPLATES PREDEFINIDOS                            │
│        (prompt_templates.py)                               │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  GENÉRICO       │  │  LEGAL          │                  │
│  ├─────────────────┤  ├─────────────────┤                  │
│  │ • Análisis      │  │ • Contratos     │                  │
│  │ • General       │  │ • Normativas    │                  │
│  │ • Default       │  │ • Precisión +++ │                  │
│  └─────────────────┘  └─────────────────┘                  │
│                                                            │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  FINANCIERO     │  │  TÉCNICO        │                  │
│  ├─────────────────┤  ├─────────────────┤                  │
│  │ • Números       │  │ • Código        │                  │
│  │ • Ratios        │  │ • Arquitectura  │                  │
│  │ • Comparativas  │  │ • Versiones     │                  │
│  └─────────────────┘  └─────────────────┘                  │
│                                                            │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  MÉDICO         │  │  COMERCIAL      │                  │
│  ├─────────────────┤  ├─────────────────┤                  │
│  │ • Evidencia     │  │ • Negocios      │                  │
│  │ • Rigor         │  │ • ROI           │                  │
│  │ • Guías clínic  │  │ • Oportunidades │                  │
│  └─────────────────┘  └─────────────────┘                  │
│                                                            │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  RRHH           │  │  INVESTIGADOR   │                  │
│  ├─────────────────┤  ├─────────────────┤                  │
│  │ • Talento       │  │ • Académico     │                  │
│  │ • Compliance    │  │ • Crítica       │                  │
│  │ • Normativo     │  │ • Citas formales│                  │
│  └─────────────────┘  └─────────────────┘                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
         │
         │ Usuario selecciona o personaliza
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Settings del Proyecto (settings.html)                    │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Dropdown: [ Seleccionar Template ▼ ]                │ │
│  │  • Genérico                                          │ │
│  │  • Legal                                             │ │
│  │  • Financiero                                        │ │
│  │  • Técnico                                           │ │
│  │  • Médico                                            │ │
│  │  • Comercial                                         │ │
│  │  • RRHH                                              │ │
│  │  • Investigador                                      │ │
│  │  • Personalizado (escribir)                          │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  [ Guardar ]                                              │
│                                                            │
└────────────────────────────────────────────────────────────┘
         │
         │ Se guarda en project.settings
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Database (SQLite app.db)                                  │
│                                                            │
│  Project                                                   │
│  ├─ id                                                    │
│  ├─ name                                                  │
│  ├─ settings                                              │
│  │  ├─ template_type: "legal"                           │
│  │  ├─ system_instruction: "[prompt del template]"      │
│  │  └─ k_value: 5                                        │
│  └─ ...                                                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
         │
         │ Se usa en get_conversational_qa_chain()
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  QA_PROMPT Dinámico Construido                             │
│  (en memoria, cada consulta)                               │
│                                                            │
│  {base_instruction}  ← Del template                       │
│  + REGLAS DE CITAS   ← Siempre igual                      │
│  + CONTEXTO          ← Docs recuperados                   │
│  + PREGUNTA          ← Del usuario                        │
│                                                            │
└────────────────────────────────────────────────────────────┘


## 3️⃣ CONSTRUCCIÓN DEL PROMPT FINAL


┌─────────────────────────────────────────────────────────────────────────┐
│                    QA_TEMPLATE_STRING (Final)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PART 1: BASE INSTRUCTION                                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ "Eres un ANALISTA EXPERTO y profesional.                        │  │
│  │  Tu objetivo es proporcionar respuestas precisas, bien          │  │
│  │  estructuradas y fundamentadas en los documentos.               │  │
│  │                                                                  │  │
│  │  CARACTERÍSTICAS:                                               │  │
│  │  - Claras y directas                                            │  │
│  │  - Profesionales                                                │  │
│  │  - Estructuradas                                                │  │
│  │  - Fundamentadas                                                │  │
│  │  - Útiles"                                                       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  PART 2: INSTRUCCIONES ESTRICTAS                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 📋 CITA OBLIGATORIA                                             │  │
│  │    Cada dato = fuente (Archivo, página/sección)                │  │
│  │                                                                  │  │
│  │ 2️⃣ PRIORIDAD DE INFORMACIÓN                                    │  │
│  │    - Datos literales = MÁS importantes                         │  │
│  │    - Conceptos = Menos crítico                                 │  │
│  │                                                                  │  │
│  │ 3️⃣ CITAS TEXTUALES                                              │  │
│  │    Si importante → frase exacta entre comillas                 │  │
│  │                                                                  │  │
│  │ 4️⃣ SI NO TIENES INFORMACIÓN                                    │  │
│  │    NO inventes → "No encontré..."                              │  │
│  │                                                                  │  │
│  │ 5️⃣ ESTRUCTURA                                                   │  │
│  │    - RESUMEN/RESPUESTA                                          │  │
│  │    - DETALLES Y ANÁLISIS                                        │  │
│  │    - CONTEXTO O NOTAS                                           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  PART 3: CONTEXTO RECUPERADO                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 🗂️ CONTEXTO DE REFERENCIA:                                     │  │
│  │                                                                  │  │
│  │ Doc 1 (relevancia: 0.94)                                        │  │
│  │ "El beneficio neto en Q3 alcanzó $2.5M según el Estado de...   │  │
│  │  Fuente: Estados_Financieros.xlsx, p. 45"                      │  │
│  │                                                                  │  │
│  │ Doc 2 (relevancia: 0.87)                                        │  │
│  │ "Comparado con Q2 que fue $2.17M, representa un crecimiento... │  │
│  │  Fuente: Balance_Trimestral.pdf, tabla 2"                      │  │
│  │                                                                  │  │
│  │ [5+ documentos más recuperados]                                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  PART 4: LA PREGUNTA                                                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ ❓ PREGUNTA DEL USUARIO:                                        │  │
│  │                                                                  │  │
│  │ "¿Cuál fue el beneficio neto del tercer trimestre?"             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                             │
                             │ Enviado a ChatOpenAI
                             │
                             ▼
                    ┌────────────────┐
                    │ LLM Procesa    │
                    │ + Genera       │
                    │ Respuesta      │
                    └────────┬───────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │ RESPUESTA ESTRUCTURADA Y CITADA        │
        │                                        │
        │ "Según el Estado de Resultados Q3     │
        │  (Estados_Financieros.xlsx, p. 45),   │
        │  el beneficio neto fue de $2.5M       │
        │  (+15% vs Q2 anterior).                │
        │                                        │
        │  Análisis: [estructura + citas]       │
        │                                        │
        │  Fuentes: [listado de referencias]    │
        └────────────────────────────────────────┘



## 4️⃣ DECISIÓN DEL ROUTER


┌──────────────────────────────────────────────────────────────────┐
│                    PREGUNTA LLEGA AL ROUTER                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Pregunta: "¿Cuál fue el beneficio neto de Q3?"                 │
│  Historial: [conversación anterior]                             │
│                                                                  │
│                          ↓                                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │           ROUTER PROMPT (mejorado)                         │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │                                                            │ │
│  │  ✓ ¿Es pregunta ESPECÍFICA? (números, nombres, fechas)   │ │
│  │    SÍ → "¿cuál fue?" ← Palabra clave detectada            │ │
│  │                                                            │ │
│  │  ✓ ¿Pide RESUMEN general? (de qué trata, principales...) │ │
│  │    NO → No pide resumen del documento                     │ │
│  │                                                            │ │
│  │  ✓ ¿Es seguimiento de conversación? (profundización)     │ │
│  │    NO → Primera pregunta sobre este tema                  │ │
│  │                                                            │ │
│  │  DECISIÓN:  Elige "chat_con_documentos"                 │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│                          ↓                                       │
│                                                                  │
│  Resultado: Ejecuta QA_CHAIN                                    │
│            (no summarizer)                                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘


OTRO EJEMPLO:

┌──────────────────────────────────────────────────────────────────┐
│  Pregunta: "Resume el documento"                                 │
│  Historial: [conversación anterior]                              │
│                                                                  │
│                          ↓                                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │           ROUTER PROMPT (mejorado)                         │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │                                                            │ │
│  │  ✓ ¿Es pregunta ESPECÍFICA?                              │ │
│  │    NO → No busca dato específico                          │ │
│  │                                                            │ │
│  │  ✓ ¿Pide RESUMEN?                                        │ │
│  │    SÍ → Palabra "resume" detectada ← Indicador claro    │ │
│  │                                                            │ │
│  │  ✓ Pide "visión general" del proyecto completo?         │ │
│  │    SÍ → CONFIRMADO                                        │ │
│  │                                                            │ │
│  │  DECISIÓN:  Elige "resumir_documento_completo"          │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│                          ↓                                       │
│                                                                  │
│  Resultado: Ejecuta SUMMARIZER                                  │ │
│            (Map-Reduce de todos los docs)                      │ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘


## 5️⃣ CICLO DE VIDA DEL PROYECTO

┌───────────────────────────────────────────────────────────────────────┐
│                    CICLO DE VIDA DEL PROYECTO                         │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  FASE 1: CREACIÓN                                                    │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Usuario crea proyecto                                          │  │
│  │ • Nombre                                                       │  │
│  │ • Ruta de documentos                                           │  │
│  │ • Modelo IA                                                    │  │
│  │                                                                │  │
│  │ Background: Ingesta + Indexing (ChromaDB)                     │  │
│  │ Status: INDEXING → READY                                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  FASE 2: CONFIGURACIÓN INICIAL (NEW)                                 │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Usuario va a Ajustes                                           │  │
│  │ • Selecciona template O personaliza prompt                     │  │
│  │ • Configura k_value                                            │  │
│  │ • Click Guardar                                                │  │
│  │                                                                │  │
│  │ Resultado: project.settings actualizado                       │  │
│  │ Beneficio: Respuestas futuras usarán esta config              │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  FASE 3: CONVERSACIÓN                                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Usuario entra a Chat                                           │  │
│  │ • Hace preguntas normalmente                                   │  │
│  │ • Sistema internamente:                                        │  │
│  │   1. Router decide qué herramienta                             │  │
│  │   2. Herramienta ejecuta con PROMPT mejorado                  │  │
│  │   3. Respuesta es citada + estructurada                        │  │
│  │                                                                │  │
│  │ Impacto: Respuestas profesionales desde la Q1                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  FASE 4: PERSONALIZACIÓN (Opcional)                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Si respuestas no se ajustan:                                   │  │
│  │ • Vuelve a Ajustes                                             │  │
│  │ • Cambia template O modifica prompt                            │  │
│  │ • Prueba nuevamente                                            │  │
│  │                                                                │  │
│  │ Caché: Automáticamente se limpia para aplicar cambios         │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  FASE 5: MANTENIMIENTO                                               │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Proyecto en operación                                          │  │
│  │ • Histórico de preguntas guardado                              │  │
│  │ • Costos calculados automáticamente                            │  │
│  │ • Puede editar y reenviar preguntas anteriores                 │  │
│  │ • Puede limpiar historial si es necesario                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘



## 6️⃣ COMPONENTES CLAVE


COMPONENTES DEL SISTEMA MEJORADO:

┌──────────────────────────────────────┐
│  Router (router.py)                  │ ← Decide herramienta
│  ├─ Prompt con reglas explícitas    │
│  └─ Temp=0 (determinista)            │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│  QA Chain (qa_chain.py)              │ ← Responde preguntas
│  ├─ Retrieval híbrido (Vec + BM25)  │
│  ├─ Reranking (FlashrankRerank)      │
│  ├─ Prompt dinámico mejorado         │
│  └─ LLM (ChatOpenAI, temp=0.2)       │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│  Templates (prompt_templates.py)     │ ← 8 prompts predefinidos
│  ├─ Genérico                         │
│  ├─ Legal                            │
│  ├─ Financiero                       │
│  ├─ Técnico                          │
│  ├─ Médico                           │
│  ├─ Comercial                        │
│  ├─ RRHH                             │
│  └─ Investigador                     │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│  Routes (routes.py)                  │ ← Orquesta todo
│  ├─ /ask → Procesa pregunta           │
│  ├─ /settings → Configuración        │
│  └─ /chat → Interfaz                 │
└──────────────────────────────────────┘



## 7️⃣ VALORES POR DEFECTO


┌─────────────────────────────────────────┐
│    CONFIGURACIÓN POR DEFECTO            │
├─────────────────────────────────────────┤
│                                         │
│  Router Temp:              0 (fixed)   │
│  LLM Temp:                 0.2         │
│  K Value (docs):           5           │
│  Chunk Size:               1500 chars  │
│  Chunk Overlap:            300 chars   │
│  Embeddings:               OpenAI      │
│  Vector Store:             Chroma      │
│  BM25 Retrieval:           k=20        │
│  Reranking Top N:          k_value     │
│  Template Default:         "generico"  │
│  Citas:                    Obligatorias│
│  Estructura:               Jerarquía   │
│                                         │
└─────────────────────────────────────────┘



