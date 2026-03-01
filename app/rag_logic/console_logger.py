# app/rag_logic/console_logger.py
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Compatibilidad LangChain (nuevo / viejo)
try:
    from langchain_core.callbacks import BaseCallbackHandler  # type: ignore
except Exception:
    from langchain.callbacks.base import BaseCallbackHandler  # type: ignore


# ==============================
# Utils
# ==============================
def _enable_windows_ansi() -> None:
    """Activa ANSI en Windows si es posible. Si falla, no pasa nada."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        return


class _C:
    _USE = sys.stdout.isatty()

    DIM = "\033[2m" if _USE else ""
    BOLD = "\033[1m" if _USE else ""
    OK = "\033[92m" if _USE else ""
    WARN = "\033[93m" if _USE else ""
    ERR = "\033[91m" if _USE else ""
    CYAN = "\033[96m" if _USE else ""
    END = "\033[0m" if _USE else ""


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("\r\n", "\n")
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _as_text(x: Any, n: int = 220) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return _truncate(x, n)
    if isinstance(x, (int, float, bool)):
        return str(x)
    # LangChain messages
    if hasattr(x, "content"):
        try:
            return _truncate(str(getattr(x, "content", "")), n)
        except Exception:
            pass
    return _truncate(_safe_json(x), n)


def _get_name(serialized: Any, fallback: str) -> str:
    if isinstance(serialized, dict):
        return serialized.get("name") or serialized.get("id") or fallback
    return fallback


def _extract_tool_calls_from_message(msg: Any) -> List[Dict[str, Any]]:
    if msg is None:
        return []

    # OpenAI-style additional_kwargs={"tool_calls":[...]}
    try:
        ak = getattr(msg, "additional_kwargs", None) or {}
        tc = ak.get("tool_calls")
        if isinstance(tc, list):
            return [t if isinstance(t, dict) else {"raw": t} for t in tc]
    except Exception:
        pass

    # Algunas versiones exponen msg.tool_calls
    try:
        tc2 = getattr(msg, "tool_calls", None)
        if isinstance(tc2, list):
            return [t if isinstance(t, dict) else {"raw": t} for t in tc2]
    except Exception:
        pass

    return []


def _summarize_tool_calls(tool_calls: List[Dict[str, Any]], max_items: int = 3) -> str:
    if not tool_calls:
        return ""
    names: List[str] = []
    for tc in tool_calls[: max_items]:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if isinstance(fn, dict):
            names.append(fn.get("name") or "unknown_tool")
        else:
            names.append(tc.get("name") or "unknown_tool")
    more = ""
    if len(tool_calls) > max_items:
        more = f" +{len(tool_calls) - max_items}"
    return f" tools={','.join(names)}{more}"


def _extract_finish_reason(response: Any) -> Optional[str]:
    # LLMResult -> generations[0][0].generation_info.finish_reason
    try:
        gens = getattr(response, "generations", None)
        if gens and gens[0] and gens[0][0]:
            gi = getattr(gens[0][0], "generation_info", None) or {}
            fr = gi.get("finish_reason")
            if fr:
                return str(fr)
    except Exception:
        pass

    # AIMessage.response_metadata.finish_reason
    try:
        gens = getattr(response, "generations", None)
        if gens and gens[0] and gens[0][0]:
            msg = getattr(gens[0][0], "message", None)
            rm = getattr(msg, "response_metadata", None) or {}
            fr = rm.get("finish_reason")
            if fr:
                return str(fr)
    except Exception:
        pass

    return None


@dataclass
class _RunInfo:
    name: str
    kind: str
    parent: Optional[str]
    start_ts: float


class ConsoleLogger(BaseCallbackHandler):
    """
    Logger COMPACTO para producción/debug rápido.

    Objetivo:
    - No inundar PowerShell con prompts y dumps enormes
    - Sí mostrar el flujo: Chain/Router -> Tool -> Retriever -> LLM End
    - Sí mostrar duraciones y herramientas elegidas
    - Sí mostrar tool_calls cuando el LLM responde con finish_reason=tool_calls

    Por defecto imprime ~1 línea por evento.
    """

    def __init__(
        self,
        # Modo: "compact" (default) o "debug" (muestra inputs/output cortos)
        mode: str = "compact",
        # Cuántos caracteres máximos para inputs/outputs si mode="debug"
        max_debug_chars: int = 220,
        # Mostrar retriever (docs recuperados)
        show_retriever: bool = True,
    ) -> None:
        super().__init__()
        _enable_windows_ansi()
        self.mode = (mode or "compact").strip().lower()
        self.max_debug_chars = max_debug_chars
        self.show_retriever = show_retriever

        self._runs: Dict[str, _RunInfo] = {}

    # --------------------------
    # Helpers
    # --------------------------
    def _register(self, run_id: Optional[str], name: str, kind: str, parent: Optional[str]) -> None:
        if not run_id:
            return
        self._runs[run_id] = _RunInfo(name=name or kind, kind=kind, parent=parent, start_ts=time.time())

    def _close(self, run_id: Optional[str]) -> None:
        if not run_id:
            return
        self._runs.pop(run_id, None)

    def _dur(self, run_id: Optional[str]) -> str:
        if not run_id or run_id not in self._runs:
            return ""
        dt = time.time() - self._runs[run_id].start_ts
        return f"{dt:.2f}s"

    def _depth(self, run_id: Optional[str]) -> int:
        if not run_id or run_id not in self._runs:
            return 0
        depth = 0
        cur = self._runs.get(run_id)
        while cur and cur.parent:
            depth += 1
            cur = self._runs.get(cur.parent)
            if depth > 8:
                break
        return depth

    def _pad(self, run_id: Optional[str]) -> str:
        # Compacto: 2 espacios por nivel
        return "  " * self._depth(run_id)

    def _print(self, line: str) -> None:
        print(line, flush=True)

    def _debug_kv(self, prefix: str, value: Any) -> str:
        return f" {prefix}={_as_text(value, self.max_debug_chars)}"

    # ==========================
    # Chain
    # ==========================
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        parent = kwargs.get("parent_run_id")
        name = _get_name(serialized, "Chain")
        self._register(run_id, name, "chain", str(parent) if parent else None)

        p = self._pad(run_id)

        # ✅ 1) Mostrar SIEMPRE la pregunta del usuario en el chain raíz (depth==0)
        q = None
        if self._depth(run_id) == 0:
            if isinstance(inputs, dict):
                q = inputs.get("input") or inputs.get("question")
            elif isinstance(inputs, str):
                q = inputs

        extra = ""
        if q:
            extra += f" {_C.DIM}Q={_C.END}{_as_text(q, self.max_debug_chars)}"

        # En debug añadimos un poquito más, pero sin inundar
        if self.mode == "debug":
            payload = inputs.get("input") if isinstance(inputs, dict) else inputs
            extra += self._debug_kv("in", payload)

        self._print(f"{p}{_C.CYAN}▶ chain{_C.END} {name}{_C.DIM} [{_now()}]{_C.END}{extra}")



    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        p = self._pad(run_id)
        dur = self._dur(run_id)

        extra = ""
        if self.mode == "debug":
            # A veces outputs={'output':...}
            out = outputs.get("output") if isinstance(outputs, dict) else outputs
            extra = self._debug_kv("out", out)

        self._print(f"{p}{_C.OK}✓ chain{_C.END} {_get_name({}, 'end')}{_C.DIM} ({dur}){_C.END}{extra}")
        self._close(run_id)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        p = self._pad(run_id)
        self._print(f"{p}{_C.ERR}✗ chain error{_C.END} {error}")
        self._close(run_id)

    # ==========================
    # Tool
    # ==========================
    def on_tool_start(self, serialized: Dict[str, Any], input_str: Any = None, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        parent = kwargs.get("parent_run_id")

        tool_name = None
        if isinstance(serialized, dict):
            tool_name = serialized.get("name") or serialized.get("id")
        tool_name = tool_name or "tool"

        self._register(run_id, tool_name, "tool", str(parent) if parent else None)
        p = self._pad(run_id)

        extra = ""
        if self.mode == "debug":
            tool_input = kwargs.get("input") or kwargs.get("inputs") or input_str
            extra = self._debug_kv("args", tool_input)

        self._print(f"{p}{_C.WARN}▶ tool{_C.END} {tool_name}{_C.DIM} [{_now()}]{_C.END}{extra}")

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        p = self._pad(run_id)
        dur = self._dur(run_id)

        extra = ""
        if self.mode == "debug":
            extra = self._debug_kv("out", output)

        self._print(f"{p}{_C.OK}✓ tool{_C.END} ({dur}){extra}")
        self._close(run_id)

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        p = self._pad(run_id)
        self._print(f"{p}{_C.ERR}✗ tool error{_C.END} {error}")
        self._close(run_id)

    # ==========================
    # LLM / ChatModel
    # ==========================
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        # En compacto NO imprimimos prompts (es el ruido principal)
        run_id = str(kwargs.get("run_id") or "")
        parent = kwargs.get("parent_run_id")
        name = _get_name(serialized, "LLM")
        self._register(run_id, name, "llm", str(parent) if parent else None)

        if self.mode == "debug":
            p = self._pad(run_id)
            # Solo 1 prompt truncado, nada más
            pr = prompts[0] if prompts else ""
            self._print(f"{p}{_C.CYAN}▶ llm{_C.END} {name}{self._debug_kv('prompt', pr)}")

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[Any]], **kwargs: Any) -> None:
        # Igual: en compacto no mostramos mensajes
        run_id = str(kwargs.get("run_id") or "")
        parent = kwargs.get("parent_run_id")
        name = _get_name(serialized, "ChatModel")
        self._register(run_id, name, "chat_model", str(parent) if parent else None)

        if self.mode == "debug":
            p = self._pad(run_id)
            conv = messages[0] if messages else []
            last = conv[-1] if conv else None
            self._print(f"{p}{_C.CYAN}▶ chat{_C.END} {name}{self._debug_kv('last', last)}")


    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        p = self._pad(run_id)
        dur = self._dur(run_id)

        finish = _extract_finish_reason(response) or "end"

        tool_calls_summary = ""
        route_line = ""

        try:
            gens = getattr(response, "generations", None)
            if gens and gens[0] and gens[0][0]:
                gen0 = gens[0][0]
                msg = getattr(gen0, "message", None)

                # tool calls
                tcs = _extract_tool_calls_from_message(msg)
                tool_calls_summary = _summarize_tool_calls(tcs)

                # ✅ 2) Si el router escribe ROUTE: ... en content, lo mostramos (1 línea)
                content = getattr(msg, "content", "") if msg else ""
                content = (content or "").strip()
                if content:
                    # Solo una línea corta para no meter ruido
                    first_line = content.splitlines()[0].strip()
                    if first_line.upper().startswith("ROUTE:"):
                        route_line = " " + _truncate(first_line, self.max_debug_chars)
        except Exception:
            pass

        if self.mode == "compact":
            self._print(
                f"{p}{_C.OK}✓ llm{_C.END} ({dur})"
                f"{_C.DIM} finish={finish}{tool_calls_summary}{_C.END}{route_line}"
            )
            self._close(run_id)
            return


    def on_chat_model_end(self, response: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self.on_llm_end(response, **kwargs)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        p = self._pad(run_id)
        self._print(f"{p}{_C.ERR}✗ llm error{_C.END} {error}")
        self._close(run_id)

    # ==========================
    # Retriever
    # ==========================
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs: Any) -> None:
        if not self.show_retriever:
            return
        run_id = str(kwargs.get("run_id") or "")
        parent = kwargs.get("parent_run_id")
        name = _get_name(serialized, "Retriever")
        self._register(run_id, name, "retriever", str(parent) if parent else None)

        p = self._pad(run_id)
        extra = ""
        if self.mode == "debug":
            extra = self._debug_kv("q", query)
        self._print(f"{p}{_C.CYAN}▶ retriever{_C.END} {name}{extra}")

    def on_retriever_end(self, documents: List[Any], **kwargs: Any) -> None:
        if not self.show_retriever:
            return
        run_id = str(kwargs.get("run_id") or "")
        p = self._pad(run_id)
        dur = self._dur(run_id)
        n = len(documents) if documents else 0

        # En debug, mostramos 1-2 fuentes
        src_extra = ""
        if self.mode == "debug":
            srcs: List[str] = []
            for d in (documents or [])[:2]:
                meta = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {})
                src = meta.get("relative_path") or meta.get("source") or meta.get("filename") or ""
                if src:
                    srcs.append(str(src))
            if srcs:
                src_extra = f" src={_truncate(' | '.join(srcs), self.max_debug_chars)}"

        self._print(f"{p}{_C.OK}✓ retriever{_C.END} ({dur}) docs={n}{src_extra}")
        self._close(run_id)

    def on_retriever_error(self, error: BaseException, **kwargs: Any) -> None:
        if not self.show_retriever:
            return
        run_id = str(kwargs.get("run_id") or "")
        p = self._pad(run_id)
        self._print(f"{p}{_C.ERR}✗ retriever error{_C.END} {error}")
        self._close(run_id)
