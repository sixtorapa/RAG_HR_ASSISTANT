# app/rag_logic/pii_guard.py
"""
Guardarril de entrada (input-side DLP): detecta identificadores financieros/
personales estructurados en texto libre ANTES de que llegue al LLM o se
persista en la base de datos propia.

Deliberadamente NO intenta detectar nombres de persona en texto libre — eso
necesita NER (Presidio, spaCy) o una llamada a un LLM clasificador, y tiene
falsos positivos/negativos reales. Lo que hay aquí son identificadores
ESTRUCTURADOS con dígito de control matemático (IBAN, tarjeta, DNI/NIE):
barato, determinista, prácticamente cero falsos positivos.

Política: BLOQUEAR, no redactar y continuar (ver discusión de la sesión).
Redactar y seguir es arriesgado — si el detector falla un campo, el dato se
cuela igual. Bloquear con un mensaje claro es el patrón "fail closed" esperado
en banca.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Finding:
    kind: str        # "IBAN" | "CARD" | "DNI_NIE"
    masked: str       # seguro de loguear: nunca el valor completo


def mask(value: str, keep_start: int = 2, keep_end: int = 2) -> str:
    """Enmascara un valor sensible para logging seguro (nunca el valor completo)."""
    cleaned = re.sub(r"\s+", "", value)
    if len(cleaned) <= keep_start + keep_end:
        return "*" * len(cleaned)
    return cleaned[:keep_start] + "*" * (len(cleaned) - keep_start - keep_end) + cleaned[-keep_end:]


# ==================== IBAN (mod-97) ====================

_IBAN_RE = re.compile(r"\b[A-Za-z]{2}[ -]?\d{2}(?:[ -]?[A-Za-z0-9]{2,4}){3,8}\b")

_IBAN_LETTER_VALUES = {chr(c): str(c - 55) for c in range(ord("A"), ord("Z") + 1)}


def _iban_checksum_valid(candidate: str) -> bool:
    iban = re.sub(r"[ -]", "", candidate).upper()
    if not (15 <= len(iban) <= 34):
        return False
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", iban):
        return False
    rearranged = iban[4:] + iban[:4]
    digits = "".join(_IBAN_LETTER_VALUES.get(ch, ch) for ch in rearranged)
    try:
        return int(digits) % 97 == 1
    except ValueError:
        return False


# ==================== Tarjeta (Luhn) ====================

_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_valid(candidate: str) -> bool:
    digits = re.sub(r"[ -]", "", candidate)
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ==================== DNI / NIE español ====================

_DNI_RE = re.compile(r"\b\d{8}[A-Za-z]\b")
_NIE_RE = re.compile(r"\b[XYZxyz]\d{7}[A-Za-z]\b")
_DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"
_NIE_PREFIX = {"X": "0", "Y": "1", "Z": "2"}


def _dni_nie_checksum_valid(candidate: str) -> bool:
    doc = candidate.strip().upper()
    if re.fullmatch(r"\d{8}[A-Z]", doc):
        number, letter = doc[:8], doc[8]
    elif re.fullmatch(r"[XYZ]\d{7}[A-Z]", doc):
        number, letter = _NIE_PREFIX[doc[0]] + doc[1:8], doc[8]
    else:
        return False
    return _DNI_LETTERS[int(number) % 23] == letter


# ==================== API pública ====================

def find_sensitive_entities(text: str) -> List[Finding]:
    """Escanea texto libre y devuelve los identificadores financieros/personales
    estructurados detectados (con checksum válido). Lista vacía = nada detectado."""
    if not text:
        return []

    findings: List[Finding] = []

    for m in _IBAN_RE.finditer(text):
        if _iban_checksum_valid(m.group(0)):
            findings.append(Finding(kind="IBAN", masked=mask(m.group(0))))

    for m in _CARD_RE.finditer(text):
        if _luhn_valid(m.group(0)):
            findings.append(Finding(kind="CARD", masked=mask(m.group(0))))

    for m in _DNI_RE.finditer(text):
        if _dni_nie_checksum_valid(m.group(0)):
            findings.append(Finding(kind="DNI_NIE", masked=mask(m.group(0))))
    for m in _NIE_RE.finditer(text):
        if _dni_nie_checksum_valid(m.group(0)):
            findings.append(Finding(kind="DNI_NIE", masked=mask(m.group(0))))

    return findings


def contains_sensitive_data(text: str) -> bool:
    return bool(find_sensitive_entities(text))
