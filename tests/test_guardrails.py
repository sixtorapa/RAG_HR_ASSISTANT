"""
test_guardrails.py — the security controls, asserted rather than assumed.

Two guardrails, two properties worth being able to demonstrate:

  RBAC fail-closed: an empty department list must DENY, not open up. That is
  the classic silent failure — treating "empty" as "unrestricted" passes the
  tests nobody wrote and leaks in production.

  Input-side DLP: IBAN, card and DNI/NIE are validated by check digit, not by
  resemblance. A checksum has no random false positives, and that is testable:
  invented numbers with the right shape must go through undetected.
"""

import pytest

from app.rag_logic.pii_guard import (
    contains_sensitive_data,
    find_sensitive_entities,
    mask,
)
from app.rag_logic.qa_chain import _build_security_filter, _combine_filters


# ══════════════════════════════════════════════════════════════════════════
# RBAC — department access control
# ══════════════════════════════════════════════════════════════════════════

class TestRBACFailClosed:

    def test_empty_list_denies_everything(self):
        """
        The central property. `[]` does not mean "unrestricted", it means
        "no access": a department no chunk can ever carry is injected,
        de modo que la consulta devuelve cero resultados.
        """
        f = _build_security_filter([])
        assert f == {"department": "__no_access__"}

    def test_an_empty_list_does_not_produce_a_null_filter(self):
        """
        The failure being prevented: returning None here would let the
        retriever search the WHOLE corpus, and a user with no permissions
        would see everything.
        """
        assert _build_security_filter([]) is not None

    def test_none_is_unrestricted_admin_only(self):
        # None is produced by User.get_allowed_departments() only when role=admin.
        assert _build_security_filter(None) is None

    def test_a_list_restricts_to_those_departments(self):
        f = _build_security_filter(["hr", "it"])
        assert f == {"department": {"$in": ["hr", "it"]}}

    def test_departments_are_sorted(self):
        """
        The same permission written in another order is the same filter. It
        matters because the
        security scope is part of the chain cache key.
        """
        assert _build_security_filter(["it", "hr"]) == _build_security_filter(["hr", "it"])

    def test_departments_are_deduplicated_and_normalised(self):
        f = _build_security_filter(["HR", "hr", " it "])
        assert f == {"department": {"$in": ["hr", "it"]}}

    def test_a_list_of_blank_strings_also_denies(self):
        """Edge case: ["", "  "] leaves no valid values -> deny, not allow."""
        assert _build_security_filter(["", "   "]) == {"department": "__no_access__"}


class TestFilterAlwaysAnded:

    def test_security_filter_is_combined_with_the_functional_one(self):
        seguridad = _build_security_filter(["hr"])
        funcional = {"relative_path_norm": "hr/politica.pdf"}
        combinado = _combine_filters(funcional, seguridad)
        assert combinado == {"$and": [funcional, seguridad]}

    def test_a_single_filter_is_not_wrapped(self):
        assert _combine_filters(None, {"department": "hr"}) == {"department": "hr"}

    def test_no_filters_returns_none(self):
        assert _combine_filters(None, None) is None

    def test_the_security_filter_cannot_be_lost_when_combining(self):
        """
        A document outside the scope cannot come back even when the
        functional filter asks for it explicitly: the two are ANDed.
        """
        seguridad = _build_security_filter([])
        combinado = _combine_filters({"relative_path_norm": "finanzas/secreto.pdf"}, seguridad)
        assert seguridad in combinado["$and"]


# ══════════════════════════════════════════════════════════════════════════
# DLP — identifier detection by check digit
# ══════════════════════════════════════════════════════════════════════════

class TestIBAN:

    # IBANs with a valid mod-97 checksum
    @pytest.mark.parametrize("iban", [
        "ES9121000418450200051332",
        "ES91 2100 0418 4502 0005 1332",
        "GB82WEST12345698765432",
        "DE89370400440532013000",
    ])
    def test_detects_a_valid_iban(self, iban):
        findings = find_sensitive_entities(f"mi cuenta es {iban}, gracias")
        assert "IBAN" in {h.kind for h in findings}

    @pytest.mark.parametrize("fake", [
        "ES9121000418450200051333",   # same IBAN with the last digit changed
        "ES0000000000000000000000",
        "XX1234567890123456789012",
    ])
    def test_does_not_detect_an_iban_with_a_broken_checksum(self, fake):
        """
        What makes the checksum valuable: a number that LOOKS like an IBAN
        but is miscalculated does not fire. Without it, any long alphanumeric
        string would be a false positive.
        """
        findings = find_sensitive_entities(f"referencia {fake}")
        assert "IBAN" not in {h.kind for h in findings}


class TestCard:

    @pytest.mark.parametrize("card", [
        "4539578763621486",          # test Visa, valid Luhn
        "4539 5787 6362 1486",
        "5500005555555559",
    ])
    def test_detects_a_valid_card(self, card):
        findings = find_sensitive_entities(f"pagué con la {card}")
        assert "CARD" in {h.kind for h in findings}

    def test_does_not_detect_just_any_long_number(self):
        findings = find_sensitive_entities("el pedido 1234567812345678 llegó tarde")
        assert "CARD" not in {h.kind for h in findings}


class TestDNIAndNIE:

    @pytest.mark.parametrize("doc", ["12345678Z", "00000000T", "X1234567L"])
    def test_detects_document_valid(self, doc):
        findings = find_sensitive_entities(f"mi documento es {doc}")
        assert "DNI_NIE" in {h.kind for h in findings}

    @pytest.mark.parametrize("doc", ["12345678A", "00000000Z", "X1234567Z"])
    def test_does_not_detect_a_wrong_control_letter(self, doc):
        findings = find_sensitive_entities(f"referencia {doc}")
        assert "DNI_NIE" not in {h.kind for h in findings}


class TestGuardrailPolicy:

    def test_clean_text_triggers_nothing(self):
        assert find_sensitive_entities("¿cuántos días de vacaciones tengo?") == []
        assert not contains_sensitive_data("¿cuál es la política de teletrabajo?")

    def test_empty_text_does_not_blow_up(self):
        assert find_sensitive_entities("") == []
        assert find_sensitive_entities(None) == []

    def test_the_value_detectado_never_records_completo(self):
        """
        Findings go to the log. Storing the whole value would turn the
        detector de fugas en la fuga.
        """
        iban = "ES9121000418450200051332"
        findings = find_sensitive_entities(f"cuenta {iban}")
        assert findings
        for h in findings:
            assert iban not in h.masked
            assert "*" in h.masked

    def test_the_mascara_keeps_the_ends(self):
        assert mask("ES9121000418450200051332").startswith("ES")
        assert mask("ES9121000418450200051332").endswith("32")

    def test_several_entities_in_the_same_text(self):
        text = "soy 12345678Z y mi cuenta ES9121000418450200051332"
        tipos = {h.kind for h in find_sensitive_entities(text)}
        assert {"DNI_NIE", "IBAN"} <= tipos
