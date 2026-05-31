"""Unit tests for dashboard rendering helpers.

Covers:
- Sub-task 7.1: _sort_key unit tests
- Sub-task 7.3: _finding_to_dict unit tests
- Sub-task 7.5: _render_currency_payload unit tests
- Sub-task 7.7: _render_automation_payload and _render_general_finding_detail unit tests
- Sub-task 7.9: _e unit tests
"""

from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone

from migration_watchdog.dashboard import (
    _e,
    _finding_to_dict,
    _render_automation_payload,
    _render_currency_payload,
    _render_general_finding_detail,
    _sort_key,
    GAP_RANK,
    RISK_RANK,
    SEVERITY_RANK,
)
from migration_watchdog.models import Dismissal, Finding, RiskLevel

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helper: build a minimal Finding instance
# ---------------------------------------------------------------------------

def _make_finding(
    finding_id="test-id",
    run_id="run-1",
    risk_level=RiskLevel.MEDIUM,
    category="currency_drift",
    title="Test finding",
    description="desc",
    finding_schema_version=None,
    auditor_payload=None,
    dismissal=None,
):
    return Finding(
        finding_id=finding_id,
        run_id=run_id,
        risk_level=risk_level,
        category=category,
        title=title,
        description=description,
        finding_schema_version=finding_schema_version,
        auditor_payload=auditor_payload,
        dismissal=dismissal,
    )


# ---------------------------------------------------------------------------
# Sub-task 7.1: _sort_key unit tests
# ---------------------------------------------------------------------------

class TestSortKey:
    """Tests for _sort_key()."""

    def test_currency_finding_uses_severity_rank(self):
        """Currency finding: primary sort key is SEVERITY_RANK[severity]."""
        f = _make_finding(
            finding_schema_version="currency/1.0",
            auditor_payload={"severity": "correctness"},
        )
        key = _sort_key(f)
        assert key[1] == SEVERITY_RANK["correctness"]

    def test_currency_finding_outdated_severity(self):
        """Currency finding with 'outdated' severity uses correct rank."""
        f = _make_finding(
            finding_schema_version="currency/1.0",
            auditor_payload={"severity": "outdated"},
        )
        key = _sort_key(f)
        assert key[1] == SEVERITY_RANK["outdated"]

    def test_automation_finding_uses_gap_rank(self):
        """Automation finding: primary sort key is GAP_RANK[gap_type]."""
        f = _make_finding(
            finding_schema_version="automation/1.0",
            auditor_payload={"gap_type": "full_gap"},
        )
        key = _sort_key(f)
        assert key[1] == GAP_RANK["full_gap"]

    def test_automation_finding_partial_gap(self):
        """Automation finding with 'partial_gap' uses correct rank."""
        f = _make_finding(
            finding_schema_version="automation/1.0",
            auditor_payload={"gap_type": "partial_gap"},
        )
        key = _sort_key(f)
        assert key[1] == GAP_RANK["partial_gap"]

    def test_legacy_finding_uses_risk_rank(self):
        """Legacy finding (no schema): primary sort key is RISK_RANK[risk_level]."""
        f = _make_finding(
            finding_schema_version=None,
            auditor_payload=None,
            risk_level=RiskLevel.HIGH,
        )
        key = _sort_key(f)
        assert key[1] == RISK_RANK["high"]

    def test_legacy_finding_medium_risk(self):
        """Legacy finding with MEDIUM risk uses correct rank."""
        f = _make_finding(
            finding_schema_version=None,
            auditor_payload=None,
            risk_level=RiskLevel.MEDIUM,
        )
        key = _sort_key(f)
        assert key[1] == RISK_RANK["medium"]

    def test_unknown_severity_falls_back_to_99(self):
        """Unknown severity value falls back to 99 (sorts last)."""
        f = _make_finding(
            finding_schema_version="currency/1.0",
            auditor_payload={"severity": "unknown_value"},
        )
        key = _sort_key(f)
        assert key[1] == 99

    def test_sort_key_returns_tuple(self):
        """_sort_key always returns a 3-tuple."""
        f = _make_finding()
        key = _sort_key(f)
        assert isinstance(key, tuple)
        assert len(key) == 3


# ---------------------------------------------------------------------------
# Sub-task 7.3: _finding_to_dict unit tests
# ---------------------------------------------------------------------------

class TestFindingToDict:
    """Tests for _finding_to_dict()."""

    def test_no_dismissal_returns_false(self):
        """Finding with no dismissal → dismissal_active == False."""
        f = _make_finding(dismissal=None)
        result = _finding_to_dict(f)
        assert result["dismissal_active"] is False

    def test_no_dismissal_dismissal_field_is_none(self):
        """Finding with no dismissal → dismissal field is None."""
        f = _make_finding(dismissal=None)
        result = _finding_to_dict(f)
        assert result["dismissal"] is None

    def test_future_cooldown_returns_true(self):
        """Dismissal with future cooldown_expires → dismissal_active == True."""
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        d = Dismissal(
            dismissed_at=datetime.now(UTC).isoformat(),
            cooldown_expires=future,
        )
        f = _make_finding(dismissal=d)
        result = _finding_to_dict(f)
        assert result["dismissal_active"] is True

    def test_past_cooldown_returns_false(self):
        """Dismissal with past cooldown_expires → dismissal_active == False."""
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        d = Dismissal(
            dismissed_at=datetime.now(UTC).isoformat(),
            cooldown_expires=past,
        )
        f = _make_finding(dismissal=d)
        result = _finding_to_dict(f)
        assert result["dismissal_active"] is False

    def test_result_contains_expected_keys(self):
        """_finding_to_dict result contains all expected top-level keys."""
        f = _make_finding()
        result = _finding_to_dict(f)
        for key in ("finding_id", "run_id", "risk_level", "category", "title",
                    "description", "status", "dismissal", "dismissal_active"):
            assert key in result, f"Missing key: {key!r}"

    def test_risk_level_serialised_as_string(self):
        """risk_level is serialised as a plain string, not a RiskLevel enum."""
        f = _make_finding(risk_level=RiskLevel.HIGH)
        result = _finding_to_dict(f)
        assert result["risk_level"] == "high"
        assert isinstance(result["risk_level"], str)


# ---------------------------------------------------------------------------
# Sub-task 7.5: _render_currency_payload unit tests
# ---------------------------------------------------------------------------

class TestRenderCurrencyPayload:
    """Tests for _render_currency_payload()."""

    def test_severity_banner_correctness(self):
        """Severity 'correctness' produces non-empty HTML."""
        f = _make_finding(finding_schema_version="currency/1.0", auditor_payload={"severity": "correctness"})
        result = _render_currency_payload({"severity": "correctness"}, f)
        assert result.strip() != ""

    def test_severity_banner_outdated(self):
        """Severity 'outdated' produces non-empty HTML."""
        f = _make_finding(finding_schema_version="currency/1.0", auditor_payload={"severity": "outdated"})
        result = _render_currency_payload({"severity": "outdated"}, f)
        assert result.strip() != ""

    def test_severity_banner_policy_change(self):
        """Severity 'policy_change' produces non-empty HTML."""
        f = _make_finding(finding_schema_version="currency/1.0", auditor_payload={"severity": "policy_change"})
        result = _render_currency_payload({"severity": "policy_change"}, f)
        assert result.strip() != ""

    def test_severity_banner_informational(self):
        """Severity 'informational' produces non-empty HTML."""
        f = _make_finding(finding_schema_version="currency/1.0", auditor_payload={"severity": "informational"})
        result = _render_currency_payload({"severity": "informational"}, f)
        assert result.strip() != ""

    def test_claim_text_appears_escaped(self):
        """claim_text is HTML-escaped and appears in the output."""
        claim = "App Runner costs $0.05/hour"
        payload = {"claim_text": claim}
        f = _make_finding(finding_schema_version="currency/1.0", auditor_payload=payload)
        result = _render_currency_payload(payload, f)
        assert html.escape(claim) in result

    def test_xss_script_tag_not_unescaped(self):
        """XSS payload in claim_text must not appear unescaped in output."""
        xss = '<script>alert("xss")</script>'
        payload = {"claim_text": xss}
        f = _make_finding(finding_schema_version="currency/1.0", auditor_payload=payload)
        result = _render_currency_payload(payload, f)
        assert "<script>" not in result

    def test_xss_escaped_form_present(self):
        """XSS payload appears in escaped form in the output."""
        xss = '<script>alert("xss")</script>'
        payload = {"claim_text": xss}
        f = _make_finding(finding_schema_version="currency/1.0", auditor_payload=payload)
        result = _render_currency_payload(payload, f)
        assert "&lt;script&gt;" in result

    def test_empty_payload_returns_string(self):
        """Empty payload returns a string (may be empty or minimal HTML)."""
        f = _make_finding(finding_schema_version="currency/1.0", auditor_payload={})
        result = _render_currency_payload({}, f)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Sub-task 7.7: _render_automation_payload and _render_general_finding_detail
# ---------------------------------------------------------------------------

class TestRenderAutomationPayload:
    """Tests for _render_automation_payload()."""

    def test_action_text_appears_in_output(self):
        """action_text is HTML-escaped and appears in the rendered output."""
        action = "Navigate to the IAM console"
        payload = {"action_text": action}
        f = _make_finding(finding_schema_version="automation/1.0", auditor_payload=payload)
        result = _render_automation_payload(payload, f)
        assert result.strip() != ""
        assert html.escape(action) in result

    def test_output_is_non_empty_with_full_payload(self):
        """A payload with multiple fields produces non-empty HTML."""
        payload = {
            "action_text": "Create an S3 bucket",
            "gap_type": "full_gap",
            "action_type": "console_navigation",
            "confidence": "high",
        }
        f = _make_finding(finding_schema_version="automation/1.0", auditor_payload=payload)
        result = _render_automation_payload(payload, f)
        assert result.strip() != ""

    def test_xss_in_action_text_is_escaped(self):
        """XSS payload in action_text must not appear unescaped."""
        xss = '<img src=x onerror=alert(1)>'
        payload = {"action_text": xss}
        f = _make_finding(finding_schema_version="automation/1.0", auditor_payload=payload)
        result = _render_automation_payload(payload, f)
        assert "<img" not in result

    def test_empty_payload_returns_string(self):
        """Empty payload returns a string."""
        f = _make_finding(finding_schema_version="automation/1.0", auditor_payload={})
        result = _render_automation_payload({}, f)
        assert isinstance(result, str)


class TestRenderGeneralFindingDetail:
    """Tests for _render_general_finding_detail()."""

    def test_description_appears_in_output(self):
        """Finding description text appears in the rendered HTML."""
        f = _make_finding(description="Update the pricing table")
        result = _render_general_finding_detail(f)
        assert result.strip() != ""

    def test_output_is_non_empty(self):
        """_render_general_finding_detail returns non-empty HTML for a basic finding."""
        f = _make_finding(
            description="Update the pricing table",
            category="pricing",
        )
        result = _render_general_finding_detail(f)
        assert result.strip() != ""

    def test_xss_in_description_is_escaped(self):
        """XSS payload in description must not appear unescaped."""
        xss = '<script>alert("xss")</script>'
        f = _make_finding(description=xss)
        result = _render_general_finding_detail(f)
        assert "<script>" not in result

    def test_empty_description_returns_empty_string(self):
        """Finding with empty description returns empty string."""
        f = _make_finding(description="")
        result = _render_general_finding_detail(f)
        assert result == ""


# ---------------------------------------------------------------------------
# Sub-task 7.9: _e unit tests
# ---------------------------------------------------------------------------

class TestEscapeHelper:
    """Tests for _e() HTML escape helper."""

    def test_escapes_angle_brackets(self):
        """< and > are escaped to &lt; and &gt;."""
        result = _e('<script>&"test"</script>')
        assert "&lt;" in result
        assert "&gt;" in result

    def test_escapes_ampersand(self):
        """& is escaped to &amp;."""
        result = _e('<script>&"test"</script>')
        assert "&amp;" in result

    def test_escapes_double_quote(self):
        """Double quotes are escaped to &quot;."""
        result = _e('<script>&"test"</script>')
        assert "&quot;" in result

    def test_none_returns_empty_string(self):
        """_e(None) returns an empty string."""
        assert _e(None) == ""

    def test_plain_text_unchanged(self):
        """Plain text without special characters is returned unchanged."""
        assert _e("hello world") == "hello world"

    def test_empty_string_unchanged(self):
        """Empty string is returned as-is."""
        assert _e("") == ""

    def test_all_special_chars_in_one_call(self):
        """All four special HTML characters are escaped in a single call."""
        result = _e('<script>&"test"</script>')
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result
        assert "&quot;" in result
