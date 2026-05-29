"""Tests for GapClassifier (Task 4.7).

All ``_fetch_source_url`` calls are mocked to return a fixed URL so tests
do not make network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure migration_watchdog resolves as a package.
# ---------------------------------------------------------------------------
_PACKAGE_ROOT = Path(__file__).parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from migration_watchdog.coverage_assessor import CoverageResult
from migration_watchdog.gap_classifier import GapClassifier
from migration_watchdog.models import RiskLevel

# Fixed URL returned by the mocked _fetch_source_url
_FIXED_URL = "https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    persona_id: str = "test-persona",
    file_path: str = "ai-anthropic-to-bedrock.md",
    coverage: str = "gap",
    gaps: list[str] | None = None,
    confidence: float = 0.9,
    suggested_additions: list[str] | None = None,
) -> CoverageResult:
    """Build a minimal CoverageResult for testing."""
    return CoverageResult(
        persona_id=persona_id,
        file_path=file_path,
        coverage=coverage,
        gaps=gaps or [],
        confidence=confidence,
        suggested_additions=suggested_additions or [],
    )


def _classify_with_mock(results, traces=None):
    """Run GapClassifier.classify() with _fetch_source_url mocked."""
    classifier = GapClassifier(run_id="test-run")
    with patch.object(GapClassifier, "_fetch_source_url", return_value=_FIXED_URL):
        return classifier.classify(results, traces or [])


# ---------------------------------------------------------------------------
# Test 1: missing_file produces HIGH severity finding
# ---------------------------------------------------------------------------


def test_missing_file_produces_high_severity_finding():
    """A missing_file CoverageResult produces a HIGH severity Finding."""
    result = _make_result(
        coverage="missing_file",
        gaps=["File 'ai-anthropic-to-bedrock.md' does not exist in the repository"],
        confidence=1.0,
    )
    findings = _classify_with_mock([result])

    assert len(findings) == 1
    finding = findings[0]
    assert finding.risk_level == RiskLevel.HIGH
    assert finding.category == "scenario_gap"
    assert finding.auditor_payload["gap_type"] == "missing_file"
    assert _FIXED_URL in finding.source_urls


# ---------------------------------------------------------------------------
# Test 2: low confidence result is suppressed
# ---------------------------------------------------------------------------


def test_low_confidence_result_suppressed():
    """A result with confidence=0.5 produces no finding (below 0.7 threshold)."""
    result = _make_result(
        coverage="gap",
        gaps=["missing Anthropic SDK guidance"],
        confidence=0.5,
    )
    findings = _classify_with_mock([result])

    assert findings == []


# ---------------------------------------------------------------------------
# Test 3: adequate coverage produces no finding
# ---------------------------------------------------------------------------


def test_adequate_coverage_produces_no_finding():
    """coverage='adequate' produces no finding."""
    result = _make_result(coverage="adequate", confidence=0.95)
    findings = _classify_with_mock([result])

    assert findings == []


# ---------------------------------------------------------------------------
# Test 4: two personas with same gap are deduplicated
# ---------------------------------------------------------------------------


def test_two_personas_same_gap_deduplicated():
    """Two CoverageResults with the same file and gap type produce one deduplicated finding."""
    result_a = _make_result(
        persona_id="persona-a",
        file_path="ai-anthropic-to-bedrock.md",
        coverage="missing_file",
        gaps=["File missing"],
        confidence=1.0,
    )
    result_b = _make_result(
        persona_id="persona-b",
        file_path="ai-anthropic-to-bedrock.md",
        coverage="missing_file",
        gaps=["File missing"],
        confidence=1.0,
    )

    findings = _classify_with_mock([result_a, result_b])

    assert len(findings) == 1
    persona_ids = findings[0].auditor_payload["persona_ids"]
    assert "persona-a" in persona_ids
    assert "persona-b" in persona_ids


# ---------------------------------------------------------------------------
# Test 5: AI gap produces HIGH severity
# ---------------------------------------------------------------------------


def test_ai_gap_produces_high_severity():
    """A gap result with AI-related gap text produces HIGH severity."""
    result = _make_result(
        coverage="gap",
        gaps=["no guidance for Anthropic SDK client initialization", "missing Bedrock model ID mapping"],
        confidence=0.85,
    )
    findings = _classify_with_mock([result])

    assert len(findings) == 1
    assert findings[0].risk_level == RiskLevel.HIGH
    assert findings[0].auditor_payload["gap_type"] == "coverage_gap_ai"


# ---------------------------------------------------------------------------
# Test 6: framework gap produces MEDIUM severity
# ---------------------------------------------------------------------------


def test_framework_gap_produces_medium_severity():
    """A gap result with framework-related gap text produces MEDIUM severity."""
    result = _make_result(
        coverage="gap",
        gaps=["no guidance for LangChain agent memory configuration"],
        confidence=0.80,
    )
    findings = _classify_with_mock([result])

    assert len(findings) == 1
    assert findings[0].risk_level == RiskLevel.MEDIUM
    assert findings[0].auditor_payload["gap_type"] == "coverage_gap_framework"


# ---------------------------------------------------------------------------
# Test 7: intentional_deferral produces LOW severity
# ---------------------------------------------------------------------------


def test_intentional_deferral_produces_low_severity():
    """coverage='intentional_deferral' produces a LOW severity finding."""
    result = _make_result(
        coverage="intentional_deferral",
        gaps=["contact your AWS account team for this migration path"],
        confidence=0.90,
    )
    findings = _classify_with_mock([result])

    assert len(findings) == 1
    assert findings[0].risk_level == RiskLevel.LOW
    assert findings[0].auditor_payload["gap_type"] == "intentional_deferral"


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


def test_unverified_coverage_produces_no_finding():
    """coverage='unverified' produces no finding."""
    result = _make_result(coverage="unverified", confidence=0.0)
    findings = _classify_with_mock([result])

    assert findings == []


def test_finding_has_correct_schema_version():
    """Findings produced by GapClassifier use the scenario_gap/1.0 schema version."""
    result = _make_result(
        coverage="missing_file",
        confidence=1.0,
    )
    findings = _classify_with_mock([result])

    assert len(findings) == 1
    assert findings[0].finding_schema_version == "scenario_gap/1.0"


def test_finding_id_is_deterministic():
    """The same gap_type + file_path always produces the same finding_id."""
    result = _make_result(
        coverage="missing_file",
        file_path="ai-anthropic-to-bedrock.md",
        confidence=1.0,
    )
    findings_a = _classify_with_mock([result])
    findings_b = _classify_with_mock([result])

    assert findings_a[0].finding_id == findings_b[0].finding_id


def test_confidence_exactly_at_threshold_is_not_suppressed():
    """A result with confidence exactly equal to the threshold (0.7) is NOT suppressed."""
    result = _make_result(
        coverage="gap",
        gaps=["missing Anthropic SDK guidance"],
        confidence=0.7,
    )
    findings = _classify_with_mock([result])

    # confidence == threshold should pass (not suppressed)
    assert len(findings) == 1


def test_no_source_url_suppresses_finding():
    """When _fetch_source_url returns None, the finding is suppressed."""
    result = _make_result(
        coverage="missing_file",
        confidence=1.0,
    )
    classifier = GapClassifier(run_id="test-run")
    with patch.object(GapClassifier, "_fetch_source_url", return_value=None):
        findings = classifier.classify([result], [])

    assert findings == []


def test_minor_gap_produces_low_severity():
    """A gap with no AI or framework keywords produces LOW severity (coverage_gap_minor)."""
    result = _make_result(
        coverage="gap",
        gaps=["missing cost optimization guidance"],
        confidence=0.75,
    )
    findings = _classify_with_mock([result])

    assert len(findings) == 1
    assert findings[0].risk_level == RiskLevel.LOW
    assert findings[0].auditor_payload["gap_type"] == "coverage_gap_minor"
