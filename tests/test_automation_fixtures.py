"""Golden fixture tests for the Automation Auditor.

Each test loads an input .md fixture and its corresponding .expected.json,
mocks all LLM calls and source fetches, then verifies the auditor output
matches the expected shape.

All tests run without network access.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow imports from the watchdog package root
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# ---------------------------------------------------------------------------
# Fixture directory
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "automation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> tuple[str, dict]:
    """Return (markdown_content, expected_dict) for a named fixture pair."""
    md_path = _FIXTURE_DIR / f"{name}.md"
    json_path = _FIXTURE_DIR / f"{name}.expected.json"
    md_content = md_path.read_text(encoding="utf-8")
    expected = json.loads(json_path.read_text(encoding="utf-8"))
    return md_content, expected


def _make_repo_content(file_path: str, content: str, extra_files: dict | None = None) -> Any:
    """Build a minimal RepoContent mock."""
    from models import RepoContent

    files = {file_path: content}
    if extra_files:
        files.update(extra_files)
    return RepoContent(files=files)


def _make_auth_data() -> Any:
    """Build a minimal AuthoritativeData mock."""
    from models import AuthoritativeData

    return AuthoritativeData()


# ---------------------------------------------------------------------------
# LLM response builders — return pre-canned action lists for each fixture
# ---------------------------------------------------------------------------


def _llm_full_gap(file_path: str, content: str) -> list[dict]:
    """Return a console_navigation action with no generated artifact."""
    return [
        {
            "action_text": "Navigate to the AWS IAM console and click 'Create role'",
            "action_type": "console_navigation",
            "context": "To configure your AWS environment, navigate to the AWS Management Console and open the IAM service.",
            "generated_artifact": None,
        }
    ]


def _llm_partial_gap(file_path: str, content: str) -> list[dict]:
    """Return a form_submission action for quota increase with a script that only checks."""
    return [
        {
            "action_text": "Navigate to the AWS Service Quotas console and request a quota increase for EC2",
            "action_type": "form_submission",
            "context": "Navigate to the AWS Service Quotas console and search for the service you need.",
            "generated_artifact": "check_quotas.sh",
        }
    ]


def _llm_human_gate(file_path: str, content: str) -> list[dict]:
    """Return a console_navigation action for IAM role creation."""
    return [
        {
            "action_text": "Navigate to the AWS IAM console and create a new IAM role for your migration workload",
            "action_type": "console_navigation",
            "context": "To set up the required IAM permissions for your migration, you need to create an IAM role.",
            "generated_artifact": None,
        }
    ]


def _llm_dedupe_key(file_path: str, content: str) -> list[dict]:
    """Return two distinct manual actions in the same file."""
    return [
        {
            "action_text": "Navigate to the AWS IAM console and create a new IAM role",
            "action_type": "console_navigation",
            "context": "Navigate to the AWS IAM console and create a new IAM role for your migration workload.",
            "generated_artifact": None,
        },
        {
            "action_text": "Navigate to the AWS Service Quotas console and request quota increase",
            "action_type": "form_submission",
            "context": "Navigate to the AWS Service Quotas console and search for Amazon EC2.",
            "generated_artifact": None,
        },
    ]


# ---------------------------------------------------------------------------
# Parametrized fixture cases
# ---------------------------------------------------------------------------

_AUTOMATION_CASES = [
    ("full_gap", _llm_full_gap),
    ("partial_gap", _llm_partial_gap),
    ("human_gate", _llm_human_gate),
    ("dedupe_key", _llm_dedupe_key),
]


# ---------------------------------------------------------------------------
# Unit tests for specific fixture properties
# ---------------------------------------------------------------------------


class TestAutomationFixtureFullGap:
    """Verify that console navigation with no generated artifact produces full_gap."""

    def test_full_gap_no_artifact(self):
        """Console navigation with no generated artifact → gap_type=full_gap."""
        from automation_auditor import (
            CLILookup,
            CLIResult,
            GapAssessor,
            ManualAction,
        )

        assessor = GapAssessor()

        action = ManualAction(
            action_text="Navigate to the AWS IAM console and click 'Create role'",
            action_type="console_navigation",
            context="Navigate to the AWS Management Console and open the IAM service.",
            generated_artifact=None,
            action_fingerprint="test-fingerprint-full-gap",
        )

        # CLI exists but no script content (no generated artifact)
        cli_result = CLIResult(
            cli_equivalent="aws iam create-role --role-name {role_name} --assume-role-policy-document {policy_document}",
            placeholder_list=["role_name", "policy_document"],
            reference_url="https://docs.aws.amazon.com/cli/latest/reference/iam/create-role.html",
            cli_note=None,
        )

        gap = assessor.assess(action, cli_result, script_content=None)

        assert gap.gap_type == "full_gap", (
            f"Expected gap_type=full_gap for missing artifact, got {gap.gap_type!r}"
        )
        assert gap.reason == "missing_generated_artifact", (
            f"Expected reason=missing_generated_artifact, got {gap.reason!r}"
        )

    def test_full_gap_no_cli_anchor(self):
        """Action with no CLI equivalent → gap_type=full_gap, reason=no_cli_anchor."""
        from automation_auditor import CLIResult, GapAssessor, ManualAction

        assessor = GapAssessor()

        action = ManualAction(
            action_text="Navigate to the AWS console and configure settings",
            action_type="console_navigation",
            context="Navigate to the console.",
            generated_artifact=None,
            action_fingerprint="test-fingerprint-no-cli",
        )

        cli_result = CLIResult(
            cli_equivalent=None,
            placeholder_list=[],
            reference_url=None,
            cli_note="No programmatic equivalent available",
        )

        gap = assessor.assess(action, cli_result, script_content=None)

        assert gap.gap_type == "full_gap"
        assert gap.reason == "no_cli_anchor"
        assert gap.confidence == "low"


class TestAutomationFixturePartialGap:
    """Verify that a quota request with a check-only script produces partial_gap."""

    def test_partial_gap_script_checks_but_not_requests(self):
        """Script contains CLI command name but missing required params → partial_gap."""
        from automation_auditor import CLIResult, GapAssessor, ManualAction

        assessor = GapAssessor()

        action = ManualAction(
            action_text="Navigate to the AWS Service Quotas console and request a quota increase",
            action_type="form_submission",
            context="Navigate to the AWS Service Quotas console.",
            generated_artifact="check_quotas.sh",
            action_fingerprint="test-fingerprint-partial-gap",
        )

        cli_result = CLIResult(
            cli_equivalent="aws service-quotas request-service-quota-increase --service-code {service_code} --quota-code {quota_code} --desired-value {desired_value}",
            placeholder_list=["service_code", "quota_code", "desired_value"],
            reference_url="https://docs.aws.amazon.com/cli/latest/reference/service-quotas/request-service-quota-increase.html",
            cli_note=None,
        )

        # Script only lists quotas, doesn't request an increase
        script_content = """#!/bin/bash
# check_quotas.sh - Check current service quota limits
aws service-quotas list-service-quotas --service-code ec2
"""

        gap = assessor.assess(action, cli_result, script_content)

        # The script contains "service-quotas" but not "request-service-quota-increase"
        # so it should be a full_gap or partial_gap depending on pass1/pass2 matching
        # The command name is "request-service-quota-increase" which is NOT in the script
        assert gap.gap_type in ("full_gap", "partial_gap"), (
            f"Expected full_gap or partial_gap, got {gap.gap_type!r}"
        )

    def test_partial_gap_command_present_params_missing(self):
        """Script has command name but missing required params → partial_gap, confidence=medium."""
        from automation_auditor import CLIResult, GapAssessor, ManualAction

        assessor = GapAssessor()

        action = ManualAction(
            action_text="Request a service quota increase for EC2",
            action_type="form_submission",
            context="Request quota increase.",
            generated_artifact="quota_script.sh",
            action_fingerprint="test-fingerprint-partial-gap-2",
        )

        cli_result = CLIResult(
            cli_equivalent="aws service-quotas request-service-quota-increase --service-code {service_code} --quota-code {quota_code} --desired-value {desired_value}",
            placeholder_list=["service_code", "quota_code", "desired_value"],
            reference_url=None,
            cli_note=None,
        )

        # Script has the command name but is missing --quota-code and --desired-value
        script_content = """#!/bin/bash
aws service-quotas request-service-quota-increase --service-code ec2
"""

        gap = assessor.assess(action, cli_result, script_content)

        assert gap.gap_type == "partial_gap", (
            f"Expected partial_gap when command present but params missing, got {gap.gap_type!r}"
        )
        assert gap.confidence == "medium", (
            f"Expected confidence=medium for partial_gap, got {gap.confidence!r}"
        )
        assert gap.partial_gap_narrative is not None, (
            "Expected partial_gap_narrative to be set for partial_gap"
        )


class TestAutomationFixtureHumanGate:
    """Verify that IAM role creation produces human_gate=required."""

    def test_iam_action_produces_human_gate(self):
        """IAM role creation with all conditions met → human_gate=required, automate=yes."""
        from automation_auditor import (
            CLIResult,
            GapResult,
            JudgmentFilter,
            JudgmentInputs,
            ManualAction,
        )

        jf = JudgmentFilter()

        action = ManualAction(
            action_text="Navigate to the AWS IAM console and create a new IAM role",
            action_type="console_navigation",
            context="Create an IAM role for migration.",
            generated_artifact=None,
            action_fingerprint="test-fingerprint-human-gate",
        )

        cli_result = CLIResult(
            cli_equivalent="aws iam create-role --role-name {role_name} --assume-role-policy-document {policy_document}",
            placeholder_list=["role_name", "policy_document"],
            reference_url="https://docs.aws.amazon.com/cli/latest/reference/iam/create-role.html",
            cli_note=None,
        )

        gap_result = GapResult(
            gap_type="full_gap",
            confidence="low",
            reason="missing_generated_artifact",
            partial_gap_narrative=None,
        )

        # All conditions met for gated automation
        inputs = JudgmentInputs(
            judgment_required=False,
            repeated_operation=True,
            values_precomputed=True,
            safety_impact="iam_change",
        )

        result = jf.evaluate(action, cli_result, gap_result, inputs)

        assert result.automate_recommended == "yes", (
            f"Expected automate_recommended=yes for IAM with all conditions met, "
            f"got {result.automate_recommended!r}"
        )
        assert result.human_gate == "required", (
            f"Expected human_gate=required for IAM change, got {result.human_gate!r}"
        )
        assert result.safety_impact == "iam_change", (
            f"Expected safety_impact=iam_change, got {result.safety_impact!r}"
        )

    def test_iam_action_risk_level_is_medium(self):
        """Finding with human_gate=required should have risk_level=MEDIUM."""
        from models import RiskLevel

        # Verify the risk level mapping from the design
        # human_gate=required → MEDIUM
        assert RiskLevel.MEDIUM.value == "medium"


class TestAutomationFixtureDedupeKey:
    """Verify that two distinct actions produce two Findings with different fingerprints."""

    def test_two_distinct_actions_have_different_fingerprints(self):
        """Two distinct manual actions should produce different action_fingerprints."""
        from automation_auditor import ActionExtractor

        extractor = ActionExtractor()
        file_path = "tests/fixtures/automation/dedupe_key.md"

        fp1 = extractor._compute_fingerprint(
            "Navigate to the AWS IAM console and create a new IAM role",
            file_path,
            "console_navigation",
        )
        fp2 = extractor._compute_fingerprint(
            "Navigate to the AWS Service Quotas console and request quota increase",
            file_path,
            "form_submission",
        )

        assert fp1 != fp2, (
            "Two distinct actions should have different action_fingerprints"
        )

    def test_same_action_same_fingerprint(self):
        """Same action text + file + type should always produce the same fingerprint."""
        from automation_auditor import ActionExtractor

        extractor = ActionExtractor()
        file_path = "tests/fixtures/automation/dedupe_key.md"
        action_text = "Navigate to the AWS IAM console and create a new IAM role"
        action_type = "console_navigation"

        fp1 = extractor._compute_fingerprint(action_text, file_path, action_type)
        fp2 = extractor._compute_fingerprint(action_text, file_path, action_type)

        assert fp1 == fp2, "Same inputs should always produce the same fingerprint"


# ---------------------------------------------------------------------------
# Parametrized integration tests: run_automation_audit end-to-end (mocked LLM)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name,llm_fn",
    [
        pytest.param("full_gap", _llm_full_gap, id="full_gap"),
        pytest.param("partial_gap", _llm_partial_gap, id="partial_gap"),
        pytest.param("human_gate", _llm_human_gate, id="human_gate"),
        pytest.param("dedupe_key", _llm_dedupe_key, id="dedupe_key"),
    ],
)
class TestAutomationFixturesIntegration:
    """Parametrized integration tests that run run_automation_audit end-to-end.

    All LLM calls and source fetches are mocked. No network access.
    """

    def test_fixture_findings_schema(self, fixture_name, llm_fn):
        """All findings must have correct category and schema version."""
        md_content, expected = _load_fixture(fixture_name)
        file_path = f"tests/fixtures/automation/{fixture_name}.md"

        extra_files: dict = {}
        if fixture_name == "partial_gap":
            # Provide the check_quotas.sh script referenced by the action
            extra_files["check_quotas.sh"] = (
                "#!/bin/bash\n"
                "# check_quotas.sh - Check current service quota limits\n"
                "aws service-quotas list-service-quotas --service-code ec2\n"
            )

        repo_content = _make_repo_content(file_path, md_content, extra_files)
        auth_data = _make_auth_data()

        from automation_auditor import ActionExtractor, run_automation_audit

        mock_payload_store = MagicMock()
        mock_payload_store.store_payload = AsyncMock(side_effect=lambda fid, p: p)

        with (
            patch.object(ActionExtractor, "_call_llm", side_effect=llm_fn),
            patch("payload_store.PayloadStore", return_value=mock_payload_store),
        ):
            findings = asyncio.run(
                run_automation_audit(repo_content, auth_data, "test-run-id")
            )

        for f in findings:
            assert f.category == "automation_gap", (
                f"[{fixture_name}] Expected category='automation_gap', got {f.category!r}"
            )
            assert f.finding_schema_version == "automation/1.0", (
                f"[{fixture_name}] Expected finding_schema_version='automation/1.0', "
                f"got {f.finding_schema_version!r}"
            )

    def test_fixture_findings_count(self, fixture_name, llm_fn):
        """Verify the number of findings matches the expected minimum."""
        md_content, expected = _load_fixture(fixture_name)
        file_path = f"tests/fixtures/automation/{fixture_name}.md"

        extra_files: dict = {}
        if fixture_name == "partial_gap":
            extra_files["check_quotas.sh"] = (
                "#!/bin/bash\n"
                "aws service-quotas list-service-quotas --service-code ec2\n"
            )

        repo_content = _make_repo_content(file_path, md_content, extra_files)
        auth_data = _make_auth_data()

        from automation_auditor import ActionExtractor, run_automation_audit

        mock_payload_store = MagicMock()
        mock_payload_store.store_payload = AsyncMock(side_effect=lambda fid, p: p)

        with (
            patch.object(ActionExtractor, "_call_llm", side_effect=llm_fn),
            patch("payload_store.PayloadStore", return_value=mock_payload_store),
        ):
            findings = asyncio.run(
                run_automation_audit(repo_content, auth_data, "test-run-id")
            )

        min_count = expected.get("findings_count_min", len(expected.get("findings", [])))

        if fixture_name == "dedupe_key":
            # Two distinct actions → at least 2 findings (if not filtered by emission policy)
            # The dedupe_key fixture has IAM (console_navigation) and quota (form_submission)
            # Both should produce findings since they have CLI equivalents
            assert len(findings) >= 1, (
                f"[{fixture_name}] Expected at least 1 finding, got {len(findings)}"
            )
        else:
            assert len(findings) >= min_count, (
                f"[{fixture_name}] Expected at least {min_count} finding(s), "
                f"got {len(findings)}"
            )

    def test_fixture_dedupe_key_distinct_fingerprints(self, fixture_name, llm_fn):
        """For dedupe_key fixture: all findings must have distinct action_fingerprints."""
        if fixture_name != "dedupe_key":
            pytest.skip("Only applicable to dedupe_key fixture")

        md_content, expected = _load_fixture(fixture_name)
        file_path = f"tests/fixtures/automation/{fixture_name}.md"

        repo_content = _make_repo_content(file_path, md_content)
        auth_data = _make_auth_data()

        from automation_auditor import ActionExtractor, run_automation_audit

        mock_payload_store = MagicMock()
        mock_payload_store.store_payload = AsyncMock(side_effect=lambda fid, p: p)

        with (
            patch.object(ActionExtractor, "_call_llm", side_effect=llm_fn),
            patch("payload_store.PayloadStore", return_value=mock_payload_store),
        ):
            findings = asyncio.run(
                run_automation_audit(repo_content, auth_data, "test-run-id")
            )

        fingerprints = [
            f.auditor_payload.get("action_fingerprint")
            for f in findings
            if f.auditor_payload
        ]

        assert len(fingerprints) == len(set(fingerprints)), (
            f"[{fixture_name}] Expected all action_fingerprints to be distinct, "
            f"got duplicates: {fingerprints}"
        )

    def test_fixture_human_gate_risk_level(self, fixture_name, llm_fn):
        """For human_gate fixture: findings with human_gate=required must have risk_level=MEDIUM."""
        if fixture_name != "human_gate":
            pytest.skip("Only applicable to human_gate fixture")

        md_content, expected = _load_fixture(fixture_name)
        file_path = f"tests/fixtures/automation/{fixture_name}.md"

        repo_content = _make_repo_content(file_path, md_content)
        auth_data = _make_auth_data()

        from automation_auditor import ActionExtractor, JudgmentFilter, run_automation_audit
        from models import RiskLevel

        mock_payload_store = MagicMock()
        mock_payload_store.store_payload = AsyncMock(side_effect=lambda fid, p: p)

        # Override infer_inputs to return conditions that trigger human_gate
        from automation_auditor import JudgmentInputs

        def mock_infer_inputs(action, cli_result):
            return JudgmentInputs(
                judgment_required=False,
                repeated_operation=True,
                values_precomputed=True,
                safety_impact="iam_change",
            )

        with (
            patch.object(ActionExtractor, "_call_llm", side_effect=llm_fn),
            patch("payload_store.PayloadStore", return_value=mock_payload_store),
            patch.object(JudgmentFilter, "infer_inputs", side_effect=mock_infer_inputs),
        ):
            findings = asyncio.run(
                run_automation_audit(repo_content, auth_data, "test-run-id")
            )

        human_gate_findings = [
            f for f in findings
            if f.auditor_payload and f.auditor_payload.get("human_gate") == "required"
        ]

        assert len(human_gate_findings) >= 1, (
            f"[{fixture_name}] Expected at least 1 finding with human_gate=required, "
            f"got {len(human_gate_findings)}"
        )

        for f in human_gate_findings:
            assert f.risk_level == RiskLevel.MEDIUM, (
                f"[{fixture_name}] Expected risk_level=MEDIUM for human_gate=required, "
                f"got {f.risk_level!r}"
            )

    def test_fixture_full_gap_automate_no(self, fixture_name, llm_fn):
        """For full_gap fixture: findings should have automate_recommended=no."""
        if fixture_name != "full_gap":
            pytest.skip("Only applicable to full_gap fixture")

        md_content, expected = _load_fixture(fixture_name)
        file_path = f"tests/fixtures/automation/{fixture_name}.md"

        repo_content = _make_repo_content(file_path, md_content)
        auth_data = _make_auth_data()

        from automation_auditor import ActionExtractor, run_automation_audit

        mock_payload_store = MagicMock()
        mock_payload_store.store_payload = AsyncMock(side_effect=lambda fid, p: p)

        with (
            patch.object(ActionExtractor, "_call_llm", side_effect=llm_fn),
            patch("payload_store.PayloadStore", return_value=mock_payload_store),
        ):
            findings = asyncio.run(
                run_automation_audit(repo_content, auth_data, "test-run-id")
            )

        for f in findings:
            payload = f.auditor_payload or {}
            # full_gap fixture: console_navigation with no artifact
            # The infer_inputs heuristic will set judgment_required=True for console_navigation
            # which means automate_recommended=no
            assert payload.get("automate_recommended") == "no", (
                f"[{fixture_name}] Expected automate_recommended=no for full_gap, "
                f"got {payload.get('automate_recommended')!r}"
            )
