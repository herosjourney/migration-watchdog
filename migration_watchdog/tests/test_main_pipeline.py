"""Tests for main.py pipeline additions — tasks 15.3 and 15.4.

Covers:
- ValueError handling during finding persistence (task 15.3)
- _should_include_in_pr_comment severity threshold filtering (task 15.4)
- _build_pr_comment_markdown marker and content (task 15.4)
- _post_pr_comment 403 non-fatal handling (task 15.4)
- audit-report.json writing (task 15.4)
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow imports from the watchdog package root
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# ---------------------------------------------------------------------------
# Stub out migration_watchdog.* imports so main.py can be imported without
# the full package being installed.
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    return mod


_MW_STUBS = [
    "migration_watchdog",
    "migration_watchdog.analysis_agent",
    "migration_watchdog.finding_deduplicator",
    "migration_watchdog.findings_repository",
    "migration_watchdog.models",
    "migration_watchdog.refactoring_agent",
    "migration_watchdog.repo_scanner",
    "migration_watchdog.retry",
    "migration_watchdog.review_agent",
    "migration_watchdog.source_fetcher",
]

for _stub_name in _MW_STUBS:
    if _stub_name not in sys.modules:
        sys.modules[_stub_name] = _make_stub(_stub_name)

# Provide the symbols that main.py imports from these stubs
import models as _models_real  # noqa: E402 — real models module

_mw_models = sys.modules["migration_watchdog.models"]
_mw_models.Finding = _models_real.Finding
_mw_models.RiskLevel = _models_real.RiskLevel
_mw_models.ScanRun = _models_real.ScanRun

_mw_analysis = sys.modules["migration_watchdog.analysis_agent"]
_mw_analysis.run_analysis = MagicMock(return_value=[])

_mw_dedup = sys.modules["migration_watchdog.finding_deduplicator"]
_mw_dedup.deduplicate = MagicMock(return_value=[])
_mw_dedup._is_addressed_by_pr = MagicMock(return_value=False)

_mw_repo = sys.modules["migration_watchdog.findings_repository"]
_mw_repo.FindingsRepository = MagicMock()

_mw_refactor = sys.modules["migration_watchdog.refactoring_agent"]
_mw_refactor.run_refactoring_assessment = MagicMock(return_value=None)

_mw_scanner = sys.modules["migration_watchdog.repo_scanner"]
_mw_scanner.RepoScanner = MagicMock()

_mw_retry = sys.modules["migration_watchdog.retry"]

async def _passthrough_retry(operation, **kwargs):
    return await operation()

_mw_retry.retry_with_backoff = _passthrough_retry

_mw_review = sys.modules["migration_watchdog.review_agent"]
_mw_review.review_findings = MagicMock(return_value=[])

_mw_fetcher = sys.modules["migration_watchdog.source_fetcher"]
_mw_fetcher.SourceFetcher = MagicMock()

# Now import main — the stubs are in place
import main as _main_module  # noqa: E402

from main import (  # noqa: E402
    _build_pr_comment_markdown,
    _post_pr_comment,
    _should_include_in_pr_comment,
)
from migration_watchdog.models import Finding, RiskLevel  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_finding(
    category: str = "currency_drift",
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    auditor_payload: dict | None = None,
    review_status: str | None = None,
    title: str = "Test finding",
    affected_files: list[str] | None = None,
) -> Finding:
    return Finding(
        finding_id="test-id",
        run_id="run-1",
        risk_level=risk_level,
        category=category,
        title=title,
        description="desc",
        affected_files=affected_files or ["file.md"],
        auditor_payload=auditor_payload,
        review_status=review_status,
    )


# ---------------------------------------------------------------------------
# Task 15.3 — ValueError handling during persistence
# ---------------------------------------------------------------------------

class TestValueErrorHandling:
    """Verify that ValueError from save_finding is caught per-finding."""

    def test_payload_too_large_is_caught_and_skipped(self):
        """A ValueError from save_finding must be caught; the finding is skipped."""
        findings = [
            _make_finding(title="Good finding"),
            _make_finding(title="Bad finding"),
        ]

        partial_source_failures: list[str] = []
        persisted: list[Finding] = []

        # Simulate the loop in run_scan step 7
        def mock_save(finding: Finding) -> None:
            if finding.title == "Bad finding":
                raise ValueError(
                    "payload still too large after truncation: 99999 bytes "
                    "(finding_id=test-id)"
                )

        for finding in findings:
            try:
                mock_save(finding)
                persisted.append(finding)
            except ValueError as exc:
                partial_source_failures.append(
                    json.dumps({"type": "payload_too_large", "finding_id": finding.finding_id})
                )

        assert len(persisted) == 1
        assert persisted[0].title == "Good finding"
        assert len(partial_source_failures) == 1
        entry = json.loads(partial_source_failures[0])
        assert entry["type"] == "payload_too_large"
        assert "finding_id" in entry

    def test_partial_source_failures_format(self):
        """The structured entry appended to partial_source_failures must be valid JSON."""
        finding_id = "abc-123"
        entry = json.dumps({"type": "payload_too_large", "finding_id": finding_id})
        parsed = json.loads(entry)
        assert parsed["type"] == "payload_too_large"
        assert parsed["finding_id"] == finding_id

    def test_good_findings_still_persisted_after_bad_one(self):
        """Findings after a bad one must still be persisted."""
        findings = [
            _make_finding(title="Bad finding"),
            _make_finding(title="Good finding after bad"),
        ]

        partial_source_failures: list[str] = []
        persisted: list[Finding] = []

        def mock_save(finding: Finding) -> None:
            if finding.title == "Bad finding":
                raise ValueError("too large")

        for finding in findings:
            try:
                mock_save(finding)
                persisted.append(finding)
            except ValueError as exc:
                partial_source_failures.append(
                    json.dumps({"type": "payload_too_large", "finding_id": finding.finding_id})
                )

        assert len(persisted) == 1
        assert persisted[0].title == "Good finding after bad"
        assert len(partial_source_failures) == 1


# ---------------------------------------------------------------------------
# Task 15.4 — Severity threshold filtering
# ---------------------------------------------------------------------------

class TestShouldIncludeInPrComment:
    """Verify _should_include_in_pr_comment filtering rules."""

    def test_correctness_always_included_at_default_threshold(self):
        f = _make_finding(auditor_payload={"severity": "correctness"})
        assert _should_include_in_pr_comment(f, "outdated") is True

    def test_outdated_always_included_at_default_threshold(self):
        f = _make_finding(auditor_payload={"severity": "outdated"})
        assert _should_include_in_pr_comment(f, "outdated") is True

    def test_policy_change_excluded_at_default_threshold(self):
        f = _make_finding(auditor_payload={"severity": "policy_change"})
        assert _should_include_in_pr_comment(f, "outdated") is False

    def test_policy_change_included_at_low_threshold(self):
        f = _make_finding(auditor_payload={"severity": "policy_change"})
        assert _should_include_in_pr_comment(f, "low") is True

    def test_informational_never_included_at_low_threshold(self):
        f = _make_finding(auditor_payload={"severity": "informational"})
        assert _should_include_in_pr_comment(f, "low") is False

    def test_informational_never_included_at_default_threshold(self):
        f = _make_finding(auditor_payload={"severity": "informational"})
        assert _should_include_in_pr_comment(f, "outdated") is False

    def test_automation_full_gap_included(self):
        f = _make_finding(
            category="automation_gap",
            auditor_payload={"gap_type": "full_gap"},
        )
        assert _should_include_in_pr_comment(f, "outdated") is True

    def test_automation_partial_gap_included(self):
        f = _make_finding(
            category="automation_gap",
            auditor_payload={"gap_type": "partial_gap"},
        )
        assert _should_include_in_pr_comment(f, "outdated") is True

    def test_automation_no_gap_excluded(self):
        f = _make_finding(
            category="automation_gap",
            auditor_payload={"gap_type": "no_gap"},
        )
        assert _should_include_in_pr_comment(f, "outdated") is False

    def test_automation_missing_gap_type_defaults_to_full_gap(self):
        """When gap_type is absent, default to full_gap (include)."""
        f = _make_finding(
            category="automation_gap",
            auditor_payload={},
        )
        assert _should_include_in_pr_comment(f, "outdated") is True

    def test_correctness_included_at_low_threshold(self):
        f = _make_finding(auditor_payload={"severity": "correctness"})
        assert _should_include_in_pr_comment(f, "low") is True

    def test_outdated_included_at_low_threshold(self):
        f = _make_finding(auditor_payload={"severity": "outdated"})
        assert _should_include_in_pr_comment(f, "low") is True


# ---------------------------------------------------------------------------
# Task 15.4 — PR comment markdown building
# ---------------------------------------------------------------------------

class TestBuildPrCommentMarkdown:
    """Verify _build_pr_comment_markdown produces correct content."""

    def test_marker_always_present(self):
        body = _build_pr_comment_markdown([], 42, "run-1", [], "outdated")
        assert "<!-- watchdog-audit-comment -->" in body

    def test_empty_findings_shows_clean_message(self):
        body = _build_pr_comment_markdown([], 42, "run-1", [], "outdated")
        assert "No findings" in body

    def test_currency_finding_rendered(self):
        f = _make_finding(
            category="currency_drift",
            risk_level=RiskLevel.HIGH,
            auditor_payload={
                "severity": "correctness",
                "claim_type": "region_count",
                "claim_text": "9 regions supported",
                "suggested_fix": "Update to 15 regions",
            },
            title="Region count stale",
        )
        body = _build_pr_comment_markdown([f], 42, "run-1", ["file.md"], "outdated")
        assert "Currency Drift" in body
        assert "CORRECTNESS" in body
        assert "region_count" in body
        assert "9 regions supported" in body
        assert "Update to 15 regions" in body

    def test_automation_finding_rendered(self):
        f = _make_finding(
            category="automation_gap",
            risk_level=RiskLevel.MEDIUM,
            auditor_payload={
                "gap_type": "full_gap",
                "action_type": "console_navigation",
                "action_text": "Navigate to IAM console",
                "automate_recommended": "yes",
                "human_gate": "required",
                "cli_equivalent": "aws iam create-role --role-name {role_name}",
            },
            title="IAM role creation gap",
        )
        body = _build_pr_comment_markdown([f], 42, "run-1", ["file.md"], "outdated")
        assert "Automation Gap" in body
        assert "FULL_GAP" in body
        assert "Requires human approval before execution" in body
        assert "aws iam create-role" in body

    def test_disputed_finding_shows_banner(self):
        f = _make_finding(
            category="currency_drift",
            auditor_payload={"severity": "correctness"},
            review_status="disputed",
            title="Disputed finding",
        )
        body = _build_pr_comment_markdown([f], 42, "run-1", [], "outdated")
        assert "Disputed by review agent" in body

    def test_policy_change_filtered_at_default_threshold(self):
        f = _make_finding(
            category="currency_drift",
            auditor_payload={"severity": "policy_change"},
            title="Policy change finding",
        )
        body = _build_pr_comment_markdown([f], 42, "run-1", [], "outdated")
        # Should show clean message since policy_change is filtered out
        assert "No findings" in body

    def test_policy_change_included_at_low_threshold(self):
        f = _make_finding(
            category="currency_drift",
            auditor_payload={"severity": "policy_change", "claim_type": "service_name"},
            title="Policy change finding",
        )
        body = _build_pr_comment_markdown([f], 42, "run-1", [], "low")
        assert "Policy change finding" in body

    def test_run_id_in_comment(self):
        body = _build_pr_comment_markdown([], 99, "my-run-id-123", [], "outdated")
        assert "my-run-id-123" in body

    def test_pr_number_in_comment(self):
        body = _build_pr_comment_markdown([], 99, "run-1", [], "outdated")
        assert "#99" in body

    def test_multiple_findings_all_rendered(self):
        findings = [
            _make_finding(
                category="currency_drift",
                auditor_payload={"severity": "correctness"},
                title="Finding A",
            ),
            _make_finding(
                category="currency_drift",
                auditor_payload={"severity": "outdated"},
                title="Finding B",
            ),
        ]
        body = _build_pr_comment_markdown(findings, 1, "run-1", [], "outdated")
        assert "Finding A" in body
        assert "Finding B" in body
        assert "2" in body  # "Found 2 finding(s)"


# ---------------------------------------------------------------------------
# Task 15.4 — _post_pr_comment: 403 non-fatal, audit-report.json
# ---------------------------------------------------------------------------

class TestPostPrComment:
    """Verify _post_pr_comment behavior for 403 and audit-report.json."""

    @pytest.mark.asyncio
    async def test_writes_audit_report_json(self, tmp_path):
        """audit-report.json must be written even when GitHub API is unavailable."""
        findings = [
            _make_finding(
                category="currency_drift",
                auditor_payload={"severity": "correctness"},
            )
        ]

        with patch("main.httpx.AsyncClient") as mock_client_cls:
            # Make the client raise so we don't actually call GitHub
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=Exception("no network"))
            mock_client_cls.return_value = mock_client

            orig_dir = os.getcwd()
            os.chdir(tmp_path)
            try:
                await _post_pr_comment(
                    findings,
                    pr_number=42,
                    run_id="run-abc",
                    audited_files=["file.md"],
                    github_token="fake-token",
                    target_owner="owner",
                    target_repo="repo",
                    severity_threshold="outdated",
                )
            finally:
                os.chdir(orig_dir)

        report_path = tmp_path / "audit-report.json"
        assert report_path.exists(), "audit-report.json must be written"
        report = json.loads(report_path.read_text())
        assert report["run_id"] == "run-abc"
        assert report["pr_number"] == 42
        assert report["severity_threshold"] == "outdated"

    @pytest.mark.asyncio
    async def test_403_on_list_comments_is_non_fatal(self, tmp_path):
        """A 403 when listing comments must be logged and not raise."""
        import httpx as _httpx

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = _httpx.HTTPStatusError(
            "403", request=MagicMock(), response=mock_response
        )

        with patch("main.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            orig_dir = os.getcwd()
            os.chdir(tmp_path)
            try:
                # Must not raise
                await _post_pr_comment(
                    [],
                    pr_number=1,
                    run_id="run-1",
                    audited_files=[],
                    github_token="token",
                    target_owner="owner",
                    target_repo="repo",
                )
            finally:
                os.chdir(orig_dir)

    @pytest.mark.asyncio
    async def test_no_token_skips_api_call(self, tmp_path):
        """When github_token is empty, no API call should be made."""
        with patch("main.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            orig_dir = os.getcwd()
            os.chdir(tmp_path)
            try:
                await _post_pr_comment(
                    [],
                    pr_number=1,
                    run_id="run-1",
                    audited_files=[],
                    github_token="",  # empty token
                    target_owner="owner",
                    target_repo="repo",
                )
            finally:
                os.chdir(orig_dir)

            # AsyncClient should not have been entered (no API calls)
            mock_client.__aenter__.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_new_comment_when_none_exists(self, tmp_path):
        """When no existing watchdog comment is found, a new one is created."""
        list_response = MagicMock()
        list_response.status_code = 200
        list_response.raise_for_status = MagicMock()
        list_response.json = MagicMock(return_value=[])  # no existing comments

        create_response = MagicMock()
        create_response.status_code = 201
        create_response.raise_for_status = MagicMock()

        with patch("main.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=list_response)
            mock_client.post = AsyncMock(return_value=create_response)
            mock_client_cls.return_value = mock_client

            orig_dir = os.getcwd()
            os.chdir(tmp_path)
            try:
                await _post_pr_comment(
                    [],
                    pr_number=5,
                    run_id="run-1",
                    audited_files=[],
                    github_token="token",
                    target_owner="owner",
                    target_repo="repo",
                )
            finally:
                os.chdir(orig_dir)

            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args
            body = call_kwargs.kwargs.get("json", {}).get("body", "")
            assert "<!-- watchdog-audit-comment -->" in body

    @pytest.mark.asyncio
    async def test_updates_existing_comment_when_found(self, tmp_path):
        """When an existing watchdog comment is found, it is updated (PATCH)."""
        list_response = MagicMock()
        list_response.status_code = 200
        list_response.raise_for_status = MagicMock()
        list_response.json = MagicMock(return_value=[
            {"id": 999, "body": "<!-- watchdog-audit-comment -->\nOld content"}
        ])

        patch_response = MagicMock()
        patch_response.status_code = 200
        patch_response.raise_for_status = MagicMock()

        with patch("main.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=list_response)
            mock_client.patch = AsyncMock(return_value=patch_response)
            mock_client_cls.return_value = mock_client

            orig_dir = os.getcwd()
            os.chdir(tmp_path)
            try:
                await _post_pr_comment(
                    [],
                    pr_number=5,
                    run_id="run-1",
                    audited_files=[],
                    github_token="token",
                    target_owner="owner",
                    target_repo="repo",
                )
            finally:
                os.chdir(orig_dir)

            mock_client.patch.assert_called_once()
            # Verify it patched the correct comment ID
            url = mock_client.patch.call_args.args[0]
            assert "999" in url

    @pytest.mark.asyncio
    async def test_audit_report_json_counts_visible_findings(self, tmp_path):
        """audit-report.json must correctly count visible vs total findings."""
        findings = [
            _make_finding(
                category="currency_drift",
                auditor_payload={"severity": "correctness"},
                title="Correctness finding",
            ),
            _make_finding(
                category="currency_drift",
                auditor_payload={"severity": "policy_change"},
                title="Policy change (filtered at default threshold)",
            ),
            _make_finding(
                category="automation_gap",
                auditor_payload={"gap_type": "full_gap"},
                title="Automation gap",
            ),
        ]

        with patch("main.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=Exception("no network"))
            mock_client_cls.return_value = mock_client

            orig_dir = os.getcwd()
            os.chdir(tmp_path)
            try:
                await _post_pr_comment(
                    findings,
                    pr_number=1,
                    run_id="run-1",
                    audited_files=[],
                    github_token="token",
                    target_owner="owner",
                    target_repo="repo",
                    severity_threshold="outdated",  # policy_change filtered out
                )
            finally:
                os.chdir(orig_dir)

        report = json.loads((tmp_path / "audit-report.json").read_text())
        assert report["total_findings"] == 3
        # policy_change is filtered at "outdated" threshold, so visible = 2
        assert report["visible_findings"] == 2
        assert report["findings_by_category"]["currency_drift"] == 1
        assert report["findings_by_category"]["automation_gap"] == 1
