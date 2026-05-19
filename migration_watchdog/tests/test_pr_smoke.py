"""Smoke test: run_scan() in PR mode with minimal env vars.

Verifies that:
1. run_scan() completes without crashing when TRIGGER_TYPE=pull_request
   and GitHub App credentials are absent (PR checkout path).
2. audit-report.json is written with the expected shape.
3. partial_source_failures is initialized before currency audit runs
   (regression test for the NameError bug).

All external calls (DynamoDB, GitHub API, LLM, S3) are mocked.
No network access required.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# ---------------------------------------------------------------------------
# Stub out migration_watchdog.* imports that require AWS/Strands at import time
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    return types.ModuleType(name)


_STUBS = [
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
    "migration_watchdog.currency_auditor",
    "migration_watchdog.automation_auditor",
    "migration_watchdog.security_auditor",
    "migration_watchdog.pr_commenter",
    "migration_watchdog.scan_config",
]

for _stub_name in _STUBS:
    if _stub_name not in sys.modules:
        sys.modules[_stub_name] = _make_stub(_stub_name)

import models as _models_real  # noqa: E402

_mw_models = sys.modules["migration_watchdog.models"]
_mw_models.Finding = _models_real.Finding
_mw_models.RiskLevel = _models_real.RiskLevel
_mw_models.ScanRun = _models_real.ScanRun
_mw_models.RepoContent = _models_real.RepoContent

_mw_analysis = sys.modules["migration_watchdog.analysis_agent"]
_mw_analysis.run_analysis = MagicMock(return_value=[])

_mw_dedup = sys.modules["migration_watchdog.finding_deduplicator"]
_mw_dedup.deduplicate = MagicMock(return_value=[])
_mw_dedup._is_addressed_by_pr = MagicMock(return_value=False)

_mw_repo = sys.modules["migration_watchdog.findings_repository"]
_mock_repo = MagicMock()
_mock_repo.list_findings = MagicMock(return_value=[])
_mock_repo.save_finding = MagicMock()
_mock_repo.save_run = MagicMock()
_mw_repo.FindingsRepository = MagicMock(return_value=_mock_repo)

_mw_refactor = sys.modules["migration_watchdog.refactoring_agent"]
_mw_refactor.run_refactoring_assessment = MagicMock(return_value=None)

_mw_scanner = sys.modules["migration_watchdog.repo_scanner"]
_mw_scanner.RepoScanner = MagicMock()

_mw_retry = sys.modules["migration_watchdog.retry"]
async def _passthrough(op, **kw): return await op()
_mw_retry.retry_with_backoff = _passthrough

_mw_review = sys.modules["migration_watchdog.review_agent"]
_mw_review.review_findings = MagicMock(return_value=[])

_mw_fetcher = sys.modules["migration_watchdog.source_fetcher"]
_mock_fetcher = MagicMock()
_mock_auth_data = _models_real.AuthoritativeData()
_mock_fetcher.fetch_all_sources = AsyncMock(return_value=_mock_auth_data)
_mw_fetcher.SourceFetcher = MagicMock(return_value=_mock_fetcher)

_mw_currency = sys.modules["migration_watchdog.currency_auditor"]
_mw_currency.run_currency_audit = AsyncMock(return_value=[])
_mw_currency.currency_dedupe_key = MagicMock(side_effect=lambda f: (f.category, frozenset(f.affected_files), f.finding_id))

_mw_automation = sys.modules["migration_watchdog.automation_auditor"]
_mw_automation.run_automation_audit = AsyncMock(return_value=[])
_mw_automation.automation_dedupe_key = MagicMock(side_effect=lambda f: (f.category, frozenset(f.affected_files), f.finding_id))

# Stub security_auditor
_mw_security = sys.modules["migration_watchdog.security_auditor"]
_mw_security.run_security_audit = AsyncMock(return_value=[])
_mw_security.security_dedupe_key = MagicMock(side_effect=lambda f: (f.category, frozenset(f.affected_files), f.finding_id))

# Stub pr_commenter — provide the four functions main.py imports
_mw_pr_commenter = sys.modules["migration_watchdog.pr_commenter"]
_mw_pr_commenter._finding_severity = MagicMock(return_value="outdated")
_mw_pr_commenter._should_include_in_pr_comment = MagicMock(return_value=True)
_mw_pr_commenter._build_pr_comment_markdown = MagicMock(return_value="<!-- watchdog-audit-comment -->")
_mw_pr_commenter._post_pr_comment = AsyncMock()

# Stub scan_config — provide ScanConfig with from_env()
import dataclasses as _dc

@_dc.dataclass
class _MockScanConfig:
    github_app_id: str = ""
    github_app_private_key: str = ""
    github_installation_id: str = ""
    dynamodb_table: str = "test-table"
    aws_region: str = "us-east-1"
    target_owner: str = "owner"
    target_repo: str = "repo"
    trigger_type: str = "pull_request"
    pr_number: int | None = 42
    pr_head_sha: str | None = "abc1234"
    pr_html_url: str | None = "https://github.com/owner/repo/pull/42"
    changed_files: list = _dc.field(default_factory=list)
    github_token: str = "fake-token"
    severity_threshold: str = "outdated"

    @classmethod
    def from_env(cls) -> "_MockScanConfig":
        import os as _os
        changed_str = _os.environ.get("CHANGED_FILES", "")
        changed = [f.strip() for f in changed_str.split() if f.strip()]
        return cls(
            trigger_type=_os.environ.get("TRIGGER_TYPE", "scheduled"),
            pr_number=int(_os.environ.get("PR_NUMBER", "0")) or None,
            pr_head_sha=_os.environ.get("PR_HEAD_SHA"),
            pr_html_url=_os.environ.get("PR_HTML_URL"),
            changed_files=changed,
            github_token=_os.environ.get("GITHUB_TOKEN", ""),
            target_owner=_os.environ.get("WATCHDOG_TARGET_OWNER", "owner"),
            target_repo=_os.environ.get("WATCHDOG_TARGET_REPO", "repo"),
            dynamodb_table=_os.environ.get("DYNAMODB_TABLE", "test-table"),
            aws_region=_os.environ.get("AWS_REGION", "us-east-1"),
        )

_mw_scan_config = sys.modules["migration_watchdog.scan_config"]
_mw_scan_config.ScanConfig = _MockScanConfig

import main as _main_module  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_MD = """# GCP to AWS Migration Guide

This guide covers migrating from GCP to AWS.

## AgentCore Availability

Amazon Bedrock AgentCore is available in 9 regions.
"""

_SAMPLE_PRICING_CACHE = """# Pricing Cache

**Last updated:** 2026-01-01

## Compute

### Fargate

| Metric | Rate |
|--------|------|
| per_vcpu_hour | $0.04048 |
"""


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestPRModeSmoke:
    """Smoke tests for run_scan() in PR-triggered mode."""

    def _run_with_pr_env(self, tmp_path: Path, changed_files: list[str]) -> dict:
        """Run run_scan() with PR env vars and return the audit-report.json content."""
        # Write fixture files to tmp_path
        for file_path in changed_files:
            full_path = tmp_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(_SAMPLE_MD, encoding="utf-8")

        # Write pricing cache
        pricing_path = tmp_path / "migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/shared/pricing-cache.md"
        pricing_path.parent.mkdir(parents=True, exist_ok=True)
        pricing_path.write_text(_SAMPLE_PRICING_CACHE, encoding="utf-8")

        env = {
            "TRIGGER_TYPE": "pull_request",
            "PR_NUMBER": "42",
            "PR_HEAD_SHA": "abc1234",
            "PR_HTML_URL": "https://github.com/owner/repo/pull/42",
            "CHANGED_FILES": " ".join(str(tmp_path / f) for f in changed_files),
            "GITHUB_TOKEN": "fake-token",
            "WATCHDOG_TARGET_OWNER": "owner",
            "WATCHDOG_TARGET_REPO": "repo",
            "DYNAMODB_TABLE": "test-table",
            "AWS_REGION": "us-east-1",
            # No GITHUB_APP_ID — PR checkout path
        }

        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        orig_env = {k: os.environ.get(k) for k in env}
        try:
            os.environ.update(env)
            # Remove App vars to ensure PR checkout path is taken
            for app_var in ("GITHUB_APP_ID", "GITHUB_APP_PRIVATE_KEY", "GITHUB_INSTALLATION_ID"):
                os.environ.pop(app_var, None)

            with patch.object(_main_module, "SourceFetcher") as mock_sf_cls, \
                 patch.object(_main_module, "boto3") as mock_boto3, \
                 patch.object(_main_module.httpx, "AsyncClient") as mock_client_cls:
                # Patch AsyncClient only — do not replace the whole httpx module,
                # or except httpx.HTTPStatusError breaks (mock is not BaseException).
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.get = AsyncMock(side_effect=Exception("no network"))
                mock_client_cls.return_value = mock_client

                # Mock boto3.resource for DynamoDB
                mock_boto3.resource = MagicMock()

                # Mock SourceFetcher with async fetch_all_sources
                mock_sf = AsyncMock()
                mock_sf.fetch_all_sources = AsyncMock(return_value=_mock_auth_data)
                mock_sf_cls.return_value = mock_sf

                asyncio.run(_main_module.run_scan())

        except SystemExit as exc:
            # run_scan calls sys.exit(1) on failure — treat as test failure
            pytest.fail(f"run_scan() called sys.exit({exc.code})")
        finally:
            os.chdir(orig_dir)
            for k, v in orig_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        report_path = tmp_path / "audit-report.json"
        assert report_path.exists(), "audit-report.json must be written by run_scan() in PR mode"
        return json.loads(report_path.read_text())

    def test_pr_mode_completes_without_crash(self, tmp_path):
        """run_scan() in PR mode must complete without raising or calling sys.exit(1)."""
        changed = ["migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/shared/ai-migration-guardrails.md"]
        report = self._run_with_pr_env(tmp_path, changed)
        assert "run_id" in report

    def test_audit_report_json_written(self, tmp_path):
        """audit-report.json must be written with the expected top-level keys."""
        changed = ["migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/shared/ai-migration-guardrails.md"]
        report = self._run_with_pr_env(tmp_path, changed)

        assert "run_id" in report, "run_id must be present"
        assert "pr_number" in report, "pr_number must be present"
        assert "findings" in report, "findings array must be present"
        assert "total_findings" in report, "total_findings must be present"
        assert "visible_findings" in report, "visible_findings must be present"
        assert report["pr_number"] == 42

    def test_audit_report_findings_is_list(self, tmp_path):
        """findings in audit-report.json must be a list (even when empty)."""
        changed = ["migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/shared/ai-migration-guardrails.md"]
        report = self._run_with_pr_env(tmp_path, changed)
        assert isinstance(report["findings"], list)

    def test_no_partial_source_failures_nameerror(self, tmp_path):
        """Regression: partial_source_failures must be defined before currency audit runs."""
        _mw_currency.run_currency_audit.return_value = []
        _mw_currency.run_currency_audit.last_partial_source_failures = [
            {"type": "migration_context_failure", "file": "test.md"}
        ]

        changed = ["migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/shared/ai-migration-guardrails.md"]
        try:
            report = self._run_with_pr_env(tmp_path, changed)
            # If we get here, no NameError occurred
            assert "run_id" in report
        finally:
            # Reset
            _mw_currency.run_currency_audit.last_partial_source_failures = []

    def test_pr_checkout_loads_pricing_cache(self, tmp_path):
        """PR checkout mode must load pricing-cache.md alongside changed files."""
        changed = ["migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/shared/ai-migration-guardrails.md"]

        # Track what files were loaded into RepoContent
        loaded_files: dict = {}

        original_run_currency = _mw_currency.run_currency_audit

        async def capture_repo_content(repo_content, *args, **kwargs):
            loaded_files.update(repo_content.files)
            return []

        _mw_currency.run_currency_audit = AsyncMock(side_effect=capture_repo_content)
        try:
            self._run_with_pr_env(tmp_path, changed)
        finally:
            _mw_currency.run_currency_audit = original_run_currency

        # pricing-cache.md must be in the snapshot
        pricing_key = "migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/shared/pricing-cache.md"
        assert any(pricing_key in k or "pricing-cache" in k for k in loaded_files), (
            f"pricing-cache.md must be loaded in PR checkout mode. "
            f"Loaded files: {list(loaded_files.keys())}"
        )
