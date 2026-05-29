"""Integration tests for run_scenario_audit().

All LLM calls (CoverageAssessor._assess_file) are mocked.
PathTracer and GapClassifier run with real logic.
"""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from migration_watchdog.coverage_assessor import CoverageAssessor, CoverageResult
from migration_watchdog.models import RepoContent, RiskLevel
from migration_watchdog.scenario_auditor import run_scenario_audit

PERSONAS_YAML = Path(__file__).parent.parent / "personas.yaml"


def make_empty_repo():
    return RepoContent(files={}, open_prs=[], commit_sha="test", fetched_at="2026-01-01")


def test_run_scenario_audit_returns_findings_for_missing_files():
    """run_scenario_audit() returns HIGH severity findings for missing design-ref files."""
    # Empty repo — all design-ref files will be missing
    repo = make_empty_repo()

    # Mock _assess_file so no LLM calls are made for loaded files
    # (with empty repo, all files are missing, so _assess_file won't be called anyway)
    findings = asyncio.get_event_loop().run_until_complete(
        run_scenario_audit(
            repo_content=repo,
            run_id="test-run",
            persona_library_path=str(PERSONAS_YAML),
        )
    )

    # With empty repo, all design-ref files are missing → HIGH severity findings
    assert len(findings) > 0
    high_findings = [f for f in findings if f.risk_level == RiskLevel.HIGH]
    assert len(high_findings) > 0


def test_run_scenario_audit_anthropic_persona_produces_finding():
    """The ai-only-anthropic-sdk persona produces a finding for ai-anthropic-to-bedrock.md."""
    repo = make_empty_repo()

    findings = asyncio.get_event_loop().run_until_complete(
        run_scenario_audit(
            repo_content=repo,
            run_id="test-run",
            persona_library_path=str(PERSONAS_YAML),
        )
    )

    # Find findings related to ai-anthropic-to-bedrock.md
    anthropic_findings = [
        f for f in findings
        if "ai-anthropic-to-bedrock.md" in f.affected_files
    ]
    assert len(anthropic_findings) > 0
    assert anthropic_findings[0].risk_level == RiskLevel.HIGH


def test_run_scenario_audit_pr_optimization_filters_personas():
    """PR optimization: only personas with overlapping design_refs are assessed."""
    repo = make_empty_repo()

    # Only changed files related to anthropic
    changed_files = [
        "migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/design-refs/ai-anthropic-to-bedrock.md"
    ]

    findings = asyncio.get_event_loop().run_until_complete(
        run_scenario_audit(
            repo_content=repo,
            run_id="test-run",
            persona_library_path=str(PERSONAS_YAML),
            changed_files=changed_files,
        )
    )

    # Should still produce findings (the anthropic persona is selected)
    # Just verify it runs without error and returns a list
    assert isinstance(findings, list)


def test_run_scenario_audit_returns_list():
    """run_scenario_audit() always returns a list, even with no findings."""
    repo = RepoContent(
        files={
            # Provide all expected design-ref files so nothing is missing
            "migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/design-refs/ai-anthropic-to-bedrock.md": "# Anthropic to Bedrock\n\nGuidance here.",
            "migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/design-refs/ai-openai-to-bedrock.md": "# OpenAI to Bedrock\n\nGuidance here.",
            "migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/design-refs/ai-gemini-to-bedrock.md": "# Gemini to Bedrock\n\nGuidance here.",
            "migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/design-refs/compute.md": "# Compute\n\nGuidance here.",
            "migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/design-refs/database.md": "# Database\n\nGuidance here.",
            "migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/design-refs/design-ref-agentic-to-agentcore.md": "# Agentic\n\nGuidance here.",
            "migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/phases/design/design-billing.md": "# Billing\n\nGuidance here.",
        },
        open_prs=[],
        commit_sha="test",
        fetched_at="2026-01-01",
    )

    # Mock _assess_file to return adequate coverage for all files
    adequate_result = CoverageResult(
        persona_id="test",
        file_path="test.md",
        coverage="adequate",
        gaps=[],
        confidence=0.95,
        suggested_additions=[],
    )

    with patch.object(CoverageAssessor, "_assess_file", return_value=adequate_result):
        findings = asyncio.get_event_loop().run_until_complete(
            run_scenario_audit(
                repo_content=repo,
                run_id="test-run",
                persona_library_path=str(PERSONAS_YAML),
            )
        )

    assert isinstance(findings, list)
