"""Tests for CoverageAssessor.

All LLM calls are mocked — no real Bedrock calls are made.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure migration_watchdog resolves as a package.
# ---------------------------------------------------------------------------
_PACKAGE_ROOT = Path(__file__).parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from migration_watchdog.coverage_assessor import CoverageAssessor, CoverageResult
from migration_watchdog.models import RepoContent
from migration_watchdog.path_tracer import ExecutionTrace
from migration_watchdog.persona_library import (
    AgenticProfile,
    AIStack,
    ExpectedPath,
    InfrastructureProfile,
    Persona,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DESIGN_REFS_PREFIX = (
    "migrate/plugins/migration-to-aws/skills/gcp-to-aws/"
    "references/design-refs/"
)


def _make_persona(
    persona_id: str = "test-persona",
    provider: str | None = "anthropic",
    models: list[str] | None = None,
    frameworks: list[str] | None = None,
    integration_pattern: str | None = "direct_sdk",
    gateway_type: str | None = None,
    is_agentic: bool = False,
    agentic_framework: str | None = None,
) -> Persona:
    """Build a minimal Persona for testing."""
    return Persona(
        id=persona_id,
        description=f"Test persona: {persona_id}",
        infrastructure=InfrastructureProfile(gcp_services=[], has_terraform=False),
        ai_stack=AIStack(
            provider=provider,
            models=models or ["claude-3-5-sonnet-20241022"],
            frameworks=frameworks or [],
            integration_pattern=integration_pattern,
            gateway_type=gateway_type,
        ),
        agentic_profile=AgenticProfile(
            is_agentic=is_agentic,
            framework=agentic_framework,
        ),
        expected_path=ExpectedPath(
            discover_outputs=["ai-workload-profile.json"],
            clarify_route="clarify-ai-only",
            design_route="design-ai",
            design_refs=["ai-anthropic-to-bedrock.md"],
        ),
    )


def _make_repo_with_files(*paths: str) -> RepoContent:
    """RepoContent with specific files present."""
    return RepoContent(
        files={p: f"# Content of {p}\n\nSome guidance here." for p in paths},
        open_prs=[],
        commit_sha="test-sha",
        fetched_at="2026-01-01",
    )


def _make_empty_repo() -> RepoContent:
    """RepoContent with no files."""
    return RepoContent(files={}, open_prs=[], commit_sha="test-sha", fetched_at="2026-01-01")


# ---------------------------------------------------------------------------
# Task 3.5 / 3.1: missing_file handling
# ---------------------------------------------------------------------------


def test_missing_file_produces_missing_file_result():
    """A persona with a missing design-ref file produces a missing_file result with confidence 1.0."""
    trace = ExecutionTrace(
        persona_id="ai-only-anthropic-sdk",
        design_refs_loaded=["ai-anthropic-to-bedrock.md"],
        files_missing=["ai-anthropic-to-bedrock.md"],
    )
    persona = _make_persona("ai-only-anthropic-sdk")
    repo = _make_empty_repo()

    assessor = CoverageAssessor()
    results = asyncio.run(assessor.assess(persona, trace, repo))

    assert len(results) == 1
    result = results[0]
    assert result.coverage == "missing_file"
    assert result.confidence == 1.0
    assert result.persona_id == "ai-only-anthropic-sdk"
    assert result.file_path == "ai-anthropic-to-bedrock.md"


def test_missing_file_result_has_gap_description():
    """The missing_file result includes a gap describing the missing file."""
    trace = ExecutionTrace(
        persona_id="test-persona",
        design_refs_loaded=["ai-anthropic-to-bedrock.md"],
        files_missing=["ai-anthropic-to-bedrock.md"],
    )
    persona = _make_persona("test-persona")
    repo = _make_empty_repo()

    assessor = CoverageAssessor()
    results = asyncio.run(assessor.assess(persona, trace, repo))

    assert len(results) == 1
    assert len(results[0].gaps) > 0
    assert "ai-anthropic-to-bedrock.md" in results[0].gaps[0]


def test_missing_file_result_has_suggested_addition():
    """The missing_file result includes a suggested addition."""
    trace = ExecutionTrace(
        persona_id="test-persona",
        design_refs_loaded=["ai-anthropic-to-bedrock.md"],
        files_missing=["ai-anthropic-to-bedrock.md"],
    )
    persona = _make_persona("test-persona", provider="anthropic")
    repo = _make_empty_repo()

    assessor = CoverageAssessor()
    results = asyncio.run(assessor.assess(persona, trace, repo))

    assert len(results) == 1
    assert len(results[0].suggested_additions) > 0


def test_no_llm_call_for_missing_files():
    """Missing files do not trigger LLM calls — _assess_file is never called."""
    trace = ExecutionTrace(
        persona_id="test-persona",
        design_refs_loaded=["ai-anthropic-to-bedrock.md"],
        files_missing=["ai-anthropic-to-bedrock.md"],
    )
    persona = _make_persona("test-persona")
    repo = _make_empty_repo()

    assessor = CoverageAssessor()
    with patch.object(CoverageAssessor, "_assess_file", new_callable=AsyncMock) as mock_assess:
        results = asyncio.run(assessor.assess(persona, trace, repo))
        mock_assess.assert_not_called()

    assert len(results) == 1
    assert results[0].coverage == "missing_file"


# ---------------------------------------------------------------------------
# Task 3.3 / 3.4: LLM-based assessment
# ---------------------------------------------------------------------------


def test_adequate_coverage_returns_adequate():
    """When _assess_file returns adequate coverage, the result has coverage='adequate'."""
    trace = ExecutionTrace(
        persona_id="gcp-infra-openai-direct",
        design_refs_loaded=["ai-openai-to-bedrock.md"],
        files_missing=[],
    )
    persona = _make_persona("gcp-infra-openai-direct", provider="openai")
    repo = _make_repo_with_files(_DESIGN_REFS_PREFIX + "ai-openai-to-bedrock.md")

    adequate_result = CoverageResult(
        persona_id="gcp-infra-openai-direct",
        file_path="ai-openai-to-bedrock.md",
        coverage="adequate",
        gaps=[],
        confidence=0.95,
        suggested_additions=[],
    )

    assessor = CoverageAssessor()
    with patch.object(CoverageAssessor, "_assess_file", new_callable=AsyncMock, return_value=adequate_result):
        results = asyncio.run(assessor.assess(persona, trace, repo))

    assert len(results) == 1
    assert results[0].coverage == "adequate"
    assert results[0].confidence == 0.95
    assert results[0].gaps == []


def test_gap_coverage_returns_gap():
    """When _assess_file returns a gap, the result has coverage='gap' with gap details."""
    trace = ExecutionTrace(
        persona_id="ai-only-anthropic-sdk",
        design_refs_loaded=["ai.md"],
        files_missing=[],
    )
    persona = _make_persona("ai-only-anthropic-sdk", provider="anthropic")
    repo = _make_repo_with_files(_DESIGN_REFS_PREFIX + "ai.md")

    gap_result = CoverageResult(
        persona_id="ai-only-anthropic-sdk",
        file_path="ai.md",
        coverage="gap",
        gaps=[
            "no guidance for Anthropic SDK client initialization",
            "missing Claude model ID mapping",
        ],
        confidence=0.85,
        suggested_additions=["Add Anthropic SDK to Bedrock migration section"],
    )

    assessor = CoverageAssessor()
    with patch.object(CoverageAssessor, "_assess_file", new_callable=AsyncMock, return_value=gap_result):
        results = asyncio.run(assessor.assess(persona, trace, repo))

    assert len(results) == 1
    assert results[0].coverage == "gap"
    assert len(results[0].gaps) == 2
    assert results[0].confidence == 0.85


def test_multiple_design_refs_assessed_separately():
    """Multiple design-ref files in the trace each produce a separate CoverageResult."""
    trace = ExecutionTrace(
        persona_id="gcp-infra-openai-direct",
        design_refs_loaded=["compute.md", "ai-openai-to-bedrock.md"],
        files_missing=[],
    )
    persona = _make_persona("gcp-infra-openai-direct", provider="openai")
    repo = _make_repo_with_files(
        _DESIGN_REFS_PREFIX + "compute.md",
        _DESIGN_REFS_PREFIX + "ai-openai-to-bedrock.md",
    )

    compute_result = CoverageResult(
        persona_id="gcp-infra-openai-direct",
        file_path="compute.md",
        coverage="adequate",
        gaps=[],
        confidence=0.9,
        suggested_additions=[],
    )
    ai_result = CoverageResult(
        persona_id="gcp-infra-openai-direct",
        file_path="ai-openai-to-bedrock.md",
        coverage="adequate",
        gaps=[],
        confidence=0.92,
        suggested_additions=[],
    )

    assessor = CoverageAssessor()
    with patch.object(
        CoverageAssessor, "_assess_file", new_callable=AsyncMock, side_effect=[compute_result, ai_result]
    ):
        results = asyncio.run(assessor.assess(persona, trace, repo))

    assert len(results) == 2
    file_paths = {r.file_path for r in results}
    assert "compute.md" in file_paths
    assert "ai-openai-to-bedrock.md" in file_paths


def test_mixed_missing_and_loaded_files():
    """A trace with both missing and loaded files produces results for both."""
    trace = ExecutionTrace(
        persona_id="test-persona",
        design_refs_loaded=["ai-anthropic-to-bedrock.md", "compute.md"],
        files_missing=["ai-anthropic-to-bedrock.md"],
    )
    persona = _make_persona("test-persona", provider="anthropic")
    repo = _make_repo_with_files(_DESIGN_REFS_PREFIX + "compute.md")

    compute_result = CoverageResult(
        persona_id="test-persona",
        file_path="compute.md",
        coverage="adequate",
        gaps=[],
        confidence=0.88,
        suggested_additions=[],
    )

    assessor = CoverageAssessor()
    with patch.object(CoverageAssessor, "_assess_file", new_callable=AsyncMock, return_value=compute_result):
        results = asyncio.run(assessor.assess(persona, trace, repo))

    assert len(results) == 2
    coverages = {r.file_path: r.coverage for r in results}
    assert coverages["ai-anthropic-to-bedrock.md"] == "missing_file"
    assert coverages["compute.md"] == "adequate"


# ---------------------------------------------------------------------------
# Task 3.3: _parse_response
# ---------------------------------------------------------------------------


def test_parse_response_adequate():
    """_parse_response correctly parses an adequate coverage JSON response."""
    assessor = CoverageAssessor()
    response = '{"coverage": "adequate", "gaps": [], "confidence": 0.95, "suggested_additions": []}'
    result = assessor._parse_response(response, "test-persona", "compute.md")

    assert result.coverage == "adequate"
    assert result.confidence == 0.95
    assert result.gaps == []
    assert result.persona_id == "test-persona"
    assert result.file_path == "compute.md"


def test_parse_response_gap():
    """_parse_response correctly parses a gap coverage JSON response."""
    assessor = CoverageAssessor()
    response = (
        '{"coverage": "gap", '
        '"gaps": ["no Anthropic SDK guidance", "missing model IDs"], '
        '"confidence": 0.8, '
        '"suggested_additions": ["Add Anthropic section"]}'
    )
    result = assessor._parse_response(response, "test-persona", "ai.md")

    assert result.coverage == "gap"
    assert len(result.gaps) == 2
    assert result.confidence == 0.8
    assert len(result.suggested_additions) == 1


def test_parse_response_with_markdown_fence():
    """_parse_response handles JSON wrapped in markdown code fences."""
    assessor = CoverageAssessor()
    response = (
        "Here is my assessment:\n\n"
        "```json\n"
        '{"coverage": "intentional_deferral", "gaps": [], "confidence": 0.75, "suggested_additions": []}\n'
        "```\n"
    )
    result = assessor._parse_response(response, "test-persona", "compute.md")

    assert result.coverage == "intentional_deferral"
    assert result.confidence == 0.75


def test_parse_response_invalid_json_returns_unverified():
    """_parse_response returns unverified result when JSON is invalid."""
    assessor = CoverageAssessor()
    result = assessor._parse_response("not valid json at all", "test-persona", "compute.md")

    assert result.coverage == "unverified"
    assert result.confidence == 0.0
    assert result.gaps == []


def test_parse_response_unknown_coverage_value_returns_unverified():
    """_parse_response returns unverified when coverage value is not a known enum."""
    assessor = CoverageAssessor()
    response = '{"coverage": "unknown_value", "gaps": [], "confidence": 0.9, "suggested_additions": []}'
    result = assessor._parse_response(response, "test-persona", "compute.md")

    assert result.coverage == "unverified"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Task 3.4: _find_file_content
# ---------------------------------------------------------------------------


def test_find_file_content_design_refs_path():
    """_find_file_content finds files under the design-refs/ prefix."""
    assessor = CoverageAssessor()
    repo = _make_repo_with_files(_DESIGN_REFS_PREFIX + "compute.md")

    content = assessor._find_file_content("compute.md", repo)
    assert content is not None
    assert "compute.md" in content


def test_find_file_content_phases_design_path():
    """_find_file_content finds files under the phases/design/ prefix."""
    phases_prefix = (
        "migrate/plugins/migration-to-aws/skills/gcp-to-aws/"
        "references/phases/design/"
    )
    assessor = CoverageAssessor()
    repo = _make_repo_with_files(phases_prefix + "compute.md")

    content = assessor._find_file_content("compute.md", repo)
    assert content is not None


def test_find_file_content_returns_none_when_not_found():
    """_find_file_content returns None when the file is not in the repo."""
    assessor = CoverageAssessor()
    repo = _make_empty_repo()

    content = assessor._find_file_content("nonexistent.md", repo)
    assert content is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_trace_returns_empty_results():
    """An empty trace (no design refs) produces no results."""
    trace = ExecutionTrace(
        persona_id="test-persona",
        design_refs_loaded=[],
        files_missing=[],
    )
    persona = _make_persona("test-persona")
    repo = _make_empty_repo()

    assessor = CoverageAssessor()
    results = asyncio.run(assessor.assess(persona, trace, repo))

    assert results == []


def test_file_not_in_repo_content_is_skipped():
    """A design-ref that is loaded (not missing) but absent from repo_content is skipped."""
    trace = ExecutionTrace(
        persona_id="test-persona",
        design_refs_loaded=["compute.md"],
        files_missing=[],
    )
    persona = _make_persona("test-persona")
    # Repo has no files — compute.md is not in files_missing but also not in repo_content
    repo = _make_empty_repo()

    assessor = CoverageAssessor()
    with patch.object(CoverageAssessor, "_assess_file", new_callable=AsyncMock) as mock_assess:
        results = asyncio.run(assessor.assess(persona, trace, repo))
        mock_assess.assert_not_called()

    # No result produced for a file that can't be found in repo_content
    assert results == []


def test_confidence_clamped_to_valid_range():
    """_parse_response clamps confidence to [0.0, 1.0]."""
    assessor = CoverageAssessor()

    # Confidence > 1.0
    response_high = '{"coverage": "adequate", "gaps": [], "confidence": 1.5, "suggested_additions": []}'
    result_high = assessor._parse_response(response_high, "test", "file.md")
    assert result_high.confidence <= 1.0

    # Confidence < 0.0
    response_low = '{"coverage": "adequate", "gaps": [], "confidence": -0.5, "suggested_additions": []}'
    result_low = assessor._parse_response(response_low, "test", "file.md")
    assert result_low.confidence >= 0.0
