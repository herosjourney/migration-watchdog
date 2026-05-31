"""Property-based tests for watchdog-coverage-and-reliability spec.

Properties 2, 3, 7, 8, 9 from the design document are implemented here
using the hypothesis library. Unit tests for exhausted-retry paths are also
included.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
import unittest.mock
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Stub out 'strands' and 'strands.models.bedrock' before any module-level
# import of security_auditor (which imports strands at the top level).
# ---------------------------------------------------------------------------
if "strands" not in sys.modules:
    _strands_stub = types.ModuleType("strands")
    _strands_stub.Agent = MagicMock()  # type: ignore[attr-defined]
    _strands_stub.tool = lambda f: f   # type: ignore[attr-defined]
    sys.modules["strands"] = _strands_stub

if "strands.models" not in sys.modules:
    _strands_models_stub = types.ModuleType("strands.models")
    sys.modules["strands.models"] = _strands_models_stub

if "strands.models.bedrock" not in sys.modules:
    _strands_bedrock_stub = types.ModuleType("strands.models.bedrock")
    _strands_bedrock_stub.BedrockModel = MagicMock()  # type: ignore[attr-defined]
    sys.modules["strands.models.bedrock"] = _strands_bedrock_stub

from migration_watchdog.alias_table import AliasTable
from migration_watchdog.automation_auditor import ActionExtractor, CLILookup, ManualAction
from migration_watchdog.coverage_assessor import CoverageAssessor, CoverageResult
from migration_watchdog.models import RepoContent
from migration_watchdog.path_tracer import PathTracer
from migration_watchdog.persona_library import (
    AgenticProfile,
    AIStack,
    ExpectedPath,
    InfrastructureProfile,
    Persona,
    PersonaLibrary,
)
from migration_watchdog.security_auditor import LLMSecurityScanner, run_security_audit


# ---------------------------------------------------------------------------
# Module-level fixtures (loaded once)
# ---------------------------------------------------------------------------

_ALIAS_TABLE = AliasTable()
_CLI_LOOKUP = CLILookup()

_PERSONAS_YAML = Path(__file__).parent.parent / "personas.yaml"
_persona_library = PersonaLibrary()
_persona_library.load(str(_PERSONAS_YAML))


# ---------------------------------------------------------------------------
# Property 2: Security scan failures are always recorded as partial failures
# ---------------------------------------------------------------------------

# Feature: watchdog-coverage-and-reliability, Property 2: Security scan failures are always recorded as partial failures
@given(
    file_stems=st.lists(
        st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        ),
        min_size=1,
        max_size=8,
    ),
    failing_indices=st.sets(st.integers(min_value=0, max_value=7)),
)
@settings(max_examples=50, deadline=None)
def test_security_scan_failures_recorded_as_partial_failures(
    file_stems: list[str],
    failing_indices: set[int],
) -> None:
    """Validates: security scan failures are always recorded in partial_failures."""
    # Build file paths that contain "generate-artifacts" so they pass the
    # security-relevant filter in run_security_audit.
    files = {
        f"generate-artifacts/{stem}.md": f"content of {stem}"
        for stem in file_stems
    }
    repo_content = RepoContent(files=files)

    # Determine which file paths should fail (by index into the files dict).
    file_paths = list(files.keys())
    # Clamp indices to the actual number of files.
    actual_failing = {i for i in failing_indices if i < len(file_paths)}
    failing_paths = {file_paths[i] for i in actual_failing}

    def mock_scan_file(file_path: str, content: str) -> list[dict]:
        if file_path in failing_paths:
            raise RuntimeError("simulated failure")
        return []

    partial_failures: list[str] = []

    with patch.object(LLMSecurityScanner, "scan_file", side_effect=mock_scan_file):
        asyncio.run(
            run_security_audit(
                repo_content=repo_content,
                run_id="test-run",
                partial_failures=partial_failures,
            )
        )

    # Assert exactly one partial_failure entry per failing file.
    assert len(partial_failures) == len(failing_paths), (
        f"Expected {len(failing_paths)} partial failure(s), got {len(partial_failures)}"
    )

    # Assert each entry is valid JSON with the required keys.
    for entry in partial_failures:
        parsed = json.loads(entry)
        assert parsed.get("type") == "security_scan_failure", (
            f"Expected type='security_scan_failure', got {parsed.get('type')!r}"
        )
        assert "file" in parsed, f"Expected 'file' key in partial failure entry: {parsed}"
        assert parsed["file"] in failing_paths, (
            f"Partial failure file {parsed['file']!r} not in expected failing paths"
        )


# ---------------------------------------------------------------------------
# Property 3: Security scan failures do not abort remaining file scans
# ---------------------------------------------------------------------------

# Feature: watchdog-coverage-and-reliability, Property 3: Security scan failures do not abort remaining file scans
@given(
    n_files=st.integers(min_value=2, max_value=8),
    fail_index=st.integers(min_value=0, max_value=7),
)
@settings(max_examples=50, deadline=None)
def test_security_scan_failure_does_not_abort_remaining_scans(
    n_files: int,
    fail_index: int,
) -> None:
    """Validates: a scan failure on one file does not abort scanning of remaining files."""
    # Clamp fail_index to valid range.
    k = fail_index % n_files

    files = {
        f"generate-artifacts/file{i}.md": f"content {i}"
        for i in range(n_files)
    }
    repo_content = RepoContent(files=files)
    file_paths = list(files.keys())
    failing_path = file_paths[k]

    call_log: list[str] = []

    def mock_scan_file(file_path: str, content: str) -> list[dict]:
        call_log.append(file_path)
        if file_path == failing_path:
            raise RuntimeError("simulated failure")
        return []

    partial_failures: list[str] = []

    with patch.object(LLMSecurityScanner, "scan_file", side_effect=mock_scan_file):
        asyncio.run(
            run_security_audit(
                repo_content=repo_content,
                run_id="test-run",
                partial_failures=partial_failures,
            )
        )

    # All N files must have been scanned despite the failure.
    assert len(call_log) == n_files, (
        f"Expected scan_file to be called {n_files} times, got {len(call_log)}"
    )


# ---------------------------------------------------------------------------
# Property 7: Alias table resolves all required model names
# ---------------------------------------------------------------------------

REQUIRED_MODEL_NAMES = [
    "us.amazon.nova-2-lite-v1:0",
    "Nova 2 Lite",
    "gemini-2.5-flash",
    "Gemini 2.5 Flash",
    "gpt-4o",
    "GPT-4o",
    "gpt-4o-mini",
    "GPT-4o mini",
    "claude-3-5-sonnet-20241022",
    "gpt-oss-120b",
    "gpt-oss-20b",
]

# Feature: watchdog-coverage-and-reliability, Property 7: Alias table resolves all required model names
@given(st.sampled_from(REQUIRED_MODEL_NAMES))
@settings(max_examples=100)
def test_alias_table_resolves_required_model_names(name: str) -> None:
    """Validates: alias table resolves all required model names to a canonical ID."""
    _canonical_id, found = _ALIAS_TABLE.resolve(name)
    assert found is True, (
        f"AliasTable.resolve({name!r}) returned found=False; "
        "expected this model name to be in the alias table"
    )


# ---------------------------------------------------------------------------
# Property 8: CLI lookup returns a match for all required keyword phrases
# ---------------------------------------------------------------------------

REQUIRED_KEYWORD_PHRASES = [
    "enable model access",
    "request model access",
    "bedrock model access",
    "create secret",
    "secrets manager",
    "store secret",
    "create agent",
    "agentcore",
    "bedrock agent",
    "create repository",
    "ecr",
    "container registry",
    "push image",
    "docker push",
    "ecr push",
]

# Feature: watchdog-coverage-and-reliability, Property 8: CLI lookup returns a match for all required keyword phrases
@given(st.sampled_from(REQUIRED_KEYWORD_PHRASES))
@settings(max_examples=100)
def test_cli_lookup_returns_match_for_required_keyword_phrases(
    keyword_phrase: str,
) -> None:
    """Validates: CLILookup returns a CLI equivalent for all required keyword phrases."""
    action = ManualAction(
        action_text=keyword_phrase,
        action_type="console_navigation",
        context="",
        generated_artifact=None,
        action_fingerprint="test",
    )
    result = _CLI_LOOKUP.lookup(action)
    assert result.cli_equivalent is not None, (
        f"CLILookup.lookup() returned cli_equivalent=None for keyword phrase {keyword_phrase!r}; "
        "expected a CLI command to be found"
    )


# ---------------------------------------------------------------------------
# Property 9: New personas route to non-empty design-ref decision points
# ---------------------------------------------------------------------------

NEW_PERSONA_IDS = [
    "gcp-cloud-functions",
    "gcp-gce",
    "gcp-firestore",
    "gcp-cloud-storage",
    "gcp-pubsub",
    "gcp-cloud-spanner",
    "ai-only-openai-assistants-api",
    "ai-only-autogen",
]

# Feature: watchdog-coverage-and-reliability, Property 9: New personas route to non-empty design-ref decision points
@given(st.sampled_from(NEW_PERSONA_IDS))
@settings(max_examples=100)
def test_new_personas_route_to_non_empty_decision_points(persona_id: str) -> None:
    """Validates: new personas produce at least one design-ref decision point."""
    persona = _persona_library.get_by_id(persona_id)
    assert persona is not None, (
        f"Persona {persona_id!r} not found in PersonaLibrary; "
        "ensure it is defined in personas.yaml"
    )

    trace = PathTracer().trace(persona, RepoContent(files={}))

    assert len(trace.decision_points) > 0, (
        f"Persona {persona_id!r} produced no decision points; "
        "expected at least one routing decision to be recorded"
    )


# ---------------------------------------------------------------------------
# Unit: CoverageAssessor exhausted retries returns unverified
# ---------------------------------------------------------------------------

def test_coverage_assessor_exhausted_retries_returns_unverified() -> None:
    """Validates: CoverageAssessor._assess_file returns unverified when all retries exhausted."""
    assessor = CoverageAssessor()

    persona = Persona(
        id="test-persona",
        description="Test persona for retry exhaustion",
        infrastructure=InfrastructureProfile(),
        ai_stack=AIStack(),
        agentic_profile=AgenticProfile(),
        expected_path=ExpectedPath(clarify_route="clarify-global", design_route="design-infra"),
    )

    with patch(
        "migration_watchdog.retry.retry_with_backoff",
        side_effect=RuntimeError("all retries exhausted"),
    ):
        result = asyncio.run(
            assessor._assess_file(persona, "test.md", "content")
        )

    assert result.coverage == "unverified", (
        f"Expected coverage='unverified', got {result.coverage!r}"
    )
    assert result.confidence == 0.0, (
        f"Expected confidence=0.0, got {result.confidence}"
    )


# ---------------------------------------------------------------------------
# Unit: ActionExtractor exhausted retries returns empty list
# ---------------------------------------------------------------------------

def test_action_extractor_exhausted_retries_returns_empty_list() -> None:
    """Validates: ActionExtractor._call_llm returns [] when all retries exhausted."""
    extractor = ActionExtractor()

    with patch(
        "migration_watchdog.retry.retry_with_backoff",
        side_effect=RuntimeError("all retries exhausted"),
    ):
        result = asyncio.run(
            extractor._call_llm("test.md", "aws migration content")
        )

    assert result == [], (
        f"Expected empty list on exhausted retries, got {result!r}"
    )
