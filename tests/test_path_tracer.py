"""Regression tests for PathTracer — all 12 personas.

Each test verifies that the traced path for a specific persona matches the
expected routing behaviour encoded in the persona's ``expected_path`` field.
Tests are deterministic: no LLM calls, no network I/O.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from migration_watchdog.models import RepoContent
from migration_watchdog.path_tracer import PathTracer
from migration_watchdog.persona_library import PersonaLibrary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PERSONAS_YAML = (
    Path(__file__).parent.parent / "migration_watchdog" / "personas.yaml"
)


def _load_persona(persona_id: str):
    """Load a single persona by ID from the canonical YAML file."""
    lib = PersonaLibrary()
    lib.load(str(PERSONAS_YAML))
    persona = lib.get_by_id(persona_id)
    assert persona is not None, f"Persona '{persona_id}' not found in personas.yaml"
    return persona


def make_empty_repo_content() -> RepoContent:
    """RepoContent with no files — tests missing_file detection."""
    return RepoContent(files={}, open_prs=[], commit_sha="test", fetched_at="2026-01-01")


def make_repo_with_files(*paths: str) -> RepoContent:
    """RepoContent with specific files present."""
    return RepoContent(
        files={p: "content" for p in paths},
        open_prs=[],
        commit_sha="test",
        fetched_at="2026-01-01",
    )


# ---------------------------------------------------------------------------
# Discover-phase tests
# ---------------------------------------------------------------------------


def test_gcp_infra_only_discover_outputs():
    """gcp-infra-only: discover returns inventory + clusters, NOT ai-workload-profile."""
    persona = _load_persona("gcp-infra-only")
    tracer = PathTracer()
    outputs = tracer._simulate_discover(persona)

    assert "gcp-resource-inventory.json" in outputs
    assert "gcp-resource-clusters.json" in outputs
    assert "ai-workload-profile.json" not in outputs


def test_anthropic_sdk_discover_outputs():
    """ai-only-anthropic-sdk: discover returns ai-workload-profile with anthropic source."""
    persona = _load_persona("ai-only-anthropic-sdk")
    tracer = PathTracer()
    outputs = tracer._simulate_discover(persona)

    assert "ai-workload-profile.json" in outputs
    assert "gcp-resource-inventory.json" not in outputs

    ai_profile = outputs["ai-workload-profile.json"]
    assert ai_profile["summary"]["ai_source"] == "anthropic"


def test_openrouter_gateway_discover_outputs():
    """ai-only-openrouter-gateway: discover returns ai-workload-profile with gateway_type openrouter."""
    persona = _load_persona("ai-only-openrouter-gateway")
    tracer = PathTracer()
    outputs = tracer._simulate_discover(persona)

    assert "ai-workload-profile.json" in outputs
    ai_profile = outputs["ai-workload-profile.json"]
    assert ai_profile["integration"]["gateway_type"] == "openrouter"


def test_billing_only_discover_outputs():
    """billing-data-only: discover returns billing-profile.json only."""
    persona = _load_persona("billing-data-only")
    tracer = PathTracer()
    outputs = tracer._simulate_discover(persona)

    assert "billing-profile.json" in outputs
    assert "gcp-resource-inventory.json" not in outputs
    assert "ai-workload-profile.json" not in outputs


def test_multi_model_discover_outputs():
    """gcp-infra-multi-model: discover returns ai-workload-profile with ai_source == 'both'."""
    persona = _load_persona("gcp-infra-multi-model")
    tracer = PathTracer()
    outputs = tracer._simulate_discover(persona)

    assert "ai-workload-profile.json" in outputs
    ai_profile = outputs["ai-workload-profile.json"]
    assert ai_profile["summary"]["ai_source"] == "both"


# ---------------------------------------------------------------------------
# Design-ref routing tests
# ---------------------------------------------------------------------------


def test_anthropic_sdk_design_refs():
    """ai-only-anthropic-sdk: design refs include ai-anthropic-to-bedrock.md, NOT ai.md."""
    persona = _load_persona("ai-only-anthropic-sdk")
    tracer = PathTracer()
    refs = tracer._trace_design(persona, preferences={})

    assert "ai-anthropic-to-bedrock.md" in refs
    assert "ai.md" not in refs


def test_openai_direct_design_refs():
    """gcp-infra-openai-direct: design refs include ai-openai-to-bedrock.md."""
    persona = _load_persona("gcp-infra-openai-direct")
    tracer = PathTracer()
    refs = tracer._trace_design(persona, preferences={})

    assert "ai-openai-to-bedrock.md" in refs


def test_gemini_direct_design_refs():
    """gcp-infra-gemini-direct: design refs include ai-gemini-to-bedrock.md."""
    persona = _load_persona("gcp-infra-gemini-direct")
    tracer = PathTracer()
    refs = tracer._trace_design(persona, preferences={})

    assert "ai-gemini-to-bedrock.md" in refs


def test_agentic_langgraph_design_refs():
    """gcp-infra-langgraph-openai-agentic: design refs include design-ref-agentic-to-agentcore.md."""
    persona = _load_persona("gcp-infra-langgraph-openai-agentic")
    tracer = PathTracer()
    refs = tracer._trace_design(persona, preferences={})

    assert "design-ref-agentic-to-agentcore.md" in refs


# ---------------------------------------------------------------------------
# Full trace tests (missing-file detection)
# ---------------------------------------------------------------------------


def test_trace_missing_file_detected():
    """ai-only-anthropic-sdk with empty repo: ai-anthropic-to-bedrock.md is in files_missing."""
    persona = _load_persona("ai-only-anthropic-sdk")
    tracer = PathTracer()
    repo = make_empty_repo_content()

    result = tracer.trace(persona, repo)

    assert "ai-anthropic-to-bedrock.md" in result.files_missing
    assert "ai-anthropic-to-bedrock.md" not in result.files_loaded


def test_trace_file_found_when_present():
    """ai-only-anthropic-sdk with repo containing the file: it's in files_loaded, not files_missing."""
    persona = _load_persona("ai-only-anthropic-sdk")
    tracer = PathTracer()

    # Provide the file at the design-refs path
    file_path = (
        "migrate/plugins/migration-to-aws/skills/gcp-to-aws/"
        "references/design-refs/ai-anthropic-to-bedrock.md"
    )
    repo = make_repo_with_files(file_path)

    result = tracer.trace(persona, repo)

    assert "ai-anthropic-to-bedrock.md" in result.files_loaded
    assert "ai-anthropic-to-bedrock.md" not in result.files_missing


# ---------------------------------------------------------------------------
# Smoke test: all 12 personas trace without error
# ---------------------------------------------------------------------------


def test_all_personas_trace_without_error():
    """All 12 personas can be traced against an empty repo without raising exceptions."""
    lib = PersonaLibrary()
    personas = lib.load(str(PERSONAS_YAML))
    assert len(personas) >= 12, f"Expected at least 12 personas, got {len(personas)}"

    tracer = PathTracer()
    repo = make_empty_repo_content()

    for persona in personas:
        result = tracer.trace(persona, repo)
        assert result.persona_id == persona.id
        assert len(result.phases_executed) > 0
