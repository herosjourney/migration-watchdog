"""Tests for PersonaLibrary — Task 1.2 / 1.4.

Validates that PersonaLibrary.load() correctly parses YAML, constructs the
nested dataclasses, enforces required-field validation, and that get_by_id()
and get_all() behave correctly before and after load().
"""

from __future__ import annotations

import os
import tempfile

import pytest

from migration_watchdog.persona_library import (
    AgenticProfile,
    AIStack,
    ExpectedPath,
    InfrastructureProfile,
    Persona,
    PersonaLibrary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_YAML = """\
- id: "anthropic-sdk-no-gcp"
  description: "Startup using Anthropic SDK directly, no GCP infrastructure"
  infrastructure:
    gcp_services: []
    has_terraform: false
    has_billing_data: false
  ai_stack:
    provider: "anthropic"
    models: ["claude-3-5-sonnet-20241022"]
    frameworks: []
    integration_pattern: "direct_sdk"
    gateway_type: null
  agentic_profile:
    is_agentic: false
    framework: null
    tools: []
    orchestration_pattern: null
  expected_path:
    discover_outputs: ["ai-workload-profile.json"]
    clarify_route: "clarify-ai-only"
    design_route: "design-ai"
    design_refs: ["ai-anthropic-to-bedrock.md"]
  known_gaps_fixed_by_prs: ["PR#30"]
"""

TWO_PERSONA_YAML = """\
- id: "gcp-infra-only"
  description: "GCP infra only, no AI workloads"
  infrastructure:
    gcp_services: ["cloud-run", "cloud-sql"]
    has_terraform: true
    has_billing_data: false
  ai_stack:
    provider: null
    models: []
    frameworks: []
    integration_pattern: null
    gateway_type: null
  agentic_profile:
    is_agentic: false
  expected_path:
    discover_outputs: ["gcp-resource-inventory.json"]
    clarify_route: "clarify-gcp"
    design_route: "design-gcp"
    design_refs: ["compute.md"]
  known_gaps_fixed_by_prs: []

- id: "anthropic-sdk-no-gcp"
  description: "Startup using Anthropic SDK directly, no GCP infrastructure"
  infrastructure:
    gcp_services: []
    has_terraform: false
    has_billing_data: false
  ai_stack:
    provider: "anthropic"
    models: ["claude-3-5-sonnet-20241022"]
    frameworks: []
    integration_pattern: "direct_sdk"
    gateway_type: null
  agentic_profile:
    is_agentic: false
  expected_path:
    discover_outputs: ["ai-workload-profile.json"]
    clarify_route: "clarify-ai-only"
    design_route: "design-ai"
    design_refs: ["ai-anthropic-to-bedrock.md"]
  known_gaps_fixed_by_prs: ["PR#30"]
"""


def _write_yaml(content: str) -> str:
    """Write YAML content to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# load() — happy path
# ---------------------------------------------------------------------------


def test_load_returns_list_of_personas():
    path = _write_yaml(MINIMAL_YAML)
    try:
        lib = PersonaLibrary()
        personas = lib.load(path)
        assert isinstance(personas, list)
        assert len(personas) == 1
        assert isinstance(personas[0], Persona)
    finally:
        os.unlink(path)


def test_load_parses_id_and_description():
    path = _write_yaml(MINIMAL_YAML)
    try:
        lib = PersonaLibrary()
        p = lib.load(path)[0]
        assert p.id == "anthropic-sdk-no-gcp"
        assert p.description == "Startup using Anthropic SDK directly, no GCP infrastructure"
    finally:
        os.unlink(path)


def test_load_parses_infrastructure():
    path = _write_yaml(MINIMAL_YAML)
    try:
        lib = PersonaLibrary()
        p = lib.load(path)[0]
        assert isinstance(p.infrastructure, InfrastructureProfile)
        assert p.infrastructure.gcp_services == []
        assert p.infrastructure.has_terraform is False
        assert p.infrastructure.has_billing_data is False
    finally:
        os.unlink(path)


def test_load_parses_ai_stack():
    path = _write_yaml(MINIMAL_YAML)
    try:
        lib = PersonaLibrary()
        p = lib.load(path)[0]
        assert isinstance(p.ai_stack, AIStack)
        assert p.ai_stack.provider == "anthropic"
        assert p.ai_stack.models == ["claude-3-5-sonnet-20241022"]
        assert p.ai_stack.frameworks == []
        assert p.ai_stack.integration_pattern == "direct_sdk"
        assert p.ai_stack.gateway_type is None
    finally:
        os.unlink(path)


def test_load_parses_agentic_profile():
    path = _write_yaml(MINIMAL_YAML)
    try:
        lib = PersonaLibrary()
        p = lib.load(path)[0]
        assert isinstance(p.agentic_profile, AgenticProfile)
        assert p.agentic_profile.is_agentic is False
        assert p.agentic_profile.framework is None
        assert p.agentic_profile.tools == []
        assert p.agentic_profile.orchestration_pattern is None
    finally:
        os.unlink(path)


def test_load_parses_expected_path():
    path = _write_yaml(MINIMAL_YAML)
    try:
        lib = PersonaLibrary()
        p = lib.load(path)[0]
        assert isinstance(p.expected_path, ExpectedPath)
        assert p.expected_path.discover_outputs == ["ai-workload-profile.json"]
        assert p.expected_path.clarify_route == "clarify-ai-only"
        assert p.expected_path.design_route == "design-ai"
        assert p.expected_path.design_refs == ["ai-anthropic-to-bedrock.md"]
    finally:
        os.unlink(path)


def test_load_parses_known_gaps():
    path = _write_yaml(MINIMAL_YAML)
    try:
        lib = PersonaLibrary()
        p = lib.load(path)[0]
        assert p.known_gaps_fixed_by_prs == ["PR#30"]
    finally:
        os.unlink(path)


def test_load_multiple_personas():
    path = _write_yaml(TWO_PERSONA_YAML)
    try:
        lib = PersonaLibrary()
        personas = lib.load(path)
        assert len(personas) == 2
        ids = [p.id for p in personas]
        assert "gcp-infra-only" in ids
        assert "anthropic-sdk-no-gcp" in ids
    finally:
        os.unlink(path)


def test_load_gcp_infra_persona_fields():
    path = _write_yaml(TWO_PERSONA_YAML)
    try:
        lib = PersonaLibrary()
        personas = lib.load(path)
        gcp = next(p for p in personas if p.id == "gcp-infra-only")
        assert gcp.infrastructure.gcp_services == ["cloud-run", "cloud-sql"]
        assert gcp.infrastructure.has_terraform is True
        assert gcp.ai_stack.provider is None
        assert gcp.expected_path.clarify_route == "clarify-gcp"
        assert gcp.expected_path.design_route == "design-gcp"
        assert gcp.expected_path.design_refs == ["compute.md"]
        assert gcp.known_gaps_fixed_by_prs == []
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# load() — validation errors
# ---------------------------------------------------------------------------


def test_load_raises_for_missing_id():
    yaml_content = """\
- description: "No id field"
  expected_path:
    clarify_route: "x"
    design_route: "y"
"""
    path = _write_yaml(yaml_content)
    try:
        lib = PersonaLibrary()
        with pytest.raises(ValueError, match="id"):
            lib.load(path)
    finally:
        os.unlink(path)


def test_load_raises_for_empty_id():
    yaml_content = """\
- id: ""
  description: "Empty id"
  expected_path:
    clarify_route: "x"
    design_route: "y"
"""
    path = _write_yaml(yaml_content)
    try:
        lib = PersonaLibrary()
        with pytest.raises(ValueError, match="id"):
            lib.load(path)
    finally:
        os.unlink(path)


def test_load_raises_for_missing_description():
    yaml_content = """\
- id: "test-persona"
  expected_path:
    clarify_route: "x"
    design_route: "y"
"""
    path = _write_yaml(yaml_content)
    try:
        lib = PersonaLibrary()
        with pytest.raises(ValueError, match="description"):
            lib.load(path)
    finally:
        os.unlink(path)


def test_load_raises_for_missing_clarify_route():
    yaml_content = """\
- id: "test-persona"
  description: "A persona"
  expected_path:
    design_route: "design-ai"
"""
    path = _write_yaml(yaml_content)
    try:
        lib = PersonaLibrary()
        with pytest.raises(ValueError, match="clarify_route"):
            lib.load(path)
    finally:
        os.unlink(path)


def test_load_raises_for_missing_design_route():
    yaml_content = """\
- id: "test-persona"
  description: "A persona"
  expected_path:
    clarify_route: "clarify-ai-only"
"""
    path = _write_yaml(yaml_content)
    try:
        lib = PersonaLibrary()
        with pytest.raises(ValueError, match="design_route"):
            lib.load(path)
    finally:
        os.unlink(path)


def test_load_raises_for_non_list_yaml():
    yaml_content = "id: not-a-list\n"
    path = _write_yaml(yaml_content)
    try:
        lib = PersonaLibrary()
        with pytest.raises(ValueError, match="list"):
            lib.load(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# get_by_id()
# ---------------------------------------------------------------------------


def test_get_by_id_returns_matching_persona():
    path = _write_yaml(TWO_PERSONA_YAML)
    try:
        lib = PersonaLibrary()
        lib.load(path)
        p = lib.get_by_id("anthropic-sdk-no-gcp")
        assert p is not None
        assert p.id == "anthropic-sdk-no-gcp"
    finally:
        os.unlink(path)


def test_get_by_id_returns_none_for_unknown_id():
    path = _write_yaml(MINIMAL_YAML)
    try:
        lib = PersonaLibrary()
        lib.load(path)
        assert lib.get_by_id("does-not-exist") is None
    finally:
        os.unlink(path)


def test_get_by_id_raises_before_load():
    lib = PersonaLibrary()
    with pytest.raises(RuntimeError, match="call load\\(\\) first"):
        lib.get_by_id("any-id")


# ---------------------------------------------------------------------------
# get_all()
# ---------------------------------------------------------------------------


def test_get_all_returns_all_personas():
    path = _write_yaml(TWO_PERSONA_YAML)
    try:
        lib = PersonaLibrary()
        lib.load(path)
        all_personas = lib.get_all()
        assert len(all_personas) == 2
    finally:
        os.unlink(path)


def test_get_all_returns_copy():
    path = _write_yaml(MINIMAL_YAML)
    try:
        lib = PersonaLibrary()
        personas = lib.load(path)
        copy1 = lib.get_all()
        copy2 = lib.get_all()
        # Each call returns a new list object
        assert copy1 is not personas
        assert copy1 is not copy2
        # But the contents are the same Persona instances
        assert copy1[0] is personas[0]
    finally:
        os.unlink(path)


def test_get_all_raises_before_load():
    lib = PersonaLibrary()
    with pytest.raises(RuntimeError, match="call load\\(\\) first"):
        lib.get_all()


# ---------------------------------------------------------------------------
# Production personas.yaml validation
# ---------------------------------------------------------------------------

from pathlib import Path

PERSONAS_YAML_PATH = Path(__file__).parent.parent / "migration_watchdog" / "personas.yaml"

EXPECTED_PERSONA_IDS = [
    "gcp-infra-only",
    "gcp-infra-openai-direct",
    "gcp-infra-langchain-openai",
    "gcp-infra-langgraph-openai-agentic",
    "gcp-infra-gemini-direct",
    "ai-only-anthropic-sdk",
    "ai-only-openrouter-gateway",
    "ai-only-litellm-proxy",
    "ai-only-crewai-openai",
    "ai-only-openai-agents-sdk",
    "gcp-infra-multi-model",
    "billing-data-only",
]

VALID_PHASE_TOKENS = {
    "clarify-global",
    "clarify-ai",
    "clarify-ai-only",
    "design-infra",
    "design-ai",
    "design-billing",
}

PR30_PERSONA_IDS = [
    "ai-only-anthropic-sdk",
    "ai-only-openrouter-gateway",
    "ai-only-litellm-proxy",
]


def test_production_personas_load_all():
    """Loads personas.yaml and asserts at least 15 personas are returned with no errors."""
    lib = PersonaLibrary()
    personas = lib.load(str(PERSONAS_YAML_PATH))
    assert len(personas) >= 15, f"Expected at least 15 personas, got {len(personas)}"


def test_production_personas_all_have_unique_ids():
    """Asserts all 12 persona IDs are unique (no duplicates)."""
    lib = PersonaLibrary()
    personas = lib.load(str(PERSONAS_YAML_PATH))
    ids = [p.id for p in personas]
    assert len(ids) == len(set(ids)), f"Duplicate persona IDs found: {[i for i in ids if ids.count(i) > 1]}"


def test_production_personas_expected_ids_present():
    """Asserts all 12 expected persona IDs are present in personas.yaml."""
    lib = PersonaLibrary()
    personas = lib.load(str(PERSONAS_YAML_PATH))
    loaded_ids = {p.id for p in personas}
    for expected_id in EXPECTED_PERSONA_IDS:
        assert expected_id in loaded_ids, f"Expected persona ID '{expected_id}' not found in personas.yaml"


def test_production_personas_valid_phase_names():
    """Asserts clarify_route and design_route for every persona contain only valid phase tokens."""
    lib = PersonaLibrary()
    personas = lib.load(str(PERSONAS_YAML_PATH))
    for persona in personas:
        for token in persona.expected_path.clarify_route.split("+"):
            assert token in VALID_PHASE_TOKENS, (
                f"Persona '{persona.id}' has invalid clarify_route token '{token}'. "
                f"Valid tokens: {VALID_PHASE_TOKENS}"
            )
        for token in persona.expected_path.design_route.split("+"):
            assert token in VALID_PHASE_TOKENS, (
                f"Persona '{persona.id}' has invalid design_route token '{token}'. "
                f"Valid tokens: {VALID_PHASE_TOKENS}"
            )


def test_production_personas_pr30_gap_anchors():
    """Asserts the three PR#30 gap personas exist and have 'PR#30' in known_gaps_fixed_by_prs."""
    lib = PersonaLibrary()
    personas = lib.load(str(PERSONAS_YAML_PATH))
    persona_map = {p.id: p for p in personas}
    for persona_id in PR30_PERSONA_IDS:
        assert persona_id in persona_map, f"PR#30 gap persona '{persona_id}' not found"
        persona = persona_map[persona_id]
        assert "PR#30" in persona.known_gaps_fixed_by_prs, (
            f"Persona '{persona_id}' is missing 'PR#30' in known_gaps_fixed_by_prs. "
            f"Got: {persona.known_gaps_fixed_by_prs}"
        )


def test_production_personas_agentic_have_framework():
    """Asserts every agentic persona (is_agentic=True) has a non-None framework."""
    lib = PersonaLibrary()
    personas = lib.load(str(PERSONAS_YAML_PATH))
    for persona in personas:
        if persona.agentic_profile.is_agentic:
            assert persona.agentic_profile.framework is not None, (
                f"Persona '{persona.id}' has is_agentic=True but framework is None"
            )


def test_production_personas_ai_only_have_no_gcp_services():
    """Asserts personas with IDs starting with 'ai-only-' have no GCP services and no Terraform."""
    lib = PersonaLibrary()
    personas = lib.load(str(PERSONAS_YAML_PATH))
    for persona in personas:
        if persona.id.startswith("ai-only-"):
            assert persona.infrastructure.gcp_services == [], (
                f"Persona '{persona.id}' starts with 'ai-only-' but has gcp_services: "
                f"{persona.infrastructure.gcp_services}"
            )
            assert persona.infrastructure.has_terraform is False, (
                f"Persona '{persona.id}' starts with 'ai-only-' but has has_terraform=True"
            )
