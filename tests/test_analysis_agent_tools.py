"""Unit tests for @tool functions in analysis_agent.py.

Tests cover:
  - compare_pricing
  - compare_models
  - check_new_content_opportunities
  - create_finding
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub out 'strands' and 'strands.models.bedrock' before importing
# analysis_agent, which imports strands at the top level.
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

import migration_watchdog.analysis_agent as _agent_module
from migration_watchdog.analysis_agent import (
    check_new_content_opportunities,
    compare_models,
    compare_pricing,
    create_finding,
)

# ---------------------------------------------------------------------------
# Fixture: reset module-level _current_findings before/after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_current_findings():
    _agent_module._current_findings.clear()
    _agent_module._current_run_id = "test-run-id"
    yield
    _agent_module._current_findings.clear()


# ---------------------------------------------------------------------------
# Minimal pricing markdown used across compare_pricing tests
# ---------------------------------------------------------------------------

_PRICING_MD = """\
**Last updated:** 2024-01-01

## Compute

### Fargate

| Metric | Rate |
|--------|------|
| per vCPU-hour | $0.04048 |
"""


# ===========================================================================
# Sub-task 4.1: compare_pricing tests
# ===========================================================================

class TestComparePricing:
    def test_exceeds_tolerance_when_price_differs_significantly(self):
        """A current price >10% different from cached should set exceeds_tolerance=True."""
        current_prices = {"fargate": {"per vcpu-hour": 0.10}}
        result_json = compare_pricing(_PRICING_MD, json.dumps(current_prices))
        result = json.loads(result_json)

        assert "entries" in result
        fargate_entries = [e for e in result["entries"] if e["service"] == "fargate"]
        assert fargate_entries, "Expected at least one fargate entry"

        entry = fargate_entries[0]
        assert entry["exceeds_tolerance"] is True, (
            f"Expected exceeds_tolerance=True for price 0.10 vs cached 0.04048, "
            f"got difference_pct={entry.get('difference_pct')}"
        )

    def test_within_tolerance_when_price_is_close(self):
        """A current price within 10% of cached should set exceeds_tolerance=False."""
        current_prices = {"fargate": {"per vcpu-hour": 0.04050}}
        result_json = compare_pricing(_PRICING_MD, json.dumps(current_prices))
        result = json.loads(result_json)

        assert "entries" in result
        fargate_entries = [e for e in result["entries"] if e["service"] == "fargate"]
        assert fargate_entries, "Expected at least one fargate entry"

        entry = fargate_entries[0]
        assert entry["exceeds_tolerance"] is False, (
            f"Expected exceeds_tolerance=False for price 0.04050 vs cached 0.04048, "
            f"got difference_pct={entry.get('difference_pct')}"
        )


# ===========================================================================
# Sub-task 4.3: compare_models tests
# ===========================================================================

_MODEL_MD = """\
| gpt-4o | some description |
| gpt-4-turbo | some description |
"""


class TestCompareModels:
    def test_model_missing_from_repo_is_flagged(self):
        """A model in the current catalog but absent from the repo markdown
        should appear in models_missing_from_repo."""
        current_models = {"models": ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"]}
        result_json = compare_models(_MODEL_MD, "openai", json.dumps(current_models))
        result = json.loads(result_json)

        assert "gpt-4o-mini" in result["models_missing_from_repo"], (
            f"Expected 'gpt-4o-mini' in models_missing_from_repo, "
            f"got {result['models_missing_from_repo']}"
        )

    def test_model_removed_from_catalog_is_flagged(self):
        """A model in the repo markdown but absent from the current catalog
        should appear in models_removed_from_catalog."""
        current_models = {"models": ["gpt-4o"]}
        result_json = compare_models(_MODEL_MD, "openai", json.dumps(current_models))
        result = json.loads(result_json)

        assert "gpt-4-turbo" in result["models_removed_from_catalog"], (
            f"Expected 'gpt-4-turbo' in models_removed_from_catalog, "
            f"got {result['models_removed_from_catalog']}"
        )


# ===========================================================================
# Sub-task 4.5: check_new_content_opportunities tests
# ===========================================================================

class TestCheckNewContentOpportunities:
    def test_no_opportunity_when_topic_covered_in_repo(self):
        """When repo files already mention 'bedrock agents', no opportunity
        for bedrock_agents should be returned."""
        repo_files = {
            "design-refs/compute.md": "This guide covers bedrock agents in detail."
        }
        open_prs: list = []
        recent_updates = [
            {"title": "bedrock agents update", "content": "bedrock agents now support..."}
        ]

        result_json = check_new_content_opportunities(
            json.dumps(repo_files),
            json.dumps(open_prs),
            json.dumps(recent_updates),
        )
        result = json.loads(result_json)

        topics = [opp["topic"] for opp in result.get("opportunities", [])]
        assert "bedrock_agents" not in topics, (
            "Expected no bedrock_agents opportunity when repo already covers the topic"
        )

    def test_opportunity_returned_when_topic_uncovered(self):
        """When repo files are empty and recent updates mention 'bedrock agents',
        an opportunity with topic='bedrock_agents' should be returned."""
        repo_files: dict = {}
        open_prs: list = []
        recent_updates = [
            {"title": "bedrock agents update", "content": "bedrock agents now support..."}
        ]

        result_json = check_new_content_opportunities(
            json.dumps(repo_files),
            json.dumps(open_prs),
            json.dumps(recent_updates),
        )
        result = json.loads(result_json)

        topics = [opp["topic"] for opp in result.get("opportunities", [])]
        assert "bedrock_agents" in topics, (
            f"Expected bedrock_agents opportunity when repo is empty and updates mention it, "
            f"got topics={topics}"
        )


# ===========================================================================
# Sub-task 4.7: create_finding tests
# ===========================================================================

class TestCreateFinding:
    def test_rejects_empty_source_urls(self):
        """create_finding should return created=False when source_urls is empty."""
        result_json = create_finding(
            category="pricing",
            title="test",
            description="desc",
            affected_files='["file.md"]',
            proposed_changes="{}",
            source_urls="[]",
        )
        result = json.loads(result_json)

        assert result["created"] is False, (
            f"Expected created=False for empty source_urls, got {result}"
        )
        assert len(_agent_module._current_findings) == 0, (
            "Expected no findings appended when source_urls is empty"
        )

    def test_creates_and_appends_finding_with_valid_source_urls(self):
        """create_finding should return created=True and append to _current_findings
        when given valid source_urls."""
        result_json = create_finding(
            category="pricing",
            title="Fargate price changed",
            description="Fargate vCPU price has changed significantly.",
            affected_files='["migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/shared/pricing-cache.md"]',
            proposed_changes="{}",
            source_urls='["https://docs.aws.amazon.com/test"]',
        )
        result = json.loads(result_json)

        assert result["created"] is True, (
            f"Expected created=True for valid source_urls, got {result}"
        )
        assert len(_agent_module._current_findings) == 1, (
            f"Expected 1 finding appended, got {len(_agent_module._current_findings)}"
        )
