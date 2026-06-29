"""Unit tests for the improvement_advisor module helper functions.

Tests _filter_reference_files to verify correct filtering of reference
files from RepoContent based on path prefixes and .md extension.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# ---------------------------------------------------------------------------
# Stub out 'strands' and 'strands.models.bedrock' before importing
# improvement_advisor (which imports strands at the top level).
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

from migration_watchdog.improvement_advisor import (
    CHUNK_SIZE,
    CHUNK_THRESHOLD,
    GCP_REFERENCES_PREFIX,
    GCP_SKILL_PATH,
    HEROKU_REFERENCES_PREFIX,
    HEROKU_SKILL_PATH,
    IMPROVEMENT_CATEGORIES,
    _chunk_file,
    _compute_finding_id,
    _filter_reference_files,
    create_improvement_finding,
)
from migration_watchdog.models import RepoContent, RiskLevel

# ---------------------------------------------------------------------------
# Hypothesis imports for property-based tests
# ---------------------------------------------------------------------------

from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Tests for _filter_reference_files
# ---------------------------------------------------------------------------


class TestFilterReferenceFiles:
    """Tests for _filter_reference_files."""

    def test_returns_gcp_reference_md_files(self):
        """Files under GCP_REFERENCES_PREFIX with .md extension are included."""
        repo = RepoContent(files={
            f"{GCP_REFERENCES_PREFIX}phases/01-assess/README.md": "# Assess",
            f"{GCP_REFERENCES_PREFIX}design-refs/compute.md": "# Compute",
        })
        result = _filter_reference_files(repo)
        assert len(result) == 2
        assert f"{GCP_REFERENCES_PREFIX}phases/01-assess/README.md" in result
        assert f"{GCP_REFERENCES_PREFIX}design-refs/compute.md" in result

    def test_returns_heroku_reference_md_files(self):
        """Files under HEROKU_REFERENCES_PREFIX with .md extension are included."""
        repo = RepoContent(files={
            f"{HEROKU_REFERENCES_PREFIX}phases/01-assess/README.md": "# Heroku Assess",
            f"{HEROKU_REFERENCES_PREFIX}shared/networking.md": "# Networking",
        })
        result = _filter_reference_files(repo)
        assert len(result) == 2
        assert f"{HEROKU_REFERENCES_PREFIX}phases/01-assess/README.md" in result
        assert f"{HEROKU_REFERENCES_PREFIX}shared/networking.md" in result

    def test_includes_skill_md_files(self):
        """GCP and Heroku SKILL.md paths are included."""
        repo = RepoContent(files={
            GCP_SKILL_PATH: "# GCP Skill",
            HEROKU_SKILL_PATH: "# Heroku Skill",
        })
        result = _filter_reference_files(repo)
        assert len(result) == 2
        assert GCP_SKILL_PATH in result
        assert HEROKU_SKILL_PATH in result

    def test_excludes_non_md_files(self):
        """Non-.md files under reference prefixes are excluded."""
        repo = RepoContent(files={
            f"{GCP_REFERENCES_PREFIX}shared/sdk-capability-map.json": "{}",
            f"{GCP_REFERENCES_PREFIX}phases/01-assess/config.yaml": "key: val",
            f"{GCP_REFERENCES_PREFIX}phases/01-assess/README.md": "# Include me",
        })
        result = _filter_reference_files(repo)
        assert len(result) == 1
        assert f"{GCP_REFERENCES_PREFIX}phases/01-assess/README.md" in result

    def test_excludes_files_outside_reference_prefixes(self):
        """Files not under reference prefixes or SKILL.md paths are excluded."""
        repo = RepoContent(files={
            "migrate/plugins/migration-to-aws/README.md": "# Plugin readme",
            "some/other/path/file.md": "# Other",
            "migrate/plugins/migration-to-aws/skills/gcp-to-aws/some-other.md": "# Not ref",
        })
        result = _filter_reference_files(repo)
        assert len(result) == 0

    def test_empty_repo_content(self):
        """Empty RepoContent returns empty dict."""
        repo = RepoContent(files={})
        result = _filter_reference_files(repo)
        assert result == {}

    def test_preserves_content(self):
        """Returned dict contains the original file content."""
        content = "# Phase 1\n\nDetailed instructions here."
        repo = RepoContent(files={
            f"{GCP_REFERENCES_PREFIX}phases/01-assess/README.md": content,
        })
        result = _filter_reference_files(repo)
        assert result[f"{GCP_REFERENCES_PREFIX}phases/01-assess/README.md"] == content

    def test_mixed_files_filters_correctly(self):
        """A mix of valid and invalid files is filtered correctly."""
        repo = RepoContent(files={
            f"{GCP_REFERENCES_PREFIX}phases/01-assess/README.md": "gcp phase",
            f"{HEROKU_REFERENCES_PREFIX}design-refs/web.md": "heroku design",
            GCP_SKILL_PATH: "gcp skill",
            HEROKU_SKILL_PATH: "heroku skill",
            f"{GCP_REFERENCES_PREFIX}shared/data.json": "json data",
            "unrelated/file.md": "unrelated",
            "migrate/plugins/migration-to-aws/skills/gcp-to-aws/other.md": "not under refs",
        })
        result = _filter_reference_files(repo)
        assert len(result) == 4
        assert f"{GCP_REFERENCES_PREFIX}phases/01-assess/README.md" in result
        assert f"{HEROKU_REFERENCES_PREFIX}design-refs/web.md" in result
        assert GCP_SKILL_PATH in result
        assert HEROKU_SKILL_PATH in result

    def test_deeply_nested_md_files_included(self):
        """Deeply nested .md files under reference prefixes are included."""
        deep_path = f"{GCP_REFERENCES_PREFIX}phases/03-data/sub/deep/nested.md"
        repo = RepoContent(files={
            deep_path: "# Deep nested",
        })
        result = _filter_reference_files(repo)
        assert len(result) == 1
        assert deep_path in result

# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

import json
import hashlib
import migration_watchdog.improvement_advisor as _ia_module


# --- Strategies ---

# Strategy for valid improvement categories
_valid_categories = st.sampled_from(list(IMPROVEMENT_CATEGORIES))

# Strategy for impact scores in valid range [3, 5]
_valid_impact = st.integers(min_value=3, max_value=5)

# Strategy for effort scores in valid range [1, 3]
_valid_effort = st.integers(min_value=1, max_value=3)

# Strategy for non-empty titles (≤120 chars)
_valid_titles = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=120,
).filter(lambda t: t.strip() != "")

# Strategy for valid affected file paths (non-empty)
_valid_file_paths = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=200,
).map(lambda s: f"migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/{s}.md")

# Strategy for source_urls: 1-5 URLs
_valid_source_urls = st.lists(
    st.just("https://docs.aws.amazon.com/example"),
    min_size=1,
    max_size=5,
)

# Strategy for 3-section descriptions
_valid_descriptions = st.tuples(
    st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""),
    st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""),
    st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""),
).map(lambda parts: f"{parts[0]}\n\n{parts[1]}\n\n{parts[2]}")


class TestPropertyFindingStructureInvariants:
    """Property 1: Finding structure invariants.

    For any Finding object created by create_improvement_finding, verify all
    structural constraints hold simultaneously.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 5.1, 5.2, 5.4, 5.5, 5.6**
    """

    # Feature: improvement-advisor, Property 1: Finding structure invariants

    @given(
        title=_valid_titles,
        description=_valid_descriptions,
        affected_file=_valid_file_paths,
        improvement_category=_valid_categories,
        impact_score=_valid_impact,
        effort_score=_valid_effort,
        source_urls=_valid_source_urls,
    )
    @settings(max_examples=100)
    def test_finding_structure_invariants(
        self,
        title: str,
        description: str,
        affected_file: str,
        improvement_category: str,
        impact_score: int,
        effort_score: int,
        source_urls: list[str],
    ):
        """All structural constraints hold for any valid Finding produced by create_improvement_finding."""
        # Setup module state
        _ia_module._current_findings = []
        _ia_module._current_run_id = "test-run-pbt"

        # Invoke the tool function directly (it's decorated with @tool but acts as identity)
        result_json = create_improvement_finding(
            title=title,
            description=description,
            affected_file=affected_file,
            improvement_category=improvement_category,
            impact_score=impact_score,
            effort_score=effort_score,
            source_urls=json.dumps(source_urls),
        )

        result = json.loads(result_json)
        assert result["created"] is True, f"Finding creation failed: {result.get('error')}"

        # Get the created finding
        assert len(_ia_module._current_findings) == 1
        finding = _ia_module._current_findings[0]

        # Verify ALL structural constraints simultaneously
        assert finding.category == "improvement"
        assert finding.risk_level == RiskLevel.MEDIUM
        assert finding.title != ""
        assert len(finding.title) <= 120
        assert len(finding.affected_files) > 0
        assert finding.finding_schema_version == "improvement/1.0"
        assert 1 <= len(finding.source_urls) <= 5
        assert "improvement_category" in finding.auditor_payload
        assert "impact_score" in finding.auditor_payload
        assert "effort_score" in finding.auditor_payload
        assert finding.auditor_payload["improvement_category"] in IMPROVEMENT_CATEGORIES
        assert isinstance(finding.auditor_payload["impact_score"], int)
        assert 3 <= finding.auditor_payload["impact_score"] <= 5
        assert isinstance(finding.auditor_payload["effort_score"], int)
        assert 1 <= finding.auditor_payload["effort_score"] <= 3


class TestPropertyReferenceFileSelection:
    """Property 2: Reference file selection.

    For any RepoContent with random file paths, _filter_reference_files returns
    exactly paths matching the full prefixes AND ending with .md.

    **Validates: Requirements 2.1, 2.2, 2.3**
    """

    # Feature: improvement-advisor, Property 2: Reference file selection

    @given(
        file_paths=st.lists(
            st.one_of(
                # Paths that SHOULD match (GCP references .md)
                st.text(
                    alphabet=st.characters(whitelist_categories=("L", "N")),
                    min_size=1,
                    max_size=30,
                ).map(lambda s: f"{GCP_REFERENCES_PREFIX}{s}.md"),
                # Paths that SHOULD match (Heroku references .md)
                st.text(
                    alphabet=st.characters(whitelist_categories=("L", "N")),
                    min_size=1,
                    max_size=30,
                ).map(lambda s: f"{HEROKU_REFERENCES_PREFIX}{s}.md"),
                # SKILL paths that SHOULD match
                st.sampled_from([GCP_SKILL_PATH, HEROKU_SKILL_PATH]),
                # Paths under references but NOT .md (should NOT match)
                st.text(
                    alphabet=st.characters(whitelist_categories=("L", "N")),
                    min_size=1,
                    max_size=30,
                ).map(lambda s: f"{GCP_REFERENCES_PREFIX}{s}.json"),
                # Paths that look similar but use short prefix (should NOT match)
                st.text(
                    alphabet=st.characters(whitelist_categories=("L", "N")),
                    min_size=1,
                    max_size=30,
                ).map(lambda s: f"skills/gcp-to-aws/references/{s}.md"),
                # Completely unrelated paths (should NOT match)
                st.text(
                    alphabet=st.characters(whitelist_categories=("L", "N")),
                    min_size=1,
                    max_size=50,
                ).map(lambda s: f"other/{s}.md"),
            ),
            min_size=0,
            max_size=20,
        ),
    )
    @settings(max_examples=100)
    def test_reference_file_selection(self, file_paths: list[str]):
        """Only paths with full prefixes AND .md extension are returned."""
        # Build RepoContent with generated paths
        files = {path: f"content of {path}" for path in file_paths}
        repo = RepoContent(files=files)

        result = _filter_reference_files(repo)

        skill_paths = {GCP_SKILL_PATH, HEROKU_SKILL_PATH}

        # Verify: every returned path satisfies the filter criteria
        for path in result:
            assert path.endswith(".md"), f"Non-.md file returned: {path}"
            assert (
                path.startswith(GCP_REFERENCES_PREFIX)
                or path.startswith(HEROKU_REFERENCES_PREFIX)
                or path in skill_paths
            ), f"Path doesn't match any valid prefix: {path}"

        # Verify: every input path that SHOULD match IS in the result
        for path in file_paths:
            if not path.endswith(".md"):
                continue
            should_match = (
                path.startswith(GCP_REFERENCES_PREFIX)
                or path.startswith(HEROKU_REFERENCES_PREFIX)
                or path in skill_paths
            )
            if should_match:
                assert path in result, f"Expected path missing from result: {path}"
            else:
                assert path not in result, f"Unexpected path in result: {path}"


class TestPropertyFileChunkingPreservesContent:
    """Property 3: File chunking preserves content.

    For any string > 12000 chars, _chunk_file produces chunks each ≤ 8000 chars,
    and stripping "--- Chunk N/M ---\\n" headers from each chunk then concatenating
    gives back the original content.

    **Validates: Requirements 2.4**
    """

    # Feature: improvement-advisor, Property 3: File chunking preserves content

    @given(
        base=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
            min_size=100,
            max_size=500,
        ),
        multiplier=st.integers(min_value=30, max_value=120),
    )
    @settings(max_examples=100)
    def test_file_chunking_preserves_content(self, base: str, multiplier: int):
        """Chunking any large string preserves content and respects size limits."""
        import re

        # Build content > 12000 chars by repeating with paragraph separators
        content = ("\n\n".join([base] * multiplier))
        # Ensure content exceeds threshold
        if len(content) <= CHUNK_THRESHOLD:
            content = content + "x" * (CHUNK_THRESHOLD - len(content) + 1)

        chunks = _chunk_file(content)

        # Must produce at least 2 chunks for content > CHUNK_THRESHOLD
        assert len(chunks) >= 2, (
            f"Expected multiple chunks for content of length {len(content)}, "
            f"got {len(chunks)}"
        )

        # Each chunk must be ≤ CHUNK_SIZE (8000) chars
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= CHUNK_SIZE + 50, (
                f"Chunk {i+1} exceeds max size: {len(chunk)} chars "
                f"(max {CHUNK_SIZE} + header overhead)"
            )

        # Strip headers and concatenate to recover original content
        header_pattern = re.compile(r"^--- Chunk \d+/\d+ ---\n")
        stripped_chunks = []
        for chunk in chunks:
            stripped = header_pattern.sub("", chunk, count=1)
            stripped_chunks.append(stripped)

        reconstructed = "".join(stripped_chunks)
        assert reconstructed == content, (
            f"Reconstructed content differs from original. "
            f"Original length: {len(content)}, reconstructed length: {len(reconstructed)}"
        )


class TestPropertyFindingIdDeterminism:
    """Property 4: Finding ID determinism.

    For any tuple (category, path, title), calling _compute_finding_id twice
    returns the same 32-char hex string. Empty path or title returns empty string.

    **Validates: Requirements 5.3**
    """

    # Feature: improvement-advisor, Property 4: Finding ID determinism

    @given(
        category=st.text(min_size=1, max_size=30),
        path=st.text(min_size=0, max_size=200),
        title=st.text(min_size=0, max_size=120),
    )
    @settings(max_examples=100)
    def test_finding_id_determinism(self, category: str, path: str, title: str):
        """_compute_finding_id is deterministic and handles empty inputs correctly."""
        result1 = _compute_finding_id(category, path, title)
        result2 = _compute_finding_id(category, path, title)

        # Determinism: same inputs always produce same output
        assert result1 == result2

        if not path or not title:
            # Empty path or title → empty string
            assert result1 == ""
        else:
            # Non-empty inputs → 32-char hex string
            assert len(result1) == 32
            assert all(c in "0123456789abcdef" for c in result1)

            # Verify against direct SHA-256 computation
            expected = hashlib.sha256(
                f"{category}|{path}|{title}".encode()
            ).hexdigest()[:32]
            assert result1 == expected
