"""Property-based tests for codebase-improvements spec.

All 7 correctness properties from the design document are implemented here
using the hypothesis library.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import unittest.mock
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from migration_watchdog.findings_repository import FindingsRepository
from migration_watchdog.models import Dismissal, Finding, RiskLevel
from migration_watchdog.scan_config import ScanConfig
from migration_watchdog.utils import has_active_dismissal


# ---------------------------------------------------------------------------
# Helper: build a minimal Finding instance
# ---------------------------------------------------------------------------

def _make_finding(
    finding_id: str = "test-id",
    run_id: str = "run-1",
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    category: str = "currency_drift",
    title: str = "Test finding",
    description: str = "desc",
    dismissal: Dismissal | None = None,
) -> Finding:
    return Finding(
        finding_id=finding_id,
        run_id=run_id,
        risk_level=risk_level,
        category=category,
        title=title,
        description=description,
        dismissal=dismissal,
    )


# ---------------------------------------------------------------------------
# Property 1: Timestamps written by FindingsRepository are timezone-aware
# ---------------------------------------------------------------------------

# Feature: codebase-improvements, Property 1: timestamps written by FindingsRepository are timezone-aware
@given(
    finding_id=st.text(min_size=1, max_size=50),
    run_id=st.text(min_size=1, max_size=50),
    category=st.text(min_size=1, max_size=50),
    title=st.text(min_size=1, max_size=100),
    description=st.text(min_size=0, max_size=200),
)
@settings(max_examples=100)
def test_finding_to_item_timestamps_are_aware(
    finding_id: str,
    run_id: str,
    category: str,
    title: str,
    description: str,
) -> None:
    """Validates: Requirements 2.2"""
    finding = _make_finding(
        finding_id=finding_id,
        run_id=run_id,
        category=category,
        title=title,
        description=description,
    )
    item = FindingsRepository._finding_to_item(finding)
    for key in ("created_at", "updated_at"):
        dt = datetime.fromisoformat(item[key])
        assert dt.tzinfo is not None, (
            f"Expected timezone-aware datetime for {key!r}, got {item[key]!r}"
        )


# ---------------------------------------------------------------------------
# Property 2: has_active_dismissal correctly classifies future dismissals
# ---------------------------------------------------------------------------

# Feature: codebase-improvements, Property 2: has_active_dismissal correctly classifies future dismissals
@given(
    future_dt=st.datetimes(
        # Use utcnow() + buffer so naive datetimes are also in the future when
        # has_active_dismissal normalises them to UTC for comparison.
        min_value=datetime.utcnow() + timedelta(hours=1),
        max_value=datetime(2100, 1, 1),
    ),
    use_utc=st.booleans(),
)
@settings(max_examples=100)
def test_has_active_dismissal_future(future_dt: datetime, use_utc: bool) -> None:
    """Validates: Requirements 3.2, 3.5"""
    if use_utc:
        # Make it UTC-aware by appending +00:00 to the isoformat string
        expires_str = future_dt.isoformat() + "+00:00"
    else:
        # Naive datetime — treated as UTC by has_active_dismissal
        expires_str = future_dt.isoformat()

    dismissal = Dismissal(
        dismissed_at=datetime.now(timezone.utc).isoformat(),
        cooldown_expires=expires_str,
    )
    finding = _make_finding(dismissal=dismissal)
    assert has_active_dismissal(finding) is True, (
        f"Expected True for future cooldown_expires={expires_str!r}"
    )


# ---------------------------------------------------------------------------
# Property 3: has_active_dismissal correctly classifies past dismissals
# ---------------------------------------------------------------------------

# Feature: codebase-improvements, Property 3: has_active_dismissal correctly classifies past dismissals
@given(
    past_dt=st.datetimes(
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2020, 1, 1),
    ),
    use_utc=st.booleans(),
)
@settings(max_examples=100)
def test_has_active_dismissal_past(past_dt: datetime, use_utc: bool) -> None:
    """Validates: Requirements 3.2, 3.6"""
    if use_utc:
        expires_str = past_dt.isoformat() + "+00:00"
    else:
        expires_str = past_dt.isoformat()

    dismissal = Dismissal(
        dismissed_at=datetime(2019, 1, 1).isoformat(),
        cooldown_expires=expires_str,
    )
    finding = _make_finding(dismissal=dismissal)
    assert has_active_dismissal(finding) is False, (
        f"Expected False for past cooldown_expires={expires_str!r}"
    )


# ---------------------------------------------------------------------------
# Property 4: _scan_findings returns all items across all pages
# ---------------------------------------------------------------------------

# Feature: codebase-improvements, Property 4: _scan_findings returns all items across all pages
@given(
    pages=st.lists(
        st.lists(
            st.fixed_dictionaries({"PK": st.text(), "SK": st.just("FINDING")}),
            min_size=0,
            max_size=5,
        ),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=100)
def test_scan_findings_pagination(pages: list[list[dict]]) -> None:
    """Validates: Requirements 5.1, 5.2, 5.3"""
    # Build a mock DynamoDB table whose scan() returns pages sequentially.
    # Each page except the last has a LastEvaluatedKey.
    call_count = [0]

    def mock_scan(**kwargs):
        idx = call_count[0]
        call_count[0] += 1
        page_items = pages[idx] if idx < len(pages) else []
        resp = {"Items": page_items}
        # All pages except the last get a LastEvaluatedKey
        if idx < len(pages) - 1:
            resp["LastEvaluatedKey"] = {"PK": f"page-{idx}"}
        return resp

    mock_table = unittest.mock.MagicMock()
    mock_table.scan.side_effect = mock_scan

    # Construct a FindingsRepository with the mock table (bypass __init__)
    repo = object.__new__(FindingsRepository)
    repo._table = mock_table  # type: ignore[attr-defined]

    result = repo._scan_findings()

    expected_count = sum(len(p) for p in pages)
    assert len(result) == expected_count, (
        f"Expected {expected_count} items across {len(pages)} pages, got {len(result)}"
    )


# ---------------------------------------------------------------------------
# Property 5: _build_pr_comment_markdown output is stable after extraction
# ---------------------------------------------------------------------------

# Feature: codebase-improvements, Property 5: _build_pr_comment_markdown output is stable after extraction
@given(
    pr_number=st.integers(min_value=1, max_value=9999),
    run_id=st.text(min_size=1, max_size=50),
    severity_threshold=st.sampled_from(["outdated", "low"]),
    num_findings=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=100)
def test_pr_comment_markdown_stable(
    pr_number: int,
    run_id: str,
    severity_threshold: str,
    num_findings: int,
) -> None:
    """Validates: Requirements 1.5"""
    from migration_watchdog.pr_commenter import _build_pr_comment_markdown

    findings = [
        _make_finding(
            finding_id=f"id-{i}",
            run_id=run_id,
            title=f"Finding {i}",
        )
        for i in range(num_findings)
    ]
    audited_files = ["file1.py", "file2.py"]

    result = _build_pr_comment_markdown(
        findings=findings,
        pr_number=pr_number,
        run_id=run_id,
        audited_files=audited_files,
        severity_threshold=severity_threshold,
    )

    assert isinstance(result, str), "Expected _build_pr_comment_markdown to return a str"
    assert "<!-- watchdog-audit-comment -->" in result, (
        "Expected marker '<!-- watchdog-audit-comment -->' in output"
    )


# ---------------------------------------------------------------------------
# Property 6: ScanConfig.from_env() raises ValueError for each missing required variable
# ---------------------------------------------------------------------------

# Feature: codebase-improvements, Property 6: ScanConfig.from_env() raises ValueError for each missing required variable
@given(
    missing_var=st.sampled_from(["WATCHDOG_TARGET_OWNER", "WATCHDOG_TARGET_REPO"])
)
@settings(max_examples=100)
def test_scan_config_missing_required(missing_var: str) -> None:
    """Validates: Requirements 6.3"""
    # Set both required vars, then delete the one under test.
    env_overrides = {
        "WATCHDOG_TARGET_OWNER": "test-owner",
        "WATCHDOG_TARGET_REPO": "test-repo",
    }
    env_overrides.pop(missing_var, None)

    # Build a clean env: start from a minimal base and apply overrides.
    clean_env = {k: v for k, v in os.environ.items()}
    # Remove the missing var from the environment
    clean_env.pop(missing_var, None)
    # Set the other required var
    for k, v in env_overrides.items():
        clean_env[k] = v

    with unittest.mock.patch.dict(os.environ, clean_env, clear=True):
        with pytest.raises(ValueError, match=missing_var):
            ScanConfig.from_env()


# ---------------------------------------------------------------------------
# Property 7: main.py contains no inline imports
# ---------------------------------------------------------------------------

# Feature: codebase-improvements, Property 7: main.py contains no inline imports
def test_no_inline_imports_in_main() -> None:
    """Validates: Requirements 1.3

    Uses ast.parse + ast.walk to inspect every FunctionDef and AsyncFunctionDef
    node in main.py and asserts no child Import or ImportFrom nodes exist inside
    function bodies.
    """
    spec = importlib.util.find_spec("migration_watchdog.main")
    assert spec is not None, "Could not find migration_watchdog.main module"
    assert spec.origin is not None, "migration_watchdog.main has no origin file"

    with open(spec.origin, "r", encoding="utf-8") as fh:
        source = fh.read()

    tree = ast.parse(source, filename=spec.origin)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Walk all descendants of this function body
            for child in ast.walk(node):
                if child is node:
                    continue
                assert not isinstance(child, (ast.Import, ast.ImportFrom)), (
                    f"Inline import found in function {node.name!r} "
                    f"at line {child.lineno} in {spec.origin}"
                )
