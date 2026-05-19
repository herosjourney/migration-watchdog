"""Shared pytest configuration for the migration_watchdog test suite.

Ensures migration_watchdog resolves as a proper package by inserting the
*parent* of the package directory into sys.path before any test imports run.

This avoids per-file sys.path hacks and the "not a package" collision that
occurs when pytest adds the flat module directory to sys.path directly.

Test isolation note
-------------------
test_main_pipeline.py and test_pr_smoke.py stub migration_watchdog.* in
sys.modules at module level. This pollutes the import cache for tests that
run after them in the same process (test_automation_fixtures.py integration
tests). Run these groups separately:

    pytest tests/test_currency_fixtures.py tests/test_automation_fixtures.py
    pytest tests/test_main_pipeline.py
    pytest tests/test_pr_smoke.py

Or use pytest-forked for full isolation:

    pytest --forked tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

# tests/ is inside the package directory (migration_watchdog/).
# Its grandparent is the directory that should be on sys.path so that
# `import migration_watchdog` resolves to the package, not the bare module.
_TESTS_DIR = Path(__file__).parent          # .../migration_watchdog/tests/
_PACKAGE_DIR = _TESTS_DIR.parent            # .../migration_watchdog/  (symlink target)
_PARENT_DIR = _PACKAGE_DIR.parent           # .../plan-3ef5b6d1-outer/

# Insert parent so migration_watchdog resolves as a package.
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

# Remove the package dir itself if pytest added it, to prevent the
# "migration_watchdog is not a package" collision.
for _p in [str(_PACKAGE_DIR), str(_TESTS_DIR)]:
    while _p in sys.path:
        sys.path.remove(_p)
