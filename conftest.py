"""pytest configuration for the migration_watchdog test suite.

Ensures that `migration_watchdog` resolves as a proper package by inserting
the *parent* of this directory into sys.path. This makes
`from migration_watchdog.payload_store import PayloadStore` work correctly
when running pytest from within the flat module directory.

Without this, Python sees the current directory's __init__.py as the
`migration_watchdog` module (not a package), causing:
    ModuleNotFoundError: No module named 'migration_watchdog.payload_store';
    'migration_watchdog' is not a package
"""

from __future__ import annotations

import sys
from pathlib import Path

# The package directory is the directory containing this conftest.py.
# Its parent is the directory that should be on sys.path so that
# `import migration_watchdog` resolves to this directory as a package.
_PACKAGE_DIR = Path(__file__).parent          # .../plan-3ef5b6d1.../
_PARENT_DIR = _PACKAGE_DIR.parent             # .../plan-3ef5b6d1-outer/

# Insert parent first so `migration_watchdog` resolves as a package,
# not as the bare __init__.py module.
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

# Remove the package dir itself from sys.path if it was added by pytest,
# to prevent the "not a package" collision.
if str(_PACKAGE_DIR) in sys.path:
    sys.path.remove(str(_PACKAGE_DIR))
