"""Alias table for resolving human-readable model names to canonical Bedrock model IDs.

The table is loaded from a bundled ``alias_table.json`` file that ships with the
package.  It can be auto-synced from a :class:`~models.BedrockLifecycle` object
fetched by ``SourceFetcher`` without making any additional network calls.

Thread-safety note: read access (``resolve``, ``get_pending_review``) is safe to
call concurrently.  ``sync_from_bedrock_lifecycle`` is intended to be called once
per run before any concurrent reads begin.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import BedrockLifecycle

logger = logging.getLogger(__name__)

# Path to the bundled JSON file that lives next to this module.
_BUNDLED_JSON = os.path.join(os.path.dirname(__file__), "alias_table.json")


class AliasTable:
    """Loads from ``alias_table.json`` bundled with the package.

    The in-memory representation mirrors the JSON structure:

    .. code-block:: json

        {
          "version": "1.0",
          "last_synced": "YYYY-MM-DD",
          "entries": { "<human name>": "<canonical model id>", ... },
          "pending_review": [ { ... }, ... ]
        }

    All public methods operate on the in-memory copy; the file on disk is
    **never** written back automatically (a human must review and commit
    changes).
    """

    def __init__(self, json_path: str | None = None) -> None:
        """Load the alias table from *json_path* or the bundled default.

        Parameters
        ----------
        json_path:
            Optional path to an alternative ``alias_table.json`` file.  When
            ``None`` the bundled file at ``alias_table.json`` (next to this
            module) is used.

        Raises
        ------
        FileNotFoundError
            If the resolved path does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.
        """
        path = json_path or _BUNDLED_JSON
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        self._version: str = data.get("version", "1.0")
        self._last_synced: str = data.get("last_synced", "")
        # Normalise keys to lower-case for case-insensitive lookup while
        # preserving the original casing in a parallel dict for display.
        raw_entries: dict[str, str] = data.get("entries", {})
        self._entries: dict[str, str] = {k.lower(): v for k, v in raw_entries.items()}
        # Keep original-cased keys so we can reconstruct the display form.
        self._original_keys: dict[str, str] = {k.lower(): k for k in raw_entries}
        self._pending_review: list[dict] = list(data.get("pending_review", []))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, name: str) -> tuple[str, bool]:
        """Return ``(canonical_id, found)`` for *name*.

        Lookup is case-insensitive.  When *name* is not found the raw value is
        returned unchanged, the miss is logged, and the name is appended to the
        ``pending_review`` list so a maintainer can add the mapping later.

        Parameters
        ----------
        name:
            Human-readable model name (e.g. ``"Claude Sonnet 4.6"``).

        Returns
        -------
        tuple[str, bool]
            ``(canonical_id, True)`` on a hit, ``(name, False)`` on a miss.
        """
        key = name.lower()
        if key in self._entries:
            return self._entries[key], True

        logger.warning(
            "AliasTable miss for %r — using raw value; adding to pending_review",
            name,
        )
        self._add_to_pending_review(
            {
                "type": "miss",
                "name": name,
                "detected_at": _utc_now(),
            }
        )
        return name, False

    def sync_from_bedrock_lifecycle(self, lifecycle: "BedrockLifecycle") -> None:
        """Auto-update the table from fetched Bedrock lifecycle data.

        Rules
        -----
        * **New unambiguous entries** (``model_name`` → ``model_id`` mapping
          that does not conflict with any existing entry) are added directly.
        * **Conflicts** — same ``model_name`` already maps to a *different*
          ``model_id`` — are flagged to ``pending_review`` and the existing
          entry is **not** overwritten.
        * **Existing entries** that match exactly are silently skipped (no
          duplicate pending_review entries are created).
        * ``last_synced`` is updated to today's date (UTC) after processing.

        Parameters
        ----------
        lifecycle:
            A :class:`~models.BedrockLifecycle` instance populated by
            ``SourceFetcher``.
        """
        for entry in lifecycle.models:
            model_name: str = entry.model_name
            model_id: str = entry.model_id

            if not model_name or not model_id:
                logger.debug(
                    "sync_from_bedrock_lifecycle: skipping entry with empty name or id"
                )
                continue

            key = model_name.lower()

            if key in self._entries:
                existing_id = self._entries[key]
                if existing_id == model_id:
                    # Exact match — nothing to do.
                    continue
                # Conflict: same name, different ID.
                logger.warning(
                    "AliasTable conflict for %r: existing=%r, new=%r — flagging for review",
                    model_name,
                    existing_id,
                    model_id,
                )
                self._add_to_pending_review(
                    {
                        "type": "conflict",
                        "name": model_name,
                        "existing_id": existing_id,
                        "new_id": model_id,
                        "detected_at": _utc_now(),
                    }
                )
            else:
                # New unambiguous entry — add it.
                logger.info(
                    "AliasTable: adding new entry %r → %r from Bedrock lifecycle",
                    model_name,
                    model_id,
                )
                self._entries[key] = model_id
                self._original_keys[key] = model_name

        self._last_synced = _utc_now()[:10]  # YYYY-MM-DD

    def get_pending_review(self) -> list[dict]:
        """Return a copy of the pending-review list.

        Each item is a dict with at minimum ``"type"`` and ``"name"`` keys.
        Callers should not mutate the returned list.
        """
        return list(self._pending_review)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_to_pending_review(self, item: dict) -> None:
        """Append *item* to pending_review, avoiding exact duplicates."""
        if item not in self._pending_review:
            self._pending_review.append(item)


def _utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string (seconds precision)."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
