"""Payload size management for auditor findings.

Manages ``auditor_payload`` dict size so that DynamoDB items stay within
the 400 KB item limit.  Payloads that exceed 50 KB after two truncation
passes are written to S3 in full; the DynamoDB item receives a truncated
version plus an ``auditor_payload_s3_key`` pointer.

Two-phase design
----------------
- ``prepare_payload()`` — pure truncation, no I/O.
- ``persist_overflow()`` — S3 write, called only when overflow occurred.
- ``store_payload()`` — convenience wrapper that calls both.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


class PayloadStore:
    """Manages auditor_payload size.  Overflows to S3 when > 50 KB.

    Two-phase design:
    - ``prepare_payload()`` — pure truncation, no I/O.
    - ``persist_overflow()`` — S3 write, called only when overflow occurred.
    - ``store_payload()`` — convenience wrapper that calls both.
    """

    MAX_PAYLOAD_BYTES = 50 * 1024  # 50 KB

    LARGE_FIELDS = [
        "claim_text",
        "action_text",
        "line_context",
        "verification_query",
        "partial_gap_narrative",
    ]

    TRUNCATE_RULES: dict[str, int] = {
        "claim_text": 500,
        "action_text": 500,
        "line_context": 500,
        "verification_query": 500,
        "partial_gap_narrative": 1000,
    }

    def __init__(self, s3_client=None, bucket_name: str | None = None) -> None:
        """Initialise the store.

        Args:
            s3_client: A boto3 S3 client (or compatible object).  May be
                ``None`` when S3 overflow is not expected (e.g. in tests).
            bucket_name: Name of the S3 bucket for overflow payloads.  When
                ``None``, the value of the ``WATCHDOG_PAYLOAD_BUCKET``
                environment variable is used.
        """
        self._s3 = s3_client
        self._bucket = bucket_name or os.environ.get("WATCHDOG_PAYLOAD_BUCKET")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_payload(
        self, finding_id: str, payload_dict: dict
    ) -> tuple[dict, str | None]:
        """Pure truncation — no I/O.

        Returns ``(payload_to_store, full_json_or_None)``.

        - If the serialized payload is within the 50 KB limit, returns
          ``(payload_dict, None)`` unchanged.
        - If it exceeds the limit, performs two truncation passes and
          returns ``(truncated_dict_with_s3_key, full_json_string)``.

        Two-pass truncation:

        1. Truncate known large text fields per ``TRUNCATE_RULES``.
        2. If still > 50 KB, strip optional large fields entirely.

        Raises:
            ValueError: If the payload still exceeds 50 KB after both
                truncation passes (e.g. due to huge non-string blobs).
                Callers in ``main.py`` must catch this, log the
                ``finding_id``, append a structured entry to
                ``ScanRun.partial_source_failures``, and skip persisting
                that finding rather than crashing the entire run.
        """
        serialized = json.dumps(payload_dict, sort_keys=True)
        if len(serialized.encode()) <= self.MAX_PAYLOAD_BYTES:
            return payload_dict, None

        # Payload exceeds limit — capture the full JSON before truncating.
        full_json = serialized
        s3_key = f"payloads/{finding_id}.json"

        # Work on a shallow copy so the caller's dict is not mutated.
        truncated = dict(payload_dict)

        # Pass 1: truncate known large text fields per TRUNCATE_RULES.
        for field, max_len in self.TRUNCATE_RULES.items():
            if field in truncated and isinstance(truncated[field], str):
                if len(truncated[field]) > max_len:
                    truncated[field] = truncated[field][:max_len]

        # Pass 2: if still too large, strip optional large fields entirely.
        if len(json.dumps(truncated, sort_keys=True).encode()) > self.MAX_PAYLOAD_BYTES:
            for field in self.LARGE_FIELDS:
                truncated.pop(field, None)

        # Add the S3 pointer so the consumer can fetch the full payload.
        truncated["auditor_payload_s3_key"] = s3_key

        # Final size check — raise rather than assert so it survives python -O.
        final_size = len(json.dumps(truncated, sort_keys=True).encode())
        if final_size > self.MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"payload still too large after truncation: {final_size} bytes "
                f"(finding_id={finding_id})"
            )

        return truncated, full_json

    async def persist_overflow(self, finding_id: str, full_json: str) -> None:
        """Write the full payload to S3.

        Called only when ``prepare_payload()`` returned a non-``None``
        ``full_json``.

        Two-phase commit: the S3 write happens **before** the DynamoDB item
        is persisted.  The caller (``store_payload``) must call this before
        the Finding is saved to DynamoDB so the S3 object exists before the
        pointer is written.  If this raises, the caller must strip
        ``auditor_payload_s3_key`` from the truncated payload before
        persisting.

        Args:
            finding_id: The finding ID used to derive the S3 key.
            full_json: The full JSON string to write.

        Raises:
            Exception: Any exception raised by the underlying S3 client is
                propagated to the caller.
        """
        s3_key = f"payloads/{finding_id}.json"
        self._s3.put_object(
            Bucket=self._bucket,
            Key=s3_key,
            Body=full_json.encode(),
            ContentType="application/json",
        )

    async def store_payload(self, finding_id: str, payload_dict: dict) -> dict:
        """Convenience wrapper: prepare + persist if needed.

        Returns the payload dict that should be stored on the Finding
        (either the original dict unchanged, or the truncated dict with
        an ``auditor_payload_s3_key`` pointer).

        Two-phase commit order:

        1. ``prepare_payload()`` — pure, no I/O, adds ``s3_key`` to the
           truncated dict.
        2. ``persist_overflow()`` — S3 write (before DynamoDB).
        3. Return truncated dict (caller persists to DynamoDB after this
           returns).

        If ``persist_overflow()`` raises, ``auditor_payload_s3_key`` is
        stripped from the truncated dict and the truncated dict is returned
        without the pointer so the DynamoDB item is not left with a dangling
        S3 reference.

        If ``prepare_payload()`` raises ``ValueError`` (payload still too
        large after both truncation passes — e.g. huge non-string blobs),
        the exception propagates to the caller.  ``main.py`` must catch it,
        log the ``finding_id``, append a structured entry to
        ``ScanRun.partial_source_failures`` (e.g.
        ``{"type": "payload_too_large", "finding_id": finding_id}``), and
        skip persisting that finding rather than crashing the entire run.

        Args:
            finding_id: Unique identifier for the finding.
            payload_dict: The auditor payload dict to store.

        Returns:
            The payload dict to assign to ``Finding.auditor_payload``.

        Raises:
            ValueError: Propagated from ``prepare_payload()`` when the
                payload cannot be reduced to ≤ 50 KB.
        """
        truncated, full_json = self.prepare_payload(finding_id, payload_dict)

        if full_json is not None:
            try:
                await self.persist_overflow(finding_id, full_json)
            except Exception:
                logger.exception(
                    "S3 overflow write failed for finding %s; "
                    "storing truncated payload without S3 key",
                    finding_id,
                )
                truncated.pop("auditor_payload_s3_key", None)

        return truncated
