"""Gap Classifier for the Scenario Simulation Agent.

Converts ``CoverageResult`` objects produced by the ``CoverageAssessor`` into
typed, severity-rated ``Finding`` objects.  Deduplicates findings across
personas and fetches authoritative AWS documentation URLs for each gap type.

Pipeline position:
    PersonaLibrary → PathTracer → CoverageAssessor → **GapClassifier** → Findings
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime

from migration_watchdog.coverage_assessor import CoverageResult
from migration_watchdog.models import Finding, RiskLevel
from migration_watchdog.path_tracer import ExecutionTrace
from migration_watchdog.source_fetcher import AwsDocsSearcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task 4.1 — Gap type → (severity, gap_type_label) mapping
# ---------------------------------------------------------------------------

_GAP_SEVERITY: dict[str, tuple[RiskLevel, str]] = {
    "missing_file":           (RiskLevel.HIGH,   "missing_file"),
    "routing_error":          (RiskLevel.HIGH,   "routing_error"),
    "coverage_gap_ai":        (RiskLevel.HIGH,   "coverage_gap"),
    "coverage_gap_framework": (RiskLevel.MEDIUM, "coverage_gap"),
    "missing_question":       (RiskLevel.MEDIUM, "missing_question"),
    "auto_skip_error":        (RiskLevel.MEDIUM, "auto_skip_error"),
    "contradictory_guidance": (RiskLevel.MEDIUM, "contradictory_guidance"),
    "coverage_gap_minor":     (RiskLevel.LOW,    "coverage_gap"),
    "intentional_deferral":   (RiskLevel.LOW,    "intentional_deferral"),
}


# ---------------------------------------------------------------------------
# GapClassifier
# ---------------------------------------------------------------------------


class GapClassifier:
    """Classifies ``CoverageResult`` objects into typed, severity-rated ``Finding`` objects.

    Args:
        run_id: Identifier for the current scan run (propagated to each Finding).
        confidence_threshold: Results with ``confidence < confidence_threshold``
            are suppressed (no Finding created).  Default: 0.7.
        docs_searcher: Optional ``AwsDocsSearcher`` instance.  A new instance is
            created if not provided.
    """

    def __init__(
        self,
        run_id: str = "",
        confidence_threshold: float = 0.7,
        docs_searcher: AwsDocsSearcher | None = None,
    ) -> None:
        self._run_id = run_id
        self._confidence_threshold = confidence_threshold
        self._docs_searcher = docs_searcher or AwsDocsSearcher()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        results: list[CoverageResult],
        traces: list[ExecutionTrace],
    ) -> list[Finding]:
        """Classify coverage results into deduplicated ``Finding`` objects.

        Steps:
        1. Skip results with ``coverage in ("adequate", "unverified")``.
        2. Suppress results where ``confidence < confidence_threshold``.
        3. Map each result to a ``(gap_type_key, RiskLevel)`` pair.
        4. Fetch an authoritative AWS documentation URL; suppress if none found.
        5. Build a ``Finding`` for each result.
        6. Deduplicate findings by ``(gap_type, affected_files_frozenset)``.

        Args:
            results: Coverage results from ``CoverageAssessor.assess()``.
            traces: Execution traces (currently unused but available for future
                routing-error detection).

        Returns:
            Deduplicated list of ``Finding`` objects.
        """
        findings: list[Finding] = []

        for result in results:
            # Step 1 — skip non-actionable coverage values
            if result.coverage in ("adequate", "unverified"):
                continue

            # Step 2 — Task 4.6: confidence threshold
            if result.confidence < self._confidence_threshold:
                logger.info(
                    "Suppressing finding for %s/%s: confidence %.2f < threshold %.2f",
                    result.persona_id,
                    result.file_path,
                    result.confidence,
                    self._confidence_threshold,
                )
                continue

            # Step 3 — classify
            gap_type_key, risk_level = self._classify_result(result)
            if gap_type_key is None:
                continue

            # Step 4 — fetch source URL
            source_url = self._fetch_source_url(gap_type_key, " ".join(result.gaps))
            if source_url is None:
                logger.warning(
                    "No source URL found for gap_type=%s; suppressing finding",
                    gap_type_key,
                )
                continue

            # Step 5 — build Finding
            finding = Finding(
                finding_id=self._make_finding_id(gap_type_key, result.file_path),
                run_id=self._run_id,
                risk_level=risk_level,
                category="scenario_gap",
                title=self._make_title(gap_type_key, result.file_path, result.persona_id),
                description=self._make_description(result, gap_type_key),
                affected_files=[result.file_path],
                proposed_changes={result.file_path: "\n".join(result.suggested_additions)},
                source_urls=[source_url],
                scan_timestamp=datetime.utcnow().isoformat(),
                status="pending",
                auditor_payload={
                    "gap_type": gap_type_key,
                    "persona_ids": [result.persona_id],
                    "coverage": result.coverage,
                    "confidence": result.confidence,
                    "gaps": result.gaps,
                },
                finding_schema_version="scenario_gap/1.0",
            )
            findings.append(finding)

        # Step 6 — deduplicate
        return self._deduplicate(findings)

    # ------------------------------------------------------------------
    # Task 4.2 — _classify_result
    # ------------------------------------------------------------------

    def _classify_result(self, result: CoverageResult) -> tuple[str | None, RiskLevel | None]:
        """Map a ``CoverageResult`` to a ``(gap_type_key, RiskLevel)`` pair.

        Returns ``(None, None)`` when no finding should be created (e.g. for
        ``"adequate"`` or ``"unverified"`` coverage values).

        Args:
            result: A single coverage assessment result.

        Returns:
            A ``(gap_type_key, RiskLevel)`` tuple, or ``(None, None)``.
        """
        if result.coverage == "missing_file":
            return ("missing_file", RiskLevel.HIGH)

        if result.coverage == "intentional_deferral":
            return ("intentional_deferral", RiskLevel.LOW)

        if result.coverage == "gap":
            # Determine severity based on what's missing.
            # AI provider gaps are HIGH; framework gaps are MEDIUM; others are LOW.
            gaps_text = " ".join(result.gaps).lower()
            if any(
                kw in gaps_text
                for kw in ["provider", "sdk", "api", "model", "bedrock", "anthropic", "openai", "gemini"]
            ):
                return ("coverage_gap_ai", RiskLevel.HIGH)
            elif any(
                kw in gaps_text
                for kw in ["framework", "langchain", "langgraph", "crewai", "autogen", "strands"]
            ):
                return ("coverage_gap_framework", RiskLevel.MEDIUM)
            else:
                return ("coverage_gap_minor", RiskLevel.LOW)

        if result.coverage in ("adequate", "unverified"):
            return (None, None)

        return (None, None)

    # ------------------------------------------------------------------
    # Task 4.3 — _deduplicate
    # ------------------------------------------------------------------

    def _deduplicate(self, findings: list[Finding]) -> list[Finding]:
        """Merge findings with the same ``(gap_type, affected_files_frozenset)`` key.

        When two findings share the same key, their ``auditor_payload["persona_ids"]``
        lists are merged (union, sorted).  The first occurrence's other fields are
        preserved.

        Args:
            findings: Raw (possibly duplicated) list of findings.

        Returns:
            Deduplicated list of findings.
        """
        seen: dict[tuple, Finding] = {}

        for finding in findings:
            gap_type = finding.auditor_payload.get("gap_type", "") if finding.auditor_payload else ""
            key = (gap_type, frozenset(finding.affected_files))

            if key in seen:
                # Merge persona IDs
                existing = seen[key]
                existing_personas = set(existing.auditor_payload.get("persona_ids", []))
                new_personas = set(finding.auditor_payload.get("persona_ids", []))
                existing.auditor_payload["persona_ids"] = sorted(existing_personas | new_personas)
            else:
                seen[key] = finding

        return list(seen.values())

    # ------------------------------------------------------------------
    # Task 4.4 — _fetch_source_url
    # ------------------------------------------------------------------

    def _fetch_source_url(self, gap_type: str, context: str) -> str | None:
        """Return an authoritative AWS documentation URL for the given gap type.

        Pre-defined URLs are returned immediately for common gap types to avoid
        unnecessary network calls.  For unknown gap types, ``AwsDocsSearcher``
        is used as a fallback.  Returns ``None`` if no URL can be found (the
        caller will suppress the finding).

        Args:
            gap_type: The gap type key (e.g. ``"missing_file"``, ``"coverage_gap_ai"``).
            context: Human-readable context string used as the search query fallback.

        Returns:
            An ``https://docs.aws.amazon.com/…`` URL, or ``None``.
        """
        _KNOWN_URLS: dict[str, str] = {
            "missing_file": "https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html",
            "routing_error": "https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html",
            "coverage_gap_ai": "https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html",
            "coverage_gap_framework": "https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html",
            "intentional_deferral": "https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html",
        }

        if gap_type in _KNOWN_URLS:
            return _KNOWN_URLS[gap_type]

        # Fall back to search for unknown gap types
        try:
            result = self._docs_searcher.search_and_fetch(
                f"AWS Bedrock migration {context[:100]}"
            )
            if result:
                urls = re.findall(r"https://docs\.aws\.amazon\.com/[^\s\"']+", result)
                return urls[0] if urls else None
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_finding_id(self, gap_type: str, file_path: str) -> str:
        """Compute a deterministic SHA-256 finding ID.

        Uses the same pattern as ``security_auditor.py``:
        ``SHA-256(gap_type|file_path)[:32]``
        """
        raw = f"{gap_type}|{file_path}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _make_title(self, gap_type: str, file_path: str, persona_id: str) -> str:
        """Build a human-readable finding title."""
        filename = file_path.split("/")[-1]
        labels: dict[str, str] = {
            "missing_file": f"Missing design-ref file: {filename}",
            "routing_error": f"Routing error for {filename}",
            "coverage_gap_ai": f"AI provider coverage gap in {filename}",
            "coverage_gap_framework": f"Framework coverage gap in {filename}",
            "coverage_gap_minor": f"Minor coverage gap in {filename}",
            "intentional_deferral": f"Intentional deferral in {filename}",
            "missing_question": f"Missing question for {filename}",
            "auto_skip_error": f"Auto-skip error in {filename}",
            "contradictory_guidance": f"Contradictory guidance in {filename}",
        }
        return labels.get(gap_type, f"Coverage gap in {filename}")

    def _make_description(self, result: CoverageResult, gap_type: str) -> str:
        """Build a human-readable finding description."""
        lines: list[str] = [
            f"Persona `{result.persona_id}` triggered a `{gap_type}` gap on `{result.file_path}`.",
        ]
        if result.gaps:
            lines.append("\n**Specific gaps identified:**")
            for gap in result.gaps:
                lines.append(f"- {gap}")
        if result.suggested_additions:
            lines.append("\n**Suggested additions:**")
            for suggestion in result.suggested_additions:
                lines.append(f"- {suggestion}")
        return "\n".join(lines)
