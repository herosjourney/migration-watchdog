"""Coverage Assessor for the Scenario Simulation Agent.

For each design-ref file loaded in an execution trace, assesses whether the
file actually covers the persona's specific stack. Uses Claude via Bedrock
(same pattern as ``currency_auditor.py`` and ``analysis_agent.py``).

One LLM call is made per (persona, design-ref file) pair. Files that are
listed in ``trace.files_missing`` receive a ``missing_file`` result with
``confidence=1.0`` — no LLM call is needed for those.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from migration_watchdog.models import RepoContent
from migration_watchdog.path_tracer import ExecutionTrace
from migration_watchdog.persona_library import Persona

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CoverageResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class CoverageResult:
    """Result of assessing whether a design-ref file covers a persona's stack.

    ``coverage`` values:
    - ``"adequate"`` — file covers the persona's stack sufficiently.
    - ``"gap"`` — file exists but does not address specific elements of the
      persona's stack.
    - ``"intentional_deferral"`` — gap is explicitly documented in the file
      (e.g., "contact your AWS account team", "this is out of scope").
    - ``"missing_file"`` — the referenced file does not exist in the repo.
    - ``"unverified"`` — LLM call failed or response could not be parsed.
    """

    persona_id: str
    file_path: str
    coverage: str          # "adequate" | "gap" | "intentional_deferral" | "missing_file" | "unverified"
    gaps: list[str] = field(default_factory=list)        # specific elements not covered
    confidence: float = 0.0                              # 0.0-1.0; findings suppressed below threshold
    suggested_additions: list[str] = field(default_factory=list)  # concrete suggestions


# ---------------------------------------------------------------------------
# LLM assessment prompt
# ---------------------------------------------------------------------------

_ASSESSMENT_SYSTEM_PROMPT = """\
You are assessing whether a migration plugin reference file provides adequate guidance
for a specific startup's technical stack.

You will be given:
1. A persona description (the startup's stack)
2. The content of a design-ref file from the migration plugin

Your task: determine whether this file provides adequate guidance for this startup.

Output a JSON object with these fields:
{
  "coverage": "adequate" | "gap" | "intentional_deferral",
  "gaps": ["list of specific elements not covered"],
  "confidence": 0.0-1.0,
  "suggested_additions": ["concrete suggestions for what to add to the file"]
}

Rules:
- "adequate": the file covers the startup's provider, models, frameworks, and integration pattern
- "gap": the file exists but is missing guidance for specific elements of this startup's stack
- "intentional_deferral": the gap is explicitly documented in the file (e.g., "contact your AWS account team", "this is out of scope")
- confidence: how confident you are in your assessment (0.0-1.0)
- gaps: list specific missing elements (e.g., "no guidance for Anthropic SDK client initialization", "missing Claude model ID mapping")
- suggested_additions: concrete, actionable suggestions

Be conservative: only flag a gap if the missing guidance would materially affect the migration outcome.
Do NOT flag gaps for minor omissions or edge cases.
"""


# ---------------------------------------------------------------------------
# Path prefixes used to locate design-ref files in the repo snapshot
# ---------------------------------------------------------------------------

_DESIGN_REFS_PREFIX = (
    "migrate/plugins/migration-to-aws/skills/gcp-to-aws/"
    "references/design-refs/"
)
_PHASES_DESIGN_PREFIX = (
    "migrate/plugins/migration-to-aws/skills/gcp-to-aws/"
    "references/phases/design/"
)


# ---------------------------------------------------------------------------
# CoverageAssessor
# ---------------------------------------------------------------------------


class CoverageAssessor:
    """Assesses whether design-ref files cover a persona's specific stack.

    For each file in the execution trace, makes one LLM call (Claude via
    Bedrock) to determine whether the file provides adequate guidance for the
    persona's stack. Files listed in ``trace.files_missing`` receive a
    ``missing_file`` result without an LLM call.
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-opus-4-7",
        region_name: str = "us-east-1",
    ) -> None:
        self._model_id = model_id
        self._region_name = region_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def assess(
        self,
        persona: Persona,
        trace: ExecutionTrace,
        repo_content: RepoContent,
    ) -> list[CoverageResult]:
        """Assess coverage for all design-ref files in the execution trace.

        Missing files receive a ``missing_file`` result with ``confidence=1.0``
        without an LLM call. Files that were loaded are assessed via LLM.

        Args:
            persona: The startup persona being assessed.
            trace: The execution trace produced by ``PathTracer.trace()``.
            repo_content: Snapshot of the repo at scan time.

        Returns:
            List of ``CoverageResult`` objects, one per design-ref file
            referenced in the trace (both loaded and missing).
        """
        results: list[CoverageResult] = []

        # ------------------------------------------------------------------ #
        # Handle missing files first — no LLM call needed                     #
        # ------------------------------------------------------------------ #
        for missing_file in trace.files_missing:
            results.append(
                CoverageResult(
                    persona_id=persona.id,
                    file_path=missing_file,
                    coverage="missing_file",
                    gaps=[f"File '{missing_file}' does not exist in the repository"],
                    confidence=1.0,
                    suggested_additions=[
                        f"Create {missing_file} with guidance for "
                        f"{persona.ai_stack.provider or 'the detected'} migrations"
                    ],
                )
            )

        # ------------------------------------------------------------------ #
        # Assess files that were loaded                                        #
        # ------------------------------------------------------------------ #
        for ref in trace.design_refs_loaded:
            if ref in trace.files_missing:
                # Already handled above
                continue

            content = self._find_file_content(ref, repo_content)
            if content is None:
                logger.warning(
                    "CoverageAssessor: content not found for '%s' in repo snapshot; skipping",
                    ref,
                )
                continue

            result = await self._assess_file(persona, ref, content)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _assess_file(
        self, persona: Persona, file_path: str, content: str
    ) -> CoverageResult:
        """Make one LLM call (with retry) to assess whether a file covers the persona's stack.

        Wraps the Bedrock Agent call in ``retry_with_backoff`` with up to 3
        retries and a 2-second base delay. If all retries are exhausted,
        returns a result with ``coverage="unverified"`` and ``confidence=0.0``.

        Args:
            persona: The startup persona being assessed.
            file_path: The filename of the design-ref file (without path prefix).
            content: The markdown content of the design-ref file.

        Returns:
            A ``CoverageResult`` parsed from the LLM response. On exhausted
            retries, returns a result with ``coverage="unverified"`` and
            ``confidence=0.0``.
        """
        from migration_watchdog.retry import retry_with_backoff

        user_prompt = (
            f"Persona: {persona.id}\n"
            f"Description: {persona.description}\n"
            f"AI Provider: {persona.ai_stack.provider}\n"
            f"Models: {persona.ai_stack.models}\n"
            f"Frameworks: {persona.ai_stack.frameworks}\n"
            f"Integration Pattern: {persona.ai_stack.integration_pattern}\n"
            f"Gateway Type: {persona.ai_stack.gateway_type}\n"
            f"Is Agentic: {persona.agentic_profile.is_agentic}\n"
            f"Agentic Framework: {persona.agentic_profile.framework}\n"
            f"\n"
            f"File: {file_path}\n"
            f"Content:\n"
            f"```markdown\n"
            f"{content[:6000]}\n"
            f"```\n"
            f"\n"
            f"Assess whether this file provides adequate guidance for this startup's stack."
        )

        async def _call() -> CoverageResult:
            from strands import Agent  # lazy import — not available in test env
            from strands.models.bedrock import BedrockModel  # lazy import

            model = BedrockModel(
                model_id=self._model_id,
                region_name=self._region_name,
                max_tokens=2000,
            )
            agent = Agent(model=model, system_prompt=_ASSESSMENT_SYSTEM_PROMPT)
            response_text = str(agent(user_prompt))
            return self._parse_response(response_text, persona.id, file_path)

        try:
            return await retry_with_backoff(_call, max_retries=3, base_delay=2.0)
        except Exception:
            logger.exception(
                "CoverageAssessor: all retries exhausted for persona='%s' file='%s'",
                persona.id,
                file_path,
            )
            return CoverageResult(
                persona_id=persona.id,
                file_path=file_path,
                coverage="unverified",
                gaps=[],
                confidence=0.0,
                suggested_additions=[],
            )

    def _parse_response(
        self, response_text: str, persona_id: str, file_path: str
    ) -> CoverageResult:
        """Parse the LLM JSON response into a ``CoverageResult``.

        Handles both bare JSON objects and JSON embedded in markdown code
        fences. On any parse failure, returns a result with
        ``coverage="unverified"`` and ``confidence=0.0``.

        Args:
            response_text: Raw text returned by the LLM.
            persona_id: ID of the persona being assessed (for the result).
            file_path: Path of the file being assessed (for the result).

        Returns:
            A ``CoverageResult`` populated from the parsed JSON, or an
            ``"unverified"`` result on failure.
        """
        text = response_text.strip()

        # Strip markdown code fences if present
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()

        # Find the first '{' to handle any leading prose
        brace_idx = text.find("{")
        if brace_idx != -1:
            text = text[brace_idx:]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "CoverageAssessor: failed to parse LLM JSON for persona='%s' file='%s'; "
                "response preview: %r",
                persona_id,
                file_path,
                response_text[:200],
            )
            return CoverageResult(
                persona_id=persona_id,
                file_path=file_path,
                coverage="unverified",
                gaps=[],
                confidence=0.0,
                suggested_additions=[],
            )

        if not isinstance(parsed, dict):
            logger.warning(
                "CoverageAssessor: LLM response for persona='%s' file='%s' is not a JSON object",
                persona_id,
                file_path,
            )
            return CoverageResult(
                persona_id=persona_id,
                file_path=file_path,
                coverage="unverified",
                gaps=[],
                confidence=0.0,
                suggested_additions=[],
            )

        coverage = parsed.get("coverage", "unverified")
        gaps = parsed.get("gaps") or []
        confidence = float(parsed.get("confidence", 0.0))
        suggested_additions = parsed.get("suggested_additions") or []

        # Validate coverage value
        valid_coverage_values = {"adequate", "gap", "intentional_deferral"}
        if coverage not in valid_coverage_values:
            logger.warning(
                "CoverageAssessor: unexpected coverage value %r for persona='%s' file='%s'; "
                "treating as unverified",
                coverage,
                persona_id,
                file_path,
            )
            coverage = "unverified"
            confidence = 0.0

        return CoverageResult(
            persona_id=persona_id,
            file_path=file_path,
            coverage=coverage,
            gaps=gaps if isinstance(gaps, list) else [],
            confidence=max(0.0, min(1.0, confidence)),
            suggested_additions=suggested_additions if isinstance(suggested_additions, list) else [],
        )

    def _find_file_content(
        self, ref: str, repo_content: RepoContent
    ) -> str | None:
        """Find the content of a design-ref file in the repo snapshot.

        Tries both the ``design-refs/`` and ``phases/design/`` path prefixes.

        Args:
            ref: The filename of the design-ref (without path prefix).
            repo_content: Snapshot of the repo at scan time.

        Returns:
            The file content string, or ``None`` if not found under either
            prefix.
        """
        design_refs_path = _DESIGN_REFS_PREFIX + ref
        phases_design_path = _PHASES_DESIGN_PREFIX + ref

        if design_refs_path in repo_content.files:
            return repo_content.files[design_refs_path]
        if phases_design_path in repo_content.files:
            return repo_content.files[phases_design_path]

        return None
