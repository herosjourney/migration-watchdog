"""Automation Auditor for the Migration Plugin Watchdog.

Extracts manual action instructions from Reference_Files, looks up
programmatic CLI/API equivalents, assesses whether generated scripts
already cover those actions, and applies safety/judgment criteria before
recommending automation. Produces Finding objects with
category="automation_gap".
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed action_type values
# ---------------------------------------------------------------------------
ALLOWED_ACTION_TYPES = frozenset(
    {
        "console_navigation",
        "copy_paste",
        "manual_calculation",
        "form_submission",
        "file_edit",
        "browser_action",
    }
)

# ---------------------------------------------------------------------------
# ManualAction dataclass
# ---------------------------------------------------------------------------


@dataclass
class ManualAction:
    """A single manual action instruction extracted from a Reference_File.

    Fields
    ------
    action_text : str
        Verbatim instruction text directing the user to perform an action.
    action_type : str
        One of the 6 allowed action types (see ALLOWED_ACTION_TYPES).
    context : str
        Surrounding paragraph or section providing context for the action.
    generated_artifact : str | None
        Path or name of the generated script/artifact that should cover this
        action, or None when no artifact is associated.
    action_fingerprint : str
        SHA-256(normalize(action_text)|source_file|action_type) — stable
        dedupe key across re-runs.
    """

    action_text: str
    action_type: str
    context: str
    generated_artifact: str | None
    action_fingerprint: str


# ---------------------------------------------------------------------------
# CLIResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class CLIResult:
    """The result of a CLI lookup for a ManualAction.

    Fields
    ------
    cli_equivalent : str | None
        AWS CLI command template with ``{placeholder}`` tokens, or ``None``
        when no programmatic equivalent is available.
    placeholder_list : list[str]
        Ordered list of placeholder names that appear in ``cli_equivalent``.
        Empty when ``cli_equivalent`` is ``None``.
    reference_url : str | None
        URL to the AWS CLI reference documentation for this command, or
        ``None`` when no match was found.
    cli_note : str | None
        Human-readable note.  Set to ``"No programmatic equivalent
        available"`` when ``cli_equivalent`` is ``None``; ``None`` otherwise.
    """

    cli_equivalent: str | None
    placeholder_list: list[str] = field(default_factory=list)
    reference_url: str | None = None
    cli_note: str | None = None


# ---------------------------------------------------------------------------
# CLILookup class
# ---------------------------------------------------------------------------

# Bundled index lives next to this module file.
_DEFAULT_INDEX_PATH = Path(__file__).parent / "cli_command_index.json"

# Sentinel returned when no match is found.
_NO_MATCH = CLIResult(
    cli_equivalent=None,
    placeholder_list=[],
    reference_url=None,
    cli_note="No programmatic equivalent available",
)


class CLILookup:
    """Looks up CLI equivalents from a bundled command index.

    The index maps action keywords to CLI command templates.  Never makes
    network calls.  Never fabricates commands — every returned command
    template comes directly from the bundled ``cli_command_index.json``.

    Parameters
    ----------
    index_path : str | None
        Path to a custom ``cli_command_index.json``.  When ``None`` (the
        default) the bundled index packaged alongside this module is used.
    """

    def __init__(self, index_path: str | None = None) -> None:
        path = Path(index_path) if index_path is not None else _DEFAULT_INDEX_PATH
        with path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        self._entries: list[dict] = raw.get("entries", [])
        self._version: str = raw.get("version", "unknown")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, action: ManualAction) -> CLIResult:
        """Find a CLI equivalent by matching action keywords against the index.

        Matching strategy
        -----------------
        1. **Exact keyword match** — every keyword token in an entry's
           ``keywords`` list is checked against the lowercased
           ``action.action_text``.  The entry with the most keyword hits
           (and at least one hit) wins.
        2. **Fuzzy / partial match** — if no entry scores on exact tokens,
           individual words from ``action.action_text`` and
           ``action.action_type`` are checked against each entry's keyword
           strings using substring containment.
        3. **No match** — returns the sentinel ``CLIResult`` with
           ``cli_equivalent=None`` and
           ``cli_note="No programmatic equivalent available"``.

        Never fabricates.  Every returned command template is from the
        bundled index.

        Parameters
        ----------
        action : ManualAction
            The manual action to look up.

        Returns
        -------
        CLIResult
            Matched result, or the no-match sentinel.
        """
        action_text_lower = action.action_text.lower()
        action_type_lower = action.action_type.lower()

        # --- Pass 1: exact keyword token matching ---
        best_entry: dict | None = None
        best_score: int = 0

        for entry in self._entries:
            score = self._exact_score(entry, action_text_lower, action_type_lower)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None:
            return self._entry_to_result(best_entry)

        # --- Pass 2: fuzzy / partial matching ---
        for entry in self._entries:
            if self._fuzzy_matches(entry, action_text_lower, action_type_lower):
                return self._entry_to_result(entry)

        return _NO_MATCH

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _exact_score(
        self,
        entry: dict,
        action_text_lower: str,
        action_type_lower: str,
    ) -> int:
        """Return the number of keyword phrases from *entry* found in the action text.

        Each keyword phrase is checked as a whole substring (not split into
        individual words) against the lowercased action text.  The action
        type is also checked against the entry's ``action_types`` list for a
        small bonus (+1) to prefer type-specific entries, but only when at
        least one keyword already matched (to avoid false positives from
        action_type alone).
        """
        score = 0
        for kw in entry.get("keywords", []):
            if kw.lower() in action_text_lower:
                score += 1
        # Small bonus when action_type matches — only applied when keywords matched
        if score > 0 and action_type_lower in [at.lower() for at in entry.get("action_types", [])]:
            score += 1
        return score

    def _fuzzy_matches(
        self,
        entry: dict,
        action_text_lower: str,
        action_type_lower: str,
    ) -> bool:
        """Return True if any individual word from the action text appears in
        any keyword string of the entry, or if the action type matches.

        This is a looser check used only when exact matching finds nothing.
        """
        # Split action text into individual words (≥3 chars to avoid noise)
        words = {w for w in re.split(r"\W+", action_text_lower) if len(w) >= 3}
        words.add(action_type_lower)

        for kw in entry.get("keywords", []):
            kw_lower = kw.lower()
            for word in words:
                if word in kw_lower:
                    return True
        return False

    @staticmethod
    def _entry_to_result(entry: dict) -> CLIResult:
        """Convert a raw index entry dict to a ``CLIResult``."""
        cli_equivalent = entry.get("cli_equivalent")
        placeholder_list = list(entry.get("placeholder_list", []))
        reference_url = entry.get("reference_url")
        return CLIResult(
            cli_equivalent=cli_equivalent,
            placeholder_list=placeholder_list,
            reference_url=reference_url,
            cli_note=None,  # a match was found — no note needed
        )

# ---------------------------------------------------------------------------
# LLM extraction prompt (ActionExtractor)
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are the Automation Auditor Extraction Agent. Your job is to identify manual
action instructions in migration Reference_Files.

A manual action instruction is a directive that tells a human to:
- Navigate to a console, UI, or web page (console_navigation)
- Copy and paste text, commands, or values (copy_paste)
- Perform a manual calculation or derive a value by hand (manual_calculation)
- Fill in and submit a form (form_submission)
- Edit a file directly (file_edit)
- Perform an action in a browser (browser_action)

DO NOT extract:
- Conceptual explanations or architectural descriptions
- Informational notes or background context
- Passive observations ("the console shows ...")
- Conditional or hypothetical statements that are not directives

For each manual action you identify, output a JSON object with these fields:
  action_text   -- the verbatim instruction text (string, required, non-empty)
  action_type   -- one of the 6 allowed types listed above (string, required)
  context       -- the surrounding sentence or paragraph (string, required, non-empty)
  generated_artifact -- path/identifier of a generated script that may cover this
                        action, or null when none is associated (string or null)

Return a JSON array of these objects. If no manual actions are found, return [].
Do not include any text outside the JSON array.
"""


def _build_extraction_prompt(file_path: str, content: str) -> str:
    """Build the user message for the extraction LLM call."""
    return (
        f"## Reference_File: {file_path}\n\n"
        "Please extract all manual action instructions from the following content.\n\n"
        f"```\n{content}\n```\n\n"
        "Return a JSON array of manual action objects as described in your instructions."
    )


# ---------------------------------------------------------------------------
# ActionExtractor
# ---------------------------------------------------------------------------


class ActionExtractor:
    """Extracts manual action instructions from Reference_Files using an LLM.

    The LLM identifies instructions that direct users to navigate, click,
    enter values, search, or submit forms.  Conceptual explanations and
    architectural descriptions are excluded.

    Each extracted action receives a stable ``action_fingerprint`` computed
    as SHA-256(normalize(action_text)|source_file|action_type).
    """

    def __init__(self, model_id: str = "us.anthropic.claude-opus-4-7") -> None:
        self._model_id = model_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(self, file_path: str, content: str) -> list[ManualAction]:
        """Extract all manual action instructions from a Reference_File.

        Steps
        -----
        1. Detect migration context -- halt file if not found (returns []).
        2. Call LLM to identify candidate manual action instructions.
        3. Parse and validate the LLM response.
        4. Compute ``action_fingerprint`` for each candidate.
        5. Deduplicate by ``action_fingerprint`` (first occurrence wins).
        6. Return validated list of ManualAction objects.

        Parameters
        ----------
        file_path:
            Path of the Reference_File being processed (used in fingerprint
            computation and for logging).
        content:
            Full text content of the Reference_File.

        Returns
        -------
        list[ManualAction]
            Validated, deduplicated list of manual actions found in the file.
            Returns an empty list when no migration context is detected or
            when the LLM finds no manual actions.
        """
        if not self._has_migration_context(content):
            logger.info(
                "automation_auditor: no migration context found in %s -- skipping",
                file_path,
            )
            return []

        raw_candidates = await self._call_llm(file_path, content)
        actions = self._parse_and_validate(raw_candidates, file_path)
        return self._deduplicate(actions)

    # ------------------------------------------------------------------
    # Fingerprint computation
    # ------------------------------------------------------------------

    def _compute_fingerprint(
        self,
        action_text: str,
        source_file: str,
        action_type: str,
    ) -> str:
        """Compute SHA-256(normalize(action_text)|source_file|action_type).

        Normalization: lowercase + collapse all whitespace runs to a single
        space + strip leading/trailing whitespace.

        Parameters
        ----------
        action_text:
            The verbatim action instruction text.
        source_file:
            Path of the Reference_File that contains the action.
        action_type:
            One of the 6 allowed action type values.

        Returns
        -------
        str
            Lowercase hex-encoded SHA-256 digest.
        """
        normalized = re.sub(r"\s+", " ", action_text.lower().strip())
        raw = f"{normalized}|{source_file}|{action_type}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_migration_context(self, content: str) -> bool:
        """Return True when the file contains recognisable migration context.

        A file is considered a Reference_File with migration context when it
        contains at least one of the following signals:
        - A YAML/TOML front-matter block (``---`` or ``+++`` at the start)
        - Keywords commonly found in migration runbooks or AWS documentation

        This is intentionally permissive -- the goal is to skip files that are
        clearly not migration-related (e.g. pure configuration files, empty
        files) rather than to enforce strict schema validation.
        """
        if not content or not content.strip():
            return False

        lower = content.lower()

        # Front-matter block
        if content.lstrip().startswith(("---", "+++")):
            return True

        # Migration / cloud / AWS keywords
        migration_keywords = [
            "migration",
            "migrate",
            "aws",
            "cloud",
            "infrastructure",
            "runbook",
            "deploy",
            "terraform",
            "kubernetes",
            "eks",
            "ecs",
            "fargate",
            "lambda",
            "s3",
            "iam",
            "vpc",
            "rds",
            "bedrock",
        ]
        return any(kw in lower for kw in migration_keywords)

    async def _call_llm(self, file_path: str, content: str) -> list[dict]:
        """Call the LLM to extract candidate manual actions.

        Wraps the Bedrock agent call in retry_with_backoff (max 3 retries,
        2 s base delay).  Returns the parsed list of raw candidate dicts from
        the LLM response, or an empty list when all retries are exhausted.
        """
        from migration_watchdog.retry import retry_with_backoff

        user_message = _build_extraction_prompt(file_path, content)

        async def _call() -> list[dict]:
            from strands import Agent
            from strands.models.bedrock import BedrockModel

            model = BedrockModel(
                model_id=self._model_id,
                region_name="us-east-1",
                max_tokens=8000,
            )
            agent = Agent(
                model=model,
                system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            )
            # Strands Agent returns an AgentResult; str() gives the text content.
            response_text = str(agent(user_message))
            return self._parse_llm_response(response_text, file_path)

        try:
            return await retry_with_backoff(_call, max_retries=3, base_delay=2.0)
        except Exception:
            logger.exception(
                "automation_auditor: all retries exhausted for %s", file_path
            )
            return []

    def _parse_llm_response(self, response_text: str, file_path: str) -> list[dict]:
        """Parse the LLM response text into a list of raw candidate dicts.

        The LLM is instructed to return a JSON array.  This method extracts
        the first JSON array found in the response (to handle cases where the
        model emits a small preamble before the JSON).
        """
        stripped = response_text.strip()

        # First attempt: the entire response is a JSON array
        if stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass

        # Second attempt: find the first [...] block in the response
        match = re.search(r"\[.*\]", stripped, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(
            "automation_auditor: could not parse LLM response as JSON array for %s",
            file_path,
        )
        return []

    def _parse_and_validate(
        self, raw_candidates: list[dict], file_path: str
    ) -> list[ManualAction]:
        """Validate raw candidate dicts and build ManualAction objects.

        Validation rules:
        - ``action_text`` must be a non-empty string.
        - ``action_type`` must be one of the 6 allowed values.
        - ``context`` must be a non-empty string.
        - ``generated_artifact`` may be a string or None/null.

        Invalid candidates are logged and skipped.
        """
        actions: list[ManualAction] = []

        for idx, candidate in enumerate(raw_candidates):
            if not isinstance(candidate, dict):
                logger.warning(
                    "automation_auditor: candidate %d in %s is not a dict -- skipping",
                    idx,
                    file_path,
                )
                continue

            action_text = candidate.get("action_text", "")
            action_type = candidate.get("action_type", "")
            context = candidate.get("context", "")
            generated_artifact = candidate.get("generated_artifact")

            # Validate required string fields
            if not isinstance(action_text, str) or not action_text.strip():
                logger.warning(
                    "automation_auditor: candidate %d in %s has empty/missing "
                    "action_text -- skipping",
                    idx,
                    file_path,
                )
                continue

            if action_type not in ALLOWED_ACTION_TYPES:
                logger.warning(
                    "automation_auditor: candidate %d in %s has invalid "
                    "action_type %r -- skipping",
                    idx,
                    file_path,
                    action_type,
                )
                continue

            if not isinstance(context, str) or not context.strip():
                logger.warning(
                    "automation_auditor: candidate %d in %s has empty/missing "
                    "context -- skipping",
                    idx,
                    file_path,
                )
                continue

            # generated_artifact must be a string or None
            if generated_artifact is not None and not isinstance(generated_artifact, str):
                generated_artifact = str(generated_artifact)

            fingerprint = self._compute_fingerprint(action_text, file_path, action_type)

            actions.append(
                ManualAction(
                    action_text=action_text,
                    action_type=action_type,
                    context=context,
                    generated_artifact=generated_artifact,
                    action_fingerprint=fingerprint,
                )
            )

        return actions

    def _deduplicate(self, actions: list[ManualAction]) -> list[ManualAction]:
        """Deduplicate by action_fingerprint; first occurrence wins."""
        seen: set[str] = set()
        unique: list[ManualAction] = []
        for action in actions:
            if action.action_fingerprint not in seen:
                seen.add(action.action_fingerprint)
                unique.append(action)
        return unique


# ---------------------------------------------------------------------------
# GapResult dataclass and GapAssessor class  (Task 11.1)
# ---------------------------------------------------------------------------


@dataclass
class GapResult:
    """The result of a gap assessment for a ManualAction.

    Fields
    ------
    gap_type : str
        One of ``"full_gap"``, ``"partial_gap"``, or ``"no_gap"``.
    confidence : str
        One of ``"high"``, ``"medium"``, or ``"low"``.
    reason : str | None
        ``"no_cli_anchor"`` when ``cli_equivalent`` is ``None``;
        ``"missing_generated_artifact"`` when ``script_content`` is ``None``
        but a CLI equivalent exists; ``None`` for all other gap types.
    partial_gap_narrative : str | None
        Human-readable description of what the script does and what it omits,
        populated only when ``gap_type="partial_gap"``.
    """

    gap_type: str
    confidence: str
    reason: str | None
    partial_gap_narrative: str | None


class GapAssessor:
    """Performs two-pass static analysis of generated_artifact script content.

    Pass 1 — CLI name match: checks for the CLI command name as a substring
    in the script content.

    Pass 2 — API operation match: checks for the AWS service name AND the API
    operation name (catches boto3, SDK wrappers, Terraform/CloudFormation
    resources that implement the same API operation).

    Early-exit priority (checked before any pass):
    1. ``cli_result.cli_equivalent is None`` → ``full_gap``,
       ``reason="no_cli_anchor"``, ``confidence="low"``.
    2. ``script_content is None`` (but CLI exists) → ``full_gap``,
       ``reason="missing_generated_artifact"``, ``confidence="low"``.

    Known limitation: does not execute scripts; cannot detect dynamic
    dispatch, aliased commands, or wrapper functions.  The ``confidence``
    score communicates this limitation.
    """

    def assess(
        self,
        action: ManualAction,
        cli_result: CLIResult,
        script_content: str | None,
    ) -> GapResult:
        """Assess the automation gap for a single ManualAction.

        Parameters
        ----------
        action:
            The manual action being assessed.
        cli_result:
            The CLI lookup result for the action.
        script_content:
            Full text of the generated artifact script, or ``None`` when no
            artifact exists or the file could not be read.

        Returns
        -------
        GapResult
            Classified gap with confidence and optional narrative.
        """
        # --- Early-exit 1: no CLI anchor ---
        if cli_result.cli_equivalent is None:
            return GapResult(
                gap_type="full_gap",
                confidence="low",
                reason="no_cli_anchor",
                partial_gap_narrative=None,
            )

        # --- Early-exit 2: CLI exists but no script to analyse ---
        if script_content is None:
            return GapResult(
                gap_type="full_gap",
                confidence="low",
                reason="missing_generated_artifact",
                partial_gap_narrative=None,
            )

        # --- Pass 1: CLI command name substring match ---
        command_name = self._extract_command_name(cli_result.cli_equivalent)
        pass1_match = bool(command_name) and command_name in script_content

        # --- Pass 2: AWS service name + API operation name match ---
        service_name, operation_name = self._extract_service_operation(
            cli_result.cli_equivalent
        )
        pass2_match = bool(service_name) and bool(operation_name) and (
            service_name in script_content and operation_name in script_content
        )

        command_present = pass1_match or pass2_match

        if not command_present:
            return GapResult(
                gap_type="full_gap",
                confidence="low",
                reason=None,
                partial_gap_narrative=None,
            )

        # Command is present — check parameter coverage
        required_flags = self._extract_required_flags(cli_result)
        all_params_found = self._all_params_present(required_flags, script_content)

        if all_params_found:
            return GapResult(
                gap_type="no_gap",
                confidence="high",
                reason=None,
                partial_gap_narrative=None,
            )

        # Command present but parameter coverage uncertain or incomplete
        if required_flags:
            # We know what params are required and some are missing → partial_gap
            missing = [f for f in required_flags if f not in script_content]
            narrative = (
                f"Script contains '{command_name}' but is missing required "
                f"parameter(s): {', '.join(missing)}. "
                f"Action: {action.action_text[:200]}"
            )
            return GapResult(
                gap_type="partial_gap",
                confidence="medium",
                reason=None,
                partial_gap_narrative=narrative,
            )

        # Command found but no required flags to check — medium confidence partial
        narrative = (
            f"Script contains '{command_name}' but parameter coverage could "
            f"not be verified (no required flags in CLI template). "
            f"Action: {action.action_text[:200]}"
        )
        return GapResult(
            gap_type="partial_gap",
            confidence="medium",
            reason=None,
            partial_gap_narrative=narrative,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_command_name(cli_equivalent: str) -> str:
        """Extract the CLI sub-command name from a command template.

        For ``aws service-quotas request-service-quota-increase ...`` this
        returns ``"request-service-quota-increase"``.  For a command with
        fewer tokens it returns the last non-placeholder token.
        """
        # Strip placeholder tokens and split on whitespace
        tokens = [
            t for t in cli_equivalent.split()
            if not t.startswith("{") and not t.startswith("--")
        ]
        # tokens[0] is "aws", tokens[1] is service, tokens[2] is sub-command
        if len(tokens) >= 3:
            return tokens[2]
        if len(tokens) == 2:
            return tokens[1]
        return tokens[0] if tokens else ""

    @staticmethod
    def _extract_service_operation(cli_equivalent: str) -> tuple[str, str]:
        """Extract (service_name, operation_name) from a CLI command template.

        For ``aws service-quotas request-service-quota-increase ...`` this
        returns ``("service-quotas", "RequestServiceQuotaIncrease")``.

        The operation name is the PascalCase conversion of the sub-command
        name (used in boto3 and CloudFormation resource types).
        """
        tokens = [
            t for t in cli_equivalent.split()
            if not t.startswith("{") and not t.startswith("--")
        ]
        if len(tokens) < 3:
            return ("", "")

        service_name = tokens[1]  # e.g. "service-quotas"

        # Convert kebab-case sub-command to PascalCase operation name
        sub_command = tokens[2]  # e.g. "request-service-quota-increase"
        operation_name = "".join(
            part.capitalize() for part in sub_command.split("-")
        )  # e.g. "RequestServiceQuotaIncrease"

        return (service_name, operation_name)

    @staticmethod
    def _extract_required_flags(cli_result: CLIResult) -> list[str]:
        """Return the ``--flag-name`` strings for each placeholder in the template.

        For a template like
        ``aws iam create-role --role-name {role_name} --assume-role-policy-document {policy_document}``
        this returns ``["--role-name", "--assume-role-policy-document"]``.
        """
        if not cli_result.cli_equivalent:
            return []

        flags: list[str] = []
        tokens = cli_result.cli_equivalent.split()
        for i, token in enumerate(tokens):
            if token.startswith("{") and i > 0 and tokens[i - 1].startswith("--"):
                flags.append(tokens[i - 1])
        return flags

    @staticmethod
    def _all_params_present(required_flags: list[str], script_content: str) -> bool:
        """Return True when every required flag appears in the script content."""
        if not required_flags:
            # No required flags to check — treat as uncertain (not high confidence)
            return False
        return all(flag in script_content for flag in required_flags)


# ---------------------------------------------------------------------------
# JudgmentInputs, JudgmentResult dataclasses and JudgmentFilter class  (Task 11.3)
# ---------------------------------------------------------------------------

# Known safety_impact values
_KNOWN_SAFETY_IMPACTS = frozenset(
    {"none", "iam_change", "quota_increase", "delete", "production_impact"}
)

# Safety impacts that are hard-blocked (no automation regardless of other inputs)
_HARD_BLOCK_IMPACTS = frozenset({"delete", "production_impact"})

# Safety impacts that allow automation with a human gate when conditions are met
_GATED_IMPACTS = frozenset({"iam_change", "quota_increase"})


@dataclass
class JudgmentInputs:
    """Structured inputs for the JudgmentFilter scoring table.

    Separates the scoring inputs from the action/CLI/gap objects so the
    filter can be tested independently with explicit values.

    Fields
    ------
    judgment_required : bool
        Whether the user must make a judgment call before proceeding.
    repeated_operation : bool
        Whether the action is a repeated operation vs one-time setup.
    values_precomputed : bool
        Whether the CLI command requires values the plugin has already
        computed.
    safety_impact : str
        Safety classification: ``"none"``, ``"iam_change"``,
        ``"quota_increase"``, ``"delete"``, or ``"production_impact"``.
        Any unknown value is treated as ``"production_impact"``.
    """

    judgment_required: bool
    repeated_operation: bool
    values_precomputed: bool
    safety_impact: str


@dataclass
class JudgmentResult:
    """The result of a JudgmentFilter evaluation.

    Fields
    ------
    automate_recommended : str
        ``"yes"`` or ``"no"``.
    rationale : str
        Human-readable explanation of the decision.
    judgment_required : bool
        Echoed from inputs.
    repeated_operation : bool
        Echoed from inputs.
    values_precomputed : bool
        Echoed from inputs.
    safety_impact : str
        Echoed from inputs (after unknown-value normalisation).
    human_gate : str | None
        ``"required"`` when a human gate is mandated; ``None`` otherwise.
    """

    automate_recommended: str
    rationale: str
    judgment_required: bool
    repeated_operation: bool
    values_precomputed: bool
    safety_impact: str
    human_gate: str | None


class JudgmentFilter:
    """Applies safety and judgment criteria before recommending automation.

    Canonical evaluation order (normative — do not reorder):

    1. ``inputs.judgment_required=True``
       → ``automate_recommended=no``, ``human_gate=None``  (hard block)

    2. ``inputs.safety_impact in {delete, production_impact}``
       → ``automate_recommended=no``, ``human_gate=None``

    3. ``inputs.safety_impact in {iam_change, quota_increase}``
       AND ``judgment_required=False``
       AND ``repeated_operation=True``
       AND ``values_precomputed=True``
       → ``automate_recommended=yes``, ``human_gate="required"``

    4. ``inputs.safety_impact in {iam_change, quota_increase}``  (other combos)
       → ``automate_recommended=no``, ``human_gate=None``

    5. ``safety_impact=none``
       AND ``judgment_required=False``
       AND ``repeated_operation=True``
       AND ``values_precomputed=True``
       → ``automate_recommended=yes``, ``human_gate=None``

    6. All other combinations
       → ``automate_recommended=no``, ``human_gate=None``

    Unknown ``safety_impact`` values are normalised to ``"production_impact"``
    before evaluation (conservative default).
    """

    def evaluate(
        self,
        action: ManualAction,
        cli_result: CLIResult,
        gap_result: GapResult,
        inputs: JudgmentInputs,
    ) -> JudgmentResult:
        """Apply the decision table and return a JudgmentResult.

        ``gap_result`` is accepted for context only and does not influence
        the ``automate_recommended`` decision.

        Parameters
        ----------
        action:
            The manual action being evaluated.
        cli_result:
            The CLI lookup result for the action.
        gap_result:
            The gap assessment result (context only; does not affect decision).
        inputs:
            Explicit judgment inputs.

        Returns
        -------
        JudgmentResult
            Decision with rationale and echoed inputs.
        """
        # Normalise unknown safety_impact values to "production_impact"
        safety_impact = (
            inputs.safety_impact
            if inputs.safety_impact in _KNOWN_SAFETY_IMPACTS
            else "production_impact"
        )

        # --- Step 1: judgment_required hard block ---
        if inputs.judgment_required:
            return JudgmentResult(
                automate_recommended="no",
                rationale=(
                    "Automation blocked: this action requires human judgment "
                    "before proceeding. The action cannot be safely automated "
                    "without a human decision point."
                ),
                judgment_required=inputs.judgment_required,
                repeated_operation=inputs.repeated_operation,
                values_precomputed=inputs.values_precomputed,
                safety_impact=safety_impact,
                human_gate=None,
            )

        # --- Step 2: hard-blocked safety impacts ---
        if safety_impact in _HARD_BLOCK_IMPACTS:
            return JudgmentResult(
                automate_recommended="no",
                rationale=(
                    f"Automation blocked: safety_impact='{safety_impact}' is a "
                    f"hard-blocked category. Actions with delete or production "
                    f"impact must not be automated."
                ),
                judgment_required=inputs.judgment_required,
                repeated_operation=inputs.repeated_operation,
                values_precomputed=inputs.values_precomputed,
                safety_impact=safety_impact,
                human_gate=None,
            )

        # --- Step 3: gated impacts — all conditions met → yes with human gate ---
        if safety_impact in _GATED_IMPACTS:
            if (
                not inputs.judgment_required
                and inputs.repeated_operation
                and inputs.values_precomputed
            ):
                return JudgmentResult(
                    automate_recommended="yes",
                    rationale=(
                        f"Automation recommended with human gate: "
                        f"safety_impact='{safety_impact}' requires explicit "
                        f"human approval before execution. All other conditions "
                        f"are met (repeated_operation=True, values_precomputed=True, "
                        f"judgment_required=False)."
                    ),
                    judgment_required=inputs.judgment_required,
                    repeated_operation=inputs.repeated_operation,
                    values_precomputed=inputs.values_precomputed,
                    safety_impact=safety_impact,
                    human_gate="required",
                )

            # --- Step 4: gated impacts — other combinations → no ---
            missing: list[str] = []
            if not inputs.repeated_operation:
                missing.append("repeated_operation=True")
            if not inputs.values_precomputed:
                missing.append("values_precomputed=True")
            return JudgmentResult(
                automate_recommended="no",
                rationale=(
                    f"Automation blocked: safety_impact='{safety_impact}' "
                    f"requires all conditions to be met for automation. "
                    f"Missing: {', '.join(missing) if missing else 'unknown condition'}."
                ),
                judgment_required=inputs.judgment_required,
                repeated_operation=inputs.repeated_operation,
                values_precomputed=inputs.values_precomputed,
                safety_impact=safety_impact,
                human_gate=None,
            )

        # --- Step 5: safety_impact=none — all conditions met → yes ---
        if safety_impact == "none":
            if inputs.repeated_operation and inputs.values_precomputed:
                return JudgmentResult(
                    automate_recommended="yes",
                    rationale=(
                        "Automation recommended: safety_impact=none, "
                        "repeated_operation=True, values_precomputed=True, "
                        "judgment_required=False. All conditions are met."
                    ),
                    judgment_required=inputs.judgment_required,
                    repeated_operation=inputs.repeated_operation,
                    values_precomputed=inputs.values_precomputed,
                    safety_impact=safety_impact,
                    human_gate=None,
                )

        # --- Step 6: all other combinations → no ---
        missing_conditions: list[str] = []
        if not inputs.repeated_operation:
            missing_conditions.append("repeated_operation=True")
        if not inputs.values_precomputed:
            missing_conditions.append("values_precomputed=True")
        return JudgmentResult(
            automate_recommended="no",
            rationale=(
                f"Automation not recommended: not all conditions are met. "
                f"Missing: {', '.join(missing_conditions) if missing_conditions else 'unknown condition'}."
            ),
            judgment_required=inputs.judgment_required,
            repeated_operation=inputs.repeated_operation,
            values_precomputed=inputs.values_precomputed,
            safety_impact=safety_impact,
            human_gate=None,
        )

    def infer_inputs(
        self,
        action: ManualAction,
        cli_result: CLIResult,
    ) -> JudgmentInputs:
        """Infer JudgmentInputs from action and CLI context.

        Used when inputs are not explicitly provided (e.g., in automated runs).

        Conservative bias: when inference is uncertain, defaults to
        ``judgment_required=True`` and ``safety_impact="production_impact"``
        so that automation recommendations stay rare and safe.  It is better
        to under-recommend automation than to recommend it incorrectly for a
        destructive action.

        Heuristics applied
        ------------------
        - ``judgment_required``: ``True`` unless the action type is
          ``copy_paste`` or ``manual_calculation`` (low-judgment types).
        - ``repeated_operation``: ``True`` when the action text contains
          repetition signals ("each", "every", "all", "per", "for each").
        - ``values_precomputed``: ``True`` when the CLI template has no
          unresolved placeholders (i.e., ``placeholder_list`` is empty) or
          when the action text references a previously computed value.
        - ``safety_impact``: inferred from action text and CLI command:
          - ``"delete"`` when action text contains "delete", "remove",
            "destroy", "terminate", "drop".
          - ``"iam_change"`` when action text or CLI contains "iam",
            "role", "policy", "permission".
          - ``"quota_increase"`` when action text or CLI contains "quota",
            "limit", "increase".
          - ``"none"`` when action type is ``copy_paste`` or
            ``manual_calculation`` and no other signals are present.
          - ``"production_impact"`` (default) for all other cases.

        Parameters
        ----------
        action:
            The manual action to infer inputs for.
        cli_result:
            The CLI lookup result for the action.

        Returns
        -------
        JudgmentInputs
            Conservatively inferred inputs.
        """
        action_lower = action.action_text.lower()
        cli_lower = (cli_result.cli_equivalent or "").lower()
        combined = f"{action_lower} {cli_lower}"

        # --- judgment_required ---
        low_judgment_types = {"copy_paste", "manual_calculation"}
        judgment_required = action.action_type not in low_judgment_types

        # --- repeated_operation ---
        repetition_signals = {"each", "every", "all ", "per ", "for each"}
        repeated_operation = any(sig in action_lower for sig in repetition_signals)

        # --- values_precomputed ---
        # True only when a CLI equivalent exists AND has no unresolved placeholders.
        # When cli_equivalent is None (_NO_MATCH), placeholder_list=[] but there
        # is no CLI to run, so values_precomputed must be False.
        values_precomputed = (
            cli_result.cli_equivalent is not None
            and len(cli_result.placeholder_list) == 0
        )

        # --- safety_impact ---
        delete_signals = {"delete", "remove", "destroy", "terminate", "drop"}
        iam_signals = {"iam", " role", "policy", "permission"}
        quota_signals = {"quota", "limit", "increase"}

        if any(sig in action_lower for sig in delete_signals):
            safety_impact = "delete"
        elif any(sig in combined for sig in iam_signals):
            safety_impact = "iam_change"
        elif any(sig in combined for sig in quota_signals):
            safety_impact = "quota_increase"
        elif action.action_type in low_judgment_types:
            safety_impact = "none"
        else:
            safety_impact = "production_impact"

        return JudgmentInputs(
            judgment_required=judgment_required,
            repeated_operation=repeated_operation,
            values_precomputed=values_precomputed,
            safety_impact=safety_impact,
        )

# ---------------------------------------------------------------------------
# Redaction helpers (shared with currency_auditor pattern)
# ---------------------------------------------------------------------------

# ARN pattern: arn:aws:<service>:<region>:<account-id>:<resource>
_ARN_RE = re.compile(
    r"arn:aws:[a-z0-9\-]+:[a-z0-9\-]*:\d{12}:[^\s\"']*",
    re.IGNORECASE,
)

# Account ID: 12-digit number preceded by an account-context keyword
_ACCOUNT_ID_RE = re.compile(
    r"(?:account(?:\s+id)?|aws\s+account)\s*[:\s]\s*(\d{12})\b",
    re.IGNORECASE,
)

# AWS access key ID pattern
_ACCESS_KEY_RE = re.compile(r"AKIA[0-9A-Z]{16}")


def _redact_value(value: str) -> str:
    """Apply redaction patterns to a single string value.

    Patterns applied (in order):
    1. ARNs → ``[REDACTED_ARN]``
    2. Account IDs (12-digit numbers preceded by account-context keyword)
       → ``[REDACTED_ACCOUNT_ID]``
    3. AWS access key IDs (``AKIA[0-9A-Z]{16}``) → ``[REDACTED_KEY]``

    Blind redaction of all 12-digit numbers is NOT performed.
    """
    value = _ARN_RE.sub("[REDACTED_ARN]", value)
    value = _ACCOUNT_ID_RE.sub(
        lambda m: m.group(0).replace(m.group(1), "[REDACTED_ACCOUNT_ID]"),
        value,
    )
    value = _ACCESS_KEY_RE.sub("[REDACTED_KEY]", value)
    return value


def _redact_payload(payload: dict) -> dict:
    """Return a shallow copy of *payload* with string values redacted.

    Only top-level string values are redacted.  Nested dicts (e.g.
    ``migration_context``) are recursively processed.
    """
    result: dict = {}
    for key, val in payload.items():
        if isinstance(val, str):
            result[key] = _redact_value(val)
        elif isinstance(val, dict):
            result[key] = _redact_payload(val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Migration context detection helpers
# ---------------------------------------------------------------------------

# Patterns used to detect source/target system from file path and content.
# These mirror the approach used in currency_auditor.py.

_SOURCE_SYSTEM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bgcp\b|\bgoogle\s+cloud\b|\bgoogle\s+cloud\s+platform\b", re.IGNORECASE), "GCP"),
    (re.compile(r"\bazure\b|\bmicrosoft\s+azure\b", re.IGNORECASE), "Azure"),
    (re.compile(r"\bon[\s\-]?prem(?:ises?)?\b|\bon[\s\-]?premise\b", re.IGNORECASE), "on-premises"),
    (re.compile(r"\bheroku\b", re.IGNORECASE), "Heroku"),
    (re.compile(r"\bdigitalocean\b", re.IGNORECASE), "DigitalOcean"),
]

_TARGET_SYSTEM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\baws\b|\bamazon\s+web\s+services\b", re.IGNORECASE), "AWS"),
]

# Path-based hints: if the file path contains "gcp-to-aws" we can infer both systems.
_PATH_GCP_TO_AWS_RE = re.compile(r"gcp[\-_]to[\-_]aws", re.IGNORECASE)
_PATH_AZURE_TO_AWS_RE = re.compile(r"azure[\-_]to[\-_]aws", re.IGNORECASE)


def _detect_migration_context(
    file_path: str,
    content: str,
) -> dict | None:
    """Detect migration context (source_system, target_system) from file path and content.

    Detection strategy (same as currency_auditor):
    1. Check file path for known migration direction patterns (e.g. ``gcp-to-aws``).
    2. Scan the first 20 lines of content for source/target system keywords.

    Returns a dict ``{"source_system": str, "target_system": str}`` when both
    systems are identified, or ``None`` when the context cannot be determined.
    """
    # --- Step 1: path-based detection ---
    if _PATH_GCP_TO_AWS_RE.search(file_path):
        return {"source_system": "GCP", "target_system": "AWS"}
    if _PATH_AZURE_TO_AWS_RE.search(file_path):
        return {"source_system": "Azure", "target_system": "AWS"}

    # --- Step 2: content-based detection (first 20 lines) ---
    lines = content.splitlines()[:20]
    header = "\n".join(lines)

    source_system: str | None = None
    target_system: str | None = None

    for pattern, system in _SOURCE_SYSTEM_PATTERNS:
        if pattern.search(header):
            source_system = system
            break

    for pattern, system in _TARGET_SYSTEM_PATTERNS:
        if pattern.search(header):
            target_system = system
            break

    if source_system and target_system:
        return {"source_system": source_system, "target_system": target_system}

    # Fallback: scan full content (some files have context buried deeper)
    if source_system is None:
        for pattern, system in _SOURCE_SYSTEM_PATTERNS:
            if pattern.search(content):
                source_system = system
                break

    if target_system is None:
        for pattern, system in _TARGET_SYSTEM_PATTERNS:
            if pattern.search(content):
                target_system = system
                break

    if source_system and target_system:
        return {"source_system": source_system, "target_system": target_system}

    return None


def _migration_context_to_scope(migration_context: dict) -> str:
    """Return a human-readable scope string from a migration_context dict."""
    source = migration_context.get("source_system", "unknown")
    target = migration_context.get("target_system", "unknown")
    return f"{source} → {target}"


# ---------------------------------------------------------------------------
# automation_dedupe_key
# ---------------------------------------------------------------------------


def automation_dedupe_key(finding: "Finding") -> tuple:  # type: ignore[name-defined]
    """Return the dedupe key for an Automation_Auditor finding.

    Key: ``(category, frozenset(affected_files), action_fingerprint)``

    The ``action_fingerprint`` is read from
    ``finding.auditor_payload["action_fingerprint"]``.

    Guard: when ``auditor_payload`` is missing or ``action_fingerprint`` is
    absent, falls back to ``finding.finding_id`` so the finding is never
    silently dropped.

    Parameters
    ----------
    finding:
        A ``Finding`` object produced by ``run_automation_audit()``.

    Returns
    -------
    tuple
        ``(category, frozenset(affected_files), action_fingerprint_or_id)``
    """
    payload = getattr(finding, "auditor_payload", None) or {}
    action_fingerprint = payload.get("action_fingerprint") or finding.finding_id
    return (
        finding.category,
        frozenset(finding.affected_files),
        action_fingerprint,
    )


# ---------------------------------------------------------------------------
# run_automation_audit
# ---------------------------------------------------------------------------

# CLI no-match emission policy: skip findings for these action types when
# cli_equivalent is None (they have no meaningful programmatic equivalent).
_CLI_NO_MATCH_SKIP_TYPES = frozenset({"copy_paste", "file_edit", "browser_action"})


async def run_automation_audit(
    repo_content: "RepoContent",  # type: ignore[name-defined]
    authoritative_data: "AuthoritativeData",  # type: ignore[name-defined]
    run_id: str,
    file_filter: list[str] | None = None,
) -> list["Finding"]:  # type: ignore[name-defined]
    """Run the Automation Auditor against all (or filtered) Reference_Files.

    For each Reference_File in ``repo_content.files`` (filtered by
    ``file_filter`` when provided):

    1. Detect migration context (source_system, target_system) — halt file if
       not found; emit structured error to ``partial_source_failures``.
    2. Extract manual actions via ``ActionExtractor``.
    3. For each action:
       a. Look up CLI equivalent via ``CLILookup``.
       b. Load script content from ``repo_content.files`` if
          ``action.generated_artifact`` is set.
       c. Assess gap via ``GapAssessor``.
       d. Apply judgment filter via ``JudgmentFilter.infer_inputs()`` then
          ``evaluate()``.
       e. Skip ``no_gap`` findings.
       f. Apply CLI no-match emission policy: skip when
          ``cli_result.cli_equivalent is None`` AND
          ``action.action_type in {copy_paste, file_edit, browser_action}``.
       g. Build ``auditor_payload`` dict.
       h. Apply redaction patterns.
       i. Set risk_level: MEDIUM when ``human_gate="required"``, LOW otherwise.
       j. Call ``PayloadStore.store_payload()`` for each Finding.
       k. Construct ``Finding`` with ``category="automation_gap"``,
          ``finding_schema_version="automation/1.0"``.
    4. Return list of Findings.

    Parameters
    ----------
    repo_content:
        Snapshot of the target repository (files dict: path → content).
    authoritative_data:
        Aggregated authoritative data (not used directly by this auditor but
        accepted for API symmetry with ``run_currency_audit``).
    run_id:
        Identifier for the current scan run (set on every Finding).
    file_filter:
        When provided, only files whose paths appear in this list are
        processed.  ``None`` means all files.

    Returns
    -------
    list[Finding]
        All automation gap findings produced during this run.
    """
    import uuid
    from datetime import datetime, timezone

    from migration_watchdog.models import Finding, RiskLevel
    from migration_watchdog.payload_store import PayloadStore

    findings: list[Finding] = []
    partial_source_failures: list[dict] = []

    extractor = ActionExtractor()
    cli_lookup = CLILookup()
    gap_assessor = GapAssessor()
    judgment_filter = JudgmentFilter()
    payload_store = PayloadStore()

    # Determine which files to process
    files_to_process: dict[str, str] = {}
    if file_filter is not None:
        filter_set = set(file_filter)
        for path, content in repo_content.files.items():
            if path in filter_set:
                files_to_process[path] = content
    else:
        files_to_process = dict(repo_content.files)

    for file_path, file_content in files_to_process.items():
        # --- Step 1: Detect migration context ---
        migration_context = _detect_migration_context(file_path, file_content)
        if migration_context is None:
            logger.info(
                "run_automation_audit: no migration context found in %s — skipping",
                file_path,
            )
            partial_source_failures.append(
                {
                    "type": "no_migration_context",
                    "file": file_path,
                    "message": (
                        "Could not determine migration context (source_system / "
                        "target_system) from file path or content. "
                        "Human review required."
                    ),
                }
            )
            continue

        scope = _migration_context_to_scope(migration_context)

        # --- Step 2: Extract manual actions ---
        try:
            actions = await extractor.extract(file_path, file_content)
        except Exception:
            logger.exception(
                "run_automation_audit: ActionExtractor failed for %s — skipping",
                file_path,
            )
            partial_source_failures.append(
                {
                    "type": "extraction_error",
                    "file": file_path,
                    "message": "ActionExtractor raised an unexpected exception.",
                }
            )
            continue

        if not actions:
            logger.info(
                "run_automation_audit: no manual actions found in %s", file_path
            )
            continue

        # --- Step 3: Process each action ---
        for action in actions:
            # a. CLI lookup
            cli_result = cli_lookup.lookup(action)

            # b. Load script content if generated_artifact is set
            script_content: str | None = None
            if action.generated_artifact:
                script_content = repo_content.files.get(action.generated_artifact)
                if script_content is None:
                    logger.debug(
                        "run_automation_audit: generated_artifact %r not found in "
                        "repo_content for action in %s",
                        action.generated_artifact,
                        file_path,
                    )

            # c. Gap assessment
            gap_result = gap_assessor.assess(action, cli_result, script_content)

            # d. Judgment filter
            judgment_inputs = judgment_filter.infer_inputs(action, cli_result)
            judgment_result = judgment_filter.evaluate(
                action, cli_result, gap_result, judgment_inputs
            )

            # e. Skip no_gap findings
            if gap_result.gap_type == "no_gap":
                logger.debug(
                    "run_automation_audit: skipping no_gap finding for action "
                    "%r in %s",
                    action.action_text[:60],
                    file_path,
                )
                continue

            # f. CLI no-match emission policy
            if (
                cli_result.cli_equivalent is None
                and action.action_type in _CLI_NO_MATCH_SKIP_TYPES
            ):
                logger.debug(
                    "run_automation_audit: CLI no-match emission policy — skipping "
                    "action_type=%r with no CLI equivalent in %s",
                    action.action_type,
                    file_path,
                )
                continue

            # g. Build auditor_payload
            auditor_payload: dict = {
                "action_text": action.action_text,
                "action_type": action.action_type,
                "action_fingerprint": action.action_fingerprint,
                "cli_equivalent": cli_result.cli_equivalent,
                "placeholder_list": cli_result.placeholder_list,
                "reference_url": cli_result.reference_url,
                "cli_note": cli_result.cli_note,
                "generated_artifact": action.generated_artifact,
                "gap_type": gap_result.gap_type,
                "confidence": gap_result.confidence,
                "reason": gap_result.reason,
                "partial_gap_narrative": gap_result.partial_gap_narrative,
                "automate_recommended": judgment_result.automate_recommended,
                "rationale": judgment_result.rationale,
                "judgment_required": judgment_result.judgment_required,
                "repeated_operation": judgment_result.repeated_operation,
                "values_precomputed": judgment_result.values_precomputed,
                "safety_impact": judgment_result.safety_impact,
                "human_gate": judgment_result.human_gate,
                "migration_context": migration_context,
                "scope": scope,
            }

            # h. Apply redaction
            auditor_payload = _redact_payload(auditor_payload)

            # i. Risk level
            risk_level = (
                RiskLevel.MEDIUM
                if judgment_result.human_gate == "required"
                else RiskLevel.LOW
            )

            # Build finding_id before payload store call
            finding_id = str(uuid.uuid4())

            # j. PayloadStore.store_payload()
            try:
                auditor_payload = await payload_store.store_payload(
                    finding_id, auditor_payload
                )
            except ValueError:
                logger.exception(
                    "run_automation_audit: payload too large for finding %s "
                    "(action %r in %s) — skipping",
                    finding_id,
                    action.action_text[:60],
                    file_path,
                )
                partial_source_failures.append(
                    {
                        "type": "payload_too_large",
                        "finding_id": finding_id,
                        "file": file_path,
                    }
                )
                continue

            # k. Construct Finding
            now = datetime.now(timezone.utc).isoformat()
            title = (
                f"Automation gap: {action.action_type} in {file_path}"
            )
            description = (
                f"Manual action detected that may be automatable. "
                f"Gap type: {gap_result.gap_type}. "
                f"Automate recommended: {judgment_result.automate_recommended}. "
                f"Action: {action.action_text[:200]}"
            )

            finding = Finding(
                finding_id=finding_id,
                run_id=run_id,
                risk_level=risk_level,
                category="automation_gap",
                title=title,
                description=description,
                affected_files=[file_path],
                proposed_changes={},
                source_urls=(
                    [cli_result.reference_url]
                    if cli_result.reference_url
                    else []
                ),
                scan_timestamp=now,
                status="pending",
                review_status=None,
                review_notes=None,
                primary_reasoning=None,
                partial_data_warning=False,
                pr_url=None,
                dismissal=None,
                auditor_payload=auditor_payload,
                finding_schema_version="automation/1.0",
            )

            findings.append(finding)

    # Attach partial_source_failures to authoritative_data if possible
    # (mirrors the pattern used in currency_auditor / main.py)
    if partial_source_failures:
        existing = getattr(authoritative_data, "partial_failures", None)
        if existing is not None:
            for failure in partial_source_failures:
                # Use json.dumps for structured dict entries so ops tooling can parse them
                if isinstance(failure, dict):
                    existing.append(json.dumps(failure))
                else:
                    existing.append(str(failure))
        logger.warning(
            "run_automation_audit: %d partial source failure(s) recorded",
            len(partial_source_failures),
        )

    logger.info(
        "run_automation_audit: produced %d finding(s) from %d file(s)",
        len(findings),
        len(files_to_process),
    )
    return findings
