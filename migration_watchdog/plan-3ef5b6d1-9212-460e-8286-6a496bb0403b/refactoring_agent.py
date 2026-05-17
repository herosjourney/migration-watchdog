"""Strands-based refactoring assessment agent using Claude Opus 4.7.

Evaluates whether the target repo's plugin structure warrants refactoring
based on current scan results, accumulated findings history, and repo
structure. Generates high-risk findings with advantages, disadvantages,
and proposed structure.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from uuid import uuid4

from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from migration_watchdog.models import (
    Finding,
    RepoContent,
    RiskLevel,
    classify_risk,
)

logger = logging.getLogger(__name__)

# Module-level state that the create_refactoring_finding tool writes to
# during a run. Reset at the start of each run_refactoring_assessment() call.
_refactoring_finding: Finding | None = None
_current_run_id: str = ""

# ---------------------------------------------------------------------------
# Strands @tool functions
# ---------------------------------------------------------------------------


@tool
def analyze_repo_structure(repo_files_json: str, findings_history_json: str) -> str:
    """Analyze the repo structure and accumulated findings to assess refactoring needs.

    Args:
        repo_files_json: JSON string of repo file paths and sizes
        findings_history_json: JSON string of historical findings

    Returns:
        JSON string with analysis results (file count, structure, patterns in findings)
    """
    try:
        repo_files: dict[str, str] = json.loads(repo_files_json)
    except (json.JSONDecodeError, TypeError):
        repo_files = {}

    try:
        findings_history: list[dict] = json.loads(findings_history_json)
    except (json.JSONDecodeError, TypeError):
        findings_history = []

    # Compute structural metrics
    file_count = len(repo_files)
    total_size = sum(len(content) for content in repo_files.values())

    # Analyze directory structure
    directories: dict[str, int] = {}
    for path in repo_files:
        parts = path.rsplit("/", 1)
        directory = parts[0] if len(parts) > 1 else "."
        directories[directory] = directories.get(directory, 0) + 1

    # Analyze findings patterns — which categories and files are repeatedly flagged
    category_counts: dict[str, int] = {}
    affected_file_counts: dict[str, int] = {}
    for finding in findings_history:
        cat = finding.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        for f in finding.get("affected_files", []):
            affected_file_counts[f] = affected_file_counts.get(f, 0) + 1

    # Identify repeatedly flagged files (flagged more than once)
    repeatedly_flagged = {
        f: count for f, count in affected_file_counts.items() if count > 1
    }

    return json.dumps(
        {
            "file_count": file_count,
            "total_content_size": total_size,
            "directories": directories,
            "findings_history_count": len(findings_history),
            "category_counts": category_counts,
            "repeatedly_flagged_files": repeatedly_flagged,
            "affected_file_counts": affected_file_counts,
        }
    )


@tool
def create_refactoring_finding(
    advantages: str, disadvantages: str, proposed_structure: str
) -> str:
    """Create a high-risk refactoring Finding.

    Auto-classifies as HIGH risk (category="refactoring").

    Args:
        advantages: Description of refactoring advantages
        disadvantages: Description of refactoring disadvantages
        proposed_structure: Description of what the refactored structure would look like

    Returns:
        JSON string of the created Finding
    """
    global _refactoring_finding, _current_run_id

    risk_level = classify_risk("refactoring")
    finding_id = str(uuid4())
    scan_timestamp = datetime.utcnow().isoformat()

    description = (
        f"**Advantages:**\n{advantages}\n\n"
        f"**Disadvantages:**\n{disadvantages}\n\n"
        f"**Proposed Structure:**\n{proposed_structure}"
    )

    finding = Finding(
        finding_id=finding_id,
        run_id=_current_run_id,
        risk_level=risk_level,
        category="refactoring",
        title="Plugin Structure Refactoring Assessment",
        description=description,
        affected_files=[],
        proposed_changes={},
        source_urls=[],
        scan_timestamp=scan_timestamp,
        status="pending",
    )

    _refactoring_finding = finding

    return json.dumps(
        {
            "created": True,
            "finding_id": finding_id,
            "risk_level": risk_level.value,
            "category": "refactoring",
            "title": finding.title,
            "advantages": advantages,
            "disadvantages": disadvantages,
            "proposed_structure": proposed_structure,
            "scan_timestamp": scan_timestamp,
        }
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

REFACTORING_SYSTEM_PROMPT = """You are the Migration Plugin Watchdog Refactoring Assessor. \
Your job is to evaluate whether the plugin's structure and architecture warrants refactoring.

Look for architectural patterns AND structural issues, including:

**Architectural patterns:**
- State machine failure recovery (do phases have replay/rewind? do they emit draft artifacts?)
- Context loading strategy (is there per-phase context budget? or does the skill blindly load all refs?)
- Multi-agent orchestration (is it single-threaded? could large migrations benefit from sub-agents?)
- Validation checkpoints and fail-closed gates
- Artifact checksumming and hand-edit detection
- Token cost measurement per phase

**Structural issues:**
- File organization and naming conventions
- Content duplication across files
- Accumulated findings patterns (are the same areas repeatedly flagged?)
- Whether new content areas (AgentCore, Strands SDK) fit naturally
- Excessive coupling between phases

Review the provided SKILL.md and phase orchestration files carefully. Look for:
- Places where the skill relies on agent discretion where it should be declarative
- Missing failure recovery documentation
- Scalability limits that should be documented but aren't

If you identify genuine improvements, create one finding per distinct recommendation using \
create_refactoring_finding. Each finding should have:
- Clear advantages (what problem does it solve?)
- Clear disadvantages and risks (what's the cost of the change?)
- A concrete description of what the new structure/pattern would look like

Create multiple findings if there are multiple independent improvements. Don't bundle \
unrelated concerns. Only suggest genuinely valuable changes — don't change for the sake \
of change."""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_refactoring_agent() -> Agent:
    """Create the Strands refactoring assessment agent with Claude Opus 4.7."""
    model = BedrockModel(
        model_id="us.anthropic.claude-opus-4-7",
        region_name="us-east-1",
        max_tokens=16000,
    )
    return Agent(
        model=model,
        system_prompt=REFACTORING_SYSTEM_PROMPT,
        tools=[analyze_repo_structure, create_refactoring_finding],
    )


# ---------------------------------------------------------------------------
# Run refactoring assessment
# ---------------------------------------------------------------------------


def _build_refactoring_prompt(
    repo_content: RepoContent,
    existing_findings: list[Finding],
    run_id: str,
) -> str:
    """Build the user message with repo structure and findings history."""
    sections: list[str] = []

    sections.append(f"## Refactoring Assessment for Run: {run_id}\n")

    # Repo structure summary
    sections.append("## Repository Structure\n")
    repo_files_summary: dict[str, str] = {}
    for path, content in repo_content.files.items():
        repo_files_summary[path] = f"{len(content)} chars"
    sections.append(
        "```json\n"
        + json.dumps(repo_files_summary, indent=2)
        + "\n```\n"
    )

    # Include key architectural files for deep review
    sections.append("## Key Architectural Files (for deep review)\n")
    arch_keywords = ["SKILL.md", "skill.md", "phases/clarify/clarify.md",
                     "phases/design/design.md", "phases/discover/discover.md",
                     "phases/estimate/estimate.md", "phases/generate/generate.md",
                     "phases/feedback/feedback.md", "design-refs/index.md"]
    for path, content in repo_content.files.items():
        if any(kw in path for kw in arch_keywords):
            sections.append(f"### {path}\n```\n{content[:3000]}\n```\n")

    # Findings history
    sections.append("## Accumulated Findings History\n")
    findings_data = [
        {
            "finding_id": f.finding_id,
            "category": f.category,
            "title": f.title,
            "affected_files": f.affected_files,
            "risk_level": f.risk_level.value if isinstance(f.risk_level, RiskLevel) else f.risk_level,
            "status": f.status,
            "scan_timestamp": f.scan_timestamp,
        }
        for f in existing_findings
    ]
    sections.append(
        "```json\n"
        + json.dumps(findings_data, indent=2)
        + "\n```\n"
    )

    sections.append(
        "\n## Instructions\n"
        "Please analyze the repository structure and accumulated findings history "
        "using the analyze_repo_structure tool. Based on the analysis, determine "
        "whether the plugin's structure warrants refactoring. If you identify a "
        "genuine structural improvement, use the create_refactoring_finding tool "
        "to create a finding with clear advantages, disadvantages, and a proposed "
        "new structure. If no refactoring is warranted, simply explain why the "
        "current structure is adequate."
    )

    return "\n".join(sections)


def run_refactoring_assessment(
    repo_content: RepoContent,
    existing_findings: list[Finding],
    run_id: str,
) -> Finding | None:
    """Run the refactoring assessment agent.

    Builds a prompt with repo structure and findings history, runs the
    Strands agent, and returns the refactoring Finding if one was created,
    or None if no refactoring is warranted.

    Args:
        repo_content: Snapshot of the target repo.
        existing_findings: Previously generated findings from this and past runs.
        run_id: Unique identifier for this scan run.

    Returns:
        A refactoring Finding if the agent recommends refactoring, else None.
    """
    global _refactoring_finding, _current_run_id

    # Reset module-level state for this run
    _refactoring_finding = None
    _current_run_id = run_id

    agent = create_refactoring_agent()

    # Build the user message with repo structure and findings history
    user_message = _build_refactoring_prompt(repo_content, existing_findings, run_id)

    logger.info("Starting refactoring assessment agent for run %s", run_id)

    # Let the agent reason and use tools
    agent(user_message)

    if _refactoring_finding is not None:
        logger.info(
            "Refactoring assessment for run %s: finding created (id=%s)",
            run_id,
            _refactoring_finding.finding_id,
        )
    else:
        logger.info(
            "Refactoring assessment for run %s: no refactoring warranted",
            run_id,
        )

    return _refactoring_finding
