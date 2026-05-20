"""Strands-based review agent using Amazon Nova 2 Pro.

Quality-checks medium and high-risk findings for accuracy and
hallucinations, providing an independent second opinion from a
different model family. Findings are never discarded — disputed
findings pass through with both models' reasoning attached.
"""

from __future__ import annotations

import json
import logging

from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from migration_watchdog.models import AuthoritativeData, Finding, ReviewResult, RiskLevel
from migration_watchdog.source_fetcher import AwsDocsSearcher

logger = logging.getLogger(__name__)

# Module-level docs searcher for the review agent
_review_docs_searcher = AwsDocsSearcher()


@tool
def search_aws_docs_for_review(query: str) -> str:
    """Search AWS documentation to verify a finding's claims.

    Use this when the pre-fetched authoritative data is insufficient to
    confirm or deny a finding. Search for the specific claim being made.

    Args:
        query: Specific search query, e.g. "AWS App Runner closed new customers 2026"

    Returns:
        Relevant text from AWS documentation.
    """
    return _review_docs_searcher.search_and_fetch(query)

# ---------------------------------------------------------------------------
# Strands @tool functions
# ---------------------------------------------------------------------------


@tool
def verify_finding_accuracy(
    finding_json: str, authoritative_data_json: str
) -> str:
    """Verify a finding's claims against the authoritative source data.

    Cross-references the finding's description and proposed changes against
    the raw authoritative data to check for hallucinations or inaccuracies.

    Args:
        finding_json: JSON string of the Finding to verify
        authoritative_data_json: JSON string of relevant authoritative source data

    Returns:
        JSON string of accuracy assessment with review_status, issues, and notes
    """
    try:
        finding_data = json.loads(finding_json)
        authoritative_data = json.loads(authoritative_data_json)
    except (json.JSONDecodeError, TypeError) as exc:
        return json.dumps(
            {
                "review_status": "disputed",
                "issues": [f"Failed to parse input data: {exc}"],
                "notes": "Could not verify finding due to malformed input data.",
            }
        )

    issues: list[str] = []
    notes_parts: list[str] = []

    # Check for source citations — findings without citations are auto-flagged
    source_urls = finding_data.get("source_urls", [])
    if not source_urls:
        return json.dumps(
            {
                "review_status": "disputed",
                "issues": ["no source citation provided"],
                "notes": (
                    "Finding has no source_urls. Cannot verify claims without "
                    "authoritative source citations."
                ),
            }
        )

    # Cross-reference the finding's claims against the authoritative data
    description = finding_data.get("description", "")
    proposed_changes = finding_data.get("proposed_changes", {})

    notes_parts.append(
        f"Verified finding '{finding_data.get('title', 'unknown')}' "
        f"against {len(authoritative_data)} authoritative data entries."
    )
    notes_parts.append(f"Finding cites {len(source_urls)} source URL(s).")

    if description:
        notes_parts.append(f"Description length: {len(description)} chars.")
    if proposed_changes:
        notes_parts.append(
            f"Proposed changes affect {len(proposed_changes)} file(s)."
        )

    # Return the structured assessment for the LLM to reason about
    return json.dumps(
        {
            "review_status": "confirmed" if not issues else "disputed",
            "issues": issues,
            "notes": " ".join(notes_parts),
            "finding_title": finding_data.get("title", ""),
            "source_urls": source_urls,
            "authoritative_data_keys": (
                list(authoritative_data.keys())
                if isinstance(authoritative_data, dict)
                else []
            ),
        }
    )


@tool
def correct_finding(finding_json: str, issues_json: str) -> str:
    """Correct a finding based on identified issues.

    Takes the original finding and a list of issues, and returns a
    corrected version of the finding data. The finding is never discarded —
    if issues are severe, the corrected version reflects the accurate data.

    Args:
        finding_json: JSON string of the Finding to correct
        issues_json: JSON array of identified inaccuracy descriptions

    Returns:
        JSON string of corrected finding data, or null if finding should be discarded
    """
    try:
        finding_data = json.loads(finding_json)
        issues = json.loads(issues_json)
    except (json.JSONDecodeError, TypeError) as exc:
        return json.dumps(
            {
                "corrected": False,
                "error": f"Failed to parse input data: {exc}",
            }
        )

    if not issues:
        # No issues to correct — finding is accurate as-is
        return json.dumps(
            {
                "corrected": False,
                "reason": "No issues identified; finding is accurate.",
            }
        )

    # Return the finding data with a correction marker for the LLM to
    # fill in the corrected description and changes
    return json.dumps(
        {
            "corrected": True,
            "original_description": finding_data.get("description", ""),
            "original_proposed_changes": finding_data.get("proposed_changes", {}),
            "issues_addressed": issues,
            "corrected_description": None,  # LLM fills this in
            "corrected_changes": None,  # LLM fills this in
        }
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

REVIEW_SYSTEM_PROMPT = (
    "You are the Migration Plugin Watchdog Review Agent. Your job is to "
    "quality-check findings generated by the primary analysis agent.\n\n"
    "For each finding you review:\n"
    "1. Use verify_finding_accuracy to cross-reference the finding against "
    "authoritative data\n"
    "2. If the pre-fetched data is insufficient, use search_aws_docs_for_review "
    "to look up the specific claim in AWS documentation\n"
    "3. VERIFY that the finding includes at least one source_url. If no source "
    'is cited, mark the finding as "disputed" with the reason "no source '
    'citation provided"\n'
    "4. Classify your assessment as one of:\n"
    '   - "confirmed": The finding is accurate, well-supported by cited '
    "authoritative data\n"
    '   - "corrected": The finding has merit but contains inaccuracies — use '
    "correct_finding to fix them\n"
    '   - "disputed": You have POSITIVE EVIDENCE that the finding is wrong — '
    "you checked and the claim is incorrect\n"
    "5. Be skeptical — look for hallucinated claims, incorrect pricing numbers, "
    "wrong model names, or guidance that doesn't match the authoritative source\n"
    "6. For disputed findings, provide detailed reasoning so the human reviewer "
    "can make an informed decision\n\n"
    "CRITICAL RULE — DISPUTED vs CONFIRMED:\n"
    "- 'disputed' means you have CHECKED and found the finding to be WRONG.\n"
    "- If you cannot verify due to missing data, malformed JSON, insufficient "
    "authoritative sources, or any technical error — return 'confirmed' and note "
    "the limitation in reviewer_notes. Do NOT return 'disputed' just because "
    "you couldn't validate.\n"
    "- If the authoritative data payload has JSON errors or is malformed, "
    "use search_aws_docs_for_review to look up the claim directly instead.\n"
    "- Only dispute when you have positive evidence the finding is factually wrong.\n\n"
    "SELF-AWARE FILE CHECK (model_deprecation and staleness findings):\n"
    "Before confirming any model_deprecation or staleness finding, check whether "
    "the finding is about a file that contains 'recompute on each run' or similar "
    "self-refresh instructions. If the file already correctly marks a model as "
    "'excluded' or 'legacy' with the right EOL date, the finding is a false "
    "positive — mark it as 'disputed' with the reason: 'File already correctly "
    "handles this via its recompute-on-run instructions.'\n\n"
    "You are a different model (Nova 2 Lite) from the primary agent (Claude "
    "Opus 4.7), providing an independent perspective. Focus on factual accuracy "
    "and source grounding, not style.\n\n"
    "IMPORTANT: Do NOT discard findings. Even if you disagree, mark them as "
    '"disputed" with your reasoning. The human reviewer makes the final call.\n\n'
    "Respond with a JSON object containing:\n"
    '- "review_status": one of "confirmed", "corrected", "disputed"\n'
    '- "issues": array of identified issues (empty if confirmed)\n'
    '- "corrected_description": corrected description if status is "corrected", '
    "else null\n"
    '- "corrected_changes": corrected proposed_changes dict if status is '
    '"corrected", else null\n'
    '- "reviewer_notes": your detailed reasoning'
)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_review_agent() -> Agent:
    """Create the Strands review agent with Nova 2 Lite.
    
    Nova 2 Lite is used instead of Nova Premier based on Amazon's Nova 2
    technical report showing Nova 2 Lite surpasses Premier on multi-step
    problem-solving at 7x lower cost and up to 5x faster inference.
    Benchmark your specific workloads if you need to compare.
    """
    model = BedrockModel(
        model_id="us.amazon.nova-2-lite-v1:0",
        region_name="us-east-1",
    )
    return Agent(
        model=model,
        system_prompt=REVIEW_SYSTEM_PROMPT,
        tools=[verify_finding_accuracy, correct_finding, search_aws_docs_for_review],
    )


# ---------------------------------------------------------------------------
# Review logic
# ---------------------------------------------------------------------------


def _build_review_prompt(
    finding: Finding, authoritative_data: AuthoritativeData
) -> str:
    """Build the user message for reviewing a single finding."""
    finding_dict = {
        "finding_id": finding.finding_id,
        "risk_level": finding.risk_level.value if isinstance(finding.risk_level, RiskLevel) else finding.risk_level,
        "category": finding.category,
        "title": finding.title,
        "description": finding.description,
        "affected_files": finding.affected_files,
        "proposed_changes": finding.proposed_changes,
        "source_urls": finding.source_urls,
        "primary_reasoning": finding.primary_reasoning,
    }

    # Build a relevant subset of authoritative data based on the finding's category
    relevant_data: dict = {}
    if finding.category in ("pricing",):
        relevant_data["aws_pricing"] = authoritative_data.aws_pricing
    if finding.category in ("model_deprecation", "new_model"):
        relevant_data["gemini_models"] = {
            "models": authoritative_data.gemini_models.models,
            "pricing": authoritative_data.gemini_models.pricing,
            "deprecations": authoritative_data.gemini_models.deprecations,
        }
        relevant_data["openai_models"] = {
            "pricing": authoritative_data.openai_models.pricing,
            "deprecations": authoritative_data.openai_models.deprecations,
        }
        relevant_data["bedrock_lifecycle"] = [
            {
                "model_name": m.model_name,
                "model_id": m.model_id,
                "status": m.status,
                "eol_date": m.eol_date,
                "replacement": m.replacement,
            }
            for m in authoritative_data.bedrock_lifecycle.models
        ]
    if finding.category in ("guidance_update", "new_content", "structural", "core_removal"):
        relevant_data["aws_docs"] = authoritative_data.aws_docs
        relevant_data["aws_blog_posts"] = authoritative_data.aws_blog_posts
        relevant_data["aws_whats_new"] = authoritative_data.aws_whats_new

    # If no specific data matched, include everything for a thorough review
    if not relevant_data:
        relevant_data["aws_docs"] = authoritative_data.aws_docs
        relevant_data["aws_pricing"] = authoritative_data.aws_pricing

    return (
        "Please review the following finding for accuracy and hallucinations.\n\n"
        f"## Finding\n```json\n{json.dumps(finding_dict, indent=2)}\n```\n\n"
        f"## Authoritative Data\n```json\n{json.dumps(relevant_data, indent=2)}\n```\n\n"
        "Use verify_finding_accuracy to cross-reference the finding against the "
        "authoritative data, then provide your assessment. If you find inaccuracies, "
        "use correct_finding to fix them. Respond with your final assessment as JSON."
    )


def _parse_review_result(agent_result: object, finding: Finding) -> ReviewResult:
    """Parse the review agent's response into a ReviewResult.

    Attempts to extract structured JSON from the agent's response text.
    Falls back to a confirmed status if parsing fails, to avoid blocking
    the pipeline.
    """
    # Extract text from the agent result
    result_text = str(agent_result)

    # Try to parse JSON from the response
    review_status = "confirmed"
    issues: list[str] = []
    corrected_description: str | None = None
    corrected_changes: dict[str, str] | None = None
    reviewer_notes = result_text

    try:
        # Try to find JSON in the response
        json_start = result_text.find("{")
        json_end = result_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(result_text[json_start:json_end])
            review_status = parsed.get("review_status", "confirmed")
            issues = parsed.get("issues", [])
            corrected_description = parsed.get("corrected_description")
            corrected_changes = parsed.get("corrected_changes")
            reviewer_notes = parsed.get("reviewer_notes", result_text)
    except (json.JSONDecodeError, ValueError):
        # If we can't parse JSON, default to unverified (not confirmed) to avoid
        # inflating false positives — a failed review should not count as confirmation
        review_status = "unverified"
        logger.warning(
            "Could not parse review agent response as JSON for finding %s; "
            "defaulting to unverified to avoid false confirmation",
            finding.finding_id,
        )

    # Validate review_status is one of the expected values
    if review_status not in ("confirmed", "corrected", "disputed", "unverified"):
        review_status = "unverified"

    return ReviewResult(
        finding_id=finding.finding_id,
        review_status=review_status,
        issues=issues,
        corrected_description=corrected_description,
        corrected_changes=corrected_changes,
        reviewer_notes=reviewer_notes,
        primary_reasoning=finding.primary_reasoning or "",
    )


def apply_review(finding: Finding, review: ReviewResult) -> Finding:
    """Apply review result to a finding. Never discards — always returns a Finding.

    - "confirmed": Finding passes through unchanged, review_notes added.
    - "corrected": Finding updated with corrected description/changes,
      original values preserved in review_notes for dashboard display.
    - "disputed": Finding passes through with review_status="disputed",
      both models' reasoning attached in review_notes.

    Args:
        finding: The original finding to apply the review to.
        review: The review result from the review agent.

    Returns:
        The updated Finding — always non-None.
    """
    finding.review_status = review.review_status

    if review.review_status == "confirmed":
        finding.review_notes = review.reviewer_notes

    elif review.review_status == "corrected":
        # Preserve original values in review_notes for dashboard display
        original_info = (
            f"Original description: {finding.description}\n"
            f"Original proposed changes: {json.dumps(finding.proposed_changes)}\n\n"
            f"Reviewer notes: {review.reviewer_notes}"
        )
        finding.review_notes = original_info

        # Apply corrections
        if review.corrected_description:
            finding.description = review.corrected_description
        if review.corrected_changes:
            finding.proposed_changes = review.corrected_changes

    elif review.review_status == "disputed":
        # Attach both models' reasoning
        finding.review_notes = (
            f"Primary model reasoning: {review.primary_reasoning}\n\n"
            f"Review model reasoning: {review.reviewer_notes}"
        )
        finding.primary_reasoning = review.primary_reasoning

    return finding


def review_findings(
    findings: list[Finding], authoritative_data: AuthoritativeData
) -> list[Finding]:
    """Review medium/high-risk findings using the Strands review agent.

    Returns ALL findings — confirmed, corrected, or disputed.
    Disputed findings are NOT discarded; they pass through with
    review_status="disputed" and both models' reasoning attached.

    Low-risk findings skip the review step entirely and are passed
    through unchanged.

    Args:
        findings: List of findings to review.
        authoritative_data: Aggregated authoritative source data.

    Returns:
        All findings with review results applied to medium/high-risk ones.
    """
    agent = create_review_agent()
    reviewed: list[Finding] = []

    for finding in findings:
        # Review medium/high-risk findings AND all model_deprecation findings
        # (model_deprecation findings are often low-risk but prone to false positives
        # from self-aware files that recompute their own status on each run).
        # Also review security HIGH findings — LLM-generated security findings can
        # hallucinate; an independent review catches false positives before they
        # reach the dashboard and alarm startups unnecessarily.
        should_review = (
            finding.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)
            or finding.category in ("model_deprecation", "new_model")
            or (finding.category == "security" and finding.risk_level == RiskLevel.HIGH)
        )
        if should_review:
            logger.info(
                "Reviewing %s-risk finding %s: %s",
                finding.risk_level.value,
                finding.finding_id,
                finding.title,
            )
            try:
                prompt = _build_review_prompt(finding, authoritative_data)
                result = agent(prompt)
                review_result = _parse_review_result(result, finding)
                updated = apply_review(finding, review_result)
                reviewed.append(updated)  # Always append — never discard
            except Exception:
                logger.exception(
                    "Review agent failed for finding %s; passing through as-is",
                    finding.finding_id,
                )
                # On failure, pass the finding through unchanged
                reviewed.append(finding)
        else:
            # Low-risk: skip review, pass through unchanged
            reviewed.append(finding)

    logger.info(
        "Review complete: %d findings reviewed out of %d total",
        sum(
            1
            for f in findings
            if f.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)
            or f.category in ("model_deprecation", "new_model")
        ),
        len(findings),
    )

    return reviewed
