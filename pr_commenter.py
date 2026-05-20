"""PR comment functions for the Migration Plugin Watchdog.

Contains functions for building and posting watchdog audit comments on GitHub PRs.
Extracted verbatim from main.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import httpx

from migration_watchdog.models import Finding, RiskLevel
from migration_watchdog.retry import retry_with_backoff

logger = logging.getLogger(__name__)

# Severity ordering for threshold filtering
_SEVERITY_ORDER = {"correctness": 0, "outdated": 1, "policy_change": 2, "informational": 3}
_THRESHOLD_INCLUDE = {
    "outdated": {"correctness", "outdated"},
    "low": {"correctness", "outdated", "policy_change"},
}


def _finding_severity(finding: Finding) -> str | None:
    """Extract severity from a finding's auditor_payload, or derive from risk_level."""
    if finding.auditor_payload:
        return finding.auditor_payload.get("severity")
    # Fallback: derive from risk_level for non-auditor findings
    if finding.risk_level == RiskLevel.HIGH:
        return "correctness"
    if finding.risk_level == RiskLevel.MEDIUM:
        return "outdated"
    return "policy_change"


def _should_include_in_pr_comment(finding: Finding, severity_threshold: str) -> bool:
    """Determine whether a finding should appear in the PR comment.

    Rules:
    - Always include: correctness, outdated
    - Always include: security findings at HIGH risk (regardless of threshold)
    - Include when threshold="low": also policy_change
    - Never include: informational
    - For automation findings (category="automation_gap"): include when gap_type != "no_gap"
    """
    # Security HIGH findings always appear in PR comments — startups must see these
    if finding.category == "security" and finding.risk_level == RiskLevel.HIGH:
        return True

    # Automation findings: include when gap_type is not no_gap
    if finding.category == "automation_gap":
        gap_type = (finding.auditor_payload or {}).get("gap_type", "full_gap")
        return gap_type != "no_gap"

    severity = _finding_severity(finding)
    if severity == "informational":
        return False

    allowed = _THRESHOLD_INCLUDE.get(severity_threshold, _THRESHOLD_INCLUDE["outdated"])
    return severity in allowed


def _build_pr_comment_markdown(
    findings: list[Finding],
    pr_number: int,
    run_id: str,
    audited_files: list[str],
    severity_threshold: str,
) -> str:
    """Build the markdown body for the PR audit comment."""
    lines: list[str] = []
    lines.append("<!-- watchdog-audit-comment -->")
    lines.append("## 🔍 Watchdog Audit Report")
    lines.append("")
    lines.append(f"**Run ID:** `{run_id}`  ")
    lines.append(f"**PR:** #{pr_number}  ")
    if audited_files:
        lines.append(f"**Files audited:** {len(audited_files)}")
    lines.append("")

    # Filter findings for this comment
    visible = [f for f in findings if _should_include_in_pr_comment(f, severity_threshold)]

    if not visible:
        lines.append("✅ No findings above the severity threshold. All audited content looks good.")
        return "\n".join(lines)

    # Group by category
    currency = [f for f in visible if f.category == "currency_drift"]
    automation = [f for f in visible if f.category == "automation_gap"]
    other = [f for f in visible if f.category not in ("currency_drift", "automation_gap")]

    lines.append(f"Found **{len(visible)}** finding(s) requiring attention.")
    lines.append("")

    if currency:
        lines.append("### 💱 Currency Drift Findings")
        lines.append("")
        for finding in currency:
            payload = finding.auditor_payload or {}
            severity = payload.get("severity", "unknown")
            claim_type = payload.get("claim_type", "")
            claim_text = payload.get("claim_text", finding.title)
            suggested_fix = payload.get("suggested_fix", "")
            affected = ", ".join(f"`{f}`" for f in finding.affected_files) if finding.affected_files else "unknown"
            risk_emoji = {"correctness": "🔴", "outdated": "🟡", "policy_change": "🔵"}.get(severity, "⚪")
            lines.append(f"#### {risk_emoji} [{severity.upper()}] {finding.title}")
            lines.append(f"- **File:** {affected}")
            if claim_type:
                lines.append(f"- **Claim type:** `{claim_type}`")
            if claim_text:
                lines.append(f"- **Claim:** {claim_text[:200]}")
            if suggested_fix:
                lines.append(f"- **Suggested fix:** {suggested_fix}")
            if finding.review_status == "disputed":
                lines.append("> ⛔ Disputed by review agent — reconcile before approving.")
            lines.append("")

    if automation:
        lines.append("### 🤖 Automation Gap Findings")
        lines.append("")
        for finding in automation:
            payload = finding.auditor_payload or {}
            gap_type = payload.get("gap_type", "unknown")
            action_type = payload.get("action_type", "")
            action_text = payload.get("action_text", finding.title)
            automate_recommended = payload.get("automate_recommended", "no")
            human_gate = payload.get("human_gate")
            cli_equivalent = payload.get("cli_equivalent")
            affected = ", ".join(f"`{f}`" for f in finding.affected_files) if finding.affected_files else "unknown"
            gap_emoji = {"full_gap": "🔴", "partial_gap": "🟡"}.get(gap_type, "⚪")
            lines.append(f"#### {gap_emoji} [{gap_type.upper()}] {finding.title}")
            lines.append(f"- **File:** {affected}")
            if action_type:
                lines.append(f"- **Action type:** `{action_type}`")
            if action_text:
                lines.append(f"- **Action:** {action_text[:200]}")
            if cli_equivalent:
                lines.append(f"- **CLI equivalent:** `{cli_equivalent}`")
            lines.append(f"- **Automate recommended:** {automate_recommended}")
            if human_gate == "required":
                lines.append("> ⚠️ Requires human approval before execution")
            lines.append("")

    if other:
        lines.append("### 📋 Other Findings")
        lines.append("")
        for finding in other:
            lines.append(f"- **{finding.title}** (`{finding.category}`, risk: {finding.risk_level})")
        lines.append("")

    lines.append("---")
    lines.append(
        f"*Posted by Watchdog · [View full report in dashboard]"
        f"(https://github.com) · Run `{run_id[:8]}`*"
    )
    return "\n".join(lines)


async def _post_pr_comment(
    findings: list[Finding],
    pr_number: int,
    run_id: str,
    audited_files: list[str],
    github_token: str,
    target_owner: str,
    target_repo: str,
    severity_threshold: str = "outdated",
) -> None:
    """Post or update a single watchdog bot comment on the triggering PR.

    Uses the ``<!-- watchdog-audit-comment -->`` marker for idempotent upsert:
    finds an existing watchdog comment and updates it, or creates a new one.

    Treats 403 responses as non-fatal (logs once, does not raise).
    Also writes ``audit-report.json`` with summary data for the job summary step.

    Args:
        findings: All reviewed findings from the current run.
        pr_number: The PR number to comment on.
        run_id: The current scan run ID.
        audited_files: List of files audited in this run.
        github_token: GitHub token for API authentication.
        target_owner: Owner of the target repository.
        target_repo: Name of the target repository.
        severity_threshold: Severity threshold for filtering findings in the comment.
    """
    marker = "<!-- watchdog-audit-comment -->"
    api_base = f"https://api.github.com/repos/{target_owner}/{target_repo}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    comment_body = _build_pr_comment_markdown(
        findings, pr_number, run_id, audited_files, severity_threshold
    )

    # Write audit-report.json for the job summary step
    visible_findings = [f for f in findings if _should_include_in_pr_comment(f, severity_threshold)]
    audit_report = {
        "run_id": run_id,
        "pr_number": pr_number,
        "audited_files": audited_files,
        "total_findings": len(findings),
        "visible_findings": len(visible_findings),
        "findings_by_category": {
            "currency_drift": sum(1 for f in visible_findings if f.category == "currency_drift"),
            "automation_gap": sum(1 for f in visible_findings if f.category == "automation_gap"),
            "other": sum(1 for f in visible_findings if f.category not in ("currency_drift", "automation_gap")),
        },
        "severity_threshold": severity_threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # "status": "completed" signals to the CI policy step that the audit ran
        # successfully to completion. If this field is absent or not "completed",
        # the policy step should treat the run as incomplete.
        "status": "completed",
        # Include serialized findings for CI policy step and job summary
        "findings": [
            {
                "finding_id": f.finding_id,
                "category": f.category,
                "title": f.title,
                "description": f.description,
                "risk_level": f.risk_level.value if hasattr(f.risk_level, "value") else str(f.risk_level),
                "auditor_payload": f.auditor_payload,
                "finding_schema_version": f.finding_schema_version,
            }
            for f in findings
        ],
    }
    try:
        with open("audit-report.json", "w") as fh:
            json.dump(audit_report, fh, indent=2)
        logger.info("Wrote audit-report.json")
    except OSError as exc:
        logger.warning("Could not write audit-report.json: %s", exc)

    if not github_token:
        logger.warning("No GITHUB_TOKEN available; skipping PR comment post")
        return

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Find existing watchdog comment on the PR
        existing_comment_id: int | None = None
        try:
            async def _list_comments() -> httpx.Response:
                return await client.get(
                    f"{api_base}/issues/{pr_number}/comments",
                    headers=headers,
                    params={"per_page": 100},
                )

            resp = await retry_with_backoff(
                _list_comments,
                max_retries=2,
                retryable_exceptions=(httpx.TransportError, httpx.TimeoutException),
            )
            if resp.status_code == 403:
                logger.warning(
                    "GitHub API returned 403 when listing PR comments for PR #%d; "
                    "skipping PR comment upsert",
                    pr_number,
                )
                return
            resp.raise_for_status()
            for comment in resp.json():
                if marker in comment.get("body", ""):
                    existing_comment_id = comment["id"]
                    break
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                logger.warning(
                    "GitHub API returned 403 when listing PR comments for PR #%d; "
                    "skipping PR comment upsert",
                    pr_number,
                )
                return
            logger.exception("Failed to list PR comments for PR #%d: %s", pr_number, exc)
            return
        except Exception as exc:
            logger.exception("Failed to list PR comments for PR #%d: %s", pr_number, exc)
            return

        # Step 2: Update existing comment or create new one
        try:
            if existing_comment_id is not None:
                async def _update_comment() -> httpx.Response:
                    return await client.patch(
                        f"{api_base}/issues/comments/{existing_comment_id}",
                        headers=headers,
                        json={"body": comment_body},
                    )

                resp = await retry_with_backoff(
                    _update_comment,
                    max_retries=2,
                    retryable_exceptions=(httpx.TransportError, httpx.TimeoutException),
                )
                if resp.status_code == 403:
                    logger.warning(
                        "GitHub API returned 403 when updating comment %d on PR #%d; "
                        "skipping PR comment upsert",
                        existing_comment_id,
                        pr_number,
                    )
                    return
                resp.raise_for_status()
                logger.info(
                    "Updated existing watchdog comment %d on PR #%d",
                    existing_comment_id,
                    pr_number,
                )
            else:
                async def _create_comment() -> httpx.Response:
                    return await client.post(
                        f"{api_base}/issues/{pr_number}/comments",
                        headers=headers,
                        json={"body": comment_body},
                    )

                resp = await retry_with_backoff(
                    _create_comment,
                    max_retries=2,
                    retryable_exceptions=(httpx.TransportError, httpx.TimeoutException),
                )
                if resp.status_code == 403:
                    logger.warning(
                        "GitHub API returned 403 when creating comment on PR #%d; "
                        "skipping PR comment upsert",
                        pr_number,
                    )
                    return
                resp.raise_for_status()
                logger.info("Created new watchdog comment on PR #%d", pr_number)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                logger.warning(
                    "GitHub API returned 403 on comment upsert for PR #%d; "
                    "skipping (non-fatal)",
                    pr_number,
                )
                return
            logger.exception(
                "Failed to upsert PR comment for PR #%d: %s", pr_number, exc
            )
        except Exception as exc:
            logger.exception(
                "Failed to upsert PR comment for PR #%d: %s", pr_number, exc
            )
