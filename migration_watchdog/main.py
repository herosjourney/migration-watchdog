"""Scan pipeline orchestrator for the Migration Plugin Watchdog.

Main entry point that runs the full weekly scan pipeline sequentially
in the GitHub Actions runner. Wires all components in sequence:
record run start -> fetch repo -> fetch sources -> run analysis agent ->
deduplicate -> review medium/high findings -> persist findings ->
run refactoring assessment -> record run complete.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from uuid import uuid4

import boto3
import httpx

from migration_watchdog.analysis_agent import run_analysis
from migration_watchdog.finding_deduplicator import deduplicate
from migration_watchdog.findings_repository import FindingsRepository
from migration_watchdog.improvement_advisor import improvement_dedupe_key, run_improvement_advisor
from migration_watchdog.models import Finding, RiskLevel, ScanRun
from migration_watchdog.refactoring_agent import run_refactoring_assessment
from migration_watchdog.repo_scanner import RepoScanner
from migration_watchdog.retry import retry_with_backoff
from migration_watchdog.review_agent import review_findings
from migration_watchdog.source_fetcher import SourceFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
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
    - Include when threshold="low": also policy_change
    - Never include: informational
    - For automation findings (category="automation_gap"): include when gap_type != "no_gap"
    """
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
        "timestamp": datetime.utcnow().isoformat(),
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


def _count_by_risk(findings: list[Finding]) -> dict[str, int]:
    """Count findings by risk level.

    Args:
        findings: List of findings to count.

    Returns:
        A dict mapping risk level value strings to counts.
    """
    counts: dict[str, int] = {}
    for finding in findings:
        key = finding.risk_level.value if isinstance(finding.risk_level, RiskLevel) else str(finding.risk_level)
        counts[key] = counts.get(key, 0) + 1
    return counts


async def run_scan() -> None:
    """Execute the full weekly scan pipeline."""
    run_id = str(uuid4())
    start_timestamp = datetime.utcnow().isoformat()

    # Read configuration from environment variables
    # GitHub App credentials are required for scheduled runs (repo scanning + PR creation).
    # For PR-triggered runs, GITHUB_TOKEN is sufficient — App vars default to empty string.
    github_app_id = os.environ.get("GITHUB_APP_ID", "")
    github_app_private_key = os.environ.get("GITHUB_APP_PRIVATE_KEY", "")
    github_installation_id = os.environ.get("GITHUB_INSTALLATION_ID", "")
    dynamodb_table = os.environ.get("DYNAMODB_TABLE", "watchdog-findings")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    target_owner = os.environ.get("WATCHDOG_TARGET_OWNER", "aws-samples")
    target_repo = os.environ.get("WATCHDOG_TARGET_REPO", "sample-agent-skills-for-aws-migration")

    # Read PR / trigger environment variables
    trigger_type = os.environ.get("TRIGGER_TYPE", "scheduled")
    pr_number_str = os.environ.get("PR_NUMBER")
    pr_number = int(pr_number_str) if pr_number_str and trigger_type == "pull_request" else None
    pr_head_sha = os.environ.get("PR_HEAD_SHA") if trigger_type == "pull_request" else None
    pr_html_url = (
        os.environ.get("PR_HTML_URL")
        or (f"https://github.com/{target_owner}/{target_repo}/pull/{pr_number}" if pr_number else None)
    )
    changed_files_str = os.environ.get("CHANGED_FILES", "")
    audited_files = [f.strip() for f in changed_files_str.split() if f.strip()] if trigger_type == "pull_request" else []

    # Initialise DynamoDB resource and repository
    dynamodb_resource = boto3.resource("dynamodb", region_name=aws_region)
    findings_repo = FindingsRepository(dynamodb_resource, dynamodb_table)
    _has_aws_credentials = True

    # 1. Record run start
    logger.info("Starting scan run %s", run_id)
    try:
        findings_repo.save_run(
            ScanRun(
                run_id=run_id,
                start_timestamp=start_timestamp,
                status="running",
                trigger_type=trigger_type,
                audited_files=audited_files,
                source_pr_number=pr_number,
                source_pr_head_sha=pr_head_sha,
                source_pr_html_url=pr_html_url,
            )
        )
    except Exception as exc:
        # NoCredentialsError or any other AWS error — skip persistence for this run.
        # This happens on fork PRs where OIDC credentials are unavailable.
        logger.warning("DynamoDB unavailable (no credentials?), skipping persistence: %s", exc)
        _has_aws_credentials = False

    try:
        # 2. Fetch repo content
        # For PR-triggered runs without GitHub App credentials, use CHANGED_FILES
        # from the checkout environment rather than fetching via the GitHub API.
        if trigger_type == "pull_request" and not github_app_id:
            logger.info("PR-triggered run without App credentials — using checkout files …")
            import os as _os
            from migration_watchdog.models import RepoContent
            files: dict[str, str] = {}
            # Load changed reference files
            for file_path in audited_files:
                try:
                    with open(file_path, encoding="utf-8") as _f:
                        files[file_path] = _f.read()
                except OSError as exc:
                    logger.warning("Could not read %s: %s", file_path, exc)
            # Always merge shared artifacts needed for full audit coverage:
            # pricing-cache.md (for price claim verification) and any referenced
            # generated_artifact scripts (for gap assessment).
            _SHARED_ARTIFACTS = [
                "migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/shared/pricing-cache.md",
                "migrate/plugins/migration-to-aws/skills/gcp-to-aws/SKILL.md",
            ]
            for artifact_path in _SHARED_ARTIFACTS:
                if artifact_path not in files:
                    try:
                        with open(artifact_path, encoding="utf-8") as _f:
                            files[artifact_path] = _f.read()
                    except OSError:
                        pass  # not present in checkout — skip silently

            # Also load any generated_artifact scripts referenced in the changed
            # markdown files so gap assessment can find them.
            # Two sources:
            # 1. Inline prose patterns: generated_artifact: scripts/foo.sh
            # 2. LLM JSON output: "generated_artifact": "scripts/foo.sh"
            import re as _re
            _ARTIFACT_PATTERNS = [
                # Inline prose / YAML-style
                _re.compile(
                    r'generated_artifact["\s:]+([^\s"\'<>]+\.(?:sh|py|tf|json))',
                    _re.IGNORECASE,
                ),
                # JSON string value
                _re.compile(
                    r'"generated_artifact"\s*:\s*"([^"]+\.(?:sh|py|tf|json))"',
                    _re.IGNORECASE,
                ),
            ]
            _REPO_ROOT = _os.getcwd()  # workflow runs from repo root
            for _content in list(files.values()):
                for _pattern in _ARTIFACT_PATTERNS:
                    for _match in _pattern.finditer(_content):
                        _artifact_rel = _match.group(1).strip()
                        # Try the path as-is, then relative to features/.../references/
                        _candidates = [
                            _artifact_rel,
                            _os.path.join("migrate/plugins/migration-to-aws/skills/gcp-to-aws/references", _artifact_rel),
                        ]
                        for _candidate in _candidates:
                            if _candidate not in files:
                                try:
                                    with open(_candidate, encoding="utf-8") as _f:
                                        files[_candidate] = _f.read()
                                        logger.debug("Loaded referenced artifact: %s", _candidate)
                                    break
                                except OSError:
                                    pass  # try next candidate
            repo_content = RepoContent(files=files, open_prs=[], commit_sha="", fetched_at="")
            logger.info("Loaded %d files from checkout (%d changed + shared artifacts)", len(files), len(audited_files))
        else:
            logger.info("Fetching repo snapshot …")
            scanner = RepoScanner(github_app_id, github_app_private_key, github_installation_id)
            repo_content = await scanner.fetch_repo_snapshot(
                target_owner, target_repo
            )
            logger.info(
                "Fetched %d files and %d open PRs",
                len(repo_content.files),
                len(repo_content.open_prs),
            )
            # Parity with PR checkout mode: ensure generated_artifact scripts
            # referenced in the markdown files are also in the snapshot.
            # RepoScanner fetches references/**/*.md but not scripts under
            # references/ — add them so gap assessment works on scheduled runs.
            _extra_scripts: dict[str, str] = {}
            import re as _re2
            import base64 as _b64
            _ARTIFACT_PATTERNS_SCHED = [
                _re2.compile(r'generated_artifact["\s:]+([^\s"\'<>]+\.(?:sh|py|tf|json))', _re2.IGNORECASE),
                _re2.compile(r'"generated_artifact"\s*:\s*"([^"]+\.(?:sh|py|tf|json))"', _re2.IGNORECASE),
            ]
            # Build auth headers from the scanner's installation token
            _scanner_token = scanner._generate_installation_token()
            _gh_headers = {
                "Authorization": f"token {_scanner_token}",
                "Accept": "application/vnd.github+json",
            }
            for _path, _content in repo_content.files.items():
                for _pat in _ARTIFACT_PATTERNS_SCHED:
                    for _m in _pat.finditer(_content):
                        _art = _m.group(1).strip()
                        if _art not in repo_content.files and _art not in _extra_scripts:
                            # Try multiple path candidates (mirrors PR checkout mode):
                            # 1. Path as-is
                            # 2. Relative to features/.../references/
                            # 3. Relative to the directory containing the referencing file
                            _ref_dir = "/".join(_path.split("/")[:-1]) if "/" in _path else ""
                            _candidates = [
                                _art,
                                f"migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/{_art}",
                                f"{_ref_dir}/{_art}" if _ref_dir else None,
                            ]
                            for _candidate in _candidates:
                                if not _candidate or _candidate in repo_content.files or _candidate in _extra_scripts:
                                    continue
                                try:
                                    async with httpx.AsyncClient(timeout=15.0) as _client:
                                        _resp = await _client.get(
                                            f"https://api.github.com/repos/{target_owner}/{target_repo}/contents/{_candidate}",
                                            headers=_gh_headers,
                                        )
                                    if _resp.status_code == 200:
                                        _data = _resp.json()
                                        _extra_scripts[_candidate] = _b64.b64decode(_data.get("content", "")).decode("utf-8")
                                        logger.debug("Fetched referenced artifact for scheduled scan: %s", _candidate)
                                        break  # found it — stop trying candidates
                                except Exception as _exc:
                                    logger.debug("Could not fetch artifact %s: %s", _candidate, _exc)
            if _extra_scripts:
                repo_content.files.update(_extra_scripts)
                logger.info("Added %d referenced artifact script(s) to scheduled scan snapshot", len(_extra_scripts))

        # 3. Fetch authoritative sources
        logger.info("Fetching authoritative sources …")
        fetcher = SourceFetcher()
        authoritative_data = await fetcher.fetch_all_sources()
        if authoritative_data.partial_failures:
            logger.warning(
                "Partial source failures: %s",
                ", ".join(authoritative_data.partial_failures),
            )

        # 4. Run analysis agent
        logger.info("Running analysis agent …")
        raw_findings = run_analysis(repo_content, authoritative_data, run_id)
        logger.info("Analysis agent produced %d raw findings", len(raw_findings))

        # 4a. Run currency audit (NEW)
        # Initialize partial_source_failures early so currency/automation audit
        # failures can be appended before step 7 defines it.
        partial_source_failures: list[str] = list(authoritative_data.partial_failures)

        from migration_watchdog.currency_auditor import run_currency_audit, currency_dedupe_key
        currency_findings: list[Finding] = []
        try:
            currency_findings = await run_currency_audit(
                repo_content, authoritative_data, run_id,
                file_filter=audited_files if trigger_type == "pull_request" else None,
            )
            logger.info("Currency audit produced %d findings", len(currency_findings))
        except ValueError as exc:
            logger.error("Currency audit payload error: %s", exc)
            authoritative_data.partial_failures.append(
                f"currency_audit_payload_error: {exc}"
            )
        except Exception as exc:
            logger.exception("Currency audit failed: %s", exc)
            authoritative_data.partial_failures.append(
                f"currency_audit_error: {exc}"
            )
        # Collect partial failures from currency audit
        currency_failures = getattr(run_currency_audit, "last_partial_source_failures", [])
        for failure in currency_failures:
            # Use json.dumps for structured entries, str() for plain strings
            if isinstance(failure, dict):
                partial_source_failures.append(json.dumps(failure))
            else:
                partial_source_failures.append(str(failure))

        # 4b. Run automation audit (NEW)
        from migration_watchdog.automation_auditor import run_automation_audit, automation_dedupe_key
        automation_findings: list[Finding] = []
        try:
            automation_findings = await run_automation_audit(
                repo_content, authoritative_data, run_id,
                file_filter=audited_files if trigger_type == "pull_request" else None,
            )
            logger.info("Automation audit produced %d findings", len(automation_findings))
        except ValueError as exc:
            logger.error("Automation audit payload error: %s", exc)
            authoritative_data.partial_failures.append(
                f"automation_audit_payload_error: {exc}"
            )
        except Exception as exc:
            logger.exception("Automation audit failed: %s", exc)
            authoritative_data.partial_failures.append(
                f"automation_audit_error: {exc}"
            )

        # 4c. Run scenario simulation audit (NEW)
        from migration_watchdog.scenario_auditor import run_scenario_audit
        scenario_findings: list[Finding] = []
        try:
            scenario_findings = await run_scenario_audit(
                repo_content=repo_content,
                run_id=run_id,
                changed_files=audited_files if trigger_type == "pull_request" else None,
            )
            logger.info("Scenario audit: %d findings", len(scenario_findings))
        except Exception as exc:
            logger.exception("Scenario audit failed: %s", exc)
            authoritative_data.partial_failures.append(
                f"scenario_audit_error: {exc}"
            )

        # 4e. Run improvement advisor (scheduled runs only)
        improvement_findings: list[Finding] = []
        if trigger_type == "scheduled":
            try:
                improvement_findings = await run_improvement_advisor(
                    repo_content=repo_content,
                    run_id=run_id,
                )
                logger.info("Improvement advisor produced %d findings", len(improvement_findings))
            except Exception as exc:
                logger.warning("Improvement advisor failed: %s", exc)
                partial_source_failures.append(
                    f"improvement_advisor_error: {exc}"
                )

        # 5. Deduplicate findings with per-category keys
        logger.info("Deduplicating findings …")
        existing_findings = findings_repo.list_findings(exclude_dismissed=False) if _has_aws_credentials else []

        # Analysis findings use default key
        deduped_analysis = deduplicate(raw_findings, repo_content.open_prs, existing_findings)

        # Currency findings use claim_id key
        deduped_currency = deduplicate(
            currency_findings, repo_content.open_prs, existing_findings,
            dedupe_key_fn=currency_dedupe_key,
        )

        # Automation findings use action_fingerprint key
        deduped_automation = deduplicate(
            automation_findings, repo_content.open_prs, existing_findings,
            dedupe_key_fn=automation_dedupe_key,
        )

        # Scenario findings use default key (finding_id is deterministic SHA-256)
        deduped_scenario = deduplicate(
            scenario_findings, repo_content.open_prs, existing_findings,
        )

        # Improvement findings use improvement_dedupe_key
        deduped_improvement = deduplicate(
            improvement_findings, repo_content.open_prs, existing_findings,
            dedupe_key_fn=improvement_dedupe_key,
        )

        all_deduped = deduped_analysis + deduped_currency + deduped_automation + deduped_scenario + deduped_improvement
        logger.info(
            "Deduplication: analysis=%d, currency=%d, automation=%d, scenario=%d, improvement=%d -> total=%d findings",
            len(deduped_analysis),
            len(deduped_currency),
            len(deduped_automation),
            len(deduped_scenario),
            len(deduped_improvement),
            len(all_deduped),
        )

        # 6. Review medium/high currency findings only
        logger.info("Reviewing medium/high currency findings …")
        currency_for_review = [
            f for f in all_deduped
            if f.category == "currency_drift"
            and f.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)
        ]
        other_findings = [
            f for f in all_deduped
            if not (f.category == "currency_drift" and f.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH))
        ]
        reviewed_currency = review_findings(currency_for_review, authoritative_data)
        reviewed_findings = reviewed_currency + other_findings
        logger.info("Review complete: %d findings", len(reviewed_findings))

        # 7. Persist findings
        logger.info("Persisting %d findings …", len(reviewed_findings))
        # partial_source_failures was initialized at step 4a; additional failures
        # from persistence are appended below.
        persisted_findings: list[Finding] = []
        for finding in reviewed_findings:
            if not _has_aws_credentials:
                persisted_findings.append(finding)
                continue
            try:
                findings_repo.save_finding(finding)
                persisted_findings.append(finding)
            except ValueError as exc:
                # PayloadStore.prepare_payload() raised ValueError — payload too large
                # after both truncation passes. Log, record in partial_source_failures,
                # and skip this finding rather than crashing the run.
                logger.error(
                    "Skipping finding %s — payload too large: %s",
                    finding.finding_id,
                    exc,
                )
                partial_source_failures.append(
                    json.dumps({"type": "payload_too_large", "finding_id": finding.finding_id})
                )
        reviewed_findings = persisted_findings

        # 7b. Post PR comment for PR-triggered runs
        if trigger_type == "pull_request" and pr_number:
            await _post_pr_comment(
                reviewed_findings,
                pr_number,
                run_id,
                audited_files,
                github_token=os.environ.get("GITHUB_TOKEN", ""),
                target_owner=target_owner,
                target_repo=target_repo,
                severity_threshold=os.environ.get("WATCHDOG_SEVERITY_THRESHOLD", "outdated"),
            )

        # 7a. Clean up existing findings now covered by open PRs
        logger.info("Checking existing findings against open PRs …")
        all_pending = findings_repo.list_findings(status="pending", exclude_dismissed=True) if _has_aws_credentials else []
        from migration_watchdog.finding_deduplicator import _is_addressed_by_pr
        cleaned = 0
        for existing in all_pending:
            for pr in repo_content.open_prs:
                if _is_addressed_by_pr(existing, pr):
                    findings_repo.update_finding_status(existing.finding_id, "superseded")
                    cleaned += 1
                    logger.info(
                        "Finding %s superseded by PR #%d: %s",
                        existing.finding_id, pr.number, pr.title,
                    )
                    break
        if cleaned:
            logger.info("Cleaned up %d findings now covered by open PRs", cleaned)

        # 8. Run refactoring assessment
        logger.info("Running refactoring assessment …")
        refactoring_finding = None
        if _has_aws_credentials:
            refactoring_finding = run_refactoring_assessment(
                repo_content, existing_findings, run_id
            )
        if refactoring_finding:
            if _has_aws_credentials:
                findings_repo.save_finding(refactoring_finding)
            reviewed_findings.append(refactoring_finding)
            logger.info("Refactoring finding created: %s", refactoring_finding.finding_id)
        else:
            logger.info("No refactoring warranted")

        # 9. Record run complete
        end_timestamp = datetime.utcnow().isoformat()
        if _has_aws_credentials:
            findings_repo.save_run(
                ScanRun(
                    run_id=run_id,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    status="completed",
                    findings_count=len(reviewed_findings),
                    findings_by_risk=_count_by_risk(reviewed_findings),
                    partial_source_failures=partial_source_failures,
                    partial_data_warning=bool(partial_source_failures),
                    trigger_type=trigger_type,
                    audited_files=audited_files,
                    source_pr_number=pr_number,
                    source_pr_head_sha=pr_head_sha,
                    source_pr_html_url=pr_html_url,
                )
            )
        logger.info(
            "Scan run %s completed: %d findings (%s)",
            run_id,
            len(reviewed_findings),
            _count_by_risk(reviewed_findings),
        )

    except Exception as exc:
        logger.exception("Scan run %s failed: %s", run_id, exc)
        if _has_aws_credentials:
            findings_repo.save_run(
                ScanRun(
                    run_id=run_id,
                    start_timestamp=start_timestamp,
                    end_timestamp=datetime.utcnow().isoformat(),
                    status="failed",
                    failure_reason=str(exc),
                    trigger_type=trigger_type,
                    audited_files=audited_files,
                    source_pr_number=pr_number,
                    source_pr_head_sha=pr_head_sha,
                    source_pr_html_url=pr_html_url,
                )
            )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_scan())
