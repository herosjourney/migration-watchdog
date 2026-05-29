"""Scan pipeline orchestrator for the Migration Plugin Watchdog.

Main entry point that runs the full weekly scan pipeline sequentially
in the GitHub Actions runner. Wires all components in sequence:
record run start -> fetch repo -> fetch sources -> run analysis agent ->
deduplicate -> review medium/high findings -> persist findings ->
run refactoring assessment -> record run complete.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from uuid import uuid4

import boto3
import httpx

from migration_watchdog.analysis_agent import run_analysis
from migration_watchdog.automation_auditor import run_automation_audit, automation_dedupe_key
from migration_watchdog.currency_auditor import run_currency_audit, currency_dedupe_key
from migration_watchdog.finding_deduplicator import deduplicate, _is_addressed_by_pr
from migration_watchdog.findings_repository import FindingsRepository
from migration_watchdog.models import Finding, RepoContent, RiskLevel, ScanRun
from migration_watchdog.pr_commenter import (
    _finding_severity,
    _should_include_in_pr_comment,
    _build_pr_comment_markdown,
    _post_pr_comment,
)
from migration_watchdog.refactoring_agent import run_refactoring_assessment
from migration_watchdog.repo_scanner import RepoScanner
from migration_watchdog.retry import retry_with_backoff
from migration_watchdog.review_agent import review_findings
from migration_watchdog.scan_config import ScanConfig
from migration_watchdog.scenario_auditor import run_scenario_audit
from migration_watchdog.security_auditor import run_security_audit, security_dedupe_key
from migration_watchdog.source_fetcher import SourceFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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


def _load_artifact_scripts(
    files: dict[str, str],
    fetch_fn: Callable[[str], str | None],
) -> None:
    """Scan *files* for generated_artifact references and load them in-place.

    Args:
        files: Mutable dict of path -> content. New entries are added here.
        fetch_fn: Callable that takes a candidate path and returns file content
                  (str) if found, or None if not found.
    """
    _ARTIFACT_PATTERNS = [
        re.compile(
            r'generated_artifact["\s:]+([^\s"\'<>]+\.(?:sh|py|tf|json))',
            re.IGNORECASE,
        ),
        re.compile(
            r'"generated_artifact"\s*:\s*"([^"]+\.(?:sh|py|tf|json))"',
            re.IGNORECASE,
        ),
    ]
    for content in list(files.values()):
        for pattern in _ARTIFACT_PATTERNS:
            for match in pattern.finditer(content):
                artifact_rel = match.group(1).strip()
                candidates = [
                    artifact_rel,
                    f"migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/{artifact_rel}",
                ]
                for candidate in candidates:
                    if candidate not in files:
                        result = fetch_fn(candidate)
                        if result is not None:
                            files[candidate] = result
                            break


async def run_scan(_source_fetcher=None) -> None:
    """Execute the full weekly scan pipeline.
    
    Args:
        _source_fetcher: Optional pre-built SourceFetcher instance (used in tests).
    """
    run_id = str(uuid4())
    start_timestamp = datetime.now(timezone.utc).isoformat()

    # Read configuration from environment variables
    config = ScanConfig.from_env()

    # Initialise DynamoDB resource and repository
    dynamodb_resource = boto3.resource("dynamodb", region_name=config.aws_region)
    findings_repo = FindingsRepository(dynamodb_resource, config.dynamodb_table)
    _has_aws_credentials = True

    # 1. Record run start
    logger.info("Starting scan run %s", run_id)
    try:
        findings_repo.save_run(
            ScanRun(
                run_id=run_id,
                start_timestamp=start_timestamp,
                status="running",
                trigger_type=config.trigger_type,
                audited_files=config.changed_files,
                source_pr_number=config.pr_number,
                source_pr_head_sha=config.pr_head_sha,
                source_pr_html_url=config.pr_html_url,
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
        if config.trigger_type == "pull_request" and not config.github_app_id:
            logger.info("PR-triggered run without App credentials — using checkout files …")
            files: dict[str, str] = {}
            # Load changed reference files
            # Normalize absolute paths to relative paths so auditor pattern matching works
            # (auditors check for patterns like "gcp-to-aws/" which won't match absolute paths)
            _cwd = os.getcwd()
            for file_path in config.changed_files:
                try:
                    with open(file_path, encoding="utf-8") as _f:
                        # Store with relative path key for consistent auditor filtering
                        rel_path = os.path.relpath(file_path, _cwd) if os.path.isabs(file_path) else file_path
                        files[rel_path] = _f.read()
                except OSError as exc:
                    logger.warning("Could not read %s: %s", file_path, exc)

            # Also normalize config.changed_files for use as file_filter later —
            # currency/automation use exact key match against repo_content.files,
            # so file_filter must use the same relative path keys.
            _normalized_changed_files = [
                os.path.relpath(f, _cwd) if os.path.isabs(f) else f
                for f in config.changed_files
            ]
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
            _REPO_ROOT = os.getcwd()  # workflow runs from repo root

            def _pr_fetch_fn(path: str) -> str | None:
                try:
                    with open(path, encoding="utf-8") as _f:
                        content = _f.read()
                        logger.debug("Loaded referenced artifact: %s", path)
                        return content
                except OSError:
                    return None

            _load_artifact_scripts(files, _pr_fetch_fn)
            repo_content = RepoContent(files=files, open_prs=[], commit_sha="", fetched_at="")
            logger.info("Loaded %d files from checkout (%d changed + shared artifacts)", len(files), len(config.changed_files))
        else:
            logger.info("Fetching repo snapshot …")
            scanner = RepoScanner(config.github_app_id, config.github_app_private_key, config.github_installation_id)
            repo_content = await scanner.fetch_repo_snapshot(
                config.target_owner, config.target_repo
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
            _scanner_token = scanner._generate_installation_token()
            _gh_headers = {
                "Authorization": f"token {_scanner_token}",
                "Accept": "application/vnd.github+json",
            }

            async def _sched_fetch_fn(path: str) -> str | None:
                try:
                    async with httpx.AsyncClient(timeout=15.0) as _client:
                        _resp = await _client.get(
                            f"https://api.github.com/repos/{config.target_owner}/{config.target_repo}/contents/{path}",
                            headers=_gh_headers,
                        )
                    if _resp.status_code == 200:
                        _data = _resp.json()
                        content = base64.b64decode(_data.get("content", "")).decode("utf-8")
                        logger.debug("Fetched referenced artifact for scheduled scan: %s", path)
                        return content
                except Exception as _exc:
                    logger.debug("Could not fetch artifact %s: %s", path, _exc)
                return None

            extra_scripts: dict[str, str] = {}
            for _path, _content in repo_content.files.items():
                for _pat in [
                    re.compile(r'generated_artifact["\s:]+([^\s"\'<>]+\.(?:sh|py|tf|json))', re.IGNORECASE),
                    re.compile(r'"generated_artifact"\s*:\s*"([^"]+\.(?:sh|py|tf|json))"', re.IGNORECASE),
                ]:
                    for _m in _pat.finditer(_content):
                        _art = _m.group(1).strip()
                        if _art not in repo_content.files and _art not in extra_scripts:
                            _ref_dir = "/".join(_path.split("/")[:-1]) if "/" in _path else ""
                            _candidates = [
                                _art,
                                f"migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/{_art}",
                                f"{_ref_dir}/{_art}" if _ref_dir else None,
                            ]
                            for _candidate in _candidates:
                                if not _candidate or _candidate in repo_content.files or _candidate in extra_scripts:
                                    continue
                                result = await _sched_fetch_fn(_candidate)
                                if result is not None:
                                    extra_scripts[_candidate] = result
                                    break
            if extra_scripts:
                repo_content.files.update(extra_scripts)
                logger.info("Added %d referenced artifact script(s) to scheduled scan snapshot", len(extra_scripts))

        # 3. Fetch authoritative sources
        logger.info("Fetching authoritative sources …")
        fetcher = _source_fetcher if _source_fetcher is not None else SourceFetcher()
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

        # Normalize file_filter for PR runs — ensure relative paths match repo_content.files keys
        # For scheduled runs this is None (scan all files).
        if config.trigger_type == "pull_request":
            try:
                _normalized_changed_files
            except NameError:
                # Scheduled path or App-credential PR path — normalize from config
                _cwd2 = os.getcwd()
                _normalized_changed_files = [
                    os.path.relpath(f, _cwd2) if os.path.isabs(f) else f
                    for f in config.changed_files
                ]

        currency_findings: list[Finding] = []
        try:
            currency_findings = await run_currency_audit(
                repo_content, authoritative_data, run_id,
                file_filter=_normalized_changed_files if config.trigger_type == "pull_request" else None,
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
        automation_findings: list[Finding] = []
        try:
            automation_findings = await run_automation_audit(
                repo_content, authoritative_data, run_id,
                file_filter=_normalized_changed_files if config.trigger_type == "pull_request" else None,
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

        # 4c. Run security audit (NEW)
        security_findings: list[Finding] = []
        try:
            security_findings = await run_security_audit(
                repo_content, run_id,
                file_filter=_normalized_changed_files if config.trigger_type == "pull_request" else None,
            )
            logger.info("Security audit produced %d findings", len(security_findings))
        except Exception as exc:
            logger.exception("Security audit failed: %s", exc)
            authoritative_data.partial_failures.append(
                f"security_audit_error: {exc}"
            )

        # 4d. Run scenario simulation audit (NEW)
        scenario_findings: list[Finding] = []
        try:
            scenario_findings = await run_scenario_audit(
                repo_content=repo_content,
                run_id=run_id,
                changed_files=config.changed_files if config.trigger_type == "pull_request" else None,
            )
            logger.info("Scenario audit: %d findings", len(scenario_findings))
        except Exception as exc:
            logger.exception("Scenario audit failed: %s", exc)
            authoritative_data.partial_failures.append(
                f"scenario_audit_error: {exc}"
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

        # Security findings use issue_type key
        deduped_security = deduplicate(
            security_findings, repo_content.open_prs, existing_findings,
            dedupe_key_fn=security_dedupe_key,
        )

        # Scenario findings use default key (finding_id is deterministic SHA-256)
        deduped_scenario = deduplicate(
            scenario_findings, repo_content.open_prs, existing_findings,
        )

        all_deduped = deduped_analysis + deduped_currency + deduped_automation + deduped_security + deduped_scenario
        logger.info(
            "Deduplication: analysis=%d, currency=%d, automation=%d, security=%d, scenario=%d -> total=%d findings",
            len(deduped_analysis),
            len(deduped_currency),
            len(deduped_automation),
            len(deduped_security),
            len(deduped_scenario),
            len(all_deduped),
        )

        # 6. Review medium/high currency findings AND security HIGH findings
        # review_agent.py already handles both categories in should_review logic;
        # main.py was incorrectly filtering to currency_drift only.
        logger.info("Reviewing medium/high currency findings and security HIGH findings …")
        findings_for_review = [
            f for f in all_deduped
            if (f.category == "currency_drift" and f.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH))
            or (f.category == "security" and f.risk_level == RiskLevel.HIGH)
        ]
        other_findings = [
            f for f in all_deduped
            if f not in findings_for_review
        ]
        reviewed_batch = review_findings(findings_for_review, authoritative_data)
        reviewed_findings = reviewed_batch + other_findings
        logger.info("Review complete: %d findings (%d reviewed, %d passed through)",
                    len(reviewed_findings), len(findings_for_review), len(other_findings))

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
        if config.trigger_type == "pull_request" and config.pr_number:
            await _post_pr_comment(
                reviewed_findings,
                config.pr_number,
                run_id,
                config.changed_files,
                github_token=config.github_token,
                target_owner=config.target_owner,
                target_repo=config.target_repo,
                severity_threshold=config.severity_threshold,
            )

        # 7a. Clean up existing findings now covered by open PRs
        logger.info("Checking existing findings against open PRs …")
        all_pending = findings_repo.list_findings(status="pending", exclude_dismissed=True) if _has_aws_credentials else []
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
        end_timestamp = datetime.now(timezone.utc).isoformat()
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
                    trigger_type=config.trigger_type,
                    audited_files=config.changed_files,
                    source_pr_number=config.pr_number,
                    source_pr_head_sha=config.pr_head_sha,
                    source_pr_html_url=config.pr_html_url,
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
                    end_timestamp=datetime.now(timezone.utc).isoformat(),
                    status="failed",
                    failure_reason=str(exc),
                    trigger_type=config.trigger_type,
                    audited_files=config.changed_files,
                    source_pr_number=config.pr_number,
                    source_pr_head_sha=config.pr_head_sha,
                    source_pr_html_url=config.pr_html_url,
                )
            )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_scan())
