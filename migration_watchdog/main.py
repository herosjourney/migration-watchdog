"""Scan pipeline orchestrator for the Migration Plugin Watchdog.

Main entry point that runs the full weekly scan pipeline sequentially
in the GitHub Actions runner. Wires all components in sequence:
record run start -> fetch repo -> fetch sources -> run analysis agent ->
deduplicate -> review medium/high findings -> persist findings ->
run refactoring assessment -> record run complete.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime
from uuid import uuid4

import boto3

from migration_watchdog.analysis_agent import run_analysis
from migration_watchdog.finding_deduplicator import deduplicate
from migration_watchdog.findings_repository import FindingsRepository
from migration_watchdog.models import Finding, RiskLevel, ScanRun
from migration_watchdog.refactoring_agent import run_refactoring_assessment
from migration_watchdog.repo_scanner import RepoScanner
from migration_watchdog.review_agent import review_findings
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


async def run_scan() -> None:
    """Execute the full weekly scan pipeline."""
    run_id = str(uuid4())
    start_timestamp = datetime.utcnow().isoformat()

    # Read configuration from environment variables
    github_app_id = os.environ["GITHUB_APP_ID"]
    github_app_private_key = os.environ["GITHUB_APP_PRIVATE_KEY"]
    github_installation_id = os.environ["GITHUB_INSTALLATION_ID"]
    dynamodb_table = os.environ.get("DYNAMODB_TABLE", "watchdog-findings")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")

    # Initialise DynamoDB resource and repository
    dynamodb_resource = boto3.resource("dynamodb", region_name=aws_region)
    findings_repo = FindingsRepository(dynamodb_resource, dynamodb_table)

    # 1. Record run start
    logger.info("Starting scan run %s", run_id)
    findings_repo.save_run(
        ScanRun(
            run_id=run_id,
            start_timestamp=start_timestamp,
            status="running",
        )
    )

    try:
        # 2. Fetch repo content
        logger.info("Fetching repo snapshot …")
        scanner = RepoScanner(github_app_id, github_app_private_key, github_installation_id)
        repo_content = await scanner.fetch_repo_snapshot(
            "aws-samples", "sample-agent-skills-for-aws-migration"
        )
        logger.info(
            "Fetched %d files and %d open PRs",
            len(repo_content.files),
            len(repo_content.open_prs),
        )

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

        # 5. Deduplicate findings
        logger.info("Deduplicating findings …")
        existing_findings = findings_repo.list_findings(exclude_dismissed=False)
        deduped_findings = deduplicate(
            raw_findings, repo_content.open_prs, existing_findings
        )
        logger.info(
            "Deduplication: %d -> %d findings",
            len(raw_findings),
            len(deduped_findings),
        )

        # 6. Review medium/high risk findings
        logger.info("Reviewing medium/high risk findings …")
        reviewed_findings = review_findings(deduped_findings, authoritative_data)
        logger.info("Review complete: %d findings", len(reviewed_findings))

        # 7. Persist findings
        logger.info("Persisting %d findings …", len(reviewed_findings))
        for finding in reviewed_findings:
            findings_repo.save_finding(finding)

        # 7a. Clean up existing findings now covered by open PRs
        logger.info("Checking existing findings against open PRs …")
        all_pending = findings_repo.list_findings(status="pending", exclude_dismissed=True)
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
        refactoring_finding = run_refactoring_assessment(
            repo_content, existing_findings, run_id
        )
        if refactoring_finding:
            findings_repo.save_finding(refactoring_finding)
            reviewed_findings.append(refactoring_finding)
            logger.info("Refactoring finding created: %s", refactoring_finding.finding_id)
        else:
            logger.info("No refactoring warranted")

        # 9. Record run complete
        end_timestamp = datetime.utcnow().isoformat()
        findings_repo.save_run(
            ScanRun(
                run_id=run_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                status="completed",
                findings_count=len(reviewed_findings),
                findings_by_risk=_count_by_risk(reviewed_findings),
                partial_source_failures=authoritative_data.partial_failures,
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
        findings_repo.save_run(
            ScanRun(
                run_id=run_id,
                start_timestamp=start_timestamp,
                end_timestamp=datetime.utcnow().isoformat(),
                status="failed",
                failure_reason=str(exc),
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_scan())
