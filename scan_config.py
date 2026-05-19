"""Structured configuration dataclass for the Migration Plugin Watchdog scan pipeline.

Reads all configuration from environment variables via ``ScanConfig.from_env()``.
The dataclass is importable and instantiable without any AWS calls or file I/O.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ScanConfig:
    """All configuration needed to run a watchdog scan pipeline.

    Fields with defaults are optional; fields without defaults are required
    when constructing directly. Use ``from_env()`` to populate from the
    environment, which enforces required variables at startup.
    """

    # GitHub App credentials (optional — only needed for scheduled runs)
    github_app_id: str
    github_app_private_key: str
    github_installation_id: str

    # AWS / DynamoDB
    dynamodb_table: str = "watchdog-findings"
    aws_region: str = "us-east-1"

    # Target repository (required via from_env)
    target_owner: str = ""
    target_repo: str = ""

    # Trigger context
    trigger_type: str = "scheduled"
    pr_number: int | None = None
    pr_head_sha: str | None = None
    pr_html_url: str | None = None
    changed_files: list[str] = field(default_factory=list)

    # Auth token (PR-triggered runs)
    github_token: str = ""

    # Severity threshold for PR comment filtering
    severity_threshold: str = "outdated"

    @classmethod
    def from_env(cls) -> "ScanConfig":
        """Read all configuration from os.environ.

        Required variables (raises ``ValueError`` if absent or empty):
          WATCHDOG_TARGET_OWNER
          WATCHDOG_TARGET_REPO

        All other variables have defaults or are optional.
        """

        def _require(name: str) -> str:
            val = os.environ.get(name, "").strip()
            if not val:
                raise ValueError(f"Required environment variable {name!r} is not set or empty")
            return val

        target_owner = _require("WATCHDOG_TARGET_OWNER")
        target_repo = _require("WATCHDOG_TARGET_REPO")

        trigger_type = os.environ.get("TRIGGER_TYPE", "scheduled")
        pr_number_str = os.environ.get("PR_NUMBER")
        pr_number = int(pr_number_str) if pr_number_str and trigger_type == "pull_request" else None
        pr_head_sha = os.environ.get("PR_HEAD_SHA") if trigger_type == "pull_request" else None
        pr_html_url = (
            os.environ.get("PR_HTML_URL")
            or (f"https://github.com/{target_owner}/{target_repo}/pull/{pr_number}" if pr_number else None)
        )
        changed_files_str = os.environ.get("CHANGED_FILES", "")
        changed_files = (
            [f.strip() for f in changed_files_str.split() if f.strip()]
            if trigger_type == "pull_request"
            else []
        )

        return cls(
            github_app_id=os.environ.get("GITHUB_APP_ID", ""),
            github_app_private_key=os.environ.get("GITHUB_APP_PRIVATE_KEY", ""),
            github_installation_id=os.environ.get("GITHUB_INSTALLATION_ID", ""),
            dynamodb_table=os.environ.get("DYNAMODB_TABLE", "watchdog-findings"),
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            target_owner=target_owner,
            target_repo=target_repo,
            trigger_type=trigger_type,
            pr_number=pr_number,
            pr_head_sha=pr_head_sha,
            pr_html_url=pr_html_url,
            changed_files=changed_files,
            github_token=os.environ.get("GITHUB_TOKEN", ""),
            severity_threshold=os.environ.get("SEVERITY_THRESHOLD", "outdated"),
        )
