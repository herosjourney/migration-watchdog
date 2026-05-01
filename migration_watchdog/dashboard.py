"""FastAPI dashboard Lambda served via Mangum and Jinja2.

Single Lambda function behind API Gateway that serves HTML pages and
API endpoints for reviewing findings, approving/declining changes,
and viewing scan run history.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Optional

import boto3
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from mangum import Mangum

from migration_watchdog.findings_repository import FindingsRepository
from migration_watchdog.models import Finding, RiskLevel, ScanRun
from migration_watchdog.pr_creator import PRCreator

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "watchdog-findings")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
DASHBOARD_API_KEY = os.environ.get("DASHBOARD_API_KEY", "")
GITHUB_APP_ID = os.environ.get("GITHUB_APP_ID", "")
GITHUB_APP_PRIVATE_KEY_ENCRYPTED = os.environ.get("GITHUB_APP_PRIVATE_KEY_ENCRYPTED", "")
GITHUB_INSTALLATION_ID = os.environ.get("GITHUB_INSTALLATION_ID", "")

# Target repo for PR creation (fork-based flow).
# Branches are pushed to the FORK, PRs are opened against the UPSTREAM.
FORK_OWNER = "herosjourney"
UPSTREAM_OWNER = "aws-samples"
TARGET_REPO = "sample-agent-skills-for-aws-migration"
DEFAULT_BRANCH = "main"

# Risk-level ordering for sorting (higher numeric value = higher risk).
_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    "low": 0,
    "medium": 1,
    "high": 2,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decrypt_private_key(encrypted_key: str) -> str:
    """Decrypt the GitHub App private key from a KMS-encrypted env var.

    If the value does not look encrypted (starts with ``-----BEGIN``),
    return it as-is so the dashboard works in local/test environments.
    """
    if not encrypted_key:
        return ""
    if encrypted_key.startswith("-----BEGIN"):
        return encrypted_key
    try:
        import base64

        kms = boto3.client("kms", region_name=AWS_REGION)
        decrypted = kms.decrypt(CiphertextBlob=base64.b64decode(encrypted_key))
        return decrypted["Plaintext"].decode("utf-8")
    except Exception:
        return encrypted_key


def _get_findings_repository() -> FindingsRepository:
    """Create a FindingsRepository backed by DynamoDB."""
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    return FindingsRepository(dynamodb, DYNAMODB_TABLE)


def _get_pr_creator() -> PRCreator:
    """Create a PRCreator with GitHub App credentials."""
    private_key = _decrypt_private_key(GITHUB_APP_PRIVATE_KEY_ENCRYPTED)
    return PRCreator(
        github_app_id=GITHUB_APP_ID,
        app_private_key=private_key,
        installation_id=GITHUB_INSTALLATION_ID,
    )


def _finding_to_dict(finding: Finding) -> dict[str, Any]:
    """Serialise a Finding to a JSON-safe dict."""
    risk = finding.risk_level.value if isinstance(finding.risk_level, RiskLevel) else finding.risk_level
    result: dict[str, Any] = {
        "finding_id": finding.finding_id,
        "run_id": finding.run_id,
        "risk_level": risk,
        "category": finding.category,
        "title": finding.title,
        "description": finding.description,
        "affected_files": finding.affected_files,
        "proposed_changes": finding.proposed_changes,
        "source_urls": finding.source_urls,
        "scan_date": finding.scan_timestamp,
        "status": finding.status,
        "review_status": finding.review_status,
        "review_notes": finding.review_notes,
        "primary_reasoning": finding.primary_reasoning,
        "partial_data_warning": finding.partial_data_warning,
        "pr_url": finding.pr_url,
    }
    if finding.dismissal is not None:
        result["dismissal"] = {
            "dismissed_at": finding.dismissal.dismissed_at,
            "cooldown_expires": finding.dismissal.cooldown_expires,
            "reason": finding.dismissal.reason,
        }
    else:
        result["dismissal"] = None
    return result


def _run_to_dict(run: ScanRun) -> dict[str, Any]:
    """Serialise a ScanRun to a JSON-safe dict."""
    return {
        "run_id": run.run_id,
        "start_timestamp": run.start_timestamp,
        "end_timestamp": run.end_timestamp,
        "status": run.status,
        "failure_reason": run.failure_reason,
        "findings_count": run.findings_count,
        "findings_by_risk": run.findings_by_risk,
        "partial_source_failures": run.partial_source_failures,
    }


def _sort_findings(
    findings: list[Finding],
    sort_by: str = "risk_level",
    sort_order: str = "desc",
) -> list[Finding]:
    """Sort findings by the requested field and order.

    Supported ``sort_by`` values:
    - ``"risk_level"``: HIGH > MEDIUM > LOW (desc) or LOW > MEDIUM > HIGH (asc)
    - ``"scan_date"``: newest first (desc) or oldest first (asc)
    """
    reverse = sort_order.lower() == "desc"

    if sort_by == "risk_level":
        return sorted(
            findings,
            key=lambda f: _RISK_ORDER.get(f.risk_level, 1),
            reverse=reverse,
        )
    elif sort_by == "scan_date":
        return sorted(
            findings,
            key=lambda f: f.scan_timestamp or "",
            reverse=reverse,
        )
    # Default: return as-is.
    return findings


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Migration Plugin Watchdog Dashboard")


# ---------------------------------------------------------------------------
# Authentication middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Simple API-key authentication middleware.

    Checks for ``X-API-Key`` header or ``api_key`` query parameter.
    Skips authentication for the ``/health`` endpoint.
    """
    if request.url.path == "/health":
        return await call_next(request)

    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")

    if not DASHBOARD_API_KEY:
        # If no API key is configured, allow all requests (dev mode).
        return await call_next(request)

    if not api_key or api_key != DASHBOARD_API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)


# ---------------------------------------------------------------------------
# Health check (unauthenticated)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check():
    """Health check endpoint — no authentication required."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# HTML dashboard routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to the main dashboard page."""
    return HTMLResponse(
        status_code=307,
        headers={"Location": "/dashboard"},
        content="",
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the main HTML dashboard page showing all findings.

    Uses an inline HTML template (the full Jinja2 template file is
    created in task 18.2).
    """
    repo = _get_findings_repository()
    findings = repo.list_findings(exclude_dismissed=True)
    findings = _sort_findings(findings, sort_by="risk_level", sort_order="desc")

    findings_data = [_finding_to_dict(f) for f in findings]

    risk_badge = {
        "low": '<span style="color:green;">🟢 Low</span>',
        "medium": '<span style="color:orange;">🟡 Medium</span>',
        "high": '<span style="color:red;">🔴 High</span>',
    }

    rows = ""
    for f in findings_data:
        review_badge = ""
        if f.get("review_status"):
            badge_colors = {
                "confirmed": "green",
                "corrected": "orange",
                "disputed": "red",
            }
            color = badge_colors.get(f["review_status"], "gray")
            review_badge = (
                f'<span style="color:{color}; font-weight:bold;">'
                f'{f["review_status"]}</span>'
            )

        extra_info = ""
        if f.get("review_status") == "disputed" and f.get("primary_reasoning"):
            extra_info += (
                "<details><summary>Primary reasoning</summary>"
                f"<p>{f['primary_reasoning']}</p></details>"
            )
            if f.get("review_notes"):
                extra_info += (
                    "<details><summary>Review reasoning</summary>"
                    f"<p>{f['review_notes']}</p></details>"
                )
        elif f.get("review_status") == "corrected" and f.get("review_notes"):
            extra_info += (
                "<details><summary>Correction notes</summary>"
                f"<p>{f['review_notes']}</p></details>"
            )

        rows += f"""
        <tr>
            <td>{risk_badge.get(f['risk_level'], f['risk_level'])}</td>
            <td>{review_badge}</td>
            <td>{f['title']}</td>
            <td>{', '.join(f['affected_files'])}</td>
            <td>{f['scan_date']}</td>
            <td>{f['status']}</td>
            <td>
                <button onclick="approve('{f['finding_id']}')">Approve</button>
                <button onclick="decline('{f['finding_id']}')">Decline</button>
            </td>
        </tr>
        {extra_info}
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Watchdog Dashboard</title>
    <style>
        body {{ font-family: sans-serif; margin: 2rem; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
        button {{ margin: 2px; padding: 4px 10px; cursor: pointer; }}
        details {{ margin-top: 4px; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Migration Plugin Watchdog Dashboard</h1>
    <p>Total findings: {len(findings_data)}</p>
    <table>
        <thead>
            <tr>
                <th>Risk</th>
                <th>Review</th>
                <th>Title</th>
                <th>Affected Files</th>
                <th>Scan Date</th>
                <th>Status</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    <script>
        async function approve(findingId) {{
            const resp = await fetch('/api/findings/' + findingId + '/approve', {{method: 'POST'}});
            const data = await resp.json();
            alert(data.pr_url ? 'PR created: ' + data.pr_url : JSON.stringify(data));
            location.reload();
        }}
        async function decline(findingId) {{
            const resp = await fetch('/api/findings/' + findingId + '/decline', {{method: 'POST'}});
            const data = await resp.json();
            alert('Finding declined');
            location.reload();
        }}
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# JSON API routes — Findings
# ---------------------------------------------------------------------------


@app.get("/api/findings")
async def list_findings(
    status: Optional[str] = Query(default=None),
    risk_level: Optional[str] = Query(default=None),
    sort_by: str = Query(default="risk_level"),
    sort_order: str = Query(default="desc"),
):
    """List findings with optional filtering and sorting.

    Query parameters:
    - ``status``: filter by finding status (pending, approved, declined)
    - ``risk_level``: filter by risk level (low, medium, high)
    - ``sort_by``: ``"risk_level"`` (default) or ``"scan_date"``
    - ``sort_order``: ``"desc"`` (default) or ``"asc"``
    """
    repo = _get_findings_repository()
    findings = repo.list_findings(
        status=status,
        risk_level=risk_level,
        exclude_dismissed=True,
    )
    findings = _sort_findings(findings, sort_by=sort_by, sort_order=sort_order)

    return {
        "findings": [_finding_to_dict(f) for f in findings],
        "total_count": len(findings),
        "filters_applied": {
            "status": status,
            "risk_level": risk_level,
            "sort_by": sort_by,
            "sort_order": sort_order,
        },
    }


@app.get("/api/findings/{finding_id}")
async def get_finding(finding_id: str):
    """Return full details for a single finding."""
    repo = _get_findings_repository()
    finding = repo.get_finding(finding_id)
    if finding is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    return _finding_to_dict(finding)


@app.post("/api/findings/{finding_id}/approve")
async def approve_finding(finding_id: str):
    """Approve a finding: update status and trigger PR creation.

    Returns the PR URL on success.
    """
    repo = _get_findings_repository()
    finding = repo.get_finding(finding_id)
    if finding is None:
        raise HTTPException(status_code=404, detail="Finding not found")

    # Update status to approved.
    repo.update_finding_status(finding_id, "approved")
    finding.status = "approved"

    # Create a PR via PRCreator.
    pr_creator = _get_pr_creator()
    pr_url = await pr_creator.create_pr(
        finding, FORK_OWNER, UPSTREAM_OWNER, TARGET_REPO, DEFAULT_BRANCH
    )

    return {"finding_id": finding_id, "status": "approved", "pr_url": pr_url}


@app.post("/api/findings/{finding_id}/decline")
async def decline_finding(finding_id: str):
    """Decline a finding: update status and record 2-month dismissal cooldown."""
    repo = _get_findings_repository()
    finding = repo.get_finding(finding_id)
    if finding is None:
        raise HTTPException(status_code=404, detail="Finding not found")

    # Update status to declined.
    repo.update_finding_status(finding_id, "declined")

    # Record dismissal with 2-month cooldown.
    repo.record_dismissal(finding_id, cooldown_months=2)

    return {"finding_id": finding_id, "status": "declined"}


# ---------------------------------------------------------------------------
# JSON API routes — Scan Runs
# ---------------------------------------------------------------------------


@app.get("/api/runs")
async def list_runs():
    """List all scan runs.

    Scans the DynamoDB table for RUN# entities and returns them sorted
    by start_timestamp descending (newest first).
    """
    repo = _get_findings_repository()
    # The FindingsRepository doesn't expose a list_runs method, so we
    # query the table directly for RUN entities.
    table = repo._table
    from boto3.dynamodb.conditions import Attr, Key

    resp = table.scan(
        FilterExpression=Attr("SK").eq("RUN") & Attr("PK").begins_with("RUN#"),
    )
    items = resp.get("Items", [])

    runs = [repo._item_to_run(item) for item in items]
    runs.sort(key=lambda r: r.start_timestamp or "", reverse=True)

    return {"runs": [_run_to_dict(r) for r in runs], "total_count": len(runs)}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Return details for a single scan run."""
    repo = _get_findings_repository()
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_to_dict(run)


# ---------------------------------------------------------------------------
# Mangum handler for AWS Lambda
# ---------------------------------------------------------------------------

handler = Mangum(app)
