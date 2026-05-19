"""FastAPI dashboard Lambda served via Mangum and Jinja2.

Single Lambda function behind API Gateway that serves HTML pages and
API endpoints for reviewing findings, approving/declining changes,
and viewing scan run history.
"""

from __future__ import annotations

import html
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Optional

import boto3
from botocore.config import Config
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from mangum import Mangum

from migration_watchdog.findings_repository import FindingsRepository
from migration_watchdog.models import Finding, RiskLevel, ScanRun
from migration_watchdog.pr_creator import PRCreator

logger = logging.getLogger(__name__)

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

# Rank dicts for _sort_key() — lower rank = higher priority in sorted output.
SEVERITY_RANK: dict[str, int] = {
    "correctness": 0,
    "outdated": 1,
    "policy_change": 2,
    "informational": 3,
}

GAP_RANK: dict[str, int] = {
    "full_gap": 0,
    "partial_gap": 1,
    "no_gap": 2,
}

RISK_RANK: dict[str, int] = {
    "high": 0,
    "medium": 1,
    "low": 2,
}

# Module-level S3 payload cache to avoid redundant fetches within a request.
_payload_cache: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _e(value: str | None) -> str:
    """HTML-escape *value*, returning an empty string for ``None``."""
    if value is None:
        return ""
    return html.escape(str(value))


def _sort_key(finding: Finding) -> tuple:
    """Return a sort tuple ``(file_key, primary, risk_key)`` for ordering findings.

    - *file_key*: first affected file path (or empty string).
    - *primary*: severity rank for currency findings, gap rank for automation
      findings, or risk rank for legacy findings — all via ``.get()`` with
      fallback ``99`` so unknown values sort last.
    - *risk_key*: RISK_RANK on ``risk_level.value`` (or the raw string when
      ``risk_level`` is already a string).
    """
    file_key = finding.affected_files[0] if finding.affected_files else ""

    payload = finding.auditor_payload or {}
    schema = finding.finding_schema_version or ""

    if schema.startswith("currency/"):
        primary = SEVERITY_RANK.get(payload.get("severity"), 99)
    elif schema.startswith("automation/"):
        primary = GAP_RANK.get(payload.get("gap_type"), 99)
    else:
        # Legacy finding — use risk level.
        risk_val = (
            finding.risk_level.value
            if isinstance(finding.risk_level, RiskLevel)
            else finding.risk_level
        )
        primary = RISK_RANK.get(risk_val, 99)

    risk_val = (
        finding.risk_level.value
        if isinstance(finding.risk_level, RiskLevel)
        else finding.risk_level
    )
    risk_key = RISK_RANK.get(risk_val, 99)

    return (file_key, primary, risk_key)


def _fetch_full_payload(s3_key: str, bucket: str) -> dict | None:
    """Fetch a full auditor payload from S3, with caching and a single retry.

    Uses a short botocore timeout (5 s read / 5 s connect) to avoid blocking
    dashboard responses.  Results are cached in the module-level
    ``_payload_cache`` dict so repeated calls within the same Lambda invocation
    are free.

    Returns ``None`` on any failure and logs a warning.
    """
    cache_key = f"{bucket}/{s3_key}"
    if cache_key in _payload_cache:
        return _payload_cache[cache_key]

    s3 = boto3.client(
        "s3",
        config=Config(read_timeout=5, connect_timeout=5),
    )

    last_exc: Exception | None = None
    for attempt in range(2):  # initial attempt + one retry
        try:
            response = s3.get_object(Bucket=bucket, Key=s3_key)
            body = response["Body"].read()
            import json as _json

            data: dict = _json.loads(body)
            _payload_cache[cache_key] = data
            return data
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                logger.warning(
                    "S3 payload fetch failed (attempt 1), retrying: bucket=%s key=%s error=%s",
                    bucket,
                    s3_key,
                    exc,
                )

    logger.warning(
        "S3 payload fetch failed after retry: bucket=%s key=%s error=%s",
        bucket,
        s3_key,
        last_exc,
    )
    return None


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


def _render_list_badges(finding: Finding, dismissal_active: bool = False) -> str:
    """Render compact inline HTML badges for the main table row.

    Only shows information NOT already visible in other columns.
    Status is in the Status column — don't repeat it here.
    """
    parts: list[str] = []

    # --- Review status (only when not None and not confirmed — confirmed is the default/expected) ---
    review_status = finding.review_status
    if review_status == "disputed":
        parts.append('<span style="color:#ef9a9a; font-size:0.82em;">⛔ disputed</span>')
    elif review_status == "corrected":
        parts.append('<span style="color:#ffb74d; font-size:0.82em;">✏️ corrected</span>')

    # --- Dismissal indicator ---
    if dismissal_active and finding.status == "pending":
        parts.append('<span style="color:rgba(255,255,255,0.4); font-size:0.8em;">🔕 dismissed</span>')

    # --- PR URL link ---
    if finding.pr_url:
        parts.append(
            f'<a href="{_e(finding.pr_url)}" target="_blank" '
            f'style="font-size:0.82em; color:#90c8ff;">[Fix PR ↗]</a>'
        )

    # --- The key signal for this finding type ---
    schema = finding.finding_schema_version or ""
    payload = finding.auditor_payload or {}

    if schema.startswith("currency/"):
        severity = payload.get("severity")
        sev_map = {
            "correctness": ("🔴 Factually wrong", "#ff5252"),
            "outdated": ("🟡 Information is stale", "#ffb74d"),
            "policy_change": ("🔵 Service/feature changed", "#64b5f6"),
            "informational": ("ℹ️ Note", "#90a4ae"),
        }
        if severity in sev_map:
            label, color = sev_map[severity]
            parts.append(f'<span style="color:{color}; font-size:0.82em;">{label}</span>')

    elif schema.startswith("automation/"):
        gap_type = payload.get("gap_type")
        cli_equivalent = payload.get("cli_equivalent")
        # If a CLI equivalent exists, the point is "this CAN be automated" not "nothing exists"
        if gap_type == "full_gap":
            if cli_equivalent:
                label, color = ("🔴 Could be automated with AWS CLI", "#ff5252")
            else:
                label, color = ("🔴 No CLI equivalent found", "#ff5252")
            parts.append(f'<span style="color:{color}; font-size:0.82em;">{label}</span>')
        elif gap_type == "partial_gap":
            parts.append('<span style="color:#ffb74d; font-size:0.82em;">🟡 Partially automated</span>')
        elif gap_type == "no_gap":
            parts.append('<span style="color:#a5d6a7; font-size:0.82em;">✅ Already automated</span>')

        if payload.get("human_gate") == "required":
            parts.append('<span style="color:#ce93d8; font-size:0.82em;">🔒 needs approval</span>')

    elif finding.category == "security":
        issue_type = (payload.get("issue_type") or "").replace("_", " ")
        if issue_type:
            parts.append(f'<span style="color:#ef9a9a; font-size:0.82em;">🔒 {_e(issue_type)}</span>')

    return " ".join(parts) if parts else '<span style="color:rgba(255,255,255,0.3); font-size:0.82em;">—</span>'


def _render_currency_payload(payload: dict, finding: Finding) -> str:
    """Render a clean, readable detail card for a currency/1.0 finding.

    Designed for readability by non-technical users. Uses a card layout with
    clear section headings, generous whitespace, and plain-English labels.
    """
    import json as _jd

    sections: list[str] = []

    # --- Truncated payload warning ---
    s3_key = payload.get("auditor_payload_s3_key")
    if s3_key and not payload.get("claim_text"):
        sections.append(
            '<div style="background:rgba(255,183,77,0.12); border:1px solid rgba(255,183,77,0.35); '
            'padding:10px 14px; border-radius:8px; color:#ffb74d; font-size:0.88em;">'
            "⚠️ Showing truncated payload — full details unavailable"
            "</div>"
        )

    # --- Severity banner (most prominent element) ---
    severity = payload.get("severity")
    sev_config = {
        "correctness": {
            "bg": "rgba(255,82,82,0.12)",
            "border": "rgba(255,82,82,0.4)",
            "icon": "🔴",
            "label": "Factually Wrong",
            "desc": "This claim in the plugin is incorrect based on current AWS documentation. It needs to be fixed before users act on it.",
            "color": "#ff5252",
        },
        "outdated": {
            "bg": "rgba(255,183,77,0.1)",
            "border": "rgba(255,183,77,0.35)",
            "icon": "🟡",
            "label": "Stale Information",
            "desc": "This was accurate at some point but is no longer current. It should be updated to reflect how AWS works today.",
            "color": "#ffb74d",
        },
        "policy_change": {
            "bg": "rgba(100,181,246,0.1)",
            "border": "rgba(100,181,246,0.35)",
            "icon": "🔵",
            "label": "Details Have Changed",
            "desc": "The service or feature still exists, but something about it has changed — a name, a limit, a status, or how it works.",
            "color": "#64b5f6",
        },
        "informational": {
            "bg": "rgba(144,164,174,0.1)",
            "border": "rgba(144,164,174,0.3)",
            "icon": "ℹ️",
            "label": "Informational Note",
            "desc": "A minor discrepancy worth noting, but not urgent.",
            "color": "#90a4ae",
        },
    }
    if severity and severity in sev_config:
        cfg = sev_config[severity]
        sections.append(
            f'<div style="background:{cfg["bg"]}; border:1px solid {cfg["border"]}; '
            f'border-radius:10px; padding:14px 18px;">'
            f'<div style="font-size:1.05em; font-weight:700; color:{cfg["color"]}; margin-bottom:6px;">'
            f'{cfg["icon"]} {cfg["label"]}</div>'
            f'<div style="color:rgba(255,255,255,0.75); font-size:0.9em; line-height:1.6;">{cfg["desc"]}</div>'
            f'</div>'
        )

    # --- The claim itself ---
    claim_text = payload.get("claim_text")
    if claim_text:
        claim_type = payload.get("claim_type", "")
        type_display = {
            "feature_availability": "Feature availability",
            "service_name": "Service name",
            "region_count": "Region count",
            "region_list": "Region list",
            "price": "Price",
            "model_id": "Model ID",
            "eol_date": "End-of-life date",
            "quota_limit": "Quota limit",
            "service_limit": "Service limit",
            "preview_status": "Preview / GA status",
            "other_factual": "Factual claim",
        }
        subtype = payload.get("claim_subtype")
        type_label = type_display.get(claim_type, claim_type.replace("_", " ").title() if claim_type else "")
        if subtype:
            type_label += f" · {_e(subtype)}"

        sections.append(
            '<div style="background:rgba(255,255,255,0.05); border-radius:10px; padding:16px 18px;">'
            '<div style="font-size:0.72em; text-transform:uppercase; letter-spacing:1px; '
            'color:rgba(255,255,255,0.4); margin-bottom:8px;">What the plugin currently says</div>'
            f'<div style="font-size:1em; color:#fff; font-style:italic; line-height:1.6; '
            f'border-left:3px solid rgba(255,255,255,0.2); padding-left:12px;">'
            f'&ldquo;{_e(claim_text)}&rdquo;</div>'
            + (f'<div style="margin-top:8px; font-size:0.78em; color:rgba(255,255,255,0.35);">'
               f'Claim type: {_e(type_label)}</div>' if type_label else '')
            + '</div>'
        )

    # --- What's actually true now ---
    actual_value = payload.get("actual_value")
    suggested_fix = payload.get("suggested_fix")

    # Detect when suggested_fix is actually a "couldn't verify" message from the AI
    # rather than a real actionable fix
    _unverifiable_phrases = (
        "the documentation excerpt",
        "the provided documentation",
        "cannot be verified",
        "could not be verified",
        "additional documentation",
        "would be required",
        "does not mention",
        "does not address",
    )
    fix_is_unverifiable = suggested_fix and any(
        p in suggested_fix.lower() for p in _unverifiable_phrases
    )

    if actual_value or suggested_fix:
        if fix_is_unverifiable and not actual_value:
            # Show as a neutral "verification note" rather than a green fix card
            sections.append(
                '<div style="background:rgba(144,164,174,0.08); border:1px solid rgba(144,164,174,0.25); '
                'border-radius:10px; padding:16px 18px;">'
                '<div style="font-size:0.72em; text-transform:uppercase; letter-spacing:1px; '
                'color:rgba(144,164,174,0.7); margin-bottom:10px;">⚠️ Verification note</div>'
                '<div style="color:rgba(255,255,255,0.6); font-size:0.88em; line-height:1.7;">'
                'The AI could not find a specific AWS documentation page to verify or refute this claim. '
                'This finding may need manual review against the relevant AWS service documentation.'
                '</div>'
                '</div>'
            )
        else:
            fix_html = '<div style="background:rgba(165,214,167,0.08); border:1px solid rgba(165,214,167,0.25); border-radius:10px; padding:16px 18px;">'
            fix_html += '<div style="font-size:0.72em; text-transform:uppercase; letter-spacing:1px; color:rgba(165,214,167,0.6); margin-bottom:10px;">How to resolve this</div>'
            if actual_value:
                fix_html += (
                    f'<div style="margin-bottom:10px;">'
                    f'<div style="font-size:0.8em; color:rgba(255,255,255,0.45); margin-bottom:4px;">Current reality</div>'
                    f'<div style="color:#a5d6a7; font-size:0.92em; line-height:1.6;">{_e(actual_value)}</div>'
                    f'</div>'
                )
            if suggested_fix and not fix_is_unverifiable:
                fix_html += (
                    f'<div>'
                    f'<div style="font-size:0.8em; color:rgba(255,255,255,0.45); margin-bottom:4px;">Suggested fix</div>'
                    f'<div style="color:#c8e6c9; font-size:0.92em; line-height:1.6;">{_e(suggested_fix)}</div>'
                    f'</div>'
                )
            fix_html += '</div>'
            sections.append(fix_html)

    # --- Source and scope (compact footer row) ---
    footer_parts: list[str] = []
    verification_source = payload.get("verification_source")
    if verification_source:
        footer_parts.append(
            f'<span><span style="color:rgba(255,255,255,0.35);">Source: </span>'
            f'<a href="{_e(verification_source)}" target="_blank" style="color:#90c8ff;">'
            f'{_e(verification_source[:60])}{"…" if len(verification_source) > 60 else ""}</a></span>'
        )
    scope = payload.get("scope")
    if scope:
        footer_parts.append(
            f'<span><span style="color:rgba(255,255,255,0.35);">Scope: </span>'
            f'<span style="color:rgba(255,255,255,0.6);">{_e(scope)}</span></span>'
        )
    if footer_parts:
        sections.append(
            '<div style="display:flex; gap:20px; flex-wrap:wrap; font-size:0.82em; '
            'padding:8px 4px; border-top:1px solid rgba(255,255,255,0.06); margin-top:2px;">'
            + " · ".join(footer_parts)
            + '</div>'
        )

    # --- AI review status ---
    review_status = finding.review_status
    review_notes = finding.review_notes
    primary_reasoning = finding.primary_reasoning

    if review_status == "disputed":
        dispute_reason = review_notes or primary_reasoning or "No reason provided."
        try:
            if '\\n' in dispute_reason or '\\u' in dispute_reason:
                dispute_reason = _jd.loads(f'"{dispute_reason}"')
        except Exception:
            pass
        sections.append(
            '<div style="background:rgba(239,83,80,0.1); border:1px solid rgba(239,83,80,0.35); '
            'padding:14px 18px; border-radius:10px;">'
            '<div style="color:#ef9a9a; font-weight:700; margin-bottom:8px;">⛔ AI Review Agent Disputed This Finding</div>'
            f'<div style="color:#ffcdd2; font-size:0.88em; line-height:1.7; white-space:pre-wrap;">{_e(dispute_reason)}</div>'
            '</div>'
        )
    elif review_status == "confirmed":
        sections.append(
            '<div style="display:flex; align-items:center; gap:10px; padding:10px 14px; '
            'background:rgba(165,214,167,0.07); border-radius:8px; font-size:0.85em;">'
            '<span style="font-size:1.1em;">✅</span>'
            '<span style="color:#a5d6a7;">Independently verified by the AI review agent — this finding is accurate.</span>'
            '</div>'
        )
    elif review_status == "corrected":
        sections.append(
            '<div style="padding:10px 14px; background:rgba(255,183,77,0.08); border-radius:8px; font-size:0.85em;">'
            '<div style="color:#ffb74d; margin-bottom:4px;">✏️ Corrected by the AI review agent</div>'
            + (f'<div style="color:rgba(255,255,255,0.6); line-height:1.6;">{_e(review_notes)}</div>' if review_notes else '')
            + '</div>'
        )

    inner = '\n'.join(
        f'<div style="margin-bottom:10px;">{s}</div>' for s in sections
    )

    return (
        "<details>"
        "<summary style='cursor:pointer; color:rgba(255,255,255,0.5); font-size:0.85em; "
        "padding:8px 0; user-select:none; list-style:none; display:flex; align-items:center; gap:6px;'>"
        "<span style='font-size:0.8em;'>▶</span> View finding details"
        "</summary>"
        f'<div style="padding:16px 4px 4px 4px;">{inner}</div>'
        "</details>"
    )


def _render_automation_payload(payload: dict, finding: Finding) -> str:
    """Render a ``<details>`` panel for an automation/1.0 finding.

    Uses plain-English labels throughout — no internal field names shown to users.
    """
    parts: list[str] = []

    # --- Plain-English translations for technical values ---
    _action_type_labels = {
        "console_navigation": "Manual console step — user must navigate the AWS console",
        "cli_command": "CLI command — can be run from the terminal",
        "api_call": "API call — can be automated via code",
        "config_change": "Configuration change",
        "iam_policy": "IAM / permissions change",
        "quota_request": "Service quota increase request",
        "manual_approval": "Requires manual approval",
        "file_edit": "File or configuration edit",
    }
    _gap_type_labels = {
        "full_gap": "🔴 Not automated — no script exists for this step yet",
        "partial_gap": "🟡 Partially automated — a script exists but doesn't cover this step",
        "no_gap": "✅ Already automated — the generated script handles this",
    }
    _gap_reason_labels = {
        "missing_generated_artifact": "The setup script for this step hasn't been generated yet",
        "cli_not_in_script": "An AWS CLI command exists for this but isn't in the script",
        "no_cli_equivalent": "No AWS CLI command exists for this action",
        "human_decision_required": "This step requires a human decision before it can run",
        "not_automatable": "This type of action can't be automated",
        "partial_coverage": "The script covers part of this but not all of it",
    }
    _confidence_labels = {
        "high": "high confidence",
        "medium": "medium confidence",
        "low": "low confidence — verify manually",
    }

    # --- Manual action (the step the plugin tells users to do) ---
    action_text = payload.get("action_text")
    if action_text is not None:
        parts.append(
            f'<div style="background:rgba(255,255,255,0.05); border-radius:8px; padding:12px 16px; margin-bottom:4px;">'
            f'<div style="font-size:0.72em; text-transform:uppercase; letter-spacing:1px; color:rgba(255,255,255,0.4); margin-bottom:6px;">What the plugin tells users to do manually</div>'
            f'<div style="color:#fff; font-style:italic; line-height:1.6;">&ldquo;{_e(action_text)}&rdquo;</div>'
            f'</div>'
        )

    # --- Action type (translated) ---
    action_type = payload.get("action_type")
    if action_type is not None:
        label = _action_type_labels.get(action_type, action_type.replace("_", " ").title())
        parts.append(f'<p style="color:rgba(255,255,255,0.65); font-size:0.88em;">📋 {_e(label)}</p>')

    # --- Gap status (translated) ---
    gap_type = payload.get("gap_type")
    confidence = payload.get("confidence")
    reason = payload.get("reason")
    if gap_type is not None:
        gap_label = _gap_type_labels.get(gap_type, gap_type.replace("_", " ").title())
        conf_label = _confidence_labels.get(confidence, "") if confidence else ""
        conf_suffix = f' <span style="color:rgba(255,255,255,0.4); font-size:0.85em;">({conf_label})</span>' if conf_label else ""
        parts.append(f'<p style="font-size:0.92em;">{gap_label}{conf_suffix}</p>')
    if reason is not None:
        reason_label = _gap_reason_labels.get(reason, reason.replace("_", " ").capitalize())
        parts.append(f'<p style="color:rgba(255,255,255,0.6); font-size:0.85em;">Why: {_e(reason_label)}</p>')

    # --- Gap detail narrative ---
    partial_gap_narrative = payload.get("partial_gap_narrative")
    if partial_gap_narrative is not None:
        parts.append(f'<p style="color:rgba(255,255,255,0.7); font-size:0.88em; line-height:1.6;">{_e(partial_gap_narrative)}</p>')

    # --- CLI equivalent ---
    cli_equivalent = payload.get("cli_equivalent")
    if cli_equivalent is not None:
        parts.append(
            '<div style="margin:8px 0;">'
            '<div style="font-size:0.78em; color:rgba(255,255,255,0.45); margin-bottom:4px;">AWS CLI command that could automate this</div>'
            f'<pre style="background:rgba(0,0,0,0.4); padding:10px 14px; border-radius:6px; overflow-x:auto; margin:0;"><code style="color:#b0e0ff; font-size:0.88em;">{_e(cli_equivalent)}</code></pre>'
            '</div>'
        )
    else:
        cli_note = payload.get("cli_note")
        if cli_note is not None:
            parts.append(f'<p style="color:rgba(255,255,255,0.6); font-size:0.88em;"><em>{_e(cli_note)}</em></p>')

    # --- Reference URL ---
    reference_url = payload.get("reference_url")
    if reference_url is not None:
        parts.append(
            f'<p style="font-size:0.85em;"><span style="color:rgba(255,255,255,0.4);">Reference: </span>'
            f'<a href="{_e(reference_url)}" target="_blank" style="color:#90c8ff;">{_e(reference_url[:80])}{"…" if len(reference_url) > 80 else ""}</a></p>'
        )

    # --- Placeholders (translated) ---
    placeholder_list = payload.get("placeholder_list")
    if placeholder_list and isinstance(placeholder_list, list):
        parts.append(
            '<p style="font-size:0.85em; color:rgba(255,255,255,0.55);">'
            '📝 Values you\'ll need to fill in: '
            + ', '.join(f'<code style="background:rgba(255,255,255,0.1); padding:1px 5px; border-radius:3px; color:#b0e0ff;">{_e(str(p))}</code>' for p in placeholder_list)
            + '</p>'
        )

    # --- Plain-English assessment ---
    judgment_required = payload.get("judgment_required")
    repeated_operation = payload.get("repeated_operation")
    values_precomputed = payload.get("values_precomputed")
    safety_impact = payload.get("safety_impact")
    automate_recommended = payload.get("automate_recommended")

    assessment_lines = []
    if judgment_required is True:
        assessment_lines.append("⚠️ A human needs to make a decision before this can run — it can't be fully automated.")
    elif judgment_required is False:
        assessment_lines.append("✅ No human decision needed — this step can be fully automated.")
    if repeated_operation is True:
        assessment_lines.append("🔁 This step runs repeatedly — automating it would save significant time.")
    if values_precomputed is False and judgment_required is False:
        assessment_lines.append("📋 Some input values need to be looked up before running the CLI command.")
    elif values_precomputed is True:
        assessment_lines.append("✅ All required values are already known.")
    if safety_impact:
        safety_labels = {
            "quota_increase": "⚠️ This increases a service quota — review the new limit before automating.",
            "iam_change": "⚠️ This changes IAM permissions — review carefully before automating.",
            "data_deletion": "🔴 This deletes data — never automate without a confirmation step.",
            "cost_impact": "💰 This may incur AWS costs — verify pricing before automating.",
        }
        assessment_lines.append(safety_labels.get(safety_impact, f"⚠️ Safety note: {_e(safety_impact)}"))

    if assessment_lines:
        parts.append(
            '<div style="background:rgba(255,255,255,0.05); border-left:3px solid rgba(255,255,255,0.2); padding:10px 14px; margin:8px 0; border-radius:3px;">'
            + '<br>'.join(f'<span style="color:rgba(255,255,255,0.75); font-size:0.88em;">{line}</span>' for line in assessment_lines)
            + '</div>'
        )

    # --- Recommendation ---
    if automate_recommended is not None:
        if automate_recommended == "yes":
            parts.append('<p style="color:#a5d6a7; font-size:0.9em;">✅ Recommendation: this step should be automated.</p>')
        else:
            parts.append('<p style="color:rgba(255,255,255,0.6); font-size:0.9em;">❌ Recommendation: keep this step manual — automation is not recommended here.</p>')

    # --- Human gate warning ---
    if payload.get("human_gate") == "required":
        parts.append(
            '<div style="background:rgba(106,27,154,0.2); border-left:4px solid #ce93d8; '
            'padding:10px 14px; border-radius:4px; color:#ce93d8; font-weight:600;">'
            "⚠️ Requires human approval before this can be executed"
            "</div>"
        )

    # --- Rationale ---
    rationale = payload.get("rationale")
    if rationale is not None:
        parts.append(f'<p style="color:rgba(255,255,255,0.65); font-size:0.88em; line-height:1.6;"><strong style="color:rgba(255,255,255,0.8);">Why flag this:</strong> {_e(rationale)}</p>')

    # --- Scope ---
    scope = payload.get("scope")
    if scope is not None:
        parts.append(f'<p style="font-size:0.82em; color:rgba(255,255,255,0.4);">Scope: {_e(scope)}</p>')

    # --- Static analysis caveat ---
    parts.append(
        '<p style="font-size:0.8em; color:rgba(255,255,255,0.3); font-style:italic; margin-top:8px;">'
        "Gap detection uses pattern matching — confirm in the PR diff before acting on this."
        "</p>"
    )

    inner = "\n".join(parts)
    return (
        "<details>"
        "<summary style='cursor:pointer; color:rgba(255,255,255,0.5); font-size:0.85em; "
        "padding:8px 0; user-select:none;'>▶ View automation details</summary>"
        f'<div style="padding:14px 4px 4px 4px;">{inner}</div>'
        "</details>"
    )


def _format_description(description: str) -> str:
    """Format a finding description as readable HTML.

    Handles:
    - JSON escape sequences (\\n, \\u2014, etc.)
    - **Bold** markdown
    - `code` backtick spans
    - Numbered lists: (1) text or 1. text
    - Bullet lists: - item or * item
    - Section headers: Advantages:, Disadvantages:, etc.
    - Long prose after a section header: split on "; (" or "; " into sub-bullets
    - Paragraph breaks (double newline)
    """
    import re as _re
    import json as _json

    # Decode JSON escape sequences if present
    try:
        if '\\n' in description or '\\u' in description:
            description = _json.loads(f'"{description}"')
    except Exception:
        pass

    def format_inline(text: str) -> str:
        """Format inline markdown: **bold**, `code`."""
        text = _e(text)
        text = _re.sub(r'\*\*([^*]+)\*\*', r'<strong style="color:#fff;">\1</strong>', text)
        text = _re.sub(r'`([^`]+)`', r'<code style="background:rgba(255,255,255,0.12);padding:1px 5px;border-radius:3px;color:#b0e0ff;font-size:0.9em;">\1</code>', text)
        return text

    def split_prose_into_bullets(text: str) -> str:
        """Split long prose with semicolons into a bullet list."""
        # Split on "; (" (numbered sub-points like "; (1) ...") or "; " before capital/number
        parts = _re.split(r';\s+(?=\(?\d+\)?[\s\.]|\([A-Z]|[A-Z][a-z])', text)
        if len(parts) <= 1:
            # Try splitting on ". " followed by capital letter for very long sentences
            parts = _re.split(r'\.\s+(?=[A-Z(])', text)
        if len(parts) <= 2:
            # Not worth splitting — just return as paragraph
            return f'<p style="margin:6px 0;">{format_inline(text)}</p>'
        items = ''.join(f'<li style="margin:4px 0;">{format_inline(p.strip().rstrip(";."))}</li>' for p in parts if p.strip())
        return f'<ul style="margin:4px 0; padding-left:20px;">{items}</ul>'

    # Section headers that introduce bullet-list content on subsequent lines
    _BULLET_SECTION_HEADERS = {
        'advantages', 'disadvantages', 'pros', 'cons', 'benefits', 'drawbacks',
        'considerations', 'tradeoffs', 'trade-offs',
    }

    lines = description.split('\n')
    html_parts: list[str] = []
    in_list = False
    list_type = None
    in_bullet_section = False  # True when inside Advantages/Disadvantages block

    def close_list():
        nonlocal in_list, list_type
        if in_list:
            html_parts.append(f'</{list_type}>')
            in_list = False
            list_type = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_list()
            in_bullet_section = False
            continue

        # Section headers: "Advantages:", "Disadvantages:", "Proposed structure:", etc.
        header_match = _re.match(
            r'^\*{0,2}(Advantages|Disadvantages|Pros|Cons|Benefits|Drawbacks|Proposed Structure|Proposed structure|Summary|Background|Impact|Solution|Recommendation|Note|Considerations|Tradeoffs)s?\*{0,2}:\s*(.*)$',
            stripped, _re.IGNORECASE
        )
        if header_match:
            close_list()
            label = header_match.group(1).title()
            rest = header_match.group(2).strip()
            # Use different styling for Proposed Structure vs Advantages/Disadvantages
            is_bullet_section = label.lower() in _BULLET_SECTION_HEADERS
            is_proposal = 'proposed' in label.lower()
            if is_proposal:
                html_parts.append(
                    f'<p style="margin:16px 0 8px 0; font-weight:bold; color:#90c8ff; '
                    f'border-bottom:1px solid rgba(100,160,255,0.25); padding-bottom:4px; font-size:1em;">📋 {_e(label)}:</p>'
                )
            else:
                html_parts.append(
                    f'<p style="margin:12px 0 4px 0; font-weight:bold; color:rgba(255,255,255,0.9); '
                    f'border-bottom:1px solid rgba(255,255,255,0.12); padding-bottom:2px;">{_e(label)}:</p>'
                )
            in_bullet_section = is_bullet_section and not is_proposal
            if rest:
                if in_bullet_section:
                    html_parts.append(f'<ul style="margin:4px 0; padding-left:20px;">')
                    html_parts.append(f'<li style="margin:6px 0; line-height:1.6;">{format_inline(rest)}</li>')
                    in_list = True
                    list_type = 'ul'
                else:
                    html_parts.append(split_prose_into_bullets(rest))
            elif in_bullet_section:
                html_parts.append(f'<ul style="margin:4px 0; padding-left:20px;">')
                in_list = True
                list_type = 'ul'
            continue

        # Numbered list items: "(1) text" or "1. text"
        num_match = _re.match(r'^[\(\[]?(\d+)[\)\]\.]\s+(.+)$', stripped)
        if num_match:
            if not in_list or list_type != 'ol':
                close_list()
                html_parts.append('<ol style="margin:4px 0; padding-left:20px;">')
                in_list = True
                list_type = 'ol'
            html_parts.append(f'<li style="margin:4px 0;">{format_inline(num_match.group(2))}</li>')
            in_bullet_section = False
            continue

        # Bullet list items: "- text" or "* text"
        bullet_match = _re.match(r'^[-*•]\s+(.+)$', stripped)
        if bullet_match:
            if not in_list or list_type != 'ul':
                close_list()
                html_parts.append('<ul style="margin:4px 0; padding-left:20px;">')
                in_list = True
                list_type = 'ul'
            html_parts.append(f'<li style="margin:6px 0; line-height:1.6;">{format_inline(bullet_match.group(1))}</li>')
            continue

        # Inside an Advantages/Disadvantages section — each paragraph is a bullet
        if in_bullet_section:
            if not in_list or list_type != 'ul':
                close_list()
                html_parts.append('<ul style="margin:4px 0; padding-left:20px;">')
                in_list = True
                list_type = 'ul'
            html_parts.append(f'<li style="margin:8px 0; line-height:1.7;">{format_inline(stripped)}</li>')
            continue

        # Long prose line — if it's long and has semicolons, split into bullets
        close_list()
        if len(stripped) > 200 and '; ' in stripped:
            html_parts.append(split_prose_into_bullets(stripped))
        else:
            html_parts.append(f'<p style="margin:6px 0; line-height:1.6;">{format_inline(stripped)}</p>')

    close_list()
    return '\n'.join(html_parts)


def _render_general_finding_detail(finding: Finding) -> str:
    """Render a ``<details>`` panel for non-currency/automation findings."""
    parts: list[str] = []

    # --- Description — split into intro + structured sections for readability ---
    description = finding.description
    if description:
        import re as _re_desc
        import json as _json_desc
        # Decode JSON escape sequences
        try:
            if '\\n' in description or '\\u' in description:
                description = _json_desc.loads(f'"{description}"')
        except Exception:
            pass

        # Split the description at the first section header (Advantages:, Disadvantages:, etc.)
        # Everything before it is the "What's being proposed" intro
        # Handles both plain "Advantages:" and bold "**Advantages:**" markdown
        section_split = _re_desc.split(
            r'(?=\*{0,2}(?:Advantages|Disadvantages|Pros|Cons|Benefits|Drawbacks|Proposed Structure|Proposed structure|Summary|Background|Impact|Solution|Recommendation|Considerations|Tradeoffs)s?\*{0,2}:)',
            description, maxsplit=1
        )
        intro = section_split[0].strip()
        rest = section_split[1].strip() if len(section_split) > 1 else ''
        # Strip any leading ** left over from bold markdown headers like **Advantages:**
        rest = rest.lstrip('*').strip()

        if intro:
            parts.append(
                '<div style="background:rgba(100,160,255,0.08); border-left:3px solid rgba(100,160,255,0.4); '
                'padding:12px 16px; border-radius:0 8px 8px 0; margin-bottom:12px;">'
                '<div style="font-size:0.72em; text-transform:uppercase; letter-spacing:1px; '
                'color:rgba(100,160,255,0.7); margin-bottom:8px;">What\'s being proposed</div>'
                f'<div style="color:rgba(255,255,255,0.85); font-size:0.9em; line-height:1.7;">{_format_description(intro)}</div>'
                '</div>'
            )
        if rest:
            parts.append(f'<div style="margin-top:4px;">{_format_description(rest)}</div>')

    # --- Dispute reason (shown prominently when disputed) ---
    review_status = finding.review_status
    review_notes = finding.review_notes
    primary_reasoning = finding.primary_reasoning

    if review_status == "disputed":
        dispute_reason = review_notes or primary_reasoning or "No reason provided."
        # Decode escape sequences in dispute reason too
        try:
            import json as _json2
            if '\\n' in dispute_reason or '\\u' in dispute_reason:
                dispute_reason = _json2.loads(f'"{dispute_reason}"')
        except Exception:
            pass
        parts.append(
            '<div style="background:rgba(198,40,40,0.15); border-left:4px solid #ef5350; '
            'padding:10px 14px; margin:8px 0; border-radius:4px;">'
            '<strong style="color:#ef9a9a;">⛔ Disputed by review agent</strong><br>'
            f'<span style="color:rgba(255,255,255,0.75); font-size:0.9em; white-space:pre-wrap;">{_e(dispute_reason)}</span>'
            '</div>'
        )
        if primary_reasoning and review_notes and primary_reasoning != review_notes:
            parts.append(
                '<details><summary style="cursor:pointer; color:rgba(255,255,255,0.5);">Primary agent reasoning</summary>'
                f'<p style="font-size:0.9em; white-space:pre-wrap; color:rgba(255,255,255,0.7);">{_e(primary_reasoning)}</p>'
                '</details>'
            )

    # --- Impact of leaving unresolved ---
    category = finding.category or ""
    impact_map = {
        "refactoring": "Without this change, the plugin's reference directory will continue to accumulate overlapping files with no clear taxonomy, making it harder to maintain and increasing the risk of contradictory guidance.",
        "new_content": "Without this content, users migrating this workload type will receive incomplete guidance and may miss important AWS services or patterns.",
        "guidance_update": "Without this update, the plugin may recommend outdated services, deprecated features, or incorrect configurations to users.",
        "pricing": "Without this update, cost estimates generated by the plugin will be inaccurate, potentially causing users to over- or under-provision resources.",
        "model_deprecation": "Without this update, the plugin may recommend models that are deprecated or unavailable, causing migration failures.",
        "structural": "Without this change, the plugin's structure may cause confusion or errors for users following the migration guidance.",
    }
    impact = impact_map.get(category, "")
    if impact:
        parts.append(
            '<div style="background:rgba(249,168,37,0.12); border-left:3px solid #f9a825; padding:8px 12px; margin:6px 0; border-radius:3px;">'
            f'<strong style="color:#ffb74d;">⚠️ Impact if unresolved:</strong> <span style="color:rgba(255,255,255,0.75);">{impact}</span>'
            '</div>'
        )

    # --- Proposed changes ---
    proposed_changes = finding.proposed_changes
    if proposed_changes and isinstance(proposed_changes, dict):
        parts.append('<p><strong>Proposed changes:</strong></p>')
        for file_path, change in proposed_changes.items():
            change_str = str(change)
            # Decode escape sequences
            try:
                import json as _json_c
                if '\\n' in change_str or '\\u' in change_str:
                    change_str = _json_c.loads(f'"{change_str}"')
            except Exception:
                pass
            short = change_str[:300]
            rest = change_str[300:]
            change_html = _e(short)
            if rest:
                change_html += (
                    f'<span id="more-{hash(file_path)}" style="display:none;">{_e(rest)}</span>'
                    f' <a href="#" onclick="document.getElementById(\'more-{hash(file_path)}\').style.display=\'inline\';this.style.display=\'none\';return false;" style="font-size:0.85em;">[show more]</a>'
                )
            parts.append(
                f'<div style="margin:6px 0; padding:8px 10px; background:rgba(255,255,255,0.06); border-radius:4px; border-left:3px solid #5c9aff;">'
                f'<div style="margin-bottom:4px; color:rgba(255,255,255,0.85); font-size:0.88em; line-height:1.6;">{change_html}</div>'
                f'<div style="font-size:0.78em; color:rgba(255,255,255,0.4); font-family:monospace;">{_e(file_path)}</div>'
                f'</div>'
            )

    # --- Source URLs ---
    source_urls = finding.source_urls
    if source_urls:
        parts.append('<p><strong>Sources:</strong></p><ul>')
        for url in source_urls:
            parts.append(f'<li><a href="{_e(url)}" target="_blank">{_e(url)}</a></li>')
        parts.append('</ul>')

    # --- Corrected notes ---
    if review_status == "corrected" and review_notes:
        parts.append(f'<p><strong>Review correction notes:</strong> {_e(review_notes)}</p>')

    if not parts:
        return ""

    inner = "\n".join(parts)
    return (
        "<details open>"
        "<summary style='cursor:pointer; font-weight:bold; color:#90c8ff;'>Finding Details</summary>"
        f'<div style="padding:8px 12px; border:1px solid rgba(255,255,255,0.1); border-radius:4px; margin-top:4px; background:rgba(0,0,0,0.35); color:#e0e0e0;">{inner}</div>'
        "</details>"
    )


def _render_auditor_payload(
    payload: Optional[dict],
    schema_version: Optional[str],
    finding: Finding,
) -> str:
    """Dispatch to the correct payload renderer based on ``schema_version``.

    - ``schema_version`` starting with ``"currency/"`` → ``_render_currency_payload()``
    - ``schema_version`` starting with ``"automation/"`` → ``_render_automation_payload()``
    - All other findings → ``_render_general_finding_detail()``
    """
    if schema_version and schema_version.startswith("currency/") and payload:
        return _render_currency_payload(payload, finding)
    if schema_version and schema_version.startswith("automation/") and payload:
        return _render_automation_payload(payload, finding)
    # For analysis, refactoring, new_content, guidance_update, etc.
    return _render_general_finding_detail(finding)


def _render_pr_sticky_banner(scan_run: ScanRun | None) -> str:
    """Render a sticky top banner when the run is associated with a PR.

    Returns an HTML string with the PR number, HEAD SHA, and links to the
    PR and its Files-changed tab.  Returns ``""`` when ``scan_run`` is
    ``None`` or ``source_pr_number`` is not set.

    All user-derived values are escaped with ``_e()``.
    """
    if scan_run is None or scan_run.source_pr_number is None:
        return ""

    pr_number = _e(str(scan_run.source_pr_number))
    head_sha = _e(str(scan_run.source_pr_head_sha or ""))
    # Truncate SHA to 7 chars for display (standard git short SHA).
    short_sha = head_sha[:7] if head_sha else ""
    pr_url = _e(scan_run.source_pr_html_url or "")
    files_url = f"{pr_url}/files" if pr_url else ""

    links = ""
    if pr_url:
        links += f' <a href="{pr_url}" target="_blank">[View PR ↗]</a>'
    if files_url:
        links += f' <a href="{files_url}" target="_blank">[Files changed ↗]</a>'

    sha_display = f" @ {short_sha}" if short_sha else ""

    return (
        '<div style="position:sticky; top:0; background:#e3f2fd; '
        'border-bottom:2px solid #1565c0; padding:8px 16px; z-index:100;">'
        f"🔍 Audit for PR #{pr_number}{sha_display}"
        f"{links}"
        "</div>"
    )


def _render_partial_data_warning(scan_run: ScanRun | None) -> str:
    """Render a page-level partial-data warning banner.

    Shown only when the run is scoped (``scan_run`` is not ``None``) and
    ``scan_run.partial_data_warning`` is ``True``.  Returns ``""`` otherwise.
    """
    if scan_run is None or not scan_run.partial_data_warning:
        return ""
    return (
        '<div style="background:#fff3e0; border-left:4px solid #e65100; '
        'padding:8px 12px; margin:8px 0;">'
        "⚠️ Partial data warning: some source fetches failed during this scan run. "
        "Some findings may be incomplete."
        "</div>"
    )


def _render_merge_checklist(
    scan_run: ScanRun | None,
    findings: list[Finding],
) -> str:
    """Render an optional merge checklist panel for PR-triggered runs.

    Shown only when ``scan_run`` is not ``None`` and
    ``scan_run.source_pr_number`` is set.  Counts are derived from the
    current ``findings`` list.

    Checklist items:
    - ✅/❌ N correctness findings with disputed review
    - ✅/❌ N human_gate=required findings with automate_recommended=yes and status=pending
    - ⚠️  N outdated findings (informational)
    - ℹ️  N automation gaps identified
    - ℹ️  N human_gate=required findings declined (informational)

    ✅ when count == 0, ❌ when count > 0 for blocking items.
    All user-derived values are escaped with ``_e()``.
    """
    if scan_run is None or scan_run.source_pr_number is None:
        return ""

    pr_number = _e(str(scan_run.source_pr_number))

    # --- Derive counts from findings list ---
    correctness_disputed = 0
    human_gate_pending = 0
    outdated_count = 0
    automation_gap_count = 0
    human_gate_declined = 0

    for f in findings:
        payload = f.auditor_payload or {}
        schema = f.finding_schema_version or ""

        if schema.startswith("currency/"):
            severity = payload.get("severity")
            if severity == "correctness" and f.review_status == "disputed":
                correctness_disputed += 1
            if severity == "outdated":
                outdated_count += 1

        if schema.startswith("automation/"):
            automation_gap_count += 1
            if (
                payload.get("human_gate") == "required"
                and payload.get("automate_recommended") == "yes"
                and f.status == "pending"
            ):
                human_gate_pending += 1
            if payload.get("human_gate") == "required" and f.status == "declined":
                human_gate_declined += 1

    def _blocking_icon(count: int) -> str:
        return "✅" if count == 0 else "❌"

    items = (
        f"<li>{_blocking_icon(correctness_disputed)} "
        f"{_e(str(correctness_disputed))} correctness findings with disputed review</li>"
        f"<li>{_blocking_icon(human_gate_pending)} "
        f"{_e(str(human_gate_pending))} human_gate=required findings with "
        f"automate_recommended=yes and status=pending</li>"
        f"<li>⚠️ {_e(str(outdated_count))} outdated findings (informational)</li>"
        f"<li>ℹ️ {_e(str(automation_gap_count))} automation gaps identified</li>"
        f"<li>ℹ️ {_e(str(human_gate_declined))} human_gate=required findings declined "
        f"(informational)</li>"
    )

    return (
        '<div style="background:#f5f5f5; border:1px solid #ddd; '
        'padding:12px; margin:12px 0;">'
        f"<h3>PR #{pr_number} Audit Summary</h3>"
        f"<ul>{items}</ul>"
        "</div>"
    )


def _finding_to_dict(finding: Finding) -> dict[str, Any]:
    """Serialise a Finding to a JSON-safe dict.

    Includes ``dismissal`` cooldown info so the badge layer can show the
    ``🔕 dismissed (cooldown)`` indicator when a dismissal is active.
    """
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
        "auditor_payload": finding.auditor_payload,
        "finding_schema_version": finding.finding_schema_version,
    }
    if finding.dismissal is not None:
        # Include full cooldown info so the badge can show the dismissed indicator.
        result["dismissal"] = {
            "dismissed_at": finding.dismissal.dismissed_at,
            "cooldown_expires": finding.dismissal.cooldown_expires,
            "reason": finding.dismissal.reason,
        }
        # Compute whether the dismissal cooldown is still active.
        try:
            from datetime import timezone as _tz
            expires = datetime.fromisoformat(finding.dismissal.cooldown_expires)
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=_tz.utc)
            result["dismissal_active"] = datetime.now(_tz.utc) < expires
        except (ValueError, TypeError):
            result["dismissal_active"] = False
    else:
        result["dismissal"] = None
        result["dismissal_active"] = False
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
async def dashboard_page(run_id: Optional[str] = Query(default=None)):
    """Serve the main HTML dashboard page showing all findings.

    When ``run_id`` is provided, findings are scoped to that scan run and
    optional PR-context banners are rendered (sticky PR banner, merge
    checklist, partial-data warning).

    Uses an inline HTML template (the full Jinja2 template file is
    created in task 18.2).
    """
    repo = _get_findings_repository()

    # --- Run-scoped vs global view ---
    scan_run: ScanRun | None = None
    if run_id is not None:
        scan_run = repo.get_run(run_id)
        findings = repo.list_findings(run_id=run_id, exclude_dismissed=True)
    else:
        findings = repo.list_findings(exclude_dismissed=True)

    findings = [f for f in findings if f.status not in ("superseded", "declined")]
    # Use _sort_key() for deterministic, schema-aware ordering.
    findings = sorted(findings, key=_sort_key)

    # Resolve S3 payload bucket from environment (may be empty in dev).
    payload_bucket = os.environ.get("WATCHDOG_PAYLOAD_BUCKET", "")

    # Enrich findings: fetch full S3 payload when auditor_payload_s3_key is present.
    enriched_findings: list[Finding] = []
    for f in findings:
        if (
            f.auditor_payload
            and f.auditor_payload.get("auditor_payload_s3_key")
            and payload_bucket
        ):
            s3_key = f.auditor_payload["auditor_payload_s3_key"]
            full_payload = _fetch_full_payload(s3_key, payload_bucket)
            if full_payload is not None:
                import copy as _copy
                f = _copy.copy(f)
                f.auditor_payload = full_payload
        enriched_findings.append(f)

    findings_data = [_finding_to_dict(f) for f in enriched_findings]

    # Group findings by agent
    agent_groups = {
        "currency": {
            "label": "💱 Currency Auditor",
            "description": "Checks factual claims in the reference files against live AWS documentation. Flags things like wrong region counts, deprecated model IDs, stale prices, and services that are no longer available to new customers. These findings are the most urgent — a correctness finding means the plugin is giving users wrong information right now.",
            "categories": {"currency_drift"},
            "findings": [],
        },
        "automation": {
            "label": "🤖 Automation Auditor",
            "description": "Reads migration guides looking for steps that tell users to do things manually — like 'go to the AWS console and click X'. For each manual step, it checks whether an AWS CLI command exists that does the same thing, and whether the generated setup scripts already use it. These findings help make the plugin progressively more automated.",
            "categories": {"automation_gap"},
            "findings": [],
        },
        "analysis": {
            "label": "🔍 Analysis Agent",
            "description": "Compares the plugin's guidance against current AWS best practices, pricing data, and model information. Flags outdated service recommendations, stale pricing tables, deprecated AI models, and missing content areas. These findings keep the plugin's advice aligned with how AWS actually works today.",
            "categories": {"pricing", "model_deprecation", "new_model", "guidance_update", "new_content", "structural", "core_removal"},
            "findings": [],
        },
        "refactoring": {
            "label": "🏗️ Refactoring Agent",
            "description": "Looks at the overall structure of the plugin's reference files and identifies patterns that suggest the organization could be improved. These findings are optional — they're suggestions for making the plugin easier to maintain, not urgent fixes. Previous refactoring suggestions have been declined, so review carefully before approving.",
            "categories": {"refactoring"},
            "findings": [],
        },
        "security": {
            "label": "🔒 Security Auditor",
            "description": "Scans reference files that describe generated code (scripts, Terraform, Python adapters) for security vulnerabilities. Flags open admin ports, secrets in shell variables, plaintext database passwords, missing deletion protection, and other security anti-patterns in generated infrastructure. These findings should be prioritized — they represent real vulnerabilities in code the plugin generates for users.",
            "categories": {"security"},
            "findings": [],
        },
    }

    for f, finding_obj in zip(findings_data, enriched_findings):
        category = f.get("category", "")
        placed = False
        for group in agent_groups.values():
            if category in group["categories"]:
                group["findings"].append((f, finding_obj))
                placed = True
                break
        if not placed:
            agent_groups["analysis"]["findings"].append((f, finding_obj))

    def render_findings_table(group_findings):
        """Render a findings table for a group."""
        if not group_findings:
            return '<p style="color:#888; font-style:italic;">No findings in this category.</p>'
        table_rows = ""
        for f, finding_obj in group_findings:
            review_badge = ""
            if f.get("review_status"):
                badge_colors = {
                    "confirmed": "#a5d6a7",
                    "corrected": "#ffb74d",
                    "disputed": "#ef9a9a",
                }
                badge_labels = {
                    "confirmed": "✅ verified",
                    "corrected": "✏️ corrected",
                    "disputed": "⛔ disputed",
                }
                badge_titles = {
                    "confirmed": "The AI review agent independently verified this finding is accurate",
                    "corrected": "The AI review agent found inaccuracies and corrected the finding",
                    "disputed": "The AI review agent disagrees — review carefully before approving",
                }
                rs = f["review_status"]
                color = badge_colors.get(rs, "#aaa")
                label = badge_labels.get(rs, rs)
                title = badge_titles.get(rs, "")
                review_badge = (
                    f'<span style="color:{color}; font-size:0.82em;" title="{_e(title)}">{label}</span>'
                )

            extra_info = ""
            if f.get("review_status") == "disputed" and f.get("primary_reasoning"):
                extra_info += (
                    "<details><summary>Primary reasoning</summary>"
                    f"<p>{_e(f['primary_reasoning'])}</p></details>"
                )
                if f.get("review_notes"):
                    extra_info += (
                        "<details><summary>Review reasoning</summary>"
                        f"<p>{_e(f['review_notes'])}</p></details>"
                    )
            elif f.get("review_status") == "corrected" and f.get("review_notes"):
                extra_info += (
                    "<details><summary>Correction notes</summary>"
                    f"<p>{_e(f['review_notes'])}</p></details>"
                )

            payload_html = _render_auditor_payload(
                f.get("auditor_payload"),
                f.get("finding_schema_version"),
                finding_obj,
            )
            if payload_html:
                extra_info += payload_html

            dismissal_active = bool(f.get("dismissal_active", False))
            badges_html = _render_list_badges(finding_obj, dismissal_active=dismissal_active)

            risk_badge_map = {
                "low": '<span style="color:#a5d6a7;">🟢 Low</span>',
                "medium": '<span style="color:#ffb74d;">🟡 Medium</span>',
                "high": '<span style="color:#ff5252;">🔴 High</span>',
            }

            # Format scan date as "May 18, 2026" — slice YYYY-MM-DD then reformat
            raw_date = f.get('scan_date', '')
            try:
                y, m, d = raw_date[:10].split('-')
                month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                formatted_date = f"{month_names[int(m)-1]} {int(d)}, {y}"
            except Exception:
                formatted_date = raw_date[:10] if raw_date else ''

            # Clean up the title for display:
            # Automation titles like "Automation gap: copy_paste in some/long/path.md"
            # are redundant since the file path is already in Affected Files.
            # Strip the filename suffix and translate action_type to plain English.
            _action_type_plain = {
                "copy_paste": "copy and paste step",
                "console_navigation": "manual console step",
                "cli_command": "CLI command step",
                "api_call": "API call step",
                "config_change": "configuration change",
                "iam_policy": "IAM permissions change",
                "quota_request": "quota increase request",
                "manual_approval": "manual approval step",
                "file_edit": "file edit step",
            }
            display_title = f.get('title', '')
            schema = f.get('finding_schema_version', '')
            _claim_type_plain = {
                "feature_availability": "Feature availability",
                "service_name": "Service name",
                "region_count": "Region count",
                "region_list": "Region list",
                "price": "Price",
                "model_id": "Model ID",
                "eol_date": "End-of-life date",
                "quota_limit": "Quota limit",
                "service_limit": "Service limit",
                "preview_status": "Preview / GA status",
                "other_factual": "Factual claim",
            }
            if schema and schema.startswith('currency/'):
                import re as _re_title
                payload_data = f.get('auditor_payload') or {}
                claim_text = payload_data.get('claim_text', '')
                claim_type = payload_data.get('claim_type', '')
                # Translate claim_type to plain English
                plain_type = _claim_type_plain.get(claim_type, claim_type.replace('_', ' ').title() if claim_type else '')
                # Strip everything from " in filename.md" to end (handles multi-word claim types)
                display_title = _re_title.sub(r'\s+in\s+\S+$', '', display_title)
                # Replace "Currency drift: <anything>" with "Stale claim: <plain_type>"
                display_title = _re_title.sub(r'^Currency drift:.*$', f'Stale claim: {plain_type}', display_title)
                # Append the actual claim text — no truncation, let the column wrap
                if claim_text:
                    display_title = f'{display_title} — "{claim_text}"'
            elif schema and schema.startswith('automation/'):
                import re as _re_title
                # Strip " in path/to/file.md" suffix
                display_title = _re_title.sub(r'\s+in\s+\S+\.md$', '', display_title)
                # Translate action_type slug in title (e.g. "copy_paste" → "copy and paste step")
                for slug, plain in _action_type_plain.items():
                    display_title = display_title.replace(slug, plain)
                # Clean up any remaining underscores
                display_title = display_title.replace('_', ' ')
                # Append the action_text so the user knows what specific step is being flagged
                action_text = (f.get('auditor_payload') or {}).get('action_text', '')
                if action_text:
                    short_action = action_text[:80] + ('…' if len(action_text) > 80 else '')
                    display_title = f'{display_title} — {short_action}'            # Status with color
            status_val = f.get('status', 'pending')
            status_colors = {
                "pending": "#b0bec5",
                "approved": "#a5d6a7",
                "declined": "#ef9a9a",
            }
            status_color = status_colors.get(status_val, "#b0bec5")
            status_html = f'<span style="color:{status_color};">{_e(status_val)}</span>'

            table_rows += f"""
            <tr>
                <td>{risk_badge_map.get(f['risk_level'], f['risk_level'])}</td>
                <td>{review_badge if review_badge else '<span style="color:rgba(255,255,255,0.45); font-size:0.82em;" title="The second AI reviewer has not checked this finding yet">🔍 Not yet reviewed</span>'}</td>
                <td>{_e(display_title)}</td>
                <td style="font-size:0.8em; color:#b0bec5;">{_e(', '.join(f['affected_files']))}</td>
                <td style="font-size:0.82em; color:#b0bec5; white-space:nowrap;">{_e(formatted_date)}</td>
                <td>{status_html}</td>
                <td>{badges_html}</td>
                <td>
                    <button onclick="approve('{_e(f['finding_id'])}')">Approve</button>
                    <button onclick="decline('{_e(f['finding_id'])}')">Decline</button>
                </td>
            </tr>"""
            if extra_info:
                table_rows += f"""
            <tr>
                <td colspan="8" style="padding:12px 16px; background:rgba(15,20,50,0.95); border-top:1px solid rgba(255,255,255,0.06);">{extra_info}</td>
            </tr>"""

        return f"""<table>
            <thead><tr>
                <th>Risk</th><th>AI Review</th><th>Title</th><th>Affected Files</th>
                <th>Scan Date</th><th>Status</th><th>Finding Signal</th><th>Actions</th>
            </tr></thead>
            <tbody>{table_rows}</tbody>
        </table>"""    # Build tab content
    tab_buttons = ""
    tab_panels = ""
    first = True
    all_finding_ids = [f['finding_id'] for f, _ in sum([g["findings"] for g in agent_groups.values()], [])]

    for key, group in agent_groups.items():
        count = len(group["findings"])
        active_class = "tab-btn active" if first else "tab-btn"
        panel_style = "" if first else "display:none;"
        first = False
        tab_buttons += f'<button class="{active_class}" onclick="showTab(\'{key}\')">{group["label"]} <span class="tab-count">({count})</span></button>'
        tab_panels += f"""
        <div id="tab-{key}" class="tab-panel" style="{panel_style}">
            <div class="agent-desc">
                <strong>{group["label"]}</strong><br>
                {group["description"]}
            </div>
            {render_findings_table(group["findings"])}
        </div>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Watchdog Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            min-height: 100vh;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 40%, #24243e 70%, #0f3460 100%);
            background-attachment: fixed;
            color: #e0e0e0;
            display: flex;
            flex-direction: column;
        }}
        /* Glassmorphism base */
        .glass {{
            background: rgba(255, 255, 255, 0.07);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
        }}
        .layout {{ display: flex; flex: 1; gap: 0; }}
        .sidebar {{
            width: 240px;
            min-width: 200px;
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-right: 1px solid rgba(255,255,255,0.1);
            padding: 20px 14px;
            overflow-y: auto;
        }}
        .sidebar h2 {{
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: rgba(255,255,255,0.4);
            margin-bottom: 14px;
        }}
        .run-item {{
            padding: 10px 12px;
            margin: 5px 0;
            border-radius: 8px;
            cursor: pointer;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.05);
            font-size: 0.82em;
            transition: all 0.2s;
        }}
        .run-item:hover {{
            background: rgba(255,255,255,0.12);
            border-color: rgba(100,160,255,0.4);
            transform: translateX(2px);
        }}
        .run-item.active {{
            background: rgba(100,160,255,0.2);
            border-color: rgba(100,160,255,0.6);
        }}
        .run-item .run-date {{ font-weight: 600; color: #e0e0e0; }}
        .run-item .run-meta {{ color: rgba(255,255,255,0.45); font-size: 0.9em; margin-top: 2px; }}
        .run-all {{
            padding: 10px 12px;
            margin: 5px 0;
            border-radius: 8px;
            cursor: pointer;
            border: 1px solid rgba(100,160,255,0.3);
            background: rgba(100,160,255,0.1);
            font-size: 0.82em;
            font-weight: 600;
            color: #90c8ff;
            transition: all 0.2s;
        }}
        .run-all:hover {{ background: rgba(100,160,255,0.2); }}
        .main {{ flex: 1; padding: 24px; overflow-x: auto; min-width: 0; }}
        h1 {{
            font-size: 1.5em;
            font-weight: 700;
            color: #fff;
            margin-bottom: 16px;
            text-shadow: 0 0 20px rgba(100,160,255,0.4);
        }}
        /* Table */
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{
            border: 1px solid rgba(255,255,255,0.08);
            padding: 10px 12px;
            text-align: left;
            vertical-align: top;
            font-size: 0.88em;
            color: #e0e0e0;
        }}
        th {{
            background: rgba(255,255,255,0.06);
            color: rgba(255,255,255,0.6);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75em;
            letter-spacing: 0.8px;
            white-space: nowrap;
        }}
        tr:hover td {{ background: rgba(255,255,255,0.03); }}
        tr:nth-child(even) td {{ background: rgba(255,255,255,0.02); }}
        /* Buttons */
        button {{
            margin: 2px;
            padding: 5px 12px;
            cursor: pointer;
            border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.15);
            background: rgba(255,255,255,0.08);
            color: #e0e0e0;
            font-size: 0.82em;
            transition: all 0.2s;
        }}
        button:hover {{
            background: rgba(255,255,255,0.15);
            border-color: rgba(255,255,255,0.3);
        }}
        button.danger {{
            background: rgba(198,40,40,0.15);
            border-color: rgba(198,40,40,0.4);
            color: #ff8a80;
        }}
        button.danger:hover {{
            background: rgba(198,40,40,0.3);
        }}
        /* Tabs */
        .tabs {{
            display: flex;
            gap: 6px;
            margin-bottom: 0;
            flex-wrap: wrap;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 0;
        }}
        .tab-btn {{
            padding: 9px 18px;
            border: 1px solid rgba(255,255,255,0.1);
            border-bottom: none;
            background: rgba(255,255,255,0.04);
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            font-size: 0.85em;
            color: rgba(255,255,255,0.55);
            transition: all 0.2s;
        }}
        .tab-btn:hover {{ background: rgba(255,255,255,0.1); color: #fff; }}
        .tab-btn.active {{
            background: rgba(100,160,255,0.18);
            color: #90c8ff;
            border-color: rgba(100,160,255,0.35);
            font-weight: 600;
        }}
        .tab-count {{ font-size: 0.82em; opacity: 0.7; }}
        .tab-panel {{ padding: 20px 0; }}
        /* Details panels */
        details {{
            margin-top: 6px;
            font-size: 0.88em;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            overflow: hidden;
        }}
        details summary {{
            padding: 8px 12px;
            cursor: pointer;
            color: rgba(255,255,255,0.7);
            font-weight: 600;
            user-select: none;
        }}
        details summary:hover {{ color: #fff; }}
        details > div {{ padding: 10px 14px; }}
        /* Toolbar */
        .toolbar {{
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            padding: 10px 14px;
            background: rgba(255,255,255,0.04);
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.07);
        }}
        /* Agent description box */
        .agent-desc {{
            background: rgba(100,160,255,0.08);
            border-left: 3px solid rgba(100,160,255,0.5);
            padding: 12px 16px;
            margin-bottom: 16px;
            border-radius: 0 8px 8px 0;
            font-size: 0.88em;
            color: rgba(255,255,255,0.75);
        }}
        .agent-desc strong {{ color: #90c8ff; }}
        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: rgba(255,255,255,0.03); }}
        ::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.15); border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.25); }}
        /* Links */
        a {{ color: #90c8ff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        code {{
            background: rgba(255,255,255,0.1);
            padding: 1px 5px;
            border-radius: 4px;
            font-size: 0.9em;
            color: #b0e0ff;
        }}
        pre {{ background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px; overflow-x: auto; }}
        pre code {{ background: none; padding: 0; }}
    </style>
</head>
<body>
    {_render_pr_sticky_banner(scan_run)}
    <div class="layout">
        <div class="sidebar">
            <h2>📋 Scan Runs</h2>
            <div class="run-all" onclick="window.location='/dashboard'">All findings</div>
            <div id="runs-list">Loading...</div>
        </div>
        <div class="main">
            <h1>🔍 Migration Watchdog</h1>
            {_render_partial_data_warning(scan_run)}
            {_render_merge_checklist(scan_run, enriched_findings)}
            <div class="toolbar">
                <span style="color:rgba(255,255,255,0.6);"><strong style="color:#fff;">{len(findings_data)}</strong> total findings</span>
                {'<span style="color:#90c8ff; font-weight:bold;">📅 Run: ' + _e(run_id[:8]) + '...</span>' if run_id else ''}
                <button class="danger" onclick="declineAll()">🗑 Decline all visible</button>
            </div>
            <div class="tabs">{tab_buttons}</div>
            {tab_panels}
        </div>
    </div>
    <script>
        function showTab(key) {{
            document.querySelectorAll('.tab-panel').forEach(p => p.style.display = 'none');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('tab-' + key).style.display = '';
            event.target.classList.add('active');
        }}

        fetch('/api/runs').then(r => r.json()).then(data => {{
            const el = document.getElementById('runs-list');
            if (!data.runs || data.runs.length === 0) {{
                el.innerHTML = '<p style="color:rgba(255,255,255,0.3);font-size:0.82em;padding:8px;">No runs yet</p>';
                return;
            }}
            const currentRunId = new URLSearchParams(window.location.search).get('run_id');
            el.innerHTML = data.runs.slice(0, 5).map(run => {{
                let date = 'unknown';
                if (run.start_timestamp) {{
                    try {{
                        const d = new Date(run.start_timestamp);
                        date = d.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric', year: 'numeric' }});
                    }} catch(e) {{
                        date = run.start_timestamp.substring(0, 10);
                    }}
                }}
                const count = run.findings_count || 0;
                const status = run.status || '';
                const statusIcon = status === 'completed' ? '✅' : status === 'failed' ? '❌' : '🔄';
                const isActive = run.run_id === currentRunId;
                return `<div class="run-item ${{isActive ? 'active' : ''}}" onclick="window.location='/dashboard?run_id=${{run.run_id}}'">
                    <div class="run-date">${{statusIcon}} ${{date}}</div>
                    <div class="run-meta">${{count}} findings · ${{run.run_id.substring(0,8)}}</div>
                </div>`;
            }}).join('');
        }}).catch(() => {{
            document.getElementById('runs-list').innerHTML = '<p style="color:rgba(255,255,255,0.3);font-size:0.82em;padding:8px;">Could not load runs</p>';
        }});

        async function approve(findingId) {{
            const resp = await fetch('/api/findings/' + findingId + '/approve', {{method: 'POST'}});
            const data = await resp.json();
            alert(data.pr_url ? 'PR created: ' + data.pr_url : JSON.stringify(data));
            location.reload();
        }}
        async function decline(findingId) {{
            await fetch('/api/findings/' + findingId + '/decline', {{method: 'POST'}});
            location.reload();
        }}
        async function declineAll() {{
            const ids = {all_finding_ids};
            if (!confirm('Decline all ' + ids.length + ' visible findings?')) return;
            for (const id of ids) {{
                await fetch('/api/findings/' + id + '/decline', {{method: 'POST'}});
            }}
            location.reload();
        }}
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


# ---------------------------------------------------------------------------
# JSON API routes — Findings
# ---------------------------------------------------------------------------


@app.get("/api/findings")
async def list_findings(
    status: Optional[str] = Query(default=None),
    risk_level: Optional[str] = Query(default=None),
    sort_by: str = Query(default="risk_level"),
    sort_order: str = Query(default="desc"),
    run_id: Optional[str] = Query(default=None),
):
    """List findings with optional filtering and sorting.

    Query parameters:
    - ``status``: filter by finding status (pending, approved, declined)
    - ``risk_level``: filter by risk level (low, medium, high)
    - ``sort_by``: ``"risk_level"`` (default) or ``"scan_date"``
    - ``sort_order``: ``"desc"`` (default) or ``"asc"``
    - ``run_id``: when provided, filter findings to this run only; legacy items
      without a ``run_id`` attribute are excluded from run-scoped views
    """
    repo = _get_findings_repository()

    if run_id is not None:
        # Scan DynamoDB directly with a FilterExpression on run_id.
        # Paginate with Limit=500 per page to keep individual requests bounded.
        from boto3.dynamodb.conditions import Attr

        table = repo._table
        filter_expr = Attr("run_id").eq(run_id)
        if status is not None:
            filter_expr = filter_expr & Attr("status").eq(status)
        if risk_level is not None:
            filter_expr = filter_expr & Attr("risk_level").eq(risk_level)

        items: list[dict] = []
        scan_kwargs: dict = {
            "FilterExpression": filter_expr,
            "Limit": 500,
        }
        while True:
            resp = table.scan(**scan_kwargs)
            items.extend(resp.get("Items", []))
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key

        findings = [repo._item_to_finding(item) for item in items]
    else:
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
            "run_id": run_id,
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
