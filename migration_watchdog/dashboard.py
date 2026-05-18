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

    Returns an HTML string with badges for risk level, review status, status
    (with optional dismissed-cooldown indicator), PR link, category, short ID
    prefix, and schema-specific badges (severity for currency, gap/confidence/
    human-gate/automate for automation).

    All user-derived values are escaped with ``_e()``.
    The disputed banner fires ONLY when ``review_status == "disputed"`` — never
    on ``None``.
    """
    parts: list[str] = []

    # --- Risk badge ---
    risk_val = (
        finding.risk_level.value
        if isinstance(finding.risk_level, RiskLevel)
        else str(finding.risk_level)
    )
    risk_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    risk_labels = {"high": "High", "medium": "Medium", "low": "Low"}
    risk_icon = risk_icons.get(risk_val.lower(), "⚪")
    risk_label = risk_labels.get(risk_val.lower(), _e(risk_val))
    parts.append(
        f'<span style="font-weight:bold;">{risk_icon} {risk_label}</span>'
    )

    # --- Review status badge (blank when None) ---
    review_status = finding.review_status
    if review_status is not None:
        badge_colors = {
            "confirmed": "#2e7d32",
            "corrected": "#e65100",
            "disputed": "#c62828",
        }
        color = badge_colors.get(review_status, "#555")
        parts.append(
            f'<span style="color:{color}; font-weight:bold; border:1px solid {color}; '
            f'padding:1px 4px; border-radius:3px;">{_e(review_status)}</span>'
        )

    # --- Status badge + optional dismissed-cooldown indicator ---
    status_val = _e(finding.status)
    status_html = f'<span style="padding:1px 4px; border-radius:3px; background:#eee;">{status_val}</span>'
    if dismissal_active and finding.status == "pending":
        status_html += (
            ' <span style="color:#555; font-size:0.85em;">🔕 dismissed (cooldown)</span>'
        )
    parts.append(status_html)

    # --- PR URL link ---
    if finding.pr_url:
        parts.append(
            f'<a href="{_e(finding.pr_url)}" target="_blank" '
            f'style="font-size:0.85em;">[Fix PR ↗]</a>'
        )

    # --- Category badge ---
    schema = finding.finding_schema_version or ""
    category = finding.category or ""
    if category == "currency_drift":
        cat_label = "currency"
        cat_color = "#1565c0"
    elif category == "automation_gap":
        cat_label = "automation"
        cat_color = "#6a1b9a"
    else:
        cat_label = "analysis"
        cat_color = "#37474f"
    parts.append(
        f'<span style="color:{cat_color}; font-size:0.8em; border:1px solid {cat_color}; '
        f'padding:1px 4px; border-radius:3px;">{cat_label}</span>'
    )

    # --- Short claim/action ID prefix (first 8 hex chars) ---
    payload = finding.auditor_payload or {}
    short_id = ""
    if schema.startswith("currency/"):
        raw_id = payload.get("claim_id") or ""
        short_id = str(raw_id)[:8]
    elif schema.startswith("automation/"):
        raw_id = payload.get("action_fingerprint") or ""
        short_id = str(raw_id)[:8]
    if short_id:
        parts.append(
            f'<span style="font-family:monospace; font-size:0.8em; color:#555;">{_e(short_id)}</span>'
        )

    # --- Currency-specific badges ---
    if schema.startswith("currency/"):
        severity = payload.get("severity")
        if severity is not None:
            sev_colors = {
                "correctness": "#c62828",
                "outdated": "#e65100",
                "policy_change": "#1565c0",
            }
            sev_color = sev_colors.get(severity, "#555")
            parts.append(
                f'<span style="color:{sev_color}; font-size:0.8em; border:1px solid {sev_color}; '
                f'padding:1px 4px; border-radius:3px;">{_e(severity)}</span>'
            )

    # --- Automation-specific badges ---
    elif schema.startswith("automation/"):
        # Gap badge
        gap_type = payload.get("gap_type")
        if gap_type is not None:
            gap_colors = {
                "full_gap": "#c62828",
                "partial_gap": "#e65100",
                "no_gap": "#2e7d32",
            }
            gap_color = gap_colors.get(gap_type, "#555")
            parts.append(
                f'<span style="color:{gap_color}; font-size:0.8em; border:1px solid {gap_color}; '
                f'padding:1px 4px; border-radius:3px;">{_e(gap_type)}</span>'
            )

        # Confidence badge
        confidence = payload.get("confidence")
        if confidence is not None:
            conf_colors = {"high": "#2e7d32", "medium": "#e65100", "low": "#c62828"}
            conf_color = conf_colors.get(confidence, "#555")
            parts.append(
                f'<span style="color:{conf_color}; font-size:0.8em;">conf:{_e(confidence)}</span>'
            )

        # Human gate badge
        if payload.get("human_gate") == "required":
            parts.append(
                '<span style="color:#6a1b9a; font-size:0.85em;">🔒 human gate</span>'
            )

        # Automate badge
        automate = payload.get("automate_recommended")
        if automate is not None:
            icon = "✅" if automate == "yes" else "❌"
            parts.append(
                f'<span style="font-size:0.85em;">{icon} {_e(automate)}</span>'
            )

    return " ".join(parts)


def _render_currency_payload(payload: dict, finding: Finding) -> str:
    """Render a ``<details>`` panel for a currency/1.0 finding.

    Shows claim text, claim type, severity (with colour), actual value,
    suggested fix, verification source link, price verification path,
    alias resolved status, price metadata, scope, and review provenance
    from the Finding (not the payload).

    All user-derived values are escaped with ``_e()``.
    """
    parts: list[str] = []

    # --- Truncated payload warning ---
    # Show only when the payload was NOT successfully enriched from S3.
    # If auditor_payload_s3_key is present but the inline fields are populated
    # (i.e. S3 fetch succeeded and the full payload replaced the truncated one),
    # the key may still be present — don't show the warning in that case.
    # We detect truncation by checking whether claim_text/action_text is absent
    # (stripped during truncation) while the s3_key is present.
    s3_key = payload.get("auditor_payload_s3_key")
    is_truncated = s3_key and not payload.get("claim_text") and not payload.get("action_text")
    if is_truncated:
        parts.append(
            '<p style="color:#e65100; font-style:italic;">'
            "⚠️ Showing truncated payload — full details unavailable"
            "</p>"
        )

    # --- Claim text ---
    claim_text = payload.get("claim_text")
    if claim_text is not None:
        parts.append(f"<p><strong>Claim:</strong> &ldquo;{_e(claim_text)}&rdquo;</p>")

    # --- Claim type + optional subtype ---
    claim_type = payload.get("claim_type")
    claim_subtype = payload.get("claim_subtype")
    if claim_type is not None:
        type_str = _e(claim_type)
        if claim_subtype:
            type_str += f" / {_e(claim_subtype)}"
        parts.append(f"<p><strong>Claim type:</strong> {type_str}</p>")

    # --- Severity with colour ---
    severity = payload.get("severity")
    if severity is not None:
        sev_colors = {
            "correctness": "#c62828",
            "outdated": "#e65100",
            "policy_change": "#1565c0",
        }
        sev_color = sev_colors.get(severity, "#555")
        parts.append(
            f'<p><strong>Severity:</strong> '
            f'<span style="color:{sev_color}; font-weight:bold;">{_e(severity)}</span></p>'
        )

    # --- Actual value ---
    actual_value = payload.get("actual_value")
    if actual_value is not None:
        parts.append(f"<p><strong>Actual value:</strong> {_e(actual_value)}</p>")

    # --- Suggested fix ---
    suggested_fix = payload.get("suggested_fix")
    if suggested_fix is not None:
        parts.append(f"<p><strong>Suggested fix:</strong> {_e(suggested_fix)}</p>")

    # --- Verification source as clickable link ---
    verification_source = payload.get("verification_source")
    if verification_source is not None:
        parts.append(
            f'<p><strong>Source:</strong> '
            f'<a href="{_e(verification_source)}" target="_blank">{_e(verification_source)}</a></p>'
        )

    # --- Price verification path (when claim_type="price") ---
    if claim_type == "price":
        price_path = payload.get("price_verification_path")
        if price_path is not None:
            parts.append(
                f"<p><strong>Verification path:</strong> {_e(price_path)}</p>"
            )

    # --- Alias resolved (when claim_type="model_id" and alias_found is present) ---
    if claim_type == "model_id":
        alias_found = payload.get("alias_found")
        if alias_found is not None:
            if alias_found:
                alias_str = "✅ yes"
            else:
                alias_str = "⚠️ no (raw value used, pending review)"
            parts.append(f"<p><strong>Alias resolved:</strong> {alias_str}</p>")

    # --- Price metadata (when claim_type="price") ---
    if claim_type == "price":
        price_meta = payload.get("price_metadata")
        if price_meta and isinstance(price_meta, dict):
            currency = _e(price_meta.get("currency", ""))
            unit = _e(price_meta.get("unit", ""))
            pricing_model = _e(price_meta.get("pricing_model", ""))
            region = _e(price_meta.get("region", ""))
            meta_parts = [p for p in [currency, unit, pricing_model, region] if p]
            if meta_parts:
                parts.append(
                    f"<p><strong>Price metadata:</strong> {' · '.join(meta_parts)}</p>"
                )

    # --- Scope ---
    scope = payload.get("scope")
    if scope is not None:
        parts.append(f"<p><strong>Scope:</strong> {_e(scope)}</p>")

    # --- Review provenance (from Finding, not payload) ---
    review_status = finding.review_status
    review_notes = finding.review_notes
    primary_reasoning = finding.primary_reasoning

    if review_status == "disputed":
        dispute_reason = review_notes or primary_reasoning or "No reason provided."
        try:
            import json as _json_d
            if '\\n' in dispute_reason or '\\u' in dispute_reason:
                dispute_reason = _json_d.loads(f'"{dispute_reason}"')
        except Exception:
            pass
        parts.append(
            '<div style="background:#ffebee; border-left:4px solid #c62828; '
            'padding:10px 14px; margin:8px 0; border-radius:4px;">'
            '<strong style="color:#c62828;">⛔ Disputed by review agent</strong><br>'
            f'<span style="color:#555; font-size:0.9em; white-space:pre-wrap;">{_e(dispute_reason)}</span>'
            '</div>'
        )
        if primary_reasoning and review_notes and primary_reasoning != review_notes:
            parts.append(
                '<details><summary style="cursor:pointer; color:#555;">Primary agent reasoning</summary>'
                f'<p style="font-size:0.9em; white-space:pre-wrap;">{_e(primary_reasoning)}</p>'
                '</details>'
            )
    elif review_status == "corrected":
        parts.append(
            f'<p><strong>Review status:</strong> '
            f'<span style="color:#e65100; font-weight:bold;">corrected</span></p>'
        )
        if review_notes:
            parts.append(
                f"<p><strong>Review notes:</strong> {_e(review_notes)}</p>"
            )
    elif review_status == "confirmed":
        parts.append(
            f'<p><strong>Review status:</strong> '
            f'<span style="color:#2e7d32; font-weight:bold;">confirmed</span></p>'
        )
        if review_notes:
            parts.append(
                f"<p><strong>Review notes:</strong> {_e(review_notes)}</p>"
            )

    inner = "\n".join(parts)
    return (
        "<details>"
        "<summary>Currency Finding Details</summary>"
        f'<div style="padding:8px 12px;">{inner}</div>'
        "</details>"
    )


def _render_automation_payload(payload: dict, finding: Finding) -> str:
    """Render a ``<details>`` panel for an automation/1.0 finding.

    Shows action text, action type, gap type + confidence + reason,
    gap detail narrative, CLI equivalent as a code block, reference URL,
    placeholder list, judgment inputs, automate recommendation, human gate
    warning, rationale, scope, and a static-analysis caveat note.

    All user-derived values are escaped with ``_e()``.
    """
    parts: list[str] = []

    # --- Action text ---
    action_text = payload.get("action_text")
    if action_text is not None:
        parts.append(
            f"<p><strong>Manual action:</strong> &ldquo;{_e(action_text)}&rdquo;</p>"
        )

    # --- Action type ---
    action_type = payload.get("action_type")
    if action_type is not None:
        parts.append(f"<p><strong>Action type:</strong> {_e(action_type)}</p>")

    # --- Gap type + confidence + reason (when reason is not None) ---
    gap_type = payload.get("gap_type")
    confidence = payload.get("confidence")
    reason = payload.get("reason")
    if gap_type is not None:
        gap_str = _e(gap_type)
        if confidence is not None:
            gap_str += f" ({_e(confidence)} confidence)"
        parts.append(f"<p><strong>Gap:</strong> {gap_str}</p>")
    if reason is not None:
        parts.append(f"<p><strong>Gap reason:</strong> {_e(reason)}</p>")

    # --- Gap detail / partial_gap_narrative ---
    partial_gap_narrative = payload.get("partial_gap_narrative")
    if partial_gap_narrative is not None:
        parts.append(
            f"<p><strong>Gap detail:</strong> {_e(partial_gap_narrative)}</p>"
        )

    # --- CLI equivalent as <pre><code> block ---
    cli_equivalent = payload.get("cli_equivalent")
    if cli_equivalent is not None:
        parts.append(
            f"<p><strong>CLI equivalent:</strong></p>"
            f"<pre><code>{_e(cli_equivalent)}</code></pre>"
        )
    else:
        cli_note = payload.get("cli_note")
        if cli_note is not None:
            parts.append(
                f"<p><strong>CLI equivalent:</strong> <em>{_e(cli_note)}</em></p>"
            )

    # --- Reference URL as clickable link ---
    reference_url = payload.get("reference_url")
    if reference_url is not None:
        parts.append(
            f'<p><strong>Reference:</strong> '
            f'<a href="{_e(reference_url)}" target="_blank">{_e(reference_url)}</a></p>'
        )

    # --- Placeholder list (comma-separated) ---
    placeholder_list = payload.get("placeholder_list")
    if placeholder_list and isinstance(placeholder_list, list):
        escaped_placeholders = ", ".join(_e(str(p)) for p in placeholder_list)
        parts.append(
            f"<p><strong>Placeholders:</strong> {escaped_placeholders}</p>"
        )

    # --- Judgment inputs section (always shown) ---
    judgment_required = payload.get("judgment_required")
    repeated_operation = payload.get("repeated_operation")
    values_precomputed = payload.get("values_precomputed")
    safety_impact = payload.get("safety_impact")

    judgment_rows = ""
    if judgment_required is not None:
        judgment_rows += f"<tr><td>judgment_required</td><td>{_e(str(judgment_required))}</td></tr>"
    if repeated_operation is not None:
        judgment_rows += f"<tr><td>repeated_operation</td><td>{_e(str(repeated_operation))}</td></tr>"
    if values_precomputed is not None:
        judgment_rows += f"<tr><td>values_precomputed</td><td>{_e(str(values_precomputed))}</td></tr>"
    if safety_impact is not None:
        judgment_rows += f"<tr><td>safety_impact</td><td>{_e(str(safety_impact))}</td></tr>"

    parts.append(
        "<p><strong>Judgment inputs:</strong></p>"
        '<table style="border-collapse:collapse; font-size:0.9em;">'
        f"{judgment_rows}"
        "</table>"
    )

    # --- Automate recommendation ---
    automate_recommended = payload.get("automate_recommended")
    if automate_recommended is not None:
        icon = "✅ Yes" if automate_recommended == "yes" else "❌ No"
        parts.append(
            f"<p><strong>Automate:</strong> {icon}</p>"
        )

    # --- Human gate warning ---
    if payload.get("human_gate") == "required":
        parts.append(
            '<p style="background:#f3e5f5; border-left:4px solid #6a1b9a; '
            'padding:8px 12px; color:#6a1b9a; font-weight:bold;">'
            "⚠️ Requires human approval before execution"
            "</p>"
        )

    # --- Rationale ---
    rationale = payload.get("rationale")
    if rationale is not None:
        parts.append(f"<p><strong>Rationale:</strong> {_e(rationale)}</p>")

    # --- Scope ---
    scope = payload.get("scope")
    if scope is not None:
        parts.append(f"<p><strong>Scope:</strong> {_e(scope)}</p>")

    # --- Static analysis caveat note ---
    parts.append(
        '<p style="font-size:0.85em; color:#555; font-style:italic;">'
        "Note: Gap detection is a static substring match — confirm in PR diff "
        "before acting on this recommendation."
        "</p>"
    )

    inner = "\n".join(parts)
    return (
        "<details>"
        "<summary>Automation Finding Details</summary>"
        f'<div style="padding:8px 12px;">{inner}</div>'
        "</details>"
    )


def _format_description(description: str) -> str:
    """Format a finding description as readable HTML.

    Handles:
    - JSON escape sequences (\\n, \\u2014, etc.)
    - **Bold** markdown
    - `code` backtick spans
    - Numbered lists (1. 2. 3.)
    - Bullet lists (- item)
    - Section headers (Advantages:, Disadvantages:, etc.)
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

    # Split into lines for processing
    lines = description.split('\n')
    html_parts: list[str] = []
    in_list = False
    list_type = None  # 'ul' or 'ol'

    def close_list():
        nonlocal in_list, list_type
        if in_list:
            html_parts.append(f'</{list_type}>')
            in_list = False
            list_type = None

    def format_inline(text: str) -> str:
        """Format inline markdown: **bold**, `code`."""
        text = _e(text)
        text = _re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        text = _re.sub(r'`([^`]+)`', r'<code style="background:#f0f0f0;padding:1px 3px;border-radius:2px;">\1</code>', text)
        return text

    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_list()
            continue

        # Section headers like "Advantages:", "Disadvantages:", "Proposed structure:"
        header_match = _re.match(r'^(Advantages|Disadvantages|Proposed structure|Summary|Background|Impact|Solution|Recommendation)s?:\s*(.*)$', stripped, _re.IGNORECASE)
        if header_match:
            close_list()
            label = header_match.group(1)
            rest = header_match.group(2)
            html_parts.append(f'<p style="margin:10px 0 4px 0;"><strong>{_e(label)}:</strong> {format_inline(rest)}</p>' if rest else f'<p style="margin:10px 0 4px 0;"><strong>{_e(label)}:</strong></p>')
            continue

        # Numbered list items: "1. text" or "(1) text"
        num_match = _re.match(r'^[\(\[]?(\d+)[\)\]\.]\s+(.+)$', stripped)
        if num_match:
            if not in_list or list_type != 'ol':
                close_list()
                html_parts.append('<ol style="margin:4px 0; padding-left:20px;">')
                in_list = True
                list_type = 'ol'
            html_parts.append(f'<li style="margin:3px 0;">{format_inline(num_match.group(2))}</li>')
            continue

        # Bullet list items: "- text" or "* text"
        bullet_match = _re.match(r'^[-*•]\s+(.+)$', stripped)
        if bullet_match:
            if not in_list or list_type != 'ul':
                close_list()
                html_parts.append('<ul style="margin:4px 0; padding-left:20px;">')
                in_list = True
                list_type = 'ul'
            html_parts.append(f'<li style="margin:3px 0;">{format_inline(bullet_match.group(1))}</li>')
            continue

        # Regular paragraph text
        close_list()
        html_parts.append(f'<p style="margin:6px 0;">{format_inline(stripped)}</p>')

    close_list()
    return '\n'.join(html_parts)


def _render_general_finding_detail(finding: Finding) -> str:
    """Render a ``<details>`` panel for non-currency/automation findings."""
    parts: list[str] = []

    # --- Description (the main content) ---
    description = finding.description
    if description:
        parts.append(f'<div style="margin-bottom:8px;">{_format_description(description)}</div>')

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
            '<div style="background:#ffebee; border-left:4px solid #c62828; '
            'padding:10px 14px; margin:8px 0; border-radius:4px;">'
            '<strong style="color:#c62828;">⛔ Disputed by review agent</strong><br>'
            f'<span style="color:#555; font-size:0.9em; white-space:pre-wrap;">{_e(dispute_reason)}</span>'
            '</div>'
        )
        if primary_reasoning and review_notes and primary_reasoning != review_notes:
            parts.append(
                '<details><summary style="cursor:pointer; color:#555;">Primary agent reasoning</summary>'
                f'<p style="font-size:0.9em; white-space:pre-wrap;">{_e(primary_reasoning)}</p>'
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
            '<div style="background:#fff8e1; border-left:3px solid #f9a825; padding:6px 10px; margin:6px 0; border-radius:3px;">'
            f'<strong>⚠️ Impact if unresolved:</strong> {impact}'
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
                f'<div style="margin:6px 0; padding:8px 10px; background:#f5f5f5; border-radius:4px; border-left:3px solid #1565c0;">'
                f'<div style="margin-bottom:4px;">{change_html}</div>'
                f'<div style="font-size:0.8em; color:#666; font-family:monospace;">{_e(file_path)}</div>'
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
        "<summary style='cursor:pointer; font-weight:bold;'>Finding Details</summary>"
        f'<div style="padding:8px 12px; border:1px solid #e0e0e0; border-radius:4px; margin-top:4px;">{inner}</div>'
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
            expires = datetime.fromisoformat(finding.dismissal.cooldown_expires)
            result["dismissal_active"] = datetime.utcnow() < expires
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

    risk_badge = {
        "low": '<span style="color:green;">🟢 Low</span>',
        "medium": '<span style="color:orange;">🟡 Medium</span>',
        "high": '<span style="color:red;">🔴 High</span>',
    }

    rows = ""
    for f, finding_obj in zip(findings_data, enriched_findings):
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

        # Render auditor payload detail panel (currency or automation).
        payload_html = _render_auditor_payload(
            f.get("auditor_payload"),
            f.get("finding_schema_version"),
            finding_obj,
        )
        if payload_html:
            extra_info += payload_html

        # Render inline badges for the row.
        dismissal_active = bool(f.get("dismissal_active", False))
        badges_html = _render_list_badges(finding_obj, dismissal_active=dismissal_active)

        rows += f"""
        <tr>
            <td>{risk_badge.get(f['risk_level'], f['risk_level'])}</td>
            <td>{review_badge}</td>
            <td>{_e(f['title'])}</td>
            <td>{_e(', '.join(f['affected_files']))}</td>
            <td>{_e(f['scan_date'])}</td>
            <td>{_e(f['status'])}</td>
            <td>{badges_html}</td>
            <td>
                <button onclick="approve('{_e(f['finding_id'])}')">Approve</button>
                <button onclick="decline('{_e(f['finding_id'])}')">Decline</button>
                <p style="font-size:0.8em; color:#555; margin:4px 0 0 0;">
                    Approving queues a fix PR. Merging that PR is a separate step on GitHub.
                </p>
            </td>
        </tr>
        """
        if extra_info:
            rows += f"""
        <tr>
            <td colspan="8" style="padding:4px 12px; background:#fafafa;">{extra_info}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
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
    {_render_pr_sticky_banner(scan_run)}
    <h1>Migration Plugin Watchdog Dashboard</h1>
    {_render_partial_data_warning(scan_run)}
    {_render_merge_checklist(scan_run, enriched_findings)}
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
                <th>Badges</th>
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
