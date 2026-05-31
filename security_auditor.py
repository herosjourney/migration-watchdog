"""Security Auditor — LLM-based security review of generated code artifacts.

Uses Claude Opus 4.7 via Strands to reason about security issues in the
plugin's reference files. Unlike static pattern matching, the LLM can
understand context, recognize novel anti-patterns, and apply the AWS
Startup Security Baseline checklist without hard-coded regexes.

Security checks performed (grounded in AWS Startup Security Baseline ACCT.01-ACCT.13):
- Open administrative ports (22/SSH, 3389/RDP, 5900/VNC) from 0.0.0.0/0
- Secrets stored in shell variables (ps aux, shell history, core dump exposure)
- Database passwords in Terraform variables (plaintext in state)
- Missing deletion_protection on Aurora/RDS clusters
- Secrets Manager rotation without compliance gate
- max_tokens set too high in Bedrock adapters (TPM quota burndown trap)
- Missing security baseline resources (GuardDuty, CloudTrail, S3 PAB, IMDSv2)
- IMDSv2 not enforced on EC2 launch templates
- S3 buckets without public access block
- Any other security anti-patterns the LLM identifies
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from migration_watchdog.models import Finding, RiskLevel
from migration_watchdog.source_fetcher import AwsDocsSearcher

logger = logging.getLogger(__name__)

# Module-level docs searcher for security verification
_security_docs_searcher = AwsDocsSearcher()


@tool
def search_aws_security_docs(query: str) -> str:
    """Search AWS security documentation to verify security best practices.

    Use this to verify current AWS security recommendations before flagging
    an issue, and to find the authoritative source URL for findings.

    Args:
        query: e.g. "AWS Systems Manager Session Manager SSH alternative",
               "Bedrock max_tokens TPM quota reservation",
               "AWS Startup Security Baseline GuardDuty"

    Returns:
        Relevant text from AWS security documentation.
    """
    return _security_docs_searcher.search_and_fetch_sync(query)


# ---------------------------------------------------------------------------
# System prompt for the security review agent
# ---------------------------------------------------------------------------

_SECURITY_SYSTEM_PROMPT = """\
You are a security auditor reviewing migration plugin reference files.

These files instruct an AI agent to generate Terraform, shell scripts, and Python code
for startups migrating from GCP to AWS. Your job is to identify security anti-patterns
in the INSTRUCTIONS — issues that would cause the generated output to be insecure.

You are grounded in the AWS Startup Security Baseline (ACCT.01-ACCT.13) and OWASP.
Use the search_aws_security_docs tool to verify current AWS security best practices
before flagging issues, and to find source URLs for your findings.

SECURITY CHECKLIST — check each of these for every file:

1. OPEN ADMIN PORTS: Does the file instruct generating security group rules that allow
   ports 22 (SSH), 3389 (RDP), or 5900 (VNC) from 0.0.0.0/0 or ::/0?
   If yes: HIGH severity. AWS best practice is AWS Systems Manager Session Manager.

2. SECRETS IN SHELL VARIABLES: Does the file instruct storing secret values in shell
   variables (e.g., SECRET_VALUE=$(gcloud secrets ...))? This exposes secrets in
   process listings (ps aux), shell history, and core dumps.
   If yes: HIGH severity. Fix: write to chmod 600 temp file with trap cleanup.

3. PLAINTEXT DB PASSWORDS: Does the file instruct using master_password = var.x in
   Terraform? This puts the password in terraform.tfvars and state in plaintext.
   If yes: HIGH severity. Fix: use random_password + Secrets Manager.

4. MISSING DELETION PROTECTION: Does the file instruct generating Aurora/RDS clusters
   without deletion_protection = true?
   If yes: MEDIUM severity. Prevents accidental terraform destroy data loss.

5. ROTATION WITHOUT COMPLIANCE GATE: Does the file instruct emitting Secrets Manager
   rotation blocks unconditionally (not gated on compliance flags like soc2/pci/hipaa)?
   Blanket rotation blocks break non-compliance stacks.
   If yes: LOW severity.

6. MAX_TOKENS TOO HIGH: Does the file instruct generating Bedrock provider adapters
   with max_tokens=4096 or higher as a default? Bedrock deducts max_tokens from TPM
   quota at request start — high defaults reduce concurrency 5-8x unnecessarily.
   If yes: MEDIUM severity. Default should be ~1024 with a tuning comment.

7. MISSING SECURITY BASELINE: Does the file describe generating Terraform infrastructure
   WITHOUT including baseline security resources? Check for absence of:
   - GuardDuty (threat detection)
   - CloudTrail (audit logging)
   - S3 account-level Public Access Block
   - IAM password policy
   If these are missing from a file that generates account-level infrastructure:
   MEDIUM severity.

8. IMDSV2 NOT ENFORCED: Does the file instruct generating EC2 launch templates or
   Auto Scaling groups without http_tokens = "required" (IMDSv2)?
   If yes: MEDIUM severity. IMDSv2 prevents SSRF attacks against instance metadata.

9. ANY OTHER SECURITY ISSUES: Use your security expertise to identify any other
   anti-patterns not covered above.

OUTPUT FORMAT:
For each issue found, output a JSON object in a JSON array:
[
  {
    "issue_type": "open_admin_port|secret_in_shell_variable|plaintext_db_password|missing_deletion_protection|rotation_without_compliance_gate|missing_max_tokens_warning|missing_security_baseline|imdsv2_not_enforced|other",
    "severity": "critical|high|medium|low",
    "title": "Short descriptive title",
    "description": "What the issue is and why it matters",
    "suggested_fix": "Specific actionable fix",
    "line_context": "Relevant quote from the file (max 200 chars)",
    "source_url": "AWS documentation URL supporting this finding"
  }
]

If no security issues are found, output an empty array: []
Do NOT output any explanation outside the JSON array.
"""


def _build_security_prompt(file_path: str, content: str) -> str:
    """Build the user message for the security review LLM call.

    For files longer than 8000 chars, sends the content in chunks and
    notes the truncation so the LLM knows it may not see the full file.
    """
    # Increase limit to 12000 chars to reduce truncation of long generate-artifacts files
    _CONTENT_LIMIT = 12000
    truncated = len(content) > _CONTENT_LIMIT
    content_to_send = content[:_CONTENT_LIMIT]
    truncation_note = (
        f"\n\n**NOTE: File is {len(content)} chars — only first {_CONTENT_LIMIT} shown. "
        "Security rules near the end of the file may not be visible.**"
        if truncated else ""
    )
    return (
        f"Review this migration plugin reference file for security issues.\n\n"
        f"File path: {file_path}\n"
        f"File length: {len(content)} chars{' (TRUNCATED)' if truncated else ''}\n\n"
        f"```markdown\n{content_to_send}\n```"
        f"{truncation_note}\n\n"
        "Output a JSON array of security issues found. Output [] if none."
    )


# ---------------------------------------------------------------------------
# LLM-based security scanner
# ---------------------------------------------------------------------------

class LLMSecurityScanner:
    """Uses Claude Opus 4.7 to reason about security issues in reference files."""

    # Files that describe generated code and should be security-checked
    SECURITY_RELEVANT_PATTERNS = [
        "generate-artifacts",
        "generate-infra",
        "generate-artifacts-scripts",
        "generate-artifacts-infra",
        "networking",
        "security",
        "design-infra",
        "baseline",
        # Design reference files — contain actual migration guidance startups follow
        "design-ref",
        "ai-migration-guardrails",
        "design-ref-agentic",
        "design-ref-harness",
        "design-ref-networking",
        "design-ref-compute",
        "design-ref-database",
    ]

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-opus-4-7",
        region_name: str = "us-east-1",
    ) -> None:
        self._model_id = model_id
        self._region_name = region_name

    def scan_file(self, file_path: str, content: str) -> list[dict]:
        """Scan a single file for security issues using Claude.

        Returns a list of issue dicts with keys:
        issue_type, severity, title, description, suggested_fix,
        line_context, source_url

        Returns [] on a clean scan (no issues found).
        Raises on failure — callers are responsible for catching exceptions
        and recording partial failures.
        """
        model = BedrockModel(
            model_id=self._model_id,
            region_name=self._region_name,
            max_tokens=4000,
        )
        agent = Agent(
            model=model,
            system_prompt=_SECURITY_SYSTEM_PROMPT,
            tools=[search_aws_security_docs],
        )
        prompt = _build_security_prompt(file_path, content)
        response_text = str(agent(prompt))
        issues = self._parse_response(response_text, file_path)
        if not issues:
            logger.debug(
                "LLMSecurityScanner: no issues found in %s (clean scan)",
                file_path,
            )
        return issues

    def _parse_response(self, response_text: str, file_path: str) -> list[dict]:
        """Parse the LLM response into a list of issue dicts."""
        text = response_text.strip()

        # Strip markdown code fences
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()

        # Find the JSON array
        bracket_idx = text.find("[")
        if bracket_idx != -1:
            text = text[bracket_idx:]
        end_idx = text.rfind("]")
        if end_idx != -1:
            text = text[:end_idx + 1]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.error(
                "LLMSecurityScanner: failed to parse JSON response for %s — "
                "returning [] due to parse failure (not a clean scan). "
                "Response preview: %r",
                file_path,
                response_text[:200],
            )
            return []

        if not isinstance(parsed, list):
            return []

        # Validate and normalise each issue
        valid_issues = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            if not item.get("issue_type") or not item.get("title"):
                continue
            # Ensure required fields have defaults
            item.setdefault("severity", "medium")
            item.setdefault("description", item.get("title", ""))
            item.setdefault("suggested_fix", "")
            item.setdefault("line_context", "")
            item.setdefault("source_url", "")
            valid_issues.append(item)

        return valid_issues


# ---------------------------------------------------------------------------
# Risk level mapping
# ---------------------------------------------------------------------------

_SEVERITY_TO_RISK = {
    "critical": RiskLevel.HIGH,
    "high": RiskLevel.HIGH,
    "medium": RiskLevel.MEDIUM,
    "low": RiskLevel.LOW,
    "informational": RiskLevel.LOW,
}


def _issue_to_finding(issue: dict, file_path: str, run_id: str) -> Finding:
    """Convert an LLM-identified security issue dict to a Finding."""
    severity = issue.get("severity", "medium").lower()
    risk = _SEVERITY_TO_RISK.get(severity, RiskLevel.MEDIUM)
    issue_type = issue.get("issue_type", "other")
    title = issue.get("title", f"Security issue in {file_path.split('/')[-1]}")
    description = issue.get("description", "")
    suggested_fix = issue.get("suggested_fix", "")
    line_context = issue.get("line_context", "")
    source_url = issue.get("source_url", "")

    # Compute a deterministic finding ID
    raw = f"{issue_type}|{file_path}|{title}"
    finding_id = hashlib.sha256(raw.encode()).hexdigest()[:32]

    source_urls = []
    if source_url:
        source_urls.append(source_url)
    # Always include the AWS Startup Security Baseline as a reference
    source_urls.append(
        "https://docs.aws.amazon.com/prescriptive-guidance/latest/aws-startup-security-baseline/welcome.html"
    )

    full_description = description
    if line_context:
        full_description += f"\n\n**Context from file:** `{line_context}`"
    if suggested_fix:
        full_description += f"\n\n**Suggested fix:** {suggested_fix}"

    return Finding(
        finding_id=finding_id,
        run_id=run_id,
        risk_level=risk,
        category="security",
        title=f"{title} in {file_path.split('/')[-1]}",
        description=full_description,
        affected_files=[file_path],
        proposed_changes={},
        source_urls=source_urls,
        scan_timestamp=datetime.now(timezone.utc).isoformat(),
        status="pending",
        auditor_payload={
            "issue_type": issue_type,
            "severity": severity,
            "suggested_fix": suggested_fix,
            "line_context": line_context,
        },
        finding_schema_version="security/1.0",
    )


def security_dedupe_key(finding: Finding) -> tuple:
    """Deduplication key for security findings."""
    payload = finding.auditor_payload or {}
    return (
        finding.category,
        frozenset(finding.affected_files),
        payload.get("issue_type", finding.finding_id),
    )


async def run_security_audit(
    repo_content: Any,
    run_id: str,
    file_filter: list[str] | None = None,
    partial_failures: list[str] | None = None,
) -> list[Finding]:
    """Run the LLM-based security audit over reference files.

    Uses Claude Opus 4.7 to reason about security issues in files that
    describe generated scripts, Terraform, and Python code.

    Args:
        repo_content: Repository content with a ``files`` dict.
        run_id: Unique identifier for this scan run.
        file_filter: Optional list of file paths to restrict scanning to.
        partial_failures: Optional mutable list; on per-file scan failure a
            JSON-encoded entry is appended and scanning continues.
    """
    scanner = LLMSecurityScanner()
    findings: list[Finding] = []
    seen_keys: set = set()
    scanned_count = 0
    skipped_count = 0

    for file_path, content in repo_content.files.items():
        # Only scan security-relevant files
        is_relevant = any(
            pattern in file_path
            for pattern in LLMSecurityScanner.SECURITY_RELEVANT_PATTERNS
        )
        if not is_relevant:
            skipped_count += 1
            continue

        # Apply file filter if provided
        if file_filter and not any(f in file_path for f in file_filter):
            skipped_count += 1
            continue

        scanned_count += 1
        logger.info("Security audit: scanning %s", file_path)
        try:
            issues = scanner.scan_file(file_path, content)
        except Exception as exc:
            logger.error(
                "LLMSecurityScanner: scan FAILED for %s — recording partial failure and continuing.",
                file_path,
                exc_info=True,
            )
            if partial_failures is not None:
                partial_failures.append(
                    json.dumps({
                        "type": "security_scan_failure",
                        "file": file_path,
                        "error": str(exc),
                    })
                )
            continue

        for issue in issues:
            finding = _issue_to_finding(issue, file_path, run_id)
            key = security_dedupe_key(finding)
            if key not in seen_keys:
                seen_keys.add(key)
                findings.append(finding)
                logger.info(
                    "Security audit: found %s in %s",
                    issue.get("issue_type", "unknown"),
                    file_path,
                )

    logger.info(
        "Security audit completed: %d findings from %d files scanned "
        "(%d files skipped as not security-relevant, %d total loaded)",
        len(findings),
        scanned_count,
        skipped_count,
        len(repo_content.files),
    )
    return findings
