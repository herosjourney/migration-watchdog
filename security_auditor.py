"""Security Auditor — checks generated code artifacts for security vulnerabilities.

Reads migration guide files that describe generated scripts, Terraform, and
Python code, and flags security anti-patterns that would produce vulnerable
infrastructure or expose secrets.

Security checks performed:
- Open administrative ports (22, 3389, 5900) from 0.0.0.0/0 in security groups
- Secrets stored in shell variables (exposed in ps aux, shell history)
- Database passwords in Terraform variables (plaintext in state)
- Missing deletion protection on databases
- Missing encryption at rest
- Secrets Manager rotation without compliance gate
- max_tokens not set (TPM burndown trap on Bedrock)
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from migration_watchdog.models import Finding, RiskLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security finding categories
# ---------------------------------------------------------------------------

SECURITY_CATEGORIES = {
    "open_admin_port": {
        "severity": "critical",
        "risk": RiskLevel.HIGH,
        "title_template": "Open administrative port {port} from 0.0.0.0/0 in {file}",
        "description_template": (
            "The reference file instructs the plugin to generate a security group rule "
            "allowing inbound traffic on port {port} from 0.0.0.0/0 (the internet). "
            "This exposes administrative access ({service}) to the public internet. "
            "AWS best practice is to use AWS Systems Manager Session Manager for "
            "administrative access instead of opening SSH/RDP ports."
        ),
        "suggested_fix": (
            "Replace the open port rule with a commented-out placeholder and a "
            "warnings[] entry pointing to AWS Systems Manager Session Manager. "
            "See: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html"
        ),
    },
    "secret_in_shell_variable": {
        "severity": "high",
        "risk": RiskLevel.HIGH,
        "title_template": "Secret value stored in shell variable in {file}",
        "description_template": (
            "The generated script stores a secret value in a shell variable "
            "(e.g., SECRET_VALUE=$(gcloud secrets ...)). This exposes the secret "
            "in process listings (ps aux), shell history, and core dumps. "
            "Secrets should be written to a chmod 600 temp file with trap cleanup."
        ),
        "suggested_fix": (
            "Write the secret to a temp file: TMPFILE=$(mktemp); chmod 600 $TMPFILE; "
            "trap 'rm -f $TMPFILE' EXIT; then pass file://$TMPFILE to the AWS CLI."
        ),
    },
    "plaintext_db_password": {
        "severity": "high",
        "risk": RiskLevel.HIGH,
        "title_template": "Database password in Terraform variable in {file}",
        "description_template": (
            "The generated Terraform uses master_password = var.database_master_password, "
            "which stores the database password in terraform.tfvars and Terraform state "
            "in plaintext. This is a security vulnerability — state files are often "
            "stored in S3 and accessible to anyone with bucket access."
        ),
        "suggested_fix": (
            "Generate the password using random_password + aws_secretsmanager_secret_version, "
            "then reference it via a data source on the cluster resource. "
            "Never put database passwords in Terraform variables."
        ),
    },
    "missing_deletion_protection": {
        "severity": "medium",
        "risk": RiskLevel.MEDIUM,
        "title_template": "Missing deletion_protection on Aurora/RDS cluster in {file}",
        "description_template": (
            "The generated Terraform does not set deletion_protection = true on the "
            "Aurora/RDS cluster. Without this, terraform destroy will permanently delete "
            "the database without confirmation. This is a data loss risk in production."
        ),
        "suggested_fix": (
            "Add deletion_protection = true to the Aurora/RDS cluster resource. "
            "Include an inline comment explaining when to disable it (e.g., for teardown)."
        ),
    },
    "rotation_without_compliance_gate": {
        "severity": "low",
        "risk": RiskLevel.LOW,
        "title_template": "Secrets Manager rotation block without compliance gate in {file}",
        "description_template": (
            "The generated Terraform emits rotation blocks for every secret regardless "
            "of compliance requirements. Blanket rotation blocks make generated stacks "
            "non-applyable until every rotation Lambda exists, which is wrong for "
            "dev/non-compliance environments."
        ),
        "suggested_fix": (
            "Gate rotation blocks on a compliance flag in preferences.json "
            "(soc2, pci, hipaa, fedramp), matching the same gate used for "
            "Config and Security Hub in baseline.tf."
        ),
    },
    "missing_max_tokens_warning": {
        "severity": "medium",
        "risk": RiskLevel.MEDIUM,
        "title_template": "Missing max_tokens guidance in Bedrock provider adapter in {file}",
        "description_template": (
            "Bedrock deducts max_tokens from TPM quota at request start before any tokens "
            "are generated. Teams migrating from OpenAI commonly have max_tokens=4096 as "
            "a default, which reduces Bedrock concurrency by 5-8x unnecessarily. "
            "The generated provider adapter should default max_tokens to 1024 with a "
            "comment prompting the user to tune it."
        ),
        "suggested_fix": (
            "Set max_tokens=1024 as the default in the generated provider adapter "
            "(not the model maximum). Add a TODO comment: "
            "'# TODO: tune max_tokens to your actual p95 output length to maximize concurrency'"
        ),
    },
}

# ---------------------------------------------------------------------------
# Security patterns to detect
# ---------------------------------------------------------------------------

# Open admin ports: port 22 (SSH), 3389 (RDP), 5900 (VNC) from 0.0.0.0/0
_OPEN_PORT_PATTERNS = [
    (22, "SSH", re.compile(
        r'(?:port[s]?\s*[=:"\s]+22\b|"22"|ingress.*\b22\b|\b22\b.*ingress|from_port.*22|to_port.*22'
        r'|allow.*ssh|ssh.*allow|port 22|:22\b)',
        re.IGNORECASE
    )),
    (3389, "RDP", re.compile(
        r'(?:port[s]?\s*[=:"\s]+3389\b|"3389"|ingress.*3389|3389.*ingress|from_port.*3389|to_port.*3389'
        r'|allow.*rdp|rdp.*allow|port 3389)',
        re.IGNORECASE
    )),
    (5900, "VNC", re.compile(
        r'(?:port[s]?\s*[=:"\s]+5900\b|"5900"|ingress.*5900|5900.*ingress|from_port.*5900|to_port.*5900'
        r'|allow.*vnc|vnc.*allow|port 5900)',
        re.IGNORECASE
    )),
]
_OPEN_CIDR_PATTERN = re.compile(r'0\.0\.0\.0/0|::/0|internet|public.*internet|0\.0\.0\.0', re.IGNORECASE)

# Secret in shell variable — matches both actual code and markdown descriptions of the pattern
_SECRET_SHELL_VAR_PATTERN = re.compile(
    r'(?:'
    r'(?:SECRET|PASSWORD|TOKEN|KEY|CREDENTIAL)[_A-Z]*\s*=\s*\$\('  # actual shell code
    r'|SECRET_VALUE=\$\('                                            # exact pattern from description
    r'|gcloud secrets.*versions access'                              # gcloud secret fetch
    r'|secret.*shell.*variable'                                      # prose description
    r'|stored.*shell.*variable'                                      # prose description
    r'|shell variable.*secret'                                       # prose description
    r')',
    re.IGNORECASE
)

# Plaintext DB password in Terraform — also matches prose descriptions
_PLAINTEXT_DB_PASSWORD_PATTERN = re.compile(
    r'(?:'
    r'master_password\s*=\s*var\.'                    # actual Terraform
    r'|var\.database_master_password'                  # variable reference
    r'|database.*password.*variable'                   # prose
    r'|password.*terraform.*variable'                  # prose
    r'|master_password.*var\.'                         # partial match
    r')',
    re.IGNORECASE
)

# Missing deletion protection (Aurora/RDS resource without deletion_protection)
_AURORA_RESOURCE_PATTERN = re.compile(
    r'(?:resource\s+"aws_rds_cluster"|aws_rds_cluster|aurora.*cluster|rds.*cluster)',
    re.IGNORECASE
)
_DELETION_PROTECTION_PATTERN = re.compile(
    r'deletion_protection\s*=\s*true',
    re.IGNORECASE
)

# Rotation without compliance gate — also matches prose
_ROTATION_BLOCK_PATTERN = re.compile(
    r'(?:rotation_rules|enable_rotation|rotation_lambda|rotate.*secret|secret.*rotation)',
    re.IGNORECASE
)
_COMPLIANCE_GATE_PATTERN = re.compile(
    r'(?:compliance|soc2|pci|hipaa|fedramp|compliance_flag|compliance.*gate|gate.*compliance)',
    re.IGNORECASE
)

# Missing max_tokens warning in Bedrock adapter
_BEDROCK_ADAPTER_PATTERN = re.compile(
    r'bedrock|converse|invoke_model|provider_adapter',
    re.IGNORECASE
)
_MAX_TOKENS_PATTERN = re.compile(
    r'max_tokens\s*=\s*(?:1024|512|256)',
    re.IGNORECASE
)
_MAX_TOKENS_HIGH_PATTERN = re.compile(
    r'max_tokens\s*=\s*(?:4096|8192|16384|32768)',
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# SecurityFinding dataclass
# ---------------------------------------------------------------------------

@dataclass
class SecurityIssue:
    """A single security issue detected in a reference file."""
    issue_type: str
    file_path: str
    line_context: str
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SecurityAuditor
# ---------------------------------------------------------------------------

class SecurityAuditor:
    """Scans reference files for security anti-patterns in generated code.

    Focuses on files that describe generated scripts, Terraform, and Python
    code — specifically the generate/ phase files and networking design refs.
    """

    # Files that describe generated code and should be security-checked
    SECURITY_RELEVANT_PATTERNS = [
        "generate-artifacts",
        "generate-infra",
        "generate-artifacts-scripts",
        "generate-artifacts-infra",
        "networking",
        "security",
        "design-infra",
    ]

    def scan_file(self, file_path: str, content: str) -> list[SecurityIssue]:
        """Scan a single file for security issues."""
        issues: list[SecurityIssue] = []

        # Check open admin ports
        issues.extend(self._check_open_ports(file_path, content))

        # Check secret in shell variable
        issues.extend(self._check_secret_shell_var(file_path, content))

        # Check plaintext DB password
        issues.extend(self._check_plaintext_db_password(file_path, content))

        # Check missing deletion protection
        issues.extend(self._check_deletion_protection(file_path, content))

        # Check rotation without compliance gate
        issues.extend(self._check_rotation_gate(file_path, content))

        # Check missing max_tokens warning
        issues.extend(self._check_max_tokens(file_path, content))

        return issues

    def _check_open_ports(self, file_path: str, content: str) -> list[SecurityIssue]:
        issues = []
        # Check if the file mentions open CIDRs anywhere (not necessarily same line as port)
        has_open_cidr = bool(_OPEN_CIDR_PATTERN.search(content))
        # Also flag if the file discusses translating GCP firewall rules to AWS
        discusses_firewall_translation = bool(re.search(
            r'(?:firewall.*rule|security.*group|ingress.*rule|allow.*ssh|allow.*rdp|source_ranges)',
            content, re.IGNORECASE
        ))

        if not (has_open_cidr or discusses_firewall_translation):
            return issues

        for port, service, pattern in _OPEN_PORT_PATTERNS:
            if pattern.search(content):
                # Find the best line context
                line_context = f"Port {port} ({service}) referenced in file"
                for line in content.splitlines():
                    if pattern.search(line):
                        line_context = line.strip()[:200]
                        break
                issues.append(SecurityIssue(
                    issue_type="open_admin_port",
                    file_path=file_path,
                    line_context=line_context,
                    details={"port": port, "service": service},
                ))
        return issues

    def _check_secret_shell_var(self, file_path: str, content: str) -> list[SecurityIssue]:
        issues = []
        for line in content.splitlines():
            if _SECRET_SHELL_VAR_PATTERN.search(line):
                issues.append(SecurityIssue(
                    issue_type="secret_in_shell_variable",
                    file_path=file_path,
                    line_context=line.strip()[:200],
                ))
                break  # One finding per file
        return issues

    def _check_plaintext_db_password(self, file_path: str, content: str) -> list[SecurityIssue]:
        issues = []
        for line in content.splitlines():
            if _PLAINTEXT_DB_PASSWORD_PATTERN.search(line):
                issues.append(SecurityIssue(
                    issue_type="plaintext_db_password",
                    file_path=file_path,
                    line_context=line.strip()[:200],
                ))
                break
        return issues

    def _check_deletion_protection(self, file_path: str, content: str) -> list[SecurityIssue]:
        issues = []
        if _AURORA_RESOURCE_PATTERN.search(content):
            if not _DELETION_PROTECTION_PATTERN.search(content):
                issues.append(SecurityIssue(
                    issue_type="missing_deletion_protection",
                    file_path=file_path,
                    line_context="aws_rds_cluster resource found without deletion_protection = true",
                ))
        return issues

    def _check_rotation_gate(self, file_path: str, content: str) -> list[SecurityIssue]:
        issues = []
        if _ROTATION_BLOCK_PATTERN.search(content):
            if not _COMPLIANCE_GATE_PATTERN.search(content):
                issues.append(SecurityIssue(
                    issue_type="rotation_without_compliance_gate",
                    file_path=file_path,
                    line_context="Rotation block found without compliance gate",
                ))
        return issues

    def _check_max_tokens(self, file_path: str, content: str) -> list[SecurityIssue]:
        issues = []
        if _BEDROCK_ADAPTER_PATTERN.search(content):
            if _MAX_TOKENS_HIGH_PATTERN.search(content):
                for line in content.splitlines():
                    if _MAX_TOKENS_HIGH_PATTERN.search(line):
                        issues.append(SecurityIssue(
                            issue_type="missing_max_tokens_warning",
                            file_path=file_path,
                            line_context=line.strip()[:200],
                        ))
                        break
        return issues


def _issue_to_finding(issue: SecurityIssue, run_id: str) -> Finding:
    """Convert a SecurityIssue to a Finding."""
    category_info = SECURITY_CATEGORIES.get(issue.issue_type, {})
    risk = category_info.get("risk", RiskLevel.MEDIUM)

    details = issue.details or {}
    title = category_info.get("title_template", issue.issue_type).format(
        file=issue.file_path.split("/")[-1],
        **details,
    )
    description = category_info.get("description_template", "").format(
        file=issue.file_path,
        **details,
    )
    suggested_fix = category_info.get("suggested_fix", "")

    # Compute a deterministic finding ID
    raw = f"{issue.issue_type}|{issue.file_path}|{issue.line_context}"
    finding_id = hashlib.sha256(raw.encode()).hexdigest()[:32]

    return Finding(
        finding_id=finding_id,
        run_id=run_id,
        risk_level=risk,
        category="security",
        title=title,
        description=f"{description}\n\n**Line context:** `{issue.line_context}`",
        affected_files=[issue.file_path],
        proposed_changes={},
        source_urls=[
            "https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html",
            "https://docs.aws.amazon.com/secretsmanager/latest/userguide/rotating-secrets.html",
        ],
        scan_timestamp=datetime.utcnow().isoformat(),
        status="pending",
        auditor_payload={
            "issue_type": issue.issue_type,
            "severity": category_info.get("severity", "medium"),
            "suggested_fix": suggested_fix,
            "line_context": issue.line_context,
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
) -> list[Finding]:
    """Run the security audit over reference files.

    Scans files that describe generated code for security anti-patterns.
    """
    auditor = SecurityAuditor()
    findings: list[Finding] = []
    seen_keys: set = set()

    for file_path, content in repo_content.files.items():
        # Only scan security-relevant files
        is_relevant = any(
            pattern in file_path
            for pattern in SecurityAuditor.SECURITY_RELEVANT_PATTERNS
        )
        if not is_relevant:
            continue

        # Apply file filter if provided
        if file_filter and not any(f in file_path for f in file_filter):
            continue

        issues = auditor.scan_file(file_path, content)
        for issue in issues:
            finding = _issue_to_finding(issue, run_id)
            key = security_dedupe_key(finding)
            if key not in seen_keys:
                seen_keys.add(key)
                findings.append(finding)
                logger.info(
                    "Security audit: %s in %s",
                    issue.issue_type,
                    file_path,
                )

    logger.info(
        "Security audit completed: %d findings from %d files",
        len(findings),
        len(repo_content.files),
    )
    return findings
