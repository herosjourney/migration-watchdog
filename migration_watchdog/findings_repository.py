"""DynamoDB data access layer for the Findings Store.

Implements persistence operations for findings, dismissals, and scan runs
using a single-table DynamoDB design with PK/SK keys and GSIs for
status-based and risk-level-based queries.
"""

from __future__ import annotations

from datetime import datetime

from boto3.dynamodb.conditions import Attr, Key
from dateutil.relativedelta import relativedelta

from migration_watchdog.models import Dismissal, Finding, RiskLevel, ScanRun


class FindingsRepository:
    """DynamoDB repository for findings, dismissals, and scan runs."""

    def __init__(self, dynamodb_resource, table_name: str) -> None:
        self._table = dynamodb_resource.Table(table_name)

    # ------------------------------------------------------------------ #
    # Finding CRUD
    # ------------------------------------------------------------------ #

    def save_finding(self, finding: Finding) -> None:
        """Persist a finding using single-table design.

        Writes two items:
        1. The Finding record  (PK=FINDING#{id}, SK=FINDING)
        2. A FindingByRun record (PK=RUN#{run_id}, SK=FINDING#{id})
        """
        item = self._finding_to_item(finding)
        self._table.put_item(Item=item)

        # FindingByRun sparse projection
        self._table.put_item(
            Item={
                "PK": f"RUN#{finding.run_id}",
                "SK": f"FINDING#{finding.finding_id}",
                "risk_level": finding.risk_level.value if isinstance(finding.risk_level, RiskLevel) else finding.risk_level,
                "title": finding.title,
                "status": finding.status,
            }
        )

    def get_finding(self, finding_id: str) -> Finding | None:
        """Retrieve a finding by ID."""
        resp = self._table.get_item(
            Key={"PK": f"FINDING#{finding_id}", "SK": "FINDING"}
        )
        item = resp.get("Item")
        if item is None:
            return None
        return self._item_to_finding(item)

    def list_findings(
        self,
        status: str | None = None,
        risk_level: str | None = None,
        run_id: str | None = None,
        exclude_dismissed: bool = True,
    ) -> list[Finding]:
        """Query findings with optional filters.

        - status  -> GSI1 (PK=status, SK=scan_timestamp)
        - risk_level -> GSI2 (PK=risk_level, SK=scan_timestamp)
        - run_id  -> main table PK=RUN#{run_id}, SK begins_with FINDING#
        - exclude_dismissed -> filter out findings with active dismissal cooldowns
        """
        if run_id is not None:
            return self._list_findings_by_run(run_id, exclude_dismissed)

        if status is not None:
            items = self._query_gsi("GSI1", status)
        elif risk_level is not None:
            items = self._query_gsi("GSI2", risk_level)
        else:
            # Full table scan for FINDING entities
            items = self._scan_findings()

        findings = [self._item_to_finding(i) for i in items]

        if exclude_dismissed:
            findings = [f for f in findings if not self._has_active_dismissal(f)]

        return findings

    def update_finding_status(self, finding_id: str, status: str) -> None:
        """Update the status attribute of a finding."""
        self._table.update_item(
            Key={"PK": f"FINDING#{finding_id}", "SK": "FINDING"},
            UpdateExpression="SET #s = :s, updated_at = :u",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": status,
                ":u": datetime.utcnow().isoformat(),
            },
        )

    # ------------------------------------------------------------------ #
    # Dismissal
    # ------------------------------------------------------------------ #

    def record_dismissal(self, finding_id: str, cooldown_months: int = 2) -> None:
        """Record a dismissal with cooldown expiry timestamp."""
        now = datetime.utcnow()
        cooldown_expires = now + relativedelta(months=cooldown_months)
        dismissed_at = now.isoformat()
        cooldown_expires_iso = cooldown_expires.isoformat()

        # Write a Dismissal record
        self._table.put_item(
            Item={
                "PK": f"FINDING#{finding_id}",
                "SK": f"DISMISSAL#{dismissed_at}",
                "cooldown_expires": cooldown_expires_iso,
                "reason": None,
            }
        )

        # Update the Finding's dismissal field
        self._table.update_item(
            Key={"PK": f"FINDING#{finding_id}", "SK": "FINDING"},
            UpdateExpression="SET dismissal = :d, updated_at = :u",
            ExpressionAttributeValues={
                ":d": {
                    "dismissed_at": dismissed_at,
                    "cooldown_expires": cooldown_expires_iso,
                    "reason": None,
                },
                ":u": datetime.utcnow().isoformat(),
            },
        )

    def is_dismissed(self, finding_id: str) -> bool:
        """Check if a finding has an active dismissal cooldown."""
        finding = self.get_finding(finding_id)
        if finding is None:
            return False
        return self._has_active_dismissal(finding)

    # ------------------------------------------------------------------ #
    # Scan Runs
    # ------------------------------------------------------------------ #

    def save_run(self, run: ScanRun) -> None:
        """Persist a scan run: PK=RUN#{run_id}, SK=RUN."""
        item = self._run_to_item(run)
        self._table.put_item(Item=item)

    def get_run(self, run_id: str) -> ScanRun | None:
        """Retrieve a scan run by ID."""
        resp = self._table.get_item(
            Key={"PK": f"RUN#{run_id}", "SK": "RUN"}
        )
        item = resp.get("Item")
        if item is None:
            return None
        return self._item_to_run(item)

    # ------------------------------------------------------------------ #
    # History
    # ------------------------------------------------------------------ #

    def get_findings_history(self, file_path: str | None = None) -> list[Finding]:
        """Get historical findings, optionally filtered by file path."""
        items = self._scan_findings()
        findings = [self._item_to_finding(i) for i in items]

        if file_path is not None:
            findings = [
                f for f in findings if file_path in f.affected_files
            ]

        return findings

    # ------------------------------------------------------------------ #
    # Private helpers — serialisation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _finding_to_item(finding: Finding) -> dict:
        """Convert a Finding dataclass to a DynamoDB item dict."""
        risk = finding.risk_level.value if isinstance(finding.risk_level, RiskLevel) else finding.risk_level

        item: dict = {
            "PK": f"FINDING#{finding.finding_id}",
            "SK": "FINDING",
            "finding_id": finding.finding_id,
            "run_id": finding.run_id,
            "risk_level": risk,
            "category": finding.category,
            "title": finding.title,
            "description": finding.description,
            "scan_timestamp": finding.scan_timestamp,
            "status": finding.status,
            "partial_data_warning": finding.partial_data_warning,
        }

        # StringSet — DynamoDB does not allow empty sets
        if finding.affected_files:
            item["affected_files"] = set(finding.affected_files)
        else:
            item["affected_files"] = None

        # Map
        item["proposed_changes"] = finding.proposed_changes if finding.proposed_changes else {}

        # List
        item["source_urls"] = finding.source_urls if finding.source_urls else []

        # Optional scalars
        item["review_status"] = finding.review_status
        item["review_notes"] = finding.review_notes
        item["primary_reasoning"] = finding.primary_reasoning
        item["pr_url"] = finding.pr_url

        # Nested map (Dismissal) or None
        if finding.dismissal is not None:
            item["dismissal"] = {
                "dismissed_at": finding.dismissal.dismissed_at,
                "cooldown_expires": finding.dismissal.cooldown_expires,
                "reason": finding.dismissal.reason,
            }
        else:
            item["dismissal"] = None

        now_iso = datetime.utcnow().isoformat()
        item["created_at"] = now_iso
        item["updated_at"] = now_iso

        return item

    @staticmethod
    def _item_to_finding(item: dict) -> Finding:
        """Convert a DynamoDB item dict to a Finding dataclass."""
        affected = item.get("affected_files")
        if affected is None:
            affected_list: list[str] = []
        elif isinstance(affected, set):
            affected_list = sorted(affected)
        else:
            affected_list = list(affected)

        dismissal_data = item.get("dismissal")
        dismissal: Dismissal | None = None
        if dismissal_data and isinstance(dismissal_data, dict) and dismissal_data.get("dismissed_at"):
            dismissal = Dismissal(
                dismissed_at=dismissal_data["dismissed_at"],
                cooldown_expires=dismissal_data["cooldown_expires"],
                reason=dismissal_data.get("reason"),
            )

        risk_val = item.get("risk_level", "medium")
        try:
            risk_level = RiskLevel(risk_val)
        except ValueError:
            risk_level = RiskLevel.MEDIUM

        return Finding(
            finding_id=item["finding_id"],
            run_id=item["run_id"],
            risk_level=risk_level,
            category=item.get("category", ""),
            title=item.get("title", ""),
            description=item.get("description", ""),
            affected_files=affected_list,
            proposed_changes=item.get("proposed_changes") or {},
            source_urls=item.get("source_urls") or [],
            scan_timestamp=item.get("scan_timestamp", ""),
            status=item.get("status", "pending"),
            review_status=item.get("review_status"),
            review_notes=item.get("review_notes"),
            primary_reasoning=item.get("primary_reasoning"),
            partial_data_warning=item.get("partial_data_warning", False),
            pr_url=item.get("pr_url"),
            dismissal=dismissal,
        )

    @staticmethod
    def _run_to_item(run: ScanRun) -> dict:
        """Convert a ScanRun dataclass to a DynamoDB item dict."""
        return {
            "PK": f"RUN#{run.run_id}",
            "SK": "RUN",
            "run_id": run.run_id,
            "start_timestamp": run.start_timestamp,
            "end_timestamp": run.end_timestamp,
            "status": run.status,
            "failure_reason": run.failure_reason,
            "findings_count": run.findings_count,
            "findings_by_risk": run.findings_by_risk if run.findings_by_risk else {},
            "partial_source_failures": run.partial_source_failures if run.partial_source_failures else [],
        }

    @staticmethod
    def _item_to_run(item: dict) -> ScanRun:
        """Convert a DynamoDB item dict to a ScanRun dataclass."""
        partial = item.get("partial_source_failures")
        if partial is None:
            partial_list: list[str] = []
        elif isinstance(partial, set):
            partial_list = sorted(partial)
        else:
            partial_list = list(partial)

        return ScanRun(
            run_id=item["run_id"],
            start_timestamp=item.get("start_timestamp", ""),
            end_timestamp=item.get("end_timestamp"),
            status=item.get("status", "running"),
            failure_reason=item.get("failure_reason"),
            findings_count=int(item.get("findings_count", 0)),
            findings_by_risk=item.get("findings_by_risk") or {},
            partial_source_failures=partial_list,
        )

    # ------------------------------------------------------------------ #
    # Private helpers — queries
    # ------------------------------------------------------------------ #

    def _query_gsi(self, index_name: str, pk_value: str) -> list[dict]:
        """Query a GSI by its partition key."""
        resp = self._table.query(
            IndexName=index_name,
            KeyConditionExpression=Key("PK").eq(pk_value)
            if index_name not in ("GSI1", "GSI2")
            else self._gsi_key_condition(index_name, pk_value),
        )
        return resp.get("Items", [])

    def _gsi_key_condition(self, index_name: str, pk_value: str):
        """Build key condition for GSI queries."""
        if index_name == "GSI1":
            return Key("status").eq(pk_value)
        elif index_name == "GSI2":
            return Key("risk_level").eq(pk_value)
        return Key("PK").eq(pk_value)

    def _list_findings_by_run(self, run_id: str, exclude_dismissed: bool) -> list[Finding]:
        """List findings associated with a specific run."""
        resp = self._table.query(
            KeyConditionExpression=Key("PK").eq(f"RUN#{run_id}")
            & Key("SK").begins_with("FINDING#"),
        )
        finding_ids = []
        for item in resp.get("Items", []):
            # Extract finding_id from SK = FINDING#{finding_id}
            sk = item["SK"]
            fid = sk.replace("FINDING#", "", 1)
            finding_ids.append(fid)

        findings = []
        for fid in finding_ids:
            f = self.get_finding(fid)
            if f is not None:
                findings.append(f)

        if exclude_dismissed:
            findings = [f for f in findings if not self._has_active_dismissal(f)]

        return findings

    def _scan_findings(self) -> list[dict]:
        """Scan the table for all Finding entities."""
        resp = self._table.scan(
            FilterExpression=Attr("SK").eq("FINDING")
            & Attr("PK").begins_with("FINDING#"),
        )
        return resp.get("Items", [])

    @staticmethod
    def _has_active_dismissal(finding: Finding) -> bool:
        """Check if a finding has an active (non-expired) dismissal."""
        if finding.dismissal is None:
            return False
        try:
            expires = datetime.fromisoformat(finding.dismissal.cooldown_expires)
            return datetime.utcnow() < expires
        except (ValueError, TypeError):
            return False
