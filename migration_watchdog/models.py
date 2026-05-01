"""Core data models for the Migration Plugin Watchdog.

Defines dataclasses and enums used across the watchdog system, including
Finding, ScanRun, RepoContent, PullRequest, Dismissal, RiskLevel,
AuthoritativeData, pricing models, model staleness models, and review results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(str, Enum):
    """Risk classification for findings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Dismissal:
    """A dismissal record with cooldown period."""

    dismissed_at: str  # ISO 8601
    cooldown_expires: str  # ISO 8601 (dismissed_at + 2 months)
    reason: str | None = None


@dataclass
class Finding:
    """A single identified discrepancy between repo content and authoritative sources."""

    finding_id: str  # UUID
    run_id: str  # scan run identifier
    risk_level: RiskLevel
    category: str  # maps to discrepancy category
    title: str
    description: str
    affected_files: list[str] = field(default_factory=list)
    proposed_changes: dict[str, str] = field(default_factory=dict)  # file_path -> proposed content
    source_urls: list[str] = field(default_factory=list)
    scan_timestamp: str = ""
    status: str = "pending"  # "pending", "approved", "declined"
    review_status: str | None = None  # "confirmed", "corrected", "disputed", or None
    review_notes: str | None = None
    primary_reasoning: str | None = None
    partial_data_warning: bool = False
    pr_url: str | None = None
    dismissal: Dismissal | None = None


@dataclass
class ScanRun:
    """Record of a single scan execution."""

    run_id: str  # UUID
    start_timestamp: str  # ISO 8601
    end_timestamp: str | None = None
    status: str = "running"  # "running", "completed", "failed"
    failure_reason: str | None = None
    findings_count: int = 0
    findings_by_risk: dict[str, int] = field(default_factory=dict)
    partial_source_failures: list[str] = field(default_factory=list)


@dataclass
class PullRequest:
    """An open pull request on the target repo."""

    number: int
    title: str
    body: str
    changed_files: list[str] = field(default_factory=list)
    author: str = ""


@dataclass
class RepoContent:
    """Snapshot of target repo state at scan time."""

    files: dict[str, str] = field(default_factory=dict)  # path -> content
    open_prs: list[PullRequest] = field(default_factory=list)
    commit_sha: str = ""
    fetched_at: str = ""


@dataclass
class GeminiModelData:
    """Current Gemini model data from authoritative sources."""

    models: list[dict] = field(default_factory=list)
    pricing: dict[str, dict] = field(default_factory=dict)
    deprecations: list[dict] = field(default_factory=list)
    fetched_at: str = ""


@dataclass
class OpenAIModelData:
    """Current OpenAI model data from authoritative sources."""

    pricing: dict[str, dict] = field(default_factory=dict)
    deprecations: list[dict] = field(default_factory=list)
    fetched_at: str = ""


@dataclass
class ModelLifecycleEntry:
    """A single model's lifecycle status from Bedrock."""

    model_name: str
    model_id: str
    status: str  # "active", "legacy", "eol"
    eol_date: str | None = None
    replacement: str | None = None


@dataclass
class BedrockLifecycle:
    """Current Bedrock model lifecycle data."""

    models: list[ModelLifecycleEntry] = field(default_factory=list)
    fetched_at: str = ""


@dataclass
class AuthoritativeData:
    """Aggregated data from all authoritative sources."""

    aws_docs: dict[str, str] = field(default_factory=dict)  # topic -> content
    aws_blog_posts: list[dict] = field(default_factory=list)
    aws_whats_new: list[dict] = field(default_factory=list)
    gemini_models: GeminiModelData = field(default_factory=GeminiModelData)
    openai_models: OpenAIModelData = field(default_factory=OpenAIModelData)
    bedrock_lifecycle: BedrockLifecycle = field(default_factory=BedrockLifecycle)
    aws_pricing: dict[str, dict] = field(default_factory=dict)  # service -> pricing data
    partial_failures: list[str] = field(default_factory=list)


@dataclass
class PricingEntry:
    """A single price extracted from the pricing cache."""

    service: str  # e.g., "fargate", "bedrock_claude_sonnet_4_6"
    metric: str  # e.g., "per_vcpu_hour", "input_per_1m_tokens"
    cached_value: float
    cached_last_updated: str  # date from pricing-cache.md header
    current_value: float | None = None  # from AWS Pricing API or provider page
    difference_pct: float | None = None
    tolerance_pct: float = 0.0  # 5-10% for infra, 15-25% for AI
    exceeds_tolerance: bool = False


@dataclass
class PricingValidationResult:
    """Result of validating the pricing cache against current prices."""

    entries: list[PricingEntry] = field(default_factory=list)
    stale_date: bool = False  # True if last-updated > 30 days old
    cache_last_updated: str = ""
    validation_timestamp: str = ""


@dataclass
class ModelComparisonEntry:
    """Comparison of a model entry in repo vs authoritative source."""

    provider: str  # "gemini", "openai", "bedrock"
    model_name: str
    repo_status: str | None = None  # status in repo file, or None if missing
    current_status: str = ""  # current status from authoritative source
    repo_pricing: dict | None = None  # {input: float, output: float} from repo
    current_pricing: dict | None = None
    repo_eol_date: str | None = None
    current_eol_date: str | None = None
    change_type: str = ""  # "new_model", "deprecated", "pricing_change", etc.


@dataclass
class ModelStalenessResult:
    """Result of comparing model data in repo vs authoritative sources."""

    gemini_changes: list[ModelComparisonEntry] = field(default_factory=list)
    openai_changes: list[ModelComparisonEntry] = field(default_factory=list)
    bedrock_lifecycle_changes: list[ModelComparisonEntry] = field(default_factory=list)
    scan_timestamp: str = ""


@dataclass
class ReviewResult:
    """Result of the Review LLM quality check on a finding."""

    finding_id: str
    review_status: str  # "confirmed", "corrected", "disputed"
    issues: list[str] = field(default_factory=list)
    corrected_description: str | None = None
    corrected_changes: dict[str, str] | None = None
    reviewer_notes: str = ""
    primary_reasoning: str = ""


# --- Helper Functions ---

LOW_CATEGORIES = {"pricing", "model_deprecation", "new_model"}
MEDIUM_CATEGORIES = {"guidance_update", "new_content"}
HIGH_CATEGORIES = {"structural", "core_removal", "refactoring"}


def classify_risk(category: str) -> RiskLevel:
    """Classify risk based on discrepancy category.

    Low: pricing updates, model deprecations, new model availability
    Medium: architectural guidance changes, new content suggestions
    High: structural reorganization, removal of core guidance, refactoring

    Defaults to Medium for unknown categories.
    """
    if category in LOW_CATEGORIES:
        return RiskLevel.LOW
    elif category in HIGH_CATEGORIES:
        return RiskLevel.HIGH
    else:
        return RiskLevel.MEDIUM
