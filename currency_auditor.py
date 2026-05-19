"""Currency Auditor — extraction layer.

Implements the ``Claim`` dataclass and ``ClaimExtractor`` class that read a
Reference_File markdown string and produce a structured list of verifiable
factual claims.

The extraction layer uses an LLM call (Strands/Bedrock, same pattern as
``analysis_agent.py``) to identify candidate claims, then computes
deterministic SHA-256 fingerprints locally and deduplicates by ``claim_id``
(first occurrence wins).

The verification layer (``ClaimVerifier``) is implemented in task 6.1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from migration_watchdog.alias_table import AliasTable
from migration_watchdog.source_fetcher import AwsDocsSearcher

logger = logging.getLogger(__name__)

# Module-level docs searcher instance (shared across all ClaimExtractor/Verifier calls)
_docs_searcher = AwsDocsSearcher()


@tool
def search_aws_docs(query: str) -> str:
    """Search AWS documentation and return relevant content for claim verification.

    Use this tool to verify factual claims against current AWS documentation
    before including them in your output. Search for the specific service,
    feature, or fact you want to verify.

    Args:
        query: A specific search query, e.g. "App Runner new customers availability",
               "Fargate region availability", "Claude Opus 4 model ID Bedrock"

    Returns:
        Relevant text excerpts from AWS documentation pages.
    """
    return _docs_searcher.search_and_fetch_sync(query)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_CLAIM_TYPES: frozenset[str] = frozenset(
    {
        "region_list",
        "price",
        "quota_limit",
        "service_name",
        "feature_availability",
        "api_name",
        "model_id",
        "eol_date",
        "region_count",
        "service_limit",      # e.g., "10,000 RPM", "2M output TPM", "500 tasks per cluster"
        "preview_status",     # e.g., "AgentCore Harness is in public preview"
        "other_factual",
    }
)

# Fuzzy / unverifiable patterns — claims containing these phrases are skipped.
_FUZZY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bmost\s+regions\b", re.IGNORECASE),
    re.compile(r"\bapproximately\b", re.IGNORECASE),
    re.compile(r"\baround\b", re.IGNORECASE),
    re.compile(r"\broughly\b", re.IGNORECASE),
]

# Quarter-format date pattern: Q1/Q2/Q3/Q4 YYYY
_QUARTER_DATE_RE = re.compile(r"\bQ[1-4]\s+\d{4}\b", re.IGNORECASE)

# Normalised date patterns for _normalize_eol_date
_YYYY_MM_DD_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTH_YEAR_RE = re.compile(
    r"^(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(?P<year>\d{4})$",
    re.IGNORECASE,
)
_BEFORE_BY_RE = re.compile(
    r"^(?:before|by)\s+(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(?P<year>\d{4})$",
    re.IGNORECASE,
)

_MONTH_LAST_DAY: dict[str, int] = {
    "january": 31,
    "february": 28,  # conservative; leap-year handling not required for EOL dates
    "march": 31,
    "april": 30,
    "may": 31,
    "june": 30,
    "july": 31,
    "august": 31,
    "september": 30,
    "october": 31,
    "november": 30,
    "december": 31,
}

_MONTH_NUMBER: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


# ---------------------------------------------------------------------------
# Claim dataclass
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """A single verifiable factual assertion extracted from a Reference_File.

    ``claim_id`` is a deterministic SHA-256 fingerprint:
    ``SHA-256(normalize(claim_text)|source_file|claim_type)``
    where ``normalize`` = lowercase + whitespace-collapse.
    """

    claim_id: str
    claim_text: str
    claim_type: str  # one of ALLOWED_CLAIM_TYPES
    claim_subtype: str | None  # required when claim_type == "other_factual"
    source_file: str
    line_context: str
    verification_query: str
    # Price-specific fields (populated when claim_type == "price")
    service: str | None = None
    metric: str | None = None
    # Model-specific field (populated when claim_type == "model_id")
    # Extraction-time only; copied 1:1 into auditor_payload["alias_found"].
    alias_found: bool | None = None


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are a factual-claim extraction assistant for a migration documentation auditor.

Your task is to read a Markdown Reference_File and extract ONLY objectively verifiable
factual claims — assertions that can be confirmed or refuted by querying AWS documentation
or AWS pricing sources.

IMPORTANT: You have access to the `search_aws_docs` tool. Before finalizing any claim,
use this tool to verify the claim against current AWS documentation. This is mandatory for:
- Service availability claims (e.g. "App Runner is available for new customers")
- Feature status claims (e.g. "preview", "GA", "deprecated", "closed")
- Region availability claims
- Any claim about a service being recommended or not recommended

If your search reveals that a claim in the document is INCORRECT or OUTDATED based on
current AWS documentation, still include it in your output — the auditor needs to know
about stale claims. Set the verification_query to reflect what the docs actually say.

DO NOT extract:
- Opinions, recommendations, or design decisions
- Qualitative assessments ("best practice", "recommended approach")
- Fuzzy or approximate statements ("most regions", "approximately", "around", "roughly")
- Quarter-format dates (e.g., "Q2 2026") — these are unverifiable
- Conceptual explanations or architectural descriptions

DO extract:
- Specific region names or counts (e.g., "available in 9 regions", "us-east-1, eu-west-1, ...")
- Specific prices or pricing tiers — use claim_type: price for ALL of these:
  - Unit prices (e.g., "$0.048 per vCPU-hour", "$0.12 per ACU-hour")
  - Monthly cost estimates (e.g., "~$43–58/mo", "approximately $200/month")
  - Price ranges tied to a service (e.g., "Aurora Serverless v2 costs ~$43–58/mo")
  - Pricing tiers (e.g., "first 1M requests free, then $0.20 per million")
  Set "service" to the normalized service key (e.g., "aurora", "fargate", "bedrock")
  Set "metric" to the normalized metric (e.g., "per_acu_hour", "per_month_estimate", "per_vcpu_hour")
- Quota limits — use claim_type: quota_limit ONLY for hard service limits, NOT prices:
  (e.g., "default limit of 500 tasks per cluster", "max 10 concurrent executions")
- Service throughput limits (e.g., "10,000 RPM shared account limit", "2M output TPM cap") — use claim_type: service_limit
- Preview/GA status (e.g., "AgentCore Harness is in public preview", "available in 4 regions only") — use claim_type: preview_status
- Service names and their current status (e.g., "Amazon ECS Fargate is generally available")
- Feature availability statements (e.g., "Spot Instances are supported in Fargate")
- API names (e.g., "RunTask API")
- Model IDs or model names (e.g., "Claude Sonnet 4.6", "anthropic.claude-sonnet-4-5-20251001-v1:0")
- EOL / end-of-support dates (e.g., "support ends June 2026")
- Other objectively verifiable factual assertions

CRITICAL CLASSIFICATION RULE — price vs quota_limit:
- If the claim contains a dollar amount ($), it is ALWAYS claim_type: price — never quota_limit
- "min 0.5 ACU, ~$43–58/mo" → price (contains dollar amount)
- "default limit of 500 tasks" → quota_limit (no dollar amount, it's a count limit)
- "10,000 RPM limit" → service_limit (throughput limit, no dollar amount)

For each claim, output a JSON object with these fields:
- "claim_text": verbatim text of the claim (exact quote from the document)
- "claim_type": one of: region_list, price, quota_limit, service_name, feature_availability,
  api_name, model_id, eol_date, region_count, service_limit, preview_status, other_factual
- "claim_subtype": non-empty string ONLY when claim_type is "other_factual" (e.g.,
  "capacity_floor", "preview_vs_ga", "sdk_version", "iam_action", "sla_throughput");
  null for all other claim types
- "line_context": the surrounding sentence or paragraph (enough to locate the claim in the file)
- "verification_query": a specific, self-contained question that can be submitted to AWS
  documentation or pricing sources to verify this claim (e.g.,
  "What is the current on-demand price per vCPU-hour for AWS Fargate in us-east-1?")
- "service": normalized service key when claim_type is "price" (e.g., "fargate",
  "claude_sonnet"); null otherwise
- "metric": normalized metric key when claim_type is "price" (e.g., "per_vcpu_hour",
  "input_per_1m_tokens"); null otherwise

Output ONLY a JSON array of claim objects. Do not include any explanation or commentary.
If no verifiable claims are found, output an empty JSON array: []
"""


def _build_extraction_prompt(file_path: str, content: str) -> str:
    """Build the user message for the LLM extraction call."""
    return (
        f"Extract all verifiable factual claims from the following Reference_File.\n\n"
        f"File path: {file_path}\n\n"
        f"```markdown\n{content}\n```\n\n"
        "Output a JSON array of claim objects as described in the system prompt."
    )


# ---------------------------------------------------------------------------
# ClaimExtractor
# ---------------------------------------------------------------------------


class ClaimExtractor:
    """Reads a Reference_File markdown string and produces a list of ``Claim`` objects.

    Uses an LLM call (Strands/Bedrock, same pattern as ``analysis_agent.py``) to
    identify candidate claims, then computes deterministic SHA-256 fingerprints
    locally and deduplicates by ``claim_id`` (first occurrence wins).
    """

    def __init__(
        self,
        alias_table: AliasTable | None = None,
        model_id: str = "us.anthropic.claude-opus-4-7",
        region_name: str = "us-east-1",
    ) -> None:
        self._alias_table = alias_table or AliasTable()
        self._model_id = model_id
        self._region_name = region_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, file_path: str, content: str) -> list[Claim]:
        """Extract all verifiable claims from a Reference_File.

        Steps:
        1. Call LLM to identify candidate claims
        2. Compute ``claim_id`` for each candidate
        3. Skip fuzzy / unverifiable claims
        4. Deduplicate by ``claim_id`` (first occurrence wins)
        5. Return validated ``Claim`` list

        Parameters
        ----------
        file_path:
            Path of the Reference_File (used in ``claim_id`` computation and
            ``source_file`` field).
        content:
            Raw markdown content of the Reference_File.

        Returns
        -------
        list[Claim]
            Validated, deduplicated list of claims.  Empty when no verifiable
            claims are found.
        """
        raw_candidates = self._call_llm(file_path, content)
        if not raw_candidates:
            logger.info("ClaimExtractor: no verifiable claims found in %s", file_path)
            return []

        seen_ids: dict[str, int] = {}  # claim_id -> first-occurrence index
        claims: list[Claim] = []

        for candidate in raw_candidates:
            claim_text: str = candidate.get("claim_text", "").strip()
            claim_type: str = candidate.get("claim_type", "").strip()
            claim_subtype: str | None = candidate.get("claim_subtype") or None
            line_context: str = candidate.get("line_context", "").strip()
            verification_query: str = candidate.get("verification_query", "").strip()
            service: str | None = candidate.get("service") or None
            metric: str | None = candidate.get("metric") or None

            # --- Validate claim_type ---
            if claim_type not in ALLOWED_CLAIM_TYPES:
                logger.warning(
                    "ClaimExtractor: skipping claim with invalid claim_type=%r in %s",
                    claim_type,
                    file_path,
                )
                continue

            # --- Validate claim_subtype for other_factual ---
            if claim_type == "other_factual" and not claim_subtype:
                logger.warning(
                    "ClaimExtractor: skipping other_factual claim without claim_subtype in %s: %r",
                    file_path,
                    claim_text[:80],
                )
                continue

            # --- Skip fuzzy / unverifiable patterns ---
            if self._is_fuzzy(claim_text):
                logger.info(
                    "ClaimExtractor: skipping fuzzy/unverifiable claim in %s: %r",
                    file_path,
                    claim_text[:80],
                )
                continue

            # --- Handle eol_date normalization ---
            if claim_type == "eol_date":
                normalized_date = self._normalize_eol_date(claim_text)
                if normalized_date is None:
                    # Quarter-format or unverifiable — skip
                    logger.info(
                        "ClaimExtractor: skipping unverifiable eol_date in %s: %r",
                        file_path,
                        claim_text[:80],
                    )
                    continue
                # Use the normalized date as the canonical claim_text for ID computation
                # but preserve original in line_context
                claim_text = normalized_date

            # --- Handle model_id alias resolution ---
            alias_found: bool | None = None
            if claim_type == "model_id":
                resolved_name, alias_found = self._resolve_model_alias(
                    claim_text, self._alias_table
                )
                claim_text = resolved_name

            # --- Compute claim_id ---
            claim_id = self._compute_claim_id(claim_text, file_path, claim_type)

            # --- Deduplicate ---
            if claim_id in seen_ids:
                logger.debug(
                    "ClaimExtractor: duplicate claim_id %s in %s (first at index %d), skipping",
                    claim_id[:16],
                    file_path,
                    seen_ids[claim_id],
                )
                continue

            seen_ids[claim_id] = len(claims)

            # --- Validate required fields ---
            if not claim_text or not line_context or not verification_query:
                logger.warning(
                    "ClaimExtractor: skipping claim with missing required fields in %s",
                    file_path,
                )
                continue

            claims.append(
                Claim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    claim_type=claim_type,
                    claim_subtype=claim_subtype,
                    source_file=file_path,
                    line_context=line_context,
                    verification_query=verification_query,
                    service=service,
                    metric=metric,
                    alias_found=alias_found,
                )
            )

        if not claims:
            logger.info(
                "ClaimExtractor: no verifiable claims after filtering in %s", file_path
            )

        return claims

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, file_path: str, content: str) -> list[dict[str, Any]]:
        """Call the LLM to extract candidate claims.

        Returns a list of raw candidate dicts (not yet validated or deduplicated).
        Returns an empty list on any error.
        """
        try:
            model = BedrockModel(
                model_id=self._model_id,
                region_name=self._region_name,
                max_tokens=8000,
            )
            agent = Agent(
                model=model,
                system_prompt=_EXTRACTION_SYSTEM_PROMPT,
                tools=[search_aws_docs],
            )
            prompt = _build_extraction_prompt(file_path, content)
            response = agent(prompt)
            # Strands Agent returns an AgentResult; the text content is in .message
            response_text: str = str(response)
            return self._parse_llm_response(response_text, file_path)
        except Exception:
            logger.exception(
                "ClaimExtractor: LLM call failed for %s; returning empty list", file_path
            )
            return []

    def _parse_llm_response(
        self, response_text: str, file_path: str
    ) -> list[dict[str, Any]]:
        """Parse the LLM response text into a list of candidate dicts.

        Handles both bare JSON arrays and JSON embedded in markdown code fences.
        """
        text = response_text.strip()

        # Strip markdown code fences if present
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()

        # Find the first '[' to handle any leading prose
        bracket_idx = text.find("[")
        if bracket_idx != -1:
            text = text[bracket_idx:]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "ClaimExtractor: failed to parse LLM JSON response for %s; "
                "response preview: %r",
                file_path,
                response_text[:200],
            )
            return []

        if not isinstance(parsed, list):
            logger.warning(
                "ClaimExtractor: LLM response for %s is not a JSON array", file_path
            )
            return []

        return [item for item in parsed if isinstance(item, dict)]

    def _compute_claim_id(
        self, claim_text: str, source_file: str, claim_type: str
    ) -> str:
        """Compute a deterministic SHA-256 fingerprint for a claim.

        ``SHA-256(normalize(claim_text)|source_file|claim_type)``
        where ``normalize`` = lowercase + whitespace-collapse.
        """
        normalized = re.sub(r"\s+", " ", claim_text.lower().strip())
        raw = f"{normalized}|{source_file}|{claim_type}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _normalize_eol_date(self, date_str: str) -> str | None:
        """Normalize an EOL date string to YYYY-MM-DD.

        Returns ``None`` for quarter-format dates (Q1/Q2/Q3/Q4 YYYY) and other
        unverifiable formats; these are logged as unverifiable and skipped by
        the caller.

        Supported input formats:
        - ``YYYY-MM-DD`` — returned as-is
        - ``Month YYYY`` (e.g., ``"June 2026"``) — normalized to ``YYYY-MM-01``
        - ``before Month YYYY`` / ``by Month YYYY`` — normalized to last day of
          that month (boundary date); the caller preserves the original in
          ``line_context``
        - ``Q1/Q2/Q3/Q4 YYYY`` — returns ``None`` (unverifiable)
        """
        text = date_str.strip()

        # Already in YYYY-MM-DD
        if _YYYY_MM_DD_RE.match(text):
            return text

        # Quarter format — unverifiable
        if _QUARTER_DATE_RE.search(text):
            logger.info(
                "ClaimExtractor: quarter-format EOL date is unverifiable: %r", text
            )
            return None

        # "before Month YYYY" / "by Month YYYY" → last day of that month
        before_match = _BEFORE_BY_RE.match(text)
        if before_match:
            month_name = before_match.group("month").lower()
            year = int(before_match.group("year"))
            month_num = _MONTH_NUMBER[month_name]
            last_day = _MONTH_LAST_DAY[month_name]
            return f"{year:04d}-{month_num:02d}-{last_day:02d}"

        # "Month YYYY" → first day of that month
        month_match = _MONTH_YEAR_RE.match(text)
        if month_match:
            month_name = month_match.group("month").lower()
            year = int(month_match.group("year"))
            month_num = _MONTH_NUMBER[month_name]
            return f"{year:04d}-{month_num:02d}-01"

        # Unrecognised format
        logger.info(
            "ClaimExtractor: unrecognised EOL date format, treating as unverifiable: %r",
            text,
        )
        return None

    def _resolve_model_alias(
        self, name: str, alias_table: AliasTable
    ) -> tuple[str, bool]:
        """Resolve a human-readable model name to a canonical Bedrock model ID.

        Delegates to ``AliasTable.resolve()``.

        Returns
        -------
        tuple[str, bool]
            ``(canonical_id, found)`` — ``found`` is ``True`` when the alias
            table contained an entry for *name*, ``False`` otherwise.
        """
        return alias_table.resolve(name)

    @staticmethod
    def _is_fuzzy(claim_text: str) -> bool:
        """Return ``True`` when *claim_text* contains a fuzzy/unverifiable pattern."""
        for pattern in _FUZZY_PATTERNS:
            if pattern.search(claim_text):
                return True
        return False


# ---------------------------------------------------------------------------
# TTL cache support (24h for pricing, 7d for docs)
# ---------------------------------------------------------------------------

_PRICING_TTL_SECONDS = 24 * 60 * 60   # 24 hours
_DOCS_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days

# Module-level caches: key -> (value, expiry_timestamp)
_pricing_cache: dict[str, tuple[Any, float]] = {}
_docs_cache: dict[str, tuple[Any, float]] = {}


def _cache_get(cache: dict, key: str) -> Any | None:
    """Return cached value if present and not expired, else None."""
    import time
    entry = cache.get(key)
    if entry is None:
        return None
    value, expiry = entry
    if time.time() > expiry:
        del cache[key]
        return None
    return value


def _cache_set(cache: dict, key: str, value: Any, ttl_seconds: float) -> None:
    """Store value in cache with TTL."""
    import time
    cache[key] = (value, time.time() + ttl_seconds)


# ---------------------------------------------------------------------------
# VerificationResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Result of verifying a single ``Claim`` against authoritative sources.

    ``status`` values:
    - ``"verified_accurate"`` — claim matches authoritative data; no Finding raised.
    - ``"finding"`` — claim does not match; a Finding should be raised.
    - ``"unverified"`` — source returned no result or timed out; no Finding raised.

    ``severity`` values (only set when ``status == "finding"``):
    - ``"correctness"`` — factually wrong (maps to HIGH risk).
    - ``"outdated"`` — stale but not necessarily wrong (maps to MEDIUM risk).
    - ``"policy_change"`` — rename / preview→GA (maps to LOW risk).
    - ``"informational"`` — informational note (maps to LOW risk).
    """

    claim_id: str
    status: str                          # "verified_accurate" | "finding" | "unverified"
    severity: str | None                 # "correctness" | "outdated" | "policy_change" | "informational"
    actual_value: str | None
    verification_source: str | None
    suggested_fix: str | None
    price_verification_path: str | None  # "pricing_cache" | "direct_query"
    price_metadata: dict | None


# ---------------------------------------------------------------------------
# ClaimVerifier
# ---------------------------------------------------------------------------


class ClaimVerifier:
    """Routes each ``Claim`` to the correct verification path and returns a
    ``VerificationResult``.

    Verification paths:
    - ``price``: pricing_cache (via ``compare_pricing_entries()``) or direct_query.
    - ``region_list`` / ``region_count``: set equality / numeric comparison.
    - ``model_id``: exact match against Bedrock lifecycle after alias resolution.
    - ``eol_date``: exact match on normalised YYYY-MM-DD.
    - ``quota_limit`` / ``api_name``: compare against AWS docs → LLM judgment.
    - ``service_name`` / ``feature_availability``: LLM judgment using live docs search.
    - ``other_factual``: compare against AWS docs → LLM judgment.
    - No source result or timeout → status="unverified".
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-opus-4-7",
        region_name: str = "us-east-1",
    ) -> None:
        self._model_id = model_id
        self._region_name = region_name

    def verify(
        self,
        claim: "Claim",
        authoritative_data: Any,  # AuthoritativeData
        pricing_cache_entries: list[Any],  # list[PricingEntry]
    ) -> VerificationResult:
        """Route *claim* to the correct verification path."""
        try:
            match claim.claim_type:
                case "price":
                    return self._verify_price(claim, authoritative_data, pricing_cache_entries)
                case "region_list" | "region_count":
                    return self._verify_region(claim, authoritative_data)
                case "model_id":
                    return self._verify_model_id(claim, authoritative_data)
                case "eol_date":
                    return self._verify_eol_date(claim, authoritative_data)
                case "quota_limit" | "api_name":
                    return self._verify_aws_docs(claim, authoritative_data, severity="outdated")
                case "service_name" | "feature_availability":
                    return self._verify_service(claim, authoritative_data)
                case "service_limit":
                    return self._verify_aws_docs(claim, authoritative_data, severity="correctness")
                case "preview_status":
                    return self._verify_service(claim, authoritative_data)
                case "other_factual":
                    return self._verify_aws_docs(claim, authoritative_data, severity="outdated")
                case _:
                    logger.warning(
                        "ClaimVerifier: unknown claim_type=%r for claim_id=%s; returning unverified",
                        claim.claim_type,
                        claim.claim_id[:16],
                    )
                    return VerificationResult(
                        claim_id=claim.claim_id,
                        status="unverified",
                        severity=None,
                        actual_value=None,
                        verification_source=None,
                        suggested_fix=None,
                        price_verification_path=None,
                        price_metadata=None,
                    )
        except Exception:
            logger.exception(
                "ClaimVerifier: unexpected error verifying claim_id=%s; returning unverified",
                claim.claim_id[:16],
            )
            return VerificationResult(
                claim_id=claim.claim_id,
                status="unverified",
                severity=None,
                actual_value=None,
                verification_source=None,
                suggested_fix=None,
                price_verification_path=None,
                price_metadata=None,
            )

    # ------------------------------------------------------------------
    # Price verification
    # ------------------------------------------------------------------

    def _verify_price(
        self,
        claim: "Claim",
        authoritative_data: Any,
        pricing_cache_entries: list[Any],
    ) -> VerificationResult:
        """Route to pricing_cache path or direct_query path."""
        from migration_watchdog.pricing_comparator import compare_pricing_entries

        cache_entry = self._find_cache_entry(claim, pricing_cache_entries)

        if cache_entry is not None:
            # --- pricing_cache path ---
            path = "pricing_cache"
            current_value: float | None = None
            if claim.service and claim.metric:
                current_value = (
                    authoritative_data.aws_pricing
                    .get(claim.service, {})
                    .get(claim.metric)
                )
            if current_value is None:
                # Cannot verify without a current price
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status="unverified",
                    severity=None,
                    actual_value=None,
                    verification_source=None,
                    suggested_fix=None,
                    price_verification_path=path,
                    price_metadata=self._build_price_metadata(claim, cache_entry),
                )

            current_prices = {
                (claim.service or ""): {(claim.metric or ""): current_value}
            }
            validation = compare_pricing_entries([cache_entry], current_prices)
            entry = validation.entries[0] if validation.entries else None

            if entry is None:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status="unverified",
                    severity=None,
                    actual_value=None,
                    verification_source=None,
                    suggested_fix=None,
                    price_verification_path=path,
                    price_metadata=self._build_price_metadata(claim, cache_entry),
                )

            price_meta = self._build_price_metadata(claim, cache_entry)

            if entry.exceeds_tolerance:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status="finding",
                    severity="outdated",
                    actual_value=str(current_value),
                    verification_source=None,
                    suggested_fix=(
                        f"Update {claim.metric or 'price'} for {claim.service or 'service'} "
                        f"from {cache_entry.cached_value} to {current_value} (USD)"
                    ),
                    price_verification_path=path,
                    price_metadata=price_meta,
                )
            return VerificationResult(
                claim_id=claim.claim_id,
                status="verified_accurate",
                severity=None,
                actual_value=str(current_value),
                verification_source=None,
                suggested_fix=None,
                price_verification_path=path,
                price_metadata=price_meta,
            )

        else:
            # --- direct_query path ---
            path = "direct_query"
            current_value = self._query_aws_pricing(claim, authoritative_data)
            if current_value is None:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status="unverified",
                    severity=None,
                    actual_value=None,
                    verification_source=None,
                    suggested_fix=None,
                    price_verification_path=path,
                    price_metadata=self._build_price_metadata(claim, None),
                )

            # Apply same tolerance logic as pricing_comparator
            cached_val = self._extract_price_from_claim(claim)
            if cached_val is None:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status="unverified",
                    severity=None,
                    actual_value=str(current_value),
                    verification_source=None,
                    suggested_fix=None,
                    price_verification_path=path,
                    price_metadata=self._build_price_metadata(claim, None),
                )

            exceeds = self._exceeds_tolerance_direct(cached_val, current_value, claim)
            price_meta = self._build_price_metadata(claim, None)

            if exceeds:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status="finding",
                    severity="outdated",
                    actual_value=str(current_value),
                    verification_source=None,
                    suggested_fix=(
                        f"Update {claim.metric or 'price'} for {claim.service or 'service'} "
                        f"from {cached_val} to {current_value} (USD)"
                    ),
                    price_verification_path=path,
                    price_metadata=price_meta,
                )
            return VerificationResult(
                claim_id=claim.claim_id,
                status="verified_accurate",
                severity=None,
                actual_value=str(current_value),
                verification_source=None,
                suggested_fix=None,
                price_verification_path=path,
                price_metadata=price_meta,
            )

    # ------------------------------------------------------------------
    # Region verification
    # ------------------------------------------------------------------

    def _verify_region(
        self,
        claim: "Claim",
        authoritative_data: Any,
    ) -> VerificationResult:
        """Set equality for region_list; numeric comparison for region_count."""
        # Look up authoritative region data from aws_docs
        authoritative_regions = self._get_authoritative_regions(claim, authoritative_data)
        if authoritative_regions is None:
            return VerificationResult(
                claim_id=claim.claim_id,
                status="unverified",
                severity=None,
                actual_value=None,
                verification_source=None,
                suggested_fix=None,
                price_verification_path=None,
                price_metadata=None,
            )

        if claim.claim_type == "region_list":
            # Parse claimed regions from claim_text (comma/space separated)
            claimed_set = self._parse_region_set(claim.claim_text)
            actual_set = set(authoritative_regions) if isinstance(authoritative_regions, (list, set)) else set()
            if claimed_set == actual_set:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status="verified_accurate",
                    severity=None,
                    actual_value=", ".join(sorted(actual_set)),
                    verification_source=self._get_docs_source(claim, authoritative_data),
                    suggested_fix=None,
                    price_verification_path=None,
                    price_metadata=None,
                )
            missing = actual_set - claimed_set
            extra = claimed_set - actual_set
            fix_parts = []
            if missing:
                fix_parts.append(f"add regions: {', '.join(sorted(missing))}")
            if extra:
                fix_parts.append(f"remove regions: {', '.join(sorted(extra))}")
            return VerificationResult(
                claim_id=claim.claim_id,
                status="finding",
                severity="correctness",
                actual_value=", ".join(sorted(actual_set)),
                verification_source=self._get_docs_source(claim, authoritative_data),
                suggested_fix=f"Update region list — {'; '.join(fix_parts)}",
                price_verification_path=None,
                price_metadata=None,
            )

        else:  # region_count
            claimed_count = self._extract_integer(claim.claim_text)
            actual_count = (
                len(authoritative_regions)
                if isinstance(authoritative_regions, (list, set))
                else authoritative_regions
            )
            if claimed_count is None:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status="unverified",
                    severity=None,
                    actual_value=str(actual_count),
                    verification_source=self._get_docs_source(claim, authoritative_data),
                    suggested_fix=None,
                    price_verification_path=None,
                    price_metadata=None,
                )
            if claimed_count == actual_count:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status="verified_accurate",
                    severity=None,
                    actual_value=str(actual_count),
                    verification_source=self._get_docs_source(claim, authoritative_data),
                    suggested_fix=None,
                    price_verification_path=None,
                    price_metadata=None,
                )
            return VerificationResult(
                claim_id=claim.claim_id,
                status="finding",
                severity="correctness",
                actual_value=str(actual_count),
                verification_source=self._get_docs_source(claim, authoritative_data),
                suggested_fix=f"Update region count from {claimed_count} to {actual_count}",
                price_verification_path=None,
                price_metadata=None,
            )

    # ------------------------------------------------------------------
    # Model ID verification
    # ------------------------------------------------------------------

    def _verify_model_id(
        self,
        claim: "Claim",
        authoritative_data: Any,
    ) -> VerificationResult:
        """Exact match against Bedrock lifecycle after alias resolution.

        When alias_found is False (human-readable name not in alias table),
        searches AWS docs live to find the canonical model ID rather than
        failing with a false correctness finding.
        """
        canonical_id = claim.claim_text
        lifecycle = getattr(authoritative_data, "bedrock_lifecycle", None)

        # If alias resolution failed, try live docs search to find the canonical ID
        if claim.alias_found is False:
            search_result = _docs_searcher.search_and_fetch(
                f"Amazon Bedrock model ID {claim.claim_text} canonical identifier"
            )
            if search_result:
                # Try to extract a canonical model ID from the search result
                import re as _re
                id_pattern = _re.compile(
                    r'\b(anthropic|amazon|meta|cohere|ai21|stability|mistral|us\.anthropic|us\.amazon|us\.meta)'
                    r'\.[a-z0-9\-\.]+(?:v\d+(?::\d+)?)?',
                    _re.IGNORECASE
                )
                matches = id_pattern.findall(search_result)
                if matches:
                    # Use the first plausible match
                    canonical_id = matches[0]
                    # Update the alias table in memory for this run
                    _docs_searcher._search_cache[f"alias:{claim.claim_text}"] = canonical_id
                else:
                    # Docs search found content but no canonical ID — unverified
                    return VerificationResult(
                        claim_id=claim.claim_id,
                        status="unverified",
                        severity=None,
                        actual_value=None,
                        verification_source=None,
                        suggested_fix=(
                            f"Model name '{claim.claim_text}' could not be resolved to a canonical "
                            f"Bedrock model ID. Add this mapping to alias_table.json."
                        ),
                        price_verification_path=None,
                        price_metadata=None,
                    )
            else:
                # No docs search result — unverified, not a correctness finding
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status="unverified",
                    severity=None,
                    actual_value=None,
                    verification_source=None,
                    suggested_fix=(
                        f"Model name '{claim.claim_text}' could not be resolved to a canonical "
                        f"Bedrock model ID. Add this mapping to alias_table.json."
                    ),
                    price_verification_path=None,
                    price_metadata=None,
                )

        if lifecycle is None or not lifecycle.models:
            return VerificationResult(
                claim_id=claim.claim_id,
                status="unverified",
                severity=None,
                actual_value=None,
                verification_source=None,
                suggested_fix=None,
                price_verification_path=None,
                price_metadata=None,
            )

        known_ids = {m.model_id for m in lifecycle.models}
        if canonical_id in known_ids:
            # Model is in the lifecycle list — check its status
            model_entry = next((m for m in lifecycle.models if m.model_id == canonical_id), None)
            if model_entry and getattr(model_entry, "status", "active") in ("eol", "legacy"):
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status="finding",
                    severity="correctness" if getattr(model_entry, "status", "") == "eol" else "outdated",
                    actual_value=f"Status: {model_entry.status}, EOL: {getattr(model_entry, 'eol_date', 'unknown')}",
                    verification_source=getattr(lifecycle, "fetched_at", None),
                    suggested_fix=(
                        f"Model {canonical_id!r} is {model_entry.status}. "
                        + (f"Use {model_entry.replacement!r} instead." if getattr(model_entry, "replacement", None) else "Check the Bedrock model catalog for an active replacement.")
                    ),
                    price_verification_path=None,
                    price_metadata=None,
                )
            # Model is in lifecycle list but active — verified accurate
            return VerificationResult(
                claim_id=claim.claim_id,
                status="verified_accurate",
                severity=None,
                actual_value=canonical_id,
                verification_source=getattr(lifecycle, "fetched_at", None),
                suggested_fix=None,
                price_verification_path=None,
                price_metadata=None,
            )

        # Model NOT in lifecycle list — this is GOOD: it means the model is active
        # (the lifecycle list only contains Legacy/EOL models, not active ones)
        return VerificationResult(
            claim_id=claim.claim_id,
            status="verified_accurate",
            severity=None,
            actual_value=canonical_id,
            verification_source=getattr(lifecycle, "fetched_at", None),
            suggested_fix=None,
            price_verification_path=None,
            price_metadata=None,
        )

    # ------------------------------------------------------------------
    # EOL date verification
    # ------------------------------------------------------------------

    def _verify_eol_date(
        self,
        claim: "Claim",
        authoritative_data: Any,
    ) -> VerificationResult:
        """Exact match on normalised YYYY-MM-DD."""
        # claim_text has already been normalised to YYYY-MM-DD by ClaimExtractor.
        claimed_date = claim.claim_text
        lifecycle = getattr(authoritative_data, "bedrock_lifecycle", None)
        if lifecycle is None or not lifecycle.models:
            return VerificationResult(
                claim_id=claim.claim_id,
                status="unverified",
                severity=None,
                actual_value=None,
                verification_source=None,
                suggested_fix=None,
                price_verification_path=None,
                price_metadata=None,
            )

        # Try to find a matching model by scanning line_context for a model name
        actual_eol = self._find_eol_date_from_lifecycle(claim, lifecycle)
        if actual_eol is None:
            return VerificationResult(
                claim_id=claim.claim_id,
                status="unverified",
                severity=None,
                actual_value=None,
                verification_source=getattr(lifecycle, "fetched_at", None),
                suggested_fix=None,
                price_verification_path=None,
                price_metadata=None,
            )

        if claimed_date == actual_eol:
            return VerificationResult(
                claim_id=claim.claim_id,
                status="verified_accurate",
                severity=None,
                actual_value=actual_eol,
                verification_source=getattr(lifecycle, "fetched_at", None),
                suggested_fix=None,
                price_verification_path=None,
                price_metadata=None,
            )
        return VerificationResult(
            claim_id=claim.claim_id,
            status="finding",
            severity="correctness",
            actual_value=actual_eol,
            verification_source=getattr(lifecycle, "fetched_at", None),
            suggested_fix=f"Update EOL date from {claimed_date} to {actual_eol}",
            price_verification_path=None,
            price_metadata=None,
        )

    # ------------------------------------------------------------------
    # AWS docs verification (quota_limit, api_name, other_factual)
    # ------------------------------------------------------------------

    def _verify_aws_docs(
        self,
        claim: "Claim",
        authoritative_data: Any,
        severity: str,
    ) -> VerificationResult:
        """Compare claim against AWS docs content; mismatch → given severity.

        Uses live AWS docs search + LLM judgment to determine if the claim
        is accurate, outdated, or incorrect based on current documentation.
        """
        # Try live search first
        live_content = _docs_searcher.search_and_fetch(claim.verification_query)
        docs_content = live_content or self._get_docs_content(claim, authoritative_data)
        source_ref = self._get_docs_source(claim, authoritative_data)

        if not docs_content:
            return VerificationResult(
                claim_id=claim.claim_id,
                status="unverified",
                severity=None,
                actual_value=None,
                verification_source=None,
                suggested_fix=None,
                price_verification_path=None,
                price_metadata=None,
            )

        # Use LLM to judge whether the claim is accurate based on docs content
        judgment = self._llm_verify_claim(claim, docs_content)
        if judgment["status"] == "verified_accurate":
            return VerificationResult(
                claim_id=claim.claim_id,
                status="verified_accurate",
                severity=None,
                actual_value=judgment.get("actual_value", claim.claim_text),
                verification_source=source_ref,
                suggested_fix=None,
                price_verification_path=None,
                price_metadata=None,
            )
        if judgment["status"] == "unverified":
            return VerificationResult(
                claim_id=claim.claim_id,
                status="unverified",
                severity=None,
                actual_value=None,
                verification_source=source_ref,
                suggested_fix=None,
                price_verification_path=None,
                price_metadata=None,
            )

        # Finding — use LLM-determined severity and suggested fix
        return VerificationResult(
            claim_id=claim.claim_id,
            status="finding",
            severity=judgment.get("severity", severity),
            actual_value=judgment.get("actual_value"),
            verification_source=source_ref,
            suggested_fix=judgment.get("suggested_fix", (
                f"Verify {claim.claim_type.replace('_', ' ')} claim against current AWS documentation: "
                f"{claim.verification_query}"
            )),
            price_verification_path=None,
            price_metadata=None,
        )

    # ------------------------------------------------------------------
    # Service name / feature availability verification
    # ------------------------------------------------------------------

    def _verify_service(
        self,
        claim: "Claim",
        authoritative_data: Any,
    ) -> VerificationResult:
        """Rename/preview→GA → policy_change; deprecation → outdated; removal → correctness.

        Uses live AWS docs search + LLM judgment to determine actual service status.
        """
        # Search with a targeted query about the specific service/feature
        search_query = f"{claim.claim_text} AWS documentation"
        live_content = _docs_searcher.search_and_fetch(search_query)
        # Also search the verification_query directly
        if not live_content:
            live_content = _docs_searcher.search_and_fetch(claim.verification_query)
        docs_content = live_content or self._get_docs_content(claim, authoritative_data)
        source_ref = self._get_docs_source(claim, authoritative_data)

        if not docs_content:
            return VerificationResult(
                claim_id=claim.claim_id,
                status="unverified",
                severity=None,
                actual_value=None,
                verification_source=source_ref,
                suggested_fix=None,
                price_verification_path=None,
                price_metadata=None,
            )

        # Use LLM to judge whether the claim is accurate based on docs content
        judgment = self._llm_verify_claim(claim, docs_content)
        if judgment["status"] == "verified_accurate":
            return VerificationResult(
                claim_id=claim.claim_id,
                status="verified_accurate",
                severity=None,
                actual_value=judgment.get("actual_value", claim.claim_text),
                verification_source=source_ref,
                suggested_fix=None,
                price_verification_path=None,
                price_metadata=None,
            )
        if judgment["status"] == "unverified":
            return VerificationResult(
                claim_id=claim.claim_id,
                status="unverified",
                severity=None,
                actual_value=None,
                verification_source=source_ref,
                suggested_fix=None,
                price_verification_path=None,
                price_metadata=None,
            )

        # Finding — use LLM-determined severity and suggested fix
        return VerificationResult(
            claim_id=claim.claim_id,
            status="finding",
            severity=judgment.get("severity", "policy_change"),
            actual_value=judgment.get("actual_value"),
            verification_source=source_ref,
            suggested_fix=judgment.get("suggested_fix", f"Verify: {claim.verification_query}"),
            price_verification_path=None,
            price_metadata=None,
        )

    def _llm_verify_claim(self, claim: "Claim", docs_content: str) -> dict:
        """Use an LLM to judge whether a claim is accurate based on docs content.

        Uses Nova 2 Lite (fast, cheap) for this binary judgment call.
        Returns a dict with keys:
        - status: "verified_accurate" | "finding" | "unverified"
        - severity: "correctness" | "outdated" | "policy_change" | "informational" (when finding)
        - actual_value: what the docs actually say (when finding)
        - suggested_fix: specific actionable fix (when finding)
        """
        try:
            prompt = f"""Verify this claim against the AWS documentation excerpt below.

CLAIM: {claim.claim_text}
QUESTION: {claim.verification_query}

DOCUMENTATION:
{docs_content[:2000]}

Respond with ONLY a JSON object (no explanation):
{{"status": "verified_accurate" | "finding" | "unverified", "severity": "correctness" | "outdated" | "policy_change" | null, "actual_value": "what docs say or null", "suggested_fix": "specific fix or null"}}

Use "unverified" if the docs don't contain enough information to answer."""

            # Use Nova 2 Lite for fast, cheap judgment calls
            model = BedrockModel(
                model_id="us.amazon.nova-2-lite-v1:0",
                region_name=self._region_name,
                max_tokens=300,
            )
            agent = Agent(model=model, system_prompt="You are a precise fact-checker. Respond only with valid JSON.", tools=[])
            response_text = str(agent(prompt))

            # Parse JSON response
            import re as _re
            fence_match = _re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
            if fence_match:
                response_text = fence_match.group(1).strip()
            bracket_idx = response_text.find("{")
            if bracket_idx != -1:
                response_text = response_text[bracket_idx:]
            end_idx = response_text.rfind("}")
            if end_idx != -1:
                response_text = response_text[:end_idx + 1]

            result = json.loads(response_text)
            if result.get("status") not in ("verified_accurate", "finding", "unverified"):
                result["status"] = "unverified"
            return result

        except Exception as exc:
            logger.debug("_llm_verify_claim failed for claim %s: %s", claim.claim_id[:16], exc)
            return {"status": "unverified", "severity": None, "actual_value": None, "suggested_fix": None}

    def _find_cache_entry(
        self,
        claim: "Claim",
        pricing_cache_entries: list[Any],
    ) -> Any | None:
        """Return the first PricingEntry matching claim.service and claim.metric, or None."""
        if not claim.service or not claim.metric:
            return None
        service_key = claim.service.lower()
        metric_key = claim.metric.lower()
        for entry in pricing_cache_entries:
            if entry.service.lower() == service_key and entry.metric.lower() == metric_key:
                return entry
        return None

    def _query_aws_pricing(
        self,
        claim: "Claim",
        authoritative_data: Any,
    ) -> float | None:
        """Look up current price from authoritative_data.aws_pricing for direct_query path."""
        if not claim.service or not claim.metric:
            return None
        cache_key = f"pricing:{claim.service}:{claim.metric}"
        cached = _cache_get(_pricing_cache, cache_key)
        if cached is not None:
            return cached
        value = (
            authoritative_data.aws_pricing
            .get(claim.service, {})
            .get(claim.metric)
        )
        if value is not None:
            _cache_set(_pricing_cache, cache_key, value, _PRICING_TTL_SECONDS)
        return value

    def _extract_price_from_claim(self, claim: "Claim") -> float | None:
        """Extract a numeric price value from claim_text (e.g. '$0.048' → 0.048)."""
        match = re.search(r"\$?([\d,]+\.?\d*)", claim.claim_text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass
        return None

    def _exceeds_tolerance_direct(
        self,
        cached_val: float,
        current_val: float,
        claim: "Claim",
    ) -> bool:
        """Apply tolerance rules for direct_query path (mirrors pricing_comparator logic)."""
        # Exempt: cached value < $0.001
        if cached_val < 0.001:
            return False
        # Absolute threshold: cached value > $10/unit → $2.00 absolute
        if cached_val > 10.0:
            return abs(current_val - cached_val) > 2.0
        # Determine tolerance by service type (AI vs infra)
        # Use 25% for AI-related services, 10% for infra
        ai_keywords = {"bedrock", "claude", "titan", "llama", "mistral", "gemini", "openai"}
        service_lower = (claim.service or "").lower()
        is_ai = any(kw in service_lower for kw in ai_keywords)
        tolerance_pct = 25.0 if is_ai else 10.0
        if cached_val != 0:
            diff_pct = abs(current_val - cached_val) / cached_val * 100
        else:
            diff_pct = 0.0 if current_val == 0 else 100.0
        return diff_pct > tolerance_pct

    def _build_price_metadata(
        self,
        claim: "Claim",
        cache_entry: Any | None,
    ) -> dict:
        """Build the price_metadata dict for a price claim."""
        meta: dict = {
            "currency": "USD",
            "unit": claim.metric,
            "pricing_model": "on-demand",
            "region": "us-east-1",
        }
        if cache_entry is not None:
            meta["price_list_date"] = getattr(cache_entry, "cached_last_updated", None)
        return meta

    def _get_authoritative_regions(
        self,
        claim: "Claim",
        authoritative_data: Any,
    ) -> list[str] | int | None:
        """Return authoritative region list or count from aws_docs, or None if unavailable."""
        cache_key = f"regions:{claim.verification_query}"
        cached = _cache_get(_docs_cache, cache_key)
        if cached is not None:
            return cached

        # Look for region data in aws_docs keyed by service-related topics
        docs = getattr(authoritative_data, "aws_docs", {})
        # Try to find a relevant docs entry
        for key, content in docs.items():
            if "region" in key.lower() or "availability" in key.lower():
                regions = self._extract_regions_from_content(content)
                if regions:
                    _cache_set(_docs_cache, cache_key, regions, _DOCS_TTL_SECONDS)
                    return regions
        return None

    def _extract_regions_from_content(self, content: str) -> list[str]:
        """Extract AWS region identifiers from docs content."""
        # Match patterns like us-east-1, eu-west-2, ap-southeast-1, etc.
        region_pattern = re.compile(
            r"\b(us|eu|ap|sa|ca|me|af|il|mx|ap)-(east|west|north|south|central|northeast|southeast|northwest|southwest)-\d\b"
        )
        return list(dict.fromkeys(m.group(0) for m in region_pattern.finditer(content)))

    def _parse_region_set(self, text: str) -> set[str]:
        """Parse a comma/space-separated list of region identifiers from text."""
        region_pattern = re.compile(
            r"\b(us|eu|ap|sa|ca|me|af|il|mx)-(east|west|north|south|central|northeast|southeast|northwest|southwest)-\d\b"
        )
        return {m.group(0) for m in region_pattern.finditer(text)}

    def _extract_integer(self, text: str) -> int | None:
        """Extract the first integer from text."""
        match = re.search(r"\b(\d+)\b", text)
        if match:
            return int(match.group(1))
        return None

    def _get_docs_content(self, claim: "Claim", authoritative_data: Any) -> str | None:
        """Return relevant docs content for a claim, with TTL caching."""
        cache_key = f"docs:{claim.verification_query}"
        cached = _cache_get(_docs_cache, cache_key)
        if cached is not None:
            return cached

        docs = getattr(authoritative_data, "aws_docs", {})
        if not docs:
            return None

        # Try to find the most relevant docs entry by matching claim_type or claim_text keywords
        claim_words = set(claim.claim_text.lower().split())
        best_key: str | None = None
        best_score = 0
        for key in docs:
            key_words = set(key.lower().split())
            score = len(claim_words & key_words)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is None:
            # Fall back to first available docs entry
            best_key = next(iter(docs), None)

        if best_key is None:
            return None

        content = docs[best_key]
        _cache_set(_docs_cache, cache_key, content, _DOCS_TTL_SECONDS)
        return content

    def _get_docs_source(self, claim: "Claim", authoritative_data: Any) -> str | None:
        """Return a source reference string for the docs used."""
        docs = getattr(authoritative_data, "aws_docs", {})
        if not docs:
            return None
        # Return the first matching key as a pseudo-URL reference
        claim_words = set(claim.claim_text.lower().split())
        best_key: str | None = None
        best_score = 0
        for key in docs:
            key_words = set(key.lower().split())
            score = len(claim_words & key_words)
            if score > best_score:
                best_score = score
                best_key = key
        return best_key or next(iter(docs), None)

    def _classify_service_severity(self, claim_text: str, docs_content: str) -> str:
        """Classify severity for service_name/feature_availability mismatches."""
        docs_lower = docs_content.lower()
        # Removal / closure signals → correctness
        removal_signals = [
            "removed", "discontinued", "no longer available", "end of life", "eol",
            "closed to new customers", "not accepting new customers", "closed for new",
            "no longer accepting", "service is closing", "will be closed",
        ]
        for signal in removal_signals:
            if signal in docs_lower:
                return "correctness"
        # Deprecation signals → outdated
        deprecation_signals = ["deprecated", "deprecation", "legacy", "will be removed"]
        for signal in deprecation_signals:
            if signal in docs_lower:
                return "outdated"
        # Rename / preview→GA signals → policy_change
        rename_signals = ["renamed", "now called", "generally available", "ga", "preview"]
        for signal in rename_signals:
            if signal in docs_lower:
                return "policy_change"
        # Default: policy_change (rename is the most common reason a service name doesn't match)
        return "policy_change"

    def _find_closest_model(self, model_id: str, models: list[Any]) -> str | None:
        """Find the closest matching model ID from the lifecycle list."""
        if not models:
            return None
        # Simple prefix match
        for m in models:
            if m.model_id.startswith(model_id[:20]):
                return m.model_id
        # Return first active model as fallback
        for m in models:
            if getattr(m, "status", "") == "active":
                return m.model_id
        return models[0].model_id if models else None

    def _find_eol_date_from_lifecycle(
        self,
        claim: "Claim",
        lifecycle: Any,
    ) -> str | None:
        """Find the EOL date for a model referenced in the claim's line_context."""
        # Try to match a model name from line_context against lifecycle entries
        context_lower = claim.line_context.lower()
        for m in lifecycle.models:
            if m.model_name.lower() in context_lower or m.model_id.lower() in context_lower:
                return getattr(m, "eol_date", None)
        return None


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------

# ARN pattern: arn:aws:<service>:<region>:<account-id>:<resource>
_ARN_RE = re.compile(
    r"arn:aws:[a-z0-9\-]+:[a-z0-9\-]*:\d{12}:[^\s\"']*",
    re.IGNORECASE,
)

# Account-context keyword followed by optional whitespace/punctuation then a 12-digit number.
# Matches: "account 123456789012", "Account ID: 123456789012", "AWS account: 123456789012"
_ACCOUNT_ID_RE = re.compile(
    r"(?:account(?:\s+id)?|aws\s+account)\s*[:\s]\s*(\d{12})\b",
    re.IGNORECASE,
)

# AWS access key ID pattern
_ACCESS_KEY_RE = re.compile(r"AKIA[0-9A-Z]{16}")


def _redact_string(value: str) -> str:
    """Apply all redaction patterns to a single string value."""
    # 1. Redact ARNs first (they contain 12-digit account IDs — avoid double-redaction)
    value = _ARN_RE.sub("[REDACTED_ARN]", value)
    # 2. Redact account IDs preceded by account-context keywords
    value = _ACCOUNT_ID_RE.sub(
        lambda m: m.group(0).replace(m.group(1), "[REDACTED_ACCOUNT_ID]"),
        value,
    )
    # 3. Redact AWS access key IDs
    value = _ACCESS_KEY_RE.sub("[REDACTED_KEY]", value)
    return value


def _redact_payload(payload: dict) -> dict:
    """Return a shallow copy of *payload* with all string values redacted.

    Recursively processes nested dicts (e.g. price_metadata, migration_context)
    so that sensitive values in nested structures are also redacted.
    """
    redacted: dict = {}
    for k, v in payload.items():
        if isinstance(v, str):
            redacted[k] = _redact_string(v)
        elif isinstance(v, dict):
            redacted[k] = _redact_payload(v)  # recurse
        else:
            redacted[k] = v
    return redacted


# ---------------------------------------------------------------------------
# Migration context detection
# ---------------------------------------------------------------------------

_MIGRATION_CONTEXT_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # "# GCP to AWS" heading
    (re.compile(r"#\s*GCP\s+to\s+AWS", re.IGNORECASE), "GCP", "AWS"),
    # "source: GCP" / "source_system: GCP"
    (re.compile(r"source(?:_system)?\s*:\s*GCP", re.IGNORECASE), "GCP", "AWS"),
    # "target: AWS" / "target_system: AWS"
    (re.compile(r"target(?:_system)?\s*:\s*AWS", re.IGNORECASE), "GCP", "AWS"),
    # "migrating from GCP to AWS"
    (re.compile(r"migrat\w*\s+from\s+GCP\s+to\s+AWS", re.IGNORECASE), "GCP", "AWS"),
]

_PATH_CONTEXT_PATTERNS: list[tuple[str, str, str]] = [
    ("gcp-to-aws/", "GCP", "AWS"),
    ("gcp_to_aws/", "GCP", "AWS"),
]


def _detect_migration_context(
    file_path: str,
    content: str,
) -> tuple[str, str] | None:
    """Detect (source_system, target_system) from file path and first 20 lines of content.

    Returns ``(source_system, target_system)`` when found, or ``None`` when the
    migration context cannot be determined.
    """
    # Check file path first
    for path_fragment, source, target in _PATH_CONTEXT_PATTERNS:
        if path_fragment in file_path:
            return source, target

    # Check first 20 lines of content
    first_lines = "\n".join(content.splitlines()[:20])
    for pattern, source, target in _MIGRATION_CONTEXT_PATTERNS:
        if pattern.search(first_lines):
            return source, target

    return None


# ---------------------------------------------------------------------------
# Severity → RiskLevel mapping
# ---------------------------------------------------------------------------

from migration_watchdog.models import Finding, RiskLevel  # noqa: E402 — import after module-level constants

_SEVERITY_TO_RISK: dict[str, RiskLevel] = {
    "correctness": RiskLevel.HIGH,
    "outdated": RiskLevel.MEDIUM,
    "policy_change": RiskLevel.LOW,
    "informational": RiskLevel.LOW,
}


# ---------------------------------------------------------------------------
# currency_dedupe_key
# ---------------------------------------------------------------------------


def currency_dedupe_key(finding: "Finding") -> tuple:  # type: ignore[name-defined]
    """Return the deduplication key for a currency_drift Finding.

    Key: ``(category, frozenset(affected_files), claim_id)``

    ``claim_id`` is read from ``finding.auditor_payload["claim_id"]``.
    Guard: when ``auditor_payload`` is missing or ``claim_id`` is absent,
    fall back to ``finding.finding_id`` so that payload-less findings are
    never collapsed into a single bucket.
    """
    payload = finding.auditor_payload
    if payload and "claim_id" in payload:
        claim_id = payload["claim_id"]
    else:
        claim_id = finding.finding_id
    return (finding.category, frozenset(finding.affected_files), claim_id)


# ---------------------------------------------------------------------------
# run_currency_audit
# ---------------------------------------------------------------------------


async def run_currency_audit(
    repo_content: "RepoContent",  # type: ignore[name-defined]
    authoritative_data: "AuthoritativeData",  # type: ignore[name-defined]
    run_id: str,
    file_filter: list[str] | None = None,
) -> list["Finding"]:  # type: ignore[name-defined]
    """Run the currency audit over all (or filtered) Reference_Files.

    For each Reference_File in ``repo_content.files`` (filtered by
    ``file_filter`` when provided):

    1. Detect migration context — halt file if not found, emit structured
       error to ``partial_source_failures``.
    2. Extract claims via ``ClaimExtractor``.
    3. Parse pricing cache entries from ``pricing-cache.md`` content.
    4. Verify each claim via ``ClaimVerifier``.
    5. For each ``VerificationResult`` with ``status="finding"``:
       - Apply redaction patterns to all string values in ``auditor_payload``.
       - Build ``auditor_payload`` dict.
       - Map severity to risk_level.
       - Call ``PayloadStore.store_payload()`` for each Finding.
       - Construct ``Finding`` with ``category="currency_drift"``,
         ``finding_schema_version="currency/1.0"``.
    6. Return list of Findings.

    ``PayloadStore.prepare_payload()`` ``ValueError`` is caught per-finding:
    the finding_id is logged, a structured entry is appended to
    ``partial_source_failures``, and that finding is skipped.
    """
    import uuid
    from datetime import datetime, timezone

    from migration_watchdog.payload_store import PayloadStore
    from migration_watchdog.pricing_comparator import parse_pricing_cache

    partial_source_failures: list[dict] = []
    findings: list[Finding] = []

    extractor = ClaimExtractor()
    verifier = ClaimVerifier()
    payload_store = PayloadStore()

    # Sync alias table from Bedrock lifecycle data fetched by SourceFetcher.
    # This keeps model-ID aliases fresh without an extra network call.
    alias_table = extractor._alias_table
    if hasattr(authoritative_data, "bedrock_lifecycle") and authoritative_data.bedrock_lifecycle.models:
        alias_table.sync_from_bedrock_lifecycle(authoritative_data.bedrock_lifecycle)

    # Parse pricing cache once (reused for all files).
    # Try the full repo path first (scheduled runs via RepoScanner),
    # then fall back to the bare filename (PR checkout mode).
    _PRICING_CACHE_PATH = "features/migration-to-aws/skills/gcp-to-aws/references/shared/pricing-cache.md"
    pricing_cache_content = (
        repo_content.files.get(_PRICING_CACHE_PATH)
        or repo_content.files.get("pricing-cache.md", "")
    )
    pricing_cache_entries = parse_pricing_cache(pricing_cache_content) if pricing_cache_content else []

    # Determine which files to audit
    files_to_audit: dict[str, str] = {}
    if file_filter is not None:
        for path in file_filter:
            if path in repo_content.files:
                files_to_audit[path] = repo_content.files[path]
    else:
        files_to_audit = dict(repo_content.files)

    scan_timestamp = datetime.now(timezone.utc).isoformat()

    for file_path, content in files_to_audit.items():
        # --- Step 1: Detect migration context ---
        context_result = _detect_migration_context(file_path, content)
        if context_result is None:
            logger.warning(
                "run_currency_audit: migration context not found in %s; skipping file",
                file_path,
            )
            partial_source_failures.append(
                {"type": "migration_context_failure", "file": file_path}
            )
            continue

        source_system, target_system = context_result
        migration_context = {
            "source_system": source_system,
            "target_system": target_system,
        }
        scope = f"{source_system} to {target_system}"

        # --- Step 2: Extract claims ---
        try:
            claims = extractor.extract(file_path, content)
        except Exception:
            logger.exception(
                "run_currency_audit: claim extraction failed for %s; skipping file",
                file_path,
            )
            partial_source_failures.append(
                {"type": "extraction_failure", "file": file_path}
            )
            continue

        if not claims:
            logger.info(
                "run_currency_audit: no claims extracted from %s", file_path
            )
            continue

        # --- Step 3: Pricing cache entries already parsed above ---

        # --- Step 4 & 5: Verify each claim and build Findings ---
        for claim in claims:
            try:
                result: VerificationResult = verifier.verify(
                    claim, authoritative_data, pricing_cache_entries
                )
            except Exception:
                logger.exception(
                    "run_currency_audit: verification failed for claim_id=%s in %s; skipping",
                    claim.claim_id[:16],
                    file_path,
                )
                continue

            if result.status != "finding":
                continue

            # Build auditor_payload
            raw_payload: dict = {
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "claim_type": claim.claim_type,
                "claim_subtype": claim.claim_subtype,
                "severity": result.severity,
                "actual_value": result.actual_value,
                "verification_source": result.verification_source,
                "suggested_fix": result.suggested_fix,
                "migration_context": migration_context,
                "scope": scope,
                "alias_found": claim.alias_found,
                "price_verification_path": result.price_verification_path,
                "price_metadata": result.price_metadata,
            }

            # Apply redaction to all string values
            redacted_payload = _redact_payload(raw_payload)

            # Map severity to risk_level
            severity = result.severity or "informational"
            risk_level = _SEVERITY_TO_RISK.get(severity, RiskLevel.LOW)

            # Generate a deterministic finding_id from claim_id + run_id
            finding_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"currency:{run_id}:{claim.claim_id}",
                )
            )

            # Build title and description
            title = (
                f"Currency drift: {claim.claim_type.replace('_', ' ')} in "
                f"{file_path.split('/')[-1]}"
            )
            description = (
                f"Claim '{claim.claim_text[:120]}' in {file_path} "
                f"does not match authoritative data. "
                f"Severity: {severity}. "
                + (f"Suggested fix: {result.suggested_fix}" if result.suggested_fix else "")
            )

            # Manage payload size via PayloadStore
            try:
                stored_payload = await payload_store.store_payload(
                    finding_id, redacted_payload
                )
            except ValueError:
                logger.exception(
                    "run_currency_audit: payload too large for finding_id=%s; skipping",
                    finding_id,
                )
                partial_source_failures.append(
                    {"type": "payload_too_large", "finding_id": finding_id}
                )
                continue

            source_urls: list[str] = []
            if result.verification_source:
                source_urls.append(result.verification_source)

            finding = Finding(
                finding_id=finding_id,
                run_id=run_id,
                risk_level=risk_level,
                category="currency_drift",
                title=title,
                description=description,
                affected_files=[file_path],
                source_urls=source_urls,
                scan_timestamp=scan_timestamp,
                auditor_payload=stored_payload,
                finding_schema_version="currency/1.0",
            )

            findings.append(finding)

    # Attach partial_source_failures to the run via a module-level list that
    # callers (main.py) can retrieve.  We store them on the function object so
    # the caller can inspect them without needing a separate return value.
    run_currency_audit.last_partial_source_failures = partial_source_failures  # type: ignore[attr-defined]

    logger.info(
        "run_currency_audit: completed run_id=%s — %d findings, %d partial failures",
        run_id,
        len(findings),
        len(partial_source_failures),
    )

    return findings
