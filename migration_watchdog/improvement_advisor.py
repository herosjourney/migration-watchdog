"""Improvement Advisor agent for the migration-watchdog pipeline.

Proactively reads the migration plugin's reference files and generates
high-impact, low-cost improvement suggestions. Runs as the 7th agent
in the scan pipeline during weekly scheduled scans only.

Uses a two-tier LLM approach:
- Claude Opus 4.7 via Bedrock for suggestion generation
- Nova 2 Lite for verification/filtering of borderline suggestions

Produces standard Finding objects that integrate with existing
deduplication, persistence, dashboard, and PR comment workflows.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone

from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from migration_watchdog.models import Finding, RepoContent, RiskLevel
from migration_watchdog.source_fetcher import AwsDocsSearcher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMPROVEMENT_CATEGORIES = (
    "ease_of_use",
    "reliability",
    "security_hardening",
    "completeness",
    "developer_experience",
)

# Full path prefixes matching RepoContent file keys (set by repo_scanner.py).
GCP_REFERENCES_PREFIX = "migrate/plugins/migration-to-aws/skills/gcp-to-aws/references/"
HEROKU_REFERENCES_PREFIX = "migrate/plugins/migration-to-aws/skills/heroku-to-aws/references/"
GCP_SKILL_PATH = "migrate/plugins/migration-to-aws/skills/gcp-to-aws/SKILL.md"
HEROKU_SKILL_PATH = "migrate/plugins/migration-to-aws/skills/heroku-to-aws/SKILL.md"

# Output and chunking limits.
MAX_FINDINGS = 50
CHUNK_THRESHOLD = 12000
CHUNK_SIZE = 8000

# High-confidence misroute title prefixes for the overlap safety net.
_OVERLAP_TITLE_PREFIXES = (
    "Update price for",
    "Update pricing for",
    "Deprecate model",
    "Remove deprecated model",
    "Add CLI command for",
    "Automate manual step",
    "Fix security vulnerability in",
    "Fix insecure pattern in",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

IMPROVEMENT_SYSTEM_PROMPT = """\
You are the Migration Plugin Watchdog Improvement Advisor — a forward-looking quality advisor \
for the GCP-to-AWS and Heroku-to-AWS migration plugin. Unlike the reactive auditors that detect \
drift from authoritative sources, your role is to proactively identify high-impact, low-cost \
improvements to the plugin's reference files that no other agent surfaces.

You will be given the content of migration reference files (phases, design-refs, shared configs, \
and SKILL.md files). For each file or batch of files, generate actionable improvement suggestions.

## Improvement Categories

Classify every suggestion into exactly ONE of these categories:

1. **ease_of_use** — Simplify user-facing instructions, reduce cognitive load. Examples: \
clearer step numbering, removing jargon, adding TL;DR summaries, improving navigation cues.

2. **reliability** — Add error handling, retry logic, validation checks. Examples: adding \
pre-flight checks before destructive steps, suggesting rollback instructions, noting timeout \
values for long-running operations.

3. **security_hardening** — Add defense-in-depth measures (NOT specific vulnerabilities — \
that is the Security_Auditor's domain). Examples: recommending least-privilege IAM patterns, \
suggesting encryption-at-rest defaults, adding network isolation notes.

4. **completeness** — Missing phases, design-refs, migration scenarios. Examples: a phase \
directory that exists in gcp-to-aws but is absent in heroku-to-aws, missing cross-references \
between related design-refs, gaps in multi-region guidance.

5. **developer_experience** — Better examples, clearer variable names, documentation. Examples: \
adding copy-paste-ready code snippets, using consistent naming conventions, adding inline \
comments explaining non-obvious values.

## Overlap Exclusion (PRIMARY MECHANISM)

You MUST NOT produce suggestions in the following domains — they are owned by other agents:

- **Currency_Auditor owns**: Pricing drift, stale factual claims about service costs, \
deprecated model IDs, end-of-life dates, outdated version numbers. Do NOT suggest updating \
prices or flagging that a model ID is deprecated.

- **Automation_Auditor owns**: Manual steps that could be automated via CLI/SDK commands. \
Do NOT suggest "this manual step could be a script" — the Automation Auditor handles that.

- **Security_Auditor owns**: Specific security vulnerabilities in generated code patterns \
(e.g., "this Terraform snippet uses overly permissive IAM policy"). Do NOT flag specific \
insecure code. You MAY suggest adding general defense-in-depth guidance (e.g., "add a note \
about enabling VPC flow logs") as that is security_hardening, not vulnerability detection.

- **Scenario_Simulation_Agent owns**: Persona routing gaps and workload-specific design-ref \
coverage (e.g., "ML workloads don't have a design-ref for Heroku"). Do NOT produce \
completeness findings about missing design-ref coverage for specific workload types or \
personas that the Scenario Agent tracks.

## Scoring Guidelines

Assign integer scores to every suggestion:

- **impact_score** (3–5 ONLY): How much value this improvement delivers to startups.
  - 3 = Meaningful improvement, noticeable quality lift
  - 4 = Significant improvement, addresses a real gap
  - 5 = Critical improvement, addresses a major quality or completeness gap

- **effort_score** (1–3 ONLY): How much work to implement.
  - 1 = Trivial (a few lines changed, copy-paste fix)
  - 2 = Small (a paragraph or section rewrite, new subsection)
  - 3 = Moderate (new file, multi-section restructuring)

Do NOT emit suggestions with impact < 3 or effort > 3. If a suggestion falls outside \
these bounds, discard it.

## Tool Usage: search_aws_docs

Use the `search_aws_docs` tool ONLY when suggesting new AWS service or feature coverage — \
for example, when proposing that the plugin should add guidance about a specific AWS service \
the plugin does not currently mention. Verify the service exists and is generally available \
before suggesting it.

Do NOT use `search_aws_docs` for internal DX/consistency improvements, ease-of-use rewrites, \
reliability improvements, or developer_experience suggestions that don't reference a specific \
new AWS service. These improvements are grounded in the file content itself and need no \
external verification.

## Output Format: Description Structure

For each suggestion, produce a description with exactly 3 sections separated by blank lines:

<section 1: Suggestion>
A clear, specific statement of what should change. Name the file, section, or pattern affected.

<section 2: Rationale>
Why this matters — what problem it solves for startup engineers using the plugin.

<section 3: Before/After Example>
Show a concrete before-and-after snippet demonstrating the improvement. Use markdown code \
fences or quoted text to illustrate the current content and the improved version.

## Source URLs

- For suggestions about new AWS service/feature coverage: provide the docs.aws.amazon.com URL \
from search_aws_docs results.
- For non-service suggestions (DX, consistency, reliability, ease_of_use): use the plugin \
repository URL (https://github.com/aws-samples/sample-agent-skills-for-aws-migration) or \
any relevant reference URL that supports the suggestion rationale.

## General Instructions

- Be specific and actionable. Vague suggestions like "improve documentation" are not useful.
- Focus on the reference file content you are given — do not speculate about files you haven't seen.
- Prefer fewer, higher-quality suggestions over many low-value ones.
- Use the `create_improvement_finding` tool for each valid suggestion that passes the \
impact/effort filter.
"""

# ---------------------------------------------------------------------------
# Module-level docs searcher for improvement advisor tools
# ---------------------------------------------------------------------------

_improvement_docs_searcher = AwsDocsSearcher()

# Module-level list that the create_improvement_finding tool appends to during a run.
# Reset at the start of each run_improvement_advisor() call.
_current_findings: list[Finding] = []
_current_run_id: str = ""


# ---------------------------------------------------------------------------
# Strands @tool functions
# ---------------------------------------------------------------------------


@tool
def search_aws_docs(query: str) -> str:
    """Search AWS documentation for current information about a service or feature.

    Use this tool to verify suggestions that propose adding coverage for a specific
    AWS service or feature. Confirms that the service appears in current documentation
    with available status.

    Args:
        query: A specific search query, e.g. "Amazon Bedrock AgentCore features",
               "AWS App Runner container deployment", "Amazon ECS Fargate pricing"

    Returns:
        JSON string with page titles, URLs, and content excerpts from AWS docs.
        Returns JSON with empty results array and error field on failure.
    """
    max_attempts = 3
    last_error: str = ""

    for attempt in range(1, max_attempts + 1):
        try:
            results = _improvement_docs_searcher.search(query, limit=3)

            if not results:
                return json.dumps({"query": query, "results": [], "error": None})

            # Build structured output with titles, URLs, and content excerpts
            output_results: list[dict] = []
            for r in results:
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("snippet", "")

                # Fetch page content for the first result to get a richer excerpt
                excerpt = snippet
                if url and not excerpt:
                    page_content = _improvement_docs_searcher.fetch_page(url)
                    excerpt = page_content[:500] if page_content else ""

                output_results.append({
                    "title": title,
                    "url": url,
                    "excerpt": excerpt,
                })

            return json.dumps({"query": query, "results": output_results, "error": None})

        except Exception as exc:
            last_error = str(exc)
            logger.debug(
                "search_aws_docs attempt %d/%d failed for %r: %s",
                attempt, max_attempts, query, exc,
            )
            if attempt < max_attempts:
                time.sleep(1.0 * attempt)  # Simple linear backoff

    # All retries exhausted
    logger.warning(
        "search_aws_docs failed after %d attempts for %r: %s",
        max_attempts, query, last_error,
    )
    return json.dumps({
        "query": query,
        "results": [],
        "error": f"Search failed after {max_attempts} attempts: {last_error}",
    })


@tool
def create_improvement_finding(
    title: str,
    description: str,
    affected_file: str,
    improvement_category: str,
    impact_score: int,
    effort_score: int,
    source_urls: str,
) -> str:
    """Create an improvement Finding from an identified suggestion.

    Validates all inputs against the improvement scoring thresholds and
    category constraints. Appends valid findings to the module-level
    accumulator list.

    Args:
        title: Short descriptive title (≤120 chars, non-empty).
        description: Detailed description with 3 sections separated by blank lines:
                     suggestion, rationale, and before/after example.
        affected_file: The file path affected by this improvement (relative to
                       the plugin root, e.g., "skills/gcp-to-aws/references/phases/...").
                       Will be prefixed with "migrate/plugins/migration-to-aws/" if needed.
        improvement_category: One of: ease_of_use, reliability, security_hardening,
                              completeness, developer_experience.
        impact_score: Integer 3-5. How much value this improvement delivers.
        effort_score: Integer 1-3. How much work to implement.
        source_urls: JSON array string of 1-5 documentation URLs supporting
                     the suggestion.

    Returns:
        JSON string indicating success or failure with reason.
    """
    global _current_findings, _current_run_id

    # --- Validate title ---
    if not title or not title.strip():
        return json.dumps({
            "created": False,
            "error": "Finding rejected: title must be non-empty.",
        })
    if len(title) > 120:
        return json.dumps({
            "created": False,
            "error": f"Finding rejected: title exceeds 120 characters (got {len(title)}).",
        })

    # --- Validate improvement_category ---
    if improvement_category not in IMPROVEMENT_CATEGORIES:
        return json.dumps({
            "created": False,
            "error": (
                f"Finding rejected: improvement_category '{improvement_category}' is invalid. "
                f"Must be one of: {', '.join(IMPROVEMENT_CATEGORIES)}."
            ),
        })

    # --- Validate impact_score ---
    try:
        impact_score = int(impact_score)
    except (TypeError, ValueError):
        return json.dumps({
            "created": False,
            "error": "Finding rejected: impact_score must be an integer.",
        })
    if impact_score < 3 or impact_score > 5:
        return json.dumps({
            "created": False,
            "error": (
                f"Finding rejected: impact_score must be in [3, 5] (got {impact_score}). "
                "Suggestions with impact < 3 should be discarded."
            ),
        })

    # --- Validate effort_score ---
    try:
        effort_score = int(effort_score)
    except (TypeError, ValueError):
        return json.dumps({
            "created": False,
            "error": "Finding rejected: effort_score must be an integer.",
        })
    if effort_score < 1 or effort_score > 3:
        return json.dumps({
            "created": False,
            "error": (
                f"Finding rejected: effort_score must be in [1, 3] (got {effort_score}). "
                "Suggestions with effort > 3 should be discarded."
            ),
        })

    # --- Validate source_urls ---
    try:
        urls_list: list[str] = json.loads(source_urls)
    except (json.JSONDecodeError, TypeError):
        # Try treating as a single URL string
        if source_urls and isinstance(source_urls, str) and not source_urls.startswith("["):
            urls_list = [source_urls]
        else:
            return json.dumps({
                "created": False,
                "error": "Finding rejected: source_urls must be a valid JSON array of URLs.",
            })

    if not isinstance(urls_list, list):
        return json.dumps({
            "created": False,
            "error": "Finding rejected: source_urls must be a JSON array.",
        })

    if len(urls_list) < 1 or len(urls_list) > 5:
        return json.dumps({
            "created": False,
            "error": (
                f"Finding rejected: source_urls must have 1-5 items (got {len(urls_list)})."
            ),
        })

    # --- Build full file path ---
    full_prefix = "migrate/plugins/migration-to-aws/"
    if affected_file.startswith(full_prefix):
        full_path = affected_file
    else:
        # Prepend the full prefix
        full_path = full_prefix + affected_file.lstrip("/")

    # --- Compute finding_id ---
    finding_id = _compute_finding_id("improvement", full_path, title)
    if not finding_id:
        return json.dumps({
            "created": False,
            "error": "Finding rejected: empty file path or title produces no finding_id.",
        })

    # --- Create Finding ---
    scan_timestamp = datetime.now(timezone.utc).isoformat()

    finding = Finding(
        finding_id=finding_id,
        run_id=_current_run_id,
        risk_level=RiskLevel.MEDIUM,
        category="improvement",
        title=title,
        description=description,
        affected_files=[full_path],
        proposed_changes={},
        source_urls=urls_list,
        scan_timestamp=scan_timestamp,
        status="pending",
        auditor_payload={
            "improvement_category": improvement_category,
            "impact_score": impact_score,
            "effort_score": effort_score,
        },
        finding_schema_version="improvement/1.0",
    )

    _current_findings.append(finding)

    return json.dumps({
        "created": True,
        "finding_id": finding_id,
        "title": title,
        "improvement_category": improvement_category,
        "impact_score": impact_score,
        "effort_score": effort_score,
        "affected_file": full_path,
    })


# ---------------------------------------------------------------------------
# Core Orchestration
# ---------------------------------------------------------------------------

# Budget constant: stop processing batches if elapsed time exceeds this fraction of 300s.
_TIME_BUDGET_SECONDS = 240.0  # 80% of 300s timeout


def _classify_file_group(path: str) -> str:
    """Classify a reference file path into a batch group.

    Groups:
    - "phases" — files under .../references/phases/
    - "design-refs" — files under .../references/design-refs/
    - "shared" — everything else under references/ (shared/, clustering/, etc.)
    - "skill" — SKILL.md files (processed separately as context, not batched)
    """
    skill_paths = {GCP_SKILL_PATH, HEROKU_SKILL_PATH}
    if path in skill_paths:
        return "skill"

    # Check for phases
    if "/references/phases/" in path:
        return "phases"
    # Check for design-refs
    if "/references/design-refs/" in path:
        return "design-refs"
    # Everything else under references/
    return "shared"


def _run_advisor_impl(repo_content: RepoContent, run_id: str) -> list[Finding]:
    """Core implementation of the Improvement Advisor agent run.

    Orchestrates batched LLM invocations over plugin reference files,
    runs completeness gap detection, applies the overlap safety net,
    and returns sorted/truncated findings.

    This function is wrapped by run_improvement_advisor() with a timeout.
    """
    global _current_findings, _current_run_id

    # --- Step 1: Filter reference files ---
    ref_files = _filter_reference_files(repo_content)
    if not ref_files:
        logger.info("No reference files found in RepoContent; returning empty.")
        return []

    # --- Step 2: Reset module-level accumulators ---
    _current_findings = []
    _current_run_id = run_id

    # --- Step 3: Extract SKILL.md content for context ---
    skill_paths = {GCP_SKILL_PATH, HEROKU_SKILL_PATH}
    skill_context_parts: list[str] = []
    for spath in sorted(skill_paths):
        if spath in ref_files:
            skill_context_parts.append(
                f"=== SKILL.md: {spath} ===\n{ref_files[spath]}\n"
            )

    skill_context = "\n".join(skill_context_parts) if skill_context_parts else ""

    # --- Step 4: Batch files by subdirectory group ---
    # Exclude SKILL.md files from batches (they are used as context only)
    batches: dict[str, list[tuple[str, str]]] = {
        "phases": [],
        "design-refs": [],
        "shared": [],
    }

    for path, content in ref_files.items():
        group = _classify_file_group(path)
        if group == "skill":
            continue  # Already extracted as context
        if not content or not content.strip():
            logger.debug("Skipping empty file: %s", path)
            continue
        batches[group].append((path, content))

    # --- Step 5: Build batch messages with chunking ---
    batch_messages: list[str] = []

    for group_name in ("phases", "design-refs", "shared"):
        group_files = batches[group_name]
        if not group_files:
            continue

        # Concatenate all file contents for this group, chunking large files
        parts: list[str] = []
        for path, content in sorted(group_files, key=lambda x: x[0]):
            chunks = _chunk_file(content)
            for chunk in chunks:
                parts.append(f"=== File: {path} ===\n{chunk}\n")

        # If a single group produces a very large message, split into sub-batches
        # to keep each LLM call manageable (~60KB target per call)
        current_batch: list[str] = []
        current_size = 0
        max_batch_size = 60000  # ~60KB per batch

        for part in parts:
            if current_size + len(part) > max_batch_size and current_batch:
                batch_messages.append("\n".join(current_batch))
                current_batch = []
                current_size = 0
            current_batch.append(part)
            current_size += len(part)

        if current_batch:
            batch_messages.append("\n".join(current_batch))

    if not batch_messages:
        logger.info("No non-empty reference files to process after filtering.")
        # Still run completeness gaps
        completeness_findings = _detect_completeness_gaps(repo_content)
        for f in completeness_findings:
            f.run_id = run_id
        return completeness_findings

    # --- Step 6: Create Strands Agent ---
    model = BedrockModel(
        model_id="us.anthropic.claude-opus-4-7",
        region_name="us-east-1",
        max_tokens=16000,
    )
    agent = Agent(
        model=model,
        system_prompt=IMPROVEMENT_SYSTEM_PROMPT,
        tools=[search_aws_docs, create_improvement_finding],
    )

    # --- Step 7: Invoke agent for each batch with time budget ---
    start_time = time.monotonic()

    for batch_idx, batch_message in enumerate(batch_messages):
        elapsed = time.monotonic() - start_time
        if elapsed >= _TIME_BUDGET_SECONDS:
            logger.warning(
                "Time budget exceeded (%.1fs / %.1fs) after %d/%d batches; "
                "emitting partial results.",
                elapsed, _TIME_BUDGET_SECONDS,
                batch_idx, len(batch_messages),
            )
            break

        # Build user message with SKILL.md context prepended
        user_message_parts: list[str] = []
        if skill_context:
            user_message_parts.append(
                "## Context: SKILL.md files\n\n" + skill_context
            )
        user_message_parts.append(
            f"## Reference files to analyze (batch {batch_idx + 1}/{len(batch_messages)})\n\n"
            "Analyze the following reference files and use `create_improvement_finding` "
            "for each valid improvement suggestion you identify.\n\n"
            + batch_message
        )
        user_message = "\n\n".join(user_message_parts)

        try:
            agent(user_message)
            logger.info(
                "Batch %d/%d completed: %d findings so far (%.1fs elapsed)",
                batch_idx + 1, len(batch_messages),
                len(_current_findings),
                time.monotonic() - start_time,
            )
        except Exception:
            logger.exception(
                "Batch %d/%d failed; continuing with next batch", batch_idx + 1, len(batch_messages)
            )

    # --- Step 8: Run completeness gap detection ---
    completeness_findings = _detect_completeness_gaps(repo_content)
    for f in completeness_findings:
        f.run_id = run_id

    # --- Step 9: Merge all findings ---
    all_findings = list(_current_findings) + completeness_findings

    # --- Step 10: Apply overlap safety net ---
    filtered_findings = _apply_overlap_safety_net(all_findings)

    # --- Step 11: Sort by impact_score desc, effort_score asc ---
    filtered_findings.sort(
        key=lambda f: (
            -(f.auditor_payload or {}).get("impact_score", 0),
            (f.auditor_payload or {}).get("effort_score", 99),
        )
    )

    # --- Step 12: Truncate to MAX_FINDINGS ---
    final_findings = filtered_findings[:MAX_FINDINGS]

    logger.info(
        "Improvement advisor run %s complete: %d findings "
        "(from %d LLM-generated + %d completeness gaps, %d filtered by safety net)",
        run_id,
        len(final_findings),
        len(_current_findings),
        len(completeness_findings),
        len(all_findings) - len(filtered_findings),
    )

    return final_findings


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _filter_reference_files(repo_content: RepoContent) -> dict[str, str]:
    """Extract migration plugin reference files from RepoContent.

    Returns a dict of path->content for files whose path starts with
    GCP_REFERENCES_PREFIX or HEROKU_REFERENCES_PREFIX, plus the SKILL.md
    files. Only includes .md files (consistent with RepoScanner filter).
    """
    result: dict[str, str] = {}
    skill_paths = {GCP_SKILL_PATH, HEROKU_SKILL_PATH}

    for path, content in repo_content.files.items():
        if not path.endswith(".md"):
            continue
        if (
            path.startswith(GCP_REFERENCES_PREFIX)
            or path.startswith(HEROKU_REFERENCES_PREFIX)
            or path in skill_paths
        ):
            result[path] = content

    return result


def _chunk_file(content: str, max_chars: int = 8000) -> list[str]:
    """Split a file's content into processable chunks.

    Files ≤ CHUNK_THRESHOLD (12000) chars are returned as a single-element list.
    Larger files are split at paragraph boundaries (double newline) into chunks
    of at most max_chars characters. If no paragraph boundary is found within
    max_chars, the split occurs at max_chars.

    Each chunk in a multi-chunk result is prepended with a context header
    like "--- Chunk 2/4 ---" (the first chunk in a single-element result
    does NOT get a header).
    """
    if len(content) <= CHUNK_THRESHOLD:
        return [content]

    raw_chunks: list[str] = []
    remaining = content

    while remaining:
        if len(remaining) <= max_chars:
            raw_chunks.append(remaining)
            break

        # Look for a paragraph boundary (double newline) within the window.
        split_pos = remaining.rfind("\n\n", 0, max_chars)
        if split_pos == -1:
            # No paragraph boundary found; hard-split at max_chars.
            split_pos = max_chars
        else:
            # Include the double newline in the current chunk.
            split_pos += 2

        raw_chunks.append(remaining[:split_pos])
        remaining = remaining[split_pos:]

    # Prepend context headers to each chunk.
    total = len(raw_chunks)
    chunks: list[str] = []
    for i, chunk in enumerate(raw_chunks, start=1):
        header = f"--- Chunk {i}/{total} ---\n"
        chunks.append(header + chunk)

    return chunks


def _compute_finding_id(category: str, file_path: str, title: str) -> str:
    """Compute a deterministic finding ID from category, file path, and title.

    Returns the first 32 hex characters of SHA-256(f"{category}|{file_path}|{title}").
    Returns empty string if file_path or title is empty (caller should skip the finding).
    """
    if not file_path or not title:
        return ""

    hash_input = f"{category}|{file_path}|{title}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:32]


def improvement_dedupe_key(finding: Finding) -> tuple:
    """Deduplication key for improvement findings.

    Returns:
        Tuple of (category, improvement_category, affected_file, title_hash)
        where title_hash is SHA-256 of the title truncated to 16 hex chars.
    """
    improvement_category = (finding.auditor_payload or {}).get("improvement_category", "")
    affected_file = finding.affected_files[0] if finding.affected_files else ""
    title_hash = hashlib.sha256(finding.title.encode()).hexdigest()[:16]

    return (finding.category, improvement_category, affected_file, title_hash)


def _apply_overlap_safety_net(findings: list[Finding]) -> list[Finding]:
    """Filter out findings that are high-confidence misroutes to other agents.

    Uses narrow title-prefix matching only. Does NOT filter based on keywords
    appearing in the description body. This is a safety net — the system prompt
    exclusion is the primary overlap prevention mechanism.
    """
    result: list[Finding] = []
    for finding in findings:
        title_lower = finding.title.lower()
        if any(title_lower.startswith(prefix.lower()) for prefix in _OVERLAP_TITLE_PREFIXES):
            logger.debug("Overlap safety net filtered: %s", finding.title)
            continue
        result.append(finding)
    return result


# OS metadata filenames to exclude when inferring directory structure.
_OS_METADATA_FILES = {".DS_Store", "Thumbs.db", "._.DS_Store", "desktop.ini"}

# Phase and design-ref path prefixes for each skill.
_GCP_PHASES_PREFIX = GCP_REFERENCES_PREFIX + "phases/"
_HEROKU_PHASES_PREFIX = HEROKU_REFERENCES_PREFIX + "phases/"
_GCP_DESIGN_REFS_PREFIX = GCP_REFERENCES_PREFIX + "design-refs/"
_HEROKU_DESIGN_REFS_PREFIX = HEROKU_REFERENCES_PREFIX + "design-refs/"


def _extract_subdirs(file_paths: set[str], prefix: str) -> set[str]:
    """Extract unique immediate subdirectory names from paths under a prefix.

    Given paths like:
        prefix + "discover/discover.md"
        prefix + "design/design-billing.md"
    Returns {"discover", "design"}.

    Excludes OS metadata filenames.
    """
    subdirs: set[str] = set()
    for path in file_paths:
        if not path.startswith(prefix):
            continue
        remainder = path[len(prefix):]
        # remainder looks like "discover/discover.md" or "design/sub/file.md"
        parts = remainder.split("/")
        if len(parts) >= 2:
            # First part is the subdirectory name
            subdir = parts[0]
            if subdir not in _OS_METADATA_FILES and subdir:
                subdirs.add(subdir)
    return subdirs


def _extract_filenames(file_paths: set[str], prefix: str) -> set[str]:
    """Extract unique filenames (without path) from paths under a prefix.

    Given paths like:
        prefix + "database.md"
        prefix + "design-ref-agentic-to-agentcore.md"
    Returns {"database.md", "design-ref-agentic-to-agentcore.md"}.

    Excludes OS metadata filenames.
    """
    filenames: set[str] = set()
    for path in file_paths:
        if not path.startswith(prefix):
            continue
        remainder = path[len(prefix):]
        # For design-refs, files are typically directly under the prefix
        # e.g., "database.md" or in subdirs "sub/file.md"
        filename = remainder.split("/")[-1]
        if filename not in _OS_METADATA_FILES and filename:
            filenames.add(filename)
    return filenames


def _detect_completeness_gaps(repo_content: RepoContent) -> list[Finding]:
    """Detect structural completeness gaps between GCP and Heroku skills.

    Compares the GCP-to-AWS phases directory structure against Heroku-to-AWS
    to identify missing phases and design-refs. Does NOT detect persona-specific
    coverage gaps (those are owned by the Scenario Agent).

    Returns:
        List of Finding objects with improvement_category="completeness".
    """
    findings: list[Finding] = []
    all_paths = set(repo_content.files.keys())
    scan_timestamp = datetime.now(timezone.utc).isoformat()

    # --- Phase gap detection ---
    gcp_phases = _extract_subdirs(all_paths, _GCP_PHASES_PREFIX)
    heroku_phases = _extract_subdirs(all_paths, _HEROKU_PHASES_PREFIX)

    # Check if heroku has any phase subdirs at all
    heroku_has_any_phases = len(heroku_phases) > 0

    if gcp_phases and not heroku_has_any_phases:
        # Entire phases directory is empty/missing for heroku — impact_score=5
        missing_phases_sorted = sorted(gcp_phases)
        title = "Heroku-to-AWS skill missing all migration phases"
        affected_file = _HEROKU_PHASES_PREFIX.rstrip("/")
        finding_id = _compute_finding_id("completeness", affected_file, title)
        if finding_id:
            description = (
                f"The heroku-to-aws skill has no phase subdirectories under "
                f"references/phases/, while gcp-to-aws has {len(gcp_phases)} phases: "
                f"{', '.join(missing_phases_sorted)}."
                f"\n\n"
                f"Migration phases provide structured guidance for each step of the "
                f"migration process. Without them, Heroku-to-AWS migrations lack "
                f"the same level of structured support as GCP-to-AWS."
                f"\n\n"
                f"Before: references/phases/ is empty for heroku-to-aws\n"
                f"After: Add phase subdirectories matching gcp-to-aws: "
                f"{', '.join(missing_phases_sorted)}"
            )
            findings.append(Finding(
                finding_id=finding_id,
                run_id="",  # Will be set by caller
                risk_level=RiskLevel.MEDIUM,
                category="improvement",
                title=title,
                description=description,
                affected_files=[affected_file],
                proposed_changes={},
                source_urls=[
                    "https://github.com/aws-samples/startups/tree/main/migrate/plugins/migration-to-aws"
                ],
                scan_timestamp=scan_timestamp,
                status="pending",
                auditor_payload={
                    "improvement_category": "completeness",
                    "impact_score": 5,
                    "effort_score": 2,
                },
                finding_schema_version="improvement/1.0",
            ))
    elif gcp_phases and heroku_has_any_phases:
        # Heroku has some phases but may be missing individual ones — impact_score=4
        missing_phases = gcp_phases - heroku_phases
        for phase in sorted(missing_phases):
            title = f"Heroku-to-AWS skill missing '{phase}' phase"
            affected_file = _HEROKU_PHASES_PREFIX + phase
            finding_id = _compute_finding_id("completeness", affected_file, title)
            if not finding_id:
                continue
            description = (
                f"The heroku-to-aws skill is missing the '{phase}' phase "
                f"subdirectory that exists in gcp-to-aws/references/phases/{phase}/."
                f"\n\n"
                f"This phase provides important migration guidance in the GCP skill "
                f"and should have an equivalent for Heroku-to-AWS migrations."
                f"\n\n"
                f"Before: references/phases/{phase}/ does not exist in heroku-to-aws\n"
                f"After: Add references/phases/{phase}/ with Heroku-specific guidance"
            )
            findings.append(Finding(
                finding_id=finding_id,
                run_id="",
                risk_level=RiskLevel.MEDIUM,
                category="improvement",
                title=title,
                description=description,
                affected_files=[affected_file],
                proposed_changes={},
                source_urls=[
                    "https://github.com/aws-samples/startups/tree/main/migrate/plugins/migration-to-aws"
                ],
                scan_timestamp=scan_timestamp,
                status="pending",
                auditor_payload={
                    "improvement_category": "completeness",
                    "impact_score": 4,
                    "effort_score": 2,
                },
                finding_schema_version="improvement/1.0",
            ))

    # --- Design-refs gap detection ---
    gcp_design_ref_files = _extract_filenames(all_paths, _GCP_DESIGN_REFS_PREFIX)
    heroku_design_ref_files = _extract_filenames(all_paths, _HEROKU_DESIGN_REFS_PREFIX)

    missing_design_refs = gcp_design_ref_files - heroku_design_ref_files
    for filename in sorted(missing_design_refs):
        # Derive topic name from filename (strip .md, replace hyphens with spaces)
        topic = filename.replace(".md", "").replace("-", " ").replace("_", " ")
        title = f"Heroku-to-AWS missing design-ref for '{topic}'"
        # Truncate title to 120 chars if needed
        if len(title) > 120:
            title = title[:117] + "..."
        affected_file = _HEROKU_DESIGN_REFS_PREFIX + filename
        finding_id = _compute_finding_id("completeness", affected_file, title)
        if not finding_id:
            continue
        gcp_ref_path = _GCP_DESIGN_REFS_PREFIX + filename
        description = (
            f"The gcp-to-aws skill has a design-ref file '{filename}' covering "
            f"the '{topic}' migration topic, but heroku-to-aws has no equivalent."
            f"\n\n"
            f"Design-refs provide detailed guidance for specific migration domains. "
            f"Having parity ensures Heroku-to-AWS users get the same depth of guidance. "
            f"Evidence: {gcp_ref_path}"
            f"\n\n"
            f"Before: No design-ref for '{topic}' in heroku-to-aws\n"
            f"After: Add {_HEROKU_DESIGN_REFS_PREFIX}{filename} covering "
            f"Heroku-specific aspects of {topic}"
        )
        findings.append(Finding(
            finding_id=finding_id,
            run_id="",
            risk_level=RiskLevel.MEDIUM,
            category="improvement",
            title=title,
            description=description,
            affected_files=[affected_file],
            proposed_changes={},
            source_urls=[
                "https://github.com/aws-samples/startups/tree/main/migrate/plugins/migration-to-aws"
            ],
            scan_timestamp=scan_timestamp,
            status="pending",
            auditor_payload={
                "improvement_category": "completeness",
                "impact_score": 4,
                "effort_score": 2,
            },
            finding_schema_version="improvement/1.0",
        ))

    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_improvement_advisor(
    repo_content: RepoContent,
    run_id: str,
    timeout: float = 300.0,
) -> list[Finding]:
    """Run the Improvement Advisor agent on plugin reference files.

    This is the public entry point for the scan pipeline. It wraps the
    synchronous _run_advisor_impl in asyncio.to_thread and applies a
    timeout via asyncio.wait_for.

    Args:
        repo_content: Snapshot of the target repo with files dict.
        run_id: Current scan run identifier.
        timeout: Maximum execution time in seconds (default 300s).

    Returns:
        List of Finding objects with category="improvement".
        Empty list if no reference files are found.

    Raises:
        asyncio.TimeoutError: If execution exceeds the timeout.
    """
    # Early exit: if no reference files exist, return empty immediately.
    ref_files = _filter_reference_files(repo_content)
    if not ref_files:
        logger.info("No reference files found; skipping improvement advisor.")
        return []

    try:
        findings = await asyncio.wait_for(
            asyncio.to_thread(_run_advisor_impl, repo_content, run_id),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Improvement advisor timed out after %.0fs", timeout
        )
        raise

    logger.info(
        "Improvement advisor produced %d findings for run %s",
        len(findings),
        run_id,
    )
    return findings
