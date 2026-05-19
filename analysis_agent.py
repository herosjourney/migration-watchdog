"""Strands-based analysis agent using Claude Opus 4.7.

Implements the primary LLM agent that compares repo content against
authoritative sources, identifies discrepancies, classifies risk levels,
and generates findings with source citations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from uuid import uuid4

import httpx
from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from migration_watchdog.models import (
    AuthoritativeData,
    Finding,
    ModelStalenessResult,
    RepoContent,
    classify_risk,
)
from migration_watchdog.pricing_comparator import compare_pricing_entries, parse_pricing_cache
from migration_watchdog.model_comparator import (
    compare_model_lifecycle,
    compare_model_lists,
    compare_model_pricing,
)
from migration_watchdog.source_fetcher import AwsDocsSearcher

logger = logging.getLogger(__name__)

# Module-level docs searcher for analysis agent tools
_analysis_docs_searcher = AwsDocsSearcher()

# Module-level list that the create_finding tool appends to during a run.
# Reset at the start of each run_analysis() call.
_current_findings: list[Finding] = []
_current_run_id: str = ""

# ---------------------------------------------------------------------------
# Strands @tool functions
# ---------------------------------------------------------------------------


@tool
def compare_pricing(repo_pricing_md: str, current_pricing_json: str) -> str:
    """Compare the repo's pricing-cache.md content against current pricing data.

    Extracts prices from the markdown, compares against current data,
    flags differences exceeding tolerance (±5-10% infra, ±15-25% AI).
    Also checks last-updated date (>30 days = stale).

    Args:
        repo_pricing_md: Raw markdown content of pricing-cache.md
        current_pricing_json: JSON string of current pricing data from AWS/providers

    Returns:
        JSON string of PricingValidationResult with entries and staleness info
    """
    cached_entries = parse_pricing_cache(repo_pricing_md)
    current_prices: dict[str, dict[str, float]] = json.loads(current_pricing_json)
    result = compare_pricing_entries(cached_entries, current_prices)

    return json.dumps(
        {
            "entries": [
                {
                    "service": e.service,
                    "metric": e.metric,
                    "cached_value": e.cached_value,
                    "current_value": e.current_value,
                    "difference_pct": e.difference_pct,
                    "tolerance_pct": e.tolerance_pct,
                    "exceeds_tolerance": e.exceeds_tolerance,
                }
                for e in result.entries
            ],
            "stale_date": result.stale_date,
            "cache_last_updated": result.cache_last_updated,
            "validation_timestamp": result.validation_timestamp,
        }
    )



@tool
def compare_models(repo_model_md: str, provider: str, current_models_json: str) -> str:
    """Compare a model mapping guide against current model/pricing data.

    Args:
        repo_model_md: Raw markdown content of ai-gemini-to-bedrock.md or ai-openai-to-bedrock.md
        provider: "gemini" or "openai"
        current_models_json: JSON string of current model data from provider

    Returns:
        JSON string of ModelStalenessResult with changes detected
    """
    current_data = json.loads(current_models_json)

    # Extract model names from the repo markdown (simple heuristic: lines
    # that look like table rows with model identifiers).
    repo_model_names: set[str] = set()
    for line in repo_model_md.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and "---" not in stripped:
            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            if cells:
                repo_model_names.add(cells[0])

    current_model_names: set[str] = set(current_data.get("models", []))
    added, removed = compare_model_lists(repo_model_names, current_model_names)

    # Lifecycle comparison if lifecycle entries are provided
    from migration_watchdog.models import ModelLifecycleEntry

    repo_lifecycle = [
        ModelLifecycleEntry(
            model_name=e.get("model_name", ""),
            model_id=e.get("model_id", ""),
            status=e.get("status", "active"),
            eol_date=e.get("eol_date"),
            replacement=e.get("replacement"),
        )
        for e in current_data.get("repo_lifecycle", [])
    ]
    current_lifecycle = [
        ModelLifecycleEntry(
            model_name=e.get("model_name", ""),
            model_id=e.get("model_id", ""),
            status=e.get("status", "active"),
            eol_date=e.get("eol_date"),
            replacement=e.get("replacement"),
        )
        for e in current_data.get("current_lifecycle", [])
    ]
    lifecycle_changes = compare_model_lifecycle(repo_lifecycle, current_lifecycle)

    # Pricing comparison if pricing data is provided
    repo_pricing: dict[str, dict] = current_data.get("repo_pricing", {})
    current_pricing: dict[str, dict] = current_data.get("current_pricing", {})
    pricing_changes = compare_model_pricing(repo_pricing, current_pricing)

    result = ModelStalenessResult(
        gemini_changes=(lifecycle_changes + pricing_changes) if provider == "gemini" else [],
        openai_changes=(lifecycle_changes + pricing_changes) if provider == "openai" else [],
        bedrock_lifecycle_changes=[],
        scan_timestamp=datetime.utcnow().isoformat(),
    )

    return json.dumps(
        {
            "provider": provider,
            "added_models": sorted(added),
            "removed_models": sorted(removed),
            "lifecycle_changes": [
                {
                    "model_name": c.model_name,
                    "repo_status": c.repo_status,
                    "current_status": c.current_status,
                    "change_type": c.change_type,
                }
                for c in lifecycle_changes
            ],
            "pricing_changes": [
                {
                    "model_name": c.model_name,
                    "repo_pricing": c.repo_pricing,
                    "current_pricing": c.current_pricing,
                    "change_type": c.change_type,
                }
                for c in pricing_changes
            ],
            "scan_timestamp": result.scan_timestamp,
        }
    )


@tool
def compare_design_ref(
    repo_content: str, file_path: str, aws_docs_json: str, recent_posts_json: str
) -> str:
    """Compare a design-ref file against current AWS best practices.

    This is a pass-through that structures the data for the LLM to reason
    about. The LLM will use the returned data to identify discrepancies
    between the repo content and current authoritative sources.

    Args:
        repo_content: Raw markdown content of the design-ref file
        file_path: Path of the file being compared (e.g., "compute.md")
        aws_docs_json: JSON string of relevant AWS documentation content
        recent_posts_json: JSON string of relevant blog posts and What's New entries

    Returns:
        JSON string of comparison results (discrepancies found)
    """
    aws_docs = json.loads(aws_docs_json) if aws_docs_json else {}
    recent_posts = json.loads(recent_posts_json) if recent_posts_json else []

    return json.dumps(
        {
            "file_path": file_path,
            "repo_content_length": len(repo_content),
            "repo_content_preview": repo_content[:2000],
            "aws_docs_topics": list(aws_docs.keys()) if isinstance(aws_docs, dict) else [],
            "aws_docs": aws_docs,
            "recent_posts_count": len(recent_posts),
            "recent_posts": recent_posts,
        }
    )


@tool
def check_new_content_opportunities(
    repo_files_json: str, open_prs_json: str, recent_updates_json: str
) -> str:
    """Check for proactive content suggestions (Bedrock Agents, AgentCore,
    Strands SDK, startup migration guidance).

    First checks repo content and open PRs for existing coverage to avoid
    duplicates.

    Args:
        repo_files_json: JSON string of repo file paths and content
        open_prs_json: JSON string of open PR titles and changed files
        recent_updates_json: JSON string of recent AWS updates on agent tooling

    Returns:
        JSON string of opportunities found
    """
    repo_files: dict[str, str] = json.loads(repo_files_json)
    open_prs: list[dict] = json.loads(open_prs_json)
    recent_updates: list[dict] = json.loads(recent_updates_json)

    # Keywords for topics we want to check coverage for
    topic_keywords = {
        "bedrock_agents": ["bedrock agents", "bedrock agent"],
        "agentcore": ["agentcore", "agent core"],
        "strands_sdk": ["strands sdk", "strands-agents", "strands agent"],
        "startup_migration": ["startup migration", "startup agentic"],
    }

    opportunities: list[dict] = []

    for topic_id, keywords in topic_keywords.items():
        # Check if repo already covers this topic
        covered_in_repo = False
        for _path, content in repo_files.items():
            content_lower = content.lower()
            if any(kw in content_lower for kw in keywords):
                covered_in_repo = True
                break

        # Check if an open PR already addresses this topic
        covered_in_pr = False
        for pr in open_prs:
            pr_text = (pr.get("title", "") + " " + pr.get("body", "")).lower()
            if any(kw in pr_text for kw in keywords):
                covered_in_pr = True
                break

        # Check if there are recent updates about this topic
        relevant_updates = []
        for update in recent_updates:
            update_text = (
                update.get("title", "") + " " + update.get("content", "")
            ).lower()
            if any(kw in update_text for kw in keywords):
                relevant_updates.append(update)

        if relevant_updates and not covered_in_repo and not covered_in_pr:
            opportunities.append(
                {
                    "topic": topic_id,
                    "keywords": keywords,
                    "relevant_updates": relevant_updates,
                    "covered_in_repo": covered_in_repo,
                    "covered_in_pr": covered_in_pr,
                }
            )

    return json.dumps({"opportunities": opportunities})


@tool
def web_search(query: str) -> str:
    """Search the web for current information to verify claims or find updates.

    Use this tool when pre-fetched authoritative data is insufficient to
    confirm or deny a discrepancy. Always prefer pre-fetched data first;
    use web search only to fill gaps or verify uncertain claims.

    Args:
        query: Search query string (e.g., "AWS ECS Fargate pricing 2026",
               "Bedrock AgentCore latest features")

    Returns:
        JSON string of search results with titles, URLs, and snippets
    """
    try:
        # Use DuckDuckGo HTML search as a simple web search approach
        url = "https://html.duckduckgo.com/html/"
        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                url,
                data={"q": query},
                headers={"User-Agent": "MigrationPluginWatchdog/1.0"},
            )
            response.raise_for_status()

        # Parse basic results from the HTML response
        results: list[dict] = []
        html = response.text
        # Simple extraction of result snippets from DuckDuckGo HTML
        import re

        # Find result blocks
        result_blocks = re.findall(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</span>',
            html,
            re.DOTALL,
        )
        for href, title_html, snippet_html in result_blocks[:5]:
            # Strip HTML tags
            title = re.sub(r"<[^>]+>", "", title_html).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
            results.append(
                {"title": title, "url": href, "snippet": snippet}
            )

        return json.dumps({"query": query, "results": results})
    except Exception as exc:
        logger.warning("Web search failed for query '%s': %s", query, exc)
        return json.dumps(
            {
                "query": query,
                "results": [],
                "error": f"Search failed: {exc}",
            }
        )


@tool
def search_aws_docs(query: str) -> str:
    """Search AWS documentation for current information about a service, feature, or topic.

    Use this tool to:
    1. Verify whether a claim in the plugin's reference files is still accurate
    2. Find newer AWS services or features that might make current guidance obsolete
    3. Check service availability, status, or recommended alternatives

    Args:
        query: A specific search query, e.g. "App Runner new customers availability 2026",
               "AWS container services comparison Fargate ECS EKS",
               "Amazon Bedrock AgentCore latest features",
               "AWS serverless compute options 2025"

    Returns:
        Relevant text excerpts from AWS documentation pages.
    """
    return _analysis_docs_searcher.search_and_fetch(query)


@tool
def search_gcp_docs(query: str) -> str:
    """Search Google Cloud documentation to verify GCP-side claims in the migration plugin.

    Use this tool to verify claims about GCP services like Cloud Run, Cloud SQL,
    GKE, Vertex AI, Gemini, Pub/Sub, Firestore, etc. The migration plugin makes
    claims about GCP services as the source of migration — these need to be accurate.

    Args:
        query: A specific search query, e.g. "Cloud Run pricing 2026",
               "Gemini 2.0 Flash context window", "GKE autopilot features",
               "Cloud SQL PostgreSQL compatibility"

    Returns:
        Relevant text excerpts from Google Cloud documentation.
    """
    # Use the AwsDocsSearcher's fetch_page for GCP docs since they're plain HTTP
    from migration_watchdog.source_fetcher import GCP_DOC_URLS
    query_lower = query.lower()
    results: list[str] = []

    # Try to find a relevant GCP docs page
    for service, url in GCP_DOC_URLS.items():
        if service.replace("_", " ") in query_lower or service in query_lower:
            content = _analysis_docs_searcher.fetch_page(url)
            if content:
                results.append(f"[{service}] {content[:1500]}")
            break

    # Also try a web search for GCP-specific queries
    if not results:
        import httpx as _httpx
        try:
            with _httpx.Client(timeout=8.0) as client:
                resp = client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": f"site:cloud.google.com {query}"},
                    headers={"User-Agent": "MigrationPluginWatchdog/1.0"},
                )
                if resp.status_code == 200:
                    import re as _re
                    snippets = _re.findall(
                        r'class="result__snippet"[^>]*>(.*?)</span>',
                        resp.text, _re.DOTALL
                    )
                    for s in snippets[:3]:
                        clean = _re.sub(r'<[^>]+>', '', s).strip()
                        if clean:
                            results.append(clean)
        except Exception:
            pass

    return '\n\n'.join(results) if results else "No GCP documentation found for this query."


@tool
def check_service_obsolescence(
    service_name: str,
    current_recommendation: str,
    use_case: str,
) -> str:
    """Check whether current plugin guidance for a service is obsolete due to newer AWS offerings.

    Use this tool proactively for every service the plugin recommends. It searches AWS docs
    for newer alternatives, updated best practices, or service status changes that would
    make the current recommendation outdated.

    Args:
        service_name: The AWS service currently recommended (e.g., "AWS App Runner",
                      "Amazon ECS Fargate", "Amazon RDS Aurora")
        current_recommendation: What the plugin currently says about this service
                                (e.g., "Use App Runner for containerized web apps")
        use_case: The migration use case this service is recommended for
                  (e.g., "containerized web application", "managed database", "ML inference")

    Returns:
        JSON string with obsolescence findings: newer alternatives, status changes,
        updated best practices found in AWS documentation.
    """
    results: dict = {
        "service": service_name,
        "use_case": use_case,
        "current_recommendation": current_recommendation,
        "status_findings": [],
        "newer_alternatives": [],
        "updated_best_practices": [],
    }

    # 1. Check current service status and preview/regional availability
    status_content = _analysis_docs_searcher.search_and_fetch(
        f"{service_name} availability status new customers preview regions 2025 2026"
    )
    if status_content:
        status_lower = status_content.lower()
        closure_signals = [
            "closed to new customers", "not accepting new customers",
            "discontinued", "end of life", "no longer available",
            "closed for new", "will be closed", "service is closing",
        ]
        for signal in closure_signals:
            if signal in status_lower:
                results["status_findings"].append({
                    "type": "service_closure",
                    "signal": signal,
                    "content_preview": status_content[:500],
                })
                break

        # Check for preview status
        preview_signals = ["public preview", "in preview", "preview only", "limited preview", "limited availability"]
        for signal in preview_signals:
            if signal in status_lower:
                results["status_findings"].append({
                    "type": "preview_status",
                    "signal": signal,
                    "content_preview": status_content[:500],
                })
                break

        # Extract regional availability if mentioned
        import re as _re
        region_pattern = _re.compile(
            r'available in (\d+) regions?|supported in (\d+) regions?|(\d+) AWS regions?',
            _re.IGNORECASE
        )
        region_match = region_pattern.search(status_content)
        if region_match:
            count = next(g for g in region_match.groups() if g)
            results["status_findings"].append({
                "type": "regional_availability",
                "region_count": count,
                "content_preview": status_content[:300],
            })

    # 2. Search for newer alternatives
    alternatives_content = _analysis_docs_searcher.search_and_fetch(
        f"AWS {use_case} best practices 2025 recommended service alternative"
    )
    if alternatives_content:
        results["newer_alternatives"].append({
            "content_preview": alternatives_content[:800],
        })

    # 3. Check for updated best practices
    best_practices_content = _analysis_docs_searcher.search_and_fetch(
        f"{service_name} best practices migration guide updated"
    )
    if best_practices_content:
        results["updated_best_practices"].append({
            "content_preview": best_practices_content[:800],
        })

    import json as _json
    return _json.dumps(results)


@tool
def create_finding(
    category: str,
    title: str,
    description: str,
    affected_files: str,
    proposed_changes: str,
    source_urls: str,
) -> str:
    """Create a structured Finding from a detected discrepancy.

    Automatically classifies risk level based on category.
    Validates that source_urls is non-empty; rejects findings without citations.

    Args:
        category: One of "pricing", "model_deprecation", "new_model",
                  "guidance_update", "new_content", "structural",
                  "core_removal", "refactoring"
        title: Short descriptive title for the finding
        description: Detailed description of the discrepancy
        affected_files: JSON array of affected file paths
        proposed_changes: JSON object mapping file_path -> proposed new content
        source_urls: JSON array of authoritative source URLs

    Returns:
        JSON string of the created Finding with finding_id and risk_level
    """
    global _current_findings, _current_run_id

    # Parse JSON string arguments
    try:
        files_list: list[str] = json.loads(affected_files)
    except (json.JSONDecodeError, TypeError):
        files_list = [affected_files] if affected_files else []

    try:
        changes_dict: dict[str, str] = json.loads(proposed_changes)
    except (json.JSONDecodeError, TypeError):
        changes_dict = {}

    try:
        urls_list: list[str] = json.loads(source_urls)
    except (json.JSONDecodeError, TypeError):
        urls_list = [source_urls] if source_urls else []

    # Validate source_urls is non-empty — reject findings without citations
    if not urls_list:
        return json.dumps(
            {
                "error": "Finding rejected: source_urls must be non-empty. "
                "Every finding must have at least one authoritative source URL.",
                "created": False,
            }
        )

    risk_level = classify_risk(category)
    finding_id = str(uuid4())
    scan_timestamp = datetime.utcnow().isoformat()

    finding = Finding(
        finding_id=finding_id,
        run_id=_current_run_id,
        risk_level=risk_level,
        category=category,
        title=title,
        description=description,
        affected_files=files_list,
        proposed_changes=changes_dict,
        source_urls=urls_list,
        scan_timestamp=scan_timestamp,
        status="pending",
    )

    _current_findings.append(finding)

    return json.dumps(
        {
            "created": True,
            "finding_id": finding_id,
            "risk_level": risk_level.value,
            "category": category,
            "title": title,
            "scan_timestamp": scan_timestamp,
        }
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """You are the Migration Plugin Watchdog Analysis Agent. Your job is to \
compare the contents of the aws-samples/sample-agent-skills-for-aws-migration GitHub repository \
against current authoritative sources and identify outdated, stale, or missing content.

You will be given:
1. The current repo content (markdown files from the migration plugin)
2. Current authoritative data (AWS docs, pricing, Gemini/OpenAI model data, blog posts)

Your tasks:
- Compare pricing-cache.md against current AWS/provider pricing using the compare_pricing tool
- Compare AI model mapping guides against current model data using the compare_models tool
- Compare each design-ref file against current AWS best practices using the compare_design_ref tool
- Check for new content opportunities (Bedrock Agents, AgentCore, Strands SDK, startup migration) \
using the check_new_content_opportunities tool
- Use the search_aws_docs tool to search AWS documentation for current service status, \
  newer alternatives, and updated best practices
- Use the search_gcp_docs tool to verify GCP-side claims — the plugin makes claims about \
  GCP services (Cloud Run, Cloud SQL, GKE, Gemini, Vertex AI, etc.) as migration sources. \
  Use this tool to verify those claims are still accurate before flagging them as issues.
- Use the check_service_obsolescence tool for EVERY service the plugin recommends — \
  proactively check whether newer AWS services or features make the current guidance obsolete
- Use the web_search tool to verify claims or find current information when pre-fetched data \
is insufficient
- For each discrepancy found, create a Finding using the create_finding tool

PROACTIVE OBSOLESCENCE CHECKING (MANDATORY):
For each service mentioned in the plugin's reference files, call check_service_obsolescence to:
1. Verify the service is still available to new customers
2. Check if newer/better AWS alternatives exist for the same use case
3. Identify if AWS has released updated best practices that supersede current guidance
4. Flag if a recommended service has been deprecated, closed, or superseded

SELF-AWARE FILES — READ INTENT BEFORE FLAGGING:
Some reference files contain explicit instructions to recompute or refresh their data on each run \
(e.g., "Recompute the Status column on each run", "Days to EOL = EOL date − today"). \
Before creating a model_deprecation or staleness finding for such a file:
1. Check whether the file contains a "recompute on each run" or similar instruction.
2. If it does, verify whether the COMPUTED VALUES are actually wrong given today's date — \
   not just whether the header date is old.
3. If a model is correctly marked `excluded` or `legacy` in the file, do NOT flag it as a \
   problem — the file is working as designed. Only flag if the status label is WRONG \
   (e.g., a model past its EOL date is still marked `active` or `legacy` instead of `excluded`).
4. A stale header date alone (e.g., "as of April 10, 2026") is NOT a finding if the data \
   values are still correct. Only flag if the data itself is wrong.

CRITICAL GROUNDING RULES:
- ONLY flag discrepancies you can support with data from the tools (pre-fetched authoritative \
data, search_aws_docs, check_service_obsolescence, or web search results). Do NOT use your \
training knowledge as a source of truth.
- Every finding MUST include at least one source_url pointing to the authoritative data that \
supports the discrepancy. If you cannot cite a source, do NOT create the finding.
- If no authoritative data was fetched for a topic and all search tools return no relevant \
results, SKIP that topic entirely rather than guessing from training data.
- When in doubt, use search_aws_docs or web_search to verify before creating a finding.
- Be thorough but precise. Only create findings for genuine discrepancies backed by cited data."""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_analysis_agent() -> Agent:
    """Create the Strands analysis agent with Claude Opus 4.7."""
    model = BedrockModel(
        model_id="us.anthropic.claude-opus-4-7",
        region_name="us-east-1",
        max_tokens=16000,
    )
    return Agent(
        model=model,
        system_prompt=ANALYSIS_SYSTEM_PROMPT,
        tools=[
            compare_pricing,
            compare_models,
            compare_design_ref,
            check_new_content_opportunities,
            search_aws_docs,
            search_gcp_docs,
            check_service_obsolescence,
            web_search,
            create_finding,
        ],
    )


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------


def _build_analysis_prompt(
    repo_content: RepoContent,
    authoritative_data: AuthoritativeData,
    run_id: str,
) -> str:
    """Build the user message with all context for the analysis agent."""
    sections: list[str] = []

    sections.append(f"## Scan Run: {run_id}\n")

    # Repo content
    sections.append("## Repository Content\n")
    for path, content in repo_content.files.items():
        sections.append(f"### File: {path}\n```\n{content}\n```\n")

    # Open PRs
    if repo_content.open_prs:
        sections.append("## Open Pull Requests\n")
        for pr in repo_content.open_prs:
            sections.append(
                f"- PR #{pr.number}: {pr.title} (by {pr.author})\n"
                f"  Changed files: {', '.join(pr.changed_files)}\n"
            )

    # Authoritative data
    sections.append("## Authoritative Data\n")

    if authoritative_data.aws_pricing:
        sections.append(
            "### Current AWS Pricing\n```json\n"
            + json.dumps(authoritative_data.aws_pricing, indent=2)
            + "\n```\n"
        )

    if authoritative_data.gemini_models.models or authoritative_data.gemini_models.pricing:
        sections.append(
            "### Gemini Model Data\n```json\n"
            + json.dumps(
                {
                    "models": authoritative_data.gemini_models.models,
                    "pricing": authoritative_data.gemini_models.pricing,
                    "deprecations": authoritative_data.gemini_models.deprecations,
                },
                indent=2,
            )
            + "\n```\n"
        )

    if authoritative_data.openai_models.pricing:
        sections.append(
            "### OpenAI Model Data\n```json\n"
            + json.dumps(
                {
                    "pricing": authoritative_data.openai_models.pricing,
                    "deprecations": authoritative_data.openai_models.deprecations,
                },
                indent=2,
            )
            + "\n```\n"
        )

    if authoritative_data.bedrock_lifecycle.models:
        sections.append(
            "### Bedrock Model Lifecycle\n```json\n"
            + json.dumps(
                [
                    {
                        "model_name": m.model_name,
                        "model_id": m.model_id,
                        "status": m.status,
                        "eol_date": m.eol_date,
                        "replacement": m.replacement,
                    }
                    for m in authoritative_data.bedrock_lifecycle.models
                ],
                indent=2,
            )
            + "\n```\n"
        )

    if authoritative_data.aws_docs:
        sections.append(
            "### AWS Documentation\n```json\n"
            + json.dumps(authoritative_data.aws_docs, indent=2)
            + "\n```\n"
        )

    if authoritative_data.aws_blog_posts:
        sections.append(
            "### Recent AWS Blog Posts\n```json\n"
            + json.dumps(authoritative_data.aws_blog_posts, indent=2)
            + "\n```\n"
        )

    if authoritative_data.aws_whats_new:
        sections.append(
            "### Recent AWS What's New\n```json\n"
            + json.dumps(authoritative_data.aws_whats_new, indent=2)
            + "\n```\n"
        )

    if authoritative_data.partial_failures:
        sections.append(
            "### ⚠️ Partial Source Failures\n"
            "The following sources could not be fetched and should be noted "
            "when creating findings:\n"
            + "\n".join(f"- {s}" for s in authoritative_data.partial_failures)
            + "\n"
        )

    sections.append(
        "\n## Instructions\n"
        "Please analyze the repository content against the authoritative data above. "
        "Use the provided tools to compare pricing, models, design references, and "
        "check for new content opportunities. Create findings for any discrepancies "
        "you identify, ensuring each finding has proper source citations."
    )

    return "\n".join(sections)


def _build_pricing_prompt(
    repo_content: RepoContent,
    authoritative_data: AuthoritativeData,
    run_id: str,
) -> str:
    """Build a focused prompt for pricing cache analysis only."""
    sections: list[str] = [f"## Pricing Analysis for Run: {run_id}\n"]

    # Only include pricing-cache.md
    pricing_path = "features/migration-to-aws/skills/gcp-to-aws/references/shared/pricing-cache.md"
    for path, content in repo_content.files.items():
        if "pricing-cache" in path:
            sections.append(f"### File: {path}\n```\n{content[:8000]}\n```\n")

    if authoritative_data.aws_pricing:
        sections.append(
            "### Current AWS Pricing\n```json\n"
            + json.dumps(authoritative_data.aws_pricing, indent=2)[:4000]
            + "\n```\n"
        )

    sections.append(
        "\n## Instructions\n"
        "Compare the pricing-cache.md against current pricing data using the compare_pricing tool. "
        "Create findings for any price discrepancies that exceed tolerance thresholds. "
        "Every finding must have source_urls."
    )
    return "\n".join(sections)


def _build_models_prompt(
    repo_content: RepoContent,
    authoritative_data: AuthoritativeData,
    run_id: str,
) -> str:
    """Build a focused prompt for AI model staleness analysis only."""
    sections: list[str] = [f"## Model Staleness Analysis for Run: {run_id}\n"]

    # Only include AI-related files — use higher truncation limit since these files are large
    ai_keywords = ["ai-gemini", "ai-openai", "ai-model-lifecycle", "ai.md"]
    for path, content in repo_content.files.items():
        if any(kw in path for kw in ai_keywords):
            # ai-gemini-to-bedrock.md and ai-openai-to-bedrock.md are ~14-18KB
            # Use 12000 chars to capture most of the content
            sections.append(f"### File: {path}\n```\n{content[:12000]}\n```\n")

    if authoritative_data.gemini_models.models or authoritative_data.gemini_models.pricing:
        sections.append(
            "### Gemini Model Data\n```json\n"
            + json.dumps({
                "models": authoritative_data.gemini_models.models,
                "pricing": authoritative_data.gemini_models.pricing,
                "deprecations": authoritative_data.gemini_models.deprecations,
            }, indent=2)[:4000]
            + "\n```\n"
        )

    if authoritative_data.openai_models.pricing:
        sections.append(
            "### OpenAI Model Data\n```json\n"
            + json.dumps({
                "pricing": authoritative_data.openai_models.pricing,
                "deprecations": authoritative_data.openai_models.deprecations,
            }, indent=2)[:4000]
            + "\n```\n"
        )

    if authoritative_data.bedrock_lifecycle.models:
        sections.append(
            "### Bedrock Model Lifecycle\n```json\n"
            + json.dumps([
                {"model_name": m.model_name, "model_id": m.model_id,
                 "status": m.status, "eol_date": m.eol_date}
                for m in authoritative_data.bedrock_lifecycle.models
            ], indent=2)[:4000]
            + "\n```\n"
        )

    sections.append(
        "\n## Instructions\n"
        "Compare the AI model mapping guides and lifecycle file against current model data. "
        "Use compare_models tool and web_search to verify. "
        "Create findings for new models, deprecated models, and pricing changes. "
        "Every finding must have source_urls."
    )
    return "\n".join(sections)


def _build_guidance_prompt(
    repo_content: RepoContent,
    authoritative_data: AuthoritativeData,
    run_id: str,
) -> str:
    """Build a focused prompt for design-ref guidance analysis only."""
    sections: list[str] = [f"## Guidance Analysis for Run: {run_id}\n"]

    # Only include design-ref files (not phases, not shared)
    design_ref_keywords = ["design-refs/compute", "design-refs/database",
                           "design-refs/storage", "design-refs/networking",
                           "design-refs/messaging", "design-refs/security",
                           "design-refs/ai.md", "design-refs/ai-gemini",
                           "design-refs/ai-openai"]
    for path, content in repo_content.files.items():
        if any(kw in path for kw in design_ref_keywords):
            sections.append(f"### File: {path}\n```\n{content[:4000]}\n```\n")

    if authoritative_data.aws_docs:
        sections.append(
            "### AWS Documentation\n```json\n"
            + json.dumps(authoritative_data.aws_docs, indent=2)[:6000]
            + "\n```\n"
        )

    if authoritative_data.aws_blog_posts:
        sections.append(
            "### Recent AWS Blog Posts\n```json\n"
            + json.dumps(authoritative_data.aws_blog_posts[:10], indent=2)[:3000]
            + "\n```\n"
        )

    sections.append(
        "\n## Instructions\n"
        "Compare each design-ref file against current AWS best practices. "
        "Pay special attention to the AI files — check if guidance on agentic workflows, "
        "Bedrock Agents, AgentCore, and Strands SDK is current. "
        "For EVERY service mentioned in the design-ref files, call check_service_obsolescence "
        "to verify it is still available to new customers and that no newer/better alternative "
        "exists. Use search_aws_docs to find current service status and updated best practices. "
        "Use search_gcp_docs to verify GCP-side claims (Cloud Run, Cloud SQL, GKE, Gemini, etc.) "
        "are still accurate — the plugin's migration rationale depends on correctly describing "
        "the GCP source services. "
        "Use compare_design_ref tool and web_search to verify. "
        "Create findings for outdated guidance and obsolete service recommendations. "
        "Every finding must have source_urls."
    )
    return "\n".join(sections)


def _build_new_content_prompt(
    repo_content: RepoContent,
    authoritative_data: AuthoritativeData,
    run_id: str,
) -> str:
    """Build a focused prompt for new content opportunities only."""
    sections: list[str] = [f"## New Content Opportunities for Run: {run_id}\n"]

    # List repo file paths (not full content)
    file_count = len(repo_content.files)
    sections.append(f"### Repository Files ({file_count} files loaded)\n")
    for path in sorted(repo_content.files.keys()):
        sections.append(f"- {path}")
    sections.append("")

    # If fewer than 10 files are loaded, this is a PR-triggered run with limited context.
    # Suppress new_content findings to avoid false "missing coverage" claims.
    if file_count < 10:
        sections.append(
            "**NOTE: Only a small subset of repository files are loaded (PR-triggered run). "
            "Do NOT create 'new_content' or 'missing coverage' findings — you cannot see the "
            "full repository and would produce false positives. Skip this analysis entirely.**\n"
        )
        sections.append(
            "\n## Instructions\n"
            "This is a PR-triggered run with limited file context. "
            "Do NOT create any new_content findings. Return without creating any findings."
        )
        return "\n".join(sections)

    # Include AI-related file content snippets so the agent can see what's covered
    sections.append("### Current AI/Agent Coverage in Repo\n")
    ai_keywords = ["ai.md", "ai-gemini", "ai-openai", "ai-model-lifecycle", "agentcore", "harness", "strands"]
    for path, content in repo_content.files.items():
        if any(kw in path for kw in ai_keywords):
            sections.append(f"#### {path} (first 2000 chars)\n```\n{content[:2000]}\n```\n")

    # Open PRs
    if repo_content.open_prs:
        sections.append("### Open Pull Requests\n")
        for pr in repo_content.open_prs:
            sections.append(f"- PR #{pr.number}: {pr.title} ({', '.join(pr.changed_files[:5])})")
        sections.append("")

    if authoritative_data.aws_whats_new:
        sections.append(
            "### Recent AWS What's New\n```json\n"
            + json.dumps(authoritative_data.aws_whats_new[:10], indent=2)[:3000]
            + "\n```\n"
        )

    sections.append(
        "\n## Instructions\n"
        "Check for new content opportunities about:\n"
        "1. **Bedrock Agents** — managed agent orchestration\n"
        "2. **AgentCore** — agent runtime, harness, registry, evaluations\n"
        "3. **Strands SDK** — open-source agent framework\n"
        "4. **Agentic workflow migration** — how startups can migrate their "
        "agentic workflows and AI agent architectures from other platforms to AWS\n"
        "5. **Startup migration guidance** — specific patterns for startups\n\n"
        "Review the 'Current AI/Agent Coverage in Repo' section to see what's already covered. "
        "Use check_new_content_opportunities and web_search tools to find recent updates. "
        "Only suggest content not already covered by repo files or open PRs. "
        "Create findings for genuine gaps. Every finding must have source_urls."
    )
    return "\n".join(sections)


def run_analysis(
    repo_content: RepoContent,
    authoritative_data: AuthoritativeData,
    run_id: str,
) -> list[Finding]:
    """Run focused analysis sub-tasks to avoid token limits.

    Splits the analysis into 4 focused agent calls:
    1. Pricing cache validation
    2. AI model staleness detection
    3. Design-ref guidance comparison
    4. New content opportunities

    Each call gets only the relevant subset of repo content and
    authoritative data, keeping prompts within token limits.
    """
    global _current_findings, _current_run_id

    _current_findings = []
    _current_run_id = run_id

    agent = create_analysis_agent()

    sub_tasks = [
        ("pricing", _build_pricing_prompt),
        ("models", _build_models_prompt),
        ("guidance", _build_guidance_prompt),
        ("new_content", _build_new_content_prompt),
    ]

    for task_name, prompt_builder in sub_tasks:
        logger.info("Starting analysis sub-task: %s for run %s", task_name, run_id)
        try:
            prompt = prompt_builder(repo_content, authoritative_data, run_id)
            agent(prompt)
            logger.info(
                "Sub-task %s completed: %d total findings so far",
                task_name, len(_current_findings),
            )
        except Exception:
            logger.exception("Sub-task %s failed; continuing with next", task_name)

    # Mark findings from partial-failure sources
    if authoritative_data.partial_failures:
        for finding in _current_findings:
            _apply_partial_data_warnings(finding, authoritative_data.partial_failures)

    findings = list(_current_findings)
    logger.info(
        "Analysis completed for run %s: %d findings generated",
        run_id, len(findings),
    )
    return findings


def _apply_partial_data_warnings(
    finding: Finding, partial_failures: list[str]
) -> None:
    """Set partial_data_warning on a finding if it depends on a failed source."""
    # Map categories to source names that they depend on
    category_source_map: dict[str, list[str]] = {
        "pricing": ["aws_pricing", "gemini", "openai"],
        "model_deprecation": ["gemini", "openai", "bedrock_lifecycle"],
        "new_model": ["gemini", "openai", "bedrock_lifecycle"],
        "guidance_update": ["aws_docs", "aws_blogs"],
        "new_content": ["aws_blogs", "aws_whats_new"],
        "structural": ["aws_docs"],
    }

    dependent_sources = category_source_map.get(finding.category, [])
    for source in partial_failures:
        source_lower = source.lower()
        if any(dep in source_lower for dep in dependent_sources):
            finding.partial_data_warning = True
            return
