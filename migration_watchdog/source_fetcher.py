"""Authoritative source data fetcher.

Retrieves current data from all authoritative sources in parallel,
including AWS documentation, AWS blogs/What's New, Gemini API docs,
OpenAI API docs, Bedrock model lifecycle, and AWS Pricing API.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

import boto3
import httpx

from migration_watchdog.models import (
    AuthoritativeData,
    BedrockLifecycle,
    GeminiModelData,
    ModelLifecycleEntry,
    OpenAIModelData,
)
from migration_watchdog.retry import retry_with_backoff

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# AWS documentation topics relevant to migration guidance
AWS_DOC_TOPICS: list[str] = [
    "compute",
    "database",
    "storage",
    "ai",
]

# AWS documentation URLs by topic
AWS_DOC_URLS: dict[str, str] = {
    "compute": "https://docs.aws.amazon.com/whitepapers/latest/aws-overview/compute-services.html",
    "database": "https://docs.aws.amazon.com/whitepapers/latest/aws-overview/database.html",
    "storage": "https://docs.aws.amazon.com/whitepapers/latest/aws-overview/storage-services.html",
    "ai": "https://docs.aws.amazon.com/whitepapers/latest/aws-overview/machine-learning.html",
}

# Gemini API documentation URLs
GEMINI_MODELS_URL = "https://ai.google.dev/gemini-api/docs/models"
GEMINI_PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
GEMINI_DEPRECATIONS_URL = "https://ai.google.dev/gemini-api/docs/deprecations"

# OpenAI API documentation URLs
OPENAI_PRICING_URL = "https://developers.openai.com/api/docs/pricing"
OPENAI_DEPRECATIONS_URL = "https://developers.openai.com/api/docs/deprecations"

# Bedrock model lifecycle URL
BEDROCK_LIFECYCLE_URL = (
    "https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html"
)

# AWS blogs and What's New URLs
AWS_BLOGS_URL = "https://aws.amazon.com/blogs/"
AWS_WHATS_NEW_URL = "https://aws.amazon.com/new/"

# GCP documentation URLs for verifying GCP-side claims in the migration plugin
GCP_DOC_URLS: dict[str, str] = {
    "cloud_run": "https://cloud.google.com/run/docs/overview/what-is-cloud-run",
    "cloud_sql": "https://cloud.google.com/sql/docs/introduction",
    "gke": "https://cloud.google.com/kubernetes-engine/docs/concepts/kubernetes-engine-overview",
    "vertex_ai": "https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform",
    "cloud_spanner": "https://cloud.google.com/spanner/docs/whatis",
    "firestore": "https://cloud.google.com/firestore/docs/overview",
    "cloud_storage": "https://cloud.google.com/storage/docs/introduction",
    "pub_sub": "https://cloud.google.com/pubsub/docs/overview",
    "cloud_functions": "https://cloud.google.com/functions/docs/concepts/overview",
}

# Keywords for filtering relevant blog/What's New posts
RELEVANCE_KEYWORDS: list[str] = [
    "bedrock",
    "agentcore",
    "strands",
    "migration",
    "fargate",
    "lambda",
    "dynamodb",
    "s3",
    "ecs",
    "eks",
    "aurora",
    "rds",
    "sagemaker",
    "startup",
    "compute",
    "database",
    "storage",
]

# AWS Pricing API service codes
AWS_PRICING_SERVICE_CODES: dict[str, str] = {
    "fargate": "AmazonECS",
    "lambda": "AWSLambda",
    "dynamodb": "AmazonDynamoDB",
    "s3": "AmazonS3",
    "bedrock": "AmazonBedrock",
    "ecs": "AmazonECS",
    "eks": "AmazonEKS",
    "aurora": "AmazonRDS",
    "rds": "AmazonRDS",
    "sagemaker": "AmazonSageMaker",
}

# Default HTTP timeout in seconds
HTTP_TIMEOUT = 30.0

# Max retries for each source fetch
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# HTML text extraction helper
# ---------------------------------------------------------------------------


class _HTMLTextExtractor(HTMLParser):
    """Simple HTML parser that extracts visible text content."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self._pieces.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._pieces)


def extract_text_from_html(html: str) -> str:
    """Extract visible text content from HTML."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# AwsDocsSearcher — live AWS documentation search
# ---------------------------------------------------------------------------


class AwsDocsSearcher:
    """Searches and fetches AWS documentation pages on demand.

    Uses the same AWS Documentation Search API that the AWS Documentation
    MCP server uses, making direct HTTP calls so it works in CI without
    a local MCP process.

    Two operations:
    - ``search(query)`` — returns a list of (title, url, snippet) tuples
    - ``fetch_page(url)`` — fetches and extracts text from a specific docs page

    Performance tuning:
    - Page fetch timeout: 8s (not 15s) — AWS docs pages load fast or not at all
    - search_and_fetch only fetches 1 page (not 3) to keep latency under 15s total
    - In-process cache prevents re-fetching the same page/query in the same run
    """

    _SEARCH_API = "https://docs.aws.amazon.com/search/doc-search.html"

    def __init__(self, timeout: float = 8.0) -> None:
        self._timeout = timeout
        # Simple in-process cache: query -> results
        self._search_cache: dict[str, list[dict]] = {}
        self._page_cache: dict[str, str] = {}

    def search(self, query: str, limit: int = 3) -> list[dict]:
        """Search AWS documentation for *query*.

        Returns a list of dicts with keys: title, url, snippet.
        Returns empty list on any error.
        """
        cache_key = f"{query}:{limit}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        results: list[dict] = []
        try:
            import httpx as _httpx
            params = {
                "searchPath": "documentation",
                "searchQuery": query,
                "this_doc_locale": "en_us",
            }
            with _httpx.Client(timeout=self._timeout) as client:
                resp = client.get(self._SEARCH_API, params=params)
                if resp.status_code == 200:
                    results = self._parse_search_response(resp.text, limit)
        except Exception as exc:
            logger.debug("AwsDocsSearcher.search failed for %r: %s", query, exc)

        # If HTML search didn't work, fall back to fetching specific service pages
        if not results:
            results = self._fallback_search(query, limit)

        self._search_cache[cache_key] = results
        return results

    def fetch_page(self, url: str) -> str:
        """Fetch and extract text content from an AWS docs page.

        Returns empty string on any error.
        """
        if url in self._page_cache:
            return self._page_cache[url]

        content = ""
        try:
            import httpx as _httpx
            with _httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                resp = client.get(url, headers={"Accept": "text/html"})
                if resp.status_code == 200:
                    content = extract_text_from_html(resp.text)[:6000]
        except Exception as exc:
            logger.debug("AwsDocsSearcher.fetch_page failed for %r: %s", url, exc)

        self._page_cache[url] = content
        return content

    def search_and_fetch(self, query: str) -> str:
        """Search for *query* and return content from the top result only.

        Fetches only 1 page to keep total latency under ~15 seconds.
        Returns up to 3000 chars of content.
        """
        results = self.search(query, limit=3)
        if not results:
            return ""

        parts: list[str] = []

        # Always include snippets from all results (no HTTP needed)
        for r in results[:3]:
            snippet = r.get("snippet", "")
            if snippet:
                parts.append(f"[{r.get('title', '')}] {snippet}")

        # Fetch only the top result's page
        top_url = results[0].get("url", "")
        if top_url:
            page_text = self.fetch_page(top_url)
            if page_text:
                parts.append(page_text[:2000])

        combined = "\n\n".join(parts)
        return combined[:3000]

    def _parse_search_response(self, html: str, limit: int) -> list[dict]:
        """Parse search results from AWS docs search HTML response."""
        results: list[dict] = []
        import json as _json
        import re as _re

        # Try to find JSON data in the response
        json_match = _re.search(r'"hits"\s*:\s*\{[^}]*"hits"\s*:\s*(\[.*?\])', html, _re.DOTALL)
        if json_match:
            try:
                hits = _json.loads(json_match.group(1))
                for hit in hits[:limit]:
                    source = hit.get("_source", {})
                    results.append({
                        "title": source.get("title", ""),
                        "url": source.get("url", ""),
                        "snippet": source.get("excerpt", ""),
                    })
                return results
            except Exception:
                pass

        # Fallback: extract links and titles from HTML
        link_pattern = _re.compile(
            r'<a[^>]+href="(https://docs\.aws\.amazon\.com[^"]+)"[^>]*>([^<]+)</a>',
            _re.IGNORECASE,
        )
        for m in link_pattern.finditer(html):
            url, title = m.group(1), m.group(2).strip()
            if title and url:
                results.append({"title": title, "url": url, "snippet": ""})
            if len(results) >= limit:
                break
        return results

    def _fallback_search(self, query: str, limit: int) -> list[dict]:
        """Fallback: return known AWS service page URLs relevant to the query.

        Does NOT fetch pages here — just returns URLs with empty snippets.
        The caller decides whether to fetch.
        """
        query_lower = query.lower()

        service_docs: dict[str, tuple[str, str]] = {
            "app runner": (
                "AWS App Runner",
                "https://docs.aws.amazon.com/apprunner/latest/dg/what-is-apprunner.html",
            ),
            "fargate": (
                "AWS Fargate",
                "https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html",
            ),
            "lambda": (
                "AWS Lambda",
                "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
            ),
            "eks": (
                "Amazon EKS",
                "https://docs.aws.amazon.com/eks/latest/userguide/what-is-eks.html",
            ),
            "ecs": (
                "Amazon ECS",
                "https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html",
            ),
            "bedrock": (
                "Amazon Bedrock",
                "https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html",
            ),
            "agentcore": (
                "Amazon Bedrock AgentCore",
                "https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html",
            ),
            "harness": (
                "Amazon Bedrock AgentCore Harness",
                "https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness.html",
            ),
            "dynamodb": (
                "Amazon DynamoDB",
                "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html",
            ),
            "s3": (
                "Amazon S3",
                "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            ),
            "rds": (
                "Amazon RDS",
                "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Welcome.html",
            ),
            "aurora": (
                "Amazon Aurora",
                "https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/CHAP_AuroraOverview.html",
            ),
            "quota": (
                "AWS Service Quotas",
                "https://docs.aws.amazon.com/servicequotas/latest/userguide/intro.html",
            ),
            "region": (
                "AWS Regions",
                "https://docs.aws.amazon.com/general/latest/gr/rande.html",
            ),
        }

        results: list[dict] = []
        for keyword, (title, url) in service_docs.items():
            if keyword in query_lower:
                results.append({"title": title, "url": url, "snippet": ""})
            if len(results) >= limit:
                break
        return results


# Module-level singleton for use as a Strands tool
_docs_searcher = AwsDocsSearcher()


# ---------------------------------------------------------------------------
# SourceFetcher
# ---------------------------------------------------------------------------


class SourceFetcher:
    """Fetches current data from all authoritative sources.

    Uses httpx (async) for all HTTP calls and ``retry_with_backoff``
    for resilience.  Partial failures are recorded in
    ``partial_failures`` but do not block the scan.
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._http_client = http_client
        self._owns_client = http_client is None
        self.partial_failures: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """Return the shared HTTP client, creating one if needed."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
            self._owns_client = True
        return self._http_client

    async def _close_client(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def _fetch_url(self, url: str) -> str:
        """Fetch a URL with retries and return the response text."""
        client = await self._get_client()

        async def _do_fetch() -> str:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text

        return await retry_with_backoff(
            _do_fetch,
            max_retries=MAX_RETRIES,
            retryable_exceptions=(httpx.HTTPStatusError, httpx.TransportError),
        )

    def _record_failure(self, source_name: str, error: Exception) -> None:
        """Log an error and record the source in partial_failures."""
        logger.error("Failed to fetch %s: %s", source_name, error)
        self.partial_failures.append(source_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_all_sources(self) -> AuthoritativeData:
        """Fetch all authoritative sources in parallel.

        Each source is fetched independently with 3 retries.
        Partial failures are recorded but do not block the scan.
        """
        self.partial_failures = []

        results = await asyncio.gather(
            self._safe_fetch_aws_docs(),
            self._safe_fetch_aws_blogs_and_whats_new(),
            self._safe_fetch_gemini_data(),
            self._safe_fetch_openai_data(),
            self._safe_fetch_bedrock_lifecycle(),
            self._safe_fetch_aws_pricing(),
            self._safe_fetch_gcp_docs(),
            return_exceptions=False,
        )

        aws_docs = results[0]
        blog_posts, whats_new_posts = results[1]
        gemini_data = results[2]
        openai_data = results[3]
        bedrock_lifecycle = results[4]
        aws_pricing = results[5]
        gcp_docs = results[6]

        try:
            await self._close_client()
        except Exception:  # noqa: BLE001
            pass

        return AuthoritativeData(
            aws_docs=aws_docs,
            aws_blog_posts=blog_posts,
            aws_whats_new=whats_new_posts,
            gemini_models=gemini_data,
            openai_models=openai_data,
            bedrock_lifecycle=bedrock_lifecycle,
            aws_pricing=aws_pricing,
            gcp_docs=gcp_docs,
            partial_failures=list(self.partial_failures),
        )

    # ------------------------------------------------------------------
    # Safe wrappers (catch exceptions, record partial failures)
    # ------------------------------------------------------------------

    async def _safe_fetch_aws_docs(self) -> dict[str, str]:
        try:
            return await self.fetch_aws_docs(AWS_DOC_TOPICS)
        except Exception as exc:  # noqa: BLE001
            self._record_failure("aws_docs", exc)
            return {}

    async def _safe_fetch_aws_blogs_and_whats_new(
        self,
    ) -> tuple[list[dict], list[dict]]:
        try:
            return await self.fetch_aws_blogs_and_whats_new()
        except Exception as exc:  # noqa: BLE001
            self._record_failure("aws_blogs_and_whats_new", exc)
            return ([], [])

    async def _safe_fetch_gemini_data(self) -> GeminiModelData:
        try:
            return await self.fetch_gemini_data()
        except Exception as exc:  # noqa: BLE001
            self._record_failure("gemini_data", exc)
            return GeminiModelData()

    async def _safe_fetch_openai_data(self) -> OpenAIModelData:
        try:
            return await self.fetch_openai_data()
        except Exception as exc:  # noqa: BLE001
            self._record_failure("openai_data", exc)
            return OpenAIModelData()

    async def _safe_fetch_bedrock_lifecycle(self) -> BedrockLifecycle:
        try:
            return await self.fetch_bedrock_lifecycle()
        except Exception as exc:  # noqa: BLE001
            self._record_failure("bedrock_lifecycle", exc)
            return BedrockLifecycle()

    async def _safe_fetch_aws_pricing(self) -> dict[str, dict]:
        try:
            return await self.fetch_aws_pricing(list(AWS_PRICING_SERVICE_CODES.keys()))
        except Exception as exc:  # noqa: BLE001
            self._record_failure("aws_pricing", exc)
            return {}

    async def _safe_fetch_gcp_docs(self) -> dict[str, str]:
        try:
            return await self.fetch_gcp_docs(list(GCP_DOC_URLS.keys()))
        except Exception as exc:  # noqa: BLE001
            self._record_failure("gcp_docs", exc)
            return {}

    # ------------------------------------------------------------------
    # Individual source fetchers
    # ------------------------------------------------------------------

    async def fetch_aws_docs(self, topics: list[str]) -> dict[str, str]:
        """Fetch AWS documentation for relevant topics.

        For each topic, fetches the corresponding AWS documentation page
        and extracts text content.

        Returns:
            A dict mapping topic name to extracted text content.
        """
        result: dict[str, str] = {}
        for topic in topics:
            url = AWS_DOC_URLS.get(topic)
            if url is None:
                continue
            try:
                html = await self._fetch_url(url)
                result[topic] = extract_text_from_html(html)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch AWS docs for %s: %s", topic, exc)
                self._record_failure(f"aws_docs_{topic}", exc)
        return result

    async def fetch_aws_blogs_and_whats_new(
        self,
    ) -> tuple[list[dict], list[dict]]:
        """Fetch recent posts from AWS blogs and What's New.

        Fetches the main blog and What's New pages, extracts text, and
        returns posts relevant to compute, database, storage, AI migration,
        Bedrock Agents, AgentCore, Strands SDK, and startup migration.

        Returns:
            A tuple of (blog_posts, whats_new_posts), each a list of dicts
            with keys: title, url, date, summary.
        """
        blog_posts: list[dict] = []
        whats_new_posts: list[dict] = []

        # Fetch blogs page
        try:
            html = await self._fetch_url(AWS_BLOGS_URL)
            blog_posts = self._parse_blog_entries(html, AWS_BLOGS_URL)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch AWS blogs: %s", exc)
            self._record_failure("aws_blogs", exc)

        # Fetch What's New page
        try:
            html = await self._fetch_url(AWS_WHATS_NEW_URL)
            whats_new_posts = self._parse_blog_entries(html, AWS_WHATS_NEW_URL)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch AWS What's New: %s", exc)
            self._record_failure("aws_whats_new", exc)

        return blog_posts, whats_new_posts

    def _parse_blog_entries(
        self, html: str, base_url: str
    ) -> list[dict]:
        """Extract blog/news entries from HTML content.

        Performs basic extraction of links and surrounding text, then
        filters for relevance using keyword matching.
        """
        entries: list[dict] = []
        text = extract_text_from_html(html)

        # Simple heuristic: split text into chunks and look for relevant content
        lines = text.split("\n")
        current_entry: dict[str, str] | None = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this line looks like a title (short, no period at end)
            if len(stripped) < 200 and not stripped.endswith("."):
                lower = stripped.lower()
                if any(kw in lower for kw in RELEVANCE_KEYWORDS):
                    if current_entry:
                        entries.append(current_entry)
                    current_entry = {
                        "title": stripped,
                        "url": base_url,
                        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "summary": "",
                    }
            elif current_entry and not current_entry["summary"]:
                current_entry["summary"] = stripped[:500]

        if current_entry:
            entries.append(current_entry)

        return entries

    async def fetch_gemini_data(self) -> GeminiModelData:
        """Fetch Gemini models, pricing, and deprecations.

        Fetches from:
        - Models: https://ai.google.dev/gemini-api/docs/models
        - Pricing: https://ai.google.dev/gemini-api/docs/pricing
        - Deprecations: https://ai.google.dev/gemini-api/docs/deprecations

        Returns:
            Structured GeminiModelData with parsed content.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Fetch models
        models: list[dict] = []
        try:
            html = await self._fetch_url(GEMINI_MODELS_URL)
            text = extract_text_from_html(html)
            models = self._parse_gemini_models(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch Gemini models: %s", exc)
            self._record_failure("gemini_models", exc)

        # Fetch pricing
        pricing: dict[str, dict] = {}
        try:
            html = await self._fetch_url(GEMINI_PRICING_URL)
            text = extract_text_from_html(html)
            pricing = self._parse_gemini_pricing(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch Gemini pricing: %s", exc)
            self._record_failure("gemini_pricing", exc)

        # Fetch deprecations
        deprecations: list[dict] = []
        try:
            html = await self._fetch_url(GEMINI_DEPRECATIONS_URL)
            text = extract_text_from_html(html)
            deprecations = self._parse_deprecations(text, "gemini")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch Gemini deprecations: %s", exc)
            self._record_failure("gemini_deprecations", exc)

        return GeminiModelData(
            models=models,
            pricing=pricing,
            deprecations=deprecations,
            fetched_at=now,
        )

    def _parse_gemini_models(self, text: str) -> list[dict]:
        """Parse Gemini model names from extracted text."""
        models: list[dict] = []
        seen: set[str] = set()
        # Look for model identifiers like "gemini-2.0-flash", "gemini-1.5-pro", etc.
        pattern = re.compile(r"gemini[-\s][\w.\-]+", re.IGNORECASE)
        for match in pattern.finditer(text):
            name = match.group(0).strip().lower().replace(" ", "-")
            if name not in seen:
                seen.add(name)
                models.append({"name": name, "raw_text": match.group(0)})
        return models

    def _parse_gemini_pricing(self, text: str) -> dict[str, dict]:
        """Parse pricing information from Gemini pricing page text."""
        pricing: dict[str, dict] = {}
        # Extract the raw text as pricing context
        if text.strip():
            pricing["raw_content"] = {"text": text[:5000]}
        return pricing

    def _parse_deprecations(self, text: str, provider: str) -> list[dict]:
        """Parse deprecation notices from extracted text."""
        deprecations: list[dict] = []
        if text.strip():
            deprecations.append({
                "provider": provider,
                "raw_content": text[:5000],
            })
        return deprecations

    async def fetch_openai_data(self) -> OpenAIModelData:
        """Fetch OpenAI pricing and deprecations.

        Fetches from:
        - Pricing: https://developers.openai.com/api/docs/pricing
        - Deprecations: https://developers.openai.com/api/docs/deprecations

        Returns:
            Structured OpenAIModelData with parsed content.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Fetch pricing
        pricing: dict[str, dict] = {}
        try:
            html = await self._fetch_url(OPENAI_PRICING_URL)
            text = extract_text_from_html(html)
            if text.strip():
                pricing["raw_content"] = {"text": text[:5000]}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch OpenAI pricing: %s", exc)
            self._record_failure("openai_pricing", exc)

        # Fetch deprecations
        deprecations: list[dict] = []
        try:
            html = await self._fetch_url(OPENAI_DEPRECATIONS_URL)
            text = extract_text_from_html(html)
            deprecations = self._parse_deprecations(text, "openai")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch OpenAI deprecations: %s", exc)
            self._record_failure("openai_deprecations", exc)

        return OpenAIModelData(
            pricing=pricing,
            deprecations=deprecations,
            fetched_at=now,
        )

    async def fetch_bedrock_lifecycle(self) -> BedrockLifecycle:
        """Fetch Bedrock model lifecycle page and parse model statuses.

        Fetches from:
        https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html

        Returns:
            Structured BedrockLifecycle with parsed model entries.
        """
        now = datetime.now(timezone.utc).isoformat()

        html = await self._fetch_url(BEDROCK_LIFECYCLE_URL)
        text = extract_text_from_html(html)
        models = self._parse_bedrock_lifecycle(text)

        return BedrockLifecycle(models=models, fetched_at=now)

    def _parse_bedrock_lifecycle(self, text: str) -> list[ModelLifecycleEntry]:
        """Parse Bedrock model lifecycle entries from extracted text.

        Looks for patterns like model names followed by status indicators
        (active, legacy, EOL) and optional dates.
        """
        entries: list[ModelLifecycleEntry] = []
        seen: set[str] = set()

        # Look for common Bedrock model ID patterns
        model_pattern = re.compile(
            r"([\w.-]+(?:claude|titan|llama|mistral|cohere|ai21|stability|meta|amazon|anthropic)[\w.\-]*)",
            re.IGNORECASE,
        )

        lines = text.split("\n")
        for line in lines:
            line_lower = line.lower()
            model_match = model_pattern.search(line)
            if model_match:
                model_id = model_match.group(1).strip()
                if model_id in seen:
                    continue
                seen.add(model_id)

                # Determine status
                status = "active"
                if "eol" in line_lower or "end of life" in line_lower:
                    status = "eol"
                elif "legacy" in line_lower or "deprecated" in line_lower:
                    status = "legacy"

                # Look for dates (YYYY-MM-DD pattern)
                date_match = re.search(r"\d{4}-\d{2}-\d{2}", line)
                eol_date = date_match.group(0) if date_match else None

                entries.append(
                    ModelLifecycleEntry(
                        model_name=model_id,
                        model_id=model_id,
                        status=status,
                        eol_date=eol_date,
                        replacement=None,
                    )
                )

        return entries

    async def fetch_gcp_docs(self, services: list[str]) -> dict[str, str]:
        """Fetch GCP documentation for specified services.

        Used to verify GCP-side claims in the migration plugin — e.g., whether
        Cloud Run still works the same way, or whether Gemini pricing has changed.

        Args:
            services: List of service keys from GCP_DOC_URLS.

        Returns:
            A dict mapping service name to extracted text content.
        """
        result: dict[str, str] = {}
        for service in services:
            url = GCP_DOC_URLS.get(service)
            if url is None:
                continue
            try:
                html = await self._fetch_url(url)
                content = extract_text_from_html(html)[:4000]
                result[service] = content
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch GCP docs for %s: %s", service, exc)
                self._record_failure(f"gcp_docs_{service}", exc)
        return result

    async def fetch_aws_pricing(self, services: list[str]) -> dict[str, dict]:
        """Query the AWS Pricing API for specified services.

        Uses boto3 to query the AWS Pricing API for each service.

        Args:
            services: List of service keys (e.g., ["fargate", "lambda", "dynamodb"]).

        Returns:
            A dict mapping service name to pricing data.
        """
        result: dict[str, dict] = {}

        for service in services:
            service_code = AWS_PRICING_SERVICE_CODES.get(service)
            if service_code is None:
                continue
            try:
                pricing_data = await self._fetch_service_pricing(service, service_code)
                result[service] = pricing_data
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to fetch AWS pricing for %s: %s", service, exc
                )
                self._record_failure(f"aws_pricing_{service}", exc)

        return result

    async def _fetch_service_pricing(
        self, service: str, service_code: str
    ) -> dict[str, Any]:
        """Fetch pricing for a single AWS service via the Pricing API.

        Uses service-specific filters to return the exact pricing dimensions
        that the plugin's pricing-cache.md tracks (e.g., Fargate per-vCPU-hour,
        Lambda per-GB-second, etc.) rather than the first page of all SKUs.
        """
        # Service-specific filters to get the right pricing dimensions.
        # Each entry is a list of additional TERM_MATCH filters beyond regionCode.
        _SERVICE_FILTERS: dict[str, list[dict]] = {
            "fargate": [
                {"Type": "TERM_MATCH", "Field": "usagetype", "Value": "AmazonECS:Fargate-vCPU-Hours:perCPU"},
            ],
            "lambda": [
                {"Type": "TERM_MATCH", "Field": "group", "Value": "AWS-Lambda-Duration"},
            ],
            "dynamodb": [
                {"Type": "TERM_MATCH", "Field": "usagetype", "Value": "AmazonDynamoDB:WriteRequestUnits"},
            ],
            "s3": [
                {"Type": "TERM_MATCH", "Field": "usagetype", "Value": "AmazonS3:TimedStorage-ByteHrs"},
            ],
            "bedrock": [
                {"Type": "TERM_MATCH", "Field": "usagetype", "Value": "AmazonBedrock:InputTokens"},
            ],
            "ecs": [
                {"Type": "TERM_MATCH", "Field": "usagetype", "Value": "AmazonECS:Fargate-vCPU-Hours:perCPU"},
            ],
            "eks": [
                {"Type": "TERM_MATCH", "Field": "usagetype", "Value": "AmazonEKS:AmazonEKS-Hours:perCluster"},
            ],
            "aurora": [
                {"Type": "TERM_MATCH", "Field": "databaseEngine", "Value": "Aurora MySQL"},
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": "db.r6g.large"},
            ],
            "rds": [
                {"Type": "TERM_MATCH", "Field": "databaseEngine", "Value": "MySQL"},
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": "db.t4g.micro"},
            ],
            "sagemaker": [
                {"Type": "TERM_MATCH", "Field": "usagetype", "Value": "AmazonSageMaker:ml.t3.medium"},
            ],
        }

        extra_filters = _SERVICE_FILTERS.get(service, [])

        def _query_pricing() -> dict[str, Any]:
            client = boto3.client("pricing", region_name="us-east-1")
            filters = [
                {
                    "Type": "TERM_MATCH",
                    "Field": "regionCode",
                    "Value": "us-east-1",
                },
            ] + extra_filters

            response = client.get_products(
                ServiceCode=service_code,
                Filters=filters,
                MaxResults=5,
            )
            price_list = response.get("PriceList", [])

            # Parse the price list to extract numeric values
            parsed_prices: dict[str, float] = {}
            import json as _json
            for item_str in price_list:
                try:
                    item = _json.loads(item_str) if isinstance(item_str, str) else item_str
                    terms = item.get("terms", {}).get("OnDemand", {})
                    for term in terms.values():
                        for dim in term.get("priceDimensions", {}).values():
                            desc = dim.get("description", "")
                            price_per_unit = dim.get("pricePerUnit", {}).get("USD", "0")
                            try:
                                price_val = float(price_per_unit)
                                if price_val > 0:
                                    # Use description as key, truncated
                                    key = desc[:60].strip().lower().replace(" ", "_")
                                    parsed_prices[key] = price_val
                            except (ValueError, TypeError):
                                pass
                except Exception:
                    pass

            return {
                "service": service,
                "service_code": service_code,
                "price_list": price_list,
                "parsed_prices": parsed_prices,
            }

        async def _async_query() -> dict[str, Any]:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _query_pricing)

        return await retry_with_backoff(
            _async_query,
            max_retries=MAX_RETRIES,
            retryable_exceptions=(Exception,),
        )
