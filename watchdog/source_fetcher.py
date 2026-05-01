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

from watchdog.models import (
    AuthoritativeData,
    BedrockLifecycle,
    GeminiModelData,
    ModelLifecycleEntry,
    OpenAIModelData,
)
from watchdog.retry import retry_with_backoff

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
            return_exceptions=False,
        )

        aws_docs = results[0]
        blog_posts, whats_new_posts = results[1]
        gemini_data = results[2]
        openai_data = results[3]
        bedrock_lifecycle = results[4]
        aws_pricing = results[5]

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
        """Fetch pricing for a single AWS service via the Pricing API."""

        def _query_pricing() -> dict[str, Any]:
            client = boto3.client("pricing", region_name="us-east-1")
            response = client.get_products(
                ServiceCode=service_code,
                Filters=[
                    {
                        "Type": "TERM_MATCH",
                        "Field": "regionCode",
                        "Value": "us-east-1",
                    },
                ],
                MaxResults=10,
            )
            return {
                "service": service,
                "service_code": service_code,
                "price_list": response.get("PriceList", []),
            }

        async def _async_query() -> dict[str, Any]:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _query_pricing)

        return await retry_with_backoff(
            _async_query,
            max_retries=MAX_RETRIES,
            retryable_exceptions=(Exception,),
        )
