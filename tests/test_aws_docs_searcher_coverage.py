"""Property-based tests for AwsDocsSearcher fallback map coverage.

Feature: aws-docs-searcher-coverage
"""

from __future__ import annotations

import asyncio

import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from migration_watchdog.source_fetcher import AwsDocsSearcher, _FALLBACK_SERVICE_DOCS


# ---------------------------------------------------------------------------
# Property 1: All fallback map URLs are valid AWS docs URLs
# (exhaustive, not Hypothesis — iterates the finite map directly)
# ---------------------------------------------------------------------------

# Feature: aws-docs-searcher-coverage, Property 1: all fallback map URLs start with https://docs.aws.amazon.com/
def test_all_fallback_urls_are_aws_docs() -> None:
    """Property 1: All fallback map URLs start with https://docs.aws.amazon.com/

    Validates: Requirements 1.3, 4.1
    """
    for keyword, (title, url) in _FALLBACK_SERVICE_DOCS.items():
        assert url.startswith("https://docs.aws.amazon.com/"), (
            f"Keyword {keyword!r} has URL {url!r} which does not start with "
            "https://docs.aws.amazon.com/"
        )


# ---------------------------------------------------------------------------
# Property 3: Fallback map keyword coverage
# ---------------------------------------------------------------------------

# Feature: aws-docs-searcher-coverage, Property 3: any fallback map keyword returns at least one result with non-empty url
@settings(max_examples=100)
@given(st.sampled_from(list(_FALLBACK_SERVICE_DOCS.keys())))
def test_fallback_keyword_returns_result(keyword: str) -> None:
    """Property 3: For any keyword in the fallback map, _fallback_search returns
    at least one result with a non-empty url field.

    Validates: Requirements 1.1, 1.2
    """
    searcher = AwsDocsSearcher()
    results = searcher._fallback_search(keyword, limit=3)
    assert len(results) >= 1, f"Expected at least one result for keyword {keyword!r}, got none"
    for result in results:
        assert result.get("url"), (
            f"Result for keyword {keyword!r} has empty url: {result!r}"
        )


# ---------------------------------------------------------------------------
# Property 4: Non-matching queries return empty list
# ---------------------------------------------------------------------------

# Feature: aws-docs-searcher-coverage, Property 4: queries with no keyword match return empty list
@settings(max_examples=100)
@given(st.text(min_size=1, max_size=50))
def test_non_matching_query_returns_empty(query: str) -> None:
    """Property 4: Queries that contain none of the known keywords return [].

    Validates: Requirement 1.4
    """
    assume(not any(kw in query.lower() for kw in _FALLBACK_SERVICE_DOCS))
    searcher = AwsDocsSearcher()
    results = searcher._fallback_search(query, limit=3)
    assert results == [], (
        f"Expected [] for non-matching query {query!r}, got {results!r}"
    )


# ---------------------------------------------------------------------------
# Property 6: No result has an empty URL
# ---------------------------------------------------------------------------

# Feature: aws-docs-searcher-coverage, Property 6: no result from _fallback_search has an empty url
@settings(max_examples=100)
@given(st.sampled_from(list(_FALLBACK_SERVICE_DOCS.keys())))
def test_no_result_has_empty_url(keyword: str) -> None:
    """Property 6: No result dict returned by _fallback_search has an empty url field.

    Validates: Requirement 4.4
    """
    searcher = AwsDocsSearcher()
    results = searcher._fallback_search(keyword, limit=10)
    for result in results:
        assert result.get("url"), (
            f"Result for keyword {keyword!r} has empty url: {result!r}"
        )


# ---------------------------------------------------------------------------
# Property 2: All URL constructor outputs are valid AWS docs URLs
# ---------------------------------------------------------------------------

# Feature: aws-docs-searcher-coverage, Property 2: all URL constructor outputs start with https://docs.aws.amazon.com/
@settings(max_examples=100, deadline=None)
@given(
    st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" "),
        min_size=3,
        max_size=30,
    ).filter(lambda s: s.strip())
)
def test_url_constructor_outputs_are_aws_docs_urls(query: str) -> None:
    """Property 2: All URL constructor outputs start with https://docs.aws.amazon.com/

    No HTTP calls are made — the method returns a candidate URL directly without
    network validation.

    Validates: Requirements 2.2, 4.2
    """
    with patch("migration_watchdog.source_fetcher.httpx.AsyncClient") as mock_client_cls:
        searcher = AwsDocsSearcher()
        results = asyncio.run(searcher._construct_url_fallback(query, limit=3))
        # No HTTP client should have been instantiated
        mock_client_cls.assert_not_called()

    for result in results:
        assert result["url"].startswith("https://docs.aws.amazon.com/"), (
            f"URL {result['url']!r} does not start with https://docs.aws.amazon.com/"
        )


# ---------------------------------------------------------------------------
# Unit tests for URL constructor (no-HTTP behaviour)
# ---------------------------------------------------------------------------

def test_url_constructor_non_empty_query_returns_aws_url() -> None:
    """A query with at least one non-filler word returns a result whose URL
    starts with https://docs.aws.amazon.com/ — no HTTP calls are made.

    Validates: Requirements 2.1, 2.2
    """
    with patch("migration_watchdog.source_fetcher.httpx.AsyncClient") as mock_client_cls:
        searcher = AwsDocsSearcher()
        result = asyncio.run(searcher._construct_url_fallback("test service", limit=3))
        mock_client_cls.assert_not_called()
    assert len(result) == 1
    assert result[0]["url"].startswith("https://docs.aws.amazon.com/")


def test_url_constructor_no_http_client_instantiated() -> None:
    """_construct_url_fallback never instantiates httpx.AsyncClient.

    Validates: Requirements 2.1, 2.3
    """
    with patch("migration_watchdog.source_fetcher.httpx.AsyncClient") as mock_client_cls:
        searcher = AwsDocsSearcher()
        asyncio.run(searcher._construct_url_fallback("s3 bucket", limit=3))
        mock_client_cls.assert_not_called()


def test_url_constructor_empty_slug_returns_empty() -> None:
    searcher = AwsDocsSearcher()
    result = asyncio.run(searcher._construct_url_fallback("aws amazon the a an", limit=3))
    assert result == []


# ---------------------------------------------------------------------------
# Pipeline ordering unit tests
# ---------------------------------------------------------------------------

def test_pipeline_uses_live_search_first() -> None:
    """When live search returns results, fallback map and URL constructor are not called."""
    live_result = [{"title": "Live Result", "url": "https://docs.aws.amazon.com/live", "snippet": ""}]
    searcher = AwsDocsSearcher()
    with patch.object(searcher, "_parse_search_response", return_value=live_result), \
         patch.object(searcher, "_fallback_search") as mock_fallback, \
         patch.object(searcher, "_construct_url_fallback") as mock_construct:
        # Patch the HTTP call to return 200 so _parse_search_response is called
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "html"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        with patch("migration_watchdog.source_fetcher.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(searcher.search("test query", limit=3))
    assert result == live_result
    mock_fallback.assert_not_called()
    mock_construct.assert_not_called()


def test_pipeline_falls_back_to_map_when_live_empty() -> None:
    """When live search returns [], fallback map is used and URL constructor is not called."""
    searcher = AwsDocsSearcher()
    with patch.object(searcher, "_parse_search_response", return_value=[]), \
         patch.object(searcher, "_construct_url_fallback") as mock_construct:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "html"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        with patch("migration_watchdog.source_fetcher.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(searcher.search("lambda function", limit=3))
    # "lambda" is in the fallback map, so result should be non-empty
    assert len(result) > 0
    mock_construct.assert_not_called()


def test_pipeline_uses_url_constructor_when_both_empty() -> None:
    """When live search and fallback map both return [], URL constructor is called."""
    constructor_result = [{"title": "Constructed", "url": "https://docs.aws.amazon.com/constructed/latest/userguide/what-is-constructed.html", "snippet": ""}]
    searcher = AwsDocsSearcher()
    with patch.object(searcher, "_parse_search_response", return_value=[]), \
         patch.object(searcher, "_construct_url_fallback", new_callable=AsyncMock, return_value=constructor_result):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "html"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        with patch("migration_watchdog.source_fetcher.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(searcher.search("xyzzy-nonexistent-service-12345", limit=3))
    assert result == constructor_result


def test_pipeline_returns_empty_when_all_stages_fail() -> None:
    """When all three stages return [], search() returns []."""
    searcher = AwsDocsSearcher()
    with patch.object(searcher, "_parse_search_response", return_value=[]), \
         patch.object(searcher, "_construct_url_fallback", new_callable=AsyncMock, return_value=[]):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "html"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        with patch("migration_watchdog.source_fetcher.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(searcher.search("xyzzy-nonexistent-service-12345", limit=3))
    assert result == []


# ---------------------------------------------------------------------------
# Property 5: search_and_fetch returns content for all fallback map keywords
# ---------------------------------------------------------------------------

# Feature: aws-docs-searcher-coverage, Property 5: search_and_fetch returns non-empty string for any fallback map keyword when live search returns nothing
@settings(max_examples=100, deadline=None)
@given(st.sampled_from(list(_FALLBACK_SERVICE_DOCS.keys())))
def test_search_and_fetch_returns_content_for_fallback_keywords(keyword: str) -> None:
    """Property 5: search_and_fetch returns a non-empty string for any fallback map keyword
    when the live search is mocked to return no results.

    Validates: Requirements 4.3, 1.2
    """
    searcher = AwsDocsSearcher()

    # Mock the live search to return no results (simulates live search failure)
    # Mock fetch_page to return a fixed non-empty string (avoids real HTTP calls)
    with patch.object(searcher, "_parse_search_response", return_value=[]), \
         patch.object(searcher, "fetch_page", new_callable=AsyncMock, return_value="AWS service documentation content"):
        # Also mock the HTTP GET so _parse_search_response is called (not an exception)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "html"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        with patch("migration_watchdog.source_fetcher.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(searcher.search_and_fetch(keyword))

    assert result, (
        f"Expected non-empty string from search_and_fetch({keyword!r}) "
        "when live search returns nothing, got empty string"
    )
