"""GitHub App authentication and repository content scanner.

Fetches the target repo's default branch contents and open pull requests
via the GitHub API using short-lived installation tokens generated from
the GitHub App private key.
"""

from __future__ import annotations

import base64
import time
from datetime import datetime, timezone

import httpx
import jwt

from watchdog.models import PullRequest, RepoContent
from watchdog.retry import retry_with_backoff

# Path prefix for markdown files to scan in the target repo.
REFERENCES_PATH = "features/migration-to-aws/skills/gcp-to-aws/references"

# GitHub API base URL.
GITHUB_API = "https://api.github.com"

# HTTP status codes that should trigger a retry.
_RETRYABLE_STATUS_CODES = {401, 403, 429, 500, 502, 503, 504}


class GitHubAPIError(Exception):
    """Raised when a GitHub API call returns an unexpected status code."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"GitHub API error {status_code}: {message}")


class RepoScanner:
    """Scans a GitHub repository for markdown content and open pull requests.

    Authenticates as a GitHub App using short-lived installation tokens
    generated from the provided private key.
    """

    def __init__(
        self,
        github_app_id: str,
        app_private_key: str,
        installation_id: str,
    ) -> None:
        self.github_app_id = github_app_id
        self.app_private_key = app_private_key
        self.installation_id = installation_id

    def _generate_installation_token(self) -> str:
        """Generate a short-lived GitHub App installation token.

        Creates a JWT signed with the GitHub App private key, then exchanges
        it for an installation access token via the GitHub API.

        Returns:
            A short-lived installation access token string.

        Raises:
            GitHubAPIError: If the token exchange request fails.
        """
        now = int(time.time())
        payload = {
            "iss": self.github_app_id,
            "iat": now - 60,
            "exp": now + 600,
        }
        encoded_jwt = jwt.encode(payload, self.app_private_key, algorithm="RS256")

        response = httpx.post(
            f"{GITHUB_API}/app/installations/{self.installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {encoded_jwt}",
                "Accept": "application/vnd.github+json",
            },
        )
        if response.status_code != 201:
            raise GitHubAPIError(response.status_code, response.text)
        return response.json()["token"]

    async def fetch_repo_snapshot(self, owner: str, repo: str) -> RepoContent:
        """Fetch default branch contents and open PRs.

        Uses the GitHub App installation token to:
        1. Determine the repo's default branch.
        2. Recursively fetch all ``.md`` files under the references path
           using the Git Trees API.
        3. Fetch the content of each discovered markdown file.
        4. Fetch all open pull requests and their changed files.

        All GitHub API calls are wrapped with ``retry_with_backoff`` to handle
        rate-limit (403/429), auth (401), and server (5xx) errors.

        Args:
            owner: GitHub repository owner (e.g. ``"aws-samples"``).
            repo: GitHub repository name.

        Returns:
            A :class:`RepoContent` dataclass with files, open PRs, commit SHA,
            and fetch timestamp.
        """
        token = self._generate_installation_token()
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        }

        async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
            # 1. Get repo metadata to find default branch and HEAD SHA.
            repo_data = await self._api_get(
                client, f"{GITHUB_API}/repos/{owner}/{repo}"
            )
            default_branch = repo_data["default_branch"]

            # 2. Get the tree SHA for the default branch.
            branch_data = await self._api_get(
                client,
                f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{default_branch}",
                params={"recursive": "1"},
            )
            commit_sha = branch_data.get("sha", "")

            # 3. Filter for .md files under the references path.
            md_files: dict[str, str] = {}
            tree_items = branch_data.get("tree", [])
            md_paths = [
                item["path"]
                for item in tree_items
                if item.get("type") == "blob"
                and item["path"].startswith(REFERENCES_PATH)
                and item["path"].endswith(".md")
            ]

            # 4. Fetch content for each markdown file.
            for path in md_paths:
                content_data = await self._api_get(
                    client,
                    f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}",
                )
                # GitHub returns base64-encoded content for files.
                encoded = content_data.get("content", "")
                md_files[path] = base64.b64decode(encoded).decode("utf-8")

            # 5. Fetch all open pull requests.
            prs_data = await self._api_get(
                client,
                f"{GITHUB_API}/repos/{owner}/{repo}/pulls",
                params={"state": "open"},
            )

            open_prs: list[PullRequest] = []
            for pr in prs_data:
                # Fetch changed files for each PR.
                pr_files_data = await self._api_get(
                    client,
                    f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr['number']}/files",
                )
                changed_files = [f["filename"] for f in pr_files_data]
                open_prs.append(
                    PullRequest(
                        number=pr["number"],
                        title=pr.get("title", ""),
                        body=pr.get("body", "") or "",
                        changed_files=changed_files,
                        author=pr.get("user", {}).get("login", ""),
                    )
                )

        return RepoContent(
            files=md_files,
            open_prs=open_prs,
            commit_sha=commit_sha,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )

    async def _api_get(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict | None = None,
    ) -> dict | list:
        """Make a GET request to the GitHub API with retry logic.

        Retries on 401, 403, 429, and 5xx status codes using exponential
        backoff.

        Args:
            client: The httpx async client to use.
            url: Full URL to request.
            params: Optional query parameters.

        Returns:
            Parsed JSON response (dict or list).

        Raises:
            GitHubAPIError: If the request fails after all retries.
        """

        async def _do_request() -> dict | list:
            response = await client.get(url, params=params)
            if response.status_code in _RETRYABLE_STATUS_CODES:
                raise GitHubAPIError(response.status_code, response.text)
            if response.status_code >= 400:
                raise GitHubAPIError(response.status_code, response.text)
            return response.json()

        return await retry_with_backoff(
            _do_request,
            max_retries=3,
            base_delay=1.0,
            retryable_exceptions=(GitHubAPIError,),
        )
