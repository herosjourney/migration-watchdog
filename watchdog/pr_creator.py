"""Pull request creator for approved findings.

Creates branches, commits updated markdown files, and opens pull requests
on the target repo for each approved finding. Assigns the designated
reviewer and uses retry with exponential backoff for GitHub API calls.
"""

from __future__ import annotations

import base64
import time

import httpx
import jwt

from watchdog.models import Finding, RiskLevel
from watchdog.retry import retry_with_backoff

# GitHub API base URL.
GITHUB_API = "https://api.github.com"

# HTTP status codes that should trigger a retry.
_RETRYABLE_STATUS_CODES = {401, 403, 429, 500, 502, 503, 504}

# Designated reviewer for all watchdog PRs.
_REVIEWER = "herosjourney"


class GitHubAPIError(Exception):
    """Raised when a GitHub API call returns an unexpected status code."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"GitHub API error {status_code}: {message}")


class PRCreator:
    """Creates pull requests on the target repo for approved findings.

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

    async def create_pr(
        self,
        finding: Finding,
        fork_owner: str,
        upstream_owner: str,
        repo: str,
        default_branch: str,
    ) -> str:
        """Create a cross-fork PR for an approved finding.

        1. Generate an installation token.
        2. Create a branch on the **fork** (``fork_owner/repo``).
        3. For each file in ``finding.proposed_changes``, update the file
           content on the new branch in the fork.
        4. Open a cross-fork PR from ``fork_owner:branch`` to
           ``upstream_owner/repo:default_branch``.
        5. Assign ``herosjourney`` as reviewer.

        All GitHub API calls are wrapped with ``retry_with_backoff`` (up to
        3 retries).

        Args:
            finding: The approved finding to create a PR for.
            fork_owner: Owner of the fork (e.g. ``"herosjourney"``).
            upstream_owner: Owner of the upstream repo (e.g. ``"aws-samples"``).
            repo: GitHub repository name.
            default_branch: The default branch to target (e.g. ``"main"``).

        Returns:
            The URL of the created pull request.
        """
        token = self._generate_installation_token()
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        }

        branch_name = self._build_branch_name(finding)

        async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
            # 1. Get the SHA of the upstream default branch HEAD.
            ref_data = await self._api_get(
                client,
                f"{GITHUB_API}/repos/{upstream_owner}/{repo}/git/ref/heads/{default_branch}",
            )
            base_sha = ref_data["object"]["sha"]

            # 2. Create the new branch on the FORK from the upstream HEAD.
            await self._api_post(
                client,
                f"{GITHUB_API}/repos/{fork_owner}/{repo}/git/refs",
                json_body={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
            )

            # 3. For each file in proposed_changes, update the file on the FORK.
            for file_path, new_content in finding.proposed_changes.items():
                # Get the current file SHA from the fork branch.
                file_data = await self._api_get(
                    client,
                    f"{GITHUB_API}/repos/{fork_owner}/{repo}/contents/{file_path}",
                    params={"ref": branch_name},
                )
                file_sha = file_data["sha"]

                # Update the file with the proposed content.
                encoded_content = base64.b64encode(
                    new_content.encode("utf-8")
                ).decode("utf-8")
                await self._api_put(
                    client,
                    f"{GITHUB_API}/repos/{fork_owner}/{repo}/contents/{file_path}",
                    json_body={
                        "message": f"watchdog: {finding.title}",
                        "content": encoded_content,
                        "sha": file_sha,
                        "branch": branch_name,
                    },
                )

            # 4. Open a cross-fork PR (from fork to upstream).
            pr_body = self._build_pr_body(finding)
            pr_data = await self._api_post(
                client,
                f"{GITHUB_API}/repos/{upstream_owner}/{repo}/pulls",
                json_body={
                    "title": f"[Watchdog] {finding.title}",
                    "body": pr_body,
                    "head": f"{fork_owner}:{branch_name}",
                    "base": default_branch,
                },
            )
            pr_number = pr_data["number"]
            pr_url = pr_data["html_url"]

            # 5. Assign reviewer.
            await self._api_post(
                client,
                f"{GITHUB_API}/repos/{upstream_owner}/{repo}/pulls/{pr_number}/requested_reviewers",
                json_body={"reviewers": [_REVIEWER]},
            )

        return pr_url

    def _build_pr_body(self, finding: Finding) -> str:
        """Build PR description with finding details, risk level, and sources.

        Args:
            finding: The finding to describe.

        Returns:
            A markdown-formatted PR body string.
        """
        risk_badge = {
            RiskLevel.LOW: "🟢 Low",
            RiskLevel.MEDIUM: "🟡 Medium",
            RiskLevel.HIGH: "🔴 High",
        }.get(finding.risk_level, str(finding.risk_level))

        lines: list[str] = []
        lines.append(f"## {finding.title}")
        lines.append("")
        lines.append(finding.description)
        lines.append("")
        lines.append(f"**Risk Level:** {risk_badge}")
        lines.append("")

        # Affected files.
        lines.append("### Affected Files")
        lines.append("")
        for f in finding.affected_files:
            lines.append(f"- `{f}`")
        lines.append("")

        # Source URLs.
        if finding.source_urls:
            lines.append("### Sources")
            lines.append("")
            for url in finding.source_urls:
                lines.append(f"- {url}")
            lines.append("")

        # Review status.
        if finding.review_status:
            lines.append(f"**Review Status:** {finding.review_status}")
            if finding.review_notes:
                lines.append("")
                lines.append(f"> {finding.review_notes}")
            lines.append("")

        lines.append("---")
        lines.append("*This PR was automatically created by the Migration Plugin Watchdog.*")

        return "\n".join(lines)

    def _build_branch_name(self, finding: Finding) -> str:
        """Generate branch name from finding category and ID.

        Format: ``watchdog/{category}-{short_id}`` where ``short_id`` is the
        first 8 characters of the finding_id.

        Args:
            finding: The finding to generate a branch name for.

        Returns:
            The branch name string.
        """
        short_id = finding.finding_id[:8]
        return f"watchdog/{finding.category}-{short_id}"

    # ------------------------------------------------------------------
    # Internal HTTP helpers with retry
    # ------------------------------------------------------------------

    async def _api_get(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict | None = None,
    ) -> dict | list:
        """Make a GET request to the GitHub API with retry logic."""

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

    async def _api_post(
        self,
        client: httpx.AsyncClient,
        url: str,
        json_body: dict,
    ) -> dict:
        """Make a POST request to the GitHub API with retry logic."""

        async def _do_request() -> dict:
            response = await client.post(url, json=json_body)
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

    async def _api_put(
        self,
        client: httpx.AsyncClient,
        url: str,
        json_body: dict,
    ) -> dict:
        """Make a PUT request to the GitHub API with retry logic."""

        async def _do_request() -> dict:
            response = await client.put(url, json=json_body)
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
