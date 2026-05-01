"""Finding deduplication logic.

Pure Python logic for deduplicating findings against open pull requests,
existing pending findings, and active dismissal cooldowns.
"""

from __future__ import annotations

from datetime import datetime, timezone

from migration_watchdog.models import Dismissal, Finding, PullRequest


def deduplicate(
    new_findings: list[Finding],
    open_prs: list[PullRequest],
    existing_findings: list[Finding],
) -> list[Finding]:
    """Remove findings already addressed by open PRs or existing findings.

    - If an open PR addresses the same files and topic, skip the finding.
    - If an existing pending finding matches (same affected files and category),
      update its scan_timestamp instead of creating a duplicate.
    - Respect dismissal cooldowns — skip findings with active dismissals.

    Returns the deduplicated list of findings.
    """
    result: list[Finding] = []

    for finding in new_findings:
        # 1. PR-based deduplication
        if any(_is_addressed_by_pr(finding, pr) for pr in open_prs):
            continue

        # 2. Dismissal cooldown respect
        matching_existing = _find_matching_existing(finding, existing_findings)
        if matching_existing is not None and _has_active_dismissal(matching_existing):
            continue

        # 3. Existing-finding deduplication — merge timestamps
        if matching_existing is not None and matching_existing.status == "pending":
            matching_existing.scan_timestamp = finding.scan_timestamp
            result.append(matching_existing)
            continue

        # 4. No match — new finding passes through
        result.append(finding)

    return result


def _is_addressed_by_pr(finding: Finding, pr: PullRequest) -> bool:
    """Return True if the PR's changed_files intersect with the finding's
    affected_files AND the PR title or body contains relevant keywords
    from the finding's title or category.
    """
    finding_files = set(finding.affected_files)
    pr_files = set(pr.changed_files)

    if not finding_files.intersection(pr_files):
        return False

    keywords = _extract_keywords(finding)
    pr_text = (pr.title + " " + pr.body).lower()

    return any(kw in pr_text for kw in keywords)


def _matches_existing(finding: Finding, existing: Finding) -> bool:
    """Return True if both findings have the same set of affected_files
    AND the same category.
    """
    return (
        set(finding.affected_files) == set(existing.affected_files)
        and finding.category == existing.category
    )


def _has_active_dismissal(finding: Finding) -> bool:
    """Return True if the finding has a dismissal with cooldown_expires > now."""
    if finding.dismissal is None:
        return False

    try:
        expires = datetime.fromisoformat(finding.dismissal.cooldown_expires)
        # Ensure timezone-aware comparison
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return expires > now
    except (ValueError, TypeError):
        return False


def _find_matching_existing(
    finding: Finding, existing_findings: list[Finding]
) -> Finding | None:
    """Find the first existing finding that matches the new finding."""
    for existing in existing_findings:
        if _matches_existing(finding, existing):
            return existing
    return None


def _extract_keywords(finding: Finding) -> list[str]:
    """Extract lowercase keywords from the finding's title and category
    for matching against PR text.
    """
    keywords: list[str] = []

    # Add category as a keyword
    category = finding.category.lower().replace("_", " ")
    keywords.append(category)
    # Also add individual words from multi-word categories
    for word in category.split():
        if len(word) > 2:
            keywords.append(word)

    # Add significant words from the title (skip short/common words)
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "shall", "for",
        "and", "but", "or", "nor", "not", "no", "so", "yet", "both",
        "each", "all", "any", "few", "more", "most", "some", "such",
        "than", "too", "very", "just", "about", "above", "after",
        "before", "between", "into", "through", "during", "with",
        "from", "up", "out", "on", "off", "over", "under", "again",
        "then", "once", "here", "there", "when", "where", "why",
        "how", "what", "which", "who", "whom", "this", "that",
        "these", "those", "in", "of", "to", "at", "by", "as", "it",
        "its", "if",
    }
    for word in finding.title.lower().split():
        # Strip punctuation
        cleaned = word.strip(".,;:!?\"'()-[]{}").lower()
        if len(cleaned) > 2 and cleaned not in stop_words:
            keywords.append(cleaned)

    return keywords
