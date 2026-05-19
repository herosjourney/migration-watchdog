from __future__ import annotations

from datetime import datetime, timezone

from migration_watchdog.models import Finding


def has_active_dismissal(finding: Finding) -> bool:
    """Return True if the finding has a dismissal with cooldown_expires > now (UTC).

    Normalises naive cooldown_expires datetimes to UTC before comparison.
    Returns False when finding.dismissal is None or the datetime is unparseable.
    """
    if finding.dismissal is None:
        return False
    try:
        expires = datetime.fromisoformat(finding.dismissal.cooldown_expires)
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return expires > datetime.now(timezone.utc)
    except (ValueError, TypeError):
        return False
