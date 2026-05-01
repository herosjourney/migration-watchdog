"""Model comparison logic for the Migration Plugin Watchdog.

Pure functions that compare repo model data against current authoritative
source data to detect staleness in model lists, lifecycle status, and pricing.
"""

from __future__ import annotations

from migration_watchdog.models import ModelComparisonEntry, ModelLifecycleEntry


def compare_model_lists(
    repo_models: set[str], current_models: set[str]
) -> tuple[set[str], set[str]]:
    """Compare two sets of model names and return (added, removed).

    Args:
        repo_models: Model names present in the repo.
        current_models: Model names from the authoritative source.

    Returns:
        A tuple of (added, removed) where:
        - added: models in current but not in repo (new models).
        - removed: models in repo but not in current (no longer available).
    """
    added = current_models - repo_models
    removed = repo_models - current_models
    return added, removed


def compare_model_lifecycle(
    repo_entries: list[ModelLifecycleEntry],
    current_entries: list[ModelLifecycleEntry],
) -> list[ModelComparisonEntry]:
    """Compare model lifecycle entries between repo and authoritative source.

    Matches entries by ``model_id``. For each matched pair, reports a
    discrepancy if ``status`` or ``eol_date`` differs. Entries only in
    *current* are reported as ``"new_model"``; entries only in *repo* are
    reported as ``"removed"``.

    Args:
        repo_entries: Lifecycle entries from the repo file.
        current_entries: Lifecycle entries from the authoritative source.

    Returns:
        A list of :class:`ModelComparisonEntry` objects describing changes.
    """
    repo_by_id: dict[str, ModelLifecycleEntry] = {e.model_id: e for e in repo_entries}
    current_by_id: dict[str, ModelLifecycleEntry] = {e.model_id: e for e in current_entries}

    results: list[ModelComparisonEntry] = []

    # Check models present in both or only in current
    for model_id, current in current_by_id.items():
        repo = repo_by_id.get(model_id)
        if repo is None:
            # New model not in repo
            results.append(
                ModelComparisonEntry(
                    provider="bedrock",
                    model_name=current.model_name,
                    repo_status=None,
                    current_status=current.status,
                    repo_eol_date=None,
                    current_eol_date=current.eol_date,
                    change_type="new_model",
                )
            )
        else:
            # Both exist — check for discrepancies
            status_changed = repo.status != current.status
            eol_changed = repo.eol_date != current.eol_date
            if status_changed or eol_changed:
                change_type = "status_change" if status_changed else "eol_date_change"
                if status_changed and eol_changed:
                    change_type = "status_and_eol_change"
                results.append(
                    ModelComparisonEntry(
                        provider="bedrock",
                        model_name=current.model_name,
                        repo_status=repo.status,
                        current_status=current.status,
                        repo_eol_date=repo.eol_date,
                        current_eol_date=current.eol_date,
                        change_type=change_type,
                    )
                )

    # Check models only in repo (removed from authoritative source)
    for model_id, repo in repo_by_id.items():
        if model_id not in current_by_id:
            results.append(
                ModelComparisonEntry(
                    provider="bedrock",
                    model_name=repo.model_name,
                    repo_status=repo.status,
                    current_status="",
                    repo_eol_date=repo.eol_date,
                    current_eol_date=None,
                    change_type="removed",
                )
            )

    return results


def compare_model_pricing(
    repo_pricing: dict[str, dict], current_pricing: dict[str, dict]
) -> list[ModelComparisonEntry]:
    """Compare model pricing between repo and authoritative source.

    Both ``repo_pricing`` and ``current_pricing`` map model names to dicts
    of the form ``{"input": float, "output": float}``.

    Reports a discrepancy for each model where input or output prices differ.

    Args:
        repo_pricing: Pricing data from the repo file.
        current_pricing: Pricing data from the authoritative source.

    Returns:
        A list of :class:`ModelComparisonEntry` with
        ``change_type="pricing_change"`` for each model with differing prices.
    """
    results: list[ModelComparisonEntry] = []

    all_models = set(repo_pricing.keys()) | set(current_pricing.keys())

    for model_name in sorted(all_models):
        repo_prices = repo_pricing.get(model_name)
        current_prices = current_pricing.get(model_name)

        if repo_prices is None or current_prices is None:
            # Model only in one side — not a pricing change, skip
            continue

        repo_input = repo_prices.get("input")
        repo_output = repo_prices.get("output")
        current_input = current_prices.get("input")
        current_output = current_prices.get("output")

        if repo_input != current_input or repo_output != current_output:
            results.append(
                ModelComparisonEntry(
                    provider="",
                    model_name=model_name,
                    repo_pricing=repo_prices,
                    current_pricing=current_prices,
                    change_type="pricing_change",
                )
            )

    return results
