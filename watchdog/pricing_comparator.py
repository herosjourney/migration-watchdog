"""Pricing cache parsing and comparison logic.

Pure functions for parsing the pricing-cache.md markdown format,
comparing cached prices against current prices with tolerance thresholds,
and checking date staleness. Used as Strands tools in the analysis agent.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from watchdog.models import PricingEntry, PricingValidationResult

# Sections that use infrastructure tolerance (10%)
_INFRA_SECTIONS = frozenset(
    {"compute", "database", "storage", "networking", "supporting services", "analytics"}
)

# Sections that use AI model tolerance (25%)
_AI_SECTIONS = frozenset({"bedrock models", "source provider pricing"})


def _normalise_key(text: str) -> str:
    """Lowercase, strip, collapse whitespace, remove non-alphanumeric chars except spaces."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_dollar_value(cell: str) -> float | None:
    """Extract a numeric dollar value from a table cell.

    Handles formats like '$0.04048', '0.0104', '$0.10', '7.58',
    and ignores non-numeric cells like 'Free tier' or '—'.
    """
    cell = cell.strip()
    if not cell or cell == "—" or cell == "N/A":
        return None
    # Remove dollar sign and commas
    cleaned = cell.replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _classify_section_tolerance(section_name: str) -> float:
    """Return tolerance percentage based on the top-level section name."""
    normalised = _normalise_key(section_name)
    # Check if it matches any infrastructure section
    for infra in _INFRA_SECTIONS:
        if infra in normalised:
            return 10.0
    # Check if it matches any AI section
    for ai in _AI_SECTIONS:
        if ai in normalised:
            return 25.0
    # Default to infrastructure tolerance
    return 10.0


def _parse_last_updated(markdown: str) -> str:
    """Extract the 'Last updated' date string from the markdown header."""
    match = re.search(r"\*\*Last updated:\*\*\s*(\d{4}-\d{2}-\d{2})", markdown)
    if match:
        return match.group(1)
    return ""


def _parse_standard_table(
    rows: list[str],
    service: str,
    last_updated: str,
    tolerance: float,
) -> list[PricingEntry]:
    """Parse a standard 2-column table (Metric | Rate) into PricingEntry objects."""
    entries: list[PricingEntry] = []
    for row in rows:
        cells = [c.strip() for c in row.split("|")]
        # Filter out empty strings from leading/trailing pipes
        cells = [c for c in cells if c]
        if len(cells) < 2:
            continue
        metric = cells[0].strip()
        rate_str = cells[1].strip()
        value = _extract_dollar_value(rate_str)
        if value is not None and metric:
            entries.append(
                PricingEntry(
                    service=_normalise_key(service),
                    metric=_normalise_key(metric),
                    cached_value=value,
                    cached_last_updated=last_updated,
                    tolerance_pct=tolerance,
                )
            )
    return entries


def _parse_multi_column_table(
    rows: list[str],
    headers: list[str],
    service: str,
    last_updated: str,
    tolerance: float,
) -> list[PricingEntry]:
    """Parse a multi-column table (e.g. Instance | $/hour | $/month) into PricingEntry objects.

    Each numeric column produces a separate PricingEntry with the metric
    being '{row_label} {column_header}'.
    """
    entries: list[PricingEntry] = []
    for row in rows:
        cells = [c.strip() for c in row.split("|")]
        cells = [c for c in cells if c]
        if len(cells) < 2:
            continue
        row_label = cells[0].strip()
        for i, cell_val in enumerate(cells[1:], start=1):
            if i < len(headers):
                col_header = headers[i]
            else:
                col_header = f"col{i}"
            value = _extract_dollar_value(cell_val)
            if value is not None and row_label:
                metric = f"{_normalise_key(row_label)} {_normalise_key(col_header)}"
                entries.append(
                    PricingEntry(
                        service=_normalise_key(service),
                        metric=metric,
                        cached_value=value,
                        cached_last_updated=last_updated,
                        tolerance_pct=tolerance,
                    )
                )
    return entries


def _parse_bedrock_model_table(
    rows: list[str],
    last_updated: str,
    tolerance: float,
) -> list[PricingEntry]:
    """Parse the Bedrock Models multi-provider quick reference table.

    Expected columns: Model | Model ID | Provider | Input $/1M | Output $/1M | Context | Tier | Status
    Also handles simpler model tables with columns: Model | Input $/1M | Output $/1M
    """
    entries: list[PricingEntry] = []
    for row in rows:
        cells = [c.strip() for c in row.split("|")]
        cells = [c for c in cells if c]
        if len(cells) < 3:
            continue

        # Try to detect which format this is by checking if we have enough columns
        # for the full format (8 columns) or a simpler format
        model_name = cells[0].strip()
        if not model_name:
            continue

        # Try to find Input $/1M and Output $/1M values
        input_val = None
        output_val = None

        if len(cells) >= 8:
            # Full format: Model | Model ID | Provider | Input $/1M | Output $/1M | Context | Tier | Status
            input_val = _extract_dollar_value(cells[3])
            output_val = _extract_dollar_value(cells[4])
        elif len(cells) >= 3:
            # Simpler format: Model | Input $/1M | Output $/1M
            # or: Model | Rate/Value | ...
            # Try the last two numeric-looking columns
            for i in range(1, len(cells)):
                val = _extract_dollar_value(cells[i])
                if val is not None:
                    if input_val is None:
                        input_val = val
                    elif output_val is None:
                        output_val = val

        service_key = _normalise_key(model_name)

        if input_val is not None:
            entries.append(
                PricingEntry(
                    service=service_key,
                    metric="input per 1m tokens",
                    cached_value=input_val,
                    cached_last_updated=last_updated,
                    tolerance_pct=tolerance,
                )
            )
        if output_val is not None:
            entries.append(
                PricingEntry(
                    service=service_key,
                    metric="output per 1m tokens",
                    cached_value=output_val,
                    cached_last_updated=last_updated,
                    tolerance_pct=tolerance,
                )
            )

    return entries


def parse_pricing_cache(markdown: str) -> list[PricingEntry]:
    """Parse the pricing-cache.md markdown format into a list of PricingEntry objects.

    The file has a header with ``**Last updated:** YYYY-MM-DD`` and sections
    (## Compute, ## Database, etc.) containing markdown tables. Each ### heading
    identifies a service. Tables may be 2-column (Metric | Rate) or multi-column
    (Instance | $/hour | $/month). The Bedrock Models section uses a different
    table format with Model | Model ID | Provider | Input $/1M | Output $/1M | ...

    Returns a list of PricingEntry objects with service, metric, cached_value,
    cached_last_updated, and tolerance_pct set based on the section type.
    """
    last_updated = _parse_last_updated(markdown)
    entries: list[PricingEntry] = []

    lines = markdown.split("\n")
    current_section = ""  # ## level heading (e.g., "Compute", "Database")
    current_service = ""  # ### level heading (e.g., "Fargate", "Lambda")
    current_tolerance = 10.0
    in_bedrock_section = False

    # Track table state
    in_table = False
    table_headers: list[str] = []
    table_rows: list[str] = []
    is_bedrock_model_table = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect ## section headings
        if line.startswith("## ") and not line.startswith("### "):
            # Flush any pending table
            if in_table and table_rows:
                entries.extend(
                    _flush_table(
                        table_rows,
                        table_headers,
                        current_service,
                        last_updated,
                        current_tolerance,
                        is_bedrock_model_table,
                    )
                )
            in_table = False
            table_rows = []
            table_headers = []
            is_bedrock_model_table = False

            section_text = line[3:].strip()
            # Remove trailing content after parentheses for matching
            section_base = re.sub(r"\s*\(.*\)$", "", section_text)
            current_section = section_base
            current_tolerance = _classify_section_tolerance(current_section)
            in_bedrock_section = "bedrock" in _normalise_key(current_section)
            current_service = ""
            i += 1
            continue

        # Detect ### service headings
        if line.startswith("### "):
            # Flush any pending table
            if in_table and table_rows:
                entries.extend(
                    _flush_table(
                        table_rows,
                        table_headers,
                        current_service,
                        last_updated,
                        current_tolerance,
                        is_bedrock_model_table,
                    )
                )
            in_table = False
            table_rows = []
            table_headers = []
            is_bedrock_model_table = False

            current_service = line[4:].strip()
            i += 1
            continue

        # Detect #### sub-headings (used in Bedrock section)
        if line.startswith("#### "):
            # Flush any pending table
            if in_table and table_rows:
                entries.extend(
                    _flush_table(
                        table_rows,
                        table_headers,
                        current_service,
                        last_updated,
                        current_tolerance,
                        is_bedrock_model_table,
                    )
                )
            in_table = False
            table_rows = []
            table_headers = []
            is_bedrock_model_table = False

            # Use the #### heading as a sub-service context
            sub_heading = line[5:].strip()
            if current_service:
                current_service = current_service  # keep the ### service
            else:
                current_service = sub_heading
            i += 1
            continue

        # Detect ##### sub-sub-headings
        if line.startswith("##### "):
            if in_table and table_rows:
                entries.extend(
                    _flush_table(
                        table_rows,
                        table_headers,
                        current_service,
                        last_updated,
                        current_tolerance,
                        is_bedrock_model_table,
                    )
                )
            in_table = False
            table_rows = []
            table_headers = []
            is_bedrock_model_table = False
            i += 1
            continue

        # Detect table header row
        if line.startswith("|") and not in_table:
            # This is a potential table header
            header_cells = [c.strip() for c in line.split("|")]
            header_cells = [c for c in header_cells if c]

            # Check if next line is a separator (|---|---|)
            if i + 1 < len(lines) and re.match(
                r"^\|[\s\-:|]+\|", lines[i + 1].strip()
            ):
                in_table = True
                table_headers = header_cells
                table_rows = []

                # Detect if this is a Bedrock model table
                header_lower = [h.lower() for h in header_cells]
                is_bedrock_model_table = in_bedrock_section and (
                    "model id" in header_lower
                    or "input $/1m" in header_lower
                    or "input" in " ".join(header_lower)
                )

                i += 2  # Skip header and separator
                continue

        # Collect table data rows
        if in_table and line.startswith("|"):
            table_rows.append(line)
            i += 1
            continue

        # End of table (non-table line while in_table)
        if in_table and not line.startswith("|"):
            if table_rows:
                entries.extend(
                    _flush_table(
                        table_rows,
                        table_headers,
                        current_service,
                        last_updated,
                        current_tolerance,
                        is_bedrock_model_table,
                    )
                )
            in_table = False
            table_rows = []
            table_headers = []
            is_bedrock_model_table = False

        i += 1

    # Flush any remaining table
    if in_table and table_rows:
        entries.extend(
            _flush_table(
                table_rows,
                table_headers,
                current_service,
                last_updated,
                current_tolerance,
                is_bedrock_model_table,
            )
        )

    return entries


def _flush_table(
    rows: list[str],
    headers: list[str],
    service: str,
    last_updated: str,
    tolerance: float,
    is_bedrock_model_table: bool,
) -> list[PricingEntry]:
    """Flush accumulated table rows into PricingEntry objects."""
    if is_bedrock_model_table:
        return _parse_bedrock_model_table(rows, last_updated, tolerance)

    if len(headers) == 2:
        return _parse_standard_table(rows, service, last_updated, tolerance)

    if len(headers) > 2:
        return _parse_multi_column_table(rows, headers, service, last_updated, tolerance)

    return []


def compare_pricing_entries(
    cached: list[PricingEntry],
    current_prices: dict[str, dict[str, float]],
) -> PricingValidationResult:
    """Compare cached pricing entries against current prices.

    Args:
        cached: List of PricingEntry objects from parse_pricing_cache.
        current_prices: Dict mapping ``{service_lower: {metric_lower: current_value}}``.

    Returns:
        PricingValidationResult with all entries updated with comparison data
        and staleness info.
    """
    result_entries: list[PricingEntry] = []
    stale = False

    if cached and cached[0].cached_last_updated:
        stale = check_date_staleness(cached[0].cached_last_updated)

    cache_last_updated = cached[0].cached_last_updated if cached else ""

    for entry in cached:
        service_key = entry.service.lower()
        metric_key = entry.metric.lower()

        service_prices = current_prices.get(service_key, {})
        current_value = service_prices.get(metric_key)

        updated = PricingEntry(
            service=entry.service,
            metric=entry.metric,
            cached_value=entry.cached_value,
            cached_last_updated=entry.cached_last_updated,
            tolerance_pct=entry.tolerance_pct,
        )

        if current_value is not None:
            updated.current_value = current_value
            if entry.cached_value != 0:
                diff_pct = abs(current_value - entry.cached_value) / entry.cached_value * 100
            else:
                diff_pct = 0.0 if current_value == 0 else 100.0
            updated.difference_pct = diff_pct
            updated.exceeds_tolerance = diff_pct > entry.tolerance_pct

        result_entries.append(updated)

    return PricingValidationResult(
        entries=result_entries,
        stale_date=stale,
        cache_last_updated=cache_last_updated,
        validation_timestamp=datetime.utcnow().isoformat(),
    )


def check_date_staleness(last_updated: str, max_age_days: int = 30) -> bool:
    """Check if a date string is older than max_age_days from today.

    Args:
        last_updated: Date string in YYYY-MM-DD format.
        max_age_days: Maximum age in days before the date is considered stale.

    Returns:
        True if the date is older than max_age_days, False otherwise.
    """
    try:
        updated_date = datetime.strptime(last_updated, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        # If we can't parse the date, consider it stale
        return True

    today = date.today()
    age = (today - updated_date).days
    return age > max_age_days
