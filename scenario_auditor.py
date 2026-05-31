"""Scenario Auditor — orchestrator for the Scenario Simulation Agent.

Wires together PersonaLibrary, PathTracer, CoverageAssessor, and GapClassifier
to simulate startup user journeys through the migration plugin and identify
routing and coverage gaps.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from migration_watchdog.coverage_assessor import CoverageAssessor
from migration_watchdog.gap_classifier import GapClassifier
from migration_watchdog.models import Finding, RepoContent
from migration_watchdog.path_tracer import PathTracer
from migration_watchdog.persona_library import PersonaLibrary

logger = logging.getLogger(__name__)

# Default path to personas.yaml — located in the migration_watchdog sub-package
_DEFAULT_PERSONAS_PATH = Path(__file__).parent / "migration_watchdog" / "personas.yaml"


async def run_scenario_audit(
    repo_content: RepoContent,
    run_id: str,
    persona_library_path: str | None = None,
    changed_files: list[str] | None = None,
) -> list[Finding]:
    """Run the scenario simulation audit over all personas.

    For each persona:
    1. PathTracer traces the execution path
    2. CoverageAssessor assesses each design-ref file
    3. GapClassifier converts results to Findings

    On PR-triggered runs, only personas whose expected_path.design_refs
    overlap with changed_files are re-assessed (optimization).

    Args:
        repo_content: Snapshot of the repo at scan time.
        run_id: Identifier for the current scan run.
        persona_library_path: Path to personas.yaml. Defaults to the bundled file.
        changed_files: List of changed file paths (PR-triggered runs only).
            When provided, only personas with overlapping design_refs are assessed.

    Returns:
        Deduplicated list of Finding objects.
    """
    # Task 5.4: read confidence threshold from env
    confidence_threshold = float(os.environ.get("SCENARIO_CONFIDENCE_THRESHOLD", "0.7"))

    personas_path = persona_library_path or str(_DEFAULT_PERSONAS_PATH)

    # Load personas
    library = PersonaLibrary()
    personas = library.load(personas_path)
    logger.info("Scenario audit: loaded %d personas from %s", len(personas), personas_path)

    # Task 5.2: PR-triggered optimization
    if changed_files:
        changed_set = set(changed_files)
        personas = [
            p for p in personas
            if any(ref in " ".join(changed_set) for ref in p.expected_path.design_refs)
               or any(cf for cf in changed_set if any(ref in cf for ref in p.expected_path.design_refs))
        ]
        logger.info(
            "Scenario audit: PR optimization — %d personas selected based on %d changed files",
            len(personas), len(changed_files)
        )

    tracer = PathTracer()
    assessor = CoverageAssessor()
    classifier = GapClassifier(run_id=run_id, confidence_threshold=confidence_threshold)

    all_results = []
    all_traces = []
    traces_completed = 0
    assessments_made = 0

    for persona in personas:
        # Trace execution path
        trace = tracer.trace(persona, repo_content)
        all_traces.append(trace)
        traces_completed += 1

        # Assess coverage
        results = await assessor.assess(persona, trace, repo_content)
        all_results.extend(results)
        assessments_made += len(results)

        logger.debug(
            "Scenario audit: persona=%s traces=%d assessments=%d",
            persona.id, traces_completed, len(results)
        )

    # Classify and deduplicate
    findings = classifier.classify(all_results, all_traces)

    # Task 5.5: log summary
    suppressed = assessments_made - len([r for r in all_results if r.coverage not in ("adequate", "unverified")])
    logger.info(
        "Scenario audit completed: %d findings from %d traces, %d assessments "
        "(%d suppressed by confidence threshold)",
        len(findings), traces_completed, assessments_made, max(0, suppressed)
    )

    return findings
