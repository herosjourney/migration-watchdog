"""Execution path tracer for the Scenario Simulation Agent.

Traces execution paths through the migration plugin's state machine for a
given startup persona, producing a structured ``ExecutionTrace`` that records
every routing decision made during the simulated run.

The path tracer does NOT execute the plugin's LLM instructions. Instead it
reads routing conditions from the reference files in the repo snapshot and
evaluates them deterministically against the persona's profile. Each
conditional branch — file loads, phase gates, auto-skip rules — is captured
as a ``DecisionPoint`` so coverage gaps can be identified precisely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from migration_watchdog.models import RepoContent
from migration_watchdog.persona_library import Persona


@dataclass
class DecisionPoint:
    """A single routing decision made during path simulation.

    Captures the phase in which the decision occurred, the human-readable
    condition that was evaluated, the outcome of that evaluation, and whether
    the file referenced by the condition was present in the repo snapshot.
    """

    phase: str
    condition: str          # the routing condition evaluated (human-readable)
    outcome: str            # "taken" | "skipped" | "missing_file"
    file_referenced: str | None  # the file path referenced by this decision
    file_exists: bool       # whether the file exists in the repo snapshot


@dataclass
class ExecutionTrace:
    """Full record of a simulated execution through the migration plugin.

    Produced by ``PathTracer.trace()`` for a single persona. Contains every
    phase that ran, every file loaded or found missing, every question fired
    or auto-skipped, and the ordered list of routing decisions that led to
    those outcomes.
    """

    persona_id: str
    phases_executed: list[str] = field(default_factory=list)
    files_loaded: list[str] = field(default_factory=list)
    files_missing: list[str] = field(default_factory=list)
    questions_fired: list[str] = field(default_factory=list)
    questions_skipped: list[str] = field(default_factory=list)
    design_refs_loaded: list[str] = field(default_factory=list)
    artifacts_expected: list[str] = field(default_factory=list)
    decision_points: list[DecisionPoint] = field(default_factory=list)


class PathTracer:
    """Simulates execution of the migration plugin for a given startup persona.

    Reads routing conditions from the plugin's reference files in the repo
    snapshot and evaluates them deterministically against the persona's
    profile. Produces an ``ExecutionTrace`` capturing every routing decision.
    """

    def trace(self, persona: Persona, repo_content: RepoContent) -> ExecutionTrace:
        """Trace the full execution path for a persona through the plugin.

        Orchestrates the discover → clarify → design simulation phases and
        records every routing decision into a complete ``ExecutionTrace``.

        Args:
            persona: The startup persona to simulate.
            repo_content: Snapshot of the repo at scan time.

        Returns:
            An ``ExecutionTrace`` recording all phases, files, questions, and
            routing decisions encountered during the simulated run.
        """
        trace = ExecutionTrace(persona_id=persona.id)

        # ------------------------------------------------------------------ #
        # Phase 1: Discover                                                    #
        # ------------------------------------------------------------------ #
        trace.phases_executed.append("discover")
        discover_outputs = self._simulate_discover(persona)
        for artifact_name in discover_outputs:
            trace.files_loaded.append(artifact_name)

        # ------------------------------------------------------------------ #
        # Phase 2: Clarify                                                     #
        # ------------------------------------------------------------------ #
        trace.phases_executed.append("clarify")
        questions_fired, questions_skipped = self._trace_clarify(
            persona, discover_outputs
        )
        trace.questions_fired.extend(questions_fired)
        trace.questions_skipped.extend(questions_skipped)

        # ------------------------------------------------------------------ #
        # Phase 3: Design                                                      #
        # ------------------------------------------------------------------ #
        trace.phases_executed.append("design")
        # Simplified: real preferences come from clarify answers; here we
        # pass an empty dict so routing is driven purely by the persona profile.
        preferences: dict = {}
        design_refs = self._trace_design(persona, preferences)
        trace.design_refs_loaded.extend(design_refs)

        # Check which design-ref files exist in the repo snapshot.
        design_refs_prefix = (
            "migrate/plugins/migration-to-aws/skills/gcp-to-aws/"
            "references/design-refs/"
        )
        phases_design_prefix = (
            "migrate/plugins/migration-to-aws/skills/gcp-to-aws/"
            "references/phases/design/"
        )

        for ref in design_refs:
            design_refs_path = design_refs_prefix + ref
            phases_design_path = phases_design_prefix + ref

            exists_in_design_refs = self._check_file_exists(
                design_refs_path, repo_content
            )
            exists_in_phases = self._check_file_exists(
                phases_design_path, repo_content
            )
            exists = exists_in_design_refs or exists_in_phases

            if exists:
                trace.files_loaded.append(ref)
            else:
                trace.files_missing.append(ref)

            trace.decision_points.append(
                DecisionPoint(
                    phase="design",
                    condition=f"ai_source or infra requires {ref}",
                    outcome="taken" if exists else "missing_file",
                    file_referenced=ref,
                    file_exists=exists,
                )
            )

        # ------------------------------------------------------------------ #
        # Expected artifacts based on routes taken                            #
        # ------------------------------------------------------------------ #
        if "gcp-resource-inventory.json" in discover_outputs:
            trace.artifacts_expected.extend(
                ["aws-design.json", "terraform/", "MIGRATION_GUIDE.md"]
            )
        if "ai-workload-profile.json" in discover_outputs:
            trace.artifacts_expected.extend(["aws-design-ai.json", "ai-migration/"])
        if "billing-profile.json" in discover_outputs:
            trace.artifacts_expected.append("aws-design-billing.json")

        return trace

    def _simulate_discover(self, persona: Persona) -> dict:
        """Simulate the discover phase and return the profile outputs it would produce.

        Produces the same artifact files that the real Discover phase would
        generate for this startup, based on the persona's infrastructure and
        AI stack profile.  Only artifacts whose triggering conditions are met
        are included in the returned dict.

        Artifacts produced:

        * ``gcp-resource-inventory.json`` — when Terraform is present OR any
          GCP services are listed.
        * ``gcp-resource-clusters.json`` — alongside the inventory, whenever
          infrastructure is present.
        * ``ai-workload-profile.json`` — when an AI provider is set.
        * ``billing-profile.json`` — when billing data is available.

        Args:
            persona: The startup persona being simulated.

        Returns:
            A dict mapping output artifact filenames to their simulated content
            dicts, e.g. ``{"ai-workload-profile.json": {...}}``.
        """
        artifacts: dict = {}

        infra = persona.infrastructure
        ai = persona.ai_stack
        agentic = persona.agentic_profile

        has_terraform = infra.has_terraform
        has_gcp_services = len(infra.gcp_services) > 0
        infra_present = has_terraform or has_gcp_services

        # --- GCP resource inventory + clusters ---
        if infra_present:
            source = "terraform" if has_terraform else "inferred"
            artifacts["gcp-resource-inventory.json"] = {
                "resources": [
                    {"type": svc, "name": svc} for svc in infra.gcp_services
                ],
                "source": source,
            }
            artifacts["gcp-resource-clusters.json"] = {
                "clusters": [],
                "source": source,
            }

        # --- AI workload profile ---
        if ai.provider is not None:
            provider = ai.provider
            # "both" is the canonical value for multi-model personas; pass
            # through any other provider value unchanged.
            ai_source = provider  # "openai", "gemini", "anthropic", "both", "other"

            artifacts["ai-workload-profile.json"] = {
                "summary": {
                    "ai_source": ai_source,
                    "inferred_from_iac": False,
                },
                "models": [
                    {"model_id": m, "provider": provider}
                    for m in ai.models
                ],
                "integration": {
                    "pattern": ai.integration_pattern or "direct_sdk",
                    "primary_sdk": provider,
                    "frameworks": ai.frameworks,
                    "gateway_type": ai.gateway_type,
                },
                "agentic_profile": {
                    "is_agentic": agentic.is_agentic,
                    "framework": agentic.framework,
                    "orchestration_pattern": agentic.orchestration_pattern,
                },
            }

        # --- Billing profile ---
        if infra.has_billing_data:
            artifacts["billing-profile.json"] = {
                "source": "billing_csv",
                "services": [],
            }

        return artifacts

    def _trace_clarify(
        self, persona: Persona, discover_outputs: dict
    ) -> tuple[list[str], list[str]]:
        """Trace the clarify phase and return questions fired and auto-skipped.

        Encodes the plugin's clarify routing logic as Python conditions based
        on the discover outputs and the persona's AI workload profile.

        Routing rules:
        - ``gcp-resource-inventory.json`` only → clarify-global (Q1–Q7)
        - ``ai-workload-profile.json`` only → clarify-ai-only (Q1-AI–Q3-AI,
          minus any auto-skipped)
        - Both → clarify-global+clarify-ai (Q1–Q7 + Q14-AI–Q16-AI, minus
          any auto-skipped AI questions)
        - ``billing-profile.json`` only → clarify-global billing path (Q1–Q3)

        Auto-skip logic for AI questions:
        - Framework question skipped when ``integration.frameworks`` is
          non-empty in the AI workload profile.
        - Model question skipped when ``models`` list is non-empty.
        - Gateway question skipped when ``integration.gateway_type`` is not
          null.

        Args:
            persona: The startup persona being simulated.
            discover_outputs: Simulated outputs from the discover phase.

        Returns:
            A tuple ``(questions_fired, questions_skipped)`` where each
            element is an ordered list of question identifiers.
        """
        has_inventory = "gcp-resource-inventory.json" in discover_outputs
        has_ai_profile = "ai-workload-profile.json" in discover_outputs
        has_billing = "billing-profile.json" in discover_outputs

        # Determine which AI questions can be auto-skipped based on the
        # content of the AI workload profile artifact.
        ai_profile = discover_outputs.get("ai-workload-profile.json", {})
        ai_integration = ai_profile.get("integration", {})
        ai_models = ai_profile.get("models", [])

        frameworks_known = bool(ai_integration.get("frameworks"))
        models_known = bool(ai_models)
        gateway_known = ai_integration.get("gateway_type") is not None

        questions_fired: list[str] = []
        questions_skipped: list[str] = []

        if has_inventory and not has_ai_profile:
            # Pure infrastructure path → clarify-global (Q1–Q7)
            questions_fired = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]

        elif has_ai_profile and not has_inventory:
            # AI-only path → clarify-ai-only (Q1-AI–Q3-AI)
            candidate_ai_questions = ["Q1-AI", "Q2-AI", "Q3-AI"]
            # Q1-AI = framework question, Q2-AI = model question,
            # Q3-AI = gateway question
            skip_map = {
                "Q1-AI": frameworks_known,
                "Q2-AI": models_known,
                "Q3-AI": gateway_known,
            }
            for q in candidate_ai_questions:
                if skip_map.get(q, False):
                    questions_skipped.append(q)
                else:
                    questions_fired.append(q)

        elif has_inventory and has_ai_profile:
            # Combined path → clarify-global + clarify-ai
            # Global questions always fire; AI supplement questions may skip.
            questions_fired = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]
            candidate_ai_questions = ["Q14-AI", "Q15-AI", "Q16-AI"]
            # Q14-AI = framework, Q15-AI = model, Q16-AI = gateway
            skip_map = {
                "Q14-AI": frameworks_known,
                "Q15-AI": models_known,
                "Q16-AI": gateway_known,
            }
            for q in candidate_ai_questions:
                if skip_map.get(q, False):
                    questions_skipped.append(q)
                else:
                    questions_fired.append(q)

        elif has_billing:
            # Billing-only path → abbreviated clarify-global (Q1–Q3)
            questions_fired = ["Q1", "Q2", "Q3"]

        return questions_fired, questions_skipped

    def _trace_design(self, persona: Persona, preferences: dict) -> list[str]:
        """Trace the design phase and return the list of design-ref files that would load.

        Encodes the plugin's design routing logic as Python conditions based
        on the persona's infrastructure and AI stack profile.

        Routing rules:
        - Infrastructure present → load compute/database/storage/networking/
          messaging refs based on detected GCP services.
        - AI provider set → load the provider-specific AI migration ref.
        - Agentic → load the agentic migration ref (default: strands approach).
        - Billing only → load design-billing.md.

        Args:
            persona: The startup persona being simulated.
            preferences: Collected preferences from the clarify phase
                (currently unused; routing is driven by the persona profile).

        Returns:
            Ordered list of design-ref filenames (without path prefix) that
            would be loaded for this persona's stack.
        """
        design_refs: list[str] = []

        infra = persona.infrastructure
        ai = persona.ai_stack
        agentic = persona.agentic_profile

        has_terraform = infra.has_terraform
        has_gcp_services = len(infra.gcp_services) > 0
        infra_present = has_terraform or has_gcp_services

        # --- Infrastructure design refs ---
        if infra_present:
            services = set(infra.gcp_services)

            compute_services = {
                "cloud_run", "gke", "gce", "cloud_functions",
            }
            database_services = {
                "cloud_sql", "cloud_sql_postgresql", "cloud_sql_mysql",
                "cloud_spanner", "firestore",
            }
            storage_services = {"cloud_storage"}
            networking_services = {"vpc", "load_balancer"}
            messaging_services = {"pubsub"}

            has_compute = bool(services & compute_services)
            has_database = bool(services & database_services)
            has_storage = bool(services & storage_services)
            has_networking = bool(services & networking_services)
            has_messaging = bool(services & messaging_services)

            if has_compute:
                design_refs.append("compute.md")
            if has_database:
                design_refs.append("database.md")
            if has_storage:
                design_refs.append("storage.md")
            if has_networking:
                design_refs.append("networking.md")
            if has_messaging:
                design_refs.append("messaging.md")

            # Default fallback: if GCP services present but none matched
            # specific categories, load compute.md and database.md.
            if has_gcp_services and not any(
                [has_compute, has_database, has_storage, has_networking, has_messaging]
            ):
                design_refs.append("compute.md")
                design_refs.append("database.md")

        # --- AI provider design refs ---
        if ai.provider is not None:
            provider = ai.provider
            if provider in ("openai", "both"):
                design_refs.append("ai-openai-to-bedrock.md")
            if provider in ("gemini", "both"):
                design_refs.append("ai-gemini-to-bedrock.md")
            if provider == "anthropic":
                design_refs.append("ai-anthropic-to-bedrock.md")
            if provider == "other":
                design_refs.append("ai.md")

        # --- Agentic design refs ---
        if agentic.is_agentic:
            # Default migration approach for agentic personas is "strands"
            # unless explicitly overridden (personas don't carry this field).
            migration_approach = preferences.get("migration_approach", "strands")
            if migration_approach == "strands":
                design_refs.append("design-ref-agentic-to-agentcore.md")
            elif migration_approach == "harness":
                design_refs.append("design-ref-harness.md")
            elif migration_approach == "retarget":
                design_refs.append("retarget-gotchas.md")
            else:
                # Unknown approach — default to strands
                design_refs.append("design-ref-agentic-to-agentcore.md")

        # --- Billing-only design ref ---
        if infra.has_billing_data and not infra_present and ai.provider is None:
            design_refs.append("design-billing.md")

        return design_refs

    def _check_file_exists(self, file_path: str, repo_content: RepoContent) -> bool:
        """Check whether a file path exists in the repo snapshot.

        Args:
            file_path: The path to check, relative to the repo root.
            repo_content: Snapshot of the repo at scan time.

        Returns:
            ``True`` if the file is present in ``repo_content.files``,
            ``False`` otherwise.
        """
        return file_path in repo_content.files
