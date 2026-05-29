"""Persona library for the Scenario Simulation Agent.

Defines dataclasses representing startup technical profiles used to simulate
execution through the migration plugin's 6-phase state machine. Each persona
captures a startup's infrastructure, AI stack, agentic profile, expected
migration path, and known gaps fixed by PRs.

The PersonaLibrary class loads personas from a YAML file so new personas can
be added without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class InfrastructureProfile:
    """GCP infrastructure profile for a startup persona."""

    gcp_services: list[str] = field(default_factory=list)
    has_terraform: bool = False
    has_billing_data: bool = False


@dataclass
class AIStack:
    """AI/LLM stack profile for a startup persona."""

    provider: str | None = None
    models: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    integration_pattern: str | None = None
    gateway_type: str | None = None


@dataclass
class AgenticProfile:
    """Agentic characteristics of a startup persona."""

    is_agentic: bool = False
    framework: str | None = None
    tools: list[str] = field(default_factory=list)
    orchestration_pattern: str | None = None


@dataclass
class ExpectedPath:
    """Expected execution path through the migration plugin for a persona."""

    discover_outputs: list[str] = field(default_factory=list)
    clarify_route: str = ""
    design_route: str = ""
    design_refs: list[str] = field(default_factory=list)


@dataclass
class Persona:
    """A startup persona representing a real distribution of migration plugin users.

    Each persona captures the startup's technical profile and the expected path
    through the plugin's state machine, enabling deterministic simulation and
    gap detection.
    """

    id: str
    description: str
    infrastructure: InfrastructureProfile
    ai_stack: AIStack
    agentic_profile: AgenticProfile
    expected_path: ExpectedPath
    known_gaps_fixed_by_prs: list[str] = field(default_factory=list)


class PersonaLibrary:
    """Loads and manages startup personas from a YAML file.

    Personas are stored as structured data so new ones can be added without
    code changes. The library supports lookup by ID and retrieval of all personas.
    """

    def __init__(self) -> None:
        self._personas: list[Persona] | None = None

    def load(self, path: str) -> list[Persona]:
        """Load personas from a YAML file at the given path.

        Validates that each entry has a non-empty ``id`` and ``description``,
        and that ``expected_path.clarify_route`` and
        ``expected_path.design_route`` are non-empty strings.

        Args:
            path: Filesystem path to the personas YAML file.

        Returns:
            List of Persona instances parsed from the file.

        Raises:
            ValueError: If a required field is missing or empty.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw_entries = yaml.safe_load(fh)

        if not isinstance(raw_entries, list):
            raise ValueError(
                f"personas YAML must be a list of entries, got {type(raw_entries).__name__}"
            )

        personas: list[Persona] = []
        for i, entry in enumerate(raw_entries):
            # --- Required top-level fields ---
            persona_id = entry.get("id", "")
            if not isinstance(persona_id, str) or not persona_id.strip():
                raise ValueError(
                    f"Persona at index {i} is missing a non-empty 'id' field"
                )

            description = entry.get("description", "")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"Persona '{persona_id}' is missing a non-empty 'description' field"
                )

            # --- Infrastructure ---
            infra_raw = entry.get("infrastructure") or {}
            infrastructure = InfrastructureProfile(
                gcp_services=infra_raw.get("gcp_services") or [],
                has_terraform=bool(infra_raw.get("has_terraform", False)),
                has_billing_data=bool(infra_raw.get("has_billing_data", False)),
            )

            # --- AI stack ---
            ai_raw = entry.get("ai_stack") or {}
            ai_stack = AIStack(
                provider=ai_raw.get("provider") or None,
                models=ai_raw.get("models") or [],
                frameworks=ai_raw.get("frameworks") or [],
                integration_pattern=ai_raw.get("integration_pattern") or None,
                gateway_type=ai_raw.get("gateway_type") or None,
            )

            # --- Agentic profile ---
            agentic_raw = entry.get("agentic_profile") or {}
            agentic_profile = AgenticProfile(
                is_agentic=bool(agentic_raw.get("is_agentic", False)),
                framework=agentic_raw.get("framework") or None,
                tools=agentic_raw.get("tools") or [],
                orchestration_pattern=agentic_raw.get("orchestration_pattern") or None,
            )

            # --- Expected path ---
            path_raw = entry.get("expected_path") or {}

            clarify_route = path_raw.get("clarify_route", "")
            if not isinstance(clarify_route, str) or not clarify_route.strip():
                raise ValueError(
                    f"Persona '{persona_id}' is missing a non-empty "
                    "'expected_path.clarify_route' field"
                )

            design_route = path_raw.get("design_route", "")
            if not isinstance(design_route, str) or not design_route.strip():
                raise ValueError(
                    f"Persona '{persona_id}' is missing a non-empty "
                    "'expected_path.design_route' field"
                )

            expected_path = ExpectedPath(
                discover_outputs=path_raw.get("discover_outputs") or [],
                clarify_route=clarify_route.strip(),
                design_route=design_route.strip(),
                design_refs=path_raw.get("design_refs") or [],
            )

            # --- Assemble persona ---
            persona = Persona(
                id=persona_id.strip(),
                description=description.strip(),
                infrastructure=infrastructure,
                ai_stack=ai_stack,
                agentic_profile=agentic_profile,
                expected_path=expected_path,
                known_gaps_fixed_by_prs=entry.get("known_gaps_fixed_by_prs") or [],
            )
            personas.append(persona)

        self._personas = personas
        return personas

    def get_by_id(self, persona_id: str) -> Persona | None:
        """Return the persona with the given ID, or None if not found.

        Args:
            persona_id: The unique identifier of the persona to retrieve.

        Returns:
            The matching Persona, or None if no persona has that ID.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
        """
        if self._personas is None:
            raise RuntimeError("PersonaLibrary not loaded — call load() first")
        for persona in self._personas:
            if persona.id == persona_id:
                return persona
        return None

    def get_all(self) -> list[Persona]:
        """Return all loaded personas.

        Returns:
            A copy of the list of all Persona instances in the library.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
        """
        if self._personas is None:
            raise RuntimeError("PersonaLibrary not loaded — call load() first")
        return list(self._personas)
