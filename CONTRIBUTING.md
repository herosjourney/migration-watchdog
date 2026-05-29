# Contributing to Migration Watchdog

Thank you for contributing to the Migration Watchdog project. This document covers the key workflows for keeping the watchdog accurate and up to date.

---

## Scenario Simulation Agent — Persona Maintenance

The Scenario Simulation Agent uses a library of startup personas (`migration_watchdog/personas.yaml`) to regression-test the migration plugin's routing and coverage logic. When a PR fixes a routing or coverage gap, a corresponding persona must be added to `personas.yaml` in the same PR.

### When to add a persona

Add a persona whenever:
- A PR fixes a routing error (e.g., wrong design-ref file loaded for a startup profile)
- A PR adds a new design-ref file that covers a previously unhandled stack
- A PR adds a new clarify question that collects information needed for correct routing
- A real user reports incorrect guidance for their stack

### How to add a persona

1. Open `migration_watchdog/personas.yaml`
2. Add a new entry following the existing schema (see existing personas for examples)
3. Set `known_gaps_fixed_by_prs` to include the PR number (e.g., `["PR#30"]`)
4. Set `expected_path.design_refs` to the files that should now be loaded for this persona
5. Run `python -m pytest tests/test_persona_library.py -v` to confirm the new persona validates

### Schema reference

Each persona requires:
- `id`: unique kebab-case identifier
- `description`: one-sentence description of the startup profile
- `infrastructure`: GCP services, has_terraform, has_billing_data
- `ai_stack`: provider, models, frameworks, integration_pattern, gateway_type
- `agentic_profile`: is_agentic, framework, tools, orchestration_pattern
- `expected_path`: discover_outputs, clarify_route, design_route, design_refs
- `known_gaps_fixed_by_prs`: list of PR numbers that added coverage for this persona

Valid `clarify_route` and `design_route` tokens: `clarify-global`, `clarify-ai`, `clarify-ai-only`, `design-infra`, `design-ai`, `design-billing`. Compound routes use `+` (e.g., `clarify-global+clarify-ai`).
