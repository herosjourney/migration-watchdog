# Migration Plugin Watchdog

Automated quality assurance for the GCP-to-AWS migration plugin at [awslabs/startups/migrate](https://github.com/awslabs/startups/tree/main/migrate). Keeps migration guidance accurate and progressively more automated by running four AI agents on every pull request and weekly schedule.

---

## The problem

The migration plugin helps startups move workloads from GCP to AWS. It generates migration plans, cost estimates, and setup scripts by reading a set of reference files — pricing tables, model ID mappings, region availability lists, service guidance, and runbook templates.

These reference files go stale. AWS releases new regions, deprecates Bedrock models, changes Fargate pricing, renames services, and moves features from preview to GA. When the reference files aren't updated, the plugin generates plans with wrong region counts, deprecated model IDs, outdated prices, and guidance that no longer reflects how AWS works. Startups following those plans hit unexpected errors, pay wrong amounts, or miss better options.

At the same time, the runbooks tell users to do things manually — "go to the Service Quotas console and request an increase" — when AWS CLI commands exist that could do the same thing automatically. The generated setup scripts partially cover these steps but often stop short.

**The Watchdog solves both problems:**
1. **Currency** — automatically detects when factual claims in the reference files diverge from authoritative AWS sources, and flags them with suggested fixes before they reach users.
2. **Automation** — identifies manual steps in the runbooks that could be scripted, checks whether the generated scripts already cover them, and recommends CLI equivalents where safe to automate.

---

## What it does

**On every PR that touches a reference file in [awslabs/startups/migrate](https://github.com/awslabs/startups/tree/main/migrate):**
- Extracts verifiable factual claims and checks them against live AWS sources
- Identifies manual console steps and checks whether generated scripts already cover them
- Posts a comment on the PR with findings and a merge checklist
- Fails the PR check if any `correctness` severity finding is detected

**Weekly on the main branch:**
- Runs the same audits across all reference files
- Persists findings to DynamoDB
- Opens GitHub Issues for `outdated` findings
- Approved findings trigger cross-fork PRs with suggested fixes

---

## The four agents

### 1. Analysis Agent
**Model:** Claude Opus 4.7 (Amazon Bedrock)

The primary LLM agent. Reads all reference files and compares them against live AWS documentation, pricing pages, Gemini/OpenAI model data, and blog posts. Identifies outdated guidance, deprecated models, missing content, and pricing drift. Produces findings with source citations.

### 2. Review Agent
**Model:** Amazon Nova 2 Lite (Amazon Bedrock)

Quality-checks medium and high-risk findings from the Analysis Agent. Uses a different model family to catch hallucinations and unsupported claims. Nova 2 Lite is used here based on Amazon's Nova 2 technical report showing it surpasses Premier on multi-step problem-solving at 7x lower cost and up to 5x faster inference. Can confirm, correct, or dispute each finding. Disputed findings pass through with both models' reasoning attached for human review.

### 3. Currency Auditor *(new)*
**Model:** Claude Opus 4.7 (Amazon Bedrock) for extraction; deterministic verification

Focuses on facts that go stale: region counts, model IDs, prices, EOL dates, service names. Extracts every verifiable factual claim from reference files, then checks each one deterministically against AWS sources. No judgment calls — a region count is either right or wrong.

**Example finding:**
```json
{
  "claim": "AgentCore Runtime is available in 9 regions",
  "actual": "15 regions as of May 2026",
  "severity": "correctness",
  "suggested_fix": "Update region count from 9 to 15 and list the 6 new regions",
  "verification_source": "https://docs.aws.amazon.com/bedrock-agentcore/..."
}
```

### 4. Automation Auditor *(new)*
**Model:** Claude Opus 4.7 (Amazon Bedrock) for extraction; static analysis for gap detection

Reads migration guides looking for instructions that tell users to do things manually — "go to the Service Quotas console and request an increase." For each one, checks whether there's an AWS CLI command that does the same thing, and whether the generated setup scripts already use it.

**Example finding:**
```json
{
  "action": "Navigate to Service Quotas console and request a quota increase",
  "gap_type": "partial_gap",
  "cli_equivalent": "aws service-quotas request-service-quota-increase --service-code {service_code} --quota-code {quota_code} --desired-value {desired_value}",
  "gap_detail": "Script checks current quota but does not call request-service-quota-increase",
  "automate_recommended": "yes",
  "human_gate": "required"
}
```

---

## Sample PR comment

When a PR touches a reference file, the Watchdog posts a comment like this:

```
🔍 Watchdog Audit Report

Run ID: `a3f1c2d4` | PR: #1234 | Files audited: 3

PR #1234 Audit Summary:
  ❌ 1 correctness finding with disputed review
  ✅ 0 human_gate=required findings pending
  ⚠️ 2 outdated findings (informational)
  ℹ️ 4 automation gaps identified

### 💱 Currency Drift Findings

#### 🔴 [CORRECTNESS] Region count stale: ai-migration-guardrails.md
- File: `references/shared/ai-migration-guardrails.md`
- Claim type: `region_count`
- Claim: AgentCore Runtime is available in 9 regions
- Suggested fix: Update region count from 9 to 15 and list the 6 new regions

### 🤖 Automation Gap Findings

#### 🟡 [PARTIAL_GAP] Quota increase not automated: setup-bedrock.sh
- File: `references/phases/generate/generate.md`
- Action: Navigate to Service Quotas console and request a quota increase
- CLI equivalent: `aws service-quotas request-service-quota-increase ...`
- ⚠️ Requires human approval before execution
```

---

## Dashboard

The Watchdog includes a FastAPI dashboard for reviewing findings and approving/declining suggested fixes.

**Features:**
- Findings table with inline badges (severity, gap type, human gate, review status)
- Expandable detail panels for currency and automation findings
- PR sticky banner when viewing a run-scoped audit
- Merge checklist showing blocking vs informational findings
- Approve → triggers `PRCreator` to open a cross-fork PR with the fix
- Decline → records a 2-month dismissal cooldown

**Accessing the dashboard:**
The dashboard is deployed as an AWS Lambda function behind API Gateway. Set `DASHBOARD_API_KEY` for authentication.

---

## Architecture

```
GitHub PR / weekly cron
        │
        ▼
  pr-audit.yml / scheduled scan
        │
        ▼
  run_scan() in main.py
        │
        ├── RepoScanner (fetch reference files via GitHub API)
        ├── SourceFetcher (AWS docs, pricing, Gemini/OpenAI data)
        │
        ├── Analysis Agent (Claude Opus 4.7) ──────────────────┐
        ├── Currency Auditor (Claude Opus 4.7 + deterministic) ─┤
        └── Automation Auditor (Claude Opus 4.7 + static)      │
                                                                │
        ┌───────────────────────────────────────────────────────┘
        │
        ├── Deduplication (pluggable key functions per category)
        ├── Review Agent (Nova Premier) — medium/high currency only
        ├── DynamoDB persistence
        ├── PR comment posting (PR-triggered runs)
        └── PRCreator (approved findings → cross-fork PRs)
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `strands-agents` | LLM agent framework (Strands) |
| `boto3` / `botocore` | AWS SDK (DynamoDB, S3, Bedrock, Pricing API) |
| `httpx` | Async HTTP (GitHub API, source fetching) |
| `fastapi` + `mangum` | Dashboard web app (Lambda) |
| `python-dateutil` | Date parsing for EOL dates |
| `PyJWT` | GitHub App authentication |

**Python:** 3.10+ required (`match/case` syntax in `currency_auditor.py`)

**AWS services used:**
- Amazon Bedrock (Claude Opus 4.7, Nova Premier)
- DynamoDB (findings persistence)
- S3 (oversized payload storage)
- AWS Pricing API (price verification)
- KMS (optional: dashboard private key encryption)

---

## Repository layout

```
migration_watchdog/          ← Python package
├── main.py                  ← Pipeline orchestrator
├── currency_auditor.py      ← Currency Auditor (claim extraction + verification)
├── automation_auditor.py    ← Automation Auditor (gap detection + judgment)
├── analysis_agent.py        ← Analysis Agent (Strands-based LLM agent)
├── review_agent.py          ← Review Agent (quality check)
├── dashboard.py             ← FastAPI dashboard
├── models.py                ← Core dataclasses (Finding, ScanRun, etc.)
├── findings_repository.py   ← DynamoDB data access layer
├── finding_deduplicator.py  ← Deduplication logic
├── payload_store.py         ← S3 overflow for large payloads
├── alias_table.py           ← Model ID alias resolution
├── alias_table.json         ← Bundled alias table
├── cli_command_index.json   ← Bundled CLI command index
├── pr_creator.py            ← Cross-fork PR creation
├── repo_scanner.py          ← GitHub API file fetcher
├── source_fetcher.py        ← AWS docs + pricing fetcher
├── pricing_comparator.py    ← Price tolerance comparison
├── tests/                   ← Test suite (88 tests)
└── DEPLOYMENT.md            ← Infrastructure setup guide
pyproject.toml
setup.py
.github/
└── workflows/
    └── ci.yml               ← Watchdog repo CI
sample-agent-skills-for-aws-migration/
└── .github/workflows/
    └── pr-audit.yml         ← Content repo PR audit workflow
```

---

## Quick start

See [DEPLOYMENT.md](migration_watchdog/DEPLOYMENT.md) for full infrastructure setup. The short version:

```bash
# Install
pip install -e .

# Run tests
pytest migration_watchdog/tests/test_currency_fixtures.py migration_watchdog/tests/test_automation_fixtures.py
pytest migration_watchdog/tests/test_main_pipeline.py
pytest migration_watchdog/tests/test_pr_smoke.py

# Required environment variables for a real run
export GITHUB_APP_ID=...
export GITHUB_APP_PRIVATE_KEY=...
export GITHUB_INSTALLATION_ID=...
export DYNAMODB_TABLE=watchdog-findings
export AWS_REGION=us-east-1
python -m migration_watchdog.main
```
