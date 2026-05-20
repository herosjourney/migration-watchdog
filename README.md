# Migration Plugin Watchdog

Automated quality assurance for the GCP-to-AWS migration plugin at
[awslabs/startups/migrate](https://github.com/awslabs/startups/tree/main/migrate).
Keeps migration guidance accurate and progressively more automated by running
four AI agents on every pull request and on a weekly schedule.

---

## The problem

The migration plugin helps startups move workloads from GCP to AWS. It generates
migration plans, cost estimates, and setup scripts by reading a set of reference
files — pricing tables, model ID mappings, region availability lists, service
guidance, and runbook templates.

These reference files go stale. AWS releases new regions, deprecates Bedrock
models, changes Fargate pricing, renames services, closes services to new
customers, and moves features from preview to GA. When the reference files
aren't updated, the plugin generates plans with wrong region counts, deprecated
model IDs, outdated prices, and guidance that no longer reflects how AWS works.
Startups following those plans hit unexpected errors, pay wrong amounts, or miss
better options.

At the same time, the runbooks tell users to do things manually — "go to the
Service Quotas console and request an increase" — when AWS CLI commands exist
that could do the same thing automatically.

**The Watchdog solves both problems:**

1. **Currency** — detects when factual claims in the reference files diverge
   from authoritative AWS sources, and flags them with suggested fixes.
2. **Automation** — identifies manual steps in the runbooks that could be
   scripted, checks whether the generated scripts already cover them, and
   recommends CLI equivalents where safe to automate.

---

## How it works

### Two scan modes

**PR-triggered scan** — runs on every PR that touches a reference file in the
target repo. Audits only the changed files. Fast (5–10 minutes). Fails CI if
any `correctness` severity finding is detected, blocking the merge.

**Weekly scheduled scan** — runs every Sunday at 06:00 UTC against the full
plugin. Audits all reference files. Persists findings to DynamoDB. Provides
the complete picture of what's stale, what's missing, and what could be
automated.

### Pipeline

```
GitHub PR / weekly cron
        │
        ▼
  pr-audit.yml (content repo) / scheduled-scan.yml (watchdog repo)
        │
        ▼
  run_scan() in main.py
        │
        ├── RepoScanner — fetches reference files via GitHub App API
        ├── SourceFetcher — fetches AWS docs, pricing, Gemini/OpenAI data
        │
        ├── Analysis Agent ──────────────────────────────────────────────┐
        ├── Currency Auditor ────────────────────────────────────────────┤
        └── Automation Auditor ──────────────────────────────────────────┤
                                                                         │
        ┌────────────────────────────────────────────────────────────────┘
        │
        ├── Deduplication (per-category key functions)
        ├── Review Agent — quality-checks medium/high currency findings
        ├── DynamoDB persistence (scheduled runs + PR runs with credentials)
        ├── PR comment posting (PR-triggered runs)
        └── PRCreator — approved findings → cross-fork fix PRs
```

---

## The four agents

### 1. Analysis Agent

**Model:** Claude Opus 4.7 (`us.anthropic.claude-opus-4-7`) via Amazon Bedrock

**Purpose:** Broad comparison of reference files against authoritative sources.
Identifies outdated guidance, deprecated models, pricing drift, and missing
content. Also proactively checks whether recommended AWS services are still
available to new customers and whether newer alternatives exist.

**Tools available:**
- `compare_pricing` — compares pricing-cache.md against current AWS Pricing API data
- `compare_models` — compares AI model mapping guides against current Gemini/OpenAI/Bedrock data. Flags models present in the current provider catalog but missing from the plugin's mapping table as `new_content` findings — no hard-coding of model names required
- `compare_design_ref` — compares design-ref files against current AWS best practices
- `check_new_content_opportunities` — identifies gaps in agentic AI coverage, new AWS services, and framework compatibility issues (weekly scans only)
- `search_aws_docs` — live search of AWS documentation via the [AWS Documentation MCP Server](https://github.com/awslabs/mcp/tree/main/src/aws-documentation-mcp-server) (`awslabs.aws-documentation-mcp-server`). Searches `docs.aws.amazon.com` for current service status, feature availability, and best practices
- `search_gcp_docs` — verifies GCP-side claims in the migration plugin against current Google Cloud documentation
- `check_service_obsolescence` — proactively checks each recommended service for closure, newer alternatives, and updated best practices using the AWS Documentation MCP Server
- `web_search` — DuckDuckGo search for current information when pre-fetched data is insufficient, including framework compatibility issues not in official docs
- `create_finding` — creates a structured Finding with source citations

**Bedrock-specific checks (weekly scans):**
The Analysis Agent explicitly checks AI migration reference files for:
- Bedrock Mantle 10,000 RPM shared account limit (separate from per-model TPM quotas)
- `max_tokens` reservation trap — Bedrock deducts `max_tokens` from TPM quota at request start
- AgentCore Harness preview status and 4-region availability
- gpt-oss on Bedrock as a migration path option
- Bedrock Mantle Responses API capabilities and Assistants API shortcut
- Framework retarget compatibility gotchas (LangChain, LangGraph, CrewAI, AutoGen)

**Limitations:**
- In PR-triggered mode, only sees the changed files plus shared artifacts. Cannot
  make "missing content" judgments about the full plugin — those are suppressed
  automatically and reserved for weekly scans.
- Every finding must include at least one source URL. Findings without citations
  are rejected by the tool.
- Does not use training knowledge as a source of truth — only tool outputs.

---

### 2. Review Agent

**Model:** Amazon Nova 2 Lite (`us.amazon.nova-2-lite-v1:0`) via Amazon Bedrock

**Purpose:** Quality-checks medium and high-risk findings from the Analysis
Agent. Uses a different model family to catch hallucinations and unsupported
claims. Can confirm, correct, or dispute each finding. Disputed findings pass
through with both models' reasoning attached for human review.

Nova 2 Lite is used here based on Amazon's Nova 2 technical report showing it
surpasses Premier on multi-step problem-solving at 7× lower cost and up to 5×
faster inference.

**Limitations:**
- Only reviews `currency_drift` findings at MEDIUM or HIGH risk. Other
  categories pass through unreviewed.
- A disputed finding is not blocked — it reaches the dashboard with a
  "disputed" badge for human judgment.

---

### 3. Currency Auditor

**Model:** Claude Opus 4.7 for claim extraction; Nova 2 Lite for verification judgment

**Purpose:** Focuses on facts that go stale — region counts, model IDs, prices,
EOL dates, service names, feature availability. Extracts every verifiable
factual claim from reference files, searches AWS documentation for current
information, and uses an LLM to judge whether each claim is accurate, outdated,
or incorrect.

**Verification flow for each claim:**
1. Extract claim text and generate a verification query (Claude Opus 4.7)
2. Search AWS docs live via the AWS Documentation MCP Server (`AwsDocsSearcher`)
3. Fetch the top result page (8s timeout)
4. Pass docs content + claim to Nova 2 Lite for a binary judgment
5. Return `verified_accurate`, `finding`, or `unverified`

**Claim types handled:**
- `price` — compared against AWS Pricing API with tolerance thresholds
- `region_list` / `region_count` — set equality / numeric comparison
- `model_id` — exact match against Bedrock lifecycle after alias resolution
- `eol_date` — exact match on normalised YYYY-MM-DD
- `service_name` / `feature_availability` — live docs search + LLM judgment
- `quota_limit` / `api_name` / `other_factual` — live docs search + LLM judgment
- `service_limit` — throughput limits (e.g. RPM caps) — live docs search + LLM judgment
- `preview_status` — preview/GA status — live docs search + LLM judgment

**File coverage:**
The Currency Auditor scans all files in the `gcp-to-aws/` path, including AI migration
phase files (`clarify-ai.md`, `design-ai.md`, `ai-migration-guardrails.md`,
`generate-artifacts-ai.md`) and design reference files (`design-ref-harness.md`,
`design-ref-agentic-to-agentcore.md`). Files without a detectable migration context
are skipped with a warning.

**Severity levels:**
- `correctness` → HIGH risk — factually wrong (e.g., service closed to new customers)
- `outdated` → MEDIUM risk — stale but not necessarily wrong
- `policy_change` → LOW risk — renamed, moved to GA, or details changed
- `informational` → LOW risk — minor discrepancy

**Limitations:**
- AWS docs search uses direct HTTP calls to `docs.aws.amazon.com`. If the search
  returns no results, falls back to fetching known service pages by keyword.
  Some claims may return `unverified` if the relevant docs page isn't in the
  fallback map.
- Fuzzy claims ("most regions", "approximately") are skipped — they cannot be
  verified deterministically.
- Quarter-format dates (Q1/Q2 YYYY) are skipped as unverifiable.
- Each claim verification takes 5–15 seconds. A file with 20 claims takes
  roughly 2–5 minutes. Claims are verified sequentially (not in parallel).

---

### 4. Automation Auditor

**Model:** Claude Opus 4.7 for action extraction; static analysis for gap detection

**Purpose:** Reads migration guides looking for instructions that tell users to
do things manually. For each manual action, checks whether an AWS CLI command
exists that does the same thing, and whether the generated setup scripts already
use it.

**Gap types:**
- `full_gap` — no CLI equivalent exists in the generated scripts
- `partial_gap` — CLI equivalent exists but the script doesn't call it
- `no_gap` — script already handles this action

**Human gate:** Actions that require human approval before execution (e.g.,
quota increases, IAM policy changes) are flagged with `human_gate: required`.
These are informational — they don't fail CI.

**Limitations:**
- Relies on the bundled `cli_command_index.json` for CLI equivalents. Commands
  not in the index may be missed.
- Cannot verify whether a generated script is semantically correct — only
  whether it contains the expected CLI call.
- Automation findings never fail CI, regardless of severity.

---

### 5. Security Auditor

**Model:** Claude Opus 4.7 (`us.anthropic.claude-opus-4-7`) via Amazon Bedrock

**Purpose:** Reads migration plugin reference files and uses LLM reasoning to identify
security anti-patterns in the plugin's *instructions* — issues that would cause the
generated Terraform, shell scripts, or Python code to be insecure. Unlike static pattern
matching, the LLM understands context and can identify novel issues not in a predefined list.

**Checklist (grounded in AWS Startup Security Baseline ACCT.01-ACCT.13):**
- Open administrative ports (22/SSH, 3389/RDP, 5900/VNC) from 0.0.0.0/0
- Secrets stored in shell variables (exposed in `ps aux`, shell history, core dumps)
- Database passwords in Terraform variables (plaintext in state)
- Missing `deletion_protection` on Aurora/RDS clusters
- Secrets Manager rotation blocks without compliance gate (breaks non-compliance stacks)
- `max_tokens` set too high in Bedrock adapters (TPM quota burndown trap)
- Missing security baseline resources (GuardDuty, CloudTrail, S3 Public Access Block, IAM password policy)
- IMDSv2 not enforced on EC2 launch templates
- Any other security anti-patterns the LLM identifies

**Files scanned:** `generate-artifacts-*`, `networking.md`, `security.md`, `design-infra.md`, `baseline.md`

**Verification:** Uses `search_aws_security_docs` to verify current AWS security best practices
and find authoritative source URLs before flagging issues.

**Limitations:**
- LLM-based — slower than static analysis (~30-60s per file)
- May produce false positives on files with complex conditional logic
- Security findings never fail CI regardless of severity

---

| Severity | Risk | CI behavior |
|----------|------|-------------|
| `correctness` | HIGH | Fails PR check — blocks merge |
| `outdated` | MEDIUM | Opens GitHub Issue (weekly scans); informational on PRs |
| `policy_change` | LOW | Informational |
| `informational` | LOW | Informational |
| Automation gaps | varies | Never fail CI |

---

## Dashboard

The Watchdog includes a FastAPI dashboard for reviewing findings and approving
or declining suggested fixes.

**Features:**
- Findings table with severity badges, gap type, human gate, and review status
- Expandable detail panels for currency and automation findings
- Approve → queues a fix PR via PRCreator (merging is a separate step on GitHub)
- Decline → records a 2-month dismissal cooldown

**Run locally:**
```bash
DYNAMODB_TABLE=watchdog-findings AWS_REGION=us-east-1 \
  python3 -m uvicorn migration_watchdog.dashboard:app --reload
```
Then open http://localhost:8000.

---

## AWS Documentation MCP Server

The Watchdog uses the
[AWS Documentation MCP Server](https://github.com/awslabs/mcp/tree/main/src/aws-documentation-mcp-server)
(`awslabs.aws-documentation-mcp-server`) as its primary source of truth for
verifying claims against current AWS documentation.

### What it does

The MCP server provides two capabilities the Watchdog relies on:

1. **Search** — queries `docs.aws.amazon.com` for a given topic and returns
   relevant page titles, URLs, and excerpts. Used to find the authoritative
   page for a claim before fetching it.

2. **Fetch** — retrieves and extracts text content from a specific AWS docs
   page. Used to get the full context needed to verify a claim.

### How it's used

In GitHub Actions (CI), the Watchdog cannot run a local MCP server process.
Instead, the `AwsDocsSearcher` class in `source_fetcher.py` replicates the
same HTTP calls the MCP server makes — direct requests to
`docs.aws.amazon.com/search/doc-search.html` and page fetches — so the same
capability works in CI without any additional infrastructure.

The MCP server is used in two places:

**Currency Auditor (`currency_auditor.py`):**
- During claim extraction, the LLM calls `search_aws_docs` to self-verify
  claims before outputting them. For example, before flagging "App Runner
  supports new customers" as a claim, it searches for current App Runner
  availability and adjusts accordingly.
- During verification, `ClaimVerifier` fetches the top search result for each
  `service_name` and `feature_availability` claim, then uses Nova 2 Lite to
  judge whether the claim is accurate, outdated, or incorrect.

**Analysis Agent (`analysis_agent.py`):**
- `search_aws_docs` tool — the LLM can search AWS docs on demand during
  analysis to verify guidance or find current best practices.
- `check_service_obsolescence` tool — for every service the plugin recommends,
  searches for current availability status, newer alternatives, and updated
  best practices.

### Fallback behavior

If the AWS docs search returns no results (which can happen for some queries),
the `AwsDocsSearcher` falls back to a hardcoded map of known service pages
(App Runner, Fargate, Lambda, EKS, ECS, Bedrock, AgentCore, Harness, DynamoDB,
S3, RDS, Aurora, etc.). Claims that don't match any known service page return
`unverified` rather than a false finding.

### Running locally with the MCP server

If you want to use the actual MCP server locally (e.g., for development or
testing), install it via `uvx`:

```bash
uvx awslabs.aws-documentation-mcp-server@latest
```

The `AwsDocsSearcher` class can be replaced with direct MCP tool calls in a
local environment where the server is running.

---

## Known limitations

**PR scan sees limited context.** The PR-triggered scan only loads the changed
files plus a few shared artifacts. The Analysis Agent cannot make "missing
coverage" judgments in this mode — those findings are suppressed and reserved
for weekly scans.

**AWS docs search is best-effort.** The `AwsDocsSearcher` makes direct HTTP
calls to `docs.aws.amazon.com`. If the search endpoint returns no results (which
happens for some queries), it falls back to a hardcoded map of service pages.
Claims that don't match any known service page return `unverified` rather than
a finding.

**Sequential claim verification.** Claims are verified one at a time. A
reference file with many claims takes proportionally longer. The weekly scan
has a 60-minute timeout.

**No automatic fix application.** The Watchdog identifies problems and suggests
fixes, but does not apply them automatically. Approved findings trigger a PR
with the suggested change — a human must review and merge it.

**Refactoring agent is meta-level.** The Refactoring Agent assesses the overall
structure of the plugin codebase, not individual files. Its findings are always
HIGH risk and always require human judgment before acting on them.

**GitHub App required for scheduled scans.** The weekly scan uses the GitHub
App to fetch all reference files from the target repo. Without valid App
credentials, the scheduled scan falls back to an empty file set and produces
no findings.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `strands-agents` | LLM agent framework (Strands SDK) |
| `boto3` / `botocore` | AWS SDK — DynamoDB, S3, Bedrock, Pricing API |
| `httpx` | Async HTTP — GitHub API, AWS docs search, source fetching |
| `fastapi` + `mangum` | Dashboard web app (runs on Lambda or locally) |
| `python-dateutil` | Date parsing for EOL date normalization |
| `PyJWT` | GitHub App JWT authentication |
| `hypothesis` | Property-based testing |
| `pytest` + `pytest-asyncio` | Test runner |

**Python:** 3.10+ required (`match/case` syntax in `currency_auditor.py`)

**AWS services:**
- Amazon Bedrock — Claude Opus 4.7 (`us.anthropic.claude-opus-4-7`), Nova 2 Lite (`us.amazon.nova-2-lite-v1:0`)
- Amazon DynamoDB — findings and scan run persistence
- Amazon S3 — oversized payload storage
- AWS Pricing API — price verification
- IAM — OIDC role for GitHub Actions

---

## Repository layout

```
migration-watchdog/
├── migration_watchdog/          ← Python package
│   ├── main.py                  ← Pipeline orchestrator (run_scan)
│   ├── scan_config.py           ← ScanConfig dataclass — reads + validates all env vars at startup
│   ├── analysis_agent.py        ← Analysis Agent (Strands + Claude Opus 4.7)
│   ├── currency_auditor.py      ← Currency Auditor (claim extraction + verification)
│   ├── automation_auditor.py    ← Automation Auditor (gap detection)
│   ├── review_agent.py          ← Review Agent (Nova 2 Lite quality check)
│   ├── refactoring_agent.py     ← Refactoring Agent (structural assessment)
│   ├── pr_commenter.py          ← PR comment building and posting (extracted from main.py)
│   ├── dashboard.py             ← FastAPI dashboard
│   ├── models.py                ← Core dataclasses (Finding, ScanRun, etc.)
│   ├── findings_repository.py   ← DynamoDB data access layer
│   ├── finding_deduplicator.py  ← Deduplication logic
│   ├── utils.py                 ← Shared helpers (has_active_dismissal)
│   ├── payload_store.py         ← S3 overflow for large payloads
│   ├── source_fetcher.py        ← AWS docs + pricing fetcher + AwsDocsSearcher (async)
│   ├── repo_scanner.py          ← GitHub API file fetcher
│   ├── pr_creator.py            ← Cross-fork PR creation
│   ├── pricing_comparator.py    ← Price tolerance comparison
│   ├── model_comparator.py      ← Model lifecycle comparison
│   ├── alias_table.py           ← Model ID alias resolution
│   ├── alias_table.json         ← Bundled alias table
│   ├── cli_command_index.json   ← Bundled CLI command index
│   ├── retry.py                 ← Exponential backoff helper
│   └── tests/                   ← Test suite
│       ├── test_codebase_improvements.py  ← Property-based tests (hypothesis)
│       ├── test_currency_fixtures.py
│       ├── test_automation_fixtures.py
│       ├── test_main_pipeline.py
│       └── test_pr_smoke.py
├── pyproject.toml
├── setup.py
└── .github/
    └── workflows/
        ├── ci.yml               ← Test suite (runs on push/PR to main)
        └── scheduled-scan.yml   ← Weekly full scan (Sundays 06:00 UTC)
```

The `pr-audit.yml` workflow lives in the **target content repo**
(`herosjourney/startups` or `awslabs/startups`), not here.

---

## Setup

See [DEPLOYMENT.md](DEPLOYMENT.md) for full infrastructure setup. The short version:

```bash
# Install
pip install -e .

# Run tests
pytest tests/test_codebase_improvements.py
pytest tests/test_currency_fixtures.py tests/test_automation_fixtures.py
pytest tests/test_main_pipeline.py
pytest tests/test_pr_smoke.py

# Required environment variables for a real run
export WATCHDOG_TARGET_OWNER=aws-samples   # required — no default
export WATCHDOG_TARGET_REPO=sample-agent-skills-for-aws-migration  # required — no default
export GITHUB_APP_ID=...
export GITHUB_APP_PRIVATE_KEY=...
export GITHUB_INSTALLATION_ID=...
export DYNAMODB_TABLE=watchdog-findings
export AWS_REGION=us-east-1
python -m migration_watchdog.main
```

**Required secrets in the watchdog repo (`herosjourney/migration-watchdog`):**

| Name | Description |
|------|-------------|
| `WATCHDOG_APP_ID` | GitHub App ID |
| `WATCHDOG_APP_PRIVATE_KEY` | GitHub App private key (PEM) |
| `WATCHDOG_AWS_ROLE_ARN` | IAM role ARN for OIDC |
| `WATCHDOG_INSTALLATION_ID` | GitHub App installation ID (for scheduled scans) |

**Required variables:**

| Name | Default | Description |
|------|---------|-------------|
| `WATCHDOG_TARGET_OWNER` | — | **Required.** Target repo owner (e.g. `awslabs`) |
| `WATCHDOG_TARGET_REPO` | — | **Required.** Target repo name (e.g. `startups`) |
| `DYNAMODB_TABLE` | `watchdog-findings` | DynamoDB table name |
| `WATCHDOG_PAYLOAD_BUCKET` | — | S3 bucket for oversized payloads |
| `AWS_REGION` | `us-east-1` | AWS region |
| `SEVERITY_THRESHOLD` | `outdated` | PR comment filter: `outdated` or `low` |
| `TAVILY_API_KEY` | — | Optional. Tavily API key for web search. Falls back to DuckDuckGo if absent. |

**Required secrets in the content repo (`herosjourney/startups`):**

Same `WATCHDOG_APP_ID`, `WATCHDOG_APP_PRIVATE_KEY`, `WATCHDOG_AWS_ROLE_ARN` plus
the same variables above.

---

## Trigger the weekly scan manually

```bash
gh workflow run scheduled-scan.yml --repo herosjourney/migration-watchdog
```

Results appear in the dashboard after the run completes (~20–45 minutes for
the full plugin).
