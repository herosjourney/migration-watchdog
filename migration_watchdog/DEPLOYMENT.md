# Deployment Guide

## Overview

The Migration Plugin Watchdog runs as two GitHub Actions workflows:

1. **`pr-audit.yml`** — lives in the upstream content repo (`aws-samples/sample-agent-skills-for-aws-migration`). Triggers on PRs that touch Reference_Files. Runs the Currency and Automation Auditors against changed files.

2. **`ci.yml`** — lives in this watchdog repo. Runs on push/PR to `main`. Runs the full test suite.

---

## Required GitHub Secrets and Variables

### In the upstream content repo (`aws-samples/sample-agent-skills-for-aws-migration`)

| Name | Type | Description |
|------|------|-------------|
| `WATCHDOG_APP_ID` | Secret | GitHub App ID for the watchdog bot |
| `WATCHDOG_APP_PRIVATE_KEY` | Secret | GitHub App private key (PEM format) |
| `WATCHDOG_PAT` | Secret | Personal Access Token fallback (when App is unavailable) |
| `WATCHDOG_AWS_ROLE_ARN` | Secret | IAM role ARN for OIDC authentication (DynamoDB + S3) |
| `WATCHDOG_REPO` | Variable | Watchdog repo in `owner/repo` format (e.g. `aws-samples/migration-watchdog`) |
| `WATCHDOG_TARGET_OWNER` | Variable | Target repo owner (default: `aws-samples`) |
| `WATCHDOG_TARGET_REPO` | Variable | Target repo name (default: `sample-agent-skills-for-aws-migration`) |
| `DYNAMODB_TABLE` | Variable | DynamoDB table name for findings persistence |
| `WATCHDOG_PAYLOAD_BUCKET` | Variable | S3 bucket name for oversized payload storage |
| `AWS_REGION` | Variable | AWS region (default: `us-east-1`) |

### In the watchdog repo (this repo)

| Name | Type | Description |
|------|------|-------------|
| `WATCHDOG_PACKAGE_SUBDIR` | Variable | Inner package directory name (same as above; used by `ci.yml`) |

---

## Required AWS Infrastructure

### IAM Role (`WATCHDOG_AWS_ROLE_ARN`)

The role needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:UpdateItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/watchdog-findings*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject"],
      "Resource": "arn:aws:s3:::YOUR_PAYLOAD_BUCKET/payloads/*"
    }
  ]
}
```

Trust policy must allow OIDC from `token.actions.githubusercontent.com` for the upstream content repo.

### DynamoDB Table

Create a table named `watchdog-findings` (or set `DYNAMODB_TABLE`) with:
- Partition key: `PK` (String)
- Sort key: `SK` (String)
- GSI1: partition key `status`, sort key `scan_timestamp`
- GSI2: partition key `risk_level`, sort key `scan_timestamp`

### S3 Bucket

Create a bucket for oversized payloads (set `WATCHDOG_PAYLOAD_BUCKET`). The dashboard IAM role needs `s3:GetObject` on `payloads/*`.

---

## Required AWS Bedrock Access

The LLM extraction steps (`ClaimExtractor`, `ActionExtractor`) use Amazon Bedrock with Claude Opus 4.7 (`us.anthropic.claude-opus-4-7`). The IAM role must also have:

```json
{
  "Effect": "Allow",
  "Action": ["bedrock:InvokeModel"],
  "Resource": "arn:aws:bedrock:us-east-1::foundation-model/us.anthropic.claude-opus-4-7"
}
```

---

## GitHub App Setup

1. Create a GitHub App in your org with permissions:
   - `pull_requests: write`
   - `checks: write`
   - `issues: write`
   - `contents: read`

2. Install the App on both the watchdog repo and the upstream content repo.

3. Store the App ID as `WATCHDOG_APP_ID` and the private key as `WATCHDOG_APP_PRIVATE_KEY`.

---

## Fork PR Behavior

Fork PRs receive only the default `github.token` with restricted permissions:
- No `WATCHDOG_APP_*` secrets available
- No `WATCHDOG_PAT` secret available
- AWS credentials step uses `continue-on-error: true` → DynamoDB/S3 persistence is skipped
- Findings are written to `audit-report.json` artifact and PR comment only
- CI policy still runs and can fail the PR on `correctness` findings

---

## Local Development

```bash
# From the watchdog repo root — migration_watchdog/ is the package directory
pip install -e .

# Verify imports work
python -c "import migration_watchdog; from migration_watchdog.payload_store import PayloadStore; print('OK')"

# Run tests in the correct order (stub isolation — see pytest.ini for details)
pytest migration_watchdog/tests/test_currency_fixtures.py migration_watchdog/tests/test_automation_fixtures.py
pytest migration_watchdog/tests/test_main_pipeline.py
pytest migration_watchdog/tests/test_pr_smoke.py

# Or all at once with process isolation (requires pytest-forked)
pip install pytest-forked
pytest migration_watchdog/tests/ --forked
```

## Repository Layout

```
migration-watchdog/           ← repo root
├── migration_watchdog/       ← Python package (flat layout, no subdirectory)
│   ├── __init__.py
│   ├── main.py
│   ├── currency_auditor.py
│   ├── automation_auditor.py
│   ├── payload_store.py
│   ├── alias_table.py
│   ├── alias_table.json
│   ├── cli_command_index.json
│   ├── dashboard.py
│   ├── models.py
│   ├── ...
│   └── tests/
│       ├── test_currency_fixtures.py
│       ├── test_automation_fixtures.py
│       ├── test_main_pipeline.py
│       └── test_pr_smoke.py
├── pyproject.toml
├── setup.py
├── DEPLOYMENT.md
└── .github/
    └── workflows/
        └── ci.yml
```
