# Hermes Agent Governance With Inquiryon AMP

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This plugin connects a Hermes agent to [Inquiryon AMP](https://github.com/inquiryon) for
governance, available in two editions — **AMP SaaS** (fully managed, `https://amp.inquiryon.com`)
or **AMP Community Edition** (self-hosted, open source). Same governance pattern either way:
intercept a tool call, ask AMP for a decision, block or allow.

Once connected, AMP can:

- log prompts and governed tool activity
- evaluate tool calls against an AMP `eval-policy`
- require HITL (Human-in-the-Loop) approval when the policy says so
- block governed actions when AMP rejects them
- add date-aware routing context for time-sensitive prompts so Hermes is more likely to use `web_search` for current information
- notify the active Hermes channel when AMP is waiting for human review and when that review is resolved

## What this plugin governs

Hermes tools are normalized into AMP policy vocabulary like this:

- `terminal` → `exec/exec`
- `read_file` → `read/read`
- `search_files` → `read/search`
- `write_file` → `write/write`
- `patch` → `write/edit`
- `web_search` → `exec/web_search`
- `web_extract` → `exec/web_extract`

Other tools are not governed unless `AMP_GOVERN_UNLISTED_TOOLS=true` (SaaS plugin, off by default). With that set, every other tool except the plugin's own three tools (`amp_evaluate_research_plan`, `amp_load_research_topics`, `amp_governance_summary`) is sent as `<tool name>/invoke` with its arguments (secret-looking keys and values redacted, capped at 2,000 chars). Each extra tool call costs one AMP request against the monthly call limit.

## Prerequisites

1. **Install Hermes.** Follow [Hermes Agent's official installation instructions](https://github.com/NousResearch/hermes-agent) — this plugin assumes Hermes is already installed, not the other way around.
2. **Verify Hermes works on its own first.** Send it a test prompt and confirm it responds normally, before adding AMP governance into the mix.
3. **Set up an AMP account**, in one of two ways:
   - **AMP SaaS** at `https://amp.inquiryon.com` — recommended for quick setup, no infrastructure to run yourself.
   - **AMP Community Edition** — self-hosted in your own environment. See [`amp-community`](https://github.com/inquiryon/amp-community).

### Using AMP SaaS?

Follow **[`AMP_SaaS_Hermes.md`](AMP_SaaS_Hermes.md)**.

### Using AMP Community Edition (OSS)?

Follow **[`AMP_OSS_Hermes.md`](AMP_OSS_Hermes.md)**.
