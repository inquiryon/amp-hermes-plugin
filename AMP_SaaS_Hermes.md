# Hermes Agent Governance With Inquiryon AMP — SaaS Edition

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

You're here because you're connecting Hermes to **Inquiryon AMP's SaaS
service** (`https://amp.inquiryon.com`). If you meant to self-host AMP
instead, go back to [`README.md`](README.md) and follow the OSS branch —
this page doesn't apply there.

This plugin (also referred to as the AMP-Hermes Plugin, or AHP) connects
a Hermes agent to AMP so AMP can:

- log prompts and governed tool activity
- evaluate tool calls against an AMP `eval-policy`
- require HITL (Human-in-the-Loop) approval when the policy says so
- block governed actions when AMP rejects them
- add date-aware routing context for time-sensitive prompts so Hermes is more likely to use `web_search` for current information
- notify the active Hermes channel when AMP is waiting for human review and when that review is resolved

If you already run Hermes, you can have this running in a few minutes.

## What You'll Experience

- ✅ Connect Hermes to AMP
- ✅ Experience runtime governance
- ✅ Observe policy evaluation
- ✅ Approve a HITL request
- ✅ View execution audit logs
- ✅ (Optional) Govern LLM token and budget usage

```
            User
              │
              ▼
      Hermes AI Agent
              │
              ▼
      AMP-Hermes Plugin (AHP)
              │
              ▼
      AMP Governance Platform
      ┌────────┼────────┐
      ▼        ▼        ▼
   Policies   HITL   Audit Logs
```

## What this plugin governs

Hermes tools are normalized into AMP policy vocabulary like this:

- `terminal` → `exec/exec`
- `read_file` → `read/read`
- `search_files` → `read/search`
- `write_file` → `write/write`
- `patch` → `write/edit`
- `web_search` → `exec/web_search`

## Additional behavior

Besides tool governance, the plugin also improves two parts of the Hermes user experience:

- for prompts that appear time-sensitive or explicitly ask for live information, the plugin injects current-date context and tells Hermes to use `web_search` instead of answering from memory
- when AMP triggers HITL, the plugin uses Hermes `send_message` to notify the active channel that the action is waiting for human review, then posts a follow-up when the reviewer approves, rejects, or cancels the request, or it times out. A *modified* approval blocks the original call (the modified version is not applied automatically yet) and tells the user to re-submit.

## Prerequisites

Before installing this plugin, make sure you already have:

- Hermes installed and running on your machine — see [`README.md`](README.md)'s Prerequisites if you haven't done this yet
- a working Hermes channel or chat surface, such as Slack
- an AMP SaaS account at `https://amp.inquiryon.com`

## Quick Start

**1. Register a Hermes Remote Agent in AMP** — in the AMP UI:

> **Tip:** Use the **AMP Quick Start** wizard to auto-complete Steps 1-3
> below (recommended).

If you'd rather do it manually, or aren't using AMP UI: log in to AMP →
**Agent Launchpad** → register a new **remote agent** for Hermes
(mention "Hermes" in the agent name or description so AMP AI can
recommend a relevant policy) → check **Allow auto-start** → **Register**.

**2. Create (or use) an AMP API Key** — open the side menu → **Settings**
to copy an existing API key or create a new one.

**3. Attach a governance policy** — on the **Agent Launchpad** page, open
your registered agent. On the side menu, under **Governance** → **Write
New Policy** → **Rule-based Policy** (also called eval-policy) → use the
AI icon to generate a starter Hermes policy → activate it. This step
should define rules for the normalized tool/action pairs listed above.

You now have `AMP_API_KEY` and `AMP_AGENT_NAME`. Find `AMP_ORG_ID` and
`AMP_USERNAME` under **My Profile**. Remember them for Step 5.

**4. Install the plugin** — on the machine where your Hermes agent runs,
run one command:

```bash
hermes plugins install inquiryon/amp-hermes-plugin/plugins/saas --enable
```

This clones the repo, installs the `plugins/saas/` subdirectory as the
`amp-governance` plugin, and enables it in one step.

<details>
<summary>Alternative: manual copy or symlink</summary>

Use manual copy for a normal local install from a downloaded repo
snapshot:

```bash
mkdir -p ~/.hermes/plugins/amp-governance

cp -r /path/to/amp-hermes-plugin/plugins/saas/* ~/.hermes/plugins/amp-governance/

hermes plugins enable amp-governance
```

Use symlink instead if you're developing the plugin locally — see
"Local Development" at the bottom of this page.

</details>

**5. Configure environment variables** — add your AMP credentials to
`~/.hermes/.env` (or wherever your Hermes install directory is). Copy
`plugins/saas/.env.example` as a starting point:

```env
AMP_BACKEND_URL=https://amp.inquiryon.com
AMP_USERNAME=your_name@email_domain.com
AMP_API_KEY=amp_k_...
AMP_ORG_ID=O-0011-AB202605010...
AMP_AGENT_NAME=your-hermes-agent-10c8
```

That's all you need — LLM token/cost observation
(`AMP_LLM_GOVERNANCE_MODE=observe`) is **on by default** in this
reference implementation. Set `AMP_LLM_GOVERNANCE_ENABLED=false` if you
only want tool-call governance and no per-LLM-call logging to AMP. Other
optional settings — HITL timeouts, notifications, `enforce` mode — are
covered in [Budget Enforcement](#budget-enforcement) in the Technical
Reference below.

Notes:

- `AGENT_NAME` is accepted as a fallback alias for `AMP_AGENT_NAME`
- `AMP_FAIL_CLOSED=true` is recommended for governance-focused deployments
- if you use a custom Hermes home, set `HERMES_HOME` and place `.env` under that directory

**6. Restart Hermes:**

```bash
hermes gateway restart
```

If the gateway is not running yet, start it the way you normally run Hermes.

**7. Run your first governed command** — send this through your Hermes
channel:

- Log in to AMP and open **Agent Log** from the side menu. Then send
  this query on your Hermes channel: `Can you give me recent news on SS
  frauds?` — a plain governed tool call.

You should see transparency logs appear on the AMP **Agent Log** page,
and this query will trigger HITL approval before Hermes executes it. See
"Verify Everything Works" below to confirm it's genuinely governed, not
just running.

## Verify Everything Works

Check these places while (or after) running the command above:

**Hermes**

- The command pauses when approval is required, with a message like:
  - `[AMP] AMP is waiting for a human reviewer to approve "web_search" before continuing. This action is paused pending review.`
- Hermes receives an approval/rejection notification once a reviewer acts:
  - `[AMP] AMP reviewer approved "web_search". Continuing now.`
  - `[AMP] AMP reviewer rejected "web_search". ...`
- For time-sensitive prompts, Hermes uses `web_search` instead of answering only from model memory.

**AMP**

- The agent's activity log timeline shows entries like: session started, user prompt logged, policy check for a normalized tool/action, policy decision, and HITL requested (if approval is required).
- The workitems page shows a pending HITL item, if one was triggered. Click the workitem ID, or go to the Agent Worktray page, to approve the request.
- If LLM observation is enabled (default), the agent log also shows per-call and per-session usage:
  - `LLM call #1 | model=claude-opus-4-8 | tokens=1450 | cost_usd=0.001234 | cost_status=estimated`
  - `Execution summary | execution_id=... | llm_calls=4 | total_tokens=5820 | total_cost_usd=0.004932 | cost_status=estimated`
  - Click `LLM` to review the model's reasoning behind the call, if your model supports it.

🎉 **Congratulations! Your Hermes agent is now governed by AMP.**

## Advanced Example: Govern LLM Budget Usage

Your Hermes agent is already under runtime governance — every tool call
(and every LLM call, in `observe` mode) is logged and policy-evaluated.
This example shows an *additional* governance capability: cost-governed
agent behavior, where AMP evaluates and approves a projected token usage
budget before the agent spends it.

The research-agent skill demonstrates:

- Research planning
- Token estimation
- Cost estimation
- Budget approval
- Runtime budget enforcement

**Install the research-agent skills** — one command each:

```bash
hermes skills install https://raw.githubusercontent.com/inquiryon/amp-hermes-plugin/main/plugins/saas/skills-pointer/amp-research/SKILL.md
hermes skills install https://raw.githubusercontent.com/inquiryon/amp-hermes-plugin/main/plugins/saas/skills-pointer/amp-research-topic/SKILL.md
```

Then copy `~/.hermes/plugins/amp-governance/examples/research_topics.yaml`
to `~/.hermes/research_topics.yaml` and edit the topics list based on your
research interests. This is only needed for "run my research topics"
mode. You can simply run **ad hoc topics** named on the spot (e.g.
"research the US market"), which don't need this file at all.

**Try it** — send through your Hermes channel:

- `Research the US market today.` — the full governed research
  workflow: a cost-estimated plan, AMP approval, governed research, and
  a real cost/usage summary in the report.
- `Run my research topics.` — evaluates the cost of running your topics
  in `research_topics.yaml`. If the estimate is higher than the budget
  specified in your AMP governance policy, it requests HITL approval;
  otherwise it runs the research directly. The report ends with the
  total cost.

See [Research Example](#research-example) in the Technical Reference for
how the plan-approval interface and skills fit together.

---

# Technical Reference

The sections below document the technical implementation of the AHP
plugin — for reference only.

## Configuration

Core environment variables, gathered here for quick lookup (see the relevant section below for full details on each):

| Variable | Purpose | Default |
|---|---|---|
| `AMP_BACKEND_URL` | AMP instance URL | — |
| `AMP_USERNAME` | AMP account username | — |
| `AMP_API_KEY` | AMP API key | — |
| `AMP_ORG_ID` | AMP organization ID | — |
| `AMP_AGENT_NAME` | Registered Hermes agent name in AMP | — |
| `AMP_LLM_GOVERNANCE_ENABLED` | Enable/disable LLM observation & enforcement | `true` |
| `AMP_LLM_GOVERNANCE_MODE` | `observe` (accumulate only) or `enforce` (pre-call policy eval + block) | `observe` |
| `AMP_LLM_GOVERNANCE_FAIL_CLOSED` | Block LLM calls if AMP is unreachable | `false` |
| `AMP_LLM_GOVERNANCE_INCLUDE_SUBAGENTS` | Track LLM calls from subagents launched by this session | `true` |
| `AMP_NOTIFICATIONS_ENABLED` | Send HITL notifications to the originating chat channel | `true` |
| `AMP_HITL_TIMEOUT_MINUTES` | HITL approval timeout. The real window is also capped by Hermes (see [Approval window](#approval-window-hermes-time-limits)) | `10` |
| `AMP_HITL_POLL_INTERVAL_SECONDS` | Poll interval while waiting on a HITL decision (minimum 1) | `3` |
| `AMP_FAIL_CLOSED` | Block governed tools when AMP is unreachable (network error, timeout, 5xx). Anything AMP *answers* that is not an allow always blocks, whatever this says: HTTP 4xx, an unreadable or non-JSON body, an unknown status. An exhausted approval window always blocks too | `true` |
| `AMP_ACTION_ID_CORRELATION` | Send the Hermes `tool_call_id` as `action_id` on the HITL request and decision poll, so AMP matches the decision to the exact call. Turn on only once your AMP HITL service stores `action_id` on the workitem | `false` |
| `AMP_ACTION_ID_WATCHDOG_SECONDS` | With `AMP_ACTION_ID_CORRELATION=true`: block when AMP has not created a review for the call after this many seconds | `60` |
| `plugins.hook_callback_timeout` (Hermes `config.yaml`, not `.env`) | Hermes ≥ v2026.9 abandons a `pre_tool_call` hook after this many seconds. HITL reviews need `600` (the max). `0` disables the cap | `30` |
| `timeouts.tools.sequential_call` / `timeouts.tools.concurrent_batch` (Hermes `config.yaml`) | Hermes's per-call / per-batch tool deadline. It also counts the time a call waits for review. Set both to `660` for a 10-minute window | `420` |
| `HERMES_AGENT_TIMEOUT` | Raise if HITL reviews take more than 30 minutes (e.g. `7200`) | — |
| `HERMES_SESSION_PLATFORM` | Platform adapter used by the notification bridge (e.g. `cli`) | — |

## Approval window (Hermes time limits)

A governed tool call waits for its reviewer inside Hermes's `pre_tool_call` hook. Three limits apply, and the **shortest one wins**:

1. `AMP_HITL_TIMEOUT_MINUTES` (plugin, default 10 min)
2. `plugins.hook_callback_timeout` (Hermes ≥ v2026.9, default **30 s**)
3. `timeouts.tools.sequential_call` / `concurrent_batch` (Hermes, default **420 s**)

The plugin stops waiting shortly before the shortest limit. The margin is 15 s, or a quarter of the limit when that is smaller. It then blocks the call with its own message, for example `AMP review for "write_file" did not complete within the 30s Hermes limit (plugins.hook_callback_timeout)`. Hermes then never has to kill the hook. Doing so would suppress the plugin for 60 s and leave the call without a clear reason. At startup the plugin logs an error when a Hermes limit is shorter than `AMP_HITL_TIMEOUT_MINUTES`.

With default Hermes settings the window is only about 22 seconds. For a 10-minute window, put this in Hermes `config.yaml`:

```yaml
plugins:
  hook_callback_timeout: 600        # max; 0 disables the cap (not recommended: a hung hook then blocks forever)
timeouts:
  tools:
    sequential_call: 660
    concurrent_batch: 660
```

Even then, `600 s` minus the margin is the ceiling, so reviewers have about **9.5 minutes**.

**One approval at a time.** Hermes authorizes the calls of a concurrent batch one after another, but they share one batch deadline. If the model issues two governed calls together, the second review only starts after the first is resolved, with whatever time is left. Ask the model to make one governed call at a time when reviews are slow.

**Retries.** When a call gives up while its review is still open, the block message gives the workitem id and tells the model not to retry until a reviewer approves. If the model retries the **identical** call (same tool and arguments) in the same session, the plugin waits on the existing review instead of opening a new workitem. An approval given in the meantime then lets the retry through once.

**Decisions are matched to the call.** AMP's decision poll is scoped to the AMP instance, not to the call. So the plugin binds each call to its own workitem id and ignores decisions for other workitems, for example an earlier approval on the same instance. An approval that carries no workitem id is not accepted unless `action_id` correlation proves it belongs to the call. With `AMP_ACTION_ID_CORRELATION=false` (the default), AMP can only report the newest workitem of the instance, so the plugin also:

- runs **one review per session at a time**: a second governed call waits for the first to finish, and is blocked with "one approval at a time" if its own window ends first;
- polls AMP **once before every governed request** (a "baseline"). Whatever workitem that poll reports existed before this call and is never accepted as its decision. This costs **one extra AMP call per governed action**, counted against the monthly call limit;
- remembers the last 50 consumed or abandoned workitems per instance in `sessions.json`, so a restarted Hermes that reuses the AMP instance cannot act on an old approval.

- only accepts a decision for a workitem it is bound to (from the request response or a *pending* poll). A decision AMP cannot tie to a workitem (`method` `auto`/`fallback`) blocks at once as an "unverifiable decision";
- when a request's workitem never appeared before the call gave up, blocks the next call on that instance for up to 5 minutes ("has not been registered yet"), so the late workitem cannot decide a different call;
- also locks across Hermes processes on the same session (a file lock under `plugin_state/amp-governance/locks/`, POSIX only).

If the baseline poll fails, a 4xx blocks the call. A network error follows `AMP_FAIL_CLOSED`; even with fail-open, the call is then decided only by a workitem the plugin saw pending. With `AMP_ACTION_ID_CORRELATION=true`, AMP matches by `action_id`, and the lock and baseline are skipped.

**One-shot runs.** `hermes -z` / `hermes chat -q` do not fire Hermes's session-finalize hook, so the plugin finishes the AMP session at the end of the turn. This is bounded to 3 s and best-effort. Under `hermes chat -q` goal loops that run several turns, each turn therefore gets its own AMP instance. In one-shot mode Hermes captures stdout/stderr, so the `AMP: waiting for approval of …` line goes to the log, not the terminal.

## Notification Bridge

When AMP triggers HITL, the AHP plugin sends a notification back to the originating channel — wherever the user is interacting with Hermes.

**Supported platforms:** Slack, Telegram, Discord, Signal, Matrix, ntfy, and any other platform whose adapter is registered in your Hermes gateway.

**CLI mode:** When Hermes runs in CLI mode (`HERMES_SESSION_PLATFORM=cli` or empty), the notification bridge is a no-op. Governance is still enforced normally; the user sees the outcome through the terminal response text.

**Notification events:**
- Action paused, awaiting human review
- Review approved
- Review modified (the original call is blocked; re-submit with the reviewer's changes)
- Review rejected (the reviewer's comment is passed on as `Reviewer note: …`, capped at 500 characters)
- Review cancelled (blocked; not a reviewer rejection)
- Review timed out, or the Hermes time limit was reached, and the action was blocked

**Failure behavior:** Notification delivery is best-effort. If `send_message` fails for any reason (channel not found, gateway unavailable, network error), the failure is logged at WARNING level and silently swallowed. Governance enforcement never depends on delivery success.

**Configuration:**
```dotenv
# Notifications are on by default. Set to false to disable entirely.
AMP_NOTIFICATIONS_ENABLED=true
```

## AMP API Reference

The sections above describe AHP's governance *behavior*; this section maps that behavior onto the concrete AMP API calls behind it — useful if you're building your own AMP integration and want the underlying HTTP contract, not just what AHP does with it.

AHP is a thin, direct consumer of AMP's REST API — `X-API-Key`-authenticated HTTP calls, no SDK. It uses 6 endpoints:

| Purpose | Endpoint | Called from |
|---|---|---|
| Start a governed session | `POST /api/agent/init` | `AmpClient.init_instance()` |
| Policy evaluation + HITL request (tool call, LLM call, or plan) | `POST /api/hitl/request` | `AmpClient.request_hitl()`, `request_llm_hitl()`, `request_plan_approval()` |
| Poll for the reviewer's decision | `GET /api/hitl/get-decision` | `AmpClient.get_hitl_decision()` |
| Write a transparency log line | `POST /api/log` | `AmpClient.log()` |
| Save an LLM call's prompt/reasoning/answer | `POST /api/agents/{instance_id}/llm_trace` | `AmpClient.save_llm_trace()` |
| Signal a lifecycle state transition | `POST /api/agent/setState` | `AmpClient.set_state()` |

**Call sequence:** `init` once at session start → `hitl/request` before each governed action → poll `hitl/get-decision` until resolved → `log` around the action → `setState` at session end. `llm_trace` is saved alongside each LLM call, independent of this sequence.

Request/response payload shapes for `/api/hitl/request` are detailed where they're used: LLM budget governance in [Budget Enforcement](#budget-enforcement) below, plan governance in [Plan Approval Interface](#plan-approval-interface), and tool-call governance uses the same endpoint with the `tool`/`action` pair from [What this plugin governs](#what-this-plugin-governs) above. `amp_client.py` is the source of truth for exact request bodies.

**Polling, not callback:** `/api/hitl/request` also accepts an optional `callback` field so AMP can push the decision to a reachable URL instead of being polled for it. AHP doesn't use it: a Hermes agent typically runs on a developer's laptop or behind a private network with no public URL for AMP to call back to, so AHP always polls `/api/hitl/get-decision` instead. If your own integration runs somewhere with a stable public endpoint, the callback path removes the need to poll.

**`notify` (not currently set by AHP):** `/api/hitl/request` also accepts an optional `hitl.notify` array (e.g. `["email"]`) — AMP sends an email alert to the reviewer when the workitem is created, so they don't have to be actively watching AMP or waiting on a poll result elsewhere. AHP doesn't set this today (`amp_client.py`'s `hitl` object omits it), but it's a straightforward addition to `request_hitl()`/`request_llm_hitl()`/`request_plan_approval()` if you want reviewers alerted by email. This is a request-body field the caller sets directly — unlike `who`/`what`/`where`, it is not resolved from a policy's `hitl_spec`. No batching or rate limiting on AMP's side — one email per workitem, so it's worth leaving off for an agent that raises many HITL requests in a short window unless that volume is actually intended.

## Runtime Governance

This section explains what runtime LLM instrumentation covers, and what it doesn't — read it before "LLM Observation" and "Budget Enforcement" below.

Runtime LLM governance instruments LLM calls made through Hermes' **normal conversational agent loop** — the `pre_api_request`/`post_api_request` hooks and `llm_execution` middleware that fire on every provider call inside `agent/conversation_loop.py`. Every "every LLM call" statement in this document means every call on *that* path.

It does **not** cover calls a plugin makes via `ctx.llm.complete()` / `ctx.llm.complete_structured()` (the host-owned LLM facade documented at `agent/plugin_llm.py`). That facade calls `agent/auxiliary_client.py::call_llm()` directly and never touches `pre_api_request`, `post_api_request`, or `llm_execution` — verified by inspecting `agent/auxiliary_client.py`, which has no hook-invocation code at all. A plugin LLM call made this way is invisible to AMP's observation and enforcement, full stop.

This is why the research-agent skill (see [Research Example](#research-example)) deliberately never uses `ctx.llm` for planning or research — every step of that workflow runs as the model's own normal conversational turn specifically so it stays inside AMP's actual coverage.

## LLM Observation

When enabled, the AHP plugin observes every LLM API call made through the normal conversational agent loop (see "Runtime Governance" above) and accumulates token usage and estimated cost per session. Observation runs in both `observe` and `enforce` modes.

**What is captured per LLM call:**
- `input_tokens`, `output_tokens`
- `cache_read_tokens`, `cache_write_tokens` (Anthropic prompt caching)
- `reasoning_tokens` (extended thinking)
- `total_tokens`
- Estimated cost in USD (from Hermes' built-in pricing table)
- Cost status: `"estimated"`, `"actual"`, `"included"`, or `"unknown"`
- Wall-clock API duration

**Accumulated per session:**
- All token fields summed across all API calls
- Total estimated cost
- Worst-case cost status (if any call has `"unknown"` cost, the session total is `"unknown"`)
- LLM call count and governed tool call count

**Unknown cost handling:** When a model is not in Hermes' pricing table (local models, custom servers, new models not yet added), cost is recorded as `0.0` with `cost_status="unknown"`. The session continues normally. Do not use observation mode as a budget enforcement mechanism; it records costs that are estimatable and marks the rest as unknown.

**Streaming:** Usage is accumulated after the stream completes. The `post_api_request` hook fires after full completion, so streaming and non-streaming calls are handled identically.

**Retried calls:** Each attempt is counted and accumulated separately. If a call fails and is retried, both the failed attempt (tokens consumed up to failure) and the retry are recorded.

**Concurrent sessions:** Usage data is isolated per `session_id`. Concurrent sessions do not share accumulation state.

**AMP logging:** Each LLM call is logged to AMP with token and cost fields. A final execution summary is logged when the session ends.

## Budget Enforcement

When `AMP_LLM_GOVERNANCE_MODE=enforce`, the AHP plugin evaluates every LLM call against AMP policy **before** the call reaches the model. The policy decision determines whether the call is allowed, requires HITL approval, or is blocked.

**Observe vs enforce:**
- `observe` — accumulates token and cost data, logs to AMP, does not block calls.
- `enforce` — additionally evaluates pre-call policy, pauses for HITL when required, and raises `LLMExecutionBlocked` to cancel blocked calls.

**Hermes capability requirement:**

Enforcement requires `LLMExecutionBlocked` from `hermes_cli.middleware`, which is available in Hermes builds that include the change from [nousresearch/hermes-agent#64662](https://github.com/nousresearch/hermes-agent/issues/64662). The AHP plugin detects the class at import time:

```python
try:
    from hermes_cli.middleware import LLMExecutionBlocked
    _LLM_BLOCKED_AVAILABLE = True
except ImportError:
    _LLM_BLOCKED_AVAILABLE = False
```

If `LLMExecutionBlocked` is not present, the enforcement middleware is **not registered** and an error is logged at startup:

```
amp-governance: AMP_LLM_GOVERNANCE_MODE=enforce is configured but
LLMExecutionBlocked is not available in the installed Hermes
(nousresearch/hermes-agent#64662). Enforcement middleware was NOT
registered. LLM calls will proceed without enforcement.
```

Observation continues normally even when enforcement is unavailable.

**Local dev:** Use the `feature/llm-execution-blocking` branch of the Hermes repo, which implements `LLMExecutionBlocked`. Set `AMP_LLM_GOVERNANCE_MODE=enforce` in `~/.hermes/.env`.

**Upstream status:** The `LLMExecutionBlocked` proposal is submitted as [nousresearch/hermes-agent#64662](https://github.com/nousresearch/hermes-agent/issues/64662) and is pending upstream review.

**Policy decisions:**

The pre-call governance signal sent to AMP (`/api/hitl/request`) includes:
- `tool="llm"`, `action="invoke"` for AMP policy matching
- Current session cost (accumulated so far)
- Estimated next-call cost
- Projected total cost after next call
- All token counts for the session

AMP returns one of:
- `no-hitl` / `allow` — call proceeds immediately
- `pending` — HITL required; call is paused
- `no_policy` — no active policy; call is blocked

**HITL flow:**

When AMP requests HITL, the AHP plugin:
1. Notifies the user in their Hermes channel: `"Execution paused pending approval in AMP."` with current/projected cost context
2. Polls AMP for the reviewer's decision (see `AMP_HITL_POLL_INTERVAL_SECONDS`)
3. On approval: resumes the LLM call normally
4. On rejection or timeout: raises `LLMExecutionBlocked` to cancel the call

The reviewer acts in the AMP UI (HITL workitems page). The notification goes to the user's Hermes channel (Slack, Telegram, Discord, etc.), so they know to expect a delay.

**Cost estimation:**

Before the call, the AHP plugin estimates the pending call cost using:
- Input estimate: `max(total_message_chars / 4, 100)` tokens
- Output estimate: 1024 tokens (conservative default)

Estimation uses Hermes' built-in pricing table. For models not in the pricing table (local, custom, new), the estimate is `(0.0, "unknown")`.

**Projected cost limitation:** The projected total cost is `accumulated_cost + estimated_next_call_cost`. It does not include future calls beyond the next one (a plan-level view is not available until the plan-approval interface — see [Research Example](#research-example) — is used). Use it as a directional signal, not a hard budget bound.

**Fail-open / fail-closed:**

LLM governance defaults to **fail-open**: if AMP is unreachable or returns an unexpected response, the LLM call is allowed to proceed. This is the opposite of tool governance (which defaults fail-closed) because blocking every LLM call on AMP unavailability would halt the agent entirely.

```dotenv
# LLM governance fail behavior. Default: false (fail-open).
AMP_LLM_GOVERNANCE_FAIL_CLOSED=false
```

This setting applies only to AMP unavailability and unexpected AMP responses. Explicit block decisions (`no_policy`, reviewer rejection, HITL timeout) always block regardless of this setting.

**Configuration:**

```dotenv
# On by default in this reference implementation. Set to false to disable
# LLM observation/enforcement entirely and use tool-call governance only.
AMP_LLM_GOVERNANCE_ENABLED=true

# Mode: "observe" (accumulate only) or "enforce" (pre-call policy eval + block).
# "enforce" requires Hermes with LLMExecutionBlocked (hermes-agent#64662).
AMP_LLM_GOVERNANCE_MODE=observe

# LLM governance fail behavior. Default: false (fail-open).
# When true, LLM calls are blocked if AMP is unreachable.
AMP_LLM_GOVERNANCE_FAIL_CLOSED=false

# Whether to track LLM calls from subagents launched by this session.
AMP_LLM_GOVERNANCE_INCLUDE_SUBAGENTS=true
```

## Research Example

An end-to-end demo of cost-governed agent behavior: a user sends a chat message, a skill loads their configured topics (or a specific topic they named on the spot), proposes a plan, waits for AMP approval, researches only after approval, and reports back — with every LLM call along the way governed by LLM observation/enforcement (see "Runtime Governance" above: this only works because the whole workflow runs as the model's own normal conversational turns, never via `ctx.llm`). Install steps are in [Advanced Example: Govern LLM Budget Usage](#advanced-example-govern-llm-budget-usage); this section covers how it works.

### Plan Approval Interface

Budget enforcement enforces a budget on every LLM call — but it doesn't have a budget until something sets one. The plan-approval interface is that missing piece: a caller uses it once, up front, to get a proposed execution plan approved by AMP and turn that approval into the runtime budget enforcement then applies for the rest of the session.

This is AHP-plugin-only plumbing. It does **not** decide what a "plan" is, read any config file, call a planning LLM, or react to a chat trigger — composing the plan and deciding when to submit it is entirely the caller's job (the research-agent skill below, or any other Hermes-side workflow). The AHP plugin's job is only: take a plan dict, get it approved, wire up the budget.

**Interface:**

```python
result = evaluate_proposed_plan(
    session_id,
    plan={
        "plan_type": "research",
        "summary": "Research five configured topics",
        "projected_cost_usd": 4.75,
        "projected_cost_status": "estimated",  # optional, default "estimated"
        "estimated_llm_calls": 18,              # optional, default 0
        "estimated_tool_calls": 25,              # optional, default 0
        "estimated_duration_minutes": 20,        # optional, default 0
        "work_units_total": 5,                   # optional, default 0
        "payload": {"topics": [...]},            # optional, opaque to AHP
    },
)
```

`evaluate_proposed_plan` is exported at module level; once installed as a Hermes plugin, `amp_governance.evaluate_proposed_plan(...)`. It delegates to `AmpGovernancePlugin.evaluate_proposed_plan`.

**Validation:** only the two governance-relevant fields are required — `plan_type` (non-empty string) and `projected_cost_usd` (non-negative number). A missing/invalid required field returns an error result immediately, without contacting AMP. Everything else is optional with safe defaults. `payload` is never interpreted by AHP — it is included in the AMP request's `context` unmodified and is meant for the caller's own downstream use (e.g. the research execution step below).

**AMP normalization:** the plan is submitted through the existing `/api/hitl/request` path (same endpoint tool/LLM governance already use) — no new AMP API, no AMP schema change:

```json
{
  "tool": "execution_plan",
  "action": "submit",
  "plan_id": "...",
  "plan_type": "research",
  "plan_projected_cost_usd": 4.75,
  "plan_projected_cost_status": "estimated",
  "plan_estimated_llm_calls": 18,
  "plan_estimated_tool_calls": 25,
  "plan_estimated_duration_minutes": 20,
  "plan_work_units_total": 5
}
```

These flat fields are promoted to policy `params` automatically, so an AMP Rule-based Policy criterion can reference them directly, e.g. auto-approve when `plan_projected_cost_usd < 5.0`, otherwise require HITL.

**Decision outcomes:**

| AMP response | Result `status` | `approved_budget_usd` |
|---|---|---|
| `no-hitl` / `allow` / `allowed` / `approved` | `approved` | set immediately |
| `pending` + HITL approve | `approved` | set after reviewer decision |
| `pending` + HITL modify | `rejected` (reason: re-submit the modified plan) | not set |
| `pending` + HITL reject, or review cancelled | `rejected` | not set |
| `pending` + HITL timeout, or the Hermes tool deadline reached | `timed_out` | not set |
| `no_policy` | `rejected` | not set |
| AMP unreachable, decision poll rejected (HTTP 401/403/404/429), or unexpected status | `error` | not set |

The full result shape:

```python
{"status": "approved", "plan_id": "...", "approved_budget_usd": 4.75, "reason": "", "workitem_id": None}
```

**Approved-budget initialization:** on approval (auto or HITL), the AHP plugin sets `approved_budget_usd` on the session's `ExecutionContext` to the plan's own `projected_cost_usd` — never a recomputed value. The `ExecutionContext` is created if `on_session_start` hadn't already created one, since budget enforcement's `llm_execution_middleware` silently allows calls when no context is tracked; without this, an approved plan would never actually constrain anything. From that point on, every LLM call in the session is evaluated against this budget by the existing enforcement path — no additional wiring needed.

### How It Works

The research-agent skill builds on the plan-approval interface above with a working sample: a skill that detects the chat trigger, loads a topics config file, builds the plan, and calls `amp_evaluate_research_plan`.

Two ways to trigger it:
- **Configured mode**: `Run my research topics.` — researches whatever's in your `research_topics.yaml`.
- **Ad hoc mode**: `Research the US market.` (or any specific topic) — researches just that topic, using sensible defaults (`research_depth: standard`, `sources_per_topic: 5`, `lookback_days: 30`) since there's no config entry for a one-off request. Same plan-approval → research → report pipeline either way; only the topic source differs.

**Three registered tools** (`ctx.register_tool()`, wired in `register()`, not listed under `plugin.yaml`'s `hooks:` since tools are a separate registration API):
- `amp_evaluate_research_plan({"plan": {...}})` — thin wrapper around `evaluate_proposed_plan()` above. Sources `session_id` from `gateway.session_context.get_session_env("HERMES_SESSION_ID", "")` (the same pattern the notification bridge already uses for platform/chat/thread) rather than trusting the model to supply one. Used identically by both configured and ad hoc mode — the AHP plugin has no concept of "mode," it only ever sees a plan dict.
- `amp_load_research_topics({"path": "..."})` — validates `~/.hermes/research_topics.yaml` (or an override path) via `research_config.load_research_topics()`: 1-5 topics required, defaults applied for `research_depth`/`sources_per_topic`/`lookback_days`. Exists so config loading is deterministic Python, not the model hand-parsing YAML. Called only in configured mode — ad hoc mode skips it entirely (see the skill's step 0).
- `amp_governance_summary()` — returns the session's current `ExecutionContext.to_summary_dict()` (cost, tokens, call counts) so the skill's final report uses real numbers instead of inventing them. Returns `{"status": "no_context"}` if nothing has been tracked yet.

**Three skills:**
- `skills/research-agent/SKILL.md` — plugin-bundled (`ctx.register_skill()`), the actual step-by-step workflow, shared by both modes. Step 0 determines which mode a given trigger is (configured vs ad hoc) and branches accordingly; everything from plan-building onward (steps 2-8) is identical regardless of mode. Plugin-bundled skills are **not** auto-listed in the system prompt's skill index (they're opt-in explicit loads only), so this alone isn't discoverable by natural language.
- `skills-pointer/amp-research/SKILL.md` — configured-mode entry point, matches phrasing like "Run my research topics."
- `skills-pointer/amp-research-topic/SKILL.md` — ad hoc-mode entry point, matches phrasing like "Research the US market." Deliberately a **separate** pointer skill rather than one pointer covering both cases — two narrowly-scoped skill descriptions give the LLM-mediated routing a much cleaner signal than one description trying to cover two different intents, which is exactly the kind of ambiguity that produced flaky routing in earlier testing. In practice, the AHP plugin's own `pre_llm_call` hook now routes straight to the real skill and decides the mode itself whenever a message mentions "research," so these pointer skills mainly matter as the natural-language-discoverable entry point when that hook isn't the one driving (e.g. a direct `skill_view` call).

Neither pointer skill is plugin-bundled (both must be installed to `~/.hermes/skills/`, see [Advanced Example: Govern LLM Budget Usage](#advanced-example-govern-llm-budget-usage)), so both *are* auto-indexed and matched by natural language. Their entire body is: call `skill_view(name="amp-governance:research-agent")` and follow it. There is deliberately no separate `ctx.register_command()` for either — a plugin-registered command's return value is sent directly as the reply and cannot hand off to the model's multi-turn loop (verified against `gateway/run.py`), so it couldn't have driven this workflow anyway. Natural language alone is the supported entry point; Hermes' per-skill slash-command derivation (`agent/skill_commands.py`) technically also gives each of these a `/amp-research` / `/amp-research-topic` command, but on Slack specifically that name is intercepted by Slack's own reserved `/` namespace before it ever reaches Hermes (confirmed live — Slack rejects it client-side unless separately registered as a native Slack Slash Command, out of scope here), so it isn't a reliable cross-platform entry point and isn't documented as one.

**Pointer skill installation:** previously a manual copy/symlink step into `~/.hermes/skills/research/`. Now a one-liner per skill via `hermes skills install <URL-to-SKILL.md>` (see [Advanced Example: Govern LLM Budget Usage](#advanced-example-govern-llm-budget-usage)) — Hermes' own skill installer accepts a direct HTTPS URL to a `SKILL.md` file, which resolves this cleanly for a human running setup. A plugin still cannot auto-place a *non-bundled, auto-indexed* skill into the user-local skill tree on the user's behalf with zero commands at install time (plugin-bundled skills are deliberately excluded from that index — see "Plan Approval Interface" above); that narrower gap remains open if it matters to you.

**Testing this workflow end-to-end:** `scripts/e2e_research_test.py adhoc|configured|freshness|all` drives a real `hermes chat` turn for each mode, auto-approves any AMP plan-approval workitem via the same API the AHP plugin itself uses (`X-API-Key` from `~/.hermes/.env`, no browser/Slack needed), and cross-checks the result against `~/.hermes/state.db` — catches wrong-skill routing, a missing/invented plan cost, a governance summary that doesn't match what actually happened, and (via the `freshness` scenario) a plain time-sensitive question getting wrongly diverted into the governed workflow instead of a direct `web_search`. Makes real LLM calls (small real cost per run); not part of `pytest tests/`. Run with the hermes-agent venv (`.../hermes-agent/venv/bin/python`, for `pyyaml`).

**Not implemented this cycle:** cron scheduling (see `hermes cron create ... --skills "amp-governance:research-agent"` for the verified path once someone wants it), topic-management UI, plan revision loops, multiple report formats, ad hoc topics longer than 5 (silently capped, see the skill's step 0), a `plan_type` distinction between the two modes (both submit `plan_type: "research"` — deliberately kept as a skill-instruction-only distinction, not a governance-signal one, for this cycle).

## Common issues

### Plugin loads but does nothing

Check:

- `plugin.yaml` is present in `~/.hermes/plugins/amp-governance`
- `__init__.py` is present in the same directory
- the plugin is enabled:

```bash
hermes plugins enable amp-governance
```

### Hermes starts but AMP governance is not configured

Check that `~/.hermes/.env` contains all required values:

- `AMP_BACKEND_URL`
- `AMP_API_KEY`
- `AMP_ORG_ID`
- `AMP_USERNAME`
- `AMP_AGENT_NAME`

### Tool calls are blocked unexpectedly

The block message says why:

| Message starts with | Meaning |
|---|---|
| `… was rejected by AMP HITL review` | A reviewer rejected it (`Reviewer note:` carries their comment) |
| `AMP reviewer approved a modified version …` | The reviewer changed the call; the plugin cannot apply the change yet, so re-submit it |
| `The AMP review of … was cancelled` | The workitem was cancelled before anyone decided |
| `… timed out waiting for AMP HITL approval` / `… did not complete within the …s Hermes limit` | Nobody decided in time (see [Approval window](#approval-window-hermes-time-limits)) |
| `AMP rejected the governance request … (HTTP 4xx)` / `AMP rejected the HITL decision poll … (HTTP 4xx)` | AMP answered but refused: bad key, wrong org, monthly call limit (429), invalid payload |
| `AMP governance is unavailable` | Network error or AMP 5xx (blocked because `AMP_FAIL_CLOSED=true`) |
| `AMP was unreachable while waiting for the review …` | The decision polls failed with network errors / 5xx until the window ended (not a timeout) |
| `… (unverifiable decision)` / `… has not been registered yet` | Decision-matching safeguards; see [Approval window](#approval-window-hermes-time-limits) |
| `No active AMP governance policy …` / an AMP policy message | The agent has no active eval policy, or its HITL routing is invalid |

Also check:

- the active AMP policy for your Hermes agent
- whether the Hermes tool was normalized into a different AMP tool/action than you expected
- whether AMP policy criteria are too broad

### HITL never resolves

Check:

- reviewer assignment in the AMP policy or HITL setup
- `AMP_HITL_TIMEOUT_MINUTES`
- AMP workitems page for pending approvals
- `HERMES_AGENT_TIMEOUT` — if HITL reviews take more than 30 minutes, set this to a higher value (e.g., `HERMES_AGENT_TIMEOUT=7200`)
- On Hermes ≥ v2026.9, `plugins.hook_callback_timeout` (default 30 s) and `timeouts.tools.*` (default 420 s) cap the wait. The plugin blocks with "did not complete within the …s Hermes limit"; see [Approval window](#approval-window-hermes-time-limits)
- Polls that AMP rejects (401/403/404/429, e.g. the monthly call limit) now block immediately with the HTTP status instead of waiting for the timeout

### Research skill stops triggering, or abandons governance mid-run

Symptoms: a message that should trigger the research workflow instead gets a
direct, ungoverned answer; or the workflow starts but `amp_evaluate_research_plan`
fails repeatedly and the model falls back to a raw `web_search`/reports a vague
"internal configuration issue" instead of stopping cleanly.

This is a known failure mode in a **long-running channel/thread**, not a config
problem: the model can anchor on a stale plan shape, topic list, or approval
from earlier in that same conversation instead of following the current skill
instructions, even right after re-loading them. It gets more likely the longer
a thread has been open and the more it has already gone through this workflow.

Fix: start a fresh Hermes session in that channel rather than continuing the
existing thread — a new session has no prior turns to anchor on. In Slack,
send `/new` (or `/hermes new` if `/new` isn't registered as a native Slack
slash command in your workspace — see `hermes slack manifest` in the upstream
Hermes docs to register core commands like `/new` natively). The equivalent
command on other platforms is also `/new`.

### Notifications are not appearing in my channel

Check:

- `AMP_NOTIFICATIONS_ENABLED` is `true` (default)
- Hermes gateway is running with a platform adapter (Slack bot token, Telegram bot token, etc.)
- `HERMES_SESSION_PLATFORM` is set correctly in the gateway for your channel
- The `send_message` tool is enabled and working in your Hermes setup

### LLM observation shows cost_status="unknown"

This is expected for:
- Local models (Ollama, LM Studio, etc.) that are not in Hermes' pricing table
- New models released after the Hermes pricing table was last updated
- Providers where Hermes cannot determine the model name

The session continues normally. Token counts are still recorded; only the cost calculation is unavailable.

### LLM enforcement is configured but not activating

Check:

- `AMP_LLM_GOVERNANCE_ENABLED=true` and `AMP_LLM_GOVERNANCE_MODE=enforce` are set
- The installed Hermes build includes `LLMExecutionBlocked` ([hermes-agent#64662](https://github.com/nousresearch/hermes-agent/issues/64662))
- Hermes gateway logs at startup for the message `"LLM enforcement enabled"`
- If the log shows `"Enforcement middleware was NOT registered"`, you need a Hermes build that includes the upstream change

For local dev, use the `feature/llm-execution-blocking` branch of the Hermes repo.

## Developer Notes

### Files in this plugin

- `plugins/saas/plugin.yaml` — Hermes plugin manifest
- `plugins/saas/__init__.py` — main plugin hooks and governance flow
- `plugins/saas/amp_client.py` — AMP API client
- `plugins/saas/config.py` — config loading from `~/.hermes/.env`
- `plugins/saas/execution_context.py` — per-session LLM usage accumulation
- `plugins/saas/notification.py` — platform-neutral notification bridge
- `plugins/saas/policy.py` — Hermes tool normalization into AMP policy vocabulary
- `plugins/saas/session_store.py` — session-to-AMP instance tracking (stores a SHA-256 digest of the API key, never the key)
- `plugins/saas/hitl_poll.py` — HITL decision polling shared by the tool, LLM and plan paths (workitem binding, outcome mapping)
- `plugins/saas/host_limits.py` — Hermes hook/tool time limits that bound the approval window
- `plugins/saas/research_config.py` — research_topics.yaml loading/validation
- `plugins/saas/skills/research-agent/SKILL.md` — plugin-bundled research workflow
- `plugins/saas/skills-pointer/amp-research/SKILL.md`, `plugins/saas/skills-pointer/amp-research-topic/SKILL.md` — small discoverable pointer skills (configured mode, ad hoc mode); install via `hermes skills install <URL>` (see [Advanced Example: Govern LLM Budget Usage](#advanced-example-govern-llm-budget-usage))
- `plugins/saas/examples/research_topics.yaml` — template config read by the research-agent skill at runtime (the planning-prompt schema is inlined in the skill itself)
- `plugins/saas/scripts/e2e_research_test.py` — end-to-end test harness for the research workflow (see [Research Example](#research-example) above)
- `plugins/saas/.env.example` — example environment variables for setup
- `LICENSE` — MIT open-source license

### Local Development

For local development, symlink install is easier than `hermes plugins
install` because Hermes will load the plugin directly from your working
repo, and edits take effect on the next gateway restart with no
reinstall step:

```bash
mkdir -p ~/.hermes/plugins
ln -s /path/to/amp-hermes-plugin/plugins/saas ~/.hermes/plugins/amp-governance
hermes plugins enable amp-governance
hermes gateway restart
```

Run the SaaS plugin's tests against a Hermes checkout (released Hermes is fine; the LLM-enforcement tests use a stand-in for `LLMExecutionBlocked` when Hermes lacks it):

```bash
cd plugins/saas
PYTHONPATH=/path/to/hermes-agent /path/to/hermes-agent/venv/bin/python -m pytest -q -p no:cacheprovider
```
