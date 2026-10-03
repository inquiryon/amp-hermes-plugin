from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class NormalizedAction:
    raw_tool_name: str
    tool: str
    action: str
    context: Dict[str, Any]


# --- Content preview (A15, gate C9) and unlisted-tool governance (A9, gate C5) ---

# Reviewer-facing preview of what a write/edit will put on disk.
CONTENT_PREVIEW_LIMIT_BYTES = 2048
# Arguments of an otherwise unlisted tool, as shown to the policy/reviewer.
UNLISTED_ARGS_LIMIT_CHARS = 2000
_TRUNCATION_MARKER = "\n...[truncated]"
_REDACTED = "[REDACTED]"

# Only this much text is ever scanned/redacted for a preview or argument summary (the
# hash and size still cover the full content). Bounds regex cost on huge writes.
_REDACTION_INPUT_LIMIT = 8192

# The plugin's own tools; everything else is governed when AMP_GOVERN_UNLISTED_TOOLS is on.
OWN_TOOLS = frozenset({"amp_evaluate_research_plan", "amp_load_research_topics", "amp_governance_summary"})

# A name is secret when its LAST component is a secret word (db_password, clientSecret,
# X-Auth-Token, api_key), so names that merely contain one (bypass_hook, tokenizer_cmd,
# secret_note, auth_mode, passenger) stay visible: redaction must not hide payloads.
_SECRET_SUFFIXES = frozenset({
    "password", "passwd", "pwd", "pass", "secret", "token", "key", "apikey", "credential",
    "credentials", "cookie", "authorization", "privatekey",
})
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NAME_SPLIT_RE = re.compile(r"[_\-.\s]+")


def is_secret_name(name: Any) -> bool:
    parts = [p for p in _NAME_SPLIT_RE.split(_CAMEL_BOUNDARY_RE.sub("_", str(name)).lower()) if p]
    return bool(parts) and parts[-1] in _SECRET_SUFFIXES


def _redact_named_value(match: "re.Match[str]") -> str:
    name, separator, value = match.group(1), match.group(2), match.group(3)
    if is_secret_name(name.strip('"')) and value.strip("'\"") and not any(c.isspace() for c in value):
        quote = value[0] if value[:1] in {"'", '"'} else ""
        return f"{name}{separator}{quote}{_REDACTED}{quote}"
    return match.group(0)


# Fallback value patterns, applied after Hermes's own redactor (agent.redact) when present.
# All quantifiers are bounded so adversarial input cannot make them super-linear.
_SECRET_VALUE_PATTERNS = (
    # Only a real PEM block (base64 body + END marker; JSON-escaped \n allowed); text
    # after a bare BEGIN stays visible.
    re.compile(
        r"-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----(?:[A-Za-z0-9+/=\s]|\\[nr]){1,16384}?"
        r"-----END [A-Z ]{0,40}PRIVATE KEY-----"
    ),
    re.compile(r"\bamp_k_[A-Za-z0-9_\-]{8,200}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,200}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,200}"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,200}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,2000}\.[A-Za-z0-9_\-]{10,2000}\.[A-Za-z0-9_\-]{10,2000}"),
)
# name/value pairs: JSON ("db_password": "x") and assignments (API_KEY=x, token: "y").
# Redacted only when the name is secret (is_secret_name) and the value is one token.
_NAMED_VALUE_PATTERNS = (
    re.compile(r'("[A-Za-z0-9_\-.]{1,80}")(\s{0,5}:\s{0,5})("[^"\n]{0,4096}")'),
    re.compile(r"\b([A-Za-z0-9_\-.]{1,80})(\s{0,5}[:=]\s{0,5})((['\"])[^'\"\n]{1,4096}\4|[^\s'\",}]{1,4096})"),
)


def redact_secrets_counted(text: str) -> Tuple[str, int]:
    """Best-effort removal of credentials; returns (text, number of redactions)."""
    original = str(text or "")
    redacted = original
    count = 0
    try:  # Hermes's redactor knows many more credential formats; use it when installed
        from agent.redact import redact_sensitive_text

        hermes_redacted = redact_sensitive_text(redacted, force=True)
        if hermes_redacted != redacted:
            count += 1
        redacted = hermes_redacted
    except Exception:
        pass
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted, n = pattern.subn(_REDACTED, redacted)
        count += n
    for pattern in _NAMED_VALUE_PATTERNS:
        before = redacted
        redacted = pattern.sub(_redact_named_value, redacted)
        if redacted != before:
            count += max(redacted.count(_REDACTED) - before.count(_REDACTED), 1)
    return redacted, count


def redact_secrets(text: str) -> str:
    """Best-effort removal of credentials from text before it leaves the machine."""
    return redact_secrets_counted(str(text or "")[:_REDACTION_INPUT_LIMIT * 4])[0]


def _truncate_utf8(text: str, limit_bytes: int) -> Tuple[str, bool]:
    data = text.encode("utf-8")
    if len(data) <= limit_bytes:
        return text, False
    return data[:limit_bytes].decode("utf-8", errors="ignore") + _TRUNCATION_MARKER, True


def content_preview_fields(content: Any) -> Dict[str, Any]:
    """Preview (<= 2 KB, secrets redacted) plus hash and size of the full content."""
    text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, default=str)
    data = text.encode("utf-8")
    redacted, redactions = redact_secrets_counted(text[:_REDACTION_INPUT_LIMIT])
    preview, truncated = _truncate_utf8(redacted, CONTENT_PREVIEW_LIMIT_BYTES)
    truncated = truncated or len(text) > _REDACTION_INPUT_LIMIT
    if truncated and not preview.endswith(_TRUNCATION_MARKER):
        preview += _TRUNCATION_MARKER
    return {
        "content_preview": preview,
        "content_preview_truncated": truncated,
        # Tells the reviewer that parts of the preview were masked (they can't see those).
        "content_redactions": redactions,
        "content_sha256": hashlib.sha256(data).hexdigest(),
        "content_bytes": len(data),
    }


def _patch_content(args: Dict[str, Any]) -> str:
    if isinstance(args.get("patch"), str) and args.get("patch"):
        return args["patch"]  # V4A patch body
    old = str(args.get("old_string") or "")
    new = str(args.get("new_string") or "")
    return f"--- old_string\n{old}\n+++ new_string\n{new}"


def _drop_secret_keys(value: Any, counter: list) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if is_secret_name(key):
                counter.append(key)
                out[key] = _REDACTED
            else:
                out[key] = _drop_secret_keys(item, counter)
        return out
    if isinstance(value, list):
        return [_drop_secret_keys(item, counter) for item in value]
    return value


def unlisted_tool_context(args: Dict[str, Any]) -> Dict[str, Any]:
    dropped: list = []
    text = json.dumps(_drop_secret_keys(args, dropped), sort_keys=True, default=str)
    preview, redactions = redact_secrets_counted(text[:_REDACTION_INPUT_LIMIT])
    truncated = len(preview) > UNLISTED_ARGS_LIMIT_CHARS or len(text) > _REDACTION_INPUT_LIMIT
    if truncated:
        preview = preview[:UNLISTED_ARGS_LIMIT_CHARS] + _TRUNCATION_MARKER
    return {
        "arguments": preview,
        "arguments_truncated": truncated,
        # Tells the reviewer how many values were masked (they can't see those).
        "arguments_redactions": len(dropped) + redactions,
    }


def normalize_tool_call(
    tool_name: str,
    args: Any,
    *,
    include_content_preview: bool = False,
    govern_unlisted: bool = False,
) -> Optional[NormalizedAction]:
    """Map a Hermes tool call to AMP's tool/action vocabulary; None = not governed.

    include_content_preview (AMP_SEND_CONTENT_PREVIEW, default off) adds a redacted
    <= 2 KB preview plus sha256/size of what write_file/patch will write, so a reviewer
    can see the change. govern_unlisted (AMP_GOVERN_UNLISTED_TOOLS, default off)
    governs every other tool except the plugin's own tools (OWN_TOOLS).
    """
    if not isinstance(args, dict):
        args = {}

    if tool_name == "terminal":
        command = str(args.get("command") or args.get("cmd") or "").strip()
        return NormalizedAction(
            raw_tool_name=tool_name,
            tool="exec",
            action="exec",
            context={
                "command": command,
                "workdir": args.get("workdir") or "",
            },
        )

    if tool_name == "read_file":
        path = str(args.get("path") or "").strip()
        return NormalizedAction(
            raw_tool_name=tool_name,
            tool="read",
            action="read",
            context={"path": path},
        )

    if tool_name == "search_files":
        path = str(args.get("path") or "").strip()
        pattern = str(args.get("pattern") or "").strip()
        target = str(args.get("target") or "content").strip() or "content"
        file_glob = str(args.get("file_glob") or "").strip()
        return NormalizedAction(
            raw_tool_name=tool_name,
            tool="read",
            action="search",
            context={
                "path": path,
                "pattern": pattern,
                "query": pattern,
                "target": target,
                "file_glob": file_glob,
            },
        )

    if tool_name == "write_file":
        path = str(args.get("path") or "").strip()
        context: Dict[str, Any] = {"path": path, "file_path": path}
        if include_content_preview:
            context.update(content_preview_fields(args.get("content") or ""))
        return NormalizedAction(raw_tool_name=tool_name, tool="write", action="write", context=context)

    if tool_name == "patch":
        path = str(args.get("path") or "").strip()
        context = {"path": path, "file_path": path}
        if include_content_preview:
            context.update(content_preview_fields(_patch_content(args)))
        return NormalizedAction(raw_tool_name=tool_name, tool="write", action="edit", context=context)

    if tool_name == "web_search":
        query = str(args.get("query") or "").strip()
        return NormalizedAction(
            raw_tool_name=tool_name,
            tool="exec",
            action="web_search",
            context={"query": query},
        )

    if tool_name == "web_extract":
        urls = args.get("urls")
        return NormalizedAction(
            raw_tool_name=tool_name,
            tool="exec",
            action="web_extract",
            context={"urls": urls if isinstance(urls, list) else []},
        )

    if govern_unlisted and tool_name and tool_name not in OWN_TOOLS:
        return NormalizedAction(
            raw_tool_name=tool_name,
            tool=tool_name,
            action="invoke",
            context=unlisted_tool_context(args),
        )

    return None
