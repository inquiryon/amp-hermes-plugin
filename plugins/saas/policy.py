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

# Argument names whose values are never sent (case-insensitive substring match).
_SECRET_KEY_RE = re.compile(r"(?i)(key|token|secret|passw|credential|auth|cookie|session)")

# Fallback value patterns, applied after Hermes's own redactor (agent.redact) when present.
_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.DOTALL),
    re.compile(r"\bamp_k_[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    re.compile(
        r"(?i)\b([A-Za-z0-9_]*(?:password|passwd|secret|token|api[_-]?key|access[_-]?key)[A-Za-z0-9_]*)"
        r"(\s*[:=]\s*)(['\"]?)[^\s'\"]+\3"
    ),
)


def redact_secrets(text: str) -> str:
    """Best-effort removal of credentials from text before it leaves the machine."""
    redacted = str(text or "")
    try:  # Hermes's redactor knows many more credential formats; use it when installed
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(redacted, force=True)
    except Exception:
        pass
    for pattern in _SECRET_VALUE_PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED}", redacted)
        else:
            redacted = pattern.sub(_REDACTED, redacted)
    return redacted


def _truncate_utf8(text: str, limit_bytes: int) -> Tuple[str, bool]:
    data = text.encode("utf-8")
    if len(data) <= limit_bytes:
        return text, False
    return data[:limit_bytes].decode("utf-8", errors="ignore") + _TRUNCATION_MARKER, True


def content_preview_fields(content: Any) -> Dict[str, Any]:
    """Preview (<= 2 KB, secrets redacted) plus hash and size of the full content."""
    text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, default=str)
    data = text.encode("utf-8")
    preview, truncated = _truncate_utf8(redact_secrets(text), CONTENT_PREVIEW_LIMIT_BYTES)
    return {
        "content_preview": preview,
        "content_preview_truncated": truncated,
        "content_sha256": hashlib.sha256(data).hexdigest(),
        "content_bytes": len(data),
    }


def _patch_content(args: Dict[str, Any]) -> str:
    if isinstance(args.get("patch"), str) and args.get("patch"):
        return args["patch"]  # V4A patch body
    old = str(args.get("old_string") or "")
    new = str(args.get("new_string") or "")
    return f"--- old_string\n{old}\n+++ new_string\n{new}"


def _drop_secret_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (_REDACTED if _SECRET_KEY_RE.search(str(key)) else _drop_secret_keys(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_drop_secret_keys(item) for item in value]
    return value


def unlisted_tool_context(args: Dict[str, Any]) -> Dict[str, Any]:
    text = json.dumps(_drop_secret_keys(args), sort_keys=True, default=str)
    preview = redact_secrets(text)
    truncated = len(preview) > UNLISTED_ARGS_LIMIT_CHARS
    if truncated:
        preview = preview[:UNLISTED_ARGS_LIMIT_CHARS] + _TRUNCATION_MARKER
    return {"arguments": preview, "arguments_truncated": truncated}


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
    governs every other tool except the plugin's own amp_* tools.
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

    if govern_unlisted and tool_name and not tool_name.startswith("amp_"):
        return NormalizedAction(
            raw_tool_name=tool_name,
            tool=tool_name,
            action="invoke",
            context=unlisted_tool_context(args),
        )

    return None
