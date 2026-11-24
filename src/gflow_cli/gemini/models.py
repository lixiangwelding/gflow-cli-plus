"""Typed models, errors, exit codes, and JSON extraction helpers for Gemini Web automation."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from gflow_cli.config import get_settings
from gflow_cli.errors import Charged, GFlowError, NextAction, exit_code_for

GEMINI_APP_URL = "https://gemini.google.com/app"
DEFAULT_GEMINI_PROFILE = "gemini_ultra"
GEMINI_PROFILE_PREFIX = "gemini_profile_"

_FENCED_JSON_RE = re.compile(
    r"```(?P<lang>[A-Za-z0-9_-]*)\s*\n(?P<body>.*?)```",
    re.DOTALL,
)


class GeminiErrorCode(StrEnum):
    """Stable machine-readable error codes for Gemini Web automation."""

    AUTH_REQUIRED = "AUTH_REQUIRED"
    RATE_LIMITED = "RATE_LIMITED"
    CHALLENGE_DETECTED = "CHALLENGE_DETECTED"
    TIMEOUT = "TIMEOUT"
    NO_IMAGE_GENERATED = "NO_IMAGE_GENERATED"
    PROFILE_LOCKED = "PROFILE_LOCKED"
    INVALID_INPUT = "INVALID_INPUT"
    BROWSER_ERROR = "BROWSER_ERROR"
    UI_DRIFT = "UI_DRIFT"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    RISK_QUARANTINED = "RISK_QUARANTINED"


class GeminiError(GFlowError):
    """Base exception for all Gemini Web browser automation failures."""

    problem_type: str = "https://gflow-cli.dev/errors/gemini-web-error"
    title: str = "Gemini Web error"
    code: GeminiErrorCode = GeminiErrorCode.BROWSER_ERROR
    exit_code: int = 15
    charged: Charged = "no"
    next_action: NextAction = "none"
    default_retryable: bool = False
    default_hard_halt: bool = False
    default_retry_after_seconds: float | None = None
    _default_remediation: str = (
        "Inspect the diagnostic screenshot/DOM snippet in tmp/gemini-debug before retrying."
    )

    def __init__(
        self,
        detail: str = "",
        *,
        code: GeminiErrorCode | None = None,
        exit_code: int | None = None,
        remediation_hint: str | None = None,
        retryable: bool | None = None,
        screenshot_path: Path | None = None,
        dom_path: Path | None = None,
        next_action: NextAction | None = None,
        cooldown_seconds: float | None = None,
        retry_after_seconds: float | None = None,
        hard_halt: bool | None = None,
        risk_reason: str | None = None,
    ) -> None:
        effective_retryable = self.default_retryable if retryable is None else retryable
        super().__init__(
            detail,
            remediation_hint=remediation_hint,
            retryable=effective_retryable,
            charged="no",
            next_action=next_action,
        )
        if code is not None:
            self.code = code
        if exit_code is not None:
            self.exit_code = exit_code
        self.screenshot_path = screenshot_path
        self.dom_path = dom_path
        effective_wait = (
            retry_after_seconds
            if retry_after_seconds is not None
            else (
                cooldown_seconds
                if cooldown_seconds is not None
                else self.default_retry_after_seconds
            )
        )
        self.cooldown_seconds: float | None = effective_wait
        self.retry_after_seconds: float | None = effective_wait
        self.hard_halt: bool = self.default_hard_halt if hard_halt is None else hard_halt
        self.risk_reason: str | None = risk_reason

    def to_error_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable error payload."""
        effective_wait = (
            self.retry_after_seconds
            if self.retry_after_seconds is not None
            else self.cooldown_seconds
        )
        payload: dict[str, Any] = {
            "ok": False,
            "status": "fail",
            "error_code": self.code.value,
            "exit_code": self.exit_code,
            "class": type(self).__name__,
            "title": self.title,
            "detail": self.detail,
            "remediation_hint": self.remediation_hint,
            "retryable": bool(self.retryable),
            "hard_halt": bool(self.hard_halt),
            "screenshot_path": str(self.screenshot_path) if self.screenshot_path else None,
            "dom_path": str(self.dom_path) if self.dom_path else None,
        }
        if self.risk_reason:
            payload["risk_reason"] = self.risk_reason
        if effective_wait is not None and not self.hard_halt:
            rounded = round(float(effective_wait), 2)
            payload["retry_after_seconds"] = rounded
            payload["cooldown_seconds"] = rounded
        return payload


class GeminiAuthRequiredError(GeminiError):
    """Gemini Web session is missing, expired, or redirected to Google sign-in."""

    problem_type = "https://gflow-cli.dev/errors/gemini-auth-required"
    title = "Gemini authentication required"
    code = GeminiErrorCode.AUTH_REQUIRED
    exit_code = 3
    next_action = "reauth"
    default_retryable = False
    default_hard_halt = True
    _default_remediation = (
        "Run `gflow gemini login --profile <name>` to sign in to Gemini in a visible browser."
    )


class GeminiRateLimitError(GeminiError):
    """Gemini Web returned HTTP 429 or rendered a rate-limit / quota-exceeded banner."""

    problem_type = "https://gflow-cli.dev/errors/gemini-rate-limited"
    title = "Gemini rate limit or quota reached"
    code = GeminiErrorCode.RATE_LIMITED
    exit_code = 4
    next_action = "wait"
    default_retryable = False
    default_hard_halt = False
    default_retry_after_seconds = 120.0
    _default_remediation = (
        "Do NOT blind-retry. Respect the cooldown window before issuing more Gemini requests."
    )


class GeminiChallengeDetectedError(GeminiError):
    """Google presented a CAPTCHA, unusual-traffic page, or account verification challenge."""

    problem_type = "https://gflow-cli.dev/errors/gemini-challenge-detected"
    title = "Google verification or CAPTCHA challenge detected"
    code = GeminiErrorCode.CHALLENGE_DETECTED
    exit_code = 10
    next_action = "reauth"
    default_retryable = False
    default_hard_halt = True
    _default_remediation = (
        "Stop automated requests immediately. Complete verification manually via "
        "`gflow gemini login --profile <name>` after cooling down."
    )


class GeminiTimeoutError(GeminiError):
    """Timed out waiting for Gemini Web page readiness, streaming completion, or image render."""

    problem_type = "https://gflow-cli.dev/errors/gemini-timeout"
    title = "Gemini Web operation timed out"
    code = GeminiErrorCode.TIMEOUT
    exit_code = 9
    next_action = "retry"
    default_retryable = True
    _default_remediation = (
        "Verify proxy/network connectivity to gemini.google.com and consider raising --timeout."
    )


class GeminiLoginTimeoutError(GeminiTimeoutError):
    """Timed out waiting for interactive Gemini Web sign-in to complete."""

    problem_type = "https://gflow-cli.dev/errors/gemini-login-timeout"
    title = "Gemini login timed out"
    code = GeminiErrorCode.TIMEOUT
    exit_code = 12
    next_action = "reauth"
    default_retryable = False
    _default_remediation = (
        "Re-run `gflow gemini login --profile <name>` and complete Google sign-in before timeout."
    )


class GeminiNoImageGeneratedError(GeminiError):
    """Gemini completed its turn without producing a valid downloadable generated image."""

    problem_type = "https://gflow-cli.dev/errors/gemini-no-image-generated"
    title = "No generated image produced"
    code = GeminiErrorCode.NO_IMAGE_GENERATED
    exit_code = 26
    next_action = "none"
    default_retryable = False
    _default_remediation = (
        "Check Gemini's text reply for a safety refusal or clarify the prompt to request image "
        "generation explicitly."
    )


class GeminiProfileLockedError(GeminiError):
    """Another process or in-process handle currently holds the isolated Gemini profile lease."""

    problem_type = "https://gflow-cli.dev/errors/gemini-profile-locked"
    title = "Gemini profile is currently locked"
    code = GeminiErrorCode.PROFILE_LOCKED
    exit_code = 11
    next_action = "wait"
    default_retryable = False
    default_hard_halt = False
    default_retry_after_seconds = 20.0
    _default_remediation = (
        "Wait for the active Gemini command using this profile to finish, close any stray Chrome "
        "window on this profile, or select a different --profile."
    )


class GeminiUiDriftError(GeminiError):
    """Expected Gemini Web composer or action controls could not be located in the DOM."""

    problem_type = "https://gflow-cli.dev/errors/gemini-ui-drift"
    title = "Gemini Web UI selector drift"
    code = GeminiErrorCode.UI_DRIFT
    exit_code = 23
    next_action = "none"
    default_retryable = False
    _default_remediation = (
        "Inspect the saved diagnostic screenshot and DOM snippet under tmp/gemini-debug/."
    )


class GeminiDownloadError(GeminiError):
    """Failed to download or validate the generated high-resolution image asset."""

    problem_type = "https://gflow-cli.dev/errors/gemini-download-failed"
    title = "Gemini image download failed"
    code = GeminiErrorCode.DOWNLOAD_FAILED
    exit_code = 6
    next_action = "retry"
    default_retryable = True
    _default_remediation = (
        "Verify proxy connectivity to googleusercontent.com and inspect tmp/gemini-debug/."
    )


class GeminiConfigError(GeminiError):
    """Invalid user input, profile name, file path, or environment configuration."""

    problem_type = "https://gflow-cli.dev/errors/gemini-invalid-input"
    title = "Invalid Gemini input or configuration"
    code = GeminiErrorCode.INVALID_INPUT
    exit_code = 11
    next_action = "fix_input"
    default_retryable = False
    _default_remediation = "Fix the invalid argument or file path and run the command again."


class GeminiRiskBlockedError(GeminiError):
    """Anti-risk guard blocked execution (kill switch, active quarantine, or Flow yield)."""

    problem_type = "https://gflow-cli.dev/errors/gemini-risk-quarantined"
    title = "Gemini request blocked by anti-risk guard"
    code = GeminiErrorCode.RISK_QUARANTINED
    exit_code = 10
    next_action = "wait"
    default_retryable = False
    default_hard_halt = False
    _default_remediation = (
        "Respect the active quarantine/cooldown or remove STOP_FLOW/STOP_GEMINI when safe."
    )


#: Alias for callers referencing ``GeminiRiskGuardError``.
GeminiRiskGuardError = GeminiRiskBlockedError


def gemini_exit_code_for(exc: BaseException) -> int:
    """Resolve the process exit code for any Gemini or gflow exception."""
    if isinstance(exc, GeminiError):
        return exc.exit_code
    if isinstance(exc, GFlowError):
        return exit_code_for(exc)
    return 1


def default_debug_dir(*, home: Path | None = None) -> Path:
    """Return the diagnostic dump directory for Gemini Web failures.

    Prefers ``GFLOW_GEMINI_DEBUG_DIR``, then ``<repo>/tmp/gemini-debug`` when running
    from a source checkout, falling back to ``<home>/gemini-debug``.
    """
    env_override = os.environ.get("GFLOW_GEMINI_DEBUG_DIR", "").strip()
    if env_override:
        return Path(env_override).expanduser()
    try:
        repo_root = Path(__file__).resolve().parents[3]
        if (repo_root / "pyproject.toml").is_file():
            return repo_root / "tmp" / "gemini-debug"
    except IndexError:
        pass
    base_home = home if home is not None else get_settings().home
    return base_home / "gemini-debug"


def _empty_str_list() -> list[str]:
    return []


@dataclass(slots=True)
class GeminiProfileStatus:
    """Status snapshot for an isolated Gemini Chrome profile."""

    ok: bool
    profile: str
    profile_dir: str
    exists: bool
    locked: bool
    lock_path: str
    lock_owner_pid: int | None
    chrome_available: bool
    chrome_binary: str | None
    logged_in: bool | None
    verified_live: bool
    account_hint: str | None = None
    detail: str = ""
    warnings: list[str] = field(default_factory=_empty_str_list)
    risk: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



@dataclass(slots=True)
class GeminiLoginResult:
    """Structured result from interactive ``login_profile``."""

    ok: bool
    profile: str
    profile_dir: str
    logged_in: bool
    chat_url: str
    account_hint: str | None
    elapsed_seconds: float
    detail: str
    warnings: list[str] = field(default_factory=_empty_str_list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



@dataclass(slots=True)
class GeminiChatResult:
    """Structured result from ``chat(...)``."""

    ok: bool
    profile: str
    chat_url: str
    model_requested: str | None
    model_selected: str | None
    text: str
    extracted_json: Any | None
    attachments: list[str]
    elapsed_seconds: float
    out_file: str | None
    warnings: list[str] = field(default_factory=_empty_str_list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



@dataclass(slots=True)
class GeminiImageMetadata:
    """Metadata recorded for each downloaded generated image."""

    path: str
    width: int
    height: int
    aspect_ratio: str
    bytes: int
    sha256: str
    mime_type: str
    download_method: str
    watermark_stripped: bool
    watermark_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



@dataclass(slots=True)
class GeminiImageResult:
    """Structured result from ``generate_image(...)``."""

    ok: bool
    profile: str
    chat_url: str
    model_requested: str | None
    model_selected: str | None
    aspect: str | None
    ref_paths: list[str]
    out_file: str
    images: list[GeminiImageMetadata]
    width: int
    height: int
    aspect_ratio: str
    bytes: int
    sha256: str
    download_method: str
    watermark_stripped: bool
    watermark_note: str | None
    response_text: str
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=_empty_str_list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _empty_batch_outcomes() -> list[GeminiImageBatchItemOutcome]:
    return []


def _empty_image_metadata_list() -> list[GeminiImageMetadata]:
    return []


def _empty_int_list() -> list[int]:
    return []


@dataclass(slots=True)
class GeminiImageBatchItem:
    """Single item in a ``generate_images_batch`` manifest."""

    id: str
    prompt: str
    out_file: str
    ref_paths: list[str] = field(default_factory=_empty_str_list)
    aspect: str | None = None
    model: str | None = None
    strip_watermark: bool | None = None
    chat_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, index: int = 0) -> GeminiImageBatchItem:
        raw_id = str(data.get("id") or f"item_{index + 1:02d}").strip()
        prompt = str(data.get("prompt") or "").strip()
        out_file = str(data.get("out_file") or "").strip()
        if not prompt:
            raise GeminiConfigError(f"Batch item {raw_id!r} is missing a non-empty 'prompt'.")
        if not out_file:
            raise GeminiConfigError(f"Batch item {raw_id!r} is missing a non-empty 'out_file'.")

        raw_refs = data.get("ref_paths")
        if raw_refs is None:
            raw_refs = data.get("refs")
        if raw_refs is None:
            raw_refs = data.get("ref")
        ref_list: list[str] = []
        if isinstance(raw_refs, str):
            if raw_refs.strip():
                ref_list = [raw_refs.strip()]
        elif isinstance(raw_refs, (list, tuple)):
            for entry in cast("list[Any] | tuple[Any, ...]", raw_refs):
                s = str(entry).strip()
                if s:
                    ref_list.append(s)

        def _opt_str(val: Any) -> str | None:
            if isinstance(val, str):
                cleaned = val.strip()
                return cleaned or None
            return None

        aspect = _opt_str(data.get("aspect"))
        model = _opt_str(data.get("model"))
        sw_raw = data.get("strip_watermark")
        strip_watermark = bool(sw_raw) if isinstance(sw_raw, bool) else None
        chat_url = _opt_str(data.get("chat_url"))

        return cls(
            id=raw_id,
            prompt=prompt,
            out_file=out_file,
            ref_paths=ref_list,
            aspect=aspect,
            model=model,
            strip_watermark=strip_watermark,
            chat_url=chat_url,
        )


@dataclass(slots=True)
class GeminiImageBatchItemOutcome:
    """Per-item outcome recorded in ``GeminiImageBatchResult``."""

    id: str
    status: str  # "completed" | "skipped" | "failed" | "deferred"
    out_file: str
    chat_url: str | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: str | None = None
    bytes: int | None = None
    sha256: str | None = None
    download_method: str | None = None
    watermark_stripped: bool = False
    watermark_note: str | None = None
    elapsed_seconds: float = 0.0
    error_code: str | None = None
    exit_code: int | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeminiImageBatchResult:
    """Structured result from ``generate_images_batch(...)``."""

    ok: bool
    partial: bool
    profile: str
    reuse_chat: bool
    chat_url: str | None
    total: int
    completed: list[GeminiImageBatchItemOutcome] = field(default_factory=_empty_batch_outcomes)
    skipped: list[GeminiImageBatchItemOutcome] = field(default_factory=_empty_batch_outcomes)
    failed: list[GeminiImageBatchItemOutcome] = field(default_factory=_empty_batch_outcomes)
    deferred: list[GeminiImageBatchItemOutcome] = field(default_factory=_empty_batch_outcomes)
    items: list[GeminiImageBatchItemOutcome] = field(default_factory=_empty_batch_outcomes)
    stop_error_code: str | None = None
    stop_exit_code: int | None = None
    stop_detail: str | None = None
    hard_halt: bool = False
    risk_reason: str | None = None
    retry_after_seconds: float | None = None
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=_empty_str_list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_image_batch_manifest(manifest_path: Path | str) -> list[GeminiImageBatchItem]:
    """Load and validate an image-batch JSON manifest (list or ``{\"items\": [...]}``)."""
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        raise GeminiConfigError(f"Image batch manifest not found: {manifest_path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GeminiConfigError(f"Failed to parse image batch manifest {path}: {exc}") from exc

    entries: list[Any]
    if isinstance(raw, list):
        entries = cast("list[Any]", raw)
    elif isinstance(raw, dict) and isinstance(cast("dict[str, Any]", raw).get("items"), list):
        entries = cast("list[Any]", cast("dict[str, Any]", raw)["items"])
    else:
        raise GeminiConfigError(
            f"Image batch manifest {path} must be a JSON list or an object with an 'items' list."
        )

    items: list[GeminiImageBatchItem] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise GeminiConfigError(f"Manifest item at index {idx} must be a JSON object.")
        items.append(GeminiImageBatchItem.from_dict(cast("Mapping[str, Any]", entry), index=idx))
    if not items:
        raise GeminiConfigError(f"Image batch manifest {path} contains zero items.")
    return items


@dataclass(slots=True)
class GeminiTaskSubmitResult:
    """Structured result from ``submit_task(...)`` (async Gemini Ultra Tasks mode)."""

    ok: bool
    profile: str
    chat_url: str
    attachments: list[str] = field(default_factory=_empty_str_list)
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=_empty_str_list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeminiTaskHarvestResult:
    """Structured result from ``harvest_task(...)`` (async Gemini Ultra Tasks harvest)."""

    ok: bool
    profile: str
    chat_url: str
    completed: bool
    status_label: str
    text: str
    extracted_json: Any | None
    exported_files: list[str] = field(default_factory=_empty_str_list)
    images: list[GeminiImageMetadata] = field(default_factory=_empty_image_metadata_list)
    downloaded_count: int = 0
    expected_images: int | None = None
    missing_image_indices: list[int] = field(default_factory=_empty_int_list)
    elapsed_seconds: float = 0.0
    out_file: str | None = None
    warnings: list[str] = field(default_factory=_empty_str_list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _try_parse_json(raw: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False, None


def extract_json_from_text(text: str) -> Any | None:
    """Extract a JSON object or array from a Gemini response.

    Priority order:
    1. Fenced ```` ```json ... ``` ```` blocks (largest valid object/array preferred).
    2. Any fenced ```` ``` ... ``` ```` block containing valid JSON object/array.
    3. Entire stripped text as valid JSON.
    4. Largest valid outermost JSON object ``{...}`` or array ``[...]`` scanned via
       ``json.JSONDecoder.raw_decode``.
    """
    if not text or not text.strip():
        return None

    json_fenced: list[tuple[int, Any]] = []
    other_fenced: list[tuple[int, Any]] = []
    for match in _FENCED_JSON_RE.finditer(text):
        lang = match.group("lang").strip().lower()
        body = match.group("body").strip()
        if not body:
            continue
        ok, parsed = _try_parse_json(body)
        if not ok:
            continue
        bucket = json_fenced if lang in ("json", "jsonc") else other_fenced
        bucket.append((len(body), parsed))

    if json_fenced:
        json_fenced.sort(key=lambda item: item[0], reverse=True)
        return json_fenced[0][1]
    if other_fenced:
        other_fenced.sort(key=lambda item: item[0], reverse=True)
        return other_fenced[0][1]

    stripped = text.strip()
    ok, direct = _try_parse_json(stripped)
    if ok and isinstance(direct, (dict, list)):
        return direct  # type: ignore[return-value]

    decoder = json.JSONDecoder()
    best_span = -1
    best_value: Any = None
    idx = 0
    length = len(text)
    while idx < length:
        ch = text[idx]
        if ch in ("{", "["):
            try:
                value: Any
                value, end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                idx += 1
                continue
            span = end - idx
            if isinstance(value, (dict, list)) and span > best_span:
                best_span = span
                best_value = cast("Any", value)
            idx = max(idx + 1, end)
        else:
            idx += 1

    return best_value
