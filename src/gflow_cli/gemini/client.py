"""Playwright browser-automation client for Google Gemini Web (chat + image generation)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import mimetypes
import random
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import structlog

from gflow_cli.gemini.models import (
    DEFAULT_GEMINI_PROFILE,
    GEMINI_APP_URL,
    GeminiAuthRequiredError,
    GeminiChallengeDetectedError,
    GeminiChatResult,
    GeminiConfigError,
    GeminiDownloadError,
    GeminiError,
    GeminiErrorCode,
    GeminiImageBatchItem,
    GeminiImageBatchItemOutcome,
    GeminiImageBatchResult,
    GeminiImageMetadata,
    GeminiImageResult,
    GeminiNoImageGeneratedError,
    GeminiRateLimitError,
    GeminiRiskBlockedError,
    GeminiTaskHarvestResult,
    GeminiTaskSubmitResult,
    GeminiTimeoutError,
    GeminiUiDriftError,
    default_debug_dir,
    extract_json_from_text,
)
from gflow_cli.gemini.profiles import (
    COMPOSER_SELECTORS,
    GeminiProfileLease,
    build_gemini_launch_kwargs,
    detect_gemini_authenticated,
    detect_gemini_composer,
    gemini_profile_dir,
    get_async_playwright,
    has_google_auth_cookies,
    is_auth_redirect_url,
    is_gemini_app_url,
    validate_gemini_profile_name,
    wrap_launch_error,
)
from gflow_cli.gemini.risk_guard import (
    DOWNLOAD_RETRY_ATTEMPTS,
    DOWNLOAD_RETRY_BASE_S,
    DOWNLOAD_RETRY_CAP_S,
    GeminiRiskLedger,
    find_active_kill_switch,
    jitter_delay,
    resolve_flow_ledger_dir,
)
from gflow_cli.gemini.watermark import (
    compute_aspect_ratio_label,
    finalize_image_file,
    inspect_image_bytes,
)
from gflow_cli.paths import extension_from_magic

if TYPE_CHECKING:
    from types import TracebackType

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Resilient Gemini Web DOM selector cascades (informed by gemflow / AIkaptan)
# ---------------------------------------------------------------------------

PROMO_DISMISS_SELECTORS: tuple[str, ...] = (
    '.cdk-overlay-container button:has-text("No, thanks")',
    '.cdk-overlay-container button:has-text("Not now")',
    '.cdk-overlay-container button:has-text("Got it")',
    '.cdk-overlay-container button:has-text("Dismiss")',
    '.cdk-overlay-container button:has-text("Close")',
    '.cdk-overlay-container button:has-text("不用了")',
    '.cdk-overlay-container button:has-text("暂不")',
    '.cdk-overlay-container button:has-text("知道了")',
    '.cdk-overlay-container button:has-text("关闭")',
    '.cdk-overlay-container button[aria-label*="Close" i]',
)

SEND_BUTTON_SELECTORS: tuple[str, ...] = (
    'button[aria-label*="Send message" i]',
    'button[aria-label*="Send" i]',
    'button[aria-label="发送"]',
    'button[aria-label*="发送"]',
    'button:has(mat-icon[fonticon="arrow_upward"])',
    "button.send-button",
    'button[data-mat-icon-name="send"]',
)

STOP_BUTTON_SELECTORS: tuple[str, ...] = (
    'button[aria-label*="Stop generating" i]',
    'button[aria-label*="Stop response" i]',
    'button[aria-label*="Stop" i]',
    'button[aria-label*="停止"]',
    'button:has(mat-icon[fonticon="stop"])',
)

RESPONSE_TURN_SELECTORS: tuple[str, ...] = (
    "model-response",
    "response-container",
    '[data-test-id="model-response"]',
)

RESPONSE_TEXT_SELECTORS: tuple[str, ...] = (
    "message-content .markdown",
    "message-content",
    ".model-response-text",
    ".markdown",
    ".response-content",
)

RESPONSE_DONE_SELECTORS: tuple[str, ...] = (
    'button[aria-label="Good response"]',
    'button[aria-label="答得好"]',
    'button[aria-label="好回答"]',
    'button[aria-label*="Copy" i]',
    'button[aria-label*="复制"]',
    'button:has(mat-icon[fonticon="thumb_up"])',
    'button:has(mat-icon[fonticon="copy"])',
    "message-actions",
)

UPLOAD_BUTTON_SELECTORS: tuple[str, ...] = (
    '[data-test-id="local-images-files-uploader-button"]',
    '[data-test-id="hidden-local-image-upload-button"]',
    '[data-test-id="hidden-local-file-upload-button"]',
    'gem-menu-item[aria-label*="上传文件"]',
    'gem-menu-item:has-text("上传文件")',
    'gem-menu-item:has(mat-icon[fonticon="attach_file"])',
    'button[aria-label*="Upload file" i]',
    'button[aria-label*="Upload image" i]',
    'button[aria-label*="Upload" i]',
    'button[aria-label*="Add file" i]',
    'button[aria-label*="上传文件"]',
    'button[aria-label*="上传"]',
    'button[aria-label*="添加文件"]',
    'button[aria-label*="添加"]',
)

UPLOAD_MENU_TRIGGER_SELECTORS: tuple[str, ...] = (
    'button[aria-label="上传和工具"]',
    'button[aria-label*="上传和工具"]',
    'button:has(mat-icon[fonticon="plus"])',
    'button[aria-label*="Open upload file menu" i]',
    'button[aria-label*="Add files" i]',
    'button[aria-label*="Add" i]',
    'button[aria-label*="打开上传"]',
    'button:has(mat-icon[fonticon="add"])',
)

UPLOAD_PREVIEW_SELECTORS: tuple[str, ...] = (
    'img[data-test-id="image-preview"][src^="blob:"]',
    'img[data-test-id="image-preview"]',
    ".uploader-file-preview-container",
    ".file-preview-wrapper",
    '[data-test-id*="file-preview"]',
    "uploader-file-preview",
)

MODEL_PICKER_SELECTORS: tuple[str, ...] = (
    'button[data-test-id="bard-mode-menu-button"]',
    "bard-mode-switcher button",
    'button[data-test-id*="model-picker"]',
    'button[data-test-id*="bard-mode"]',
    ".logo-pill-btn",
    'button[aria-label*="model" i]',
    'button[aria-label*="模型"]',
)

MODEL_OPTION_SELECTORS: tuple[str, ...] = (
    'gem-menu[data-test-id="gem-mode-menu"] gem-menu-item[role="menuitemradio"]',
    '[role="menu"] [role="menuitemradio"]',
    ".cdk-overlay-popover gem-menu-item",
    'gem-menu-item[data-test-id^="bard-mode-option-"]',
    ".cdk-overlay-container button",
    '.cdk-overlay-container [role="menuitem"]',
    '.cdk-overlay-container [role="menuitemradio"]',
    '.cdk-overlay-container [role="option"]',
)

TOOLS_BUTTON_SELECTORS: tuple[str, ...] = (
    'button:has-text("Tools")',
    'button:has-text("工具")',
    'button[aria-label*="Tools" i]',
    'button[aria-label="上传和工具"]',
    'button[aria-label*="工具"]',
    'button:has(mat-icon[fonticon="plus"])',
)

CREATE_IMAGE_OPTION_SELECTORS: tuple[str, ...] = (
    '.cdk-overlay-container button:has-text("Create images")',
    '.cdk-overlay-container button:has-text("Generate images")',
    '.cdk-overlay-container button:has-text("Create image")',
    '.cdk-overlay-container button:has-text("生成图片")',
    '.cdk-overlay-container button:has-text("创建图片")',
    '.cdk-overlay-container button:has-text("制作图片")',
    '.cdk-overlay-container gem-menu-item:has-text("生成图片")',
    '.cdk-overlay-container gem-menu-item:has-text("创建图片")',
    '.cdk-overlay-container gem-menu-item:has-text("Create images")',
    '.cdk-overlay-container [role="menuitem"]:has-text("image")',
)

DOWNLOAD_IMAGE_BUTTON_SELECTORS: tuple[str, ...] = (
    'button[aria-label*="Download full size" i]',
    'button[aria-label*="Download" i]',
    'button[aria-label*="下载完整尺寸"]',
    'button[aria-label*="下载原图"]',
    'button[aria-label*="下载"]',
    '[data-test-id="download-generated-image-button"]',
    'button:has(mat-icon[fonticon="download"])',
    'a[aria-label*="Download" i]',
    'button[data-test-id*="download"]',
)

# Gemini Ultra "任务" (Tasks / Neon Cloud Agent) selectors (verified live in lr26)
TASK_MODE_PILL_SELECTORS: tuple[str, ...] = (
    'button.gem-segmented-button-action:has-text("任务")',
    'button.gem-segmented-button-action:has-text("Tasks")',
    'button[role="radio"]:has-text("任务")',
    'button[role="radio"]:has-text("Tasks")',
    'button[role="tab"]:has-text("任务")',
    'button[role="tab"]:has-text("Tasks")',
    'button:has-text("任务")',
    'button:has-text("Tasks")',
)

TASK_ARTIFACT_DOWNLOAD_SELECTORS: tuple[str, ...] = (
    'a[download]',
    'a:has-text(".json")',
    'button:has-text(".json")',
    'button:has-text("导出")',
    'button:has-text("Export")',
)

TASK_IMAGE_DOWNLOAD_SELECTORS: tuple[str, ...] = (
    'model-response button[aria-label*="下载完整尺寸"]',
    'response-container button[aria-label*="下载完整尺寸"]',
    'generated-image button[aria-label*="下载完整尺寸"]',
    'single-image button[aria-label*="下载完整尺寸"]',
    'button[aria-label*="下载完整尺寸"]',
    'model-response button[aria-label*="Download full size" i]',
    'response-container button[aria-label*="Download full size" i]',
    'button[aria-label*="Download full size" i]',
    '[data-test-id="download-generated-image-button"]',
    'model-response button:has(mat-icon[fonticon="download"])',
)

_IMAGE_FILE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})

# ---------------------------------------------------------------------------
# Risk-signal & URL normalization regexes
# ---------------------------------------------------------------------------

_CHALLENGE_URL_MARKERS: tuple[str, ...] = (
    "google.com/sorry",
    "sorry/index",
    "v3/signin/challenge",
    "v3/signin/rejected",
    "recaptcha",
)

_CHALLENGE_TEXT_RE = re.compile(
    r"(confirm it['’]s you|verify it['’]s you|unusual traffic|unusual activity|"
    r"verify you are human|complete the captcha|i['’]m not a robot|"
    r"验证您的身份|异常流量|人机身份验证)",
    re.IGNORECASE,
)

_RATE_LIMIT_TEXT_RE = re.compile(
    r"(too many requests|reached your limit|image generation limit|"
    r"quota exceeded|rate limit|try again later|temporarily unavailable|"
    r"you['’]ve reached the limit|已达到.*上限|请求过于频繁|稍后再试|次数已达上限)",
    re.IGNORECASE,
)

_GOOGLEUSERCONTENT_SIZE_RE = re.compile(r"=(?:s\d+|w\d+(?:-h\d+)?)(?:-[A-Za-z0-9_-]+)*(?=[?&#]|$)")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_SECRET_REDACT_RE = re.compile(
    r"(Bearer\s+[A-Za-z0-9._\-+/=]{12,}|"
    r"(?:SAPISID|__Secure-1PSID|__Secure-3PSID|SID|HSID|SSID)=[^;\s\"'<>]{8,}|"
    r"AIza[0-9A-Za-z\-_]{20,})",
    re.IGNORECASE,
)
_ASPECT_TOKEN_RE = re.compile(r"\b\d{1,2}\s*[:x×]\s*\d{1,2}\b", re.IGNORECASE)


def upgrade_googleusercontent_url(url: str, *, size_spec: str = "s0") -> str:
    """Rewrite a ``googleusercontent.com`` image URL to request full resolution (``=s0``).

    Handles ``=s512``, ``=w1024-h576-rj``, or URLs with no size suffix yet.
    Non-Google-CDN URLs (``data:``, ``blob:``, other hosts) are returned unchanged.
    """
    if not url or url.startswith(("data:", "blob:")):
        return url
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if "googleusercontent.com" not in host:
        return url
    spec = size_spec.lstrip("=")
    if _GOOGLEUSERCONTENT_SIZE_RE.search(url):
        return _GOOGLEUSERCONTENT_SIZE_RE.sub(f"={spec}", url, count=1)
    if "=" in parsed.path.rsplit("/", 1)[-1]:
        return url
    if parsed.query:
        base, query = url.split("?", 1)
        return f"{base}={spec}?{query}"
    return f"{url}={spec}"


def inject_aspect_hint(prompt: str, aspect: str | None) -> str:
    """Ensure ``aspect`` guidance (e.g. ``16:9``) is cleanly present in ``prompt``."""
    cleaned_prompt = prompt.strip()
    if not aspect or not aspect.strip():
        return cleaned_prompt
    norm_aspect = aspect.strip()
    # Avoid duplicating if prompt already contains this exact aspect token.
    compact = re.sub(r"\s+", "", norm_aspect).lower()
    for match in _ASPECT_TOKEN_RE.finditer(cleaned_prompt):
        found = re.sub(r"\s+", "", match.group(0)).lower().replace("x", ":").replace("×", ":")
        if found == compact:
            return cleaned_prompt
    orientation = ""
    if compact == "16:9":
        orientation = " widescreen horizontal landscape"
    elif compact == "9:16":
        orientation = " vertical portrait"
    elif compact == "1:1":
        orientation = " square"
    return (
        f"{cleaned_prompt}\n\n"
        f"Image composition requirement: generate in strict {norm_aspect}"
        f"{orientation} aspect ratio."
    )


def classify_page_risk(
    url: str,
    body_text: str = "",
    *,
    http_statuses: Sequence[int] = (),
) -> GeminiError | None:
    """Classify auth, CAPTCHA/WAF challenge, or 429 rate-limit states without blind retry."""
    low_url = (url or "").lower()
    statuses = set(http_statuses)

    if 429 in statuses or (body_text and _RATE_LIMIT_TEXT_RE.search(body_text)):
        return GeminiRateLimitError(
            f"Gemini Web rate limit or quota reached (url={url}, statuses={sorted(statuses)})."
        )

    if (
        403 in statuses
        or any(marker in low_url for marker in _CHALLENGE_URL_MARKERS)
        or (body_text and _CHALLENGE_TEXT_RE.search(body_text))
    ):
        return GeminiChallengeDetectedError(
            f"Google verification / CAPTCHA / WAF challenge detected "
            f"(url={url}, statuses={sorted(statuses)})."
        )

    if is_auth_redirect_url(low_url):
        return GeminiAuthRequiredError(
            f"Gemini Web session redirected to Google sign-in ({url})."
        )

    return None


def sanitize_dom_snippet(html: str, *, max_chars: int = 24_000) -> str:
    """Strip scripts/styles and redact potential credentials before writing debug DOM dumps."""
    stripped = _SCRIPT_STYLE_RE.sub("", html or "")
    redacted = _SECRET_REDACT_RE.sub("[REDACTED]", stripped)
    if len(redacted) > max_chars:
        return redacted[:max_chars] + "\n<!-- truncated -->\n"
    return redacted


async def save_failure_diagnostics(
    page: Any,
    *,
    debug_dir: Path,
    tag: str,
) -> tuple[Path | None, Path | None]:
    """Capture a diagnostic PNG screenshot and sanitized DOM snippet on failure."""
    screenshot_path: Path | None = None
    dom_path: Path | None = None
    safe_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", tag).strip("_") or "error"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None, None

    if page is None:
        return None, None

    try:
        shot = debug_dir / f"{stamp}_{safe_tag}.png"
        await page.screenshot(path=str(shot), full_page=False, timeout=10_000)
        screenshot_path = shot
    except Exception:
        screenshot_path = None

    try:
        html = await page.content()
        dom = debug_dir / f"{stamp}_{safe_tag}.html"
        dom.write_text(sanitize_dom_snippet(str(html)), encoding="utf-8")
        dom_path = dom
    except Exception:
        dom_path = None

    return screenshot_path, dom_path


def _validate_existing_files(paths: Sequence[Path | str], *, label: str) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if not p.is_file():
            raise GeminiConfigError(f"{label} file not found: {raw}")
        resolved.append(p)
    return resolved


class GeminiWebClient:
    """Playwright persistent-context automation client for ``https://gemini.google.com/app``.

    Manages:
    - Exclusive ``GeminiProfileLease`` on ``<home>/gemini_profile_<profile>``
    - Anti-risk preflight, pacing, and quarantine accounting via ``GeminiRiskLedger``
    - Chat/script generation with multi-file attachments & JSON extraction
    - Text-to-image and reference-guided image generation with full-res download
    """

    def __init__(
        self,
        *,
        profile: str = DEFAULT_GEMINI_PROFILE,
        headless: bool = False,
        chrome_executable: str | None = None,
        proxy: str | None = None,
        home: Path | None = None,
        debug_dir: Path | None = None,
        min_image_dim: int = 512,
        enforce_risk_guard: bool = True,
        flow_ledger_dir: Path | str | None = None,
        linked_flow_profile: str | None = None,
        allow_flow_overlap: bool = False,
    ) -> None:
        self.profile = validate_gemini_profile_name(profile)
        self.headless = headless
        self.chrome_executable = chrome_executable
        self.proxy = proxy
        self.home = home
        self.profile_dir = gemini_profile_dir(self.profile, home=home)
        self.debug_dir = debug_dir if debug_dir is not None else default_debug_dir(home=home)
        self.min_image_dim = max(64, int(min_image_dim))
        self.enforce_risk_guard = enforce_risk_guard
        self.flow_ledger_dir = flow_ledger_dir
        self.linked_flow_profile = linked_flow_profile
        self.allow_flow_overlap = allow_flow_overlap
        self.ledger = GeminiRiskLedger(home=home)

        self._lease: GeminiProfileLease | None = None
        self._pw_cm: Any = None
        self._pw: Any = None
        self._context: Any = None
        self._page: Any = None
        self._observed_statuses: list[int] = []

    async def __aenter__(self) -> GeminiWebClient:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def start(self) -> None:
        """Acquire profile lease and launch isolated Chrome persistent context."""
        if self._context is not None:
            return
        if not self.profile_dir.is_dir():
            raise GeminiAuthRequiredError(
                f"Gemini profile '{self.profile}' does not exist at {self.profile_dir}. "
                f"Run `gflow gemini login --profile {self.profile}` first."
            )

        lease = GeminiProfileLease(self.profile_dir, home=self.home)
        try:
            await lease.aacquire()
        except GeminiError as exc:
            if self.enforce_risk_guard:
                cooldown = self.ledger.record_outcome(
                    self.profile,
                    exit_code=exc.exit_code,
                    reason=exc.code.value,
                )
                if cooldown is not None and isinstance(exc, GeminiRateLimitError):
                    exc.retry_after_seconds = cooldown
                    exc.cooldown_seconds = cooldown
            raise
        self._lease = lease

        launch_kwargs = build_gemini_launch_kwargs(
            self.profile_dir,
            headless=self.headless,
            chrome_executable=self.chrome_executable,
            proxy=self.proxy,
            visible_login=False,
        )
        async_playwright = get_async_playwright()
        self._pw_cm = async_playwright()
        try:
            self._pw = await self._pw_cm.__aenter__()
            self._context = await self._pw.chromium.launch_persistent_context(**launch_kwargs)
            self._page = (
                self._context.pages[0] if self._context.pages else await self._context.new_page()
            )
            self._attach_response_observer(self._page)
        except Exception as exc:
            await self.close()
            wrapped = wrap_launch_error(exc, self.profile_dir)
            if self.enforce_risk_guard:
                self.ledger.record_outcome(
                    self.profile,
                    exit_code=wrapped.exit_code,
                    reason=wrapped.code.value,
                )
            raise wrapped from exc

    async def close(self) -> None:
        """Close Playwright context and release profile lease cleanly (bounded teardown)."""
        try:
            if self._context is not None:
                ctx, self._context = self._context, None
                with contextlib.suppress(BaseException):
                    await asyncio.wait_for(ctx.close(), timeout=15.0)
            self._page = None
            if self._pw_cm is not None:
                cm, self._pw_cm = self._pw_cm, None
                self._pw = None
                with contextlib.suppress(BaseException):
                    await asyncio.wait_for(cm.__aexit__(None, None, None), timeout=10.0)
        finally:
            if self._lease is not None:
                lease, self._lease = self._lease, None
                lease.release()

    def _attach_response_observer(self, page: Any) -> None:
        self._observed_statuses.clear()

        def _on_response(response: Any) -> None:
            try:
                status = int(response.status)
                url = str(response.url or "").lower()
            except Exception:
                return
            if status in (403, 429) and (
                "gemini.google.com" in url or "accounts.google.com" in url
            ):
                self._observed_statuses.append(status)

        with contextlib.suppress(Exception):
            page.on("response", _on_response)

    async def _visible_body_text(self, page: Any) -> str:
        try:
            text = await page.evaluate(
                "() => (document.body && document.body.innerText ? "
                "document.body.innerText.slice(0, 4000) : '')"
            )
            return str(text or "")
        except Exception:
            return ""

    async def _check_page_risk_or_raise(self, page: Any, *, tag: str) -> None:
        url = str(getattr(page, "url", "") or "")
        body = await self._visible_body_text(page)
        risk_err = classify_page_risk(url, body, http_statuses=self._observed_statuses)
        if risk_err is not None:
            shot, dom = await save_failure_diagnostics(page, debug_dir=self.debug_dir, tag=tag)
            risk_err.screenshot_path = shot
            risk_err.dom_path = dom
            raise risk_err

    async def _dismiss_promo_overlays(self, page: Any) -> None:
        for selector in PROMO_DISMISS_SELECTORS:
            try:
                loc = page.locator(selector).first
                if await loc.is_visible(timeout=250):
                    await loc.click(timeout=1_500)
                    await asyncio.sleep(jitter_delay(0.25))
            except Exception:
                continue
        with contextlib.suppress(Exception):
            backdrop = page.locator(".cdk-overlay-backdrop").first
            if await backdrop.is_visible(timeout=250):
                await page.keyboard.press("Escape")
                await asyncio.sleep(jitter_delay(0.2))

    async def _is_page_hydrated(self, page: Any) -> bool:
        """Confirm Angular SSR shell has finished hydrating before interacting with composer.

        In Gemini's ``lr26`` UI, ``rich-textarea`` renders in SSR HTML before Angular
        attaches click handlers; clicking Send while ``document.readyState !== 'complete'``
        or while body text is still empty silently drops the click.
        """
        try:
            probe = await page.evaluate(
                """() => ({
                    readyState: document.readyState || '',
                    textLen: (document.body && document.body.innerText
                        ? document.body.innerText.trim().length : 0),
                })"""
            )
        except Exception:
            return True
        if not isinstance(probe, dict):
            return True
        data = cast("dict[str, Any]", probe)
        return data.get("readyState") == "complete" and int(data.get("textLen") or 0) > 10

    async def _navigate_and_ready(
        self,
        page: Any,
        *,
        chat_url: str | None,
        new_chat: bool,
        timeout_seconds: float,
    ) -> None:
        target = GEMINI_APP_URL if (new_chat or not chat_url) else chat_url
        self._observed_statuses.clear()
        nav_timeout_ms = max(15_000, int(min(timeout_seconds, 60.0) * 1000))
        try:
            await page.goto(target, wait_until="load", timeout=nav_timeout_ms)
        except Exception:
            await page.goto(target, wait_until="domcontentloaded", timeout=nav_timeout_ms)

        deadline = time.monotonic() + min(timeout_seconds, 45.0)
        ctx = self._context if self._context is not None else getattr(page, "context", None)
        guest_checks = 0
        while time.monotonic() < deadline:
            await self._dismiss_promo_overlays(page)
            await self._check_page_risk_or_raise(page, tag="ready_check")
            is_auth, _hint = await detect_gemini_authenticated(ctx, page)
            if is_auth:
                if await self._is_page_hydrated(page):
                    await asyncio.sleep(jitter_delay(0.8))
                    return
            url = str(getattr(page, "url", "") or "")
            if is_gemini_app_url(url) and not await has_google_auth_cookies(ctx):
                # If the page already rendered the guest composer or sign-in CTA with no
                # Google auth cookies, fail fast as AUTH_REQUIRED instead of using guest mode.
                guest_checks += 1
                if guest_checks >= 2 or await detect_gemini_composer(page):
                    shot, dom = await save_failure_diagnostics(
                        page, debug_dir=self.debug_dir, tag="guest_mode_unauthenticated"
                    )
                    raise GeminiAuthRequiredError(
                        "Gemini Web loaded in logged-out guest mode (no Google auth session "
                        f"cookies found for profile '{self.profile}'). "
                        f"Run `gflow gemini login --profile {self.profile}` first.",
                        screenshot_path=shot,
                        dom_path=dom,
                    )
            await asyncio.sleep(jitter_delay(0.8))

        await self._check_page_risk_or_raise(page, tag="ready_timeout")
        shot, dom = await save_failure_diagnostics(
            page, debug_dir=self.debug_dir, tag="no_composer"
        )
        raise GeminiAuthRequiredError(
            f"Gemini prompt composer did not appear within ready timeout (url={page.url}). "
            "Your Google session may have expired.",
            screenshot_path=shot,
            dom_path=dom,
        )

    async def _find_composer(self, page: Any) -> Any:
        for selector in COMPOSER_SELECTORS:
            try:
                loc = page.locator(selector).first
                if await loc.is_visible(timeout=600):
                    return loc
            except Exception:
                continue
        shot, dom = await save_failure_diagnostics(page, debug_dir=self.debug_dir, tag="composer")
        raise GeminiUiDriftError(
            "Could not locate Gemini Web prompt composer input.",
            screenshot_path=shot,
            dom_path=dom,
        )

    async def _switch_model_best_effort(
        self,
        page: Any,
        model: str,
    ) -> tuple[str | None, str | None]:
        target_low = model.strip().lower()
        if not target_low:
            return None, None
        picker: Any = None
        for selector in MODEL_PICKER_SELECTORS:
            try:
                loc = page.locator(selector).first
                if await loc.is_visible(timeout=600):
                    picker = loc
                    break
            except Exception:
                continue
        if picker is None:
            return (
                None,
                f"Model picker button not found; keeping current Gemini model "
                f"(requested={model!r}).",
            )

        try:
            current_label = (await picker.inner_text(timeout=1_000)).strip()
            current_aria = ""
            with contextlib.suppress(Exception):
                current_aria = str(await picker.get_attribute("aria-label") or "").strip()
            if target_low in current_label.lower() or target_low in current_aria.lower():
                return current_label or current_aria, None

            await picker.click(timeout=2_500)
            await asyncio.sleep(jitter_delay(0.4))

            for opt_selector in MODEL_OPTION_SELECTORS:
                options = page.locator(opt_selector)
                count = 0
                with contextlib.suppress(Exception):
                    count = await options.count()
                if count <= 0:
                    continue
                for idx in range(min(count, 20)):
                    opt = options.nth(idx)
                    try:
                        if not await opt.is_visible(timeout=300):
                            continue
                        text = " ".join((await opt.inner_text(timeout=600)).split())
                        opt_aria = ""
                        opt_mode_id = ""
                        opt_test_id = ""
                        with contextlib.suppress(Exception):
                            opt_aria = str(await opt.get_attribute("aria-label") or "")
                        with contextlib.suppress(Exception):
                            opt_mode_id = str(await opt.get_attribute("data-mode-id") or "")
                        with contextlib.suppress(Exception):
                            opt_test_id = str(await opt.get_attribute("data-test-id") or "")
                        haystack = " ".join((text, opt_aria, opt_mode_id, opt_test_id)).lower()
                        if target_low not in haystack:
                            continue

                        await opt.click(timeout=2_000)
                        await asyncio.sleep(jitter_delay(0.4))

                        after_label = ""
                        after_aria = ""
                        with contextlib.suppress(Exception):
                            after_label = (await picker.inner_text(timeout=1_000)).strip()
                        with contextlib.suppress(Exception):
                            after_aria = str(await picker.get_attribute("aria-label") or "").strip()

                        verified = (
                            target_low in after_label.lower()
                            or target_low in after_aria.lower()
                            or (
                                bool(after_label or after_aria)
                                and (after_label != current_label or after_aria != current_aria)
                            )
                        )
                        if not verified:
                            still_open = False
                            with contextlib.suppress(Exception):
                                still_open = bool(await opt.is_visible(timeout=250))
                            if still_open:
                                with contextlib.suppress(Exception):
                                    await opt.click(force=True, timeout=2_000)
                                    await asyncio.sleep(jitter_delay(0.4))
                                    after_label = (await picker.inner_text(timeout=1_000)).strip()
                                    after_aria = str(
                                        await picker.get_attribute("aria-label") or ""
                                    ).strip()
                                    verified = (
                                        target_low in after_label.lower()
                                        or target_low in after_aria.lower()
                                        or after_label != current_label
                                        or after_aria != current_aria
                                    )

                        log.info(
                            "gemini.model_switched",
                            requested=model,
                            clicked_option=text,
                            before_label=current_label,
                            after_label=after_label,
                            after_aria=after_aria,
                            verified=verified,
                            selector=opt_selector,
                        )
                        if verified and (
                            target_low in after_label.lower() or target_low in after_aria.lower()
                        ):
                            return after_label or text, None
                        return text or after_label, None
                    except Exception:
                        continue
            with contextlib.suppress(Exception):
                await page.keyboard.press("Escape")
        except Exception as exc:
            with contextlib.suppress(Exception):
                await page.keyboard.press("Escape")
            return None, f"Model switch best-effort failed ({exc}); keeping current Gemini model."

        return None, f"Requested model {model!r} not found in Gemini picker; keeping current model."

    async def _activate_image_tool_best_effort(self, page: Any) -> tuple[bool, str | None]:
        try:
            composer = await self._find_composer(page)
            await composer.click(timeout=2_000)
            await asyncio.sleep(jitter_delay(0.25))
        except Exception:
            pass

        tools_btn: Any = None
        for selector in TOOLS_BUTTON_SELECTORS:
            try:
                loc = page.locator(selector).first
                if await loc.is_visible(timeout=600):
                    tools_btn = loc
                    break
            except Exception:
                continue
        if tools_btn is None:
            return (
                False,
                "Gemini 'Tools' button not found; relying on native prompt image generation.",
            )

        try:
            await tools_btn.click(timeout=2_000)
            await asyncio.sleep(jitter_delay(0.35))
            for opt_sel in CREATE_IMAGE_OPTION_SELECTORS:
                try:
                    opt = page.locator(opt_sel).first
                    if await opt.is_visible(timeout=500):
                        await opt.click(timeout=2_000)
                        await asyncio.sleep(jitter_delay(0.35))
                        return True, None
                except Exception:
                    continue
            with contextlib.suppress(Exception):
                await page.keyboard.press("Escape")
        except Exception as exc:
            with contextlib.suppress(Exception):
                await page.keyboard.press("Escape")
            return False, f"Image tool activation skipped ({exc}); using native prompt generation."

        return (
            False,
            "Gemini 'Create images' tool option not found; using native prompt generation.",
        )

    async def _wait_for_upload_preview(self, page: Any, *, expected_count: int) -> bool:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            for selector in UPLOAD_PREVIEW_SELECTORS:
                try:
                    cnt = await page.locator(selector).count()
                    if cnt >= min(1, expected_count):
                        await asyncio.sleep(jitter_delay(0.5))
                        return True
                except Exception:
                    continue
            await asyncio.sleep(0.4)
        return False

    async def _paste_files_via_clipboard(self, page: Any, paths: Sequence[Path]) -> bool:
        composer = await self._find_composer(page)
        await composer.click(timeout=2_000)
        payloads: list[dict[str, str]] = []
        for path in paths:
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            payloads.append(
                {
                    "name": path.name,
                    "mime": mime,
                    "b64": base64.b64encode(path.read_bytes()).decode("ascii"),
                }
            )
        try:
            ok = await composer.evaluate(
                """(el, files) => {
                    const dt = new DataTransfer();
                    for (const f of files) {
                        const bin = atob(f.b64);
                        const arr = new Uint8Array(bin.length);
                        for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
                        dt.items.add(new File([arr], f.name, { type: f.mime }));
                    }
                    const evt = new ClipboardEvent('paste', {
                        bubbles: true,
                        cancelable: true,
                        clipboardData: dt,
                    });
                    el.dispatchEvent(evt);
                    return dt.files.length > 0;
                }""",
                payloads,
            )
            return bool(ok)
        except Exception:
            return False

    async def _upload_files(self, page: Any, paths: Sequence[Path]) -> list[str]:
        """Upload 1..N files using 3-tier fallback (input -> filechooser -> synthetic paste)."""
        if not paths:
            return []
        warnings: list[str] = []
        str_paths = [str(p) for p in paths]
        uploaded = False

        # Tier 1: Direct hidden or visible input[type="file"].
        try:
            file_inputs = page.locator('input[type="file"]')
            if await file_inputs.count() > 0:
                await file_inputs.first.set_input_files(str_paths)
                uploaded = True
        except Exception:
            uploaded = False

        # Tier 2: File chooser via upload button (optionally opening '+' upload menu first).
        if not uploaded:
            for trigger_list in (UPLOAD_BUTTON_SELECTORS, UPLOAD_MENU_TRIGGER_SELECTORS):
                for selector in trigger_list:
                    try:
                        btn = page.locator(selector).first
                        if not await btn.is_visible(timeout=600):
                            continue
                        try:
                            async with page.expect_file_chooser(timeout=4_000) as fc_info:
                                await btn.click(timeout=2_000)
                            chooser = await fc_info.value
                            await chooser.set_files(str_paths)
                            uploaded = True
                            break
                        except Exception:
                            # Clicking '+' may open a menu containing "Upload files".
                            for menu_sel in UPLOAD_BUTTON_SELECTORS:
                                try:
                                    menu_item = page.locator(menu_sel).first
                                    if await menu_item.is_visible(timeout=500):
                                        async with page.expect_file_chooser(timeout=4_000) as fc2:
                                            await menu_item.click(timeout=2_000)
                                        chooser2 = await fc2.value
                                        await chooser2.set_files(str_paths)
                                        uploaded = True
                                        break
                                except Exception:
                                    continue
                            if uploaded:
                                break
                    except Exception:
                        continue
                if uploaded:
                    break

        # Tier 3: Synthetic clipboard paste onto focused composer (AIkaptan pattern).
        if not uploaded:
            uploaded = await self._paste_files_via_clipboard(page, paths)

        if not uploaded:
            shot, dom = await save_failure_diagnostics(
                page, debug_dir=self.debug_dir, tag="upload_failed"
            )
            raise GeminiUiDriftError(
                f"Failed to upload {len(paths)} attachment(s) into Gemini Web composer.",
                screenshot_path=shot,
                dom_path=dom,
            )

        preview_ok = await self._wait_for_upload_preview(page, expected_count=len(paths))
        if not preview_ok:
            warnings.append(
                "Attachment upload preview selector was not confirmed before submit; "
                "proceeding after brief settle."
            )
            await asyncio.sleep(jitter_delay(1.5))
        return warnings

    async def _click_send_once(self, page: Any) -> bool:
        for selector in SEND_BUTTON_SELECTORS:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible(timeout=600) and await btn.is_enabled(timeout=600):
                    await btn.click(timeout=2_500)
                    return True
            except Exception:
                continue
        return False

    async def _submission_started(
        self,
        page: Any,
        composer: Any,
        *,
        prior_url: str,
        prior_turns: int,
    ) -> bool:
        if await self._is_streaming_active(page):
            return True
        if str(getattr(page, "url", "") or "") != prior_url:
            return True
        if await self._count_response_turns(page) > prior_turns:
            return True
        try:
            remaining = (await composer.inner_text(timeout=800)).strip()
            return not remaining
        except Exception:
            return True

    async def _enter_prompt_and_submit(self, page: Any, prompt: str) -> None:
        composer = await self._find_composer(page)
        prior_url = str(getattr(page, "url", "") or "")
        prior_turns = await self._count_response_turns(page)
        await composer.click(timeout=2_000)
        await asyncio.sleep(jitter_delay(0.25))

        filled = False
        try:
            await composer.fill(prompt, timeout=10_000)
            current_text = (await composer.inner_text(timeout=1_500)).strip()
            filled = bool(current_text)
        except Exception:
            filled = False

        if not filled:
            # Multi-line safe fallback: insert_text per line + Shift+Enter (never bare Enter).
            lines = prompt.splitlines() or [prompt]
            for idx, line in enumerate(lines):
                if line:
                    await page.keyboard.insert_text(line)
                if idx < len(lines) - 1:
                    await page.keyboard.press("Shift+Enter")

        await asyncio.sleep(jitter_delay(0.45))

        clicked = await self._click_send_once(page)
        if not clicked:
            await page.keyboard.press("Enter")

        # In real browser sessions, confirm the click was accepted by hydrated Angular
        # (composer cleared, stop button visible, URL changed, or new turn rendered).
        # Never re-submit once submission has started.
        if self._context is None:
            return

        for attempt in range(2):
            for _ in range(5):
                await asyncio.sleep(0.5)
                if await self._submission_started(
                    page,
                    composer,
                    prior_url=prior_url,
                    prior_turns=prior_turns,
                ):
                    return
            log.warning(
                "gemini.submit_not_confirmed_retrying_click",
                attempt=attempt + 2,
                url=str(getattr(page, "url", "") or ""),
            )
            if not await self._click_send_once(page):
                with contextlib.suppress(Exception):
                    await composer.click(timeout=1_500)
                await page.keyboard.press("Enter")

    async def _response_turns_locator(self, page: Any) -> Any:
        for selector in RESPONSE_TURN_SELECTORS:
            try:
                loc = page.locator(selector)
                if await loc.count() > 0:
                    return loc
            except Exception:
                continue
        return page.locator(RESPONSE_TURN_SELECTORS[0])

    async def _count_response_turns(self, page: Any) -> int:
        best = 0
        for selector in RESPONSE_TURN_SELECTORS:
            try:
                cnt = await page.locator(selector).count()
                best = max(best, cnt)
            except Exception:
                continue
        return best

    async def _snapshot_existing_image_srcs(self, page: Any) -> set[str]:
        try:
            srcs = await page.evaluate(
                """() => Array.from(document.querySelectorAll('img'))
                    .map(img => img.currentSrc || img.src || '')
                    .filter(Boolean)"""
            )
            return {str(s) for s in srcs}
        except Exception:
            return set()

    async def _is_streaming_active(self, page: Any) -> bool:
        for selector in STOP_BUTTON_SELECTORS:
            try:
                if await page.locator(selector).first.is_visible(timeout=200):
                    return True
            except Exception:
                continue
        return False

    async def _extract_turn_text(self, turn_loc: Any) -> str:
        for selector in RESPONSE_TEXT_SELECTORS:
            try:
                sub = turn_loc.locator(selector)
                if await sub.count() > 0:
                    text = (await sub.first.inner_text(timeout=1_500)).strip()
                    if text:
                        return text
            except Exception:
                continue
        try:
            return (await turn_loc.inner_text(timeout=1_500)).strip()
        except Exception:
            return ""

    async def _scroll_turn_to_bottom(self, turn_loc: Any) -> None:
        """Scroll the latest response turn to its footer so lazy buttons/images mount."""
        with contextlib.suppress(Exception):
            await turn_loc.evaluate(
                "(el) => el.scrollIntoView({ behavior: 'instant', block: 'end' })"
            )

    async def _has_turn_done_actions(self, turn_loc: Any) -> bool:
        for selector in RESPONSE_DONE_SELECTORS:
            try:
                loc = turn_loc.locator(selector)
                if await loc.count() > 0:
                    return True
                if await loc.first.is_visible(timeout=200):
                    return True
            except Exception:
                continue
        return False

    async def _collect_turn_generated_images(
        self,
        turn_loc: Any,
        *,
        prior_srcs: set[str],
    ) -> list[dict[str, Any]]:
        """Return candidate generated image records inside the latest model response turn.

        Explicitly filters out upload previews (``data-test-id="image-preview"`` and
        ``.file-preview-wrapper`` / ``.uploader-file-preview-container``), tiny icons/avatars
        (``naturalWidth < 256`` or ``naturalHeight < 256``), and pre-existing ``prior_srcs``.
        """
        try:
            candidates = await turn_loc.evaluate(
                """(root) => {
                    const out = [];
                    const imgs = Array.from(root.querySelectorAll('img'));
                    imgs.forEach((img, idx) => {
                        const src = img.currentSrc || img.src || '';
                        if (!src) return;
                        if (img.getAttribute('data-test-id') === 'image-preview') return;
                        const skipSel = [
                            '.file-preview-wrapper',
                            '.uploader-file-preview-container',
                            'user-query',
                        ].join(', ');
                        if (img.closest(skipSel)) return;
                        const isCandidateSrc = (
                            src.includes('googleusercontent.com') ||
                            src.startsWith('data:image/') ||
                            src.startsWith('blob:')
                        );
                        if (!isCandidateSrc) return;
                        const rect = img.getBoundingClientRect();
                        const visible = rect.width >= 64 && rect.height >= 64;
                        const nw = img.naturalWidth || 0;
                        const nh = img.naturalHeight || 0;
                        if (!visible || !img.complete || nw < 256 || nh < 256) return;
                        out.push({ index: idx, src, naturalWidth: nw, naturalHeight: nh });
                    });
                    return out;
                }"""
            )
        except Exception:
            return []

        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in cast("list[dict[str, Any]]", candidates or []):
            src = str(item.get("src") or "")
            if not src or src in prior_srcs or src in seen:
                continue
            seen.add(src)
            results.append(item)
        return results

    async def _wait_for_turn_completion(
        self,
        page: Any,
        *,
        prior_turns: int,
        prior_srcs: set[str],
        timeout_seconds: float,
        expect_image: bool = False,
    ) -> tuple[str, list[dict[str, Any]], Any]:
        deadline = time.monotonic() + float(timeout_seconds)
        last_signature = ""
        stable_since: float | None = None
        last_text = ""
        last_images: list[dict[str, Any]] = []
        turn_loc: Any = None
        seen_new_turn = False

        while time.monotonic() < deadline:
            await self._check_page_risk_or_raise(page, tag="generation_poll")

            turns = await self._response_turns_locator(page)
            turn_count = await turns.count()
            if turn_count > prior_turns:
                seen_new_turn = True
                turn_loc = turns.nth(turn_count - 1)
                last_text = await self._extract_turn_text(turn_loc)
                if expect_image:
                    last_images = await self._collect_turn_generated_images(
                        turn_loc,
                        prior_srcs=prior_srcs,
                    )

            streaming = await self._is_streaming_active(page)
            if seen_new_turn and turn_loc is not None:
                if not streaming and (last_text or last_images):
                    await self._scroll_turn_to_bottom(turn_loc)
                    # Re-read once scrolled to ensure no clipped/lazy-rendered tail.
                    refreshed_text = await self._extract_turn_text(turn_loc)
                    if len(refreshed_text) >= len(last_text):
                        last_text = refreshed_text
                    if expect_image:
                        last_images = await self._collect_turn_generated_images(
                            turn_loc,
                            prior_srcs=prior_srcs,
                        )

                img_sig = ",".join(i["src"] for i in last_images)
                sig = f"{len(last_text)}:{last_text[-120:]}|imgs:{img_sig}"
                has_content = bool(last_text) or bool(last_images)
                done_actions = await self._has_turn_done_actions(turn_loc)

                if not streaming and has_content:
                    if sig != last_signature or stable_since is None:
                        last_signature = sig
                        stable_since = time.monotonic()
                    else:
                        stable_for = time.monotonic() - stable_since
                        needed = 2.0 if done_actions else 4.0
                        if expect_image and not last_images:
                            # Give generated image cards up to 6-8s before refusal fail-fast.
                            needed = 6.0 if done_actions else 8.0
                        if stable_for >= needed:
                            return last_text, last_images, turn_loc
                else:
                    last_signature = sig
                    stable_since = None

            await asyncio.sleep(0.5)

        # Graceful deadline salvage: if Gemini ALREADY finished generating (not streaming)
        # right before deadline with substantial content, do not discard a completed reply.
        if (
            seen_new_turn
            and turn_loc is not None
            and not await self._is_streaming_active(page)
            and (last_text or last_images)
        ):
            await self._scroll_turn_to_bottom(turn_loc)
            final_text = await self._extract_turn_text(turn_loc)
            final_images = (
                await self._collect_turn_generated_images(turn_loc, prior_srcs=prior_srcs)
                if expect_image
                else last_images
            )
            if final_text == last_text and (not expect_image or final_images or last_text):
                return final_text, final_images, turn_loc

        shot, dom = await save_failure_diagnostics(
            page,
            debug_dir=self.debug_dir,
            tag="image_timeout" if expect_image else "chat_timeout",
        )
        raise GeminiTimeoutError(
            f"Timed out after {timeout_seconds}s waiting for Gemini response to finish.",
            screenshot_path=shot,
            dom_path=dom,
        )

    async def _fetch_url_bytes_with_retry(self, page: Any, url: str) -> bytes | None:
        """Download-only transport retry (1s->2s->4s ±25%, NEVER re-submits prompt)."""
        headers = {"Referer": "https://gemini.google.com/"}
        for attempt in range(DOWNLOAD_RETRY_ATTEMPTS):
            try:
                resp = await page.request.get(url, headers=headers, timeout=45_000)
                status = int(resp.status)
                if 200 <= status < 300:
                    body = await resp.body()
                    if body:
                        return bytes(body)
                if 400 <= status < 500 and status != 429:
                    return None
            except Exception as exc:
                log.warning(
                    "gemini.image.fetch_retry",
                    attempt=attempt + 1,
                    error=str(exc),
                )
            if attempt < DOWNLOAD_RETRY_ATTEMPTS - 1:
                delay = min(DOWNLOAD_RETRY_CAP_S, DOWNLOAD_RETRY_BASE_S * (2**attempt))
                await asyncio.sleep(jitter_delay(delay))
        return None

    async def _fetch_blob_bytes(self, page: Any, blob_url: str) -> bytes | None:
        try:
            b64 = await page.evaluate(
                """async (u) => {
                    const r = await fetch(u);
                    if (!r.ok) return null;
                    const buf = await r.arrayBuffer();
                    const bytes = new Uint8Array(buf);
                    let bin = '';
                    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
                    return btoa(bin);
                }""",
                blob_url,
            )
            if isinstance(b64, str) and b64:
                return base64.b64decode(b64)
        except Exception:
            return None
        return None

    async def _extract_img_via_canvas(self, img_loc: Any) -> bytes | None:
        """Fallback: snapshot the live decoded ``<img>`` bitmap via an offscreen canvas.

        Works even after Gemini revokes a preview ``blob:`` object URL with
        ``URL.revokeObjectURL``, because the decoded bitmap stays loaded in the DOM.
        """
        try:
            data_url = await img_loc.evaluate(
                """(img) => {
                    const w = img.naturalWidth || 0;
                    const h = img.naturalHeight || 0;
                    if (w < 64 || h < 64) return null;
                    const c = document.createElement('canvas');
                    c.width = w;
                    c.height = h;
                    const ctx = c.getContext('2d');
                    if (!ctx) return null;
                    ctx.drawImage(img, 0, 0, w, h);
                    return c.toDataURL('image/png');
                }"""
            )
            if isinstance(data_url, str) and data_url.startswith("data:image/"):
                _, b64_part = data_url.split(",", 1)
                return base64.b64decode(b64_part)
        except Exception:
            return None
        return None

    async def _download_single_image(
        self,
        page: Any,
        turn_loc: Any,
        candidate: dict[str, Any],
        *,
        out_path: Path,
        strip_watermark: bool,
        target_aspect: str | None,
    ) -> GeminiImageMetadata:
        img_idx = int(candidate["index"])
        img_src = str(candidate["src"])
        img_loc = turn_loc.locator("img").nth(img_idx)
        # Clean up any stale temp files left by a prior interrupted run.
        for stale_suffix in (".pw_dl.tmp", ".dl.tmp", ".wm.tmp"):
            stale = out_path.with_name(f".{out_path.name}{stale_suffix}")
            with contextlib.suppress(OSError):
                stale.unlink(missing_ok=True)

        # Priority 1: Native "Download full size" button -> Playwright expect_download().
        # Scope strictly inside turn_loc so header buttons like "下载应用" are never clicked.
        try:
            await img_loc.scroll_into_view_if_needed(timeout=4_000)
            await img_loc.hover(timeout=3_000)
            await asyncio.sleep(jitter_delay(0.45))
            for btn_sel in DOWNLOAD_IMAGE_BUTTON_SELECTORS:
                btn = turn_loc.locator(btn_sel).first
                try:
                    if await btn.count() == 0 or not await btn.is_visible(timeout=600):
                        continue
                    async with page.expect_download(timeout=60_000) as dl_info:
                        await btn.click(timeout=3_000)
                    download = await dl_info.value
                    tmp_dl = out_path.with_name(f".{out_path.name}.pw_dl.tmp")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        await download.save_as(str(tmp_dl))
                        raw = tmp_dl.read_bytes()
                    finally:
                        tmp_dl.unlink(missing_ok=True)
                    return finalize_image_file(
                        raw,
                        out_path,
                        min_width=self.min_image_dim,
                        min_height=self.min_image_dim,
                        strip_watermark=strip_watermark,
                        target_aspect=target_aspect,
                        download_method="download_button",
                    )
                except Exception as exc:
                    log.warning(
                        "gemini.image.p1_download_attempt_failed",
                        selector=btn_sel,
                        error=str(exc),
                    )
                    continue
        except Exception as exc:
            log.warning("gemini.image.p1_hover_failed", error=str(exc))

        # Priority 2 & 3: Authenticated browser-context fetch (=s0 -> =s2048-rj -> raw src)
        # or revoked-blob canvas extraction.
        raw_bytes: bytes | None = None
        if img_src.startswith("data:image/"):
            with contextlib.suppress(Exception):
                _, b64_part = img_src.split(",", 1)
                raw_bytes = base64.b64decode(b64_part)
        elif img_src.startswith("blob:"):
            raw_bytes = await self._fetch_blob_bytes(page, img_src)
            if not raw_bytes:
                raw_bytes = await self._extract_img_via_canvas(img_loc)
        else:
            candidate_urls: list[str] = []
            for url_opt in (
                upgrade_googleusercontent_url(img_src, size_spec="s0"),
                upgrade_googleusercontent_url(img_src, size_spec="s2048-rj"),
                img_src,
            ):
                if url_opt not in candidate_urls:
                    candidate_urls.append(url_opt)
            for fetch_url in candidate_urls:
                raw_bytes = await self._fetch_url_bytes_with_retry(page, fetch_url)
                if raw_bytes:
                    break

        if not raw_bytes:
            shot, dom = await save_failure_diagnostics(
                page, debug_dir=self.debug_dir, tag="image_download_failed"
            )
            raise GeminiDownloadError(
                "Could not download generated image via native button or authenticated fetch.",
                screenshot_path=shot,
                dom_path=dom,
            )

        try:
            return finalize_image_file(
                raw_bytes,
                out_path,
                min_width=self.min_image_dim,
                min_height=self.min_image_dim,
                strip_watermark=strip_watermark,
                target_aspect=target_aspect,
                download_method="img_src_fetch",
            )
        except GeminiDownloadError as exc:
            shot, dom = await save_failure_diagnostics(
                page, debug_dir=self.debug_dir, tag="image_invalid"
            )
            exc.screenshot_path = shot
            exc.dom_path = dom
            raise

    async def chat(
        self,
        prompt: str,
        *,
        attachments: Sequence[Path | str] = (),
        model: str | None = None,
        chat_url: str | None = None,
        new_chat: bool = False,
        extract_json: bool = False,
        out_file: Path | str | None = None,
        timeout_seconds: int = 180,
    ) -> GeminiChatResult:
        """Send a text/multimodal prompt to Gemini Web and return the completed response."""
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise GeminiConfigError("Prompt must not be empty.")
        attachment_paths = _validate_existing_files(attachments, label="Attachment")
        started = time.monotonic()
        warnings: list[str] = []

        if self.enforce_risk_guard:
            await self.ledger.await_preflight(
                self.profile,
                kind="chat",
                flow_ledger_dir=self.flow_ledger_dir,
                linked_flow_profile=self.linked_flow_profile,
                allow_flow_overlap=self.allow_flow_overlap,
            )

        if self._page is None:
            await self.start()
        page = self._page
        assert page is not None

        try:
            await self._navigate_and_ready(
                page,
                chat_url=chat_url,
                new_chat=new_chat,
                timeout_seconds=float(timeout_seconds),
            )
            model_selected: str | None = None
            if model:
                model_selected, warn = await self._switch_model_best_effort(page, model)
                if warn:
                    warnings.append(warn)

            if attachment_paths:
                warnings.extend(await self._upload_files(page, attachment_paths))

            prior_turns = await self._count_response_turns(page)
            prior_srcs = await self._snapshot_existing_image_srcs(page)

            if self.enforce_risk_guard:
                self.ledger.record_dispatch(self.profile, kind="chat")

            await self._enter_prompt_and_submit(page, cleaned_prompt)
            text, _imgs, _turn = await self._wait_for_turn_completion(
                page,
                prior_turns=prior_turns,
                prior_srcs=prior_srcs,
                timeout_seconds=float(timeout_seconds),
                expect_image=False,
            )

            extracted: Any | None = extract_json_from_text(text) if extract_json else None
            if extract_json and extracted is None:
                warnings.append("Requested extract_json=True but no valid JSON block was parsed.")

            saved_out: str | None = None
            if out_file is not None:
                out_path = Path(out_file).expanduser()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if extract_json and extracted is not None:
                    import json

                    out_path.write_text(
                        json.dumps(extracted, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                else:
                    out_path.write_text(text, encoding="utf-8")
                saved_out = str(out_path)

            if self.enforce_risk_guard:
                self.ledger.record_outcome(self.profile, exit_code=0)

            return GeminiChatResult(
                ok=True,
                profile=self.profile,
                chat_url=str(page.url or chat_url or GEMINI_APP_URL),
                model_requested=model,
                model_selected=model_selected,
                text=text,
                extracted_json=extracted,
                attachments=[str(p) for p in attachment_paths],
                elapsed_seconds=round(time.monotonic() - started, 2),
                out_file=saved_out,
                warnings=warnings,
            )
        except GeminiError as exc:
            if self.enforce_risk_guard:
                cooldown = self.ledger.record_outcome(
                    self.profile,
                    exit_code=exc.exit_code,
                    reason=exc.code.value,
                )
                if cooldown is not None and isinstance(exc, GeminiRateLimitError):
                    exc.retry_after_seconds = cooldown
                    exc.cooldown_seconds = cooldown
            raise

    async def generate_image(
        self,
        prompt: str,
        *,
        out_file: Path | str,
        ref_paths: Sequence[Path | str] = (),
        aspect: str | None = "16:9",
        model: str | None = None,
        chat_url: str | None = None,
        new_chat: bool = False,
        strip_watermark: bool = False,
        timeout_seconds: int = 240,
        reuse_current_page: bool = False,
    ) -> GeminiImageResult:
        """Generate or edit an image via Gemini Web and save full-resolution output."""
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise GeminiConfigError("Prompt must not be empty.")
        refs = _validate_existing_files(ref_paths, label="Reference image")
        out_path = Path(out_file).expanduser()
        effective_prompt = inject_aspect_hint(cleaned_prompt, aspect)
        started = time.monotonic()
        warnings: list[str] = []

        if self.enforce_risk_guard:
            await self.ledger.await_preflight(
                self.profile,
                kind="image",
                flow_ledger_dir=self.flow_ledger_dir,
                linked_flow_profile=self.linked_flow_profile,
                allow_flow_overlap=self.allow_flow_overlap,
            )

        if self._page is None:
            await self.start()
        page = self._page
        assert page is not None

        try:
            can_reuse = False
            if reuse_current_page and not new_chat:
                ctx = (
                    self._context
                    if self._context is not None
                    else getattr(page, "context", None)
                )
                self._observed_statuses.clear()
                await self._dismiss_promo_overlays(page)
                await self._check_page_risk_or_raise(page, tag="reuse_ready_check")
                is_auth, _hint = await detect_gemini_authenticated(ctx, page)
                can_reuse = is_auth
            if not can_reuse:
                await self._navigate_and_ready(
                    page,
                    chat_url=chat_url,
                    new_chat=new_chat,
                    timeout_seconds=float(timeout_seconds),
                )
            model_selected: str | None = None
            if model and not can_reuse:
                model_selected, warn = await self._switch_model_best_effort(page, model)
                if warn:
                    warnings.append(warn)

            if not can_reuse:
                _tool_ok, tool_warn = await self._activate_image_tool_best_effort(page)
                if tool_warn:
                    warnings.append(tool_warn)

            if refs:
                warnings.extend(await self._upload_files(page, refs))

            prior_turns = await self._count_response_turns(page)
            prior_srcs = await self._snapshot_existing_image_srcs(page)

            if self.enforce_risk_guard:
                self.ledger.record_dispatch(self.profile, kind="image")

            await self._enter_prompt_and_submit(page, effective_prompt)
            response_text, candidates, turn_loc = await self._wait_for_turn_completion(
                page,
                prior_turns=prior_turns,
                prior_srcs=prior_srcs,
                timeout_seconds=float(timeout_seconds),
                expect_image=True,
            )

            if not candidates:
                shot, dom = await save_failure_diagnostics(
                    page, debug_dir=self.debug_dir, tag="no_image_generated"
                )
                snippet = " ".join(response_text.split())[:240]
                raise GeminiNoImageGeneratedError(
                    "Gemini completed its response without generating a valid image"
                    + (f" (reply={snippet!r})." if snippet else "."),
                    screenshot_path=shot,
                    dom_path=dom,
                )

            saved_images: list[GeminiImageMetadata] = []
            for idx, cand in enumerate(candidates):
                target = (
                    out_path
                    if idx == 0
                    else out_path.with_name(f"{out_path.stem}_{idx + 1}{out_path.suffix}")
                )
                meta = await self._download_single_image(
                    page,
                    turn_loc,
                    cand,
                    out_path=target,
                    strip_watermark=strip_watermark,
                    target_aspect=aspect,
                )
                saved_images.append(meta)

            primary = saved_images[0]
            if self.enforce_risk_guard:
                self.ledger.record_outcome(self.profile, exit_code=0)

            return GeminiImageResult(
                ok=True,
                profile=self.profile,
                chat_url=str(page.url or chat_url or GEMINI_APP_URL),
                model_requested=model,
                model_selected=model_selected,
                aspect=aspect,
                ref_paths=[str(p) for p in refs],
                out_file=primary.path,
                images=saved_images,
                width=primary.width,
                height=primary.height,
                aspect_ratio=primary.aspect_ratio,
                bytes=primary.bytes,
                sha256=primary.sha256,
                download_method=primary.download_method,
                watermark_stripped=primary.watermark_stripped,
                watermark_note=primary.watermark_note,
                response_text=response_text,
                elapsed_seconds=round(time.monotonic() - started, 2),
                warnings=warnings,
            )
        except GeminiError as exc:
            if self.enforce_risk_guard:
                cooldown = self.ledger.record_outcome(
                    self.profile,
                    exit_code=exc.exit_code,
                    reason=exc.code.value,
                )
                if cooldown is not None and isinstance(exc, GeminiRateLimitError):
                    exc.retry_after_seconds = cooldown
                    exc.cooldown_seconds = cooldown
            raise

    # -----------------------------------------------------------------------
    # Gemini Ultra "任务" (Tasks / Neon Cloud Agent) async workflow
    # -----------------------------------------------------------------------

    async def _activate_tasks_mode(self, page: Any) -> None:
        """Activate Gemini Ultra Tasks (「任务」) pill in the main workspace."""
        narrow_count = 6  # First 6 selectors are specific segmented-button / radio / tab pills.

        async def _is_selected(loc: Any) -> bool:
            for attr in ("aria-pressed", "aria-selected", "aria-checked"):
                with contextlib.suppress(Exception):
                    if str(await loc.get_attribute(attr) or "").strip().lower() == "true":
                        return True
            return False

        for sel_idx, selector in enumerate(TASK_MODE_PILL_SELECTORS):
            try:
                loc = page.locator(selector)
                count = min(await loc.count(), 8)
                for i in range(count):
                    cand = loc.nth(i)
                    if not await cand.is_visible(timeout=500):
                        continue
                    if sel_idx >= narrow_count:
                        txt = ""
                        with contextlib.suppress(Exception):
                            txt = (await cand.inner_text(timeout=500)).strip()
                        if txt.lower() not in {"任务", "tasks"}:
                            continue
                        box: Any = None
                        with contextlib.suppress(Exception):
                            box = await cand.bounding_box()
                        if isinstance(box, Mapping):
                            box_map = cast("Mapping[str, Any]", box)
                            if float(box_map.get("x", 0.0)) <= 220.0:
                                continue
                    if await _is_selected(cand):
                        return
                    with contextlib.suppress(Exception):
                        await cand.scroll_into_view_if_needed(timeout=1_500)
                    await cand.click(timeout=3_000)
                    await asyncio.sleep(jitter_delay(0.8))
                    if await _is_selected(cand):
                        return
                    # Click succeeded cleanly even if aria-pressed is rendered on a parent node.
                    return
            except Exception:
                continue

        # Fallback: JS coordinate probe in main area (rect.x > 220) matching exact "任务" / "Tasks".
        try:
            clicked = await page.evaluate(
                """() => {
                    const els = Array.from(
                        document.querySelectorAll('button, [role="radio"], [role="tab"]')
                    );
                    for (const el of els) {
                        const txt = (el.innerText || el.textContent || '').trim();
                        if (txt !== '任务' && txt.toLowerCase() !== 'tasks') continue;
                        const rect = el.getBoundingClientRect();
                        if (rect.width <= 0 || rect.height <= 0 || rect.x <= 220) continue;
                        el.click();
                        return true;
                    }
                    return false;
                }"""
            )
            if clicked:
                await asyncio.sleep(jitter_delay(0.8))
                return
        except Exception:
            pass

        shot, dom = await save_failure_diagnostics(
            page, debug_dir=self.debug_dir, tag="tasks_mode_missing"
        )
        raise GeminiUiDriftError(
            "Could not locate Gemini Ultra '任务' / 'Tasks' mode pill in the UI.",
            screenshot_path=shot,
            dom_path=dom,
        )

    async def _detect_task_done_badge(self, page: Any) -> bool:
        """Return True if Gemini Tasks workspace displays a visible Completed / 完成 badge.

        Walks both light DOM and open shadow roots, handles Angular Material multi-child
        buttons/chips and leading ``mat-icon`` ligature text (e.g. ``check_circle 已完成``),
        and excludes sidebar navigation, composer inputs, and user prompt bubbles.
        """
        try:
            found = await page.evaluate(
                """() => {
                    const doneRe = /^(完成|已完成|任务已完成|done|completed|finished)$/i;
                    const iconWordsRe = new RegExp(
                        '\\\\b(check_circle_outline|check_circle|task_alt|'
                        + 'verified|done_all|done|check)\\\\b',
                        'gi',
                    );
                    const skipSel = [
                        'nav',
                        'bard-sidenav',
                        'side-navigation',
                        'rich-textarea',
                        'textarea',
                        '[contenteditable="true"]',
                        'user-query',
                        '[data-test-id="user-query"]',
                    ].join(', ');
                    const isDone = (s) => s.length > 0 && s.length <= 24 && doneRe.test(s);

                    const walk = (root) => {
                        const nodes = root.querySelectorAll('*');
                        for (const el of nodes) {
                            if (el.shadowRoot && walk(el.shadowRoot)) return true;
                            if (el.children.length > 6) continue;
                            if (el.closest && el.closest(skipSel)) continue;

                            const directText = Array.from(el.childNodes)
                                .filter(n => n.nodeType === 3)
                                .map(n => (n.textContent || '').trim())
                                .filter(Boolean)
                                .join(' ')
                                .trim();
                            const rawText = (el.innerText || el.textContent || '').trim();
                            const stripped = rawText
                                .replace(iconWordsRe, ' ')
                                .replace(/\\s+/g, ' ')
                                .trim();

                            if (!isDone(directText) && !isDone(rawText) && !isDone(stripped)) {
                                continue;
                            }

                            const rect = el.getBoundingClientRect();
                            if (rect.width > 0 && rect.height > 0
                                && rect.width <= 360 && rect.height <= 90) {
                                return true;
                            }
                        }
                        return false;
                    };
                    return walk(document);
                }"""
            )
            return bool(found)
        except Exception:
            return False

    async def _has_task_done_actions(self, page: Any, turn_loc: Any | None) -> bool:
        """Check thumbs-up/copy footer actions inside the latest turn or main task workspace."""
        if turn_loc is not None and await self._has_turn_done_actions(turn_loc):
            return True
        for selector in RESPONSE_DONE_SELECTORS:
            try:
                loc = page.locator(selector)
                if await loc.count() > 0:
                    return True
            except Exception:
                continue
        return False

    async def _count_task_rendered_images(self, page: Any) -> int:
        """Count rendered generated <img> elements (>=256px) in the Tasks workspace."""
        try:
            cnt = await page.evaluate(
                """() => {
                    const seen = new Set();
                    for (const img of Array.from(document.querySelectorAll('img'))) {
                        const src = img.currentSrc || img.src || '';
                        if (!src || seen.has(src)) continue;
                        if (img.getAttribute('data-test-id') === 'image-preview') continue;
                        const skipImg = (
                            '.file-preview-wrapper, .uploader-file-preview-container, user-query'
                        );
                        if (img.closest(skipImg)) {
                            continue;
                        }
                        const isCand = (
                            src.includes('googleusercontent.com')
                            || src.startsWith('data:image/')
                            || src.startsWith('blob:')
                        );
                        if (!isCand) continue;
                        const nw = img.naturalWidth || 0;
                        const nh = img.naturalHeight || 0;
                        if (nw < 256 || nh < 256) continue;
                        seen.add(src);
                    }
                    return seen.size;
                }"""
            )
            return int(cnt or 0)
        except Exception:
            return 0

    async def _scroll_task_conversation(self, page: Any, *, fraction: float = 1.0) -> None:
        """Scroll window and Gemini conversation containers to mount lazy-loaded DOM items."""
        clamped = max(0.0, min(1.0, float(fraction)))
        with contextlib.suppress(Exception):
            await page.evaluate(
                """(frac) => {
                    const sels = [
                        'infinite-scroller',
                        '.conversation-container',
                        '[data-test-id="chat-history-container"]',
                        'main',
                    ];
                    const seen = new Set();
                    for (const sel of sels) {
                        for (const el of document.querySelectorAll(sel)) {
                            if (seen.has(el)) continue;
                            seen.add(el);
                            if (el.scrollHeight > el.clientHeight) {
                                el.scrollTop = Math.floor(el.scrollHeight * frac);
                            }
                        }
                    }
                    const docHeight = document.documentElement
                        ? document.documentElement.scrollHeight
                        : document.body.scrollHeight;
                    window.scrollTo(0, Math.floor(docHeight * frac));
                }""",
                clamped,
            )

    async def _extract_task_all_text(self, page: Any) -> str:
        """Extract combined response text across all model turns in a Tasks conversation."""
        turns = await self._response_turns_locator(page)
        count = 0
        with contextlib.suppress(Exception):
            count = await turns.count()
        parts: list[str] = []
        for idx in range(count):
            chunk = await self._extract_turn_text(turns.nth(idx))
            if chunk:
                parts.append(chunk)
        if parts:
            return "\n\n".join(parts)
        with contextlib.suppress(Exception):
            main_loc = page.locator("main").first
            if await main_loc.count() > 0:
                return (await main_loc.inner_text(timeout=1_500)).strip()
        return ""

    async def _extract_task_code_blocks(self, page: Any) -> list[str]:
        """Extract all visible/mounted code blocks from the Tasks conversation."""
        try:
            raw = await page.evaluate(
                """() => {
                    const out = [];
                    const seen = new Set();
                    const sels = [
                        'model-response pre code',
                        'response-container pre code',
                        'code-block pre',
                        'pre code',
                        'pre',
                    ];
                    for (const sel of sels) {
                        for (const el of document.querySelectorAll(sel)) {
                            const txt = (el.innerText || el.textContent || '').trim();
                            if (!txt || seen.has(txt)) continue;
                            seen.add(txt);
                            out.push(txt);
                        }
                    }
                    return out;
                }"""
            )
            return [str(item) for item in cast("list[Any]", raw or []) if str(item).strip()]
        except Exception:
            return []

    async def _find_task_image_buttons(self, page: Any) -> tuple[Any | None, list[int]]:
        """Locate full-resolution image download buttons in Tasks output (skipping app CTA)."""
        for selector in TASK_IMAGE_DOWNLOAD_SELECTORS:
            try:
                loc = page.locator(selector)
                count = await loc.count()
                if count <= 0:
                    continue
                valid_indices: list[int] = []
                for idx in range(count):
                    btn = loc.nth(idx)
                    label = ""
                    txt = ""
                    with contextlib.suppress(Exception):
                        label = str(await btn.get_attribute("aria-label") or "")
                    with contextlib.suppress(Exception):
                        txt = (await btn.inner_text(timeout=300)).strip()
                    if _is_app_download_label(label, txt):
                        continue
                    valid_indices.append(idx)
                if valid_indices:
                    return loc, valid_indices
            except Exception:
                continue
        return None, []

    async def _count_task_image_download_buttons(self, page: Any) -> int:
        _loc, indices = await self._find_task_image_buttons(page)
        return len(indices)

    async def _harvest_task_exported_files(
        self,
        page: Any,
        *,
        out_dir: Path | None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Download exported artifact files (e.g. .json) from a completed Tasks workspace.

        Never clicks top-right "下载应用" / "Download app" buttons or full-size image buttons.
        """
        exported_paths: list[str] = []
        json_texts: list[str] = []
        warnings: list[str] = []
        seen_signatures: set[str] = set()

        for selector in TASK_ARTIFACT_DOWNLOAD_SELECTORS:
            try:
                loc = page.locator(selector)
                count = min(await loc.count(), 12)
            except Exception:
                continue
            for idx in range(count):
                el = loc.nth(idx)
                try:
                    if not await el.is_visible(timeout=500):
                        continue
                    label = str(await el.get_attribute("aria-label") or "")
                    txt = (await el.inner_text(timeout=400)).strip()
                    href = str(await el.get_attribute("href") or "")
                    if _is_app_download_label(label, txt) or _is_image_download_label(label, txt):
                        continue
                    sig = f"{label.strip().lower()}|{txt.lower()}|{href}"
                    if sig in seen_signatures:
                        continue
                    seen_signatures.add(sig)

                    async with page.expect_download(timeout=15_000) as dl_info:
                        await el.click(force=True, timeout=5_000)
                    download = await dl_info.value
                    suggested = str(getattr(download, "suggested_filename", "") or "").strip()
                    safe_name = Path(suggested).name if suggested else f"task_export_{idx + 1}.json"

                    if out_dir is not None:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        dest = out_dir / safe_name
                        stem, suffix = dest.stem, dest.suffix
                        dup_n = 2
                        while dest.exists():
                            dest = out_dir / f"{stem}_{dup_n}{suffix}"
                            dup_n += 1
                        await download.save_as(str(dest))
                        exported_paths.append(str(dest))
                        raw_bytes = dest.read_bytes()
                    else:
                        with tempfile.TemporaryDirectory(prefix="gemini_task_exp_") as tmp_dir:
                            tmp_file = Path(tmp_dir) / safe_name
                            await download.save_as(str(tmp_file))
                            raw_bytes = tmp_file.read_bytes()

                    if len(raw_bytes) <= 5 * 1024 * 1024:
                        decoded = raw_bytes.decode("utf-8", errors="ignore").strip()
                        if safe_name.lower().endswith(".json") or decoded.startswith(("{", "[")):
                            json_texts.append(decoded)
                except Exception as exc:
                    warnings.append(f"Skipped task artifact download ({selector}[{idx}]): {exc}")
                    continue

        return exported_paths, json_texts, warnings

    async def _harvest_task_images(
        self,
        page: Any,
        *,
        out_dir: Path,
        out_names: Sequence[str],
        download_timeout: int,
        min_image_dim: int,
        strip_watermark: bool,
        aspect: str | None,
        expected_images: int | None,
    ) -> tuple[list[GeminiImageMetadata], list[int], list[str]]:
        """Download all full-resolution generated images from a completed Tasks workspace.

        Uses 50s default Playwright ``expect_download`` timeout per card with 1 automatic
        retry, SHA256 deduplication, and 1-based ``missing_image_indices`` reporting.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        for frac in (0.0, 0.5, 1.0):
            await self._scroll_task_conversation(page, fraction=frac)
            await asyncio.sleep(0.35)

        btn_locator, valid_indices = await self._find_task_image_buttons(page)
        saved_images: list[GeminiImageMetadata] = []
        missing_slots: list[int] = []
        warnings: list[str] = []
        seen_sha256: dict[str, str] = {}

        timeout_ms = max(10, int(download_timeout)) * 1000
        logical_idx = 0

        if btn_locator is not None and valid_indices:
            for dom_idx in valid_indices:
                btn = btn_locator.nth(dom_idx)
                raw_bytes: bytes | None = None
                last_error: str | None = None

                for attempt in range(2):
                    tmp_dl = out_dir / f".task_img_{logical_idx + 1:02d}_{dom_idx}.pw_dl.tmp"
                    try:
                        with contextlib.suppress(Exception):
                            await btn.scroll_into_view_if_needed(timeout=2_500)
                        async with page.expect_download(timeout=timeout_ms) as dl_info:
                            await btn.click(force=True, timeout=10_000)
                        download = await dl_info.value
                        await download.save_as(str(tmp_dl))
                        raw_bytes = tmp_dl.read_bytes()
                        if raw_bytes:
                            break
                        last_error = "Downloaded file was empty."
                    except Exception as exc:
                        last_error = str(exc)
                        log.warning(
                            "gemini.task.image_download_attempt_failed",
                            card=logical_idx + 1,
                            dom_index=dom_idx,
                            attempt=attempt + 1,
                            error=last_error,
                        )
                        if attempt == 0:
                            await asyncio.sleep(1.0)
                    finally:
                        tmp_dl.unlink(missing_ok=True)

                if not raw_bytes:
                    slot_1based = logical_idx + 1
                    missing_slots.append(slot_1based)
                    warnings.append(
                        f"Task image card #{slot_1based} failed after 2 attempts: {last_error}"
                    )
                    logical_idx += 1
                    continue

                raw_sha = hashlib.sha256(raw_bytes).hexdigest()
                if raw_sha in seen_sha256:
                    warnings.append(
                        f"Duplicate image SHA256 skipped on DOM card index {dom_idx} "
                        f"(matches {seen_sha256[raw_sha]})."
                    )
                    continue

                detected_ext = extension_from_magic(raw_bytes[:12]) or ".jpg"
                custom_name = (
                    out_names[logical_idx].strip()
                    if logical_idx < len(out_names) and out_names[logical_idx].strip()
                    else ""
                )
                if custom_name:
                    candidate_path = out_dir / Path(custom_name).name
                    target_path = (
                        candidate_path
                        if candidate_path.suffix
                        else candidate_path.with_suffix(detected_ext)
                    )
                else:
                    target_path = out_dir / f"task_img_{logical_idx + 1:02d}{detected_ext}"

                try:
                    meta = finalize_image_file(
                        raw_bytes,
                        target_path,
                        min_width=min_image_dim,
                        min_height=min_image_dim,
                        strip_watermark=strip_watermark,
                        target_aspect=aspect,
                        download_method="download_button",
                    )
                except GeminiDownloadError as exc:
                    slot_1based = logical_idx + 1
                    missing_slots.append(slot_1based)
                    warnings.append(
                        f"Task image card #{slot_1based} failed validation: {exc.detail}"
                    )
                    logical_idx += 1
                    continue

                if meta.sha256 in seen_sha256:
                    warnings.append(
                        f"Duplicate finalized image SHA256 skipped on DOM card index {dom_idx} "
                        f"(matches {seen_sha256[meta.sha256]})."
                    )
                    Path(meta.path).unlink(missing_ok=True)
                    continue

                seen_sha256[raw_sha] = meta.path
                seen_sha256[meta.sha256] = meta.path
                saved_images.append(meta)
                logical_idx += 1
                await asyncio.sleep(0.35)

        if expected_images is not None and logical_idx < expected_images:
            for missing_idx in range(logical_idx + 1, expected_images + 1):
                missing_slots.append(missing_idx)

        return saved_images, sorted(set(missing_slots)), warnings

    async def submit_task(
        self,
        prompt: str,
        *,
        attachments: Sequence[Path | str] = (),
        model: str | None = None,
        timeout_seconds: int = 120,
    ) -> GeminiTaskSubmitResult:
        """Fire-and-forget submission in Gemini Ultra 「任务」 (Tasks) async mode.

        Activates Tasks mode, uploads attachments, submits the prompt, waits only until the
        cloud conversation URL resolves to ``https://gemini.google.com/app/<task_id>``, and
        returns immediately so the caller can close Chrome and release the profile lock.
        """
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise GeminiConfigError("Prompt must not be empty.")
        attachment_paths = _validate_existing_files(attachments, label="Attachment")
        started = time.monotonic()
        warnings: list[str] = []
        dispatch_kind = (
            "image"
            if any(p.suffix.lower() in _IMAGE_FILE_SUFFIXES for p in attachment_paths)
            else "chat"
        )

        if self.enforce_risk_guard:
            await self.ledger.await_preflight(
                self.profile,
                kind=dispatch_kind,
                flow_ledger_dir=self.flow_ledger_dir,
                linked_flow_profile=self.linked_flow_profile,
                allow_flow_overlap=self.allow_flow_overlap,
            )

        if self._page is None:
            await self.start()
        page = self._page
        assert page is not None

        try:
            await self._navigate_and_ready(
                page,
                chat_url=GEMINI_APP_URL,
                new_chat=True,
                timeout_seconds=float(timeout_seconds),
            )
            await self._activate_tasks_mode(page)

            if model:
                _model_selected, model_warn = await self._switch_model_best_effort(page, model)
                if model_warn:
                    warnings.append(model_warn)

            if attachment_paths:
                warnings.extend(await self._upload_files(page, attachment_paths))

            prior_turns = await self._count_response_turns(page)

            if self.enforce_risk_guard:
                self.ledger.record_dispatch(self.profile, kind=dispatch_kind)

            await self._enter_prompt_and_submit(page, cleaned_prompt)

            url_wait_budget = min(max(10.0, float(timeout_seconds)), 60.0)
            deadline = time.monotonic() + url_wait_budget
            resolved_url: str | None = None

            while time.monotonic() < deadline:
                await self._check_page_risk_or_raise(page, tag="task_submit_poll")
                current_url = str(getattr(page, "url", "") or "")
                if _is_resolved_task_url(current_url):
                    resolved_url = current_url
                    break
                await asyncio.sleep(0.5)

            if not resolved_url:
                shot, dom = await save_failure_diagnostics(
                    page, debug_dir=self.debug_dir, tag="task_submit_timeout"
                )
                raise GeminiTimeoutError(
                    f"Submitted Gemini Task did not resolve to a persistent /app/<task_id> URL "
                    f"within {int(url_wait_budget)}s (current={getattr(page, 'url', '')!r}).",
                    screenshot_path=shot,
                    dom_path=dom,
                )

            # Brief confirmation settle so the cloud task starts before closing Chrome.
            if self._context is not None:
                settle_deadline = time.monotonic() + 3.0
                while time.monotonic() < settle_deadline:
                    if await self._is_streaming_active(page):
                        break
                    if await self._count_response_turns(page) > prior_turns:
                        break
                    await asyncio.sleep(0.5)

            if self.enforce_risk_guard:
                self.ledger.record_outcome(self.profile, exit_code=0)

            return GeminiTaskSubmitResult(
                ok=True,
                profile=self.profile,
                chat_url=resolved_url,
                attachments=[str(p) for p in attachment_paths],
                elapsed_seconds=round(time.monotonic() - started, 2),
                warnings=warnings,
            )
        except GeminiError as exc:
            if self.enforce_risk_guard:
                cooldown = self.ledger.record_outcome(
                    self.profile,
                    exit_code=exc.exit_code,
                    reason=exc.code.value,
                )
                if cooldown is not None and isinstance(exc, GeminiRateLimitError):
                    exc.retry_after_seconds = cooldown
                    exc.cooldown_seconds = cooldown
            raise

    async def harvest_task(
        self,
        chat_url: str,
        *,
        out_dir: Path | str | None = None,
        out_file: Path | str | None = None,
        out_names: Sequence[str] = (),
        download_images: bool = True,
        extract_json: bool = True,
        expected_images: int | None = None,
        wait_timeout: int = 600,
        download_timeout: int = 50,
        poll_interval: float = 10.0,
        min_image_dim: int = 512,
        strip_watermark: bool = False,
        aspect: str | None = None,
    ) -> GeminiTaskHarvestResult:
        """Open a submitted Gemini Tasks URL, poll completion, and harvest JSON + images.

        Read-only on the risk ledger (checks preflight without counting as a new dispatch).
        When ``wait_timeout=0`` and the task is still running, returns immediately with
        ``ok=False, completed=False, status_label="running"`` without raising an error.
        """
        target_url = chat_url.strip()
        if not target_url or not is_gemini_app_url(target_url):
            raise GeminiConfigError(
                f"Invalid Gemini task URL: {chat_url!r}. Expected a https://gemini.google.com/ URL."
            )
        if expected_images is not None and expected_images < 0:
            raise GeminiConfigError("expected_images must be >= 0.")

        started = time.monotonic()
        warnings: list[str] = []

        if self.enforce_risk_guard:
            self.ledger.check_preflight(
                self.profile,
                kind="chat",
                flow_ledger_dir=self.flow_ledger_dir,
                linked_flow_profile=self.linked_flow_profile,
                allow_flow_overlap=self.allow_flow_overlap,
            )

        if self._page is None:
            await self.start()
        page = self._page
        assert page is not None

        try:
            await self._navigate_and_ready(
                page,
                chat_url=target_url,
                new_chat=False,
                timeout_seconds=max(45.0, float(wait_timeout)),
            )
            if self._context is not None:
                await asyncio.sleep(1.0)

            poll_budget = max(0.0, float(wait_timeout))
            deadline = time.monotonic() + poll_budget
            sleep_step = max(0.5, float(poll_interval))
            last_sig = ""
            stable_since: float | None = None
            completed = False
            latest_text = ""
            last_logged_state = ""
            last_log_ts = 0.0

            while True:
                await self._check_page_risk_or_raise(page, tag="task_harvest_poll")
                streaming = await self._is_streaming_active(page)
                if not streaming:
                    await self._scroll_task_conversation(page, fraction=1.0)
                done_badge = await self._detect_task_done_badge(page)
                turns = await self._response_turns_locator(page)
                turn_count = 0
                with contextlib.suppress(Exception):
                    turn_count = await turns.count()

                latest_turn: Any | None = None
                if turn_count > 0:
                    latest_turn = turns.nth(turn_count - 1)
                    await self._scroll_turn_to_bottom(latest_turn)
                has_done_actions = await self._has_task_done_actions(page, latest_turn)
                latest_text = await self._extract_task_all_text(page)

                img_btn_count = await self._count_task_image_download_buttons(page)
                dom_img_count = await self._count_task_rendered_images(page)
                has_payload = (
                    bool(latest_text.strip()) or img_btn_count > 0 or dom_img_count > 0
                )

                now_m = time.monotonic()
                state_key = (
                    f"s={streaming}|b={done_badge}|t={turn_count}|da={has_done_actions}|"
                    f"ib={img_btn_count}|di={dom_img_count}|tl={len(latest_text)}"
                )
                if state_key != last_logged_state or (now_m - last_log_ts) >= 30.0:
                    log.info(
                        "gemini.task.harvest_poll",
                        elapsed_seconds=round(now_m - started, 1),
                        streaming=streaming,
                        done_badge=done_badge,
                        turn_count=turn_count,
                        has_done_actions=has_done_actions,
                        img_btn_count=img_btn_count,
                        dom_img_count=dom_img_count,
                        text_len=len(latest_text),
                    )
                    last_logged_state = state_key
                    last_log_ts = now_m

                if done_badge and not streaming:
                    completed = True
                    break

                if not streaming and has_payload:
                    if poll_budget <= 0.0 and has_done_actions:
                        completed = True
                        break
                    sig = (
                        f"{len(latest_text)}:{latest_text[-200:]}:"
                        f"{img_btn_count}:{dom_img_count}:{turn_count}"
                    )
                    if sig != last_sig or stable_since is None:
                        last_sig = sig
                        stable_since = now_m
                    else:
                        enough_images = (
                            expected_images is not None
                            and expected_images > 0
                            and img_btn_count >= expected_images
                        )
                        required_stable = 6.0 if (has_done_actions or enough_images) else 12.0
                        if (now_m - stable_since) >= required_stable:
                            completed = True
                            break
                else:
                    last_sig = ""
                    stable_since = None

                if poll_budget <= 0.0:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                await asyncio.sleep(min(sleep_step, remaining))

            current_url = str(getattr(page, "url", "") or target_url)
            resolved_out_dir = Path(out_dir).expanduser() if out_dir is not None else None

            # Before giving up on a blocking wait timeout, scroll the full workspace to mount
            # lazy cards/badges and check whether completion or partial images can be salvaged.
            if not completed and poll_budget > 0.0:
                for frac in (0.0, 0.5, 1.0):
                    await self._scroll_task_conversation(page, fraction=frac)
                    await asyncio.sleep(0.3)
                if (
                    await self._detect_task_done_badge(page)
                    and not await self._is_streaming_active(page)
                ):
                    completed = True

            if not completed:
                if poll_budget <= 0.0:
                    return GeminiTaskHarvestResult(
                        ok=False,
                        profile=self.profile,
                        chat_url=current_url,
                        completed=False,
                        status_label="running",
                        text=latest_text,
                        extracted_json=None,
                        exported_files=[],
                        images=[],
                        downloaded_count=0,
                        expected_images=expected_images,
                        missing_image_indices=[],
                        elapsed_seconds=round(time.monotonic() - started, 2),
                        out_file=None,
                        warnings=warnings,
                    )

                shot, dom = await save_failure_diagnostics(
                    page, debug_dir=self.debug_dir, tag="task_harvest_timeout"
                )
                if shot or dom:
                    warnings.append(
                        f"Task harvest wait_timeout expired (diagnostics: shot={shot}, dom={dom})."
                    )

                salvage_images: list[GeminiImageMetadata] = []
                salvage_missing: list[int] = []
                if (
                    download_images
                    and resolved_out_dir is not None
                    and await self._count_task_image_download_buttons(page) > 0
                ):
                    resolved_out_dir.mkdir(parents=True, exist_ok=True)
                    salvage_images, salvage_missing, img_w = await self._harvest_task_images(
                        page,
                        out_dir=resolved_out_dir,
                        out_names=out_names,
                        download_timeout=download_timeout,
                        min_image_dim=min_image_dim,
                        strip_watermark=strip_watermark,
                        aspect=aspect,
                        expected_images=expected_images,
                    )
                    warnings.extend(img_w)
                    if salvage_images:
                        warnings.append(
                            f"Salvaged {len(salvage_images)} already-rendered full-res image(s) "
                            f"after wait_timeout expiry."
                        )

                salvage_text = await self._extract_task_all_text(page) or latest_text
                salvage_json: Any | None = (
                    extract_json_from_text(salvage_text) if extract_json else None
                )
                all_expected_salvaged = (
                    bool(salvage_images)
                    and expected_images is not None
                    and len(salvage_images) >= expected_images
                    and not salvage_missing
                )
                if all_expected_salvaged:
                    if self.enforce_risk_guard:
                        self.ledger.record_outcome(self.profile, exit_code=0)
                    return GeminiTaskHarvestResult(
                        ok=True,
                        profile=self.profile,
                        chat_url=current_url,
                        completed=True,
                        status_label="completed",
                        text=salvage_text,
                        extracted_json=salvage_json,
                        exported_files=[],
                        images=salvage_images,
                        downloaded_count=len(salvage_images),
                        expected_images=expected_images,
                        missing_image_indices=[],
                        elapsed_seconds=round(time.monotonic() - started, 2),
                        out_file=None,
                        warnings=warnings,
                    )

                return GeminiTaskHarvestResult(
                    ok=False,
                    profile=self.profile,
                    chat_url=current_url,
                    completed=False,
                    status_label="timed_out",
                    text=salvage_text,
                    extracted_json=salvage_json,
                    exported_files=[],
                    images=salvage_images,
                    downloaded_count=len(salvage_images),
                    expected_images=expected_images,
                    missing_image_indices=salvage_missing,
                    elapsed_seconds=round(time.monotonic() - started, 2),
                    out_file=None,
                    warnings=warnings,
                )

            # Scroll through the full completed workspace before harvesting artifacts & images.
            for frac in (0.0, 0.5, 1.0):
                await self._scroll_task_conversation(page, fraction=frac)
                await asyncio.sleep(0.3)

            full_text = await self._extract_task_all_text(page) or latest_text
            code_blocks = await self._extract_task_code_blocks(page)

            if resolved_out_dir is not None:
                resolved_out_dir.mkdir(parents=True, exist_ok=True)

            exported_files, exported_json_texts, exp_warnings = (
                await self._harvest_task_exported_files(page, out_dir=resolved_out_dir)
            )
            warnings.extend(exp_warnings)

            extracted: Any | None = None
            if extract_json:
                for raw_json in reversed(exported_json_texts):
                    candidate = extract_json_from_text(raw_json)
                    if candidate is not None:
                        extracted = candidate
                        break
                if extracted is None:
                    for block in reversed(code_blocks):
                        candidate = extract_json_from_text(block)
                        if candidate is not None:
                            extracted = candidate
                            break
                if extracted is None:
                    extracted = extract_json_from_text(full_text)

            saved_images: list[GeminiImageMetadata] = []
            missing_indices: list[int] = []
            if download_images:
                if resolved_out_dir is not None:
                    saved_images, missing_indices, img_warnings = await self._harvest_task_images(
                        page,
                        out_dir=resolved_out_dir,
                        out_names=out_names,
                        download_timeout=download_timeout,
                        min_image_dim=min_image_dim,
                        strip_watermark=strip_watermark,
                        aspect=aspect,
                        expected_images=expected_images,
                    )
                    warnings.extend(img_warnings)
                elif expected_images is not None and expected_images > 0:
                    warnings.append(
                        "download_images=True with expected_images set, "
                        "but out_dir was not provided."
                    )
                    missing_indices = list(range(1, expected_images + 1))

            if extract_json and extracted is None and (out_file is not None or not saved_images):
                warnings.append("Requested extract_json=True but no valid JSON block was parsed.")

            saved_out: str | None = None
            if out_file is not None:
                out_path = Path(out_file).expanduser()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if extract_json and extracted is not None:
                    out_path.write_text(
                        json.dumps(extracted, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                else:
                    out_path.write_text(full_text, encoding="utf-8")
                saved_out = str(out_path)

            downloaded_count = len(saved_images)
            ok = (
                completed
                and (expected_images is None or downloaded_count >= expected_images)
                and not missing_indices
            )
            status_label = "completed" if ok else "incomplete"

            if self.enforce_risk_guard and ok:
                self.ledger.record_outcome(self.profile, exit_code=0)

            return GeminiTaskHarvestResult(
                ok=ok,
                profile=self.profile,
                chat_url=current_url,
                completed=True,
                status_label=status_label,
                text=full_text,
                extracted_json=extracted,
                exported_files=exported_files,
                images=saved_images,
                downloaded_count=downloaded_count,
                expected_images=expected_images,
                missing_image_indices=missing_indices,
                elapsed_seconds=round(time.monotonic() - started, 2),
                out_file=saved_out,
                warnings=warnings,
            )
        except GeminiError as exc:
            if self.enforce_risk_guard:
                cooldown = self.ledger.record_outcome(
                    self.profile,
                    exit_code=exc.exit_code,
                    reason=exc.code.value,
                )
                if cooldown is not None and isinstance(exc, GeminiRateLimitError):
                    exc.retry_after_seconds = cooldown
                    exc.cooldown_seconds = cooldown
            raise


def _is_resolved_task_url(url: str) -> bool:
    """Return True if ``url`` is a persistent Gemini task URL (``/app/<task_id>``)."""
    if not url or not is_gemini_app_url(url):
        return False
    parsed = urlsplit(url)
    parts = [p for p in parsed.path.split("/") if p]
    for idx, part in enumerate(parts):
        if part.lower() == "app" and idx + 1 < len(parts):
            task_id = parts[idx + 1].strip()
            return len(task_id) >= 4 and task_id.lower() not in {"new", "app"}
    return False


def _is_app_download_label(aria_label: str, text: str) -> bool:
    combined = f"{aria_label} {text}".lower()
    return any(
        needle in combined
        for needle in (
            "下载应用",
            "download app",
            "get app",
            "install app",
            "安装应用",
            "获取应用",
        )
    )


def _is_image_download_label(aria_label: str, text: str) -> bool:
    combined = f"{aria_label} {text}".lower()
    return any(
        needle in combined
        for needle in (
            "下载完整尺寸",
            "download full size",
            "下载原图",
            "full-size image",
        )
    )


async def chat(
    prompt: str,
    *,
    profile: str = DEFAULT_GEMINI_PROFILE,
    attachments: Sequence[Path | str] = (),
    model: str | None = None,
    chat_url: str | None = None,
    new_chat: bool = False,
    extract_json: bool = False,
    out_file: Path | str | None = None,
    headless: bool = False,
    timeout_seconds: int = 180,
    chrome_executable: str | None = None,
    proxy: str | None = None,
    home: Path | None = None,
    debug_dir: Path | None = None,
    enforce_risk_guard: bool = True,
    flow_ledger_dir: Path | str | None = None,
    linked_flow_profile: str | None = None,
    allow_flow_overlap: bool = False,
) -> GeminiChatResult:
    """One-shot Gemini Web chat helper managing profile lease and browser lifecycle."""
    client = GeminiWebClient(
        profile=profile,
        headless=headless,
        chrome_executable=chrome_executable,
        proxy=proxy,
        home=home,
        debug_dir=debug_dir,
        enforce_risk_guard=enforce_risk_guard,
        flow_ledger_dir=flow_ledger_dir,
        linked_flow_profile=linked_flow_profile,
        allow_flow_overlap=allow_flow_overlap,
    )
    try:
        return await client.chat(
            prompt,
            attachments=attachments,
            model=model,
            chat_url=chat_url,
            new_chat=new_chat,
            extract_json=extract_json,
            out_file=out_file,
            timeout_seconds=timeout_seconds,
        )
    finally:
        await client.close()


async def generate_image(
    prompt: str,
    *,
    out_file: Path | str,
    profile: str = DEFAULT_GEMINI_PROFILE,
    ref_paths: Sequence[Path | str] = (),
    aspect: str | None = "16:9",
    model: str | None = None,
    chat_url: str | None = None,
    new_chat: bool = False,
    strip_watermark: bool = False,
    headless: bool = False,
    timeout_seconds: int = 240,
    chrome_executable: str | None = None,
    proxy: str | None = None,
    home: Path | None = None,
    debug_dir: Path | None = None,
    enforce_risk_guard: bool = True,
    flow_ledger_dir: Path | str | None = None,
    linked_flow_profile: str | None = None,
    allow_flow_overlap: bool = False,
) -> GeminiImageResult:
    """One-shot Gemini Web image generation helper managing lease and browser lifecycle."""
    client = GeminiWebClient(
        profile=profile,
        headless=headless,
        chrome_executable=chrome_executable,
        proxy=proxy,
        home=home,
        debug_dir=debug_dir,
        enforce_risk_guard=enforce_risk_guard,
        flow_ledger_dir=flow_ledger_dir,
        linked_flow_profile=linked_flow_profile,
        allow_flow_overlap=allow_flow_overlap,
    )
    try:
        return await client.generate_image(
            prompt,
            out_file=out_file,
            ref_paths=ref_paths,
            aspect=aspect,
            model=model,
            chat_url=chat_url,
            new_chat=new_chat,
            strip_watermark=strip_watermark,
            timeout_seconds=timeout_seconds,
        )
    finally:
        await client.close()


def _probe_existing_image_outcome(
    item: GeminiImageBatchItem,
    *,
    min_image_dim: int,
) -> GeminiImageBatchItemOutcome | None:
    """Return a ``skipped`` outcome if ``item.out_file`` already holds a valid image."""
    path = Path(item.out_file).expanduser()
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
        width, height, _mime = inspect_image_bytes(data)
    except (OSError, GeminiDownloadError):
        return None
    if width < min_image_dim or height < min_image_dim:
        return None
    return GeminiImageBatchItemOutcome(
        id=item.id,
        status="skipped",
        out_file=str(path),
        width=width,
        height=height,
        aspect_ratio=compute_aspect_ratio_label(width, height),
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        detail="Existing valid image preserved (--resume).",
    )


_BATCH_STOP_EXIT_CODES: frozenset[int] = frozenset({3, 4, 10})


async def generate_images_batch(
    items: Sequence[GeminiImageBatchItem | Mapping[str, Any]],
    *,
    profile: str = DEFAULT_GEMINI_PROFILE,
    reuse_chat: bool = True,
    chat_url: str | None = None,
    gap_range: tuple[float, float] = (25.0, 45.0),
    resume: bool = True,
    aspect: str | None = "16:9",
    model: str | None = None,
    strip_watermark: bool = False,
    headless: bool = False,
    timeout_seconds: int = 240,
    chrome_executable: str | None = None,
    proxy: str | None = None,
    home: Path | None = None,
    debug_dir: Path | None = None,
    min_image_dim: int = 512,
    enforce_risk_guard: bool = True,
    flow_ledger_dir: Path | str | None = None,
    linked_flow_profile: str | None = None,
    allow_flow_overlap: bool = False,
) -> GeminiImageBatchResult:
    """Generate a batch of images sequentially in ONE Chrome session and ONE conversation.

    Uses a single :class:`GeminiProfileLease` and :class:`GeminiWebClient` across all items.
    When ``reuse_chat=True`` (default), Turns 2..N continue in the same open Gemini tab and
    conversation memory (saving Chrome cold-starts and preserving room/style consistency)
    while still downloading each shot as a separate full-resolution image.

    On any risk signal (``RATE_LIMITED`` [4], ``CHALLENGE_DETECTED`` [10],
    ``AUTH_REQUIRED`` [3], ``RISK_QUARANTINED`` / kill switch [10]), remaining items are
    immediately marked ``deferred``, Chrome closes cleanly, and partial progress +
    ``retry_after_seconds`` are returned in :class:`GeminiImageBatchResult`.
    """
    started = time.monotonic()
    validated_profile = validate_gemini_profile_name(profile)
    normalized: list[GeminiImageBatchItem] = []
    for idx, raw_item in enumerate(items):
        if isinstance(raw_item, GeminiImageBatchItem):
            normalized.append(raw_item)
        else:
            normalized.append(GeminiImageBatchItem.from_dict(raw_item, index=idx))
    if not normalized:
        raise GeminiConfigError("generate_images_batch requires at least one item.")

    gap_low = max(0.0, float(min(gap_range[0], gap_range[1])))
    gap_high = max(gap_low, float(max(gap_range[0], gap_range[1])))

    outcomes_by_index: dict[int, GeminiImageBatchItemOutcome] = {}
    seen_sha256: dict[str, str] = {}
    warnings: list[str] = []

    if resume:
        for idx, item in enumerate(normalized):
            existing = _probe_existing_image_outcome(item, min_image_dim=min_image_dim)
            if existing is not None:
                outcomes_by_index[idx] = existing
                if existing.sha256:
                    seen_sha256[existing.sha256] = item.id

    pending_indices = [idx for idx in range(len(normalized)) if idx not in outcomes_by_index]
    if not pending_indices:
        all_items = [outcomes_by_index[i] for i in range(len(normalized))]
        return GeminiImageBatchResult(
            ok=True,
            partial=False,
            profile=validated_profile,
            reuse_chat=reuse_chat,
            chat_url=chat_url,
            total=len(normalized),
            completed=[],
            skipped=list(all_items),
            failed=[],
            deferred=[],
            items=all_items,
            elapsed_seconds=round(time.monotonic() - started, 2),
            warnings=warnings,
        )

    client = GeminiWebClient(
        profile=validated_profile,
        headless=headless,
        chrome_executable=chrome_executable,
        proxy=proxy,
        home=home,
        debug_dir=debug_dir,
        min_image_dim=min_image_dim,
        enforce_risk_guard=enforce_risk_guard,
        flow_ledger_dir=flow_ledger_dir,
        linked_flow_profile=linked_flow_profile,
        allow_flow_overlap=allow_flow_overlap,
    )

    active_chat_url: str | None = chat_url
    generated_count = 0
    stop_error_code: str | None = None
    stop_exit_code: int | None = None
    stop_detail: str | None = None
    hard_halt = False
    risk_reason: str | None = None
    retry_after_seconds: float | None = None

    def _defer_remaining(start_pos: int, reason: str) -> None:
        for rem_idx in pending_indices[start_pos:]:
            if rem_idx not in outcomes_by_index:
                rem_item = normalized[rem_idx]
                outcomes_by_index[rem_idx] = GeminiImageBatchItemOutcome(
                    id=rem_item.id,
                    status="deferred",
                    out_file=str(Path(rem_item.out_file).expanduser()),
                    error_code=stop_error_code,
                    exit_code=stop_exit_code,
                    detail=reason,
                )

    def _record_risk_stop(exc: GeminiError) -> None:
        nonlocal stop_error_code, stop_exit_code, stop_detail
        nonlocal hard_halt, risk_reason, retry_after_seconds
        stop_error_code = exc.code.value
        stop_exit_code = exc.exit_code
        stop_detail = exc.detail
        hard_halt = bool(exc.hard_halt)
        risk_reason = exc.risk_reason
        if not exc.hard_halt:
            wait_val = (
                exc.retry_after_seconds
                if exc.retry_after_seconds is not None
                else exc.cooldown_seconds
            )
            retry_after_seconds = round(float(wait_val), 2) if wait_val is not None else None
        else:
            retry_after_seconds = None

    resolved_flow_dir = resolve_flow_ledger_dir(flow_ledger_dir)

    try:
        for pos, idx in enumerate(pending_indices):
            item = normalized[idx]

            # Randomized anti-risk gap between consecutive generated shots in the same session.
            if generated_count > 0 and gap_high > 0:
                sleep_s = random.uniform(gap_low, gap_high)  # noqa: S311
                log.info(
                    "gemini.image_batch.gap_sleep",
                    profile=validated_profile,
                    next_item=item.id,
                    sleep_seconds=round(sleep_s, 2),
                )
                await asyncio.sleep(sleep_s)
                if enforce_risk_guard:
                    ks = find_active_kill_switch(home=home, flow_ledger_dir=resolved_flow_dir)
                    if ks is not None:
                        ks_err = GeminiRiskBlockedError(
                            f"Kill switch file appeared at {ks} during batch gap; halting.",
                            hard_halt=True,
                            risk_reason="kill_switch",
                        )
                        _record_risk_stop(ks_err)
                        _defer_remaining(pos, ks_err.detail)
                        break

            # Pre-item risk preflight (checks STOP_FLOW/STOP_GEMINI, global_halted,
            # linked Flow lock/quarantine, profile quarantine, 429 cooldown, burst cap).
            if enforce_risk_guard:
                try:
                    client.ledger.check_preflight(
                        validated_profile,
                        kind="image",
                        flow_ledger_dir=flow_ledger_dir,
                        linked_flow_profile=linked_flow_profile,
                        allow_flow_overlap=allow_flow_overlap,
                    )
                except GeminiError as exc:
                    _record_risk_stop(exc)
                    _defer_remaining(pos, exc.detail)
                    break

            item_aspect = item.aspect if item.aspect is not None else aspect
            item_model = item.model if item.model is not None else model
            item_sw = item.strip_watermark if item.strip_watermark is not None else strip_watermark
            can_reuse_turn = reuse_chat and generated_count > 0 and active_chat_url is not None
            item_chat_url = active_chat_url if reuse_chat else (item.chat_url or chat_url)
            item_new_chat = (not can_reuse_turn) and (item_chat_url is None)

            try:
                res = await client.generate_image(
                    item.prompt,
                    out_file=item.out_file,
                    ref_paths=item.ref_paths,
                    aspect=item_aspect,
                    model=item_model,
                    chat_url=item_chat_url,
                    new_chat=item_new_chat,
                    strip_watermark=item_sw,
                    timeout_seconds=timeout_seconds,
                    reuse_current_page=can_reuse_turn,
                )
            except GeminiError as exc:
                if (
                    exc.exit_code in _BATCH_STOP_EXIT_CODES
                    or exc.code == GeminiErrorCode.PROFILE_LOCKED
                ):
                    _record_risk_stop(exc)
                    outcomes_by_index[idx] = GeminiImageBatchItemOutcome(
                        id=item.id,
                        status="failed",
                        out_file=str(Path(item.out_file).expanduser()),
                        chat_url=active_chat_url,
                        error_code=exc.code.value,
                        exit_code=exc.exit_code,
                        detail=exc.detail,
                    )
                    _defer_remaining(pos + 1, f"Deferred after {exc.code.value}: {exc.detail}")
                    break

                outcomes_by_index[idx] = GeminiImageBatchItemOutcome(
                    id=item.id,
                    status="failed",
                    out_file=str(Path(item.out_file).expanduser()),
                    chat_url=active_chat_url,
                    error_code=exc.code.value,
                    exit_code=exc.exit_code,
                    detail=exc.detail,
                )
                continue

            generated_count += 1
            active_chat_url = res.chat_url
            if res.sha256 in seen_sha256:
                dup_of = seen_sha256[res.sha256]
                warnings.append(
                    f"Duplicate image SHA256 detected for item '{item.id}' (matches '{dup_of}')."
                )
            else:
                seen_sha256[res.sha256] = item.id

            outcomes_by_index[idx] = GeminiImageBatchItemOutcome(
                id=item.id,
                status="completed",
                out_file=res.out_file,
                chat_url=res.chat_url,
                width=res.width,
                height=res.height,
                aspect_ratio=res.aspect_ratio,
                bytes=res.bytes,
                sha256=res.sha256,
                download_method=res.download_method,
                watermark_stripped=res.watermark_stripped,
                watermark_note=res.watermark_note,
                elapsed_seconds=res.elapsed_seconds,
            )
    finally:
        await client.close()

    all_outcomes = [outcomes_by_index[i] for i in range(len(normalized))]
    completed = [o for o in all_outcomes if o.status == "completed"]
    skipped = [o for o in all_outcomes if o.status == "skipped"]
    failed = [o for o in all_outcomes if o.status == "failed"]
    deferred = [o for o in all_outcomes if o.status == "deferred"]

    ok = (not failed) and (not deferred) and (stop_error_code is None)
    partial = (not ok) and bool(completed or skipped)
    if not ok and stop_exit_code is None and failed:
        stop_error_code = failed[0].error_code
        stop_exit_code = failed[0].exit_code
        stop_detail = failed[0].detail

    return GeminiImageBatchResult(
        ok=ok,
        partial=partial,
        profile=validated_profile,
        reuse_chat=reuse_chat,
        chat_url=active_chat_url,
        total=len(normalized),
        completed=completed,
        skipped=skipped,
        failed=failed,
        deferred=deferred,
        items=all_outcomes,
        stop_error_code=stop_error_code,
        stop_exit_code=stop_exit_code,
        stop_detail=stop_detail,
        hard_halt=hard_halt,
        risk_reason=risk_reason,
        retry_after_seconds=retry_after_seconds,
        elapsed_seconds=round(time.monotonic() - started, 2),
        warnings=warnings,
    )


async def submit_task(
    prompt: str,
    *,
    profile: str = DEFAULT_GEMINI_PROFILE,
    attachments: Sequence[Path | str] = (),
    model: str | None = None,
    headless: bool = False,
    timeout_seconds: int = 120,
    chrome_executable: str | None = None,
    proxy: str | None = None,
    home: Path | None = None,
    debug_dir: Path | None = None,
    enforce_risk_guard: bool = True,
    flow_ledger_dir: Path | str | None = None,
    linked_flow_profile: str | None = None,
    allow_flow_overlap: bool = False,
) -> GeminiTaskSubmitResult:
    """One-shot Gemini Ultra Tasks submit helper managing profile lease and browser lifecycle."""
    client = GeminiWebClient(
        profile=profile,
        headless=headless,
        chrome_executable=chrome_executable,
        proxy=proxy,
        home=home,
        debug_dir=debug_dir,
        enforce_risk_guard=enforce_risk_guard,
        flow_ledger_dir=flow_ledger_dir,
        linked_flow_profile=linked_flow_profile,
        allow_flow_overlap=allow_flow_overlap,
    )
    try:
        return await client.submit_task(
            prompt,
            attachments=attachments,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    finally:
        await client.close()


async def harvest_task(
    chat_url: str,
    *,
    profile: str = DEFAULT_GEMINI_PROFILE,
    out_dir: Path | str | None = None,
    out_file: Path | str | None = None,
    out_names: Sequence[str] = (),
    download_images: bool = True,
    extract_json: bool = True,
    expected_images: int | None = None,
    wait_timeout: int = 600,
    download_timeout: int = 50,
    poll_interval: float = 10.0,
    min_image_dim: int = 512,
    strip_watermark: bool = False,
    aspect: str | None = None,
    headless: bool = False,
    chrome_executable: str | None = None,
    proxy: str | None = None,
    home: Path | None = None,
    debug_dir: Path | None = None,
    enforce_risk_guard: bool = True,
    flow_ledger_dir: Path | str | None = None,
    linked_flow_profile: str | None = None,
    allow_flow_overlap: bool = False,
) -> GeminiTaskHarvestResult:
    """One-shot Gemini Ultra Tasks harvest helper managing profile lease and browser lifecycle."""
    client = GeminiWebClient(
        profile=profile,
        headless=headless,
        chrome_executable=chrome_executable,
        proxy=proxy,
        home=home,
        debug_dir=debug_dir,
        min_image_dim=min_image_dim,
        enforce_risk_guard=enforce_risk_guard,
        flow_ledger_dir=flow_ledger_dir,
        linked_flow_profile=linked_flow_profile,
        allow_flow_overlap=allow_flow_overlap,
    )
    try:
        return await client.harvest_task(
            chat_url,
            out_dir=out_dir,
            out_file=out_file,
            out_names=out_names,
            download_images=download_images,
            extract_json=extract_json,
            expected_images=expected_images,
            wait_timeout=wait_timeout,
            download_timeout=download_timeout,
            poll_interval=poll_interval,
            min_image_dim=min_image_dim,
            strip_watermark=strip_watermark,
            aspect=aspect,
        )
    finally:
        await client.close()

