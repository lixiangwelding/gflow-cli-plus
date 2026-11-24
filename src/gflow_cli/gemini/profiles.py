"""Isolated Gemini Chrome profile management, OS locking, and login/status workflows."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast
from urllib.parse import urlsplit

import structlog
from rich.console import Console

from gflow_cli.browser_manager import (
    browser_proxy_kwargs,
    disable_features_args,
    is_chrome_available,
    is_playwright_chrome_channel_available,
    profile_in_use_error,
    resolved_chrome_binary,
)
from gflow_cli.config import BrowserEngine, get_settings
from gflow_cli.gemini.models import (
    DEFAULT_GEMINI_PROFILE,
    GEMINI_APP_URL,
    GEMINI_PROFILE_PREFIX,
    GeminiConfigError,
    GeminiError,
    GeminiLoginResult,
    GeminiLoginTimeoutError,
    GeminiProfileLockedError,
    GeminiProfileStatus,
)

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

if TYPE_CHECKING:
    from types import TracebackType

log = structlog.get_logger(__name__)

_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SAFE_LOCK_NAME_RE = re.compile(r"[^\w\-]")
_METADATA_VERSION = 1
_METADATA_MAX_BYTES = 4095
_PROFILE_META_FILE = ".gemini_profile_meta.json"

#: Selectors indicating Gemini Web's logged-in prompt composer is rendered and ready.
COMPOSER_SELECTORS: tuple[str, ...] = (
    'rich-textarea div.ql-editor.textarea[contenteditable="true"]',
    'div.ql-editor.textarea[contenteditable="true"]',
    'rich-textarea div.ql-editor[contenteditable="true"]',
    'div.ql-editor.textarea',
    'rich-textarea [contenteditable="true"][role="textbox"]',
    'div.ql-editor[contenteditable="true"][role="textbox"]',
    '[role="textbox"][contenteditable="true"][aria-label*="prompt" i]',
    '[role="textbox"][contenteditable="true"][aria-label*="Gemini" i]',
    '[role="textbox"][contenteditable="true"][data-placeholder]',
    'rich-textarea [contenteditable="true"]',
    'textarea[aria-label*="prompt" i]',
)

#: Selectors for signed-out sign-in buttons on Gemini landing / guest pages.
SIGNED_OUT_SIGNIN_SELECTORS: tuple[str, ...] = (
    'a[href*="accounts.google.com/ServiceLogin"]',
    'a[href*="accounts.google.com/signin"]',
    'a[href*="accounts.google.com/AccountChooser"]',
    'a[aria-label*="Sign in" i]',
    'button[aria-label*="Sign in" i]',
    'a[aria-label*="登录"]',
    'button[aria-label*="登录"]',
    'button:has-text("Sign in")',
    'a:has-text("Sign in")',
    'button:has-text("登录")',
    'a:has-text("登录")',
)

#: Google authentication session cookies that prove a signed-in Google session.
GOOGLE_AUTH_COOKIE_NAMES: frozenset[str] = frozenset(
    {
        "__Secure-1PSID",
        "__Secure-3PSID",
        "SAPISID",
        "__Secure-1PAPISID",
        "SID",
    }
)

#: Origins queried on ``BrowserContext.cookies(...)`` to check Google auth cookies.
GOOGLE_AUTH_COOKIE_URLS: tuple[str, ...] = (
    "https://gemini.google.com",
    "https://google.com",
    "https://accounts.google.com",
)

#: Account avatar / profile button selectors for best-effort email/account hint extraction.
ACCOUNT_BUTTON_SELECTORS: tuple[str, ...] = (
    'a[aria-label*="Google Account" i]',
    'button[aria-label*="Google Account" i]',
    'a[aria-label*="Google 账号" i]',
    'button[aria-label*="Google 账号" i]',
    'a[href*="SignOutOptions"]',
)

_AUTH_URL_MARKERS: tuple[str, ...] = (
    "accounts.google.com",
    "signin",
    "servicelogin",
    "consent.google.com",
    "v3/signin",
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Process-local lock registry, completely separate from Flow's profile_lease._registry.
_gemini_registry: dict[Path, GeminiProfileLease] = {}
_gemini_registry_guard = threading.Lock()


class _LeaseMetadata(TypedDict):
    version: int
    pid: int
    acquired_at: float
    profile_name: str
    owner_token: str


def validate_gemini_profile_name(name: str) -> str:
    """Validate and normalize a Gemini profile name.

    Rejects traversal attempts, whitespace, and empty strings so the resulting
    ``gemini_profile_<name>`` directory is always strictly isolated inside gflow home.
    """
    cleaned = name.strip()
    if not cleaned or not _PROFILE_NAME_RE.fullmatch(cleaned):
        raise GeminiConfigError(
            f"Invalid Gemini profile name {name!r}. "
            "Use 1-64 ASCII letters, digits, hyphens, or underscores."
        )
    return cleaned


def resolve_gemini_home(home: Path | None = None) -> Path:
    """Return the root data directory for Gemini profiles and locks."""
    if home is not None:
        return Path(home).expanduser()
    return get_settings().home


def gemini_profile_dir(
    profile_name: str = DEFAULT_GEMINI_PROFILE,
    *,
    home: Path | None = None,
) -> Path:
    """Return the isolated Chrome user-data-dir for a Gemini profile.

    Always lives at ``<home>/gemini_profile_<name>``, guaranteeing zero collision
    with Flow's ``<home>/profile_<name>`` directories.
    """
    validated = validate_gemini_profile_name(profile_name)
    base_home = resolve_gemini_home(home)
    target = base_home / f"{GEMINI_PROFILE_PREFIX}{validated}"
    # Defensive namespace assertion: never touch Flow's profile_* namespace.
    if not target.name.startswith(GEMINI_PROFILE_PREFIX):
        raise GeminiConfigError(f"Unsafe Gemini profile path resolved: {target}")
    try:
        target.resolve(strict=False).relative_to(base_home.resolve(strict=False))
    except ValueError as exc:
        raise GeminiConfigError(
            f"Gemini profile path {target} escapes home directory {base_home}."
        ) from exc
    return target


def gemini_locks_dir(*, home: Path | None = None) -> Path:
    """Return the dedicated lock directory for Gemini profiles (separate from Flow locks)."""
    return resolve_gemini_home(home) / "gemini_locks"


def _canonicalize(profile_dir: Path | str) -> Path:
    return Path(profile_dir).expanduser().resolve()


def _lock_filename(canonical: Path) -> str:
    digest = hashlib.sha256(str(canonical).encode("utf-8", "surrogateescape")).hexdigest()
    safe_name = _SAFE_LOCK_NAME_RE.sub("_", canonical.name) or "gemini_profile"
    return f"{safe_name}-{digest[:16]}.lock"


def _lock_nonblocking(fd: int) -> None:
    if sys.platform == "win32":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _read_lock_owner_pid(fd: int) -> int | None:
    try:
        os.lseek(fd, 1, os.SEEK_SET)
        raw = os.read(fd, _METADATA_MAX_BYTES)
        parsed = json.loads(raw.decode("utf-8"))
        if isinstance(parsed, dict):
            pid = cast("dict[str, Any]", parsed).get("pid")
            if isinstance(pid, int) and not isinstance(pid, bool):
                return pid
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return None
    return None


class GeminiProfileLease:
    """Exclusive OS + in-process lock for a single isolated Gemini Chrome profile.

    Mirrors :class:`gflow_cli.profile_lease.ProfileLease` while keeping lock files
    under ``<home>/gemini_locks/`` so Flow's live locks are never touched.
    """

    def __init__(
        self,
        profile_dir: Path | str,
        *,
        home: Path | None = None,
        wait_seconds: float = 0.0,
    ) -> None:
        self._canonical = _canonicalize(profile_dir)
        self._home = resolve_gemini_home(home)
        self.lock_path = gemini_locks_dir(home=self._home) / _lock_filename(self._canonical)
        self._fd: int | None = None
        self._metadata: _LeaseMetadata | None = None
        self._wait_seconds = max(0.0, float(wait_seconds))

    @property
    def owner_pid(self) -> int | None:
        return self._metadata["pid"] if self._metadata is not None else None

    def _register_same_process(self) -> None:
        with _gemini_registry_guard:
            existing = _gemini_registry.get(self._canonical)
            if existing is not None:
                pid = existing.owner_pid or os.getpid()
                raise GeminiProfileLockedError(
                    f"Gemini profile {self._canonical} is already held in this process (pid={pid})."
                )
            _gemini_registry[self._canonical] = self

    def _unregister(self) -> None:
        with _gemini_registry_guard:
            if _gemini_registry.get(self._canonical) is self:
                del _gemini_registry[self._canonical]

    def _acquire_kernel_lock(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.is_symlink():
            raise GeminiConfigError(f"Refusing to follow symlinked lock path: {self.lock_path}")

        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(self.lock_path, flags, 0o600)
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
            try:
                _lock_nonblocking(fd)
            except OSError as exc:
                owner_pid = _read_lock_owner_pid(fd)
                owner_info = f" (owner_pid={owner_pid})" if owner_pid is not None else ""
                raise GeminiProfileLockedError(
                    f"Gemini profile {self._canonical} is locked by another process{owner_info}."
                ) from exc

            metadata: _LeaseMetadata = {
                "version": _METADATA_VERSION,
                "pid": os.getpid(),
                "acquired_at": time.time(),
                "profile_name": self._canonical.name,
                "owner_token": uuid.uuid4().hex,
            }
            payload = json.dumps(metadata).encode("utf-8")
            os.lseek(fd, 1, os.SEEK_SET)
            os.write(fd, payload)
            os.ftruncate(fd, 1 + len(payload))
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._metadata = metadata

    def acquire(self) -> GeminiProfileLease:
        self._register_same_process()
        deadline = time.monotonic() + self._wait_seconds if self._wait_seconds > 0 else None
        try:
            while True:
                try:
                    self._acquire_kernel_lock()
                    return self
                except GeminiProfileLockedError:
                    if deadline is None or time.monotonic() >= deadline:
                        raise
                    time.sleep(min(0.5, max(0.05, deadline - time.monotonic())))
        except BaseException:
            self._unregister()
            raise

    def try_acquire(self) -> bool:
        prev_wait = self._wait_seconds
        self._wait_seconds = 0.0
        try:
            self.acquire()
            return True
        except GeminiProfileLockedError:
            return False
        finally:
            self._wait_seconds = prev_wait

    def release(self) -> None:
        if self._fd is not None:
            fd, self._fd = self._fd, None
            try:
                _unlock(fd)
            finally:
                os.close(fd)
        self._metadata = None
        self._unregister()

    def __enter__(self) -> GeminiProfileLease:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    async def aacquire(self) -> GeminiProfileLease:
        if self._wait_seconds <= 0:
            return self.acquire()
        self._register_same_process()
        deadline = time.monotonic() + self._wait_seconds
        try:
            while True:
                try:
                    self._acquire_kernel_lock()
                    return self
                except GeminiProfileLockedError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    await asyncio.sleep(min(0.5, remaining))
        except BaseException:
            self._unregister()
            raise

    async def __aenter__(self) -> GeminiProfileLease:
        return await self.aacquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def probe_profile_lock(
    profile_dir: Path | str,
    *,
    home: Path | None = None,
) -> tuple[bool, int | None, Path]:
    """Non-destructively probe whether ``profile_dir`` is locked.

    Returns ``(is_locked, owner_pid, lock_path)``. Never disturbs a running holder.
    """
    canonical = _canonicalize(profile_dir)
    resolved_home = resolve_gemini_home(home)
    lock_path = gemini_locks_dir(home=resolved_home) / _lock_filename(canonical)

    with _gemini_registry_guard:
        existing = _gemini_registry.get(canonical)
        if existing is not None:
            return True, existing.owner_pid or os.getpid(), lock_path

    if not lock_path.exists():
        return False, None, lock_path

    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(lock_path, flags)
    except OSError:
        return False, None, lock_path

    try:
        try:
            _lock_nonblocking(fd)
        except OSError:
            return True, _read_lock_owner_pid(fd), lock_path
        _unlock(fd)
        return False, None, lock_path
    finally:
        os.close(fd)


def _normalize_proxy_url(raw: str) -> str | None:
    cleaned = raw.strip()
    if not cleaned:
        return None
    if cleaned.startswith("socks5h://"):
        cleaned = "socks5://" + cleaned[len("socks5h://") :]
    parsed = urlsplit(cleaned)
    if parsed.scheme not in ("http", "https", "socks5"):
        return None
    if not parsed.hostname:
        return None
    return cleaned


def resolve_browser_proxy(proxy: str | None = None) -> str | None:
    """Resolve the browser proxy from explicit arg, gflow settings, or standard proxy envs.

    Precedence:
    1. Explicit ``proxy`` argument
    2. ``GFLOW_GEMINI_PROXY`` env var
    3. ``GFLOW_CLI_BROWSER_PROXY`` (via ``get_settings().browser_proxy``)
    4. ``HTTPS_PROXY`` / ``https_proxy`` / ``HTTP_PROXY`` / ``http_proxy`` / ``ALL_PROXY``
       (supports Clash / local system proxies cleanly).
    """
    candidates: list[str | None] = [
        proxy,
        os.environ.get("GFLOW_GEMINI_PROXY"),
    ]
    with contextlib.suppress(Exception):
        candidates.append(get_settings().browser_proxy)
    for env_key in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        candidates.append(os.environ.get(env_key))

    for candidate in candidates:
        if not candidate:
            continue
        normalized = _normalize_proxy_url(candidate)
        if normalized:
            return normalized
    return None


def build_gemini_launch_kwargs(
    profile_dir: Path,
    *,
    headless: bool,
    chrome_executable: str | None = None,
    proxy: str | None = None,
    visible_login: bool = False,
) -> dict[str, Any]:
    """Build Playwright ``launch_persistent_context`` kwargs for Gemini Web.

    Follows gflow's proven anti-detection conventions:
    - Real macOS Google Chrome via ``channel="chrome"`` (or explicit ``executable_path``)
    - ``chromium_sandbox=True`` and ``ignore_default_args=["--enable-automation"]``
    - ``--disable-blink-features=AutomationControlled`` + ``--password-store=basic``
    - Respects ``GFLOW_GEMINI_PROXY`` / ``GFLOW_CLI_BROWSER_PROXY`` / ``HTTPS_PROXY`` /
      ``ALL_PROXY``
    """
    effective_headless = False if visible_login else headless
    args: list[str] = [
        "--disable-blink-features=AutomationControlled",
        "--no-first-run",
        "--no-default-browser-check",
        "--password-store=basic",
        "--window-size=1440,960",
        *disable_features_args(BrowserEngine.PLAYWRIGHT),
    ]
    if visible_login:
        args.append("--window-position=80,60")
    elif not effective_headless:
        win_pos = os.environ.get("GFLOW_GEMINI_WINDOW_POSITION", "").strip()
        if win_pos:
            args.append(f"--window-position={win_pos}")

    kwargs: dict[str, Any] = {
        "user_data_dir": str(profile_dir),
        "headless": effective_headless,
        "accept_downloads": True,
        "chromium_sandbox": True,
        "ignore_default_args": ["--enable-automation"],
        "args": args,
    }
    if effective_headless:
        kwargs["viewport"] = {"width": 1440, "height": 960}
    else:
        kwargs["no_viewport"] = True

    if chrome_executable:
        kwargs["executable_path"] = chrome_executable
    elif is_playwright_chrome_channel_available():
        kwargs["channel"] = "chrome"
    else:
        discovered = resolved_chrome_binary()
        if discovered:
            kwargs["executable_path"] = discovered
        else:
            meta = _read_profile_meta(profile_dir)
            if meta.get("chrome_channel") == "chrome":
                raise GeminiConfigError(
                    f"Gemini profile {profile_dir} was created with system Google Chrome, "
                    "but Google Chrome is no longer available. Refusing silent downgrade to "
                    "bundled Chromium (which can invalidate Chrome 130+ session cookies)."
                )

    resolved_proxy = resolve_browser_proxy(proxy)
    if resolved_proxy:
        kwargs.update(browser_proxy_kwargs(resolved_proxy))

    return kwargs


def is_auth_redirect_url(url: str) -> bool:
    """Return True when ``url`` indicates a Google sign-in / account redirect."""
    low = (url or "").lower()
    return any(marker in low for marker in _AUTH_URL_MARKERS)


def is_gemini_app_url(url: str) -> bool:
    """Return True when ``url`` is on the ``gemini.google.com`` host."""
    low = (url or "").lower()
    return "gemini.google.com" in low and not is_auth_redirect_url(low)


async def has_google_auth_cookies(context: Any) -> bool:
    """Return True iff ``context`` holds at least one valid Google auth session cookie."""
    if context is None or not hasattr(context, "cookies"):
        return False
    try:
        cookies = await context.cookies(list(GOOGLE_AUTH_COOKIE_URLS))
    except TypeError:
        try:
            cookies = await context.cookies()
        except Exception:
            return False
    except Exception:
        return False

    now = time.time()
    for cookie in cookies or ():
        if isinstance(cookie, dict):
            c_dict = cast("dict[str, Any]", cookie)
            name = str(c_dict.get("name") or "")
            value = str(c_dict.get("value") or "").strip()
            expires = c_dict.get("expires")
        else:
            name = str(getattr(cookie, "name", "") or "")
            value = str(getattr(cookie, "value", "") or "").strip()
            expires = getattr(cookie, "expires", None)
        if name not in GOOGLE_AUTH_COOKIE_NAMES or not value:
            continue
        if isinstance(expires, (int, float)) and expires > 0 and float(expires) <= now:
            continue
        return True
    return False


async def _has_visible_signin_cta(page: Any) -> bool:
    """Return True when Gemini renders a visible Sign-in / 登录 button (logged-out guest UI)."""
    for sign_in_sel in SIGNED_OUT_SIGNIN_SELECTORS:
        try:
            btn = page.locator(sign_in_sel).first
            if await btn.is_visible(timeout=250):
                return True
        except Exception:
            continue
    try:
        found = await page.evaluate(
            """() => {
                const sel = [
                    'header a', 'header button',
                    'nav a', 'nav button',
                    '[role="banner"] a', '[role="banner"] button',
                ].join(', ');
                const els = document.querySelectorAll(sel);
                for (const el of els) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width <= 0 || rect.height <= 0) continue;
                    const text = (el.innerText || el.textContent || '').trim().toLowerCase();
                    const aria = (el.getAttribute('aria-label') || '').trim().toLowerCase();
                    const href = (el.getAttribute('href') || '').toLowerCase();
                    if (
                        text === 'sign in' || text === '登录' ||
                        aria.includes('sign in') || aria.includes('登录') ||
                        href.includes('accounts.google.com/servicelogin') ||
                        href.includes('accounts.google.com/signin')
                    ) {
                        return true;
                    }
                }
                return false;
            }"""
        )
        return bool(found)
    except Exception:
        return False


async def detect_gemini_composer(page: Any) -> bool:
    """Return True when Gemini's prompt composer is visible and no Sign-in CTA is shown."""
    url = str(getattr(page, "url", "") or "")
    if not is_gemini_app_url(url):
        return False

    has_composer = False
    for selector in COMPOSER_SELECTORS:
        try:
            loc = page.locator(selector).first
            if await loc.is_visible(timeout=500):
                has_composer = True
                break
        except Exception:
            continue
    if not has_composer:
        return False

    if await _has_visible_signin_cta(page):
        return False
    return True


async def detect_gemini_authenticated(context: Any, page: Any) -> tuple[bool, str | None]:
    """Strictly verify BOTH Google auth session cookies AND signed-in Gemini Web UI.

    Prevents false positives on Gemini's logged-out guest composer, where ``rich-textarea``
    is rendered even before signing in.
    """
    url = str(getattr(page, "url", "") or "")
    if not is_gemini_app_url(url):
        return False, None
    if not await has_google_auth_cookies(context):
        return False, None
    if not await detect_gemini_composer(page):
        return False, None
    account_hint = await extract_account_hint(page)
    return True, account_hint


async def extract_account_hint(page: Any) -> str | None:
    """Best-effort extract the signed-in Google account email/label from the header."""
    for selector in ACCOUNT_BUTTON_SELECTORS:
        try:
            loc = page.locator(selector).first
            if await loc.count() == 0:
                continue
            label = await loc.get_attribute("aria-label", timeout=500)
            if not label:
                label = await loc.get_attribute("title", timeout=500)
            if label:
                match = _EMAIL_RE.search(label)
                if match:
                    return match.group(0)
                cleaned = " ".join(label.split())
                if cleaned:
                    return cleaned[:80]
        except Exception:
            continue
    return None


def _has_cookies_store(profile_dir: Path) -> bool:
    return any(
        candidate.is_file()
        for candidate in (
            profile_dir / "Default" / "Network" / "Cookies",
            profile_dir / "Default" / "Cookies",
            profile_dir / "Cookies",
        )
    )


def _read_profile_meta(profile_dir: Path) -> dict[str, Any]:
    meta_path = profile_dir / _PROFILE_META_FILE
    if not meta_path.is_file():
        return {}
    try:
        parsed = json.loads(meta_path.read_text(encoding="utf-8"))
        return cast("dict[str, Any]", parsed) if isinstance(parsed, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_profile_meta(
    profile_dir: Path,
    *,
    profile_name: str,
    account_hint: str | None,
    logged_in: bool,
) -> None:
    meta_path = profile_dir / _PROFILE_META_FILE
    payload = {
        "profile": profile_name,
        "last_login_utc": datetime.now(UTC).isoformat(),
        "logged_in": logged_in,
        "account_hint": account_hint,
        "chrome_channel": "chrome" if is_playwright_chrome_channel_available() else None,
    }
    with contextlib.suppress(OSError):
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def wrap_launch_error(exc: BaseException, profile_dir: Path) -> GeminiError:
    in_use = profile_in_use_error(exc, profile_dir)
    if in_use is not None:
        return GeminiProfileLockedError(in_use.detail)
    return GeminiError(
        f"Failed to launch Chrome for Gemini profile {profile_dir}: {exc}",
        remediation_hint=(
            "Ensure Google Chrome is installed and no other Chrome window is using this "
            "Gemini profile directory."
        ),
    )


def get_async_playwright() -> Any:
    """Return ``async_playwright`` from the configured engine (patchright opt-in or playwright)."""
    import importlib

    engine_env = os.environ.get("GFLOW_CLI_BROWSER_ENGINE", "").strip().lower()
    if engine_env == "patchright":
        try:
            mod = importlib.import_module("patchright.async_api")
            return mod.async_playwright
        except (ImportError, AttributeError):
            log.warning("gemini.patchright_requested_but_unavailable_falling_back_to_playwright")
    from playwright.async_api import async_playwright

    return async_playwright


async def login_profile(
    profile_name: str = DEFAULT_GEMINI_PROFILE,
    *,
    timeout_seconds: int = 300,
    chrome_executable: str | None = None,
    proxy: str | None = None,
    home: Path | None = None,
) -> GeminiLoginResult:
    """Launch a visible isolated Chrome window for manual Google Ultra sign-in to Gemini.

    Detects when the Gemini Web composer is logged in and ready, waits briefly for
    Chrome to flush session cookies, and closes cleanly. Also handles user closing
    the window manually or pressing Ctrl+C without corrupting the profile.
    """
    async_playwright = get_async_playwright()

    validated = validate_gemini_profile_name(profile_name)
    profile_dir = gemini_profile_dir(validated, home=home)
    profile_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    warnings: list[str] = []

    if timeout_seconds < 5:
        raise GeminiConfigError("timeout_seconds must be at least 5 seconds.")

    async with GeminiProfileLease(profile_dir, home=home):
        launch_kwargs = build_gemini_launch_kwargs(
            profile_dir,
            headless=False,
            chrome_executable=chrome_executable,
            proxy=proxy,
            visible_login=True,
        )
        log.info(
            "gemini.login.launch",
            profile=validated,
            profile_dir=str(profile_dir),
            channel=launch_kwargs.get("channel"),
            has_proxy="proxy" in launch_kwargs,
        )

        async with async_playwright() as pw:
            try:
                context = await pw.chromium.launch_persistent_context(**launch_kwargs)
            except Exception as exc:
                raise wrap_launch_error(exc, profile_dir) from exc

            logged_in = False
            account_hint: str | None = None
            final_url = GEMINI_APP_URL
            browser_closed = asyncio.Event()

            def _on_context_close(_ctx: Any = None) -> None:
                browser_closed.set()

            context.on("close", _on_context_close)

            try:
                page = context.pages[0] if context.pages else await context.new_page()
                with contextlib.suppress(Exception):
                    await page.goto(GEMINI_APP_URL, wait_until="domcontentloaded", timeout=45_000)

                err_console = Console(stderr=True)
                err_console.print(
                    f"[bold cyan]Gemini Login ({validated})[/bold cyan]: "
                    "Chrome opened at https://gemini.google.com/app.\n"
                    "  1. If Gemini opened in guest mode, click [bold]Sign in[/bold] (top-right) "
                    "and sign in to your Google Ultra account.\n"
                    "  2. Keep the window open until Gemini loads while signed in — "
                    "session cookies will auto-save and close after 2-3s.\n"
                    f"  3. Waiting up to {timeout_seconds}s (press Ctrl+C or close window to stop)."
                )

                deadline = time.monotonic() + float(timeout_seconds)
                consecutive_ready = 0
                while time.monotonic() < deadline:
                    if browser_closed.is_set():
                        break
                    active_pages = [p for p in context.pages if not p.is_closed()]
                    if not active_pages:
                        browser_closed.set()
                        break
                    page = active_pages[-1]
                    final_url = str(page.url or GEMINI_APP_URL)

                    is_auth, detected_hint = await detect_gemini_authenticated(context, page)
                    if is_auth:
                        consecutive_ready += 1
                        if detected_hint:
                            account_hint = detected_hint
                        if consecutive_ready >= 2:
                            logged_in = True
                            # Allow Chrome to flush SQLite cookie journal before closing.
                            await asyncio.sleep(2.5)
                            break
                    else:
                        consecutive_ready = 0

                    await asyncio.sleep(1.5)
                else:
                    if not logged_in:
                        raise GeminiLoginTimeoutError(
                            f"Gemini sign-in was not detected within {timeout_seconds}s "
                            f"(last_url={final_url})."
                        )
            except KeyboardInterrupt:
                warnings.append("Interrupted by user (Ctrl+C); saved current profile state.")
            finally:
                with contextlib.suppress(Exception):
                    await context.close()

    elapsed = round(time.monotonic() - started, 2)
    if logged_in:
        _write_profile_meta(
            profile_dir,
            profile_name=validated,
            account_hint=account_hint,
            logged_in=True,
        )
        with contextlib.suppress(Exception):
            from gflow_cli.gemini.risk_guard import GeminiRiskLedger

            GeminiRiskLedger(home=home).clear_manual_quarantine(validated)
        detail = f"Signed in to Gemini Web and saved persistent session in {profile_dir}."
    else:
        has_cookies = _has_cookies_store(profile_dir)
        detail = (
            "Browser window closed before Gemini composer readiness was confirmed; "
            + (
                "cookie store exists on disk — verify with `gflow gemini status --verify-live`."
                if has_cookies
                else "no session cookies detected yet."
            )
        )

    return GeminiLoginResult(
        ok=logged_in,
        profile=validated,
        profile_dir=str(profile_dir),
        logged_in=logged_in,
        chat_url=final_url,
        account_hint=account_hint,
        elapsed_seconds=elapsed,
        detail=detail,
        warnings=warnings,
    )


async def check_profile_status(
    profile_name: str = DEFAULT_GEMINI_PROFILE,
    *,
    verify_live: bool = False,
    headless: bool = True,
    timeout_seconds: int = 30,
    chrome_executable: str | None = None,
    proxy: str | None = None,
    home: Path | None = None,
) -> GeminiProfileStatus:
    """Check isolated Gemini profile existence, lock status, and optional live session."""
    validated = validate_gemini_profile_name(profile_name)
    profile_dir = gemini_profile_dir(validated, home=home)
    exists = profile_dir.is_dir()
    locked, owner_pid, lock_path = probe_profile_lock(profile_dir, home=home)
    chrome_binary = chrome_executable or resolved_chrome_binary()
    chrome_avail = bool(chrome_executable) or is_chrome_available()
    meta = _read_profile_meta(profile_dir) if exists else {}
    account_hint = meta.get("account_hint") if isinstance(meta.get("account_hint"), str) else None
    warnings: list[str] = []
    risk_snapshot: dict[str, Any] | None = None
    with contextlib.suppress(Exception):
        from gflow_cli.gemini.risk_guard import GeminiRiskLedger

        snap = GeminiRiskLedger(home=home).get_profile_risk_snapshot(validated)
        risk_snapshot = snap
        if snap.get("kill_switch_active"):
            warnings.append(f"Kill switch active: {snap.get('kill_switch_path')}")
        if snap.get("quarantined"):
            reason = snap.get("quarantine_reason") or "active cooldown"
            warnings.append(f"Profile is quarantined ({reason}).")

    if not verify_live:
        has_cookies = _has_cookies_store(profile_dir) if exists else False
        if exists and not has_cookies:
            warnings.append(
                "Profile directory exists but has no Chrome Cookies store yet; "
                "run `gflow gemini login`."
            )
        detail = (
            f"Profile '{validated}' is locked by pid={owner_pid}."
            if locked
            else (
                f"Profile '{validated}' exists on disk"
                + (" (cookies present)." if has_cookies else " (no cookies yet).")
                if exists
                else f"Profile '{validated}' does not exist yet. Run `gflow gemini login`."
            )
        )
        return GeminiProfileStatus(
            ok=exists and not locked,
            profile=validated,
            profile_dir=str(profile_dir),
            exists=exists,
            locked=locked,
            lock_path=str(lock_path),
            lock_owner_pid=owner_pid,
            chrome_available=chrome_avail,
            chrome_binary=chrome_binary,
            logged_in=None,
            verified_live=False,
            account_hint=account_hint,
            detail=detail,
            warnings=warnings,
            risk=risk_snapshot,
        )

    if locked:
        return GeminiProfileStatus(
            ok=False,
            profile=validated,
            profile_dir=str(profile_dir),
            exists=exists,
            locked=True,
            lock_path=str(lock_path),
            lock_owner_pid=owner_pid,
            chrome_available=chrome_avail,
            chrome_binary=chrome_binary,
            logged_in=None,
            verified_live=False,
            account_hint=account_hint,
            detail=(
                f"Cannot verify live status while profile '{validated}' is locked "
                f"(pid={owner_pid})."
            ),
            warnings=warnings,
            risk=risk_snapshot,
        )

    if not exists:
        return GeminiProfileStatus(
            ok=False,
            profile=validated,
            profile_dir=str(profile_dir),
            exists=False,
            locked=False,
            lock_path=str(lock_path),
            lock_owner_pid=None,
            chrome_available=chrome_avail,
            chrome_binary=chrome_binary,
            logged_in=False,
            verified_live=True,
            account_hint=None,
            detail=f"Profile '{validated}' does not exist yet. Run `gflow gemini login`.",
            warnings=warnings,
            risk=risk_snapshot,
        )

    async_playwright = get_async_playwright()

    logged_in = False
    live_hint = account_hint
    detail = ""

    async with GeminiProfileLease(profile_dir, home=home):
        launch_kwargs = build_gemini_launch_kwargs(
            profile_dir,
            headless=headless,
            chrome_executable=chrome_executable,
            proxy=proxy,
            visible_login=False,
        )
        async with async_playwright() as pw:
            try:
                context = await pw.chromium.launch_persistent_context(**launch_kwargs)
            except Exception as exc:
                raise wrap_launch_error(exc, profile_dir) from exc
            try:
                page = context.pages[0] if context.pages else await context.new_page()
                await page.goto(
                    GEMINI_APP_URL,
                    wait_until="domcontentloaded",
                    timeout=max(10_000, int(timeout_seconds * 1000)),
                )
                deadline = time.monotonic() + float(timeout_seconds)
                guest_checks = 0
                while time.monotonic() < deadline:
                    url = str(page.url or "")
                    if is_auth_redirect_url(url):
                        logged_in = False
                        detail = f"Redirected to Google sign-in ({url})."
                        break
                    is_auth, detected_hint = await detect_gemini_authenticated(context, page)
                    if is_auth:
                        logged_in = True
                        live_hint = detected_hint or live_hint
                        detail = (
                            "Live Gemini Web session verified (auth cookies + prompt composer "
                            "ready)."
                        )
                        break
                    if is_gemini_app_url(url) and not await has_google_auth_cookies(context):
                        guest_checks += 1
                        if guest_checks >= 3:
                            logged_in = False
                            detail = (
                                "Gemini loaded in logged-out guest mode (no Google auth session "
                                "cookies found). Run `gflow gemini login`."
                            )
                            break
                    await asyncio.sleep(1.0)
                else:
                    logged_in = False
                    detail = (
                        f"Timed out after {timeout_seconds}s waiting for Gemini composer "
                        f"(url={page.url})."
                    )
            finally:
                with contextlib.suppress(Exception):
                    await context.close()

    return GeminiProfileStatus(
        ok=logged_in,
        profile=validated,
        profile_dir=str(profile_dir),
        exists=True,
        locked=False,
        lock_path=str(lock_path),
        lock_owner_pid=None,
        chrome_available=chrome_avail,
        chrome_binary=chrome_binary,
        logged_in=logged_in,
        verified_live=True,
        account_hint=live_hint,
        detail=detail,
        warnings=warnings,
        risk=risk_snapshot,
    )
