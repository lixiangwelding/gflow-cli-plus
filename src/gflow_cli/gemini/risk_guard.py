"""Persistent anti-risk ledger, backoff/quarantine state machine, and Flow cross-guard.

Aligned with the proven anti-risk rules in ``flow_risk_ledger.py`` and
``docs/ACCOUNT_SAFETY.md``:
- Pace multi-prompt runs with randomized jitter (global gap, per-profile gap, burst cap).
- Zero blind retries on CAPTCHA / WAF / challenge / auth-expired / 429 prompt submit.
- Respect both ``STOP_GEMINI`` and ``STOP_FLOW`` kill switch files before launch and after sleep.
- Strictly read-only inspection of Flow's ledger (``<flow_ledger_dir>/state.json`` and
  ``profile_<linked>.lock``) so Gemini Web and Flow never collide on the same Google Ultra account.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import structlog

from gflow_cli.gemini.models import (
    GeminiAuthRequiredError,
    GeminiRateLimitError,
    GeminiRiskBlockedError,
)
from gflow_cli.gemini.profiles import resolve_gemini_home, validate_gemini_profile_name

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Anti-risk pacing & quarantine constants (aligned with flow_risk_ledger.py)
# ---------------------------------------------------------------------------

GLOBAL_GAP_RANGE: tuple[float, float] = (8.0, 15.0)
CHAT_PROFILE_GAP_RANGE: tuple[float, float] = (10.0, 25.0)
IMAGE_PROFILE_GAP_RANGE: tuple[float, float] = (15.0, 35.0)
ELEVATED_PROFILE_GAP_RANGE: tuple[float, float] = (45.0, 90.0)

WAF_QUARANTINE_S: float = 2700.0  # 45 minutes
WAF_ELEVATED_DURATION_S: float = 7200.0  # 2 hours post-WAF at 45-90s gap
WAF_GLOBAL_PAUSE_RANGE: tuple[float, float] = (30.0, 60.0)
SHARED_IP_WINDOW_S: float = 1800.0  # 30 minutes
SHARED_IP_WAF_THRESHOLD: int = 2

RATE_LIMIT_WINDOW_S: float = 1800.0  # 30 minutes
RATE_LIMIT_BACKOFF_STEPS: tuple[float, float, float] = (120.0, 300.0, 900.0)
PROFILE_LOCK_BACKOFF_S: float = 20.0

BURST_WINDOW_SECONDS: float = 600.0  # 10 minutes
DEFAULT_MAX_IMAGE_PER_BURST: int = 8  # Below Flow's ~14/10m cliff

FLOW_STAGGER_RANGE: tuple[float, float] = (10.0, 20.0)

# Download-only transport retry (NEVER re-submits prompt)
DOWNLOAD_RETRY_ATTEMPTS: int = 3
DOWNLOAD_RETRY_BASE_S: float = 1.0
DOWNLOAD_RETRY_CAP_S: float = 60.0

KILL_SWITCH_FILENAMES: tuple[str, ...] = ("STOP_GEMINI", "STOP_FLOW")


def jitter_delay(seconds: float, *, spread: float = 0.25) -> float:
    """Randomize a UI micro-delay or backoff interval by +/- ``spread`` (default 25%)."""
    if seconds <= 0:
        return 0.0
    factor = random.uniform(max(0.1, 1.0 - spread), 1.0 + spread)  # noqa: S311 - non-crypto pacing jitter
    return seconds * factor


def max_image_per_burst() -> int:
    """Resolve rolling 10-minute image burst ceiling (0 disables burst cap)."""
    raw = os.environ.get("GFLOW_GEMINI_MAX_PER_BURST", "").strip()
    if not raw:
        return DEFAULT_MAX_IMAGE_PER_BURST
    try:
        parsed = int(raw)
        return max(0, parsed)
    except ValueError:
        return DEFAULT_MAX_IMAGE_PER_BURST


def resolve_flow_ledger_dir(flow_ledger_dir: Path | str | None = None) -> Path | None:
    """Resolve optional Flow ledger directory from explicit arg or ``GFLOW_FLOW_LEDGER_DIR``."""
    if flow_ledger_dir is not None:
        return Path(flow_ledger_dir).expanduser()
    env_val = os.environ.get("GFLOW_FLOW_LEDGER_DIR", "").strip()
    if env_val:
        return Path(env_val).expanduser()
    return None


def resolve_linked_flow_profile(linked_flow_profile: str | None = None) -> str | None:
    """Resolve optional Flow profile linked to the same Google Ultra account."""
    if linked_flow_profile is not None:
        cleaned = linked_flow_profile.strip()
        return cleaned or None
    for env_key in ("GFLOW_GEMINI_LINKED_FLOW_PROFILE", "GFLOW_LINKED_FLOW_PROFILE"):
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            return env_val
    return None


def find_active_kill_switch(
    *,
    home: Path | None = None,
    flow_ledger_dir: Path | None = None,
    cwd: Path | None = None,
) -> Path | None:
    """Return the first existing ``STOP_GEMINI`` or ``STOP_FLOW`` kill switch file, if any."""
    resolved_home = resolve_gemini_home(home)
    search_dirs: list[Path] = [
        cwd if cwd is not None else Path.cwd(),
        resolved_home,
        resolved_home / "gemini_ledger",
    ]
    if flow_ledger_dir is not None:
        search_dirs.append(flow_ledger_dir)
        search_dirs.append(flow_ledger_dir.parent)

    seen: set[Path] = set()
    for directory in search_dirs:
        resolved_dir = directory.expanduser()
        if resolved_dir in seen:
            continue
        seen.add(resolved_dir)
        for name in KILL_SWITCH_FILENAMES:
            candidate = resolved_dir / name
            if candidate.exists():
                return candidate
    return None


def _is_file_flock_held(lock_path: Path) -> bool:
    """Read-only non-blocking probe of an external advisory lock file (never creates/mutates)."""
    if not lock_path.is_file():
        return False
    if sys.platform == "win32":
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
    else:
        # POSIX flock works on O_RDONLY descriptors; never open Flow's live lock for writing.
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags)
    except OSError:
        return False
    try:
        if sys.platform == "win32":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


@dataclass(slots=True)
class FlowGuardSnapshot:
    """Result of strictly read-only Flow ledger cross-checking."""

    blocked: bool
    reason: str | None = None
    cooldown_seconds: float | None = None
    flow_in_flight: bool = False
    block_kind: str | None = None
    hard_halt: bool = False


def _extract_flow_waf_event_ts(
    data: dict[str, Any],
    *,
    now: float,
    window_s: float,
) -> tuple[float, str] | None:
    """Find the most recent Flow WAF (rc=10) event across ALL profiles within ``window_s``."""
    latest_ts: float | None = None
    latest_profile = "unknown"

    def _consider(raw_ts: Any, profile_name: str) -> None:
        nonlocal latest_ts, latest_profile
        if isinstance(raw_ts, bool) or not isinstance(raw_ts, (int, float)):
            return
        ts = float(raw_ts)
        if ts <= 0.0:
            return
        age = now - ts
        if 0.0 <= age < window_s and (latest_ts is None or ts > latest_ts):
            latest_ts = ts
            latest_profile = profile_name or "unknown"

    for key in ("recent_waf_events", "waf_events"):
        events = data.get(key)
        if not isinstance(events, list):
            continue
        for item in cast("list[Any]", events):
            if isinstance(item, dict):
                ev = cast("dict[str, Any]", item)
                ev_code = ev.get("exit_code")
                if ev_code is not None and ev_code != 10:
                    continue
                ev_ts = ev.get("ts") if ev.get("ts") is not None else ev.get("timestamp")
                ev_prof = str(ev.get("profile") or "unknown")
                _consider(ev_ts, ev_prof)
            else:
                _consider(item, "unknown")

    profiles_map = data.get("profiles")
    if isinstance(profiles_map, dict):
        for pname, raw_entry in cast("dict[str, Any]", profiles_map).items():
            if not isinstance(raw_entry, dict):
                continue
            p_data = cast("dict[str, Any]", raw_entry)
            if p_data.get("last_waf_at") is not None:
                _consider(p_data.get("last_waf_at"), str(pname))
            last_exit = p_data.get("last_exit_code")
            last_outcome = str(p_data.get("last_outcome") or "").lower()
            if last_exit == 10 or "waf" in last_outcome:
                _consider(p_data.get("last_outcome_ts"), str(pname))

    if latest_ts is None:
        return None
    return latest_ts, latest_profile


def inspect_flow_ledger_guard(
    flow_ledger_dir: Path | None,
    *,
    linked_flow_profile: str | None = None,
    now: float | None = None,
    shared_ip_window_s: float = SHARED_IP_WINDOW_S,
) -> FlowGuardSnapshot:
    """Inspect Flow's ledger strictly read-only to prevent same-account / shared-IP collisions.

    Checks:
    1. ``<flow_ledger_dir>/state.json`` ``global_halted == True`` -> hard halt (shared IP hot).
    2. ANY Flow profile's recent ``rc=10`` WAF event within ``shared_ip_window_s`` (1800s / 30m)
       -> recoverable yield (``block_kind="flow_shared_ip_waf_window"``, ``hard_halt=False``,
       exact remaining ``cooldown_seconds``), enforced even when ``linked_flow_profile`` is None.
    3. ``linked_flow_profile`` quarantined in Flow ledger -> block (never checks other accounts).
    4. ``<flow_ledger_dir>/profile_<linked>.lock`` held -> reports ``flow_in_flight=True``
       (never false-blocks when only non-linked profiles like ``acc2``/``acc3`` are running).
    """
    if flow_ledger_dir is None or not flow_ledger_dir.exists():
        return FlowGuardSnapshot(blocked=False)

    current = time.time() if now is None else now
    state_path = flow_ledger_dir / "state.json"
    if state_path.is_file():
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = None
        if isinstance(raw, dict):
            data = cast("dict[str, Any]", raw)
            if bool(data.get("global_halted")):
                halt_reason = str(data.get("global_halt_reason") or "flow_global_halted")
                return FlowGuardSnapshot(
                    blocked=True,
                    reason=f"Flow ledger is globally halted ({halt_reason}); shared IP is hot.",
                    block_kind="flow_global_halted",
                    hard_halt=True,
                )
            waf_hit = _extract_flow_waf_event_ts(data, now=current, window_s=shared_ip_window_s)
            if waf_hit is not None:
                waf_ts, waf_prof = waf_hit
                remaining = max(1.0, (waf_ts + shared_ip_window_s) - current)
                return FlowGuardSnapshot(
                    blocked=True,
                    reason=(
                        f"Flow shared-IP 30m WAF silence window active "
                        f"(profile='{waf_prof}', {remaining:.0f}s remaining); yielding Gemini."
                    ),
                    cooldown_seconds=remaining,
                    block_kind="flow_shared_ip_waf_window",
                    hard_halt=False,
                )
            if linked_flow_profile:
                profiles_map = data.get("profiles")
                if isinstance(profiles_map, dict):
                    p_entry = cast("dict[str, Any]", profiles_map).get(linked_flow_profile)
                    if isinstance(p_entry, dict):
                        p_data = cast("dict[str, Any]", p_entry)
                        if bool(p_data.get("quarantined_manual")):
                            return FlowGuardSnapshot(
                                blocked=True,
                                reason=(
                                    f"Linked Flow profile '{linked_flow_profile}' requires manual "
                                    "auth check; refusing Gemini launch on same account."
                                ),
                                block_kind="flow_profile_quarantined_manual",
                                hard_halt=True,
                            )
                        q_until = float(p_data.get("quarantined_until") or 0.0)
                        if q_until > current:
                            remaining = q_until - current
                            return FlowGuardSnapshot(
                                blocked=True,
                                reason=(
                                    f"Linked Flow profile '{linked_flow_profile}' is quarantined "
                                    f"for {remaining:.0f}s; refusing Gemini launch on same account."
                                ),
                                cooldown_seconds=remaining,
                                block_kind="flow_profile_quarantined",
                                hard_halt=False,
                            )

    if linked_flow_profile:
        lock_file = flow_ledger_dir / f"profile_{linked_flow_profile}.lock"
        if _is_file_flock_held(lock_file):
            return FlowGuardSnapshot(
                blocked=True,
                reason=(
                    f"Linked Flow profile '{linked_flow_profile}' currently has an active "
                    "in-flight operation; refusing concurrent Gemini launch on same account."
                ),
                cooldown_seconds=45.0,
                flow_in_flight=True,
                block_kind="flow_in_flight",
                hard_halt=False,
            )

    return FlowGuardSnapshot(blocked=False)


def _default_state() -> dict[str, Any]:
    return {
        "version": 1,
        "global_halted": False,
        "global_halt_reason": None,
        "global_not_before": 0.0,
        "last_global_dispatch": 0.0,
        "recent_waf_events": [],
        "profiles": {},
    }


def _default_profile_entry() -> dict[str, Any]:
    return {
        "quarantined_manual": False,
        "quarantined_until": 0.0,
        "quarantine_reason": None,
        "elevated_until": 0.0,
        "rate_limit_until": 0.0,
        "lock_backoff_until": 0.0,
        "next_pacing_at": 0.0,
        "last_dispatch": 0.0,
        "consecutive_429": 0,
        "last_429_at": 0.0,
        "image_timestamps": [],
        "last_exit_code": 0,
    }


@dataclass(slots=True)
class PreflightDecision:
    """Outcome of a pre-launch anti-risk check."""

    allowed: bool
    pacing_wait_seconds: float = 0.0
    warnings: tuple[str, ...] = ()


class GeminiRiskLedger:
    """Persistent per-home Gemini anti-risk ledger stored at ``<home>/gemini_ledger/state.json``."""

    def __init__(self, *, home: Path | None = None) -> None:
        self._home = resolve_gemini_home(home)
        self.ledger_dir = self._home / "gemini_ledger"
        self.state_path = self.ledger_dir / "state.json"
        self.lock_path = self.ledger_dir / ".state.lock"

    @contextlib.contextmanager
    def _locked_state(self) -> Any:
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(self.lock_path, flags, 0o600)
        try:
            if sys.platform == "win32":
                os.lseek(fd, 0, os.SEEK_SET)
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                    os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                if sys.platform == "win32":
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read_state_unlocked(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return _default_state()
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return _default_state()
        if not isinstance(raw, dict):
            return _default_state()
        state = _default_state()
        state.update(cast("dict[str, Any]", raw))
        if not isinstance(state.get("profiles"), dict):
            state["profiles"] = {}
        if not isinstance(state.get("recent_waf_events"), list):
            state["recent_waf_events"] = []
        return state

    def _write_state_unlocked(self, state: dict[str, Any]) -> None:
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_name(f".state.{os.getpid()}.tmp")
        try:
            tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp_path, self.state_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _ensure_profile(state: dict[str, Any], profile: str) -> dict[str, Any]:
        profiles = cast("dict[str, dict[str, Any]]", state["profiles"])
        entry = profiles.get(profile)
        if not isinstance(entry, dict):
            entry = _default_profile_entry()
            profiles[profile] = entry
        else:
            for key, default_val in _default_profile_entry().items():
                entry.setdefault(key, default_val if not isinstance(default_val, list) else [])
        return entry

    def check_preflight(
        self,
        profile: str,
        *,
        kind: str = "chat",
        flow_ledger_dir: Path | str | None = None,
        linked_flow_profile: str | None = None,
        allow_flow_overlap: bool = False,
        now: float | None = None,
        cwd: Path | None = None,
    ) -> PreflightDecision:
        """Validate kill switches, quarantine, rate limits, burst cap, and Flow ledger.

        Raises typed ``GeminiError`` immediately on hard stops (kill switch, quarantine,
        rate limit cooldown, burst ceiling, or Flow collision). Returns a
        :class:`PreflightDecision` containing any required inter-request pacing sleep.
        """
        validated = validate_gemini_profile_name(profile)
        current = time.time() if now is None else now
        resolved_flow_dir = resolve_flow_ledger_dir(flow_ledger_dir)
        resolved_linked = resolve_linked_flow_profile(linked_flow_profile)

        kill_switch = find_active_kill_switch(
            home=self._home,
            flow_ledger_dir=resolved_flow_dir,
            cwd=cwd,
        )
        if kill_switch is not None:
            raise GeminiRiskBlockedError(
                f"Kill switch file active at {kill_switch}; halting before browser launch.",
                hard_halt=True,
                risk_reason="kill_switch",
            )

        if resolved_flow_dir is not None:
            # Always check Flow's global_halted (shared IP); only enforce linked-profile
            # lock/quarantine when allow_flow_overlap is False.
            flow_guard = inspect_flow_ledger_guard(
                resolved_flow_dir,
                linked_flow_profile=None if allow_flow_overlap else resolved_linked,
                now=current,
            )
            if flow_guard.blocked:
                raise GeminiRiskBlockedError(
                    flow_guard.reason or "Blocked by active Flow anti-risk guard.",
                    cooldown_seconds=flow_guard.cooldown_seconds,
                    retry_after_seconds=flow_guard.cooldown_seconds,
                    hard_halt=flow_guard.hard_halt,
                    risk_reason=flow_guard.block_kind or "flow_guard",
                )

        with self._locked_state():
            state = self._read_state_unlocked()
            if bool(state.get("global_halted")):
                reason = str(state.get("global_halt_reason") or "shared_ip_waf_cascade")
                raise GeminiRiskBlockedError(
                    f"Gemini ledger is globally halted ({reason}). "
                    "Investigate WAF/challenge events and reset ledger only after cooling down.",
                    hard_halt=True,
                    risk_reason="global_halted",
                )

            entry = self._ensure_profile(state, validated)
            if bool(entry.get("quarantined_manual")):
                q_reason = str(entry.get("quarantine_reason") or "auth_manual_check_required")
                raise GeminiAuthRequiredError(
                    f"Gemini profile '{validated}' is manually quarantined ({q_reason}). "
                    f"Run `gflow gemini login --profile {validated}` to recover.",
                    hard_halt=True,
                    risk_reason=q_reason,
                )

            q_until = float(entry.get("quarantined_until") or 0.0)
            if q_until > current:
                remaining = q_until - current
                q_reason = str(entry.get("quarantine_reason") or "waf_or_rate_limit")
                raise GeminiRiskBlockedError(
                    f"Gemini profile '{validated}' is quarantined for {remaining:.0f}s "
                    f"(reason={q_reason}).",
                    cooldown_seconds=remaining,
                    retry_after_seconds=remaining,
                    hard_halt=False,
                    risk_reason=f"profile_quarantined:{q_reason}",
                )

            rl_until = float(entry.get("rate_limit_until") or 0.0)
            if rl_until > current:
                remaining = rl_until - current
                raise GeminiRateLimitError(
                    f"Gemini profile '{validated}' is in 429 rate-limit cooldown for "
                    f"{remaining:.0f}s.",
                    cooldown_seconds=remaining,
                    retry_after_seconds=remaining,
                    risk_reason="rate_limit_cooldown",
                )

            # Rolling 10-minute burst cap for image generation.
            if kind == "image":
                cap = max_image_per_burst()
                if cap > 0:
                    recent = [
                        float(ts)
                        for ts in cast("list[Any]", entry.get("image_timestamps", []))
                        if current - float(ts) < BURST_WINDOW_SECONDS
                    ]
                    if len(recent) >= cap:
                        oldest = min(recent)
                        wait_s = max(1.0, (oldest + BURST_WINDOW_SECONDS) - current)
                        raise GeminiRateLimitError(
                            f"Gemini profile '{validated}' hit image burst cap "
                            f"({len(recent)}/{cap} in {int(BURST_WINDOW_SECONDS)}s); "
                            f"wait {wait_s:.0f}s.",
                            cooldown_seconds=wait_s,
                            retry_after_seconds=wait_s,
                            risk_reason="burst_cap",
                        )

            pacing_targets = [
                float(state.get("global_not_before") or 0.0),
                float(entry.get("next_pacing_at") or 0.0),
                float(entry.get("lock_backoff_until") or 0.0),
            ]
            wait_until = max(pacing_targets)
            pacing_wait = max(0.0, wait_until - current)

        return PreflightDecision(allowed=True, pacing_wait_seconds=pacing_wait)

    async def await_preflight(
        self,
        profile: str,
        *,
        kind: str = "chat",
        flow_ledger_dir: Path | str | None = None,
        linked_flow_profile: str | None = None,
        allow_flow_overlap: bool = False,
        respect_pacing: bool = True,
    ) -> PreflightDecision:
        """Run preflight, honor inter-request pacing sleep, and re-check kill switches."""
        decision = self.check_preflight(
            profile,
            kind=kind,
            flow_ledger_dir=flow_ledger_dir,
            linked_flow_profile=linked_flow_profile,
            allow_flow_overlap=allow_flow_overlap,
        )
        if respect_pacing and decision.pacing_wait_seconds > 0:
            log.info(
                "gemini.risk_guard.pacing_sleep",
                profile=profile,
                kind=kind,
                sleep_seconds=round(decision.pacing_wait_seconds, 2),
            )
            await asyncio.sleep(decision.pacing_wait_seconds)
            # Re-verify kill switches after sleeping so STOP_FLOW/STOP_GEMINI always wins.
            resolved_flow_dir = resolve_flow_ledger_dir(flow_ledger_dir)
            kill_switch = find_active_kill_switch(
                home=self._home,
                flow_ledger_dir=resolved_flow_dir,
            )
            if kill_switch is not None:
                raise GeminiRiskBlockedError(
                    f"Kill switch file appeared at {kill_switch} during pacing sleep; halting.",
                    hard_halt=True,
                    risk_reason="kill_switch",
                )
        return decision

    def record_dispatch(
        self,
        profile: str,
        *,
        kind: str = "chat",
        now: float | None = None,
    ) -> None:
        """Record a prompt/image dispatch and schedule randomized next-request pacing."""
        validated = validate_gemini_profile_name(profile)
        current = time.time() if now is None else now

        with self._locked_state():
            state = self._read_state_unlocked()
            entry = self._ensure_profile(state, validated)

            global_gap = random.uniform(*GLOBAL_GAP_RANGE)  # noqa: S311
            elevated_until = float(entry.get("elevated_until") or 0.0)
            if current < elevated_until:
                profile_gap = random.uniform(*ELEVATED_PROFILE_GAP_RANGE)  # noqa: S311
            elif kind == "image":
                profile_gap = random.uniform(*IMAGE_PROFILE_GAP_RANGE)  # noqa: S311
            else:
                profile_gap = random.uniform(*CHAT_PROFILE_GAP_RANGE)  # noqa: S311

            state["last_global_dispatch"] = current
            state["global_not_before"] = max(
                float(state.get("global_not_before") or 0.0),
                current + global_gap,
            )
            entry["last_dispatch"] = current
            entry["next_pacing_at"] = max(
                float(entry.get("next_pacing_at") or 0.0),
                current + profile_gap,
            )

            if kind == "image":
                recent = [
                    float(ts)
                    for ts in cast("list[Any]", entry.get("image_timestamps", []))
                    if current - float(ts) < BURST_WINDOW_SECONDS
                ]
                recent.append(current)
                entry["image_timestamps"] = recent

            self._write_state_unlocked(state)

    def record_outcome(
        self,
        profile: str,
        *,
        exit_code: int,
        reason: str = "",
        retry_after: float | None = None,
        now: float | None = None,
    ) -> float | None:
        """Update backoff/quarantine state according to the exact anti-risk matrix.

        Returns the resulting numeric cooldown seconds when applicable (``exit_code``
        in ``{4, 10, 11}``), or ``None`` otherwise.
        """
        validated = validate_gemini_profile_name(profile)
        current = time.time() if now is None else now
        applied_cooldown: float | None = None

        with self._locked_state():
            state = self._read_state_unlocked()
            entry = self._ensure_profile(state, validated)
            entry["last_exit_code"] = int(exit_code)

            if exit_code == 0:
                entry["consecutive_429"] = 0
                entry["last_429_at"] = 0.0
            elif exit_code == 10:
                # CAPTCHA / Challenge / WAF / 403 -> 45m quarantine + 2h elevated + global pause.
                q_until = current + WAF_QUARANTINE_S
                applied_cooldown = WAF_QUARANTINE_S
                prev_q = float(entry.get("quarantined_until") or 0.0)
                entry["quarantined_until"] = max(prev_q, q_until)
                entry["quarantine_reason"] = reason or "challenge_or_waf"
                entry["elevated_until"] = max(
                    float(entry.get("elevated_until") or 0.0),
                    q_until + WAF_ELEVATED_DURATION_S,
                )
                global_pause = random.uniform(*WAF_GLOBAL_PAUSE_RANGE)  # noqa: S311
                state["global_not_before"] = max(
                    float(state.get("global_not_before") or 0.0),
                    current + global_pause,
                )
                waf_events = [
                    float(ts)
                    for ts in cast("list[Any]", state.get("recent_waf_events", []))
                    if current - float(ts) <= SHARED_IP_WINDOW_S
                ]
                waf_events.append(current)
                state["recent_waf_events"] = waf_events
                if len(waf_events) >= SHARED_IP_WAF_THRESHOLD:
                    state["global_halted"] = True
                    state["global_halt_reason"] = "shared_ip_waf_cascade"
            elif exit_code == 3:
                # Auth expired / sign-in redirect -> manual quarantine until fresh login.
                entry["quarantined_manual"] = True
                entry["quarantine_reason"] = reason or "auth_manual_check_required"
            elif exit_code == 4:
                # 429 / Quota / Limit -> 120s -> 300s -> 900s + 3rd triggers 45m quarantine.
                last_429 = float(entry.get("last_429_at") or 0.0)
                if current - last_429 <= RATE_LIMIT_WINDOW_S:
                    consec = int(entry.get("consecutive_429") or 0) + 1
                else:
                    consec = 1
                entry["consecutive_429"] = consec
                entry["last_429_at"] = current
                step_idx = min(consec - 1, len(RATE_LIMIT_BACKOFF_STEPS) - 1)
                cooldown = max(RATE_LIMIT_BACKOFF_STEPS[step_idx], float(retry_after or 0.0))
                applied_cooldown = cooldown
                entry["rate_limit_until"] = max(
                    float(entry.get("rate_limit_until") or 0.0),
                    current + cooldown,
                )
                if consec >= 3:
                    entry["quarantined_until"] = max(
                        float(entry.get("quarantined_until") or 0.0),
                        current + WAF_QUARANTINE_S,
                    )
                    entry["quarantine_reason"] = reason or "repeated_rate_limit_429"
                    applied_cooldown = max(cooldown, WAF_QUARANTINE_S)
            elif exit_code == 11:
                # Profile locked -> short 20s backoff, no quarantine.
                applied_cooldown = PROFILE_LOCK_BACKOFF_S
                entry["lock_backoff_until"] = max(
                    float(entry.get("lock_backoff_until") or 0.0),
                    current + PROFILE_LOCK_BACKOFF_S,
                )
            # exit_code == 26 (no image / safety refusal) -> no quarantine, no retry.

            self._write_state_unlocked(state)
        return applied_cooldown

    def clear_manual_quarantine(self, profile: str, *, now: float | None = None) -> None:
        """Clear manual auth quarantine after a successful interactive ``login_profile``."""
        validated = validate_gemini_profile_name(profile)
        current = time.time() if now is None else now
        with self._locked_state():
            state = self._read_state_unlocked()
            entry = self._ensure_profile(state, validated)
            entry["quarantined_manual"] = False
            timed_active = float(entry.get("quarantined_until") or 0.0) > current
            if not timed_active or entry.get("quarantine_reason") in (
                "auth_manual_check_required",
                "AUTH_REQUIRED",
            ):
                entry["quarantine_reason"] = None
            self._write_state_unlocked(state)

    def get_profile_risk_snapshot(
        self,
        profile: str,
        *,
        now: float | None = None,
        flow_ledger_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Return a non-mutating risk & cooldown snapshot for ``gflow gemini status``."""
        validated = validate_gemini_profile_name(profile)
        current = time.time() if now is None else now
        resolved_flow_dir = resolve_flow_ledger_dir(flow_ledger_dir)
        kill_switch = find_active_kill_switch(home=self._home, flow_ledger_dir=resolved_flow_dir)

        state = self._read_state_unlocked()
        profiles_map = cast("dict[str, dict[str, Any]]", state.get("profiles", {}))
        entry = profiles_map.get(validated, _default_profile_entry())

        q_until = float(entry.get("quarantined_until") or 0.0)
        elevated_until = float(entry.get("elevated_until") or 0.0)
        rl_until = float(entry.get("rate_limit_until") or 0.0)
        recent_images = [
            float(ts)
            for ts in cast("list[Any]", entry.get("image_timestamps", []))
            if current - float(ts) < BURST_WINDOW_SECONDS
        ]
        return {
            "ledger_path": str(self.state_path),
            "global_halted": bool(state.get("global_halted")),
            "global_halt_reason": state.get("global_halt_reason"),
            "kill_switch_active": kill_switch is not None,
            "kill_switch_path": str(kill_switch) if kill_switch else None,
            "quarantined_manual": bool(entry.get("quarantined_manual")),
            "quarantined": bool(entry.get("quarantined_manual")) or q_until > current,
            "quarantine_remaining_seconds": round(max(0.0, q_until - current), 1),
            "quarantine_reason": entry.get("quarantine_reason"),
            "elevated": elevated_until > current,
            "elevated_remaining_seconds": round(max(0.0, elevated_until - current), 1),
            "rate_limit_remaining_seconds": round(max(0.0, rl_until - current), 1),
            "burst_count_10m": len(recent_images),
            "burst_limit_10m": max_image_per_burst(),
        }
