"""Tests for Gemini profile namespace isolation, OS locking, and offline status."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gflow_cli.gemini import (
    DEFAULT_GEMINI_PROFILE,
    GEMINI_PROFILE_PREFIX,
    GeminiConfigError,
    GeminiProfileLease,
    GeminiProfileLockedError,
    check_profile_status,
    gemini_locks_dir,
    gemini_profile_dir,
    probe_profile_lock,
    validate_gemini_profile_name,
)


def test_profile_namespace_is_strictly_isolated_from_flow_profiles(tmp_path: Path) -> None:
    home = tmp_path / "gflow_home"
    gemini_dir = gemini_profile_dir("default", home=home)
    flow_default = home / "profile_default"

    assert gemini_dir.name == f"{GEMINI_PROFILE_PREFIX}default"
    assert gemini_dir != flow_default
    assert gemini_dir.parent == home
    assert gemini_profile_dir(home=home).name == f"{GEMINI_PROFILE_PREFIX}{DEFAULT_GEMINI_PROFILE}"
    assert gemini_locks_dir(home=home) == home / "gemini_locks"


@pytest.mark.parametrize(
    "bad_name",
    ["", "   ", "../profile_default", "a/b", "a\\b", ".hidden", "-dash", "bad name", "x" * 65],
)
def test_validate_profile_name_rejects_unsafe_inputs(bad_name: str) -> None:
    with pytest.raises(GeminiConfigError):
        validate_gemini_profile_name(bad_name)


def test_profile_lease_blocks_in_process_contention_and_releases(tmp_path: Path) -> None:
    home = tmp_path / "gflow_home"
    pdir = gemini_profile_dir("gemini_ultra", home=home)

    locked_before, pid_before, _ = probe_profile_lock(pdir, home=home)
    assert locked_before is False
    assert pid_before is None

    with GeminiProfileLease(pdir, home=home):
        locked_during, pid_during, lock_path = probe_profile_lock(pdir, home=home)
        assert locked_during is True
        assert pid_during == os.getpid()
        assert lock_path.parent == home / "gemini_locks"

        with pytest.raises(GeminiProfileLockedError) as exc_info:
            with GeminiProfileLease(pdir, home=home):
                pass
        assert exc_info.value.exit_code == 11
        assert exc_info.value.code.value == "PROFILE_LOCKED"

    locked_after, pid_after, _ = probe_profile_lock(pdir, home=home)
    assert locked_after is False
    assert pid_after is None


def test_profile_lease_blocks_cross_process_contention(tmp_path: Path) -> None:
    home = tmp_path / "gflow_home"
    pdir = gemini_profile_dir("cross_proc", home=home)
    pdir.mkdir(parents=True, exist_ok=True)

    holder_code = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from gflow_cli.gemini.profiles import GeminiProfileLease\n"
        "pdir = Path(sys.argv[1])\n"
        "home = Path(sys.argv[2])\n"
        "with GeminiProfileLease(pdir, home=home):\n"
        "    print('LOCKED', flush=True)\n"
        "    sys.stdin.readline()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(pdir), str(home)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        ready = proc.stdout.readline().strip()
        assert ready == "LOCKED", proc.stderr.read() if proc.stderr else ""

        locked, owner_pid, _ = probe_profile_lock(pdir, home=home)
        assert locked is True
        assert owner_pid == proc.pid

        with pytest.raises(GeminiProfileLockedError):
            with GeminiProfileLease(pdir, home=home):
                pass
    finally:
        if proc.stdin:
            proc.stdin.write("\n")
            proc.stdin.flush()
        proc.wait(timeout=10)


@pytest.mark.asyncio
async def test_check_profile_status_offline_states(tmp_path: Path) -> None:
    home = tmp_path / "gflow_home"
    missing = await check_profile_status("gemini_ultra", verify_live=False, home=home)
    assert missing.ok is False
    assert missing.exists is False
    assert missing.locked is False
    assert missing.verified_live is False

    pdir = gemini_profile_dir("gemini_ultra", home=home)
    (pdir / "Default").mkdir(parents=True, exist_ok=True)
    (pdir / "Default" / "Cookies").write_bytes(b"sqlite-placeholder")

    ready = await check_profile_status("gemini_ultra", verify_live=False, home=home)
    assert ready.ok is True
    assert ready.exists is True
    assert ready.locked is False
    assert ready.risk is not None

    with GeminiProfileLease(pdir, home=home):
        locked_status = await check_profile_status("gemini_ultra", verify_live=False, home=home)
        assert locked_status.ok is False
        assert locked_status.locked is True
        assert locked_status.lock_owner_pid == os.getpid()

        live_while_locked = await check_profile_status("gemini_ultra", verify_live=True, home=home)
        assert live_while_locked.ok is False
        assert live_while_locked.locked is True
        assert live_while_locked.verified_live is False
