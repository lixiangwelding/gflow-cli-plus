"""Tests for Gemini persistent anti-risk ledger, backoff/quarantine matrix, and Flow guard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gflow_cli.gemini import (
    BURST_WINDOW_SECONDS,
    ELEVATED_PROFILE_GAP_RANGE,
    PROFILE_LOCK_BACKOFF_S,
    RATE_LIMIT_BACKOFF_STEPS,
    WAF_ELEVATED_DURATION_S,
    WAF_QUARANTINE_S,
    GeminiAuthRequiredError,
    GeminiRateLimitError,
    GeminiRiskBlockedError,
    GeminiRiskLedger,
    inspect_flow_ledger_guard,
    resolve_linked_flow_profile,
)


def test_waf_exit_10_quarantine_elevated_and_shared_ip_cascade(tmp_path: Path) -> None:
    ledger = GeminiRiskLedger(home=tmp_path / "home")
    t0 = 1_700_000_000.0

    # 1st WAF -> 45m quarantine + 2h elevated after quarantine
    ledger.record_outcome("gemini_ultra", exit_code=10, reason="CHALLENGE_DETECTED", now=t0)
    snap = ledger.get_profile_risk_snapshot("gemini_ultra", now=t0 + 10.0)
    assert snap["quarantined"] is True
    assert snap["elevated"] is True
    assert snap["global_halted"] is False

    with pytest.raises(GeminiRiskBlockedError) as exc_info:
        ledger.check_preflight("gemini_ultra", kind="image", now=t0 + 60.0)
    assert exc_info.value.exit_code == 10
    assert exc_info.value.cooldown_seconds is not None
    assert exc_info.value.cooldown_seconds >= WAF_QUARANTINE_S - 65.0

    # After 45m quarantine expires, profile is still in 2h elevated window (45-90s gap)
    t_after_q = t0 + WAF_QUARANTINE_S + 5.0
    decision = ledger.check_preflight("gemini_ultra", kind="image", now=t_after_q)
    assert decision.allowed is True
    ledger.record_dispatch("gemini_ultra", kind="image", now=t_after_q)
    next_decision = ledger.check_preflight("gemini_ultra", kind="image", now=t_after_q)
    assert (
        ELEVATED_PROFILE_GAP_RANGE[0]
        <= next_decision.pacing_wait_seconds
        <= ELEVATED_PROFILE_GAP_RANGE[1]
    )
    snap_after = ledger.get_profile_risk_snapshot("gemini_ultra", now=t_after_q)
    assert snap_after["elevated"] is True
    assert snap_after["elevated_remaining_seconds"] <= WAF_ELEVATED_DURATION_S

    # 2nd WAF within 30m -> global_halted (shared_ip_waf_cascade)
    ledger.record_outcome("acc2", exit_code=10, reason="CHALLENGE_DETECTED", now=t0 + 600.0)
    with pytest.raises(GeminiRiskBlockedError, match="globally halted"):
        ledger.check_preflight("acc3", kind="chat", now=t0 + 610.0)


def test_auth_exit_3_sets_manual_quarantine_cleared_by_login(tmp_path: Path) -> None:
    ledger = GeminiRiskLedger(home=tmp_path / "home")
    t0 = 1_700_000_000.0

    ledger.record_outcome("gemini_ultra", exit_code=3, reason="AUTH_REQUIRED", now=t0)
    with pytest.raises(GeminiAuthRequiredError) as exc_info:
        ledger.check_preflight("gemini_ultra", kind="chat", now=t0 + 5.0)
    assert exc_info.value.exit_code == 3

    ledger.clear_manual_quarantine("gemini_ultra", now=t0 + 10.0)
    assert ledger.check_preflight("gemini_ultra", kind="chat", now=t0 + 10.0).allowed is True


def test_rate_limit_exit_4_escalates_and_quarantines_on_third(tmp_path: Path) -> None:
    ledger = GeminiRiskLedger(home=tmp_path / "home")
    t0 = 1_700_000_000.0

    # 1st 429 -> 120s
    ledger.record_outcome("gemini_ultra", exit_code=4, reason="RATE_LIMITED", now=t0)
    with pytest.raises(GeminiRateLimitError) as e1:
        ledger.check_preflight("gemini_ultra", kind="image", now=t0 + 1.0)
    assert e1.value.cooldown_seconds is not None
    assert abs(e1.value.cooldown_seconds - (RATE_LIMIT_BACKOFF_STEPS[0] - 1.0)) < 1.0

    # 2nd 429 -> 300s
    ledger.record_outcome("gemini_ultra", exit_code=4, reason="RATE_LIMITED", now=t0 + 200.0)
    with pytest.raises(GeminiRateLimitError) as e2:
        ledger.check_preflight("gemini_ultra", kind="image", now=t0 + 201.0)
    assert e2.value.cooldown_seconds is not None
    assert abs(e2.value.cooldown_seconds - (RATE_LIMIT_BACKOFF_STEPS[1] - 1.0)) < 1.0

    # 3rd 429 within 30m -> 45m quarantine (GeminiRiskBlockedError exit 10)
    ledger.record_outcome("gemini_ultra", exit_code=4, reason="RATE_LIMITED", now=t0 + 600.0)
    with pytest.raises(GeminiRiskBlockedError) as e3:
        ledger.check_preflight("gemini_ultra", kind="image", now=t0 + 601.0)
    assert e3.value.exit_code == 10
    assert e3.value.cooldown_seconds is not None
    assert e3.value.cooldown_seconds >= WAF_QUARANTINE_S - 5.0


def test_profile_lock_exit_11_and_no_image_exit_26_do_not_quarantine(tmp_path: Path) -> None:
    ledger = GeminiRiskLedger(home=tmp_path / "home")
    t0 = 1_700_000_000.0

    ledger.record_outcome("gemini_ultra", exit_code=11, reason="PROFILE_LOCKED", now=t0)
    decision = ledger.check_preflight("gemini_ultra", kind="chat", now=t0 + 2.0)
    assert decision.allowed is True
    assert abs(decision.pacing_wait_seconds - (PROFILE_LOCK_BACKOFF_S - 2.0)) < 1.0

    ledger.record_outcome("gemini_ultra", exit_code=26, reason="NO_IMAGE_GENERATED", now=t0 + 30.0)
    snap = ledger.get_profile_risk_snapshot("gemini_ultra", now=t0 + 31.0)
    assert snap["quarantined"] is False


def test_rolling_10m_image_burst_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GFLOW_GEMINI_MAX_PER_BURST", "8")
    ledger = GeminiRiskLedger(home=tmp_path / "home")
    t0 = 1_700_000_000.0

    for i in range(8):
        ledger.record_dispatch("gemini_ultra", kind="image", now=t0 + i * 40.0)

    with pytest.raises(GeminiRateLimitError, match="burst cap") as exc_info:
        ledger.check_preflight("gemini_ultra", kind="image", now=t0 + 320.0)
    assert exc_info.value.exit_code == 4

    # After rolling window expires past first dispatch, allowed again
    decision = ledger.check_preflight(
        "gemini_ultra", kind="image", now=t0 + BURST_WINDOW_SECONDS + 10.0
    )
    assert decision.allowed is True


@pytest.mark.parametrize("filename", ["STOP_GEMINI", "STOP_FLOW"])
def test_kill_switch_files_halt_preflight_immediately(tmp_path: Path, filename: str) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / filename).write_text("halt", encoding="utf-8")

    ledger = GeminiRiskLedger(home=home)
    with pytest.raises(GeminiRiskBlockedError, match="Kill switch"):
        ledger.check_preflight("gemini_ultra", kind="chat", cwd=tmp_path)


def test_readonly_flow_ledger_cross_guard_blocks_halt_quarantine_and_flock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flow_dir = tmp_path / "flow-ledger"
    flow_dir.mkdir(parents=True, exist_ok=True)
    t0 = 1_700_000_000.0

    # 1. global_halted in Flow state.json
    (flow_dir / "state.json").write_text(
        json.dumps({"global_halted": True, "global_halt_reason": "waf_cascade"}),
        encoding="utf-8",
    )
    snap_halt = inspect_flow_ledger_guard(flow_dir, linked_flow_profile="default", now=t0)
    assert snap_halt.blocked is True
    assert "globally halted" in (snap_halt.reason or "")

    # 2. linked_flow_profile quarantined in Flow state.json
    (flow_dir / "state.json").write_text(
        json.dumps(
            {
                "global_halted": False,
                "profiles": {"default": {"quarantined_until": t0 + 1800.0}},
            }
        ),
        encoding="utf-8",
    )
    snap_q = inspect_flow_ledger_guard(flow_dir, linked_flow_profile="default", now=t0)
    assert snap_q.blocked is True
    assert snap_q.cooldown_seconds == pytest.approx(1800.0)

    # 3. Active profile_default.lock held by another process (read-only flock probe)
    (flow_dir / "state.json").write_text(
        json.dumps({"global_halted": False, "profiles": {"default": {}}}),
        encoding="utf-8",
    )
    lock_file = flow_dir / "profile_default.lock"
    lock_file.write_text("", encoding="utf-8")

    holder_code = (
        "import fcntl, sys\n"
        "f = open(sys.argv[1], 'r+')\n"
        "fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "print('HELD', flush=True)\n"
        "sys.stdin.readline()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(lock_file)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "HELD"
        snap_lock = inspect_flow_ledger_guard(flow_dir, linked_flow_profile="default", now=t0)
        assert snap_lock.blocked is True
        assert snap_lock.flow_in_flight is True

        # Ledger preflight enforces Flow cross-guard via env vars (both env aliases supported)
        monkeypatch.setenv("GFLOW_FLOW_LEDGER_DIR", str(flow_dir))
        monkeypatch.delenv("GFLOW_GEMINI_LINKED_FLOW_PROFILE", raising=False)
        monkeypatch.setenv("GFLOW_LINKED_FLOW_PROFILE", "default")
        assert resolve_linked_flow_profile() == "default"

        ledger = GeminiRiskLedger(home=tmp_path / "home")
        with pytest.raises(GeminiRiskBlockedError, match="in-flight") as in_flight_exc:
            ledger.check_preflight("gemini_ultra", kind="image", now=t0)
        in_flight_doc = in_flight_exc.value.to_error_dict()
        assert in_flight_doc["hard_halt"] is False
        assert isinstance(in_flight_doc["retry_after_seconds"], float)

        # Non-linked account (acc2) does NOT false-block when linked_flow_profile='default'
        acc2_lock = flow_dir / "profile_acc2.lock"
        acc2_lock.write_text("", encoding="utf-8")
        snap_acc2 = inspect_flow_ledger_guard(flow_dir, linked_flow_profile="acc2", now=t0)
        assert snap_acc2.blocked is False
    finally:
        if proc.stdin:
            proc.stdin.write("\n")
            proc.stdin.flush()
        proc.wait(timeout=10)


def test_allow_flow_overlap_still_hard_halts_on_flow_global_halted(tmp_path: Path) -> None:
    flow_dir = tmp_path / "flow-ledger"
    flow_dir.mkdir(parents=True, exist_ok=True)
    (flow_dir / "state.json").write_text(
        json.dumps({"global_halted": True, "global_halt_reason": "shared_ip_waf"}),
        encoding="utf-8",
    )
    ledger = GeminiRiskLedger(home=tmp_path / "home")
    with pytest.raises(GeminiRiskBlockedError, match="globally halted") as exc_info:
        ledger.check_preflight(
            "gemini_ultra",
            kind="image",
            flow_ledger_dir=flow_dir,
            linked_flow_profile=None,
            allow_flow_overlap=True,
        )
    err_doc = exc_info.value.to_error_dict()
    assert err_doc["hard_halt"] is True
    assert "retry_after_seconds" not in err_doc


def test_error_retry_after_and_hard_halt_contract() -> None:
    from gflow_cli.gemini import GeminiChallengeDetectedError

    rl_doc = GeminiRateLimitError("429").to_error_dict()
    assert rl_doc["hard_halt"] is False
    assert isinstance(rl_doc["retry_after_seconds"], float)
    assert rl_doc["retry_after_seconds"] >= 120.0

    chal_doc = GeminiChallengeDetectedError("captcha").to_error_dict()
    assert chal_doc["hard_halt"] is True
    assert "retry_after_seconds" not in chal_doc

    auth_doc = GeminiAuthRequiredError("expired").to_error_dict()
    assert auth_doc["hard_halt"] is True
    assert "retry_after_seconds" not in auth_doc


def test_flow_shared_ip_30m_waf_window_blocks_all_accounts_and_clears_after_1800s(
    tmp_path: Path,
) -> None:
    flow_dir = tmp_path / "flow-ledger"
    flow_dir.mkdir(parents=True, exist_ok=True)
    t0 = 1_700_000_000.0

    # WAF event recorded on acc2 (different from linked_flow_profile="default")
    (flow_dir / "state.json").write_text(
        json.dumps(
            {
                "global_halted": False,
                "recent_waf_events": [
                    {
                        "profile": "acc2",
                        "ts": t0,
                        "exit_code": 10,
                        "reason": "waf_unusual_activity",
                    }
                ],
                "profiles": {
                    "default": {"last_exit_code": 0, "last_outcome_ts": t0 - 100.0},
                    "acc2": {
                        "last_exit_code": 10,
                        "last_outcome": "profile_quarantined_waf",
                        "last_outcome_ts": t0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    ledger = GeminiRiskLedger(home=tmp_path / "home")

    # Even with linked_flow_profile="default" AND allow_flow_overlap=True, must yield for 30m window
    with pytest.raises(GeminiRiskBlockedError, match="30m WAF silence window") as exc_info:
        ledger.check_preflight(
            "gemini_ultra",
            kind="image",
            flow_ledger_dir=flow_dir,
            linked_flow_profile="default",
            allow_flow_overlap=True,
            now=t0 + 300.0,
        )
    err_doc = exc_info.value.to_error_dict()
    assert err_doc["exit_code"] == 10
    assert err_doc["hard_halt"] is False
    assert err_doc["risk_reason"] == "flow_shared_ip_waf_window"
    assert err_doc["retry_after_seconds"] == pytest.approx(1500.0, abs=1.0)

    # Direct inspect_flow_ledger_guard agrees
    snap = inspect_flow_ledger_guard(flow_dir, linked_flow_profile="default", now=t0 + 300.0)
    assert snap.blocked is True
    assert snap.block_kind == "flow_shared_ip_waf_window"
    assert snap.hard_halt is False
    assert snap.cooldown_seconds == pytest.approx(1500.0, abs=1.0)

    # Clears cleanly once > 1800s old
    decision = ledger.check_preflight(
        "gemini_ultra",
        kind="image",
        flow_ledger_dir=flow_dir,
        linked_flow_profile="default",
        allow_flow_overlap=True,
        now=t0 + 1805.0,
    )
    assert decision.allowed is True

    # Profile-level last_exit_code=10 + last_outcome_ts fallback (without recent_waf_events list)
    (flow_dir / "state.json").write_text(
        json.dumps(
            {
                "global_halted": False,
                "recent_waf_events": [],
                "profiles": {
                    "acc3": {
                        "last_exit_code": 10,
                        "last_outcome": "profile_quarantined_waf",
                        "last_outcome_ts": t0 + 2000.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(GeminiRiskBlockedError) as exc_fallback:
        ledger.check_preflight(
            "gemini_ultra",
            kind="chat",
            flow_ledger_dir=flow_dir,
            linked_flow_profile="default",
            allow_flow_overlap=True,
            now=t0 + 2600.0,
        )
    fb_doc = exc_fallback.value.to_error_dict()
    assert fb_doc["risk_reason"] == "flow_shared_ip_waf_window"
    assert fb_doc["hard_halt"] is False
    assert fb_doc["retry_after_seconds"] == pytest.approx(1200.0, abs=1.0)
