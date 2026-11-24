"""CLI tests for `gflow gemini` (login, status, chat, image)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from gflow_cli.cli import main
from gflow_cli.gemini.models import (
    GeminiAuthRequiredError,
    GeminiChallengeDetectedError,
    GeminiChatResult,
    GeminiDownloadError,
    GeminiImageBatchItem,
    GeminiImageBatchItemOutcome,
    GeminiImageBatchResult,
    GeminiImageMetadata,
    GeminiImageResult,
    GeminiLoginResult,
    GeminiLoginTimeoutError,
    GeminiNoImageGeneratedError,
    GeminiProfileLockedError,
    GeminiProfileStatus,
    GeminiRateLimitError,
    GeminiRiskBlockedError,
    GeminiTimeoutError,
    GeminiUiDriftError,
)


def _sample_chat_result(
    *,
    profile: str = "gemini_ultra",
    text: str = '{"scenes": [1, 2]}',
    extracted_json: Any = None,
    attachments: list[str] | None = None,
    out_file: str | None = None,
    chat_url: str = "https://gemini.google.com/app/abc123",
    model_requested: str | None = None,
    warnings: list[str] | None = None,
) -> GeminiChatResult:
    return GeminiChatResult(
        ok=True,
        profile=profile,
        chat_url=chat_url,
        model_requested=model_requested,
        model_selected=model_requested,
        text=text,
        extracted_json=extracted_json,
        attachments=attachments or [],
        elapsed_seconds=1.25,
        out_file=out_file,
        warnings=warnings or [],
    )


def _sample_image_result(
    out_file: Path,
    *,
    profile: str = "gemini_ultra",
    ref_paths: list[str] | None = None,
    aspect: str = "16:9",
    strip_watermark: bool = False,
    model: str | None = None,
) -> GeminiImageResult:
    meta = GeminiImageMetadata(
        path=str(out_file),
        width=1344,
        height=768,
        aspect_ratio="16:9",
        bytes=42_000,
        sha256="deadbeef" * 8,
        mime_type="image/png",
        download_method="download_button",
        watermark_stripped=strip_watermark,
        watermark_note="Cropped bottom-right watermark" if strip_watermark else None,
    )
    return GeminiImageResult(
        ok=True,
        profile=profile,
        chat_url="https://gemini.google.com/app/img999",
        model_requested=model,
        model_selected=model,
        aspect=aspect,
        ref_paths=ref_paths or [],
        out_file=str(out_file),
        images=[meta],
        width=meta.width,
        height=meta.height,
        aspect_ratio=meta.aspect_ratio,
        bytes=meta.bytes,
        sha256=meta.sha256,
        download_method=meta.download_method,
        watermark_stripped=meta.watermark_stripped,
        watermark_note=meta.watermark_note,
        response_text="Generated image",
        elapsed_seconds=8.5,
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Help tree & registration
# ---------------------------------------------------------------------------


def test_gemini_help_json_exposes_all_subcommands_with_spends_credits_false() -> None:
    result = CliRunner().invoke(main, ["gemini", "--help", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    paths = {" ".join(cmd["path"]): cmd["spends_credits"] for cmd in payload["commands"]}
    assert paths == {
        "gemini login": False,
        "gemini status": False,
        "gemini chat": False,
        "gemini image": False,
        "gemini image-batch": False,
        "gemini task-submit": False,
        "gemini task-harvest": False,
    }


# ---------------------------------------------------------------------------
# CLI argument validation
# ---------------------------------------------------------------------------


def test_chat_requires_prompt_or_prompt_file() -> None:
    res = CliRunner().invoke(main, ["gemini", "chat"])
    assert res.exit_code == 2
    assert "PROMPT" in res.output or "--prompt-file" in res.output


def test_chat_rejects_both_prompt_and_prompt_file(tmp_path: Path) -> None:
    pf = tmp_path / "p.txt"
    pf.write_text("hello", encoding="utf-8")
    res = CliRunner().invoke(main, ["gemini", "chat", "inline", "--prompt-file", str(pf)])
    assert res.exit_code == 2
    assert "not both" in res.output


def test_chat_rejects_empty_prompt_file(tmp_path: Path) -> None:
    pf = tmp_path / "empty.txt"
    pf.write_text("   \n\t ", encoding="utf-8")
    res = CliRunner().invoke(main, ["gemini", "chat", "--prompt-file", str(pf)])
    assert res.exit_code == 2
    assert "empty" in res.output


def test_chat_rejects_chat_url_with_new_chat() -> None:
    res = CliRunner().invoke(
        main,
        [
            "gemini",
            "chat",
            "hello",
            "--chat-url",
            "https://gemini.google.com/app/1",
            "--new-chat",
        ],
    )
    assert res.exit_code == 2
    assert "--chat-url" in res.output and "--new-chat" in res.output


def test_chat_rejects_non_positive_timeout() -> None:
    res = CliRunner().invoke(main, ["gemini", "chat", "hello", "--timeout", "0"])
    assert res.exit_code == 2
    assert "positive integer" in res.output


def test_image_rejects_invalid_aspect() -> None:
    res = CliRunner().invoke(main, ["gemini", "image", "draw a bear", "--aspect", "wide"])
    assert res.exit_code == 2
    assert "aspect" in res.output.lower()


def test_image_rejects_missing_ref_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"
    res = CliRunner().invoke(main, ["gemini", "image", "draw a bear", "--ref", str(missing)])
    assert res.exit_code == 2


def test_status_rejects_path_traversal_profile() -> None:
    res = CliRunner().invoke(main, ["gemini", "status", "--profile", "../profile_default"])
    assert res.exit_code == 2


# ---------------------------------------------------------------------------
# login & status subcommands
# ---------------------------------------------------------------------------


def test_login_json_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_res = GeminiLoginResult(
        ok=True,
        profile="gemini_ultra",
        profile_dir=str(tmp_path / "gemini_profile_gemini_ultra"),
        logged_in=True,
        chat_url="https://gemini.google.com/app",
        account_hint="ultra@example.com",
        elapsed_seconds=4.2,
        detail="Signed in",
        warnings=[],
    )
    mock_login = AsyncMock(return_value=fake_res)
    monkeypatch.setattr("gflow_cli.gemini.login_profile", mock_login)

    res = CliRunner().invoke(
        main, ["gemini", "login", "--profile", "gemini_ultra", "--timeout", "120", "--json"]
    )
    assert res.exit_code == 0, res.output
    doc = json.loads(res.stdout)
    assert doc["status"] == "ok"
    assert doc["ok"] is True
    assert doc["command"] == "gemini login"
    assert doc["logged_in"] is True
    assert doc["account_hint"] == "ultra@example.com"
    mock_login.assert_awaited_once_with(profile_name="gemini_ultra", timeout_seconds=120)


def test_status_json_offline_and_verify_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdir = tmp_path / "gemini_profile_gemini_ultra"
    fake_status = GeminiProfileStatus(
        ok=True,
        profile="gemini_ultra",
        profile_dir=str(pdir),
        exists=True,
        locked=False,
        lock_path=str(tmp_path / "gemini_locks" / "gemini_ultra.lock"),
        lock_owner_pid=None,
        chrome_available=True,
        chrome_binary="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        logged_in=True,
        verified_live=True,
        account_hint="ultra@example.com",
        detail="Live verified",
        warnings=[],
        risk={
            "kill_switch_active": False,
            "quarantined": False,
            "burst_count_10m": 2,
            "burst_limit_10m": 8,
        },
    )
    mock_status = AsyncMock(return_value=fake_status)
    monkeypatch.setattr("gflow_cli.gemini.check_profile_status", mock_status)

    res = CliRunner().invoke(
        main, ["gemini", "status", "--profile", "gemini_ultra", "--verify-live", "--json"]
    )
    assert res.exit_code == 0, res.output
    doc = json.loads(res.stdout)
    assert doc["status"] == "ok"
    assert doc["command"] == "gemini status"
    assert doc["verified_live"] is True
    assert doc["logged_in"] is True
    assert doc["risk"]["burst_count_10m"] == 2
    mock_status.assert_awaited_once_with(
        profile_name="gemini_ultra", verify_live=True, headless=True
    )


def test_status_not_ready_exits_1_with_json_and_plain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_status = GeminiProfileStatus(
        ok=False,
        profile="gemini_ultra",
        profile_dir=str(tmp_path / "gemini_profile_gemini_ultra"),
        exists=False,
        locked=False,
        lock_path=str(tmp_path / "gemini_locks" / "gemini_ultra.lock"),
        lock_owner_pid=None,
        chrome_available=True,
        chrome_binary=None,
        logged_in=None,
        verified_live=False,
        detail="Profile does not exist yet.",
        warnings=["Kill switch active: STOP_GEMINI"],
        risk={
            "kill_switch_active": True,
            "kill_switch_path": "/tmp/STOP_GEMINI",
            "quarantined": True,
            "quarantine_reason": "challenge_or_waf",
            "elevated": True,
            "burst_count_10m": 8,
            "burst_limit_10m": 8,
        },
    )
    monkeypatch.setattr(
        "gflow_cli.gemini.check_profile_status", AsyncMock(return_value=fake_status)
    )

    res_json = CliRunner().invoke(main, ["gemini", "status", "--json"])
    assert res_json.exit_code == 1
    doc = json.loads(res_json.stdout)
    assert doc["status"] == "fail"
    assert doc["ok"] is False
    assert doc["exists"] is False

    res_plain = CliRunner().invoke(main, ["gemini", "status"])
    assert res_plain.exit_code == 1
    assert "ACTIVE" in res_plain.output
    assert "burst_10m       : 8/8" in res_plain.output


def test_status_locked_exits_11_with_json_and_plain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    locked_status = GeminiProfileStatus(
        ok=False,
        profile="gemini_ultra",
        profile_dir=str(tmp_path / "gemini_profile_gemini_ultra"),
        exists=True,
        locked=True,
        lock_path=str(tmp_path / "gemini_locks" / "gemini_ultra.lock"),
        lock_owner_pid=43210,
        chrome_available=True,
        chrome_binary="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        logged_in=None,
        verified_live=False,
        detail="Profile 'gemini_ultra' is locked by pid=43210.",
        warnings=[],
        risk={"burst_count_10m": 3, "burst_limit_10m": 8},
    )
    monkeypatch.setattr(
        "gflow_cli.gemini.check_profile_status", AsyncMock(return_value=locked_status)
    )

    res_json = CliRunner().invoke(main, ["gemini", "status", "--json"])
    assert res_json.exit_code == 11
    doc = json.loads(res_json.stdout)
    assert doc["status"] == "fail"
    assert doc["ok"] is False
    assert doc["locked"] is True
    assert doc["lock_owner_pid"] == 43210

    res_plain = CliRunner().invoke(main, ["gemini", "status"])
    assert res_plain.exit_code == 11
    assert "Status: locked." in res_plain.output
    assert "burst_10m       : 3/8" in res_plain.output


# ---------------------------------------------------------------------------
# chat subcommand
# ---------------------------------------------------------------------------


def test_chat_prompt_file_attach_extract_json_and_json_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pf = tmp_path / "script_prompt.txt"
    pf.write_text("Line 1\nLine 2: return JSON", encoding="utf-8")
    att1 = tmp_path / "playbook.md"
    att2 = tmp_path / "ref.jpg"
    att1.write_text("rules", encoding="utf-8")
    att2.write_bytes(b"\xff\xd8\xff\xd9")
    out_file = tmp_path / "script.json"

    expected_json = {"episode_id": "ep01", "shots": [{"id": "seg01"}]}
    fake_res = _sample_chat_result(
        text="```json\n" + json.dumps(expected_json) + "\n```",
        extracted_json=expected_json,
        attachments=[str(att1.resolve()), str(att2.resolve())],
        out_file=str(out_file),
        model_requested="2.5 Pro",
    )
    mock_chat = AsyncMock(return_value=fake_res)
    monkeypatch.setattr("gflow_cli.gemini.chat", mock_chat)

    res = CliRunner().invoke(
        main,
        [
            "gemini",
            "chat",
            "--prompt-file",
            str(pf),
            "--attach",
            str(att1),
            "--attach",
            str(att2),
            "--model",
            "2.5 Pro",
            "--new-chat",
            "--extract-json",
            "--out-file",
            str(out_file),
            "--profile",
            "ultra_writer",
            "--headless",
            "--timeout",
            "90",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    doc = json.loads(res.stdout)
    assert doc["status"] == "ok"
    assert doc["ok"] is True
    assert doc["command"] == "gemini chat"
    assert doc["extracted_json"] == expected_json
    assert doc["out_file"] == str(out_file)
    assert len(doc["attachments"]) == 2

    mock_chat.assert_awaited_once_with(
        "Line 1\nLine 2: return JSON",
        profile="ultra_writer",
        attachments=[att1.resolve(), att2.resolve()],
        model="2.5 Pro",
        chat_url=None,
        new_chat=True,
        extract_json=True,
        out_file=out_file,
        headless=True,
        timeout_seconds=90,
    )


def test_chat_plain_output_defaults_to_headed(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_chat = AsyncMock(return_value=_sample_chat_result(text="Hello from Gemini"))
    monkeypatch.setattr("gflow_cli.gemini.chat", mock_chat)

    res = CliRunner().invoke(main, ["gemini", "chat", "hi"])
    assert res.exit_code == 0, res.output
    assert "Hello from Gemini" in res.stdout
    assert mock_chat.await_args is not None
    assert mock_chat.await_args.kwargs["headless"] is False


# ---------------------------------------------------------------------------
# image subcommand
# ---------------------------------------------------------------------------


def test_image_multi_ref_strip_watermark_and_json_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ref1 = tmp_path / "master.jpg"
    ref2 = tmp_path / "seg01.jpg"
    ref1.write_bytes(b"\xff\xd8\xff\xd9")
    ref2.write_bytes(b"\xff\xd8\xff\xd9")
    pf = tmp_path / "img_prompt.txt"
    pf.write_text("Draw two bears cooking soup", encoding="utf-8")
    out_file = tmp_path / "seg02.png"

    fake_res = _sample_image_result(
        out_file,
        ref_paths=[str(ref1.resolve()), str(ref2.resolve())],
        aspect="16:9",
        strip_watermark=True,
        model="Imagen",
    )
    mock_img = AsyncMock(return_value=fake_res)
    monkeypatch.setattr("gflow_cli.gemini.generate_image", mock_img)

    res = CliRunner().invoke(
        main,
        [
            "gemini",
            "image",
            "--prompt-file",
            str(pf),
            "--ref",
            str(ref1),
            "--ref",
            str(ref2),
            "--out-file",
            str(out_file),
            "--aspect",
            "16:9",
            "--model",
            "Imagen",
            "--chat-url",
            "https://gemini.google.com/app/c1",
            "--strip-watermark",
            "--profile",
            "gemini_ultra",
            "--timeout",
            "200",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    doc = json.loads(res.stdout)
    assert doc["status"] == "ok"
    assert doc["ok"] is True
    assert doc["command"] == "gemini image"
    assert doc["out_file"] == str(out_file)
    assert doc["local_path"] == str(out_file)
    assert doc["width"] == 1344
    assert doc["height"] == 768
    assert doc["aspect_ratio"] == "16:9"
    assert doc["watermark_stripped"] is True
    assert len(doc["ref_paths"]) == 2
    assert len(doc["images"]) == 1

    mock_img.assert_awaited_once_with(
        "Draw two bears cooking soup",
        out_file=out_file,
        profile="gemini_ultra",
        ref_paths=[ref1.resolve(), ref2.resolve()],
        aspect="16:9",
        model="Imagen",
        chat_url="https://gemini.google.com/app/c1",
        strip_watermark=True,
        headless=False,
        timeout_seconds=200,
    )


def test_image_auto_assigns_out_file_when_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_gen(prompt: str, **kwargs: Any) -> GeminiImageResult:
        captured["prompt"] = prompt
        captured.update(kwargs)
        return _sample_image_result(Path(kwargs["out_file"]))

    monkeypatch.setattr("gflow_cli.gemini.generate_image", _fake_gen)

    res = CliRunner().invoke(main, ["gemini", "image", "a sunny park", "--json"])
    assert res.exit_code == 0, res.output
    doc = json.loads(res.stdout)
    assert doc["ok"] is True
    assert "gemini/images/" in doc["out_file"].replace("\\", "/")
    assert doc["out_file"].endswith(".png")
    assert captured["strip_watermark"] is False


# ---------------------------------------------------------------------------
# Typed error exit codes & --json error envelopes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc_factory", "expected_exit", "expected_code"),
    [
        (lambda: GeminiAuthRequiredError("sign in required"), 3, "AUTH_REQUIRED"),
        (lambda: GeminiRateLimitError("429 quota", cooldown_seconds=120.0), 4, "RATE_LIMITED"),
        (lambda: GeminiDownloadError("fetch failed"), 6, "DOWNLOAD_FAILED"),
        (lambda: GeminiTimeoutError("timed out"), 9, "TIMEOUT"),
        (lambda: GeminiChallengeDetectedError("captcha shown"), 10, "CHALLENGE_DETECTED"),
        (lambda: GeminiRiskBlockedError("STOP_FLOW active"), 10, "RISK_QUARANTINED"),
        (lambda: GeminiProfileLockedError("locked by pid=99"), 11, "PROFILE_LOCKED"),
        (lambda: GeminiLoginTimeoutError("login timed out"), 12, "TIMEOUT"),
        (lambda: GeminiUiDriftError("composer missing"), 23, "UI_DRIFT"),
        (lambda: GeminiNoImageGeneratedError("safety refusal"), 26, "NO_IMAGE_GENERATED"),
    ],
)
def test_typed_gemini_errors_map_exit_codes_and_json_envelope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exc_factory: Any,
    expected_exit: int,
    expected_code: str,
) -> None:
    exc = exc_factory()
    exc.screenshot_path = tmp_path / "fail.png"
    exc.dom_path = tmp_path / "fail.html"
    monkeypatch.setattr("gflow_cli.gemini.chat", AsyncMock(side_effect=exc))

    res_json = CliRunner().invoke(main, ["gemini", "chat", "hello", "--json"])
    assert res_json.exit_code == expected_exit
    doc = json.loads(res_json.stdout)
    assert doc["ok"] is False
    assert doc["status"] == "fail"
    assert doc["command"] == "gemini chat"
    assert doc["error_code"] == expected_code
    assert doc["exit_code"] == expected_exit
    assert doc["screenshot_path"] == str(tmp_path / "fail.png")
    assert doc["dom_path"] == str(tmp_path / "fail.html")
    assert doc["error"]["error_code"] == expected_code

    res_plain = CliRunner().invoke(main, ["gemini", "chat", "hello"])
    assert res_plain.exit_code == expected_exit
    assert expected_code in res_plain.output


def test_keyboard_interrupt_exits_130_with_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gflow_cli.gemini.chat", AsyncMock(side_effect=KeyboardInterrupt))
    res = CliRunner().invoke(main, ["gemini", "chat", "hello", "--json"])
    assert res.exit_code == 130
    doc = json.loads(res.stdout)
    assert doc["ok"] is False
    assert doc["error_code"] == "INTERRUPTED"
    assert doc["exit_code"] == 130


# ---------------------------------------------------------------------------
# image-batch subcommand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_gap", ["45-25", "-5", "abc", "10-20-30", ""])
def test_image_batch_rejects_invalid_gap(tmp_path: Path, bad_gap: str) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{"id": "seg01", "prompt": "bear", "out_file": str(tmp_path / "seg01.png")}]),
        encoding="utf-8",
    )
    res = CliRunner().invoke(
        main, ["gemini", "image-batch", "--manifest", str(manifest), "--gap", bad_gap]
    )
    assert res.exit_code == 2


def test_image_batch_rejects_invalid_manifest_json(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.json"
    manifest.write_text("{not valid json", encoding="utf-8")
    res = CliRunner().invoke(main, ["gemini", "image-batch", "--manifest", str(manifest)])
    assert res.exit_code == 2


def test_image_batch_cli_success_and_flag_forwarding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out1 = tmp_path / "seg01.jpg"
    out2 = tmp_path / "seg02.jpg"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "seg01",
                        "prompt": "Shot 1 prompt",
                        "out_file": str(out1),
                        "ref_paths": [str(tmp_path / "master.jpg")],
                        "aspect": "16:9",
                    },
                    {
                        "id": "seg02",
                        "prompt": "Shot 2 prompt",
                        "out_file": str(out2),
                        "ref_paths": [str(out1)],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    c1 = GeminiImageBatchItemOutcome(
        id="seg01",
        status="completed",
        out_file=str(out1),
        chat_url="https://gemini.google.com/app/batch1",
        width=2752,
        height=1536,
        aspect_ratio="16:9",
        bytes=1_900_000,
        sha256="a" * 64,
        download_method="download_button",
        watermark_stripped=False,
        elapsed_seconds=52.1,
    )
    c2 = GeminiImageBatchItemOutcome(
        id="seg02",
        status="skipped",
        out_file=str(out2),
        width=2752,
        height=1536,
        aspect_ratio="16:9",
        bytes=1_850_000,
        sha256="b" * 64,
        download_method="existing",
        watermark_stripped=False,
        elapsed_seconds=0.01,
    )
    fake_batch_res = GeminiImageBatchResult(
        ok=True,
        partial=False,
        profile="gemini_ultra",
        reuse_chat=True,
        chat_url="https://gemini.google.com/app/batch1",
        total=2,
        completed=[c1],
        skipped=[c2],
        failed=[],
        deferred=[],
        items=[c1, c2],
        elapsed_seconds=55.0,
    )
    mock_batch = AsyncMock(return_value=fake_batch_res)
    monkeypatch.setattr("gflow_cli.gemini.generate_images_batch", mock_batch)

    res = CliRunner().invoke(
        main,
        [
            "gemini",
            "image-batch",
            "--manifest",
            str(manifest),
            "--profile",
            "gemini_ultra",
            "--reuse-chat",
            "--gap",
            "30-50",
            "--resume",
            "--aspect",
            "16:9",
            "--strip-watermark",
            "--headless",
            "--timeout",
            "210",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    doc = json.loads(res.stdout)
    assert doc["status"] == "ok"
    assert doc["ok"] is True
    assert doc["command"] == "gemini image-batch"
    assert doc["reuse_chat"] is True
    assert doc["total"] == 2
    assert len(doc["completed"]) == 1
    assert len(doc["skipped"]) == 1

    assert mock_batch.await_args is not None
    passed_items = mock_batch.await_args.args[0]
    assert len(passed_items) == 2
    assert isinstance(passed_items[0], GeminiImageBatchItem)
    assert passed_items[0].id == "seg01"
    assert passed_items[1].ref_paths == [str(out1)]
    kwargs = mock_batch.await_args.kwargs
    assert kwargs["profile"] == "gemini_ultra"
    assert kwargs["reuse_chat"] is True
    assert kwargs["gap_range"] == (30.0, 50.0)
    assert kwargs["resume"] is True
    assert kwargs["aspect"] == "16:9"
    assert kwargs["strip_watermark"] is True
    assert kwargs["headless"] is True
    assert kwargs["timeout_seconds"] == 210


def test_image_batch_mid_batch_rate_limit_emits_partial_json_and_exits_4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out1 = tmp_path / "seg01.jpg"
    out2 = tmp_path / "seg02.jpg"
    out3 = tmp_path / "seg03.jpg"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"id": "seg01", "prompt": "p1", "out_file": str(out1)},
                {"id": "seg02", "prompt": "p2", "out_file": str(out2)},
                {"id": "seg03", "prompt": "p3", "out_file": str(out3)},
            ]
        ),
        encoding="utf-8",
    )

    o1 = GeminiImageBatchItemOutcome(id="seg01", status="completed", out_file=str(out1))
    o2 = GeminiImageBatchItemOutcome(
        id="seg02",
        status="failed",
        out_file=str(out2),
        error_code="RATE_LIMITED",
        exit_code=4,
        detail="429 quota",
    )
    o3 = GeminiImageBatchItemOutcome(id="seg03", status="deferred", out_file=str(out3))
    partial_res = GeminiImageBatchResult(
        ok=False,
        partial=True,
        profile="gemini_ultra",
        reuse_chat=True,
        chat_url="https://gemini.google.com/app/batch-rl",
        total=3,
        completed=[o1],
        skipped=[],
        failed=[o2],
        deferred=[o3],
        items=[o1, o2, o3],
        stop_error_code="RATE_LIMITED",
        stop_exit_code=4,
        stop_detail="429 quota",
        hard_halt=False,
        risk_reason="rate_limit_cooldown",
        retry_after_seconds=120.0,
        elapsed_seconds=80.0,
    )
    monkeypatch.setattr(
        "gflow_cli.gemini.generate_images_batch", AsyncMock(return_value=partial_res)
    )

    res_json = CliRunner().invoke(
        main, ["gemini", "image-batch", "--manifest", str(manifest), "--json"]
    )
    assert res_json.exit_code == 4
    doc = json.loads(res_json.stdout)
    assert doc["status"] == "fail"
    assert doc["ok"] is False
    assert doc["partial"] is True
    assert doc["stop_error_code"] == "RATE_LIMITED"
    assert doc["stop_exit_code"] == 4
    assert doc["hard_halt"] is False
    assert doc["retry_after_seconds"] == 120.0
    assert len(doc["completed"]) == 1
    assert len(doc["deferred"]) == 1

    res_plain = CliRunner().invoke(main, ["gemini", "image-batch", "--manifest", str(manifest)])
    assert res_plain.exit_code == 4
    assert "RATE_LIMITED" in res_plain.output
    assert "120.0s" in res_plain.output
