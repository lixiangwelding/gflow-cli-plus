"""Unit tests for `load_image_batch_manifest` and `generate_images_batch` session batching."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from gflow_cli.gemini import (
    GeminiChallengeDetectedError,
    GeminiConfigError,
    GeminiImageBatchItem,
    GeminiImageMetadata,
    GeminiImageResult,
    GeminiProfileLockedError,
    GeminiRateLimitError,
    GeminiWebClient,
    generate_images_batch,
    load_image_batch_manifest,
)


def _png_bytes(
    width: int = 1280, height: int = 720, color: tuple[int, int, int] = (200, 180, 140)
) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _image_result(
    out_path: Path,
    *,
    chat_url: str = "https://gemini.google.com/app/session-1",
    sha_char: str = "a",
    color: tuple[int, int, int] = (200, 180, 140),
) -> GeminiImageResult:
    data = _png_bytes(color=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    meta = GeminiImageMetadata(
        path=str(out_path),
        width=1280,
        height=720,
        aspect_ratio="16:9",
        bytes=len(data),
        sha256=sha_char * 64,
        mime_type="image/png",
        download_method="download_button",
        watermark_stripped=False,
        watermark_note=None,
    )
    return GeminiImageResult(
        ok=True,
        profile="gemini_ultra",
        chat_url=chat_url,
        model_requested=None,
        model_selected=None,
        aspect="16:9",
        ref_paths=[],
        out_file=str(out_path),
        images=[meta],
        width=meta.width,
        height=meta.height,
        aspect_ratio=meta.aspect_ratio,
        bytes=meta.bytes,
        sha256=meta.sha256,
        download_method=meta.download_method,
        watermark_stripped=False,
        watermark_note=None,
        response_text="ok",
        elapsed_seconds=1.5,
        warnings=[],
    )


def test_load_image_batch_manifest_supports_list_dict_and_ref_aliases(tmp_path: Path) -> None:
    m1 = tmp_path / "list.json"
    m1.write_text(
        json.dumps(
            [
                {
                    "id": "seg01",
                    "prompt": "Prompt 1",
                    "out_file": str(tmp_path / "seg01.png"),
                    "ref": str(tmp_path / "master.jpg"),
                },
                {
                    "prompt": "Prompt 2",
                    "out_file": str(tmp_path / "seg02.png"),
                    "refs": [str(tmp_path / "seg01.png")],
                },
            ]
        ),
        encoding="utf-8",
    )
    items1 = load_image_batch_manifest(m1)
    assert len(items1) == 2
    assert items1[0].id == "seg01"
    assert items1[0].ref_paths == [str(tmp_path / "master.jpg")]
    assert items1[1].ref_paths == [str(tmp_path / "seg01.png")]

    m2 = tmp_path / "obj.json"
    m2.write_text(
        json.dumps(
            {"items": [{"id": "a", "prompt": "p", "out_file": str(tmp_path / "a.png")}]}
        ),
        encoding="utf-8",
    )
    items2 = load_image_batch_manifest(m2)
    assert [i.id for i in items2] == ["a"]

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(GeminiConfigError):
        load_image_batch_manifest(empty)


@pytest.mark.asyncio
async def test_batch_resume_all_existing_skips_chrome_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out1 = tmp_path / "seg01.png"
    out2 = tmp_path / "seg02.png"
    out1.write_bytes(_png_bytes(1280, 720, color=(10, 20, 30)))
    out2.write_bytes(_png_bytes(1280, 720, color=(40, 50, 60)))

    start_called = False

    async def _fail_start(self: Any) -> Any:
        nonlocal start_called
        start_called = True
        raise AssertionError("Chrome must not start when all items are already valid on disk")

    monkeypatch.setattr(GeminiWebClient, "start", _fail_start)

    items = [
        GeminiImageBatchItem(id="seg01", prompt="p1", out_file=str(out1)),
        GeminiImageBatchItem(id="seg02", prompt="p2", out_file=str(out2)),
    ]
    res = await generate_images_batch(
        items,
        profile="gemini_ultra",
        resume=True,
        home=tmp_path / "home",
        enforce_risk_guard=False,
    )
    assert res.ok is True
    assert res.partial is False
    assert len(res.skipped) == 2
    assert len(res.completed) == 0
    assert start_called is False


@pytest.mark.asyncio
async def test_batch_single_session_reuses_chat_and_sleeps_between_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out1 = tmp_path / "seg01.png"
    out2 = tmp_path / "seg02.png"
    master = tmp_path / "master.png"
    master.write_bytes(_png_bytes())

    lifecycle = {"closes": 0}
    client_ids: list[int] = []
    calls: list[dict[str, Any]] = []
    sleeps: list[float] = []

    async def _fake_close(self: Any) -> None:
        lifecycle["closes"] += 1
        self._page = None

    async def _fake_gen(self: Any, prompt: str, **kwargs: Any) -> GeminiImageResult:
        client_ids.append(id(self))
        calls.append({"prompt": prompt, **kwargs})
        idx = len(calls)
        return _image_result(
            Path(kwargs["out_file"]),
            chat_url="https://gemini.google.com/app/shared-ep01",
            sha_char="a" if idx == 1 else "b",
            color=(10 * idx, 20 * idx, 30 * idx),
        )

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(GeminiWebClient, "close", _fake_close)
    monkeypatch.setattr(GeminiWebClient, "generate_image", _fake_gen)
    monkeypatch.setattr("gflow_cli.gemini.client.asyncio.sleep", _fake_sleep)

    items = [
        GeminiImageBatchItem(
            id="seg01", prompt="Shot 1", out_file=str(out1), ref_paths=[str(master)]
        ),
        GeminiImageBatchItem(
            id="seg02", prompt="Shot 2", out_file=str(out2), ref_paths=[str(out1)]
        ),
    ]
    res = await generate_images_batch(
        items,
        profile="gemini_ultra",
        reuse_chat=True,
        gap_range=(25.0, 45.0),
        resume=True,
        home=tmp_path / "home",
        enforce_risk_guard=False,
    )
    assert res.ok is True
    assert res.chat_url == "https://gemini.google.com/app/shared-ep01"
    assert len(res.completed) == 2
    assert len(set(client_ids)) == 1
    assert lifecycle == {"closes": 1}
    assert calls[0]["reuse_current_page"] is False
    assert calls[1]["reuse_current_page"] is True
    assert len(sleeps) == 1
    assert 25.0 <= sleeps[0] <= 45.0


@pytest.mark.asyncio
async def test_batch_mid_run_rate_limit_and_challenge_stop_cleanly_and_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out1 = tmp_path / "seg01.png"
    out2 = tmp_path / "seg02.png"
    out3 = tmp_path / "seg03.png"
    closes = 0

    async def _fake_start(self: Any) -> Any:
        self._page = object()
        return self._page

    async def _fake_close(self: Any) -> None:
        nonlocal closes
        closes += 1
        self._page = None

    call_count = 0

    async def _gen_then_429(self: Any, _prompt: str, **kwargs: Any) -> GeminiImageResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _image_result(
                Path(kwargs["out_file"]),
                chat_url="https://gemini.google.com/app/ep-rl",
                sha_char="c",
            )
        raise GeminiRateLimitError("Too many requests", retry_after_seconds=300.0)

    async def _fast_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(GeminiWebClient, "start", _fake_start)
    monkeypatch.setattr(GeminiWebClient, "close", _fake_close)
    monkeypatch.setattr(GeminiWebClient, "generate_image", _gen_then_429)
    monkeypatch.setattr("gflow_cli.gemini.client.asyncio.sleep", _fast_sleep)

    items = [
        GeminiImageBatchItem(id="seg01", prompt="p1", out_file=str(out1)),
        GeminiImageBatchItem(id="seg02", prompt="p2", out_file=str(out2)),
        GeminiImageBatchItem(id="seg03", prompt="p3", out_file=str(out3)),
    ]
    res = await generate_images_batch(
        items,
        profile="gemini_ultra",
        reuse_chat=True,
        gap_range=(0.0, 0.0),
        home=tmp_path / "home",
        enforce_risk_guard=False,
    )
    assert res.ok is False
    assert res.partial is True
    assert call_count == 2
    assert [o.id for o in res.completed] == ["seg01"]
    assert [o.id for o in res.failed] == ["seg02"]
    assert [o.id for o in res.deferred] == ["seg03"]
    assert res.stop_error_code == "RATE_LIMITED"
    assert res.stop_exit_code == 4
    assert res.hard_halt is False
    assert res.retry_after_seconds == 300.0
    assert res.chat_url == "https://gemini.google.com/app/ep-rl"
    assert closes == 1

    # Now verify hard_halt=True on CHALLENGE_DETECTED (exit 10)
    async def _gen_challenge(self: Any, _prompt: str, **_kwargs: Any) -> GeminiImageResult:
        raise GeminiChallengeDetectedError("CAPTCHA shown")

    monkeypatch.setattr(GeminiWebClient, "generate_image", _gen_challenge)
    res_chal = await generate_images_batch(
        [GeminiImageBatchItem(id="seg02", prompt="p2", out_file=str(out2))],
        profile="gemini_ultra",
        home=tmp_path / "home",
        enforce_risk_guard=False,
    )
    assert res_chal.ok is False
    assert res_chal.stop_error_code == "CHALLENGE_DETECTED"
    assert res_chal.stop_exit_code == 10
    assert res_chal.hard_halt is True
    assert res_chal.retry_after_seconds is None
    assert closes == 2


@pytest.mark.asyncio
async def test_batch_profile_locked_stops_immediately_with_exit_11_and_20s_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0
    closes = 0

    async def _fake_gen(self: Any, _prompt: str, **_kwargs: Any) -> GeminiImageResult:
        nonlocal calls
        calls += 1
        raise GeminiProfileLockedError("Profile gemini_ultra locked by pid=999")

    async def _fake_close(self: Any) -> None:
        nonlocal closes
        closes += 1

    monkeypatch.setattr(GeminiWebClient, "generate_image", _fake_gen)
    monkeypatch.setattr(GeminiWebClient, "close", _fake_close)

    items = [
        GeminiImageBatchItem(id="seg01", prompt="p1", out_file=str(tmp_path / "seg01.png")),
        GeminiImageBatchItem(id="seg02", prompt="p2", out_file=str(tmp_path / "seg02.png")),
        GeminiImageBatchItem(id="seg03", prompt="p3", out_file=str(tmp_path / "seg03.png")),
    ]
    res = await generate_images_batch(
        items,
        profile="gemini_ultra",
        home=tmp_path / "home",
        enforce_risk_guard=False,
    )
    assert res.ok is False
    assert calls == 1
    assert [o.id for o in res.failed] == ["seg01"]
    assert [o.id for o in res.deferred] == ["seg02", "seg03"]
    assert res.stop_error_code == "PROFILE_LOCKED"
    assert res.stop_exit_code == 11
    assert res.hard_halt is False
    assert res.retry_after_seconds == 20.0
    assert closes == 1
