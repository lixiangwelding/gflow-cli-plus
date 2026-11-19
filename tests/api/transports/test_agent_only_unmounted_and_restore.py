"""Pure-offline unit tests for unmounted `<video>` recovery and hardened `restore_defaults`."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from structlog.testing import capture_logs

from gflow_cli.api import video as video_api
from gflow_cli.api.transports import agent_only_composer as aoc
from gflow_cli.config import reset_settings

CLIP_UUID = "11111111-2222-4333-8444-555555555555"


@pytest.mark.asyncio
async def test_unmounted_polls_returns_empty_src_after_two_listed_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composer = aoc.AgentOnlyComposer()
    monkeypatch.setattr(composer, "_hover_video_src", AsyncMock(return_value=""))
    monkeypatch.setattr(aoc, "_head_carries", AsyncMock(return_value=True))

    page = MagicMock()
    head = {"video": "", "img": "https://flow.google.com/asb/tok-new=s512"}
    state = {"pending": 0, "finished": [head], "options": []}
    run = aoc._Run("video", 60.0, time.monotonic() + 60.0, 0, 1, time.time())  # pyright: ignore[reportPrivateUsage]
    run.saw_pending = True
    run.listed = ("tok-new", CLIP_UUID)

    first = await composer._ready_video(page, state, [], None, run, None)  # pyright: ignore[reportPrivateUsage]
    assert first is None
    assert run.unmounted_polls == 1

    with capture_logs() as logs:
        second = await composer._ready_video(page, state, [], None, run, None)  # pyright: ignore[reportPrivateUsage]
    assert second == aoc.TileMedia(CLIP_UUID, "")
    assert run.unmounted_polls == 2
    assert any(
        e["event"] == "migrated.agent_only.video_ready_unmounted_player"
        and e["media_id"] == CLIP_UUID
        for e in logs
    )


@pytest.mark.asyncio
async def test_run_agent_video_exports_720p_when_media_src_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GFLOW_CLI_AGENT_CONFIRM", "account")
    reset_settings()

    fake_mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32
    saved_calls: list[tuple[str, bytes]] = []

    async def save(media_id: str, body: bytes) -> Path:
        out = tmp_path / f"{media_id}.mp4"
        out.write_bytes(body)
        saved_calls.append((media_id, body))
        return out

    upscale_mock = AsyncMock(return_value=fake_mp4)
    monkeypatch.setattr(
        "gflow_cli.api.transports.migrated_video_upscale.upscale_video_migrated",
        upscale_mock,
    )
    monkeypatch.setattr(
        aoc.AgentOnlyComposer,
        "apply_defaults",
        AsyncMock(return_value=aoc.DefaultsSnapshot("video", 0, 0, "Veo 3.1 - Fast")),
    )
    monkeypatch.setattr(
        aoc.AgentOnlyComposer,
        "generate",
        AsyncMock(return_value=[aoc.TileMedia(CLIP_UUID, "")]),
    )
    monkeypatch.setattr(aoc.AgentOnlyComposer, "restore_defaults", AsyncMock())

    page = MagicMock()
    result = await aoc.run_agent_video(
        page,
        video_api.GenerateVideoRequest(prompt="test shot"),
        project_id="proj-123",
        save=save,
        poll_timeout_s=30.0,
        download=True,
        on_started=None,
    )

    upscale_mock.assert_awaited_once_with(
        page,
        project_id="proj-123",
        media_id=CLIP_UUID,
        scale="720p",
    )
    assert saved_calls == [(CLIP_UUID, fake_mp4)]
    assert result.local_path == tmp_path / f"{CLIP_UUID}.mp4"
    assert result.status.media_id == CLIP_UUID


@pytest.mark.asyncio
async def test_restore_defaults_presses_escape_before_opening_pane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    composer = aoc.AgentOnlyComposer()

    async def press(key: str) -> None:
        order.append(f"press:{key}")

    async def open_pane(_page: Any) -> None:
        order.append("open_pane")

    monkeypatch.setattr(composer, "_open_pane", open_pane)
    monkeypatch.setattr(composer, "_check_radio", AsyncMock())
    monkeypatch.setattr(composer, "_choose_model", AsyncMock())
    monkeypatch.setattr(composer, "_save", AsyncMock())

    page = MagicMock()
    page.keyboard.press = press

    await composer.restore_defaults(
        page, aoc.DefaultsSnapshot("video", 0, 0, "Veo 3.1 - Fast")
    )
    assert order == ["press:Escape", "open_pane"]
