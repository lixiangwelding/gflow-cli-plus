# SPDX-License-Identifier: MIT
"""Offline tests for local AIGC commercial workflow MCP tools (`workflow_tools.py`)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from gflow_cli.mcp import workflow_tools


def _make_sample_mp4(path: Path) -> bool:
    """Create a tiny 2-second synthetic MP4 + sine audio with ffmpeg if installed."""
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).is_file():
        return False
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x180:rate=15:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0 and path.is_file()


def test_workflow_tools_registered_and_kept_in_no_spend(
    mcp_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gflow_cli.mcp import server as server_mod

    names = set(mcp_server._tool_manager._tools.keys())
    assert "gflow_qa_video_strip" in names
    assert "gflow_synthesize_voiceover" in names

    # Non-spend local workflow tools remain available under GFLOW_MCP_NO_SPEND=1.
    saved = {n: mcp_server._tool_manager.get_tool(n) for n in server_mod._SPEND_TOOLS}
    try:
        monkeypatch.setenv("GFLOW_MCP_NO_SPEND", "1")
        server_mod._apply_no_spend_policy()
        after = set(mcp_server._tool_manager._tools.keys())
        assert "gflow_qa_video_strip" in after
        assert "gflow_synthesize_voiceover" in after
    finally:
        for name, tool in saved.items():
            if tool is not None and mcp_server._tool_manager.get_tool(name) is None:
                mcp_server._tool_manager._tools[name] = tool


@pytest.mark.asyncio
async def test_gflow_qa_video_strip_extracts_3frame_contact_strip(tmp_path: Path) -> None:
    clip = tmp_path / "S01_raw.mp4"
    if not _make_sample_mp4(clip):
        pytest.skip("ffmpeg not available")
    strip = tmp_path / "qa" / "S01_strip.jpg"
    res = await workflow_tools.gflow_qa_video_strip(
        video_path=str(clip),
        out_image_path=str(strip),
        duration_sec=2.0,
    )
    assert res["status"] == "completed"
    assert Path(res["out_image_path"]).is_file()
    assert res["resolution"] == "320x180"
    assert res["has_audio"] is True
    assert res["duration_sec"] >= 1.8
    assert len(res["sample_timestamps_sec"]) == 3


@pytest.mark.asyncio
async def test_gflow_qa_video_strip_rejects_missing_video(tmp_path: Path) -> None:
    res = await workflow_tools.gflow_qa_video_strip(
        video_path=str(tmp_path / "missing.mp4"),
        out_image_path=str(tmp_path / "out.jpg"),
    )
    assert res["status"] == "error"
    assert res["error"]["title"] == "Invalid Video Path"


@pytest.mark.asyncio
async def test_gflow_synthesize_voiceover_invokes_edge_tts_and_probes_duration(
    tmp_path: Path,
) -> None:
    out_mp3 = tmp_path / "vo" / "line1.mp3"

    async def fake_exec(argv: list[str], *, timeout_s: float = 90.0) -> tuple[int, str, str]:
        assert any(arg.startswith("--rate=-5%") for arg in argv)
        out_mp3.write_bytes(b"ID3-fake-mp3-bytes")
        return 0, "", ""

    with (
        patch.object(workflow_tools, "_find_binary", side_effect=lambda name, **_: f"/bin/{name}"),
        patch.object(workflow_tools, "_run_exec", side_effect=fake_exec),
        patch.object(
            workflow_tools,
            "_probe_media",
            AsyncMock(return_value={"format": {"duration": "3.456"}}),
        ),
    ):
        res = await workflow_tools.gflow_synthesize_voiceover(
            text="艺术光，让每一道光线严丝合缝。",
            out_mp3_path=str(out_mp3),
            voice="zh-CN-XiaoxiaoNeural",
            rate="-5%",
        )

    assert res["status"] == "completed"
    assert res["duration_sec"] == 3.456
    assert res["voice"] == "zh-CN-XiaoxiaoNeural"
    assert res["file_size_bytes"] > 0


def test_qa_audit_board_cli_generates_strips_and_manifest(tmp_path: Path) -> None:
    shots = tmp_path / "shots"
    shots.mkdir()
    if not _make_sample_mp4(shots / "S01_raw.mp4"):
        pytest.skip("ffmpeg not available")
    shutil.copy2(shots / "S01_raw.mp4", shots / "S02_raw.mp4")

    out_dir = tmp_path / "qa_out"
    script = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / "aigc-commercial-pipeline"
        / "scripts"
        / "qa_audit_board.py"
    )
    proc = subprocess.run(
        ["python3", str(script), "--shots-dir", str(shots), "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "S01_raw_qa_strip.jpg").is_file()
    assert (out_dir / "S02_raw_qa_strip.jpg").is_file()
    assert (out_dir / "ALL_SHOTS_AUDIT.jpg").is_file()
    assert (out_dir / "audit_manifest.json").is_file()
