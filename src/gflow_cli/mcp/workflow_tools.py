# SPDX-License-Identifier: MIT
"""Local, credit-free AIGC commercial workflow MCP tools (QA strips & neural voiceover).

Exposes two local media utilities that spend zero Google Flow credits and remain
available even when ``--no-spend`` / ``GFLOW_MCP_NO_SPEND=1`` is active:

1. ``gflow_qa_video_strip`` — extracts a 3-frame (``start 0.4s / mid / end-0.35s``)
   side-by-side contact strip via ``ffprobe`` + ``ffmpeg`` and returns resolution,
   duration, audio-stream presence, and file size so MCP agents can visually catch
   motion defects (morphing, mid-air beams, foreground camera occlusion).
2. ``gflow_synthesize_voiceover`` — synthesizes neural TTS audio via ``edge-tts``
   and probes the exact output MP3 duration in seconds via ``ffprobe``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import structlog

from gflow_cli.errors import ConfigurationError
from gflow_cli.mcp.server import server
from gflow_cli.mcp.tools import (
    _bad_param,  # pyright: ignore[reportPrivateUsage]
    _guarded,  # pyright: ignore[reportPrivateUsage]
)

log = structlog.get_logger(__name__)

_FALLBACK_BIN_DIRS = (
    Path.home() / ".local" / "bin",
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
)


def _find_binary(name: str, *, install_hint: str) -> str:
    """Resolve an executable from ``PATH`` or common user/Homebrew binary directories."""
    found = shutil.which(name)
    if found:
        return found
    for directory in _FALLBACK_BIN_DIRS:
        candidate = directory / name
        if candidate.is_file():
            return str(candidate)
    raise ConfigurationError(
        detail=f"Required local binary {name!r} was not found on PATH.",
        remediation_hint=install_hint,
    )


async def _run_exec(argv: list[str], *, timeout_s: float = 90.0) -> tuple[int, str, str]:
    """Run a local media binary without a shell and capture UTF-8 stdout/stderr."""
    proc = await asyncio.create_subprocess_exec(  # noqa: S603 - fixed binary + local media args
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    return (
        proc.returncode or 0,
        out_b.decode("utf-8", errors="replace"),
        err_b.decode("utf-8", errors="replace"),
    )


async def _probe_media(ffprobe_bin: str, media_path: Path) -> dict[str, Any]:
    """Probe streams and container format metadata via ``ffprobe`` JSON output."""
    code, stdout, stderr = await _run_exec(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-show_entries",
            "stream=codec_type,width,height",
            "-of",
            "json",
            str(media_path),
        ],
        timeout_s=30.0,
    )
    if code != 0:
        raise ConfigurationError(
            detail=f"ffprobe failed on {media_path.name} (exit {code}): {stderr.strip()[:200]}",
            remediation_hint="Verify the media file is a valid MP4/MP3 container.",
        )
    parsed: dict[str, Any] = json.loads(stdout or "{}")
    return parsed


def _parse_video_probe(
    probe: dict[str, Any], *, fallback_duration: float, fallback_size: int
) -> tuple[int, int, float, bool, int]:
    """Extract ``(width, height, duration_sec, has_audio, file_size_bytes)`` from probe JSON."""
    streams: list[dict[str, Any]] = list(probe.get("streams") or [])
    fmt: dict[str, Any] = dict(probe.get("format") or {})
    width = 0
    height = 0
    has_audio = False
    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and not width:
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
        elif codec_type == "audio":
            has_audio = True
    raw_dur = fmt.get("duration")
    duration = float(raw_dur) if raw_dur is not None else 0.0
    if duration <= 0.0:
        duration = max(fallback_duration, 0.5)
    raw_size = fmt.get("size")
    size = int(raw_size) if raw_size is not None else fallback_size
    return width, height, duration, has_audio, size


def _sample_points(duration_sec: float) -> tuple[float, float, float]:
    """Compute ``(start 0.4s, mid, end-0.35s)`` clamped safely for short clips."""
    start_s = min(0.4, duration_sec * 0.2)
    mid_s = duration_sec * 0.5
    end_margin = min(0.35, duration_sec * 0.2)
    end_s = max(mid_s, duration_sec - end_margin)
    return round(start_s, 3), round(mid_s, 3), round(end_s, 3)


def _validate_target_file(
    raw_path: str, *, label: str
) -> tuple[Path | None, dict[str, Any] | None]:
    """Resolve an output file path and ensure its parent directory exists."""
    if not raw_path.strip():
        return None, _bad_param(f"Invalid {label}", f"{label} must not be empty.")
    target = Path(raw_path).expanduser().resolve()
    if target.exists() and target.is_dir():
        return None, _bad_param(f"Invalid {label}", f"{label} '{raw_path}' names a directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target, None


@server.tool(
    name="gflow_qa_video_strip",
    description=(
        "Extract a 3-point (start 0.4s / mid / end-0.35s) side-by-side QA frame strip "
        "from a local video using ffprobe + ffmpeg, returning resolution, duration, "
        "audio stream presence, file size, and the saved strip image path for visual "
        "inspection of motion defects (morphing, mid-air beams, camera occlusion). "
        "Spends zero credits."
    ),
)
@_guarded
async def gflow_qa_video_strip(
    video_path: str,
    out_image_path: str,
    duration_sec: float = 5.0,
) -> dict[str, Any]:
    """Build a 3x1 start/mid/end visual QA contact strip and probe clip metadata."""
    src = Path(video_path).expanduser().resolve()
    if not src.is_file():
        return _bad_param("Invalid Video Path", f"Video '{video_path}' does not exist.")
    dst, err = _validate_target_file(out_image_path, label="Output Image Path")
    if err is not None or dst is None:
        return err or _bad_param("Invalid Output Image Path", "Could not resolve out_image_path.")

    ffprobe_bin = _find_binary("ffprobe", install_hint="Install ffmpeg (`brew install ffmpeg`).")
    ffmpeg_bin = _find_binary("ffmpeg", install_hint="Install ffmpeg (`brew install ffmpeg`).")

    probe = await _probe_media(ffprobe_bin, src)
    width, height, actual_dur, has_audio, size_bytes = _parse_video_probe(
        probe, fallback_duration=duration_sec, fallback_size=src.stat().st_size
    )
    t_start, t_mid, t_end = _sample_points(actual_dur)

    filter_graph = (
        "[0:v]scale=640:-2[v0];"
        "[1:v]scale=640:-2[v1];"
        "[2:v]scale=640:-2[v2];"
        "[v0][v1][v2]hstack=inputs=3[out]"
    )
    code, _, stderr = await _run_exec(
        [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{t_start:.3f}",
            "-i",
            str(src),
            "-ss",
            f"{t_mid:.3f}",
            "-i",
            str(src),
            "-ss",
            f"{t_end:.3f}",
            "-i",
            str(src),
            "-frames:v",
            "1",
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            str(dst),
        ],
        timeout_s=60.0,
    )
    if code != 0 or not dst.is_file():
        raise ConfigurationError(
            detail=f"ffmpeg QA strip extraction failed (exit {code}): {stderr.strip()[:220]}",
            remediation_hint=(
                "Ensure the video has a decodable video stream and valid output suffix."
            ),
        )

    log.info(
        "mcp.workflow.qa_video_strip",
        video=src.name,
        strip=str(dst),
        resolution=f"{width}x{height}",
        duration_sec=round(actual_dur, 3),
        has_audio=has_audio,
    )
    return {
        "status": "completed",
        "video_path": str(src),
        "out_image_path": str(dst),
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "duration_sec": round(actual_dur, 3),
        "sample_timestamps_sec": [t_start, t_mid, t_end],
        "has_audio": has_audio,
        "file_size_bytes": size_bytes,
    }


@server.tool(
    name="gflow_synthesize_voiceover",
    description=(
        "Synthesize neural voiceover MP3 locally via edge-tts and return its exact "
        "ffprobe duration in seconds for frame-accurate AIGC edit timing. "
        "Spends zero credits."
    ),
)
@_guarded
async def gflow_synthesize_voiceover(
    text: str,
    out_mp3_path: str,
    voice: str = "zh-CN-XiaoxiaoNeural",
    rate: str = "+0%",
) -> dict[str, Any]:
    """Generate an MP3 voiceover via ``edge-tts`` and measure exact duration with ``ffprobe``."""
    cleaned = text.strip()
    if not cleaned:
        return _bad_param("Empty Voiceover Text", "Parameter 'text' must not be empty.")
    dst, err = _validate_target_file(out_mp3_path, label="Output MP3 Path")
    if err is not None or dst is None:
        return err or _bad_param("Invalid Output MP3 Path", "Could not resolve out_mp3_path.")

    edge_tts_bin = _find_binary(
        "edge-tts",
        install_hint="Install edge-tts (`uv tool install edge-tts` or `pip install edge-tts`).",
    )
    ffprobe_bin = _find_binary("ffprobe", install_hint="Install ffmpeg (`brew install ffmpeg`).")

    code, _, stderr = await _run_exec(
        [
            edge_tts_bin,
            f"--voice={voice}",
            f"--rate={rate}",
            f"--text={cleaned}",
            f"--write-media={dst}",
        ],
        timeout_s=90.0,
    )
    if code != 0 or not dst.is_file():
        raise ConfigurationError(
            detail=f"edge-tts synthesis failed (exit {code}): {stderr.strip()[:220]}",
            remediation_hint="Check network reachability and verify the voice/rate format.",
        )

    probe = await _probe_media(ffprobe_bin, dst)
    fmt: dict[str, Any] = dict(probe.get("format") or {})
    raw_dur: Any = fmt.get("duration")
    duration_sec = round(float(raw_dur), 3) if raw_dur is not None else 0.0
    size_bytes = dst.stat().st_size
    log.info(
        "mcp.workflow.voiceover_synthesized",
        out_mp3_path=str(dst),
        voice=voice,
        rate=rate,
        duration_sec=duration_sec,
        bytes=size_bytes,
    )
    return {
        "status": "completed",
        "out_mp3_path": str(dst),
        "voice": voice,
        "rate": rate,
        "duration_sec": duration_sec,
        "file_size_bytes": size_bytes,
    }
