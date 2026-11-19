#!/usr/bin/env python3
"""Generate 3-frame (start 0.4s / mid / end-0.35s) visual QA boards for shot directories.

Standalone CLI helper for `skills/aigc-commercial-pipeline`. Requires only Python 3.11+
standard library plus `ffmpeg` and `ffprobe` on PATH (or `/opt/homebrew/bin`).

Usage:
    python3 skills/aigc-commercial-pipeline/scripts/qa_audit_board.py \
        --shots-dir shots --out-dir qa --pattern "S*_raw.mp4"
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_FALLBACK_DIRS = (
    Path.home() / ".local" / "bin",
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
)


def _find_bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for directory in _FALLBACK_DIRS:
        candidate = directory / name
        if candidate.is_file():
            return str(candidate)
    msg = f"Required binary {name!r} not found on PATH."
    raise RuntimeError(msg)


def _probe_video(ffprobe_bin: str, video_path: Path) -> dict[str, Any]:
    proc = subprocess.run(  # noqa: S603 - resolved local ffprobe binary, no shell
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
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = f"ffprobe failed for {video_path}: {proc.stderr.strip()}"
        raise RuntimeError(msg)
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    width = 0
    height = 0
    has_audio = False
    for stream in streams:
        if stream.get("codec_type") == "video" and not width:
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
        elif stream.get("codec_type") == "audio":
            has_audio = True
    duration = float(fmt.get("duration") or 5.0)
    size_bytes = int(fmt.get("size") or video_path.stat().st_size)
    return {
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "duration_sec": round(duration, 3),
        "has_audio": has_audio,
        "file_size_bytes": size_bytes,
    }


def _sample_times(duration_sec: float) -> tuple[float, float, float]:
    t_start = min(0.4, duration_sec * 0.2)
    t_mid = duration_sec * 0.5
    t_end = max(t_mid, duration_sec - min(0.35, duration_sec * 0.2))
    return round(t_start, 3), round(t_mid, 3), round(t_end, 3)


def _extract_strip(
    ffmpeg_bin: str,
    video_path: Path,
    out_strip: Path,
    times: tuple[float, float, float],
    tile_width: int,
) -> None:
    t0, t1, t2 = times
    filter_graph = (
        f"[0:v]scale={tile_width}:-2[v0];"
        f"[1:v]scale={tile_width}:-2[v1];"
        f"[2:v]scale={tile_width}:-2[v2];"
        "[v0][v1][v2]hstack=inputs=3[out]"
    )
    proc = subprocess.run(  # noqa: S603 - resolved local ffmpeg binary, no shell
        [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{t0:.3f}",
            "-i",
            str(video_path),
            "-ss",
            f"{t1:.3f}",
            "-i",
            str(video_path),
            "-ss",
            f"{t2:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            str(out_strip),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not out_strip.is_file():
        msg = f"ffmpeg strip failed for {video_path.name}: {proc.stderr.strip()[:240]}"
        raise RuntimeError(msg)


def _stack_master_board(ffmpeg_bin: str, strips: list[Path], out_board: Path) -> Path | None:
    if not strips:
        return None
    if len(strips) == 1:
        shutil.copy2(strips[0], out_board)
        return out_board
    inputs: list[str] = []
    for strip in strips:
        inputs.extend(["-i", str(strip)])
    chain = "".join(f"[{idx}:v]" for idx in range(len(strips)))
    filter_graph = f"{chain}vstack=inputs={len(strips)}[out]"
    proc = subprocess.run(  # noqa: S603 - resolved local ffmpeg binary, no shell
        [
            ffmpeg_bin,
            "-y",
            *inputs,
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            str(out_board),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return out_board if proc.returncode == 0 and out_board.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots-dir", type=Path, required=True, help="Directory of MP4 clips.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for strips.")
    parser.add_argument(
        "--pattern", default="S*_raw.mp4", help="Glob pattern (default: S*_raw.mp4)"
    )
    parser.add_argument("--tile-width", type=int, default=640, help="Per-frame tile width in px.")
    args = parser.parse_args(argv)

    shots_dir: Path = args.shots_dir.expanduser().resolve()
    out_dir: Path = args.out_dir.expanduser().resolve()
    if not shots_dir.is_dir():
        print(f"Error: shots directory not found: {shots_dir}", file=sys.stderr)
        return 2

    clips = sorted(shots_dir.glob(args.pattern))
    if not clips:
        print(f"No clips matched {args.pattern!r} in {shots_dir}", file=sys.stderr)
        return 1

    ffprobe_bin = _find_bin("ffprobe")
    ffmpeg_bin = _find_bin("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    strip_paths: list[Path] = []
    for clip in clips:
        meta = _probe_video(ffprobe_bin, clip)
        times = _sample_times(meta["duration_sec"])
        strip_path = out_dir / f"{clip.stem}_qa_strip.jpg"
        _extract_strip(ffmpeg_bin, clip, strip_path, times, args.tile_width)
        strip_paths.append(strip_path)
        entry = {
            "shot": clip.name,
            "video_path": str(clip),
            "strip_path": str(strip_path),
            "sample_timestamps_sec": list(times),
            **meta,
        }
        records.append(entry)
        print(
            f"[OK] {clip.name}: {meta['resolution']} {meta['duration_sec']}s "
            f"audio={meta['has_audio']} -> {strip_path.name}"
        )

    master_board = _stack_master_board(ffmpeg_bin, strip_paths, out_dir / "ALL_SHOTS_AUDIT.jpg")
    manifest_path = out_dir / "audit_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "shots_dir": str(shots_dir),
                "pattern": args.pattern,
                "count": len(records),
                "master_board": str(master_board) if master_board else None,
                "shots": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(records)} QA strips + manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
