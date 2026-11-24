"""Image validation, dimension inspection, and optional Gemini watermark cleanup."""

from __future__ import annotations

import hashlib
import io
import math
import os
import struct
from pathlib import Path
from typing import Any

from gflow_cli.gemini.models import GeminiDownloadError, GeminiImageMetadata
from gflow_cli.paths import extension_from_magic, looks_like_video

_EXT_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

_CANONICAL_RATIOS: tuple[tuple[str, float], ...] = (
    ("16:9", 16 / 9),
    ("9:16", 9 / 16),
    ("1:1", 1.0),
    ("4:3", 4 / 3),
    ("3:4", 3 / 4),
    ("3:2", 3 / 2),
    ("2:3", 2 / 3),
    ("21:9", 21 / 9),
)


def parse_aspect_ratio(aspect: str | None) -> float | None:
    """Parse an aspect label like ``'16:9'`` into a positive float ratio, or ``None``."""
    if not aspect:
        return None
    cleaned = aspect.strip().lower().replace("x", ":")
    if ":" not in cleaned:
        return None
    parts = cleaned.split(":", 1)
    try:
        w = float(parts[0].strip())
        h = float(parts[1].strip())
    except ValueError:
        return None
    if w <= 0 or h <= 0 or not (math.isfinite(w) and math.isfinite(h)):
        return None
    return w / h


def compute_aspect_ratio_label(width: int, height: int) -> str:
    """Return a clean aspect ratio label such as ``'16:9'`` or ``'1:1'``."""
    if width <= 0 or height <= 0:
        return "unknown"
    ratio = width / height
    for label, target in _CANONICAL_RATIOS:
        if abs(ratio - target) / target <= 0.03:
            return label
    divisor = math.gcd(width, height)
    rw, rh = width // divisor, height // divisor
    if rw <= 50 and rh <= 50:
        return f"{rw}:{rh}"
    return f"{ratio:.2f}:1"


def _parse_png_dims(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    return None


def _parse_gif_dims(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 10 and data[:6] in (b"GIF87a", b"GIF89a"):
        w, h = struct.unpack("<HH", data[6:10])
        return int(w), int(h)
    return None


def _parse_webp_dims(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        w = 1 + int.from_bytes(data[24:27], "little")
        h = 1 + int.from_bytes(data[27:30], "little")
        return w, h
    if chunk == b"VP8 " and len(data) >= 30:
        w, h = struct.unpack("<HH", data[26:30])
        return int(w & 0x3FFF), int(h & 0x3FFF)
    if chunk == b"VP8L" and len(data) >= 25:
        bits = struct.unpack("<I", data[21:25])[0]
        w = (bits & 0x3FFF) + 1
        h = ((bits >> 14) & 0x3FFF) + 1
        return int(w), int(h)
    return None


def _parse_jpeg_dims(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    idx = 2
    n = len(data)
    sof_markers = frozenset(
        {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }
    )
    while idx + 4 < n:
        if data[idx] != 0xFF:
            idx += 1
            continue
        while idx < n and data[idx] == 0xFF:
            idx += 1
        if idx >= n:
            break
        marker = data[idx]
        idx += 1
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if idx + 2 > n:
            break
        seg_len = struct.unpack(">H", data[idx : idx + 2])[0]
        if seg_len < 2 or idx + seg_len > n:
            break
        if marker in sof_markers and seg_len >= 7:
            h, w = struct.unpack(">HH", data[idx + 3 : idx + 7])
            return int(w), int(h)
        idx += seg_len
    return None


def inspect_image_bytes(data: bytes) -> tuple[int, int, str]:
    """Validate image bytes and return ``(width, height, mime_type)``.

    Uses Pillow when installed, with a pure-Python header parser fallback so
    validation never breaks in minimal environments.
    """
    if not data:
        raise GeminiDownloadError("Downloaded image payload is empty.")
    if looks_like_video(data[:12]):
        raise GeminiDownloadError("Downloaded payload is a video container, not an image.")
    ext = extension_from_magic(data[:12])
    if ext is None:
        snippet = data[:64].decode("utf-8", errors="replace").strip()
        raise GeminiDownloadError(
            f"Downloaded payload does not have valid image magic bytes (head={snippet[:32]!r})."
        )
    mime = _EXT_TO_MIME.get(ext, "application/octet-stream")

    try:
        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            return int(width), int(height), mime
    except ImportError:
        pass
    except Exception as exc:
        raise GeminiDownloadError(f"Failed to decode downloaded image: {exc}") from exc

    dims: tuple[int, int] | None = None
    if ext == ".png":
        dims = _parse_png_dims(data)
    elif ext == ".jpg":
        dims = _parse_jpeg_dims(data)
    elif ext == ".webp":
        dims = _parse_webp_dims(data)
    elif ext == ".gif":
        dims = _parse_gif_dims(data)

    if dims is None or dims[0] <= 0 or dims[1] <= 0:
        raise GeminiDownloadError(f"Could not determine dimensions of {ext} image payload.")
    return dims[0], dims[1], mime


def _save_pil_image_atomic(img: Any, image_path: Path) -> None:
    suffix = image_path.suffix.lower()
    tmp_path = image_path.with_name(f".{image_path.name}.wm.tmp")
    save_kwargs: dict[str, Any] = {}
    fmt: str | None = None
    if suffix in (".jpg", ".jpeg", ".jpe"):
        fmt = "JPEG"
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        save_kwargs = {"quality": 95, "subsampling": 0}
    elif suffix == ".png":
        fmt = "PNG"
    elif suffix == ".webp":
        fmt = "WEBP"
        save_kwargs = {"quality": 95}
    try:
        if fmt:
            img.save(tmp_path, format=fmt, **save_kwargs)
        else:
            img.save(tmp_path, **save_kwargs)
        os.replace(tmp_path, image_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def strip_gemini_watermark(
    image_path: Path,
    *,
    target_aspect: str | None = None,
    crop_fraction: float = 0.06,
) -> tuple[bool, str]:
    """Optionally trim the bottom-right Gemini sparkle watermark via clean top-left crop.

    Preserves the exact original aspect ratio (or ``target_aspect`` when provided) so
    downstream 16:9 video pipelines never receive a stretched or distorted frame.
    Returns ``(applied, note)``.
    """
    if not image_path.is_file():
        return False, f"Image file not found for watermark cleanup: {image_path}"
    bounded_fraction = min(max(float(crop_fraction), 0.01), 0.25)

    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return False, "Pillow not installed; watermark left untouched"

    with Image.open(image_path) as img:
        orig_w, orig_h = img.size
        max_w = max(1, int(round(orig_w * (1.0 - bounded_fraction))))
        max_h = max(1, int(round(orig_h * (1.0 - bounded_fraction))))

        target_ratio = parse_aspect_ratio(target_aspect)
        if target_ratio is not None:
            if max_w / max_h >= target_ratio:
                crop_h = max_h
                crop_w = max(1, min(max_w, int(round(crop_h * target_ratio))))
            else:
                crop_w = max_w
                crop_h = max(1, min(max_h, int(round(crop_w / target_ratio))))
        else:
            crop_w, crop_h = max_w, max_h

        if crop_w < 64 or crop_h < 64:
            return False, f"Image too small ({orig_w}x{orig_h}) for safe watermark crop"

        cropped = img.crop((0, 0, crop_w, crop_h))
        _save_pil_image_atomic(cropped, image_path)

    return True, f"Cropped bottom-right watermark ({orig_w}x{orig_h} -> {crop_w}x{crop_h})"


def finalize_image_file(
    data: bytes,
    out_path: Path,
    *,
    min_width: int = 512,
    min_height: int = 512,
    strip_watermark: bool = False,
    target_aspect: str | None = None,
    download_method: str = "download_button",
) -> GeminiImageMetadata:
    """Validate, atomically write, optionally de-watermark, and hash an image file."""
    width, height, _mime_type = inspect_image_bytes(data)
    if width < min_width or height < min_height:
        raise GeminiDownloadError(
            f"Downloaded image is too small ({width}x{height}; minimum {min_width}x{min_height}) "
            "— likely an avatar, icon, or unloaded preview thumbnail."
        )

    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.dl.tmp")
    try:
        tmp_path.write_bytes(data)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    watermark_stripped = False
    watermark_note: str | None = None
    if strip_watermark:
        watermark_stripped, watermark_note = strip_gemini_watermark(
            out_path,
            target_aspect=target_aspect,
        )

    final_bytes = out_path.read_bytes()
    final_w, final_h, final_mime = inspect_image_bytes(final_bytes)
    digest = hashlib.sha256(final_bytes).hexdigest()

    return GeminiImageMetadata(
        path=str(out_path),
        width=final_w,
        height=final_h,
        aspect_ratio=compute_aspect_ratio_label(final_w, final_h),
        bytes=len(final_bytes),
        sha256=digest,
        mime_type=final_mime,
        download_method=download_method,
        watermark_stripped=watermark_stripped,
        watermark_note=watermark_note,
    )
