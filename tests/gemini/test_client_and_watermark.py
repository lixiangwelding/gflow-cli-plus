"""Tests for JSON extraction, watermark cleanup, URL upgrade, and mocked Playwright flows."""

from __future__ import annotations

import contextlib
import io
import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from gflow_cli.gemini import (
    GeminiAuthRequiredError,
    GeminiChallengeDetectedError,
    GeminiDownloadError,
    GeminiNoImageGeneratedError,
    GeminiRateLimitError,
    GeminiWebClient,
    classify_page_risk,
    detect_gemini_authenticated,
    extract_json_from_text,
    finalize_image_file,
    has_google_auth_cookies,
    inject_aspect_hint,
    strip_gemini_watermark,
    upgrade_googleusercontent_url,
)


def _make_png_bytes(width: int = 1280, height: int = 720) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(240, 200, 160))
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# JSON extraction, URL upgrade & aspect hint helpers
# ---------------------------------------------------------------------------


def test_extract_json_from_fenced_and_raw_text() -> None:
    fenced = 'Here is the plan:\n```json\n{"episode": "ep01", "shots": [1, 2]}\n```\nDone.'
    assert extract_json_from_text(fenced) == {"episode": "ep01", "shots": [1, 2]}

    bare = 'Result: [{"id": "seg01"}, {"id": "seg02"}] thanks!'
    assert extract_json_from_text(bare) == [{"id": "seg01"}, {"id": "seg02"}]

    assert extract_json_from_text("no json here at all") is None


def test_upgrade_googleusercontent_url_and_aspect_hint() -> None:
    u1 = "https://lh3.googleusercontent.com/gg/abc123=s512-rj"
    assert upgrade_googleusercontent_url(u1) == "https://lh3.googleusercontent.com/gg/abc123=s0"

    u2 = "https://lh3.googleusercontent.com/gg/abc123=w1024-h576?authuser=0"
    assert (
        upgrade_googleusercontent_url(u2)
        == "https://lh3.googleusercontent.com/gg/abc123=s0?authuser=0"
    )

    u3 = "https://example.com/photo=s512"
    assert upgrade_googleusercontent_url(u3) == u3

    assert "16:9" in inject_aspect_hint("Two cute bears", "16:9")
    already = "Two cute bears in strict 16:9 widescreen"
    assert inject_aspect_hint(already, "16:9") == already


# ---------------------------------------------------------------------------
# Image validation & optional watermark cleanup
# ---------------------------------------------------------------------------


def test_finalize_image_file_validates_dimensions_magic_and_watermark(tmp_path: Path) -> None:
    png_16_9 = _make_png_bytes(1280, 720)
    out_raw = tmp_path / "raw.png"
    meta_raw = finalize_image_file(png_16_9, out_raw, strip_watermark=False, target_aspect="16:9")
    assert meta_raw.width == 1280
    assert meta_raw.height == 720
    assert meta_raw.aspect_ratio == "16:9"
    assert meta_raw.watermark_stripped is False
    assert len(meta_raw.sha256) == 64

    out_clean = tmp_path / "clean.png"
    meta_clean = finalize_image_file(
        png_16_9, out_clean, strip_watermark=True, target_aspect="16:9"
    )
    assert meta_clean.watermark_stripped is True
    assert meta_clean.width < 1280 and meta_clean.height < 720
    assert meta_clean.aspect_ratio == "16:9"

    # Reject small thumbnail / avatar
    tiny = _make_png_bytes(128, 128)
    with pytest.raises(GeminiDownloadError, match="too small"):
        finalize_image_file(tiny, tmp_path / "tiny.png", min_width=512, min_height=512)

    # Reject non-image bytes
    with pytest.raises(GeminiDownloadError, match="magic bytes"):
        finalize_image_file(b"<html>not an image</html>", tmp_path / "bad.png")

    # Missing file watermark helper returns graceful False
    applied, note = strip_gemini_watermark(tmp_path / "does_not_exist.png")
    assert applied is False
    assert "not found" in note.lower()


# ---------------------------------------------------------------------------
# Risk signal classification
# ---------------------------------------------------------------------------


def test_classify_page_risk_detects_auth_challenge_and_rate_limit() -> None:
    auth_err = classify_page_risk(
        url="https://accounts.google.com/v3/signin",
        body_text="Sign in",
    )
    assert isinstance(auth_err, GeminiAuthRequiredError)
    assert auth_err.exit_code == 3

    chal_err = classify_page_risk(
        url="https://gemini.google.com/app",
        body_text="Please verify you are human to continue",
    )
    assert isinstance(chal_err, GeminiChallengeDetectedError)
    assert chal_err.exit_code == 10

    rl_err = classify_page_risk(
        url="https://gemini.google.com/app",
        body_text="You've reached your image generation limit. Try again later.",
    )
    assert isinstance(rl_err, GeminiRateLimitError)
    assert rl_err.exit_code == 4


# ---------------------------------------------------------------------------
# Mocked Playwright end-to-end flows on GeminiWebClient
# ---------------------------------------------------------------------------


class _FakeLocator:
    def __init__(
        self,
        *,
        visible: bool = False,
        text: str = "",
        count: int = 0,
        images: list[dict[str, Any]] | None = None,
        children: dict[str, _FakeLocator] | None = None,
        attrs: dict[str, str] | None = None,
    ) -> None:
        self._visible = visible
        self._text = text
        self._count = count
        self._images = images if images is not None else []
        self._children = children or {}
        self._attrs = attrs or {}
        self.filled_values: list[str] = []
        self.uploaded_files: list[list[str]] = []
        self.clicked: int = 0

    @property
    def first(self) -> _FakeLocator:
        return self

    def nth(self, _idx: int) -> _FakeLocator:
        return self

    def locator(self, selector: str) -> _FakeLocator:
        return self._children.get(selector, _FakeLocator(visible=False, count=0))

    async def count(self) -> int:
        return self._count

    async def is_visible(self, **_kwargs: Any) -> bool:
        return self._visible

    async def is_enabled(self, **_kwargs: Any) -> bool:
        return self._visible

    async def click(self, **_kwargs: Any) -> None:
        self.clicked += 1

    async def fill(self, value: str, **_kwargs: Any) -> None:
        self.filled_values.append(value)
        self._text = value

    async def inner_text(self, **_kwargs: Any) -> str:
        return self._text

    async def get_attribute(self, name: str, **_kwargs: Any) -> str | None:
        return self._attrs.get(name)

    async def set_input_files(self, files: list[str], **_kwargs: Any) -> None:
        self.uploaded_files.append(list(files))

    async def scroll_into_view_if_needed(self, **_kwargs: Any) -> None:
        return None

    async def hover(self, **_kwargs: Any) -> None:
        return None

    async def evaluate(self, _script: str, *_args: Any) -> Any:
        return list(self._images)


class _FakeDownload:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def save_as(self, path: str) -> None:
        Path(path).write_bytes(self._payload)


class _FakeDownloadInfo:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    @property
    async def value(self) -> _FakeDownload:
        return _FakeDownload(self._payload)


class _FakeRequestResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    async def body(self) -> bytes:
        return self._body


class _FakeRequestContext:
    def __init__(self, responses: dict[str, tuple[int, bytes]]) -> None:
        self._responses = responses
        self.requested_urls: list[str] = []

    async def get(self, url: str, **_kwargs: Any) -> _FakeRequestResponse:
        self.requested_urls.append(url)
        status, body = self._responses.get(url, (404, b""))
        return _FakeRequestResponse(status, body)


class _FakeKeyboard:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def insert_text(self, text: str) -> None:
        self.events.append(("insert_text", text))

    async def press(self, key: str) -> None:
        self.events.append(("press", key))


class _FakeContext:
    def __init__(self, cookies: list[dict[str, Any]] | None = None) -> None:
        self._cookies = (
            list(cookies)
            if cookies is not None
            else [{"name": "__Secure-1PSID", "value": "test-session", "expires": -1}]
        )

    async def cookies(self, _urls: list[str] | None = None) -> list[dict[str, Any]]:
        return list(self._cookies)


class _FakePage:
    def __init__(
        self,
        *,
        url: str = "https://gemini.google.com/app/test-chat",
        response_text: str = "",
        turn_images: list[dict[str, Any]] | None = None,
        download_button_bytes: bytes | None = None,
        http_responses: dict[str, tuple[int, bytes]] | None = None,
        cookies: list[dict[str, Any]] | None = None,
        show_signin: bool = False,
        account_label: str | None = None,
    ) -> None:
        self.url = url
        self.context = _FakeContext(cookies=cookies)
        self.keyboard = _FakeKeyboard()
        self._show_signin = show_signin
        self._account_label = account_label
        self._response_text = response_text
        self._turn_images = turn_images or []
        self._download_button_bytes = download_button_bytes
        self.request = _FakeRequestContext(http_responses or {})
        self.file_input = _FakeLocator(count=1, visible=True)
        self.composer = _FakeLocator(count=1, visible=True)
        self.send_btn = _FakeLocator(count=1, visible=True)
        self.download_btn = _FakeLocator(
            count=1 if download_button_bytes is not None else 0,
            visible=download_button_bytes is not None,
        )
        self.turn_count = 0
        self.turn_loc = _FakeLocator(
            count=1,
            visible=True,
            text=response_text,
            images=self._turn_images,
            children={
                "message-content .markdown": _FakeLocator(visible=True, text=response_text),
                'button[aria-label="Good response"]': _FakeLocator(visible=True, count=1),
                'button[aria-label*="Download full size" i]': self.download_btn,
                "img": _FakeLocator(count=len(self._turn_images), visible=True),
            },
        )
        self._turns_wrapper = _FakeLocator(count=0)

    def locator(self, selector: str) -> _FakeLocator:
        if selector == 'input[type="file"]':
            return self.file_input
        if selector == 'rich-textarea div.ql-editor[contenteditable="true"]':
            return self.composer
        if selector == 'button[aria-label*="Send message" i]':
            return self.send_btn
        if selector == 'button:has-text("Sign in")' and self._show_signin:
            return _FakeLocator(visible=True, count=1)
        if selector == 'a[aria-label*="Google Account" i]' and self._account_label:
            return _FakeLocator(
                visible=True,
                count=1,
                attrs={"aria-label": self._account_label},
            )
        if selector == "model-response":
            self.turn_loc._count = self.turn_count
            return self.turn_loc
        if selector == 'img[data-test-id="image-preview"][src^="blob:"]':
            return _FakeLocator(visible=True, count=1)
        if selector == 'button[aria-label*="Download full size" i]':
            return self.download_btn
        return _FakeLocator(visible=False, count=0)

    async def goto(self, url: str, **_kwargs: Any) -> None:
        self.url = url

    async def evaluate(self, script: str, *_args: Any) -> Any:
        if "ServiceLogin" in script or "sign in" in script.lower():
            return self._show_signin
        if "document.body" in script and "innerText" in script:
            return "Gemini workspace"
        if "querySelectorAll('img')" in script:
            return []
        return None

    async def screenshot(self, path: str, **_kwargs: Any) -> None:
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")

    async def content(self) -> str:
        return "<html><body>Gemini</body></html>"

    @contextlib.asynccontextmanager
    async def expect_download(self, **_kwargs: Any) -> Any:
        assert self._download_button_bytes is not None
        yield _FakeDownloadInfo(self._download_button_bytes)


@pytest.mark.asyncio
async def test_mocked_playwright_chat_extract_json_and_attachment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {"title": "Bears", "shots": [{"id": "seg01"}]}
    page = _FakePage(response_text="```json\n" + json.dumps(payload) + "\n```")
    att = tmp_path / "brief.txt"
    att.write_text("story brief", encoding="utf-8")
    out_file = tmp_path / "out.json"

    async def _fast_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("gflow_cli.gemini.client.asyncio.sleep", _fast_sleep)

    client = GeminiWebClient(
        profile="gemini_ultra",
        home=tmp_path / "home",
        debug_dir=tmp_path / "debug",
        enforce_risk_guard=False,
    )
    client._page = page

    orig_submit = client._enter_prompt_and_submit

    async def _submit_and_spawn_turn(p: Any, prompt: str) -> None:
        await orig_submit(p, prompt)
        page.turn_count = 1

    monkeypatch.setattr(client, "_enter_prompt_and_submit", _submit_and_spawn_turn)
    clock = itertools.count(0.0, 1.5)
    monkeypatch.setattr("gflow_cli.gemini.client.time.monotonic", lambda: next(clock))

    res = await client.chat(
        "Generate storyboard JSON",
        attachments=[att],
        extract_json=True,
        out_file=out_file,
        timeout_seconds=30,
    )
    assert res.ok is True
    assert res.extracted_json == payload
    assert out_file.is_file()
    assert json.loads(out_file.read_text(encoding="utf-8")) == payload
    assert page.file_input.uploaded_files == [[str(att.resolve())]]
    assert page.composer.filled_values == ["Generate storyboard JSON"]


@pytest.mark.asyncio
async def test_mocked_playwright_image_download_p1_button(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    png_bytes = _make_png_bytes(1344, 768)
    page = _FakePage(
        response_text="Here is your image",
        turn_images=[
            {
                "index": 0,
                "src": "https://lh3.googleusercontent.com/gg/gen1=s512",
                "naturalWidth": 1344,
                "naturalHeight": 768,
            }
        ],
        download_button_bytes=png_bytes,
    )
    ref_img = tmp_path / "master.png"
    ref_img.write_bytes(png_bytes)
    out_file = tmp_path / "seg01.png"

    async def _fast_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("gflow_cli.gemini.client.asyncio.sleep", _fast_sleep)

    client = GeminiWebClient(
        profile="gemini_ultra",
        home=tmp_path / "home",
        debug_dir=tmp_path / "debug",
        enforce_risk_guard=False,
    )
    client._page = page

    orig_submit = client._enter_prompt_and_submit

    async def _submit_and_spawn_turn(p: Any, prompt: str) -> None:
        await orig_submit(p, prompt)
        page.turn_count = 1

    monkeypatch.setattr(client, "_enter_prompt_and_submit", _submit_and_spawn_turn)
    clock = itertools.count(0.0, 1.5)
    monkeypatch.setattr("gflow_cli.gemini.client.time.monotonic", lambda: next(clock))

    res = await client.generate_image(
        "Two bears drinking tea",
        out_file=out_file,
        ref_paths=[ref_img],
        aspect="16:9",
        strip_watermark=True,
        timeout_seconds=30,
    )
    assert res.ok is True
    assert res.download_method == "download_button"
    assert res.watermark_stripped is True
    assert res.aspect_ratio == "16:9"
    assert out_file.is_file()
    assert page.file_input.uploaded_files == [[str(ref_img.resolve())]]


@pytest.mark.asyncio
async def test_mocked_playwright_image_download_p2_s0_upgrade_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    png_bytes = _make_png_bytes(1408, 768)
    raw_src = "https://lh3.googleusercontent.com/gg/gen2=s512-rj"
    upgraded_s0 = "https://lh3.googleusercontent.com/gg/gen2=s0"
    page = _FakePage(
        response_text="Generated image",
        turn_images=[{"index": 0, "src": raw_src, "naturalWidth": 1024, "naturalHeight": 576}],
        download_button_bytes=None,  # Force P2 fallback
        http_responses={upgraded_s0: (200, png_bytes)},
    )
    out_file = tmp_path / "seg02.png"

    async def _fast_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("gflow_cli.gemini.client.asyncio.sleep", _fast_sleep)

    client = GeminiWebClient(
        profile="gemini_ultra",
        home=tmp_path / "home",
        debug_dir=tmp_path / "debug",
        enforce_risk_guard=False,
    )
    client._page = page

    orig_submit = client._enter_prompt_and_submit

    async def _submit_and_spawn_turn(p: Any, prompt: str) -> None:
        await orig_submit(p, prompt)
        page.turn_count = 1

    monkeypatch.setattr(client, "_enter_prompt_and_submit", _submit_and_spawn_turn)
    clock = itertools.count(0.0, 1.5)
    monkeypatch.setattr("gflow_cli.gemini.client.time.monotonic", lambda: next(clock))

    res = await client.generate_image("A cozy room", out_file=out_file, timeout_seconds=30)
    assert res.ok is True
    assert res.download_method == "img_src_fetch"
    assert res.width == 1408
    assert res.height == 768
    assert page.request.requested_urls == [upgraded_s0]


@pytest.mark.asyncio
async def test_mocked_playwright_image_refusal_fails_fast_with_exit_26(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    page = _FakePage(
        response_text="I cannot generate that image due to safety guidelines.",
        turn_images=[],  # No image produced
    )
    async def _fast_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("gflow_cli.gemini.client.asyncio.sleep", _fast_sleep)

    client = GeminiWebClient(
        profile="gemini_ultra",
        home=tmp_path / "home",
        debug_dir=tmp_path / "debug",
        enforce_risk_guard=False,
    )
    client._page = page

    orig_submit = client._enter_prompt_and_submit

    async def _submit_and_spawn_turn(p: Any, prompt: str) -> None:
        await orig_submit(p, prompt)
        page.turn_count = 1

    monkeypatch.setattr(client, "_enter_prompt_and_submit", _submit_and_spawn_turn)
    clock = itertools.count(0.0, 2.5)
    monkeypatch.setattr("gflow_cli.gemini.client.time.monotonic", lambda: next(clock))

    with pytest.raises(GeminiNoImageGeneratedError) as exc_info:
        await client.generate_image(
            "blocked prompt", out_file=tmp_path / "nope.png", timeout_seconds=30
        )
    assert exc_info.value.exit_code == 26
    assert exc_info.value.retryable is False
    assert exc_info.value.screenshot_path is not None
    assert exc_info.value.screenshot_path.is_file()


# ---------------------------------------------------------------------------
# Strict cookie + DOM authentication regression tests (guest-mode protection)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guest_mode_without_auth_cookies_fails_auth_and_blocks_chat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    guest_page = _FakePage(
        url="https://gemini.google.com/app",
        cookies=[],
        show_signin=False,
    )
    ok, hint = await detect_gemini_authenticated(guest_page.context, guest_page)
    assert (ok, hint) == (False, None)

    expired_ctx = _FakeContext(
        cookies=[{"name": "__Secure-1PSID", "value": "stale", "expires": 1.0}]
    )
    assert await has_google_auth_cookies(expired_ctx) is False

    async def _fast_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("gflow_cli.gemini.client.asyncio.sleep", _fast_sleep)

    client = GeminiWebClient(
        profile="gemini_ultra",
        home=tmp_path / "home",
        debug_dir=tmp_path / "debug",
        enforce_risk_guard=False,
    )
    client._page = guest_page
    client._context = guest_page.context

    with pytest.raises(GeminiAuthRequiredError) as exc_info:
        await client.chat("hello from guest", timeout_seconds=10)
    assert exc_info.value.exit_code == 3
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_valid_cookie_with_visible_signin_cta_is_not_authenticated() -> None:
    page = _FakePage(
        url="https://gemini.google.com/app",
        cookies=[{"name": "__Secure-1PSID", "value": "valid-cookie", "expires": -1}],
        show_signin=True,
    )
    assert await has_google_auth_cookies(page.context) is True
    ok, hint = await detect_gemini_authenticated(page.context, page)
    assert (ok, hint) == (False, None)


@pytest.mark.asyncio
async def test_valid_cookie_composer_without_signin_returns_authenticated_and_hint() -> None:
    page = _FakePage(
        url="https://gemini.google.com/app",
        cookies=[{"name": "__Secure-1PSID", "value": "valid-cookie", "expires": -1}],
        show_signin=False,
        account_label="Google Account: Ultra User (ultra@example.com)",
    )
    ok, hint = await detect_gemini_authenticated(page.context, page)
    assert ok is True
    assert hint == "ultra@example.com"
