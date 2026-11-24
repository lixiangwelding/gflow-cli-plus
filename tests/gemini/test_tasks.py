"""Unit tests for Gemini Ultra Tasks (`任务`) mode: CLI + mocked engine + risk guards."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from gflow_cli.cli import main
from gflow_cli.gemini import (
    GeminiImageMetadata,
    GeminiRiskBlockedError,
    GeminiRiskLedger,
    GeminiTaskHarvestResult,
    GeminiTaskSubmitResult,
    GeminiUiDriftError,
    GeminiWebClient,
    harvest_task,
    submit_task,
)


def _make_png(
    width: int = 1280, height: int = 720, color: tuple[int, int, int] = (210, 170, 130)
) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI subcommand tests: `gflow gemini task-submit` & `gflow gemini task-harvest`
# ---------------------------------------------------------------------------


def test_task_submit_cli_validation_and_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Missing prompt -> exit 2
    assert CliRunner().invoke(main, ["gemini", "task-submit"]).exit_code == 2

    pf = tmp_path / "task_prompt.txt"
    pf.write_text("Generate 50 fresh Yierbubu stories", encoding="utf-8")
    f1 = tmp_path / "story_memory_export.json"
    f2 = tmp_path / "master.jpg"
    f1.write_text('{"episodes": []}', encoding="utf-8")
    f2.write_bytes(b"\xff\xd8\xff\xd9")

    # Both PROMPT and --prompt-file -> exit 2
    res_both = CliRunner().invoke(
        main, ["gemini", "task-submit", "inline", "--prompt-file", str(pf)]
    )
    assert res_both.exit_code == 2

    # Non-positive timeout -> exit 2
    res_bad_to = CliRunner().invoke(main, ["gemini", "task-submit", "hi", "--timeout", "0"])
    assert res_bad_to.exit_code == 2

    fake_res = GeminiTaskSubmitResult(
        ok=True,
        profile="gemini_ultra",
        chat_url="https://gemini.google.com/app/5b9d19b9557b5e25",
        attachments=[str(f1.resolve()), str(f2.resolve())],
        elapsed_seconds=12.4,
        warnings=[],
    )
    mock_submit = AsyncMock(return_value=fake_res)
    monkeypatch.setattr("gflow_cli.gemini.submit_task", mock_submit)

    res = CliRunner().invoke(
        main,
        [
            "gemini",
            "task-submit",
            "--prompt-file",
            str(pf),
            "--file",
            str(f1),
            "--file",
            str(f2),
            "--profile",
            "gemini_ultra",
            "--headless",
            "--timeout",
            "95",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    doc = json.loads(res.stdout)
    assert doc["status"] == "ok"
    assert doc["ok"] is True
    assert doc["command"] == "gemini task-submit"
    assert doc["chat_url"] == "https://gemini.google.com/app/5b9d19b9557b5e25"
    assert len(doc["attachments"]) == 2
    mock_submit.assert_awaited_once_with(
        "Generate 50 fresh Yierbubu stories",
        profile="gemini_ultra",
        attachments=[f1.resolve(), f2.resolve()],
        headless=True,
        timeout_seconds=95,
    )


def test_task_harvest_cli_validation_and_status_exit_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_url = "https://gemini.google.com/app/5b9d19b9557b5e25"

    # Validation errors -> exit 2
    assert CliRunner().invoke(main, ["gemini", "task-harvest"]).exit_code == 2
    assert (
        CliRunner()
        .invoke(main, ["gemini", "task-harvest", "--url", task_url, "--wait-timeout", "-1"])
        .exit_code
        == 2
    )
    assert (
        CliRunner()
        .invoke(main, ["gemini", "task-harvest", "--url", task_url, "--download-timeout", "0"])
        .exit_code
        == 2
    )
    assert (
        CliRunner()
        .invoke(main, ["gemini", "task-harvest", "--url", task_url, "--expected-images", "0"])
        .exit_code
        == 2
    )

    # 1. Completed -> exit 0
    img_meta = GeminiImageMetadata(
        path=str(tmp_path / "task_img_01.jpg"),
        width=2752,
        height=1536,
        aspect_ratio="16:9",
        bytes=1_900_000,
        sha256="a" * 64,
        mime_type="image/jpeg",
        download_method="download_button",
        watermark_stripped=False,
    )
    res_completed = GeminiTaskHarvestResult(
        ok=True,
        profile="gemini_ultra",
        chat_url=task_url,
        completed=True,
        status_label="completed",
        text='{"stories": [{"id": "T01"}]}',
        extracted_json={"stories": [{"id": "T01"}]},
        exported_files=[str(tmp_path / "stories.json")],
        images=[img_meta],
        downloaded_count=1,
        expected_images=1,
        missing_image_indices=[],
        elapsed_seconds=42.0,
        out_file=str(tmp_path / "out.json"),
    )
    mock_harvest = AsyncMock(return_value=res_completed)
    monkeypatch.setattr("gflow_cli.gemini.harvest_task", mock_harvest)

    r_ok = CliRunner().invoke(
        main,
        [
            "gemini",
            "task-harvest",
            "--url",
            task_url,
            "--out-dir",
            str(tmp_path),
            "--out-file",
            str(tmp_path / "out.json"),
            "--download-images",
            "--extract-json",
            "--expected-images",
            "1",
            "--wait-timeout",
            "300",
            "--download-timeout",
            "50",
            "--json",
        ],
    )
    assert r_ok.exit_code == 0, r_ok.output
    doc_ok = json.loads(r_ok.stdout)
    assert doc_ok["status"] == "ok"
    assert doc_ok["ok"] is True
    assert doc_ok["completed"] is True
    assert doc_ok["downloaded_count"] == 1
    assert doc_ok["extracted_json"] == {"stories": [{"id": "T01"}]}

    # 2. Non-blocking poll (wait_timeout=0) still running -> exit 0 + status="running"
    res_running = GeminiTaskHarvestResult(
        ok=False,
        profile="gemini_ultra",
        chat_url=task_url,
        completed=False,
        status_label="running",
        text="",
        extracted_json=None,
        exported_files=[],
        images=[],
        downloaded_count=0,
        expected_images=6,
        missing_image_indices=[],
        elapsed_seconds=2.1,
    )
    monkeypatch.setattr("gflow_cli.gemini.harvest_task", AsyncMock(return_value=res_running))
    r_run = CliRunner().invoke(
        main,
        [
            "gemini",
            "task-harvest",
            "--url",
            task_url,
            "--no-download-images",
            "--wait-timeout",
            "0",
            "--json",
        ],
    )
    assert r_run.exit_code == 0, r_run.output
    doc_run = json.loads(r_run.stdout)
    assert doc_run["status"] == "running"
    assert doc_run["ok"] is False
    assert doc_run["completed"] is False

    # 3. Timed out -> exit 9
    res_timeout = GeminiTaskHarvestResult(
        ok=False,
        profile="gemini_ultra",
        chat_url=task_url,
        completed=False,
        status_label="timed_out",
        text="",
        extracted_json=None,
        exported_files=[],
        images=[],
        downloaded_count=0,
        expected_images=6,
        missing_image_indices=[],
        elapsed_seconds=600.0,
    )
    monkeypatch.setattr("gflow_cli.gemini.harvest_task", AsyncMock(return_value=res_timeout))
    r_to = CliRunner().invoke(main, ["gemini", "task-harvest", "--url", task_url, "--json"])
    assert r_to.exit_code == 9
    assert json.loads(r_to.stdout)["status_label"] == "timed_out"

    # 4. Incomplete (missing image indices) -> exit 1 with partial JSON intact
    res_inc = GeminiTaskHarvestResult(
        ok=False,
        profile="gemini_ultra",
        chat_url=task_url,
        completed=True,
        status_label="incomplete",
        text="",
        extracted_json=None,
        exported_files=[],
        images=[img_meta],
        downloaded_count=1,
        expected_images=2,
        missing_image_indices=[2],
        elapsed_seconds=55.0,
    )
    monkeypatch.setattr("gflow_cli.gemini.harvest_task", AsyncMock(return_value=res_inc))
    r_inc = CliRunner().invoke(main, ["gemini", "task-harvest", "--url", task_url, "--json"])
    assert r_inc.exit_code == 1
    doc_inc = json.loads(r_inc.stdout)
    assert doc_inc["status"] == "fail"
    assert doc_inc["missing_image_indices"] == [2]
    assert doc_inc["downloaded_count"] == 1


# ---------------------------------------------------------------------------
# Mocked Playwright engine tests for `submit_task` and `harvest_task`
# ---------------------------------------------------------------------------


class _FakeTaskContext:
    async def cookies(self, _urls: list[str] | None = None) -> list[dict[str, Any]]:
        return [{"name": "__Secure-1PSID", "value": "valid-ultra-session", "expires": -1}]


class _FakeTaskButton:
    def __init__(
        self,
        *,
        visible: bool = True,
        text: str = "",
        aria_pressed: str = "false",
        attrs: dict[str, str] | None = None,
        on_click: Any = None,
    ) -> None:
        self._visible = visible
        self._text = text
        self._attrs = dict(attrs or {})
        self._attrs.setdefault("aria-pressed", aria_pressed)
        self._on_click = on_click
        self.clicks = 0

    @property
    def first(self) -> _FakeTaskButton:
        return self

    def nth(self, _idx: int) -> _FakeTaskButton:
        return self

    async def count(self) -> int:
        return 1 if self._visible else 0

    async def is_visible(self, **_kwargs: Any) -> bool:
        return self._visible

    async def is_enabled(self, **_kwargs: Any) -> bool:
        return self._visible

    async def inner_text(self, **_kwargs: Any) -> str:
        return self._text

    async def get_attribute(self, name: str, **_kwargs: Any) -> str | None:
        return self._attrs.get(name)

    async def scroll_into_view_if_needed(self, **_kwargs: Any) -> None:
        return None

    async def click(self, **_kwargs: Any) -> None:
        self.clicks += 1
        self._attrs["aria-pressed"] = "true"
        if callable(self._on_click):
            self._on_click()


class _FakeButtonList:
    def __init__(self, items: list[_FakeTaskButton]) -> None:
        self._items = items

    @property
    def first(self) -> _FakeTaskButton:
        return self._items[0] if self._items else _FakeTaskButton(visible=False)

    def nth(self, idx: int) -> _FakeTaskButton:
        return self._items[idx]

    async def count(self) -> int:
        return len(self._items)


class _FakeTaskComposer:
    def __init__(self) -> None:
        self._text = ""
        self.filled: list[str] = []

    @property
    def first(self) -> _FakeTaskComposer:
        return self

    async def is_visible(self, **_kwargs: Any) -> bool:
        return True

    async def click(self, **_kwargs: Any) -> None:
        return None

    async def fill(self, value: str, **_kwargs: Any) -> None:
        self.filled.append(value)
        self._text = value

    async def inner_text(self, **_kwargs: Any) -> str:
        return self._text


class _FakeFileInput:
    def __init__(self) -> None:
        self.uploads: list[list[str]] = []

    @property
    def first(self) -> _FakeFileInput:
        return self

    async def count(self) -> int:
        return 1

    async def set_input_files(self, files: list[str], **_kwargs: Any) -> None:
        self.uploads.append(list(files))


class _FakeDownloadObj:
    def __init__(self, payload: bytes, suggested_filename: str = "file.bin") -> None:
        self._payload = payload
        self.suggested_filename = suggested_filename

    async def save_as(self, path: str) -> None:
        Path(path).write_bytes(self._payload)


class _FakeDownloadHandle:
    def __init__(self, obj: _FakeDownloadObj) -> None:
        self._obj = obj

    @property
    async def value(self) -> _FakeDownloadObj:
        return self._obj


class _FakeTaskPage:
    def __init__(
        self,
        *,
        url: str = "https://gemini.google.com/app",
        has_task_pill: bool = True,
        resolved_url_on_send: str | None = "https://gemini.google.com/app/5b9d19b9557b5e25",
        done_badge: bool = False,
        streaming: bool = False,
        all_text: str = "",
        export_buttons: list[_FakeTaskButton] | None = None,
        image_buttons: list[_FakeTaskButton] | None = None,
        download_script: list[bytes | Exception | tuple[bytes, str]] | None = None,
    ) -> None:
        self.url = url
        self.context = _FakeTaskContext()
        self._has_task_pill = has_task_pill
        self._done_badge = done_badge
        self._streaming = streaming
        self._all_text = all_text
        self._download_script = list(download_script or [])
        self.download_timeouts: list[int] = []

        self.task_pill = _FakeTaskButton(visible=has_task_pill, text="任务", aria_pressed="false")
        self.composer = _FakeTaskComposer()
        self.file_input = _FakeFileInput()

        def _on_send() -> None:
            if resolved_url_on_send:
                self.url = resolved_url_on_send

        self.send_btn = _FakeTaskButton(visible=True, text="发送", on_click=_on_send)
        self.stop_btn = _FakeTaskButton(visible=streaming, text="停止")
        self.export_buttons = _FakeButtonList(export_buttons or [])
        self.image_buttons = _FakeButtonList(image_buttons or [])

    def locator(self, selector: str) -> Any:
        if selector == 'button.gem-segmented-button-action:has-text("任务")':
            return self.task_pill
        if selector == 'rich-textarea div.ql-editor[contenteditable="true"]':
            return self.composer
        if selector == 'input[type="file"]':
            return self.file_input
        if selector == 'img[data-test-id="image-preview"][src^="blob:"]':
            return _FakeTaskButton(visible=True)
        if selector == 'button[aria-label*="Send message" i]':
            return self.send_btn
        if selector == 'button[aria-label*="Stop generating" i]':
            return self.stop_btn
        if selector == 'button:has-text("导出")':
            return self.export_buttons
        if selector == 'button[aria-label*="下载完整尺寸"]':
            return self.image_buttons
        return _FakeTaskButton(visible=False)

    async def goto(self, url: str, **_kwargs: Any) -> None:
        self.url = url

    async def evaluate(self, script: str, *_args: Any) -> Any:
        if "ServiceLogin" in script or "sign in" in script.lower():
            return False
        if "doneRe" in script:
            return self._done_badge
        if "document.body" in script and "innerText" in script:
            return "Gemini Tasks workspace"
        if "model-response" in script and "join" in script:
            return self._all_text
        if "pre code" in script:
            return []
        if "rect.x <= 220" in script:
            return False
        return None

    async def screenshot(self, path: str, **_kwargs: Any) -> None:
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")

    async def content(self) -> str:
        return "<html><body>Gemini Tasks</body></html>"

    @contextlib.asynccontextmanager
    async def expect_download(self, timeout: int = 30_000, **_kwargs: Any) -> Any:
        self.download_timeouts.append(timeout)
        if not self._download_script:
            raise TimeoutError("no scripted download remaining")
        entry = self._download_script.pop(0)
        if isinstance(entry, Exception):
            raise entry
        if isinstance(entry, tuple):
            payload, name = entry
            yield _FakeDownloadHandle(_FakeDownloadObj(payload, suggested_filename=name))
        else:
            yield _FakeDownloadHandle(_FakeDownloadObj(entry, suggested_filename="image.png"))


@pytest.mark.asyncio
async def test_submit_task_activates_pill_uploads_files_and_releases_chrome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    page = _FakeTaskPage(
        url="https://gemini.google.com/app",
        has_task_pill=True,
        resolved_url_on_send="https://gemini.google.com/app/task999abc",
    )
    f_json = tmp_path / "story_memory_export.json"
    f_img = tmp_path / "master.jpg"
    f_json.write_text('{"episodes": []}', encoding="utf-8")
    f_img.write_bytes(_make_png(640, 640))

    lifecycle = {"closes": 0}

    async def _fake_start(self: Any) -> Any:
        self._page = page
        self._context = page.context
        return page

    async def _fake_close(self: Any) -> None:
        lifecycle["closes"] += 1
        self._page = None
        self._context = None

    async def _fast_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(GeminiWebClient, "start", _fake_start)
    monkeypatch.setattr(GeminiWebClient, "close", _fake_close)
    monkeypatch.setattr("gflow_cli.gemini.client.asyncio.sleep", _fast_sleep)

    res = await submit_task(
        "Create 50 stories with Python dedup",
        profile="gemini_ultra",
        attachments=[f_json, f_img],
        home=tmp_path / "home",
        debug_dir=tmp_path / "debug",
        enforce_risk_guard=False,
    )
    assert res.ok is True
    assert res.chat_url == "https://gemini.google.com/app/task999abc"
    assert page.task_pill.clicks == 1
    assert page.file_input.uploads == [[str(f_json.resolve()), str(f_img.resolve())]]
    assert page.composer.filled == ["Create 50 stories with Python dedup"]
    assert lifecycle["closes"] == 1


@pytest.mark.asyncio
async def test_submit_task_missing_pill_raises_ui_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    page = _FakeTaskPage(has_task_pill=False)

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
    client._context = page.context

    with pytest.raises(GeminiUiDriftError) as exc_info:
        await client.submit_task("test prompt", timeout_seconds=30)
    assert exc_info.value.exit_code == 23
    assert exc_info.value.screenshot_path is not None


@pytest.mark.asyncio
async def test_harvest_task_non_blocking_poll_when_still_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    page = _FakeTaskPage(
        url="https://gemini.google.com/app/running123",
        done_badge=False,
        streaming=True,
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
    client._context = page.context

    res = await client.harvest_task(
        "https://gemini.google.com/app/running123",
        download_images=False,
        wait_timeout=0,
    )
    assert res.ok is False
    assert res.completed is False
    assert res.status_label == "running"


@pytest.mark.asyncio
async def test_harvest_task_multi_image_50s_retry_dedup_and_json_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    img1 = _make_png(1280, 720, color=(200, 100, 50))
    img2 = _make_png(1280, 720, color=(50, 100, 200))
    artifact_json = json.dumps({"stories": [{"id": "T01", "title": "草莓蛋糕大作战"}]}).encode(
        "utf-8"
    )

    export_btn = _FakeTaskButton(visible=True, text="导出", attrs={"aria-label": "导出 JSON"})
    card_btns = [
        _FakeTaskButton(visible=True, attrs={"aria-label": "下载完整尺寸"}),
        _FakeTaskButton(visible=True, attrs={"aria-label": "下载完整尺寸"}),
        _FakeTaskButton(visible=True, attrs={"aria-label": "下载完整尺寸"}),
    ]
    # Sequence: 1st download is artifact export JSON; then card1 succeeds; card2 times out once
    # and succeeds on retry; card3 duplicates img1 (deduped via SHA256!).
    page = _FakeTaskPage(
        url="https://gemini.google.com/app/done777",
        done_badge=True,
        streaming=False,
        all_text="Task completed with exported JSON and 3 image cards.",
        export_buttons=[export_btn],
        image_buttons=card_btns,
        download_script=[
            (artifact_json, "yierbubu_stories.json"),
            img1,
            TimeoutError("download timed out on first click"),
            img2,
            img1,
        ],
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
    client._context = page.context

    out_dir = tmp_path / "harvest"
    out_file = tmp_path / "stories_out.json"
    res = await client.harvest_task(
        "https://gemini.google.com/app/done777",
        out_dir=out_dir,
        out_file=out_file,
        download_images=True,
        extract_json=True,
        expected_images=3,
        wait_timeout=30,
        download_timeout=50,
    )
    assert res.completed is True
    assert res.status_label == "incomplete"
    assert res.ok is False
    assert res.downloaded_count == 2
    assert res.missing_image_indices == [3]
    assert len(res.images) == 2
    assert res.extracted_json == {"stories": [{"id": "T01", "title": "草莓蛋糕大作战"}]}
    assert out_file.is_file()
    assert json.loads(out_file.read_text(encoding="utf-8")) == res.extracted_json
    assert len(res.exported_files) == 1
    # Verify 50s download timeout (50_000 ms) was used for the image cards
    assert 50_000 in page.download_timeouts


@pytest.mark.asyncio
async def test_tasks_enforce_risk_guard_and_harvest_is_readonly_on_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "STOP_FLOW").write_text("halt", encoding="utf-8")

    # Both submit_task and harvest_task must block before launching Chrome when STOP_FLOW exists
    async def _never_start(self: Any) -> Any:
        raise AssertionError("Chrome must not start when STOP_FLOW kill switch is active")

    monkeypatch.setattr(GeminiWebClient, "start", _never_start)

    with pytest.raises(GeminiRiskBlockedError) as e_sub:
        await submit_task("blocked task", home=home)
    assert e_sub.value.exit_code == 10
    assert e_sub.value.hard_halt is True

    with pytest.raises(GeminiRiskBlockedError) as e_harv:
        await harvest_task("https://gemini.google.com/app/t1", home=home)
    assert e_harv.value.exit_code == 10
    assert e_harv.value.hard_halt is True

    # Remove kill switch and verify harvest_task does NOT increment burst counter on ledger
    (home / "STOP_FLOW").unlink()
    page = _FakeTaskPage(
        url="https://gemini.google.com/app/t1",
        done_badge=True,
        streaming=False,
        all_text='{"ok": true}',
    )

    async def _fast_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("gflow_cli.gemini.client.asyncio.sleep", _fast_sleep)

    client = GeminiWebClient(
        profile="gemini_ultra",
        home=home,
        debug_dir=tmp_path / "debug",
        enforce_risk_guard=True,
    )
    client._page = page
    client._context = page.context

    res = await client.harvest_task(
        "https://gemini.google.com/app/t1",
        download_images=False,
        extract_json=True,
        wait_timeout=10,
    )
    assert res.ok is True
    snap = GeminiRiskLedger(home=home).get_profile_risk_snapshot("gemini_ultra")
    assert snap["burst_count_10m"] == 0
