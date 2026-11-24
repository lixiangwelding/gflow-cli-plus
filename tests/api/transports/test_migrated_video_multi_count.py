"""Multi-candidate video generation (--count 2..4) on flow.google.com."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

from gflow_cli import json_output
from gflow_cli.api.transports import _mc_batch_observe, migrated_composer
from gflow_cli.api.transports.batchexecute import generation_records
from gflow_cli.api.video import (
    Aspect,
    GenerateVideoRequest,
    Mode,
    VideoModel,
    VideoResult,
    VideoStatus,
)
from gflow_cli.cli_video_reporting import _emit_video_result
from gflow_cli.errors import MediaDownloadError
from gflow_cli.paths import relocate_video_output

PROJ = "22222222-2222-4222-8222-222222222222"
WF1 = "11111111-1111-4111-8111-111111111111"
WF2 = "11111111-1111-4111-8111-222222222222"
WF3 = "11111111-1111-4111-8111-333333333333"
WF4 = "11111111-1111-4111-8111-444444444444"
M1 = "aaaaaaaa-aaaa-4aaa-8aaa-111111111111"
M2 = "aaaaaaaa-aaaa-4aaa-8aaa-222222222222"
M3 = "aaaaaaaa-aaaa-4aaa-8aaa-333333333333"
M4 = "aaaaaaaa-aaaa-4aaa-8aaa-444444444444"
PROMPT = "two cute plush bears chasing a glowing donut"


def _make_raw_record(
    wf: str,
    media: str,
    status: int,
    *,
    video_url: str | None = None,
    size: int | None = None,
    upload_id: str | None = None,
) -> list[Any]:
    prompt_node: Any = [None, None, [[[PROMPT]]]]
    if upload_id:
        prompt_node = [None, None, [[[PROMPT]], upload_id]]
    details: list[Any] = [
        [1788563121, 855319000],
        PROMPT,
        None,
        None,
        None,
        None,
        [None, [["omni_flash_i2v_3s", 1, None, None, 2, 1]], prompt_node, None, 1],
        None,
        [status],
        1,
    ]
    if video_url is not None or size is not None:
        details += [None, [], None, size]
    media_info: list[Any] = [
        [
            None,
            926545,
            None,
            None,
            None,
            None,
            None,
            PROMPT,
            video_url,
            None,
            None,
            None,
            "omni_flash_i2v_3s",
            "",
            None,
            False,
            2,
        ],
        [None, None, [3]],
        [wf],
    ]
    return [wf, PROJ, media, "CAE", None, details, None, media_info]


def _envelope(frames: list[tuple[str, Any]]) -> str:
    chunks = []
    for rpcid, payload in frames:
        wrb = ["wrb.fr", rpcid, json.dumps(payload), None, None, None, "generic"]
        body = json.dumps([wrb, ["di", 100]])
        chunks.append(f"{len(body) + 1}\n{body}")
    tail = json.dumps([["e", 4, None, None, 100]])
    return ")]}'\n\n" + "\n".join(chunks) + f"\n{len(tail) + 1}\n{tail}\n"


class _FakeResponse:
    def __init__(self, url: str, body: str) -> None:
        self.url = url
        self._body = body
        self.request = None

    async def text(self) -> str:
        return self._body


class _FakeLocator:
    def __init__(self, page: _FakeBatchPage) -> None:
        self._page = page

    def filter(self, **_kw: Any) -> _FakeLocator:
        return self

    @property
    def first(self) -> _FakeLocator:
        return self

    async def count(self) -> int:
        return 1

    async def is_visible(self) -> bool:
        return True

    async def is_disabled(self) -> bool:
        return False

    async def is_enabled(self) -> bool:
        return True

    async def click(self, **_kw: Any) -> None:
        await self._page.trigger_submit()


@dataclass
class _FakeBatchPage:
    scripted_responses: list[tuple[float, str, str]]
    listeners: dict[str, list[Any]] = field(default_factory=lambda: {"response": [], "request": []})
    tasks: list[asyncio.Task[Any]] = field(default_factory=lambda: [])

    def on(self, event: str, fn: Any) -> None:
        self.listeners.setdefault(event, []).append(fn)

    def remove_listener(self, event: str, fn: Any) -> None:
        if fn in self.listeners.get(event, []):
            self.listeners[event].remove(fn)

    async def route(self, _glob: str, _handler: Any) -> None:
        return None

    async def unroute(self, _glob: str, _handler: Any) -> None:
        return None

    def locator(self, _selector: str) -> _FakeLocator:
        return _FakeLocator(self)

    async def trigger_submit(self) -> None:
        async def _emit(delay_s: float, url: str, body: str) -> None:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            resp = _FakeResponse(url, body)
            for fn in list(self.listeners.get("response", [])):
                res = fn(resp)
                if asyncio.iscoroutine(res):
                    await res

        for delay_s, url, body in self.scripted_responses:
            self.tasks.append(asyncio.create_task(_emit(delay_s, url, body)))

    async def drain(self) -> None:
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)


# --- batchexecute.generation_records ----------------------------------------


def test_generation_records_extracts_all_candidates_distinct_and_shared_workflow() -> None:
    payload = [
        None,
        881,
        [
            [_make_raw_record(WF1, M1, 6)],
            [_make_raw_record(WF1, M2, 6)],
            [
                _make_raw_record(
                    WF2, M3, 3, video_url="https://flow-content.google/v/3.mp4", size=99
                )
            ],
        ],
    ]
    recs = generation_records("eb1hJf", payload)
    assert [r.media_id for r in recs] == [M1, M2, M3]
    assert recs[0].workflow_id == WF1
    assert recs[1].workflow_id == WF1
    assert recs[2].video_url == "https://flow-content.google/v/3.mp4"
    assert recs[2].size_bytes == 99


def test_generation_records_merges_duplicate_media_id_without_losing_url() -> None:
    payload = [
        [_make_raw_record(WF1, M1, 3, video_url="https://flow-content.google/v/1.mp4", size=1234)],
        [_make_raw_record(WF1, M1, 3, video_url=None, size=None)],
    ]
    recs = generation_records("as29s", payload)
    assert len(recs) == 1
    assert recs[0].media_id == M1
    assert recs[0].video_url == "https://flow-content.google/v/1.mp4"
    assert recs[0].size_bytes == 1234


# --- _mc_batch_observe end-to-end -------------------------------------------


@pytest.mark.asyncio
async def test_batch_waits_for_all_candidates_and_relocates_numbered_files(tmp_path: Path) -> None:
    url1 = "https://flow-content.google/v/c1.mp4?Signature=1"
    url2 = "https://flow-content.google/v/c2.mp4?Signature=2"
    url3 = "https://flow-content.google/v/c3.mp4?Signature=3"
    url4 = "https://flow-content.google/v/c4.mp4?Signature=4"
    submit_env = _envelope(
        [
            (
                "eb1hJf",
                [
                    None,
                    881,
                    [
                        [_make_raw_record(WF1, M1, 6)],
                        [_make_raw_record(WF2, M2, 6)],
                        [_make_raw_record(WF3, M3, 6)],
                        [_make_raw_record(WF4, M4, 6)],
                    ],
                ],
            )
        ]
    )
    # Candidate #1 finishes early while #2-#4 are still running; must wait for all 4.
    poll_1 = _envelope(
        [
            (
                "as29s",
                [
                    [_make_raw_record(WF1, M1, 3, video_url=url1, size=1001)],
                    [_make_raw_record(WF2, M2, 2)],
                    [_make_raw_record(WF3, M3, 2)],
                    [_make_raw_record(WF4, M4, 2)],
                ],
            )
        ]
    )
    poll_2 = _envelope(
        [
            (
                "as29s",
                [
                    [_make_raw_record(WF2, M2, 3, video_url=url2, size=1002)],
                    [_make_raw_record(WF3, M3, 3, video_url=url3, size=1003)],
                    [_make_raw_record(WF4, M4, 3, video_url=url4, size=1004)],
                ],
            )
        ]
    )
    page = _FakeBatchPage(
        [
            (0.01, "https://flow.google.com/batchexecute?rpcids=eb1hJf", submit_env),
            (0.03, "https://flow.google.com/batchexecute?rpcids=as29s", poll_1),
            (0.06, "https://flow.google.com/batchexecute?rpcids=as29s", poll_2),
        ]
    )
    composer = migrated_composer.MigratedComposer()
    started_ids: list[str] = []

    async def fake_download(_page: Any, rec: Any, save: Any) -> Path:
        return save(rec.media_id, f"MP4-{rec.media_id}".encode())

    composer.download = AsyncMock(side_effect=fake_download)  # type: ignore[method-assign]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    def saver(media_id: str, data: bytes) -> Path:
        p = raw_dir / f"{media_id}.mp4"
        p.write_bytes(data)
        return p

    result = await _mc_batch_observe.run_video_batch(
        composer,
        page,  # type: ignore[arg-type]
        expected_count=4,
        poll_timeout_s=5.0,
        on_started=lambda s: started_ids.append(s.media_id),
        project_id=PROJ,
        save=saver,
        download=True,
    )
    await page.drain()

    assert started_ids == [M1, M2, M3, M4]
    assert len(result.all_results) == 4
    assert all(r.status.succeeded for r in result.all_results)

    target = tmp_path / "out" / "seg01.mp4"
    relocated = relocate_video_output(result, target, force_numbered=True)
    paths = [r.local_path for r in relocated.all_results]
    assert paths == [
        tmp_path / "out" / "seg01_1.mp4",
        tmp_path / "out" / "seg01_2.mp4",
        tmp_path / "out" / "seg01_3.mp4",
        tmp_path / "out" / "seg01_4.mp4",
    ]
    assert not target.exists()
    for path, mid in zip(paths, [M1, M2, M3, M4], strict=True):
        assert path is not None and path.read_bytes() == f"MP4-{mid}".encode()


@pytest.mark.asyncio
async def test_batch_partial_success_salvages_good_candidate_and_json_contract(
    tmp_path: Path,
) -> None:
    url2 = "https://flow-content.google/v/c2.mp4?Signature=2"
    submit_env = _envelope(
        [
            (
                "eb1hJf",
                [None, 881, [[_make_raw_record(WF1, M1, 6)], [_make_raw_record(WF1, M2, 6)]]],
            )
        ]
    )
    # Candidate #1 fails (status=4), candidate #2 succeeds with URL.
    poll_env = _envelope(
        [
            (
                "as29s",
                [
                    [_make_raw_record(WF1, M1, 4)],
                    [_make_raw_record(WF1, M2, 3, video_url=url2, size=2048)],
                ],
            )
        ]
    )
    page = _FakeBatchPage(
        [
            (0.01, "https://flow.google.com/batchexecute?rpcids=eb1hJf", submit_env),
            (0.03, "https://flow.google.com/batchexecute?rpcids=as29s", poll_env),
        ]
    )
    composer = migrated_composer.MigratedComposer()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    async def fake_download(_page: Any, rec: Any, save: Any) -> Path:
        return save(rec.media_id, b"VALID_MP4")

    composer.download = AsyncMock(side_effect=fake_download)  # type: ignore[method-assign]

    def saver(media_id: str, data: bytes) -> Path:
        p = raw_dir / f"{media_id}.mp4"
        p.write_bytes(data)
        return p

    result = await _mc_batch_observe.run_video_batch(
        composer,
        page,  # type: ignore[arg-type]
        expected_count=2,
        poll_timeout_s=5.0,
        on_started=None,
        project_id=PROJ,
        save=saver,
        download=True,
    )
    await page.drain()

    relocated = relocate_video_output(result, tmp_path / "seg02.mp4", force_numbered=True)
    assert relocated.all_results[0].status.succeeded is False
    assert relocated.all_results[0].local_path is None
    assert relocated.all_results[1].status.succeeded is True
    assert relocated.all_results[1].local_path == tmp_path / "seg02_2.mp4"
    assert (tmp_path / "seg02_2.mp4").read_bytes() == b"VALID_MP4"

    req = GenerateVideoRequest(
        prompt=PROMPT,
        mode=Mode.I2V,
        start_image_ref_id=M1,
        aspect=Aspect.LANDSCAPE,
        model=VideoModel.OMNI_1_2_FLASH,
        duration=6,
        count=2,
    )
    payload = json_output.video_result(command="video i2v", request=req, result=relocated)
    assert payload["status"] == "ok"
    assert payload["succeeded"] is True
    assert payload["partial"] is True
    assert payload["requested_count"] == 2
    assert payload["observed_count"] == 2
    assert payload["succeeded_count"] == 1
    assert payload["failed_count"] == 1
    assert payload["local_path"] == str(tmp_path / "seg02_2.mp4")
    assert payload["local_paths"] == [str(tmp_path / "seg02_2.mp4")]
    assert len(payload["candidates"]) == 2
    assert payload["candidates"][0]["succeeded"] is False
    assert payload["candidates"][1]["succeeded"] is True


@pytest.mark.asyncio
async def test_batch_download_error_on_one_clip_preserves_other_paid_clip(tmp_path: Path) -> None:
    rec1 = generation_records(
        "as29s",
        [[_make_raw_record(WF1, M1, 3, video_url="https://flow-content.google/v/1.mp4", size=50)]],
    )[0]
    rec2 = generation_records(
        "as29s",
        [[_make_raw_record(WF2, M2, 3, video_url="https://flow-content.google/v/2.mp4", size=60)]],
    )[0]
    composer = migrated_composer.MigratedComposer()

    async def flaky_download(_page: Any, rec: Any, save: Any) -> Path:
        if rec.media_id == M1:
            raise MediaDownloadError(detail="transient CDN hiccup")
        return save(rec.media_id, b"CLIP2")

    composer.download = AsyncMock(side_effect=flaky_download)  # type: ignore[method-assign]

    def saver(media_id: str, data: bytes) -> Path:
        p = tmp_path / f"{media_id}.mp4"
        p.write_bytes(data)
        return p

    result = await _mc_batch_observe.download_batch_results(
        composer,
        page=None,  # type: ignore[arg-type]
        records=[rec1, rec2],
        project_id=PROJ,
        save=saver,
        download=True,
    )
    assert result.all_results[0].status.succeeded is False
    assert result.all_results[1].status.succeeded is True
    assert result.all_results[1].local_path == tmp_path / f"{M2}.mp4"


def test_single_count_json_and_relocation_remain_unchanged(tmp_path: Path) -> None:
    src = tmp_path / "raw.mp4"
    src.write_bytes(b"SINGLE")
    res = VideoResult(
        status=VideoStatus(media_id=M1, status="MEDIA_GENERATION_STATUS_SUCCESSFUL"),
        local_path=src,
        project_id=PROJ,
        workflow_id=WF1,
    )
    target = tmp_path / "out" / "seg01.mp4"
    relocated = relocate_video_output(res, target, force_numbered=False)
    assert relocated.local_path == target
    assert target.read_bytes() == b"SINGLE"

    req = GenerateVideoRequest(
        prompt=PROMPT,
        mode=Mode.I2V,
        start_image_ref_id=M1,
        aspect=Aspect.LANDSCAPE,
        model=VideoModel.OMNI_1_2_FLASH,
        duration=6,
        count=1,
    )
    payload = json_output.video_result(command="video i2v", request=req, result=relocated)
    assert payload["status"] == "ok"
    assert payload["local_path"] == str(target)
    assert "partial" not in payload
    assert "candidates" not in payload
    assert "local_paths" not in payload


def test_emit_video_result_multi_count_total_failure_exits_1(capsys: Any) -> None:
    req = GenerateVideoRequest(
        prompt=PROMPT,
        mode=Mode.I2V,
        start_image_ref_id=M1,
        aspect=Aspect.LANDSCAPE,
        model=VideoModel.OMNI_1_2_FLASH,
        duration=6,
        count=2,
    )
    r1 = VideoResult(
        status=VideoStatus(
            media_id=M1, status="MEDIA_GENERATION_STATUS_FAILED", error_message="policy"
        ),
        local_path=None,
        project_id=PROJ,
        workflow_id=WF1,
    )
    r2 = VideoResult(
        status=VideoStatus(
            media_id=M2, status="MEDIA_GENERATION_STATUS_FAILED", error_message="policy"
        ),
        local_path=None,
        project_id=PROJ,
        workflow_id=WF2,
    )
    combined = VideoResult(
        status=r1.status,
        local_path=None,
        project_id=PROJ,
        workflow_id=WF1,
        extra_results=(r2,),
    )
    with pytest.raises(SystemExit) as exc_info:
        _emit_video_result("video i2v", req, combined, "json", True, Console(quiet=True))
    assert exc_info.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "fail"
    assert out["succeeded_count"] == 0
    assert out["failed_count"] == 2
    assert out["charged"] == "maybe"
    assert out["next_action"] == "reconcile"


@pytest.mark.asyncio
async def test_batch_adopts_late_sibling_sharing_workflow_id(tmp_path: Path) -> None:
    url1 = "https://flow-content.google/v/s1.mp4?Signature=1"
    url2 = "https://flow-content.google/v/s2.mp4?Signature=2"
    # Submit reply only lists M1; status poll later reveals sibling M2 under the same WF1.
    submit_env = _envelope([("eb1hJf", [None, 881, [[_make_raw_record(WF1, M1, 6)]]])])
    poll_env = _envelope(
        [
            (
                "as29s",
                [
                    [_make_raw_record(WF1, M1, 3, video_url=url1, size=500)],
                    [_make_raw_record(WF1, M2, 3, video_url=url2, size=600)],
                ],
            )
        ]
    )
    page = _FakeBatchPage(
        [
            (0.01, "https://flow.google.com/batchexecute?rpcids=eb1hJf", submit_env),
            (0.03, "https://flow.google.com/batchexecute?rpcids=as29s", poll_env),
        ]
    )
    composer = migrated_composer.MigratedComposer()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    async def fake_download(_page: Any, rec: Any, save: Any) -> Path:
        return save(rec.media_id, f"SIB-{rec.media_id}".encode())

    composer.download = AsyncMock(side_effect=fake_download)  # type: ignore[method-assign]

    def saver(media_id: str, data: bytes) -> Path:
        p = raw_dir / f"{media_id}.mp4"
        p.write_bytes(data)
        return p

    result = await _mc_batch_observe.run_video_batch(
        composer,
        page,  # type: ignore[arg-type]
        expected_count=2,
        poll_timeout_s=5.0,
        on_started=None,
        project_id=PROJ,
        save=saver,
        download=True,
    )
    await page.drain()

    assert [r.status.media_id for r in result.all_results] == [M1, M2]
    assert all(r.status.succeeded and r.local_path is not None for r in result.all_results)

