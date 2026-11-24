"""Multi-candidate video submit, observation, and download for ``flow.google.com`` (--count 2..4).

Kept separate from ``_mc_submit_observe.py`` and ``migrated_composer.py`` so single-clip
runs (``count == 1``) keep their exact untouched execution path and module size/complexity
stays well within ``scripts/ci/check_size.py`` limits.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import structlog

from gflow_cli.api.transports._mc_common import (
    BATCHEXECUTE_GLOB,
    LISTING_READ_BUDGET_S,
    LISTING_RPC,
    MIGRATED_PROJECT_URL,
    MULTI_SUBMIT_GRACE_S,
    RESULT_URL_GRACE_S,
    STATUS_RPCS,
    SUBMIT_BUTTON,
    SUBMIT_ENABLE_BUDGET_S,
    SUBMIT_RPCS,
    _ligature,
    _outcome_unknown,
    _submit_body_problem,
    _submit_refusal,
    body_rpcid,
    request_body,
    url_rpcid,
)
from gflow_cli.api.transports._mc_submit_observe import (
    _aborted_before_send,
    read_project_listing,
)
from gflow_cli.api.transports.batchexecute import (
    GenerationRecord,
    _merge_record,
    generation_records,
    generation_records_carrying,
    parse_frames,
)
from gflow_cli.api.video import VideoResult, VideoStarted, VideoStatus
from gflow_cli.errors import GFlowError, WireFormatError

if TYPE_CHECKING:
    from pathlib import Path

    from playwright.async_api import Page

    from gflow_cli.api.transports.migrated_composer import ClipSaver, MigratedComposer
    from gflow_cli.api.video import VideoStartedCallback

log = structlog.get_logger(__name__)


@dataclass
class _MultiClipTracker:
    """Tracks ``1..N`` candidate clips by ``media_id`` until every clip reaches terminal state."""

    expected_count: int
    order: list[str] = field(default_factory=lambda: [])
    records: dict[str, GenerationRecord] = field(default_factory=lambda: {})
    done_no_url_since: dict[str, float] = field(default_factory=lambda: {})
    started_sent: set[str] = field(default_factory=lambda: set())

    def register(self, rec: GenerationRecord, *, now: float) -> bool:
        """Add or update ``rec``; return True if ``rec.media_id`` is newly tracked."""
        is_new = rec.media_id not in self.records
        if is_new:
            self.order.append(rec.media_id)
            self.records[rec.media_id] = rec
        else:
            self.records[rec.media_id] = _merge_record(self.records[rec.media_id], rec)
        merged = self.records[rec.media_id]
        if merged.is_done and not merged.video_url:
            self.done_no_url_since.setdefault(merged.media_id, now)
        elif merged.video_url or merged.is_failed:
            self.done_no_url_since.pop(merged.media_id, None)
        return is_new

    def update_known(self, rec: GenerationRecord, *, now: float) -> bool:
        """Update a tracked clip by ``media_id`` or adopt a late sibling sharing ``workflow_id``."""
        if rec.media_id in self.records:
            self.register(rec, now=now)
            return True
        if len(self.order) < self.expected_count and any(
            cur.workflow_id == rec.workflow_id for cur in self.records.values()
        ):
            self.register(rec, now=now)
            return True
        return False

    def is_settled(self, media_id: str, *, now: float, url_grace_s: float) -> bool:
        rec = self.records[media_id]
        if rec.is_failed:
            return True
        if not rec.is_done:
            return False
        if rec.video_url:
            return True
        since = self.done_no_url_since.get(media_id)
        return since is not None and (now - since) >= url_grace_s

    def all_terminal(self) -> bool:
        return bool(self.order) and all(
            self.records[m].is_done or self.records[m].is_failed for m in self.order
        )

    def all_settled(self, *, now: float, url_grace_s: float) -> bool:
        return bool(self.order) and all(
            self.is_settled(m, now=now, url_grace_s=url_grace_s) for m in self.order
        )

    def next_grace_wait(self, *, now: float, url_grace_s: float) -> float | None:
        """Seconds remaining until all URL-less done clips finish grace; None if any still runs."""
        if not self.all_terminal():
            return None
        waits = [
            max(0.0, url_grace_s - (now - self.done_no_url_since[m]))
            for m in self.order
            if m in self.done_no_url_since and not self.records[m].video_url
        ]
        return max(waits) if waits else 0.0

    def ordered_records(self) -> list[GenerationRecord]:
        return [self.records[m] for m in self.order]


@dataclass
class _BatchWatchState:
    """Mutable synchronization state shared between Playwright listeners and the observer loop."""

    tracker: _MultiClipTracker
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    first_seen: asyncio.Event = field(default_factory=asyncio.Event)
    all_seen: asyncio.Event = field(default_factory=asyncio.Event)
    error_ready: asyncio.Event = field(default_factory=asyncio.Event)
    route_error: GFlowError | None = None
    refusal_error: GFlowError | None = None
    parse_error: WireFormatError | None = None
    observed_errors: list[GFlowError] = field(default_factory=lambda: [])
    submit_requests: list[tuple[Any, str]] = field(default_factory=lambda: [])
    seen_rpcs: set[str] = field(default_factory=lambda: set())
    lost_rpc: str | None = None

    def note_record(self, rec: GenerationRecord, *, now: float) -> bool:
        added = self.tracker.register(rec, now=now)
        self.first_seen.set()
        if len(self.tracker.order) >= self.tracker.expected_count:
            self.all_seen.set()
        self.changed.set()
        return added

    def fail_with(self, err: GFlowError, *, is_route: bool = False) -> None:
        if is_route:
            if self.route_error is None:
                self.route_error = err
        elif self.refusal_error is None:
            self.refusal_error = err
        self.error_ready.set()
        self.changed.set()


async def _flush_on_started(
    tracker: _MultiClipTracker,
    on_started: VideoStartedCallback | None,
    project_id: str | None,
) -> None:
    if on_started is None:
        return
    for mid in list(tracker.order):
        if mid in tracker.started_sent:
            continue
        tracker.started_sent.add(mid)
        rec = tracker.records[mid]
        started = VideoStarted(
            media_id=rec.media_id,
            project_id=project_id or rec.project_id,
            flow_operation_id=rec.workflow_id,
            workflow_id=rec.workflow_id,
        )
        maybe = on_started(started)
        if asyncio.iscoroutine(maybe):
            await maybe


def _adopt_from_payload(
    state: _BatchWatchState,
    rid: str,
    payload: Any,
    adopt_ids: tuple[str, ...],
) -> None:
    if not adopt_ids:
        return
    now = time.monotonic()
    for rec in generation_records_carrying(rid, payload, adopt_ids):
        if rec.media_id not in state.tracker.records:
            log.info(
                "migrated.batch_video_adopted",
                rpc=rid,
                media_id=rec.media_id,
                workflow_id=rec.workflow_id,
                status=rec.status,
            )
        state.note_record(rec, now=now)


def _handle_submit_frame(
    state: _BatchWatchState,
    rid: str,
    payload: Any,
    *,
    now: float,
) -> None:
    recs = generation_records(rid, payload)
    if not recs:
        if not state.tracker.order and state.parse_error is None:
            state.parse_error = WireFormatError(
                detail=(
                    f"batchexecute {rid}: no generation record "
                    "([uuid, uuid, uuid, _, _, [details…], …]) in the reply"
                ),
                route=f"batchexecute:{rid}",
            )
            state.error_ready.set()
        return
    for rec in recs:
        if state.note_record(rec, now=now):
            log.info(
                "migrated.submit_observed",
                rpc=rid,
                workflow_id=rec.workflow_id,
                media_id=rec.media_id,
                status=rec.status,
                batch_index=len(state.tracker.order),
                expected_count=state.tracker.expected_count,
            )


def _handle_status_frame(
    state: _BatchWatchState,
    rid: str,
    payload: Any,
    *,
    now: float,
    adopt_ids: tuple[str, ...],
) -> None:
    if len(state.tracker.order) < state.tracker.expected_count and adopt_ids:
        _adopt_from_payload(state, rid, payload, adopt_ids)
    updated = False
    for rec in generation_records(rid, payload):
        if state.tracker.update_known(rec, now=now):
            updated = True
            log.info(
                "migrated.status",
                rpc=rid,
                media_id=rec.media_id,
                status=rec.status,
                bytes=rec.size_bytes,
            )
    if updated:
        if len(state.tracker.order) >= state.tracker.expected_count:
            state.all_seen.set()
        state.changed.set()


def _handle_batch_frames(
    state: _BatchWatchState,
    text: str,
    *,
    submit_rpcs: tuple[str, ...],
    adopt_ids: tuple[str, ...],
) -> None:
    refusal = _submit_refusal(text, submit_rpcs)
    if refusal is not None and not state.tracker.order:
        state.fail_with(refusal)
        return
    now = time.monotonic()
    for rid, payload in parse_frames(text):
        state.seen_rpcs.add(rid)
        if rid in submit_rpcs:
            _handle_submit_frame(state, rid, payload, now=now)
        elif rid in STATUS_RPCS:
            _handle_status_frame(state, rid, payload, now=now, adopt_ids=adopt_ids)


def _record_lost_submit(
    state: _BatchWatchState,
    response: Any,
    rpcid: str | None,
    *,
    submit_rpcs: tuple[str, ...],
    adopt_ids: tuple[str, ...],
) -> None:
    sent = getattr(response, "request", None)
    sub = (
        rpcid
        if rpcid in submit_rpcs
        else next((r for q, r in state.submit_requests if q is sent), None)
    )
    if sub is None or state.tracker.order:
        return
    if adopt_ids:
        state.lost_rpc = sub
        log.warning(
            "migrated.video_submit_reply_lost",
            rpc=sub,
            note="batch submit reply lost; following clips via uploaded frame ids, no re-submit",
        )
        return
    state.fail_with(
        _outcome_unknown(
            f"migrated host: the {sub} submit reply arrived but its body was unreadable"
        )
    )


async def _wait_event(event: asyncio.Event, timeout_s: float) -> bool:
    if event.is_set():
        return True
    try:
        await asyncio.wait_for(event.wait(), timeout=max(0.01, timeout_s))
        return True
    except TimeoutError:
        return event.is_set()


async def _await_any(events: tuple[asyncio.Event, ...], timeout_s: float) -> None:
    if any(ev.is_set() for ev in events):
        return
    tasks = [asyncio.create_task(ev.wait()) for ev in events]
    try:
        await asyncio.wait(
            tasks, timeout=max(0.01, timeout_s), return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()


async def _await_batch_submit(
    state: _BatchWatchState,
    page: Page,
    *,
    submit_budget_s: float,
    deadline: float,
    project_id: str | None,
    adopt_ids: tuple[str, ...],
    submit_rpcs: tuple[str, ...],
) -> None:
    await _await_any((state.first_seen, state.error_ready), submit_budget_s)
    if state.first_seen.is_set() and not state.all_seen.is_set() and not state.error_ready.is_set():
        grace = min(MULTI_SUBMIT_GRACE_S, max(0.0, deadline - time.monotonic()))
        if grace > 0.0:
            await _await_any((state.all_seen, state.error_ready), grace)
    if state.lost_rpc and not state.tracker.order and not state.error_ready.is_set() and project_id:
        listing = await read_project_listing(
            page,
            project_id,
            budget_s=max(0.0, min(LISTING_READ_BUDGET_S, deadline - time.monotonic())),
        )
        if listing is not None:
            _adopt_from_payload(state, LISTING_RPC, listing, adopt_ids)
    if state.lost_rpc and not state.tracker.order and not state.error_ready.is_set():
        rem = max(0.0, deadline - time.monotonic())
        if rem > 0.0:
            await _await_any((state.first_seen, state.error_ready), rem)
    _raise_if_submit_failed(
        state,
        submit_budget_s=submit_budget_s,
        project_id=project_id,
        submit_rpcs=submit_rpcs,
    )


def _raise_if_submit_failed(
    state: _BatchWatchState,
    *,
    submit_budget_s: float,
    project_id: str | None,
    submit_rpcs: tuple[str, ...],
) -> None:
    if state.route_error is not None:
        raise state.route_error
    if state.observed_errors:
        raise state.observed_errors[0]
    if state.refusal_error is not None and not state.tracker.order:
        raise state.refusal_error
    if state.parse_error is not None and not state.tracker.order:
        raise state.parse_error
    if state.tracker.order:
        return
    if state.lost_rpc:
        where = MIGRATED_PROJECT_URL.format(project_id=project_id) if project_id else "Flow"
        raise _outcome_unknown(
            f"migrated host: the {state.lost_rpc} submit reply arrived but its body was "
            f"unreadable, and no status reply named a clip carrying this run's upload; "
            f"it may still finish in {where}"
        )
    seen = ", ".join(sorted(state.seen_rpcs)) or "none"
    raise _outcome_unknown(
        f"migrated host: no {'/'.join(submit_rpcs)} reply within "
        f"{submit_budget_s:.0f}s of clicking submit (rpcs seen: {seen})"
    )


async def _await_batch_terminal(
    state: _BatchWatchState,
    page: Page,
    *,
    deadline: float,
    project_id: str | None,
    on_started: VideoStartedCallback | None,
    adopt_ids: tuple[str, ...],
) -> list[GenerationRecord]:
    tracker = state.tracker
    while True:
        await _flush_on_started(tracker, on_started, project_id)
        now = time.monotonic()
        if tracker.all_settled(now=now, url_grace_s=RESULT_URL_GRACE_S):
            break
        remaining = deadline - now
        if remaining <= 0.0:
            break
        grace_wait = tracker.next_grace_wait(now=now, url_grace_s=RESULT_URL_GRACE_S)
        wait_slice = min(remaining, grace_wait) if grace_wait is not None else min(remaining, 5.0)
        state.changed.clear()
        await _wait_event(state.changed, max(0.05, wait_slice))

    # Best-effort listing refresh if any clip stayed non-terminal at deadline.
    if not tracker.all_terminal() and project_id:
        listing = await read_project_listing(
            page, project_id, budget_s=min(LISTING_READ_BUDGET_S, 15.0)
        )
        if listing is not None:
            now = time.monotonic()
            if len(tracker.order) < tracker.expected_count and adopt_ids:
                _adopt_from_payload(state, LISTING_RPC, listing, adopt_ids)
            for rec in generation_records(LISTING_RPC, listing):
                tracker.update_known(rec, now=now)
            await _flush_on_started(tracker, on_started, project_id)

    final_records = tracker.ordered_records()
    if not tracker.all_terminal() and not any(r.is_done for r in final_records):
        ids = ", ".join(r.workflow_id for r in final_records)
        raise _outcome_unknown(
            f"migrated host: batch generations ({ids}) were not terminal within the poll timeout"
        )
    for idx, rec in enumerate(final_records, start=1):
        log.info(
            "migrated.batch_result",
            candidate_index=idx,
            media_id=rec.media_id,
            status=rec.status,
            done=rec.is_done,
            url_host=urlsplit(rec.video_url).hostname if rec.video_url else None,
            bytes=rec.size_bytes,
        )
    return final_records


def _check_submit_body(
    rpcid: str | None,
    body: str,
    *,
    submit_rpcs: tuple[str, ...],
    expect_media_id: str | None,
    expect_end_media_id: str | None,
    expect_reference_ids: tuple[str, ...],
) -> str | None:
    if rpcid is None or rpcid not in submit_rpcs:
        return None
    return _submit_body_problem(
        rpcid,
        body,
        expect_media_id=expect_media_id,
        expect_end_media_id=expect_end_media_id,
        expect_reference_ids=expect_reference_ids,
    )


def _make_request_listener(
    state: _BatchWatchState,
    *,
    submit_rpcs: tuple[str, ...],
    expect_media_id: str | None,
    expect_end_media_id: str | None,
    expect_reference_ids: tuple[str, ...],
) -> Any:
    def on_request(request: Any) -> None:
        url = str(getattr(request, "url", ""))
        if "batchexecute" not in url or state.route_error is not None:
            return
        body = request_body(request)
        rpcid = url_rpcid(url) or body_rpcid(body)
        if rpcid is None:
            return
        state.seen_rpcs.add(rpcid)
        if rpcid not in submit_rpcs:
            return
        state.submit_requests.append((request, rpcid))
        try:
            problem = _check_submit_body(
                rpcid,
                body,
                submit_rpcs=submit_rpcs,
                expect_media_id=expect_media_id,
                expect_end_media_id=expect_end_media_id,
                expect_reference_ids=expect_reference_ids,
            )
        except Exception:  # noqa: BLE001 - listener must never raise; route guard judges too
            return
        if problem is not None:
            state.observed_errors.append(
                WireFormatError(detail=problem, route=f"batchexecute:{rpcid}")
            )

    return on_request


def _make_response_listener(
    state: _BatchWatchState,
    *,
    submit_rpcs: tuple[str, ...],
    adopt_ids: tuple[str, ...],
) -> Any:
    async def on_response(response: Any) -> None:
        url = str(getattr(response, "url", ""))
        if "batchexecute" not in url:
            return
        rpcid = url_rpcid(url)
        if rpcid is not None:
            state.seen_rpcs.add(rpcid)
            if rpcid not in submit_rpcs and rpcid not in STATUS_RPCS:
                return
        try:
            text = await response.text()
        except Exception:  # noqa: BLE001 - dropped Chrome reply body
            _record_lost_submit(
                state, response, rpcid, submit_rpcs=submit_rpcs, adopt_ids=adopt_ids
            )
            return
        _handle_batch_frames(state, text, submit_rpcs=submit_rpcs, adopt_ids=adopt_ids)

    return on_response


def _make_route_guard(
    state: _BatchWatchState,
    *,
    submit_rpcs: tuple[str, ...],
    expect_media_id: str | None,
    expect_end_media_id: str | None,
    expect_reference_ids: tuple[str, ...],
) -> Any:
    async def guard(route: Any, request: Any) -> None:
        rpcid: str | None = None
        try:
            body = request_body(request)
            rpcid = url_rpcid(str(getattr(request, "url", ""))) or body_rpcid(body)
            problem = _check_submit_body(
                rpcid,
                body,
                submit_rpcs=submit_rpcs,
                expect_media_id=expect_media_id,
                expect_end_media_id=expect_end_media_id,
                expect_reference_ids=expect_reference_ids,
            )
        except Exception as exc:  # noqa: BLE001 - unsettled route hangs page
            problem = (
                f"migrated host: the {rpcid} submit could not be checked ({type(exc).__name__})"
                if rpcid is not None and rpcid in submit_rpcs
                else None
            )
        if problem is None:
            await route.fallback()
            return
        await route.abort()
        log.warning("migrated.submit_refused_before_send", rpc=rpcid)
        state.fail_with(_aborted_before_send(problem, str(rpcid)), is_route=True)

    return guard


def _make_batch_listeners(
    state: _BatchWatchState,
    *,
    submit_rpcs: tuple[str, ...],
    expect_media_id: str | None,
    expect_end_media_id: str | None,
    expect_reference_ids: tuple[str, ...],
    adopt_ids: tuple[str, ...],
) -> tuple[Any, Any, Any]:
    return (
        _make_request_listener(
            state,
            submit_rpcs=submit_rpcs,
            expect_media_id=expect_media_id,
            expect_end_media_id=expect_end_media_id,
            expect_reference_ids=expect_reference_ids,
        ),
        _make_response_listener(state, submit_rpcs=submit_rpcs, adopt_ids=adopt_ids),
        _make_route_guard(
            state,
            submit_rpcs=submit_rpcs,
            expect_media_id=expect_media_id,
            expect_end_media_id=expect_end_media_id,
            expect_reference_ids=expect_reference_ids,
        ),
    )


async def submit_and_observe_batch(
    composer: MigratedComposer,
    page: Page,
    *,
    expected_count: int,
    poll_timeout_s: float,
    on_started: VideoStartedCallback | None,
    project_id: str | None,
    expect_media_id: str | None = None,
    expect_end_media_id: str | None = None,
    expect_reference_ids: tuple[str, ...] = (),
    adopt_ids: tuple[str, ...] = (),
    submit_rpcs: tuple[str, ...] = SUBMIT_RPCS,
) -> list[GenerationRecord]:
    """Submit a ``--count 2..4`` batch and observe every clip until terminal state."""
    state = _BatchWatchState(tracker=_MultiClipTracker(expected_count=expected_count))
    on_request, on_response, guard = _make_batch_listeners(
        state,
        submit_rpcs=submit_rpcs,
        expect_media_id=expect_media_id,
        expect_end_media_id=expect_end_media_id,
        expect_reference_ids=expect_reference_ids,
        adopt_ids=adopt_ids,
    )
    page.on("response", on_response)
    page.on("request", on_request)
    try:
        await page.route(BATCHEXECUTE_GLOB, guard)
        submit = page.locator("button").filter(has=_ligature(page, "arrow_forward")).first
        await composer._await_submit_enabled(
            page,
            submit,
            missing=(
                "migrated host: the submit button (arrow_forward) is missing "
                "after the prompt was typed (host=migrated)"
            ),
            disabled=(
                "migrated host: the submit button (arrow_forward) stayed disabled for "
                f"{SUBMIT_ENABLE_BUDGET_S:.0f}s after the prompt was typed (host=migrated)"
            ),
        )
        deadline = time.monotonic() + poll_timeout_s
        await composer._click(page, submit, named=SUBMIT_BUTTON, timeout=5000)
        log.info("migrated.submit_clicked", expected_count=expected_count)
        await _await_batch_submit(
            state,
            page,
            submit_budget_s=min(60.0, poll_timeout_s),
            deadline=deadline,
            project_id=project_id,
            adopt_ids=adopt_ids,
            submit_rpcs=submit_rpcs,
        )
        return await _await_batch_terminal(
            state,
            page,
            deadline=deadline,
            project_id=project_id,
            on_started=on_started,
            adopt_ids=adopt_ids,
        )
    finally:
        page.remove_listener("response", on_response)
        page.remove_listener("request", on_request)
        try:
            await page.unroute(BATCHEXECUTE_GLOB, guard)
        except Exception as exc:  # noqa: BLE001 - must not mask real outcome
            log.debug("migrated.submit_unroute_failed", error=type(exc).__name__)


async def download_batch_results(
    composer: MigratedComposer,
    page: Page,
    records: list[GenerationRecord],
    *,
    project_id: str,
    save: ClipSaver,
    download: bool,
) -> VideoResult:
    """Download every completed clip in ``records`` without discarding partial successes."""
    results: list[VideoResult] = []
    done_count = 0
    downloaded_count = 0
    first_download_error: GFlowError | None = None

    for idx, rec in enumerate(records, start=1):
        local_path: Path | None = None
        err_msg = None if rec.is_done else f"migrated host reported status {rec.status}"
        succeeded = rec.is_done
        if download and rec.is_done:
            done_count += 1
            try:
                local_path = await composer.download(page, rec, save)
                downloaded_count += 1
            except GFlowError as exc:
                if first_download_error is None:
                    first_download_error = exc
                log.warning(
                    "migrated.candidate_download_failed",
                    candidate_index=idx,
                    media_id=rec.media_id,
                    error_class=type(exc).__name__,
                )
                succeeded = False
                err_msg = exc.detail
        status = VideoStatus(
            media_id=rec.media_id,
            status=(
                "MEDIA_GENERATION_STATUS_SUCCESSFUL"
                if succeeded
                else "MEDIA_GENERATION_STATUS_FAILED"
            ),
            error_message=err_msg,
        )
        results.append(
            VideoResult(
                status=status,
                local_path=local_path,
                project_id=project_id,
                flow_operation_id=rec.workflow_id,
                workflow_id=rec.workflow_id,
            )
        )

    if download and done_count > 0 and downloaded_count == 0 and first_download_error is not None:
        raise first_download_error
    return replace(results[0], extra_results=tuple(results[1:]))


async def run_video_batch(
    composer: MigratedComposer,
    page: Page,
    *,
    expected_count: int,
    poll_timeout_s: float,
    on_started: VideoStartedCallback | None,
    project_id: str,
    save: ClipSaver,
    download: bool,
    expect_media_id: str | None = None,
    expect_end_media_id: str | None = None,
    expect_reference_ids: tuple[str, ...] = (),
    adopt_ids: tuple[str, ...] = (),
) -> VideoResult:
    """Observe all ``expected_count`` clips and download every succeeded candidate."""
    records = await submit_and_observe_batch(
        composer,
        page,
        expected_count=expected_count,
        poll_timeout_s=poll_timeout_s,
        on_started=on_started,
        project_id=project_id,
        expect_media_id=expect_media_id,
        expect_end_media_id=expect_end_media_id,
        expect_reference_ids=expect_reference_ids,
        adopt_ids=adopt_ids,
    )
    return await download_batch_results(
        composer,
        page,
        records,
        project_id=project_id,
        save=save,
        download=download,
    )
