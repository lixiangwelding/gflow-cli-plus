"""Tests for the 2026-11-20/21 Flow Agent UI selector cascade and side-panel recovery.

Covers both pure-offline unit behaviour and real-CSS-engine checks (using system Chrome
when bundled Playwright Chromium is absent).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from structlog.testing import capture_logs

from gflow_cli.api.transports import _mc_common, _mc_editor_gate
from gflow_cli.api.transports.migrated_composer import MigratedComposer

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import AsyncIterator

    from playwright.async_api import Page


# --- Selector cascade invariants (pure offline) -----------------------------------------


def test_selector_cascades_preserve_legacy_and_append_2026_11_selectors() -> None:
    assert "button.agent-mode-chip[aria-pressed='true']" in _mc_common.AGENT_MODE_CHIP
    assert "flow-agent-mode-switch button[aria-pressed='true']" in _mc_common.AGENT_MODE_CHIP

    assert "button.agent-mode-chip" in _mc_common.AGENT_MODE_CHIP_ANY
    assert "flow-agent-mode-switch button" in _mc_common.AGENT_MODE_CHIP_ANY

    assert "tune" in _mc_common.AGENT_ONLY_ANCHOR
    assert (
        "flow-agent-panel button:has(mat-icon:text-is('settings_2'))"
        in _mc_common.AGENT_ONLY_ANCHOR
    )
    assert (
        "flow-creative-agent-prompt-box button:has(mat-icon:text-is('settings_2'))"
        in _mc_common.AGENT_ONLY_ANCHOR
    )

    assert _mc_common.AGENT_PANEL_CLOSE_LIGATURES == ("close", "right_panel_close")
    assert "Get started" in _mc_common.DIALOG_DISMISS_CTA_CANDIDATES
    assert "开始使用" in _mc_common.DIALOG_DISMISS_CTA_CANDIDATES
    # Rights-consent buttons must never appear in changelog dismissal candidates.
    for forbidden in ("I agree", "Agree", "我同意"):
        assert forbidden not in _mc_common.DIALOG_DISMISS_CTA_CANDIDATES


@pytest.mark.asyncio
async def test_exit_agent_mode_fail_closed_when_probe_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collapse_spy = AsyncMock(return_value="right_panel_close")
    monkeypatch.setattr(MigratedComposer, "_agent_chip_pressed", AsyncMock(return_value=None))
    monkeypatch.setattr(MigratedComposer, "_collapse_agent_panel", collapse_spy)

    state, err = await MigratedComposer._exit_agent_mode(MagicMock())  # pyright: ignore[reportPrivateUsage]

    assert state == "absent"
    assert err is None
    collapse_spy.assert_not_awaited()


@pytest.mark.asyncio
async def test_exit_agent_mode_returns_absent_when_collapsed_panel_mounts_no_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truly agent-only account: panel collapses, no switch appears -> absent -> agent_only."""
    monkeypatch.setattr(_mc_editor_gate, "AGENT_PANEL_COLLAPSE_SETTLE_S", 0.05)
    monkeypatch.setattr(
        MigratedComposer, "_agent_chip_pressed", AsyncMock(side_effect=[False, False])
    )
    monkeypatch.setattr(
        MigratedComposer, "_collapse_agent_panel", AsyncMock(return_value="right_panel_close")
    )
    page = MagicMock()
    page.locator.return_value.first.wait_for = AsyncMock(side_effect=TimeoutError("no switch"))

    state, err = await MigratedComposer._exit_agent_mode(page)  # pyright: ignore[reportPrivateUsage]

    assert state == "absent"
    assert err is None


# --- Real CSS-engine verification against captured 2026-11-21 markup --------------------


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def _real_browser() -> AsyncIterator[Any]:
    playwright_api = pytest.importorskip("playwright.async_api")
    async with playwright_api.async_playwright() as pw:
        browser = None
        for kwargs in ({}, {"channel": "chrome"}):
            try:
                browser = await pw.chromium.launch(headless=True, **kwargs)
                break
            except Exception:
                continue
        if browser is None:
            pytest.skip("neither bundled Chromium nor system Chrome available")
        try:
            yield browser
        finally:
            await browser.close()


@pytest_asyncio.fixture(loop_scope="module")
async def real_page(_real_browser: Any) -> AsyncIterator[Page]:
    ctx = await _real_browser.new_context()
    try:
        yield await ctx.new_page()
    finally:
        await ctx.close()


async def _run_ensure_editor(page: Page, html: str, timeout_s: float = 10.0) -> str:
    async def _serve(route: Any) -> None:
        await route.fulfill(body=html, content_type="text/html; charset=utf-8")

    await page.route("https://flow.google.com/**", _serve)
    return await MigratedComposer().ensure_editor(page, "p-cascade", timeout_s=timeout_s)


@pytest.mark.asyncio(loop_scope="module")
async def test_real_dom_side_panel_collapse_then_spark_switch_unlocks_classic(
    real_page: Page,
) -> None:
    from tests.api.transports.test_agent_only_composer import SIDE_PANEL_WITH_SWITCH

    with capture_logs() as logs:
        kind = await _run_ensure_editor(real_page, SIDE_PANEL_WITH_SWITCH)

    assert kind == "classic"
    assert await real_page.evaluate("window.clicks") == {"collapse": 1, "switch": 1}
    assert any(
        e["event"] == "migrated.agent_panel_closed"
        and e.get("matched_label") == "right_panel_close"
        for e in logs
    )
    assert any(
        e["event"] == "migrated.agent_mode_exited"
        and e.get("matched_label") == "flow-agent-mode-switch"
        for e in logs
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_real_dom_floating_spark_switch_unlocks_classic(real_page: Page) -> None:
    from tests.api.transports.test_agent_only_composer import FLOATING_AGENT_SWITCH

    with capture_logs() as logs:
        kind = await _run_ensure_editor(real_page, FLOATING_AGENT_SWITCH)

    assert kind == "classic"
    assert await real_page.evaluate("window.clicks") == {"switch": 1}
    assert any(
        e["event"] == "migrated.agent_mode_exited"
        and e.get("matched_label") == "flow-agent-mode-switch"
        for e in logs
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_real_dom_healthy_classic_with_spark_off_never_clicks_switch(
    real_page: Page,
) -> None:
    from tests.api.transports.test_agent_only_composer import CLASSIC_WITH_UNPRESSED_SWITCH

    kind = await _run_ensure_editor(real_page, CLASSIC_WITH_UNPRESSED_SWITCH)

    assert kind == "classic"
    assert await real_page.evaluate("window.clicks") == {"switch": 0}


@pytest.mark.asyncio(loop_scope="module")
async def test_real_dom_agent_only_settings_2_without_switch_detected(
    real_page: Page, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.api.transports.test_agent_only_composer import AGENT_ONLY_SETTINGS_2

    monkeypatch.setattr(_mc_editor_gate, "AGENT_ONLY_SETTLE_S", 0.1)
    monkeypatch.setattr(_mc_editor_gate, "AGENT_ONLY_EARLY_S", 0.3)

    kind = await _run_ensure_editor(real_page, AGENT_ONLY_SETTINGS_2)

    assert kind == "agent_only"


@pytest.mark.asyncio(loop_scope="module")
async def test_real_dom_changelog_modal_dismissed_via_zh_cta(real_page: Page) -> None:
    from tests.api.transports.test_agent_only_composer import CHANGELOG_MODAL_CTA

    with capture_logs() as logs:
        kind = await _run_ensure_editor(real_page, CHANGELOG_MODAL_CTA)

    assert kind == "classic"
    assert await real_page.evaluate("window.ctaClicked") == 1
    assert any(
        e["event"] == "migrated.dialog_dismissed"
        and e.get("via") == "cta"
        and e.get("matched_label") == "开始使用"
        for e in logs
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_real_dom_upload_rights_dialog_not_clicked_by_dismiss_dialog(
    real_page: Page,
) -> None:
    """Safety guard: _dismiss_dialog must NEVER click an 'I agree' / '我同意' consent button."""
    html = """
    <div role="dialog" id="rights">
      <p>Rights confirmation</p>
      <button id="agree">我同意</button>
    </div>
    <script>
      window.agreed = 0;
      document.getElementById('agree').onclick = () => { window.agreed++; };
    </script>
    """
    await real_page.set_content(html)
    await MigratedComposer._dismiss_dialog(real_page)  # pyright: ignore[reportPrivateUsage]
    await asyncio.sleep(0.05)
    assert await real_page.evaluate("window.agreed") == 0
