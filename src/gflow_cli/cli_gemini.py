"""`gflow gemini` command group — Google Gemini Web (Ultra) browser automation.

Subcommands:

* ``login``  — interactive login in an isolated Chrome profile (``gemini_profile_<name>``).
* ``status`` — inspect isolated profile existence, lock state, and optional live session check.
* ``chat``   — send a text/file prompt to Gemini Web (with optional JSON block extraction).
* ``image``  — generate or reference-edit an image on Gemini Web and save high-res output.

All Gemini profiles live in a dedicated ``gemini_profile_<name>`` namespace with an
independent per-profile lock directory so Flow profiles (``profile_*``) are never touched.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import uuid
from collections.abc import Callable, Coroutine
from datetime import date, datetime
from pathlib import Path
from typing import Any

import click
import structlog
from rich.console import Console
from rich.markup import escape

from gflow_cli import json_output
from gflow_cli._cli_helpers import safe_path_text, spends_credits
from gflow_cli.config import get_settings
from gflow_cli.errors import GFlowError, exit_code_for

console = Console()
err_console = Console(stderr=True)
logger = structlog.get_logger(__name__)

DEFAULT_GEMINI_PROFILE = "gemini_ultra"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_positive_timeout(
    _ctx: click.Context, param: click.Parameter, value: int
) -> int:
    if value <= 0:
        raise click.BadParameter("must be a positive integer (seconds).", param=param)
    return value


def _validate_non_negative_int(
    _ctx: click.Context, param: click.Parameter, value: int
) -> int:
    if value < 0:
        raise click.BadParameter("must be a non-negative integer (>= 0).", param=param)
    return value


def _validate_optional_positive_int(
    _ctx: click.Context, param: click.Parameter, value: int | None
) -> int | None:
    if value is not None and value <= 0:
        raise click.BadParameter("must be a positive integer (> 0).", param=param)
    return value


def _validate_positive_float(
    _ctx: click.Context, param: click.Parameter, value: float
) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise click.BadParameter("must be a positive number (> 0).", param=param)
    return value


def _validate_aspect(
    _ctx: click.Context, param: click.Parameter, value: str
) -> str:
    cleaned = value.strip()
    if not cleaned or ":" not in cleaned:
        raise click.BadParameter(
            f"invalid aspect ratio {value!r}; expected W:H (e.g. 16:9, 9:16, 1:1).",
            param=param,
        )
    parts = cleaned.split(":", 1)
    if not all(p.strip().isdigit() and int(p.strip()) > 0 for p in parts):
        raise click.BadParameter(
            f"invalid aspect ratio {value!r}; width and height must be positive integers.",
            param=param,
        )
    return f"{int(parts[0].strip())}:{int(parts[1].strip())}"


def _parse_gap_range(
    _ctx: click.Context, param: click.Parameter, value: str
) -> tuple[float, float]:
    """Parse a ``MIN-MAX`` or single-number gap specification in seconds."""
    cleaned = value.strip()
    if not cleaned:
        raise click.BadParameter("gap must be 'MIN-MAX' or a non-negative number.", param=param)
    parts = cleaned.split("-")
    if len(parts) not in (1, 2) or any(not p.strip() for p in parts):
        raise click.BadParameter(
            f"invalid gap range {value!r}; expected 'MIN-MAX' (e.g. 25-45) or a single number.",
            param=param,
        )
    try:
        nums = [float(p.strip()) for p in parts]
    except ValueError as exc:
        raise click.BadParameter(
            f"invalid gap range {value!r}; values must be numbers.", param=param
        ) from exc
    if any(not math.isfinite(n) or n < 0.0 for n in nums):
        raise click.BadParameter(
            f"invalid gap range {value!r}; values must be finite and >= 0.", param=param
        )
    lo = nums[0]
    hi = nums[1] if len(nums) == 2 else nums[0]
    if lo > hi:
        raise click.BadParameter(
            f"invalid gap range {value!r}; minimum ({lo}) cannot exceed maximum ({hi}).",
            param=param,
        )
    return (lo, hi)


def _resolve_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    """Resolve mutually exclusive PROMPT argument vs ``--prompt-file``."""
    if prompt is not None and prompt_file is not None:
        raise click.UsageError("Provide either PROMPT or --prompt-file, not both.")
    if prompt is None and prompt_file is None:
        raise click.UsageError("Provide a PROMPT argument or --prompt-file PATH.")
    if prompt_file is not None:
        try:
            text = prompt_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise click.UsageError(f"Cannot read --prompt-file {prompt_file}: {exc}") from exc
        if not text.strip():
            raise click.UsageError(f"--prompt-file {prompt_file} is empty.")
        return text.strip()
    if prompt is None or not prompt.strip():
        raise click.UsageError("PROMPT must not be empty.")
    return prompt.strip()


def _validate_profile(profile: str) -> str:
    from gflow_cli.gemini import GeminiConfigError, validate_gemini_profile_name

    try:
        return validate_gemini_profile_name(profile)
    except (GeminiConfigError, ValueError) as exc:
        raise click.BadParameter(str(exc), param_hint="--profile") from exc


def _default_image_out_file() -> Path:
    settings = get_settings()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return (
        settings.output_dir
        / "gemini"
        / "images"
        / date.today().isoformat()
        / f"gemini_{stamp}_{short_id}.png"
    )


def _build_gemini_error_payload(exc: Any, *, cli_command: str) -> dict[str, Any]:
    err_dict = dict(exc.to_error_dict())
    err_dict.setdefault("ok", False)
    err_dict.setdefault("status", "fail")
    return {
        **err_dict,
        "command": cli_command,
        "error": err_dict,
    }


def _print_gemini_error(exc: Any) -> None:
    title = getattr(exc, "title", type(exc).__name__)
    code = getattr(exc, "code", "ERROR")
    detail = getattr(exc, "detail", None) or str(exc)
    hint = getattr(exc, "remediation_hint", None)
    err_console.print(
        f"[red]{escape(str(title))} ({escape(str(code))}):[/red] {escape(str(detail))}"
    )
    if hint:
        err_console.print(f"[yellow]-> {escape(str(hint))}[/yellow]")
    screenshot = getattr(exc, "screenshot_path", None)
    dom_path = getattr(exc, "dom_path", None)
    if screenshot:
        err_console.print(f"[dim]Diagnostic screenshot: {escape(str(screenshot))}[/dim]")
    if dom_path:
        err_console.print(f"[dim]Diagnostic DOM: {escape(str(dom_path))}[/dim]")


def _emit_gflow_error(exc: GFlowError, *, cli_command: str, as_json: bool) -> int:
    code = exit_code_for(exc)
    if as_json:
        payload = json_output.error_payload(exc)
        payload["ok"] = False
        payload["command"] = cli_command
        json_output.emit(payload)
    else:
        err_console.print(f"[red]{escape(exc.title)}:[/red] {escape(exc.detail or str(exc))}")
        if exc.remediation_hint:
            err_console.print(f"[yellow]-> {escape(exc.remediation_hint)}[/yellow]")
    return code


def _emit_unexpected_error(exc: Exception, *, cli_command: str, as_json: bool) -> int:
    if as_json:
        debug_exc = exc if get_settings().debug_traceback else None
        payload = json_output.unexpected_payload(debug=debug_exc)
        payload["ok"] = False
        payload["command"] = cli_command
        payload["error_code"] = "UNEXPECTED_ERROR"
        json_output.emit(payload)
    else:
        err_console.print(f"[red]Unexpected Gemini error:[/red] {escape(str(exc))}")
    return 1


def _emit_interrupted(*, cli_command: str, as_json: bool) -> int:
    if as_json:
        json_output.emit(
            {
                "ok": False,
                "status": "fail",
                "command": cli_command,
                "error_code": "INTERRUPTED",
                "exit_code": 130,
                "detail": "Interrupted by user.",
            }
        )
    else:
        err_console.print("[yellow]Interrupted.[/yellow]")
    return 130


def _run_gemini_coro(
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    *,
    cli_command: str,
    as_json: bool,
) -> Any:
    """Execute an async Gemini operation with typed error handling and JSON parity."""
    from gflow_cli.gemini import GeminiError, gemini_exit_code_for

    structlog.contextvars.bind_contextvars(cli_command=cli_command)
    try:
        return asyncio.run(coro_factory())
    except GeminiError as exc:
        code = gemini_exit_code_for(exc)
        if as_json:
            json_output.emit(_build_gemini_error_payload(exc, cli_command=cli_command))
        else:
            _print_gemini_error(exc)
        sys.exit(code)
    except GFlowError as exc:
        sys.exit(_emit_gflow_error(exc, cli_command=cli_command, as_json=as_json))
    except (KeyboardInterrupt, click.Abort):
        sys.exit(_emit_interrupted(cli_command=cli_command, as_json=as_json))
    except SystemExit:
        raise
    except Exception as exc:
        sys.exit(_emit_unexpected_error(exc, cli_command=cli_command, as_json=as_json))
    finally:
        structlog.contextvars.unbind_contextvars("cli_command")


def _print_warnings(warnings: list[str] | tuple[str, ...]) -> None:
    for warning in warnings:
        err_console.print(f"[yellow]Warning:[/yellow] {escape(str(warning))}")


def _print_status_risk(risk: dict[str, Any] | None) -> None:
    if not isinstance(risk, dict):
        return
    if risk.get("kill_switch_active"):
        ks_path = escape(str(risk.get("kill_switch_path", "")))
        console.print(f"  kill_switch     : [red]ACTIVE[/red] ({ks_path})")
    if risk.get("global_halted"):
        halt_reason = escape(str(risk.get("global_halt_reason", "")))
        console.print(f"  global_halted   : [red]True[/red] ({halt_reason})")
    if risk.get("quarantined"):
        q_reason = escape(str(risk.get("quarantine_reason", "")))
        console.print(f"  quarantined     : [yellow]True[/yellow] ({q_reason})")
    if risk.get("elevated"):
        console.print("  elevated_pacing : [yellow]True (45-90s gap)[/yellow]")
    if risk.get("burst_count_10m") is not None:
        cap = risk.get("burst_limit_10m", 8)
        console.print(f"  burst_10m       : {risk['burst_count_10m']}/{cap}")


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group()
def gemini() -> None:
    """Drive Google Gemini Web (Ultra) via an isolated Chrome profile.

    Provides ``login``, ``status``, ``chat``, ``image``, ``image-batch``,
    ``task-submit``, and ``task-harvest``.
    Uses a separate ``gemini_profile_<name>`` directory so Flow profiles are untouched.
    """


# ---------------------------------------------------------------------------
# login subcommand
# ---------------------------------------------------------------------------


@spends_credits(False)
@gemini.command(
    "login",
    short_help="Open an isolated Chrome window to sign in to Gemini Web.",
)
@click.option(
    "--profile",
    default=DEFAULT_GEMINI_PROFILE,
    show_default=True,
    envvar="GFLOW_CLI_GEMINI_PROFILE",
    help="Isolated Gemini profile name (stored under gemini_profile_<name>).",
)
@click.option(
    "--timeout",
    default=300,
    show_default=True,
    type=int,
    callback=_validate_positive_timeout,
    help="Maximum seconds to wait for manual login before timing out.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON on stdout.")
def gemini_login(profile: str, timeout: int, as_json: bool) -> None:
    """Launch a fresh headed Chrome window to https://gemini.google.com/app for sign-in."""
    from gflow_cli.gemini import login_profile

    profile_name = _validate_profile(profile)
    if not as_json:
        console.print(
            f"Launching isolated Chrome for Gemini profile [bold]{escape(profile_name)}[/bold]..."
        )
        console.print(
            "[dim]Sign in to your Google Ultra account in the opened window. "
            "This command will detect when Gemini is ready and save the session.[/dim]"
        )

    result = _run_gemini_coro(
        lambda: login_profile(profile_name=profile_name, timeout_seconds=timeout),
        cli_command="gemini login",
        as_json=as_json,
    )
    data = result.to_dict()
    ok = bool(data.get("ok") and data.get("logged_in"))
    if as_json:
        json_output.emit(
            {
                "status": "ok" if ok else "fail",
                "command": "gemini login",
                **data,
            }
        )
    else:
        if ok:
            console.print(
                f"[green]Gemini session saved.[/green] Profile: [bold]{escape(profile_name)}[/bold]"
            )
            console.print(f"  profile_dir: {escape(str(data.get('profile_dir', '')))}")
            if data.get("account_hint"):
                console.print(f"  account    : {escape(str(data['account_hint']))}")
            if data.get("chat_url"):
                console.print(f"  url        : {escape(str(data['chat_url']))}")
        else:
            console.print(
                f"[yellow]Gemini login did not complete:[/yellow] "
                f"{escape(str(data.get('detail') or 'not logged in'))}"
            )
        _print_warnings(data.get("warnings") or [])

    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------------------
# status subcommand
# ---------------------------------------------------------------------------


@spends_credits(False)
@gemini.command(
    "status",
    short_help="Inspect isolated Gemini profile state and optional live login.",
)
@click.option(
    "--profile",
    default=DEFAULT_GEMINI_PROFILE,
    show_default=True,
    envvar="GFLOW_CLI_GEMINI_PROFILE",
    help="Isolated Gemini profile name.",
)
@click.option(
    "--verify-live",
    is_flag=True,
    default=False,
    help="Launch Chrome against https://gemini.google.com/app to verify live login state.",
)
@click.option(
    "--headed/--headless",
    default=False,
    show_default=True,
    help="Browser visibility when --verify-live is enabled.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON on stdout.")
def gemini_status(profile: str, verify_live: bool, headed: bool, as_json: bool) -> None:
    """Check isolated profile existence, lock state, and optionally live session health."""
    from gflow_cli.gemini import check_profile_status

    profile_name = _validate_profile(profile)
    result = _run_gemini_coro(
        lambda: check_profile_status(
            profile_name=profile_name,
            verify_live=verify_live,
            headless=not headed,
        ),
        cli_command="gemini status",
        as_json=as_json,
    )
    data = result.to_dict()
    ok = bool(data.get("ok"))
    if as_json:
        json_output.emit(
            {
                "status": "ok" if ok else "fail",
                "command": "gemini status",
                **data,
            }
        )
    else:
        console.print(f"Gemini profile: [bold]{escape(profile_name)}[/bold]")
        console.print(f"  profile_dir     : {escape(str(data.get('profile_dir', '')))}")
        console.print(f"  exists          : {data.get('exists')}")
        console.print(f"  locked          : {data.get('locked')}")
        if data.get("lock_owner_pid") is not None:
            console.print(f"  lock_owner_pid  : {data['lock_owner_pid']}")
        console.print(f"  chrome_available: {data.get('chrome_available')}")
        if data.get("verified_live"):
            console.print(f"  logged_in       : {data.get('logged_in')}")
        if data.get("account_hint"):
            console.print(f"  account         : {escape(str(data['account_hint']))}")
        if data.get("detail"):
            console.print(f"  detail          : {escape(str(data['detail']))}")
        _print_status_risk(data.get("risk"))
        if ok:
            console.print("[green]Status: ready[/green]")
        elif data.get("locked"):
            console.print(
                "[yellow]Status: locked.[/yellow] Another Gemini process is using this profile."
            )
        else:
            console.print(
                f"[yellow]Status: not ready.[/yellow] "
                f"Run [bold]gflow gemini login --profile {escape(profile_name)}[/bold]."
            )
        _print_warnings(data.get("warnings") or [])

    if not ok:
        sys.exit(11 if data.get("locked") else 1)


# ---------------------------------------------------------------------------
# chat subcommand
# ---------------------------------------------------------------------------


@spends_credits(False)
@gemini.command(
    "chat",
    short_help="Send a prompt (and optional attachments) to Gemini Web.",
)
@click.argument("prompt", required=False, default=None)
@click.option(
    "--prompt-file",
    "prompt_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
    help="Read prompt text from a UTF-8 file (mutually exclusive with PROMPT).",
)
@click.option(
    "--attach",
    "attachments",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
    help="Attach a local file to the prompt (repeatable).",
)
@click.option("--model", default=None, help="Optional Gemini Web model name to select best-effort.")
@click.option(
    "--chat-url",
    "chat_url",
    default=None,
    help="Continue an existing Gemini conversation URL (mutually exclusive with --new-chat).",
)
@click.option(
    "--new-chat",
    "new_chat",
    is_flag=True,
    default=False,
    help="Force starting a fresh conversation at https://gemini.google.com/app.",
)
@click.option(
    "--extract-json",
    "extract_json",
    is_flag=True,
    default=False,
    help="Extract a JSON object/array from the response text.",
)
@click.option(
    "--out-file",
    "out_file",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write response text (or extracted JSON when --extract-json is set) to PATH.",
)
@click.option(
    "--profile",
    default=DEFAULT_GEMINI_PROFILE,
    show_default=True,
    envvar="GFLOW_CLI_GEMINI_PROFILE",
    help="Isolated Gemini profile name.",
)
@click.option(
    "--headed/--headless",
    default=True,
    show_default=True,
    help="Run real Chrome visibly (--headed, safer against Google WAF) or hidden (--headless).",
)
@click.option(
    "--timeout",
    default=180,
    show_default=True,
    type=int,
    callback=_validate_positive_timeout,
    help="Maximum seconds to wait for response generation.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON on stdout.")
def gemini_chat(
    prompt: str | None,
    prompt_file: Path | None,
    attachments: tuple[Path, ...],
    model: str | None,
    chat_url: str | None,
    new_chat: bool,
    extract_json: bool,
    out_file: Path | None,
    profile: str,
    headed: bool,
    timeout: int,
    as_json: bool,
) -> None:
    """Run a Gemini Web chat turn for scriptwriting or structured JSON output."""
    from gflow_cli.gemini import chat as run_gemini_chat

    resolved_prompt = _resolve_prompt(prompt, prompt_file)
    if chat_url and new_chat:
        raise click.UsageError("Cannot combine --chat-url with --new-chat.")
    profile_name = _validate_profile(profile)

    result = _run_gemini_coro(
        lambda: run_gemini_chat(
            resolved_prompt,
            profile=profile_name,
            attachments=list(attachments),
            model=model,
            chat_url=chat_url,
            new_chat=new_chat,
            extract_json=extract_json,
            out_file=out_file,
            headless=not headed,
            timeout_seconds=timeout,
        ),
        cli_command="gemini chat",
        as_json=as_json,
    )
    data = result.to_dict()
    ok = bool(data.get("ok"))
    if as_json:
        json_output.emit(
            {
                "status": "ok" if ok else "fail",
                "command": "gemini chat",
                **data,
            }
        )
    else:
        if extract_json and data.get("extracted_json") is not None:
            click.echo(json.dumps(data["extracted_json"], ensure_ascii=False, indent=2))
        else:
            click.echo(str(data.get("text") or ""))
        if data.get("out_file"):
            err_console.print(
                f"[dim]Saved output: {escape(safe_path_text(Path(str(data['out_file']))))}[/dim]"
            )
        if data.get("chat_url"):
            err_console.print(f"[dim]Chat URL: {escape(str(data['chat_url']))}[/dim]")
        _print_warnings(data.get("warnings") or [])

    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------------------
# image subcommand
# ---------------------------------------------------------------------------


@spends_credits(False)
@gemini.command(
    "image",
    short_help="Generate or reference-edit an image via Gemini Web.",
)
@click.argument("prompt", required=False, default=None)
@click.option(
    "--prompt-file",
    "prompt_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
    help="Read prompt text from a UTF-8 file (mutually exclusive with PROMPT).",
)
@click.option(
    "--ref",
    "refs",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
    help="Local reference image path for guided editing/consistency (repeatable).",
)
@click.option(
    "--out-file",
    "out_file",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination path for the downloaded image (auto-assigned if omitted).",
)
@click.option(
    "--aspect",
    default="16:9",
    show_default=True,
    callback=_validate_aspect,
    help="Target aspect ratio hint (e.g. 16:9, 9:16, 1:1).",
)
@click.option("--model", default=None, help="Optional Gemini Web model name to select best-effort.")
@click.option(
    "--chat-url",
    "chat_url",
    default=None,
    help="Optional Gemini conversation URL to continue.",
)
@click.option(
    "--strip-watermark/--no-strip-watermark",
    "strip_watermark",
    default=False,
    show_default=True,
    help="Optionally trim/clean the bottom-right Gemini sparkle watermark.",
)
@click.option(
    "--profile",
    default=DEFAULT_GEMINI_PROFILE,
    show_default=True,
    envvar="GFLOW_CLI_GEMINI_PROFILE",
    help="Isolated Gemini profile name.",
)
@click.option(
    "--headed/--headless",
    default=True,
    show_default=True,
    help="Run real Chrome visibly (--headed, safer against Google WAF) or hidden (--headless).",
)
@click.option(
    "--timeout",
    default=240,
    show_default=True,
    type=int,
    callback=_validate_positive_timeout,
    help="Maximum seconds to wait for image generation and download.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON on stdout.")
def gemini_image(
    prompt: str | None,
    prompt_file: Path | None,
    refs: tuple[Path, ...],
    out_file: Path | None,
    aspect: str,
    model: str | None,
    chat_url: str | None,
    strip_watermark: bool,
    profile: str,
    headed: bool,
    timeout: int,
    as_json: bool,
) -> None:
    """Generate a high-resolution image (or reference-guided edit) on Gemini Web."""
    from gflow_cli.gemini import generate_image as run_gemini_image

    resolved_prompt = _resolve_prompt(prompt, prompt_file)
    profile_name = _validate_profile(profile)
    target_out = out_file if out_file is not None else _default_image_out_file()

    result = _run_gemini_coro(
        lambda: run_gemini_image(
            resolved_prompt,
            out_file=target_out,
            profile=profile_name,
            ref_paths=list(refs),
            aspect=aspect,
            model=model,
            chat_url=chat_url,
            strip_watermark=strip_watermark,
            headless=not headed,
            timeout_seconds=timeout,
        ),
        cli_command="gemini image",
        as_json=as_json,
    )
    data = result.to_dict()
    ok = bool(data.get("ok"))
    saved_str = str(data.get("out_file") or target_out)
    if as_json:
        json_output.emit(
            {
                "status": "ok" if ok else "fail",
                "command": "gemini image",
                **data,
                "local_path": saved_str,
            }
        )
    else:
        if ok:
            console.print(
                f"[bold green]Saved:[/bold green] {escape(safe_path_text(Path(saved_str)))}"
            )
            dims = f"{data.get('width', '?')}x{data.get('height', '?')}"
            console.print(
                f"  size: {escape(dims)} ({escape(str(data.get('aspect_ratio') or aspect))})  "
                f"bytes: {data.get('bytes', 0)}  refs: {len(refs)}"
            )
            if data.get("watermark_stripped"):
                note = escape(str(data.get("watermark_note") or ""))
                console.print(f"  [dim]watermark: stripped ({note})[/dim]")
            if data.get("chat_url"):
                console.print(f"  [dim]chat_url: {escape(str(data['chat_url']))}[/dim]")
        else:
            err_console.print("[red]Gemini image generation did not succeed.[/red]")
        _print_warnings(data.get("warnings") or [])

    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------------------
# image-batch subcommand
# ---------------------------------------------------------------------------


def _print_image_batch_summary(data: dict[str, Any]) -> None:
    completed: list[Any] = data.get("completed") or []
    skipped: list[Any] = data.get("skipped") or []
    failed: list[Any] = data.get("failed") or []
    deferred: list[Any] = data.get("deferred") or []
    all_items: list[Any] = data.get("items") or []
    total = data.get("total", len(all_items))
    ok = bool(data.get("ok"))

    header_style = "bold green" if ok else "bold yellow"
    state_label = "completed" if ok else "stopped (partial)"
    console.print(
        f"[{header_style}]Gemini image-batch {state_label}:[/{header_style}] "
        f"total={total} completed={len(completed)} skipped={len(skipped)} "
        f"failed={len(failed)} deferred={len(deferred)}"
    )
    if data.get("chat_url"):
        console.print(f"  [dim]chat_url: {escape(str(data['chat_url']))}[/dim]")
    if data.get("stop_error_code"):
        stop_code = escape(str(data["stop_error_code"]))
        stop_detail = escape(str(data.get("stop_detail") or ""))
        err_console.print(f"[red]Stopped ({stop_code}):[/red] {stop_detail}")
    if data.get("retry_after_seconds") is not None:
        err_console.print(
            f"[yellow]Retry after: {data['retry_after_seconds']}s[/yellow]"
        )
    _print_warnings(data.get("warnings") or [])


@spends_credits(False)
@gemini.command(
    "image-batch",
    short_help="Generate a manifest of images sequentially inside one Chrome session.",
)
@click.option(
    "--manifest",
    "manifest_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
    help="Path to JSON manifest (list of items or {\"items\": [...]}).",
)
@click.option(
    "--profile",
    default=DEFAULT_GEMINI_PROFILE,
    show_default=True,
    envvar="GFLOW_CLI_GEMINI_PROFILE",
    help="Isolated Gemini profile name.",
)
@click.option(
    "--reuse-chat/--separate-chats",
    "reuse_chat",
    default=True,
    show_default=True,
    help="Keep one Gemini conversation across all shots (recommended) or start fresh per shot.",
)
@click.option(
    "--gap",
    "gap_range",
    default="25-45",
    show_default=True,
    callback=_parse_gap_range,
    help="Randomized pause in seconds between consecutive shots (e.g. '25-45' or '0').",
)
@click.option(
    "--resume/--no-resume",
    "resume",
    default=True,
    show_default=True,
    help="Skip shots whose output image already exists and passes validation.",
)
@click.option(
    "--aspect",
    default="16:9",
    show_default=True,
    callback=_validate_aspect,
    help="Default aspect ratio hint for manifest items that omit 'aspect'.",
)
@click.option(
    "--model",
    default=None,
    help="Optional default Gemini Web model name to select best-effort.",
)
@click.option(
    "--chat-url",
    "chat_url",
    default=None,
    help="Optional existing Gemini conversation URL to start the batch in.",
)
@click.option(
    "--strip-watermark/--no-strip-watermark",
    "strip_watermark",
    default=False,
    show_default=True,
    help="Default watermark cleanup setting for manifest items.",
)
@click.option(
    "--headed/--headless",
    default=True,
    show_default=True,
    help="Run real Chrome visibly (--headed, safer against Google WAF) or hidden (--headless).",
)
@click.option(
    "--timeout",
    default=240,
    show_default=True,
    type=int,
    callback=_validate_positive_timeout,
    help="Maximum seconds to wait per image generation.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON on stdout.")
def gemini_image_batch(
    manifest_path: Path,
    profile: str,
    reuse_chat: bool,
    gap_range: tuple[float, float],
    resume: bool,
    aspect: str,
    model: str | None,
    chat_url: str | None,
    strip_watermark: bool,
    headed: bool,
    timeout: int,
    as_json: bool,
) -> None:
    """Generate a sequence of images in a single Chrome session & optional shared chat."""
    from gflow_cli.gemini import (
        GeminiConfigError,
        generate_images_batch,
        load_image_batch_manifest,
    )

    profile_name = _validate_profile(profile)
    try:
        items = load_image_batch_manifest(manifest_path)
    except (GeminiConfigError, ValueError, OSError) as exc:
        raise click.UsageError(f"Invalid --manifest {manifest_path}: {exc}") from exc

    result = _run_gemini_coro(
        lambda: generate_images_batch(
            items,
            profile=profile_name,
            reuse_chat=reuse_chat,
            chat_url=chat_url,
            gap_range=gap_range,
            resume=resume,
            aspect=aspect,
            model=model,
            strip_watermark=strip_watermark,
            headless=not headed,
            timeout_seconds=timeout,
        ),
        cli_command="gemini image-batch",
        as_json=as_json,
    )
    data = result.to_dict()
    ok = bool(data.get("ok"))
    if as_json:
        json_output.emit(
            {
                "status": "ok" if ok else "fail",
                "command": "gemini image-batch",
                **data,
            }
        )
    else:
        _print_image_batch_summary(data)

    if not ok:
        stop_code = data.get("stop_exit_code")
        exit_code = int(stop_code) if isinstance(stop_code, int) and stop_code > 0 else 1
        sys.exit(exit_code)


# ---------------------------------------------------------------------------
# task-submit & task-harvest subcommands (Gemini Ultra Tasks / 任务 mode)
# ---------------------------------------------------------------------------


def _print_task_harvest_summary(data: dict[str, Any]) -> None:
    status_label = str(data.get("status_label") or ("completed" if data.get("ok") else "unknown"))
    ok = bool(data.get("ok"))
    style = "bold green" if ok else ("bold cyan" if status_label == "running" else "bold yellow")
    console.print(
        f"[{style}]Gemini task-harvest ({escape(status_label)}):[/{style}] "
        f"completed={bool(data.get('completed'))} "
        f"downloaded={data.get('downloaded_count', 0)}"
    )
    if data.get("chat_url"):
        console.print(f"  [dim]url       : {escape(str(data['chat_url']))}[/dim]")
    if data.get("out_file"):
        out_label = safe_path_text(Path(str(data["out_file"])))
        console.print(f"  [dim]out_file  : {escape(out_label)}[/dim]")
    exported: list[Any] = data.get("exported_files") or []
    if exported:
        console.print(f"  [dim]exported  : {len(exported)} file(s)[/dim]")
    missing: list[Any] = data.get("missing_image_indices") or []
    if missing:
        err_console.print(f"[yellow]Missing image indices: {escape(str(missing))}[/yellow]")
    _print_warnings(data.get("warnings") or [])


def _task_harvest_exit_code(data: dict[str, Any]) -> int:
    if bool(data.get("ok")):
        return 0
    status_label = str(data.get("status_label") or "")
    if status_label == "running":
        return 0
    if status_label == "timed_out":
        return 9
    return 1


@spends_credits(False)
@gemini.command(
    "task-submit",
    short_help="Submit an async Gemini Ultra Task (任务) and release Chrome immediately.",
)
@click.argument("prompt", required=False, default=None)
@click.option(
    "--prompt-file",
    "prompt_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
    help="Read task prompt from a UTF-8 file (mutually exclusive with PROMPT).",
)
@click.option(
    "--file",
    "--attach",
    "files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
    help="Attach a local file (e.g. story_memory_export.json or reference image; repeatable).",
)
@click.option(
    "--profile",
    default=DEFAULT_GEMINI_PROFILE,
    show_default=True,
    envvar="GFLOW_CLI_GEMINI_PROFILE",
    help="Isolated Gemini profile name.",
)
@click.option(
    "--headed/--headless",
    default=True,
    show_default=True,
    help="Run real Chrome visibly (--headed, safer against Google WAF) or hidden (--headless).",
)
@click.option(
    "--timeout",
    default=120,
    show_default=True,
    type=int,
    callback=_validate_positive_timeout,
    help="Maximum seconds to wait for task submission and URL resolution.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON on stdout.")
def gemini_task_submit(
    prompt: str | None,
    prompt_file: Path | None,
    files: tuple[Path, ...],
    profile: str,
    headed: bool,
    timeout: int,
    as_json: bool,
) -> None:
    """Submit an async cloud Task in Gemini Ultra 「任务」 mode and return its URL."""
    from gflow_cli.gemini import submit_task

    resolved_prompt = _resolve_prompt(prompt, prompt_file)
    profile_name = _validate_profile(profile)

    result = _run_gemini_coro(
        lambda: submit_task(
            resolved_prompt,
            profile=profile_name,
            attachments=list(files),
            headless=not headed,
            timeout_seconds=timeout,
        ),
        cli_command="gemini task-submit",
        as_json=as_json,
    )
    data = result.to_dict()
    ok = bool(data.get("ok"))
    if as_json:
        json_output.emit(
            {
                "status": "ok" if ok else "fail",
                "command": "gemini task-submit",
                **data,
            }
        )
    else:
        if ok:
            console.print(
                f"[bold green]Gemini task submitted:[/bold green] "
                f"{escape(str(data.get('chat_url') or ''))}"
            )
            console.print(
                f"  [dim]profile: {escape(profile_name)}  "
                f"files: {len(files)}  elapsed: {data.get('elapsed_seconds', 0)}s[/dim]"
            )
        else:
            err_console.print("[red]Gemini task submission did not succeed.[/red]")
        _print_warnings(data.get("warnings") or [])

    if not ok:
        sys.exit(1)


@spends_credits(False)
@gemini.command(
    "task-harvest",
    short_help="Poll and harvest text/JSON/images from an async Gemini Ultra Task URL.",
)
@click.option(
    "--url",
    "--chat-url",
    "url",
    required=True,
    help="Gemini Task conversation URL returned by `gflow gemini task-submit`.",
)
@click.option(
    "--out-dir",
    "out_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for downloaded task images and exported files.",
)
@click.option(
    "--out-file",
    "out_file",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write harvested text (or extracted JSON when --extract-json is set) to PATH.",
)
@click.option(
    "--download-images/--no-download-images",
    "download_images",
    default=True,
    show_default=True,
    help="Download full-resolution generated images from the completed task.",
)
@click.option(
    "--extract-json",
    "extract_json",
    is_flag=True,
    default=False,
    help="Extract structured JSON from inline code blocks or exported JSON artifacts.",
)
@click.option(
    "--expected-images",
    "expected_images",
    default=None,
    type=int,
    callback=_validate_optional_positive_int,
    help="Optional expected number of generated images to validate against.",
)
@click.option(
    "--wait-timeout",
    "wait_timeout",
    default=600,
    show_default=True,
    type=int,
    callback=_validate_non_negative_int,
    help="Seconds to wait for cloud task completion (0 = non-blocking status poll).",
)
@click.option(
    "--download-timeout",
    "download_timeout",
    default=50,
    show_default=True,
    type=int,
    callback=_validate_positive_timeout,
    help="Per-image full-size download timeout in seconds.",
)
@click.option(
    "--poll-interval",
    "poll_interval",
    default=10.0,
    show_default=True,
    type=float,
    callback=_validate_positive_float,
    help="Polling interval in seconds while waiting for completion.",
)
@click.option(
    "--profile",
    default=DEFAULT_GEMINI_PROFILE,
    show_default=True,
    envvar="GFLOW_CLI_GEMINI_PROFILE",
    help="Isolated Gemini profile name.",
)
@click.option(
    "--headed/--headless",
    default=True,
    show_default=True,
    help="Run real Chrome visibly (--headed, safer against Google WAF) or hidden (--headless).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON on stdout.")
def gemini_task_harvest(
    url: str,
    out_dir: Path | None,
    out_file: Path | None,
    download_images: bool,
    extract_json: bool,
    expected_images: int | None,
    wait_timeout: int,
    download_timeout: int,
    poll_interval: float,
    profile: str,
    headed: bool,
    as_json: bool,
) -> None:
    """Harvest text, JSON artifacts, and full-size images from a Gemini Ultra Task."""
    from gflow_cli.gemini import harvest_task

    cleaned_url = url.strip()
    if not cleaned_url:
        raise click.BadParameter("must not be empty.", param_hint="--url")
    profile_name = _validate_profile(profile)

    result = _run_gemini_coro(
        lambda: harvest_task(
            cleaned_url,
            profile=profile_name,
            out_dir=out_dir,
            out_file=out_file,
            download_images=download_images,
            extract_json=extract_json,
            expected_images=expected_images,
            wait_timeout=wait_timeout,
            download_timeout=download_timeout,
            poll_interval=poll_interval,
            headless=not headed,
        ),
        cli_command="gemini task-harvest",
        as_json=as_json,
    )
    data = result.to_dict()
    ok = bool(data.get("ok"))
    status_label = str(data.get("status_label") or "")
    if as_json:
        status_str = "ok" if ok else ("running" if status_label == "running" else "fail")
        json_output.emit(
            {
                "status": status_str,
                "command": "gemini task-harvest",
                **data,
            }
        )
    else:
        _print_task_harvest_summary(data)

    code = _task_harvest_exit_code(data)
    if code != 0:
        sys.exit(code)
