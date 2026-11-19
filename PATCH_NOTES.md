# gflow-cli-plus Patches

Forked & enhanced from `ffroliva/gflow-cli` for full production reliability with Google Flow's newer **agent-only composer**, macOS real-Chrome sessions, and commercial AIGC workflows:

1. **Agent-only composer local attachment upload (`image i2i` / `video i2v` / `video r2v`)**
   - Unlocks local `--ref` for `gflow image i2i`, local `--initial-frame` (+ optional `--end-frame` dual-frame interpolation) for `gflow video i2v`, and local `--ref` for `gflow video r2v` on accounts served the migrated `agent-only composer`.
   - Uploads attachments via the composer `+` asset picker (`filechooser`) with unique filename tags, filters to Images, and dispatches natural-language directives preserving frame/reference intent while starting fresh chat threads (`edit_square`) to prevent long-context errors.
2. **Unmounted virtualized `<video>` player recovery & hardened settings restore**
   - Fixes `TransportTimeoutError` (exit 9) when `flow.google.com` resolves a finished clip UUID via `Zzl0ze` project listing (`_listed_uuid`) but virtualizes the grid `<video>` tag: after `unmounted_polls >= 2`, `AgentOnlyComposer` logs `migrated.agent_only.video_ready_unmounted_player`, returns `TileMedia(uuid, "")`, and automatically exports/downloads the 720p MP4 via `upscale_video_migrated`.
   - Sends `Escape` before reopening Agent settings in `restore_defaults` so lingering popovers/overlays never trigger `migrated.agent_only.defaults_restore_failed`, and falls back to in-browser `fetch()` when Node `APIRequestContext` encounters transient proxy/TLS `ECONNRESET`.
3. **Fast-fail on bilingual (EN/ZH) agent error turns & Chinese credit-gate parsing**
   - Detects `"出了点问题"` / `"请重试"` / `"Something went wrong"` / `"Try again"` when no clip is pending instead of idling for the 600s timeout, and parses Chinese credit-confirmation wording (`"要消耗 20 个点数，开始生成这 1 个视频吗？"`).
4. **macOS real-Chrome login marker persistence**
   - Preserves `.gflow_browser_strategy = chrome` when interactive real-Chrome login succeeded (`Signed in.`) even if local `browser_cookie3` cookie decryption throws `verification_error` (`ImportError` / Keychain), preventing unwanted Chromium downgrade loops.
5. **Local AIGC Commercial Workflow MCP Tools & Skill (`zero-spend`)**
   - Registers `gflow_qa_video_strip` (`ffprobe` + `ffmpeg` 3-frame `0.4s / mid / end-0.35s` side-by-side motion QA strip + stream metadata) and `gflow_synthesize_voiceover` (`edge-tts` neural voiceover + exact `ffprobe` duration) in `src/gflow_cli/mcp/workflow_tools.py` (available even under `--no-spend`).
   - Ships [`skills/aigc-commercial-pipeline`](skills/aigc-commercial-pipeline/SKILL.md) covering `nano2` consistency masters, composite ref-boards, `veo-fast` `i2v` / `--end-frame` with multi-profile WAF load-balancing, mandatory 3-frame visual QA gates (`references/veo-motion-pitfalls.md`, `scripts/qa_audit_board.py`), and `edge-tts` + Veo stereo + sidechain-ducked `-16 LUFS` mastering.
