# gflow-cli-plus Patches

Forked & enhanced from `ffroliva/gflow-cli` for full compatibility with Google Flow's newer **agent-only composer** and macOS real-Chrome sessions:

1. **Agent-only composer local attachment upload (`image i2i` / `video i2v` / `video r2v`)**
   - Unlocks local `--ref` for `gflow image i2i`, local `--initial-frame` (+ optional `--end-frame`) for `gflow video i2v`, and local `--ref` for `gflow video r2v` on accounts served the migrated `agent-only composer`.
   - Uploads attachments via mounted `<input type="file">` or the composer `+` button / upload menu (`filechooser`), then dispatches clear natural-language directives preserving frame/reference intent.
2. **Fast-fail on bilingual (EN/ZH) agent error turns**
   - Detects `"出了点问题"` / `"请重试"` / `"Something went wrong"` / `"Try again"` when no clip is pending instead of idling for the full 600s video timeout.
3. **macOS real-Chrome login marker persistence**
   - Preserves `.gflow_browser_strategy = chrome` when interactive real-Chrome login succeeded (`Signed in.`) even if local `browser_cookie3` cookie decryption throws `verification_error` (`ImportError` / Keychain), preventing unwanted Chromium downgrade loops.
