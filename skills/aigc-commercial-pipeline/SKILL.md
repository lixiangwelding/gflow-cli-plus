---
name: aigc-commercial-pipeline
version: "1.0"
description: >
  End-to-end AIGC commercial video production pipeline on Google Flow (`gflow-cli-plus`):
  Nano Banana 2.1 (`nano2`) consistency master plates, composite reference boards for
  agent-only multi-ref `i2i`, Veo 3.1 Fast (`veo-fast`) `i2v` and `--end-frame` dual-frame
  interpolation with multi-profile WAF load-balancing, mandatory 3-frame (`start/mid/end`)
  visual QA gates, and `edge-tts` + Veo native stereo foley + sidechain-ducked score +
  EBU R128 `-16 LUFS` FFmpeg/kinocut mastering.
---

# AIGC Commercial Pipeline (`gflow-cli-plus`)

Produce broadcast-grade, visually consistent multi-shot commercials on Google Flow's migrated and **agent-only** composers without wasting Veo credits on preventable motion defects.

## Progressive Workflow Overview

Execute the pipeline in five gated stages. Never advance a shot to timeline assembly until its keyframe and 3-frame motion strip pass visual QA.

```text
[1. Master Plates (nano2)]
        │
        ▼
[2. Composite Ref-Boards & Shot Keyframes (image i2i)]
        │
        ▼
[3. Veo 3.1 Fast Motion (video i2v / --end-frame + Multi-Profile WAF Pool)]
        │
        ▼
[4. Mandatory 3-Frame QA Gate (start 0.4s / mid / end-0.35s)] ──(fail)──► Fix Frame or Prompt
        │ (pass)
        ▼
[5. Audio & Mastering (edge-tts + Veo Stereo + Sidechain Ducking + -16 LUFS)]
```

---

## Stage 1 — Consistency Master Generation (`nano2`)

Lock product geometry, hero character identity, and primary environment architecture **before** generating shot frames.

- **Default batch/master model:** Always use `--model nano2` (Nano Banana 2.1) for fast, zero-credit iteration (`nano-pro` is daily-capped).
- **Three canonical master plates:**
  1. `M01_product_ortho.png` — Clean studio or neutral-lit hero product anchor showing exact industrial design, ports, buttons, and brand mark.
  2. `M02_talent_sheet.png` — Hero talent wardrobe, hairstyle, and facial structure under neutral key light (keep **at most one face-bearing reference** per generation).
  3. `M03_env_master.png` — Wide architectural plate establishing wall materials, floor reflectance, and color temperature.

```bash
gflow --profile default image t2i \
  "Clean product hero photo of [product], matte anodized body, crisp工業 details, neutral studio lighting" \
  --model nano2 --aspect 16:9 --project "$PROJECT_ID" --out-file masters/M01_product_ortho.png
```

---

## Stage 2 — Composite Ref-Boards for Multi-Ref `i2i` (Agent-Only Composer)

On accounts served Google Flow's **agent-only composer**, attaching multiple separate files can cause the Gemini agent to weight one image and drift on another, or lengthen upload-picker turns.

- **Composite Reference Board pattern:** When a shot must simultaneously preserve **Product + Room + Talent/Prop**, stitch the approved masters side-by-side into a single labeled or clean 16:9 reference board (`ffmpeg` `hstack`/`xstack` or Pillow) and pass it via `--ref` (or pass 1–2 tightly cropped local `--ref` files).
- **Pre-`i2v` clean-plate scrub:** Inspect every generated `Sxx_start.png` at 100% zoom before spending Veo credits. If the wall behind the product contains **small distracting wall switches, sockets, thermostat boxes, or stray cables**, run a quick `gflow image i2i --ref Sxx_start.png` cleanup pass to erase them first (see [`references/veo-motion-pitfalls.md`](references/veo-motion-pitfalls.md)).

```bash
# Example: Stitch product + environment masters into a single composite ref board
ffmpeg -y -i masters/M01_product_ortho.png -i masters/M03_env_master.png \
  -filter_complex "[0:v]scale=960:1080:force_original_aspect_ratio=decrease,pad=960:1080:(ow-iw)/2:(oh-ih)/2[l];[1:v]scale=960:1080:force_original_aspect_ratio=decrease,pad=960:1080:(ow-iw)/2:(oh-ih)/2[r];[l][r]hstack=inputs=2" \
  refs/board_S02.png

# Generate locked shot start-frame on agent-only composer
gflow --profile default image i2i \
  "Medium cinematic shot placing the exact device from the left reference flush against the clean plaster wall from the right reference, warm morning key light, clean uncluttered wall" \
  --ref refs/board_S02.png --model nano2 --aspect 16:9 \
  --project "$PROJECT_ID" --out-file frames/S02_start.png
```

---

## Stage 3 — `veo-fast` `i2v`, Dual-Frame `--end-frame`, & Multi-Profile WAF Balancing

Animate approved stills with **Veo 3.1 Fast** (`--model veo-fast`) for optimal commercial motion fidelity per credit.

### Single-Frame `i2v` vs Dual-Frame `--initial-frame` + `--end-frame`
- **Standard `i2v` (`--initial-frame` only):** Best for controlled push-ins, subtle handheld breathing, product surface light sweeps, and locked 3D title cards.
- **Dual-frame interpolation (`--initial-frame Sxx_start.png --end-frame Sxx_end.png`):** Use when a shot transitions between two exact states (e.g., unmounted -> wall-mounted clicks into place, room dark -> laser grid aligned, macro detail -> hero packshot). Both frames are uploaded in order and bound via natural-language start/end directives on the agent-only composer.

### Multi-Profile WAF & Concurrency Load-Balancing
- Never hammer a single Chrome profile with back-to-back tight loops or concurrent browser processes (profile lease locks serialize same-profile work, and burst traffic raises reCAPTCHA/WAF friction).
- Shard shots across authenticated real-Chrome profiles (`default`, `acc2`, `acc3`, ...), running **at most 1 active generation per profile** with staggered dispatch (`10–25s` offset between profiles):
  - Worker A (`--profile default`): `S01`, `S04`, `S07`
  - Worker B (`--profile acc2`): `S02`, `S05`, `S08`
  - Worker C (`--profile acc3`): `S03`, `S06`, `S09`
- **Virtualized DOM player resilience:** On `flow.google.com`, finished agent-only video tiles sometimes keep `<video>` unmounted under opaque `/asb/` thumbnails. `gflow-cli-plus` automatically resolves the clip UUID from the `Zzl0ze` project listing after 2 unmounted polls and exports the 720p MP4 directly via the clip viewer — never blindly re-submit a timed-out video without checking the project first.

```bash
gflow --profile acc2 video i2v \
  "Slow controlled dolly-in; the device remains rigidly mounted flush against the wall while a crisp green alignment line projects strictly flat along the wall surface; unobstructed foreground sightline" \
  --model veo-fast --aspect 16:9 \
  --initial-frame frames/S02_start.png --end-frame frames/S02_end.png \
  --project "$PROJECT_ACC2" --out-file shots/S02_raw.mp4
```

---

## Stage 4 — Mandatory 3-Frame Visual QA Gate (`start / mid / end`)

**Never judge a Veo clip from frame 0 alone.** Frame 0 is your input keyframe; defects emerge in the middle and tail (`morphing`, `mid-air laser detachment`, `foreground occlusion wipe`, `text melting`).

1. **Per-shot MCP QA strip (zero credits):** Call MCP tool `gflow_qa_video_strip` right after each clip downloads:
   - Extracts `start (0.4s)`, `mid (50%)`, and `end (duration - 0.35s)` side-by-side into `qa/S02_strip.jpg` and returns exact resolution, duration, and audio-stream presence.
2. **Batch directory audit:** Run [`scripts/qa_audit_board.py`](scripts/qa_audit_board.py) over `shots/` to build individual strips plus a unified contact sheet (`qa/ALL_SHOTS_AUDIT.jpg`) and `qa/audit_manifest.json`:
   ```bash
   python3 skills/aigc-commercial-pipeline/scripts/qa_audit_board.py \
     --shots-dir shots --out-dir qa --pattern "S*_raw.mp4"
   ```
3. **Inspect every strip visually** (`read_file` / vision inspection on the `.jpg` strip):
   - Check geometry stability across all 3 columns (`0.4s` -> `mid` -> `end-0.35s`).
   - If any defect appears, consult [`references/veo-motion-pitfalls.md`](references/veo-motion-pitfalls.md) for the exact prompt/keyframe surgical fix before re-generating only the failed shot.

---

## Stage 5 — Neural VO, Native Stereo Foley, Sidechain Ducking & `-16 LUFS` Mastering

Veo always synthesizes native stereo room tone / foley (`[CONSTRAINT]`). Treat audio as a three-stem commercial mix rather than stripping everything or letting stems clash:

1. **Frame-accurate Voiceover (`edge-tts`):**
   - Call MCP tool `gflow_synthesize_voiceover(text=..., out_mp3_path=..., voice="zh-CN-XiaoxiaoNeural", rate="+5%")` per line or full script.
   - Read back the exact `duration_sec` returned by `ffprobe` **before** locking picture cut points so speech is never clipped or rushed.
2. **Three-Stem Commercial Mix Architecture:**
   - **Stem A — Voiceover (`VO`):** Center-anchored, high-passed at `80Hz`, light presence boost, untouched dynamic priority.
   - **Stem B — Veo Native Stereo Ambience/Foley (`SFX`):** Keep subtle tactile foley (button clicks, mechanical slide, room air) trimmed to `-18dB .. -22dB`; mute (`volume=0`) only on shots where Veo hallucinated unwanted speech or gibberish vocals.
   - **Stem C — Musical Score (`BGM`):** Auto-ducked under `VO` via `sidechaincompress` (`threshold=0.03:ratio=6:attack=25:release=350`) with clean fade-in/out.
3. **EBU R128 `-16 LUFS` Final Mastering (`ffmpeg` / `kinocut`):**
   - Normalize integrated loudness to **`-16 LUFS`** with **`-1.5 dBTP`** true-peak ceiling (`loudnorm=I=-16:TP=-1.5:LRA=11`) for web/mobile commercial delivery (`H.264 High@L4.1`, `yuv420p`, `aac 192k`, `-movflags +faststart`).

```bash
ffmpeg -y \
  -i picture_locked.mp4 \
  -i vo_track.mp3 \
  -i bgm_score.mp3 \
  -filter_complex "\
    [0:a]volume=0.18[sfx]; \
    [1:a]highpass=f=80,acompressor=threshold=-18dB:ratio=3,asplit=2[vo_mix][vo_sc]; \
    [2:a]volume=0.38[bgm_pre]; \
    [bgm_pre][vo_sc]sidechaincompress=threshold=0.03:ratio=6:attack=25:release=350[bgm_duck]; \
    [vo_mix][sfx][bgm_duck]amix=inputs=3:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[aout]" \
  -map 0:v -map "[aout]" \
  -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p \
  -c:a aac -b:a 192k -movflags +faststart \
  final_commercial_master.mp4
```

---

## Bundled References & Scripts

- **[`references/veo-motion-pitfalls.md`](references/veo-motion-pitfalls.md)** — Field-tested diagnosis & prompt engineering fixes for mid-air laser/beam detachment, lateral-pan foreground camera occlusion, distracting wall switches in `i2v`, and rock-steady 3D Chinese title cards.
- **[`scripts/qa_audit_board.py`](scripts/qa_audit_board.py)** — Standalone CLI batch tool generating per-shot 3-frame strips (`0.4s / mid / end-0.35s`) and a master multi-row contact sheet for any directory of `S*_raw.mp4` clips.
