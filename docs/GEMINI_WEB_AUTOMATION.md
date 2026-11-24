# Gemini Web 自动化使用指南（`gflow gemini`）

`gflow gemini` 通过独立的真实 Chrome 配置文件（`gemini_profile_<name>`）驱动 Google Gemini Web（`https://gemini.google.com/app`），用于：

1. **剧本创作 / 结构化 JSON 生成**（`gflow gemini chat`）
2. **文生图与参考图一致性生图**（`gflow gemini image`）

它与 Google Flow 的浏览器配置（`profile_default` / `profile_acc2` / `profile_acc3`）在目录和文件锁层面**完全隔离**，不会干扰正在运行的 `gflow video` / `gflow image` 任务，也不会消耗 Flow 的 Veo 积分或 Flow 每日生图额度。

---

## 1. 首次登录 Google Ultra 账号

首次使用前，执行交互式登录命令。该命令会弹出一个全新的独立 Chrome 窗口（不影响你日常使用的 Chrome 或正在运行的 Flow 窗口）：

```bash
gflow gemini login --profile gemini_ultra --timeout 300
```

操作步骤：

1. 在弹出的 Chrome 窗口中登录你的 Google Ultra 账号，并进入 `https://gemini.google.com/app`。
2. 当页面出现 Gemini 输入框后，CLI 会自动检测到登录完成、保存持久化会话并退出（退出码 `0`）。
3. 核验本地配置与会话状态：

```bash
# 仅离线检查目录与锁状态（不启动 Chrome）
gflow gemini status --profile gemini_ultra

# 启动浏览器核验线上登录态（默认无头检查，可加 --headed 观察）
gflow gemini status --profile gemini_ultra --verify-live --json
```

> **提示**：可以通过环境变量 `GFLOW_CLI_GEMINI_PROFILE` 指定默认 profile 名称（未设置时默认为 `gemini_ultra`）。配置目录位于 `$GFLOW_CLI_HOME/gemini_profile_<name>`，文件锁位于 `$GFLOW_CLI_HOME/gemini_locks/`。

---

## 2. 文本对话与结构化 JSON 生成（`gflow gemini chat`）

### 常用示例

```bash
# 直接传 prompt
gflow gemini chat "写一个 3 镜头的可爱双熊日常分镜脚本" --profile gemini_ultra

# 从文件读取长 prompt，提取 JSON 并写入文件
gflow gemini chat \
  --prompt-file ./prompts/episode_script.txt \
  --extract-json \
  --out-file ./out/script.json \
  --profile gemini_ultra \
  --json

# 附带参考文件（--attach 可重复传入多个文件）
gflow gemini chat "根据这份规范检查分镜 JSON" \
  --attach ./refs/style_guide.md \
  --attach ./refs/master_ref.jpg \
  --extract-json \
  --json

# 在同一会话中继续多轮修复（传入上一轮返回的 chat_url）
gflow gemini chat --prompt-file ./prompts/lint_fix.txt \
  --chat-url "https://gemini.google.com/app/xxxxxxxx" \
  --extract-json \
  --json
```

### 核心参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `PROMPT` | 无 | 位置参数，与 `--prompt-file` 二选一（不可同时传入） |
| `--prompt-file PATH` | 无 | 从 UTF-8 文件读取多行 prompt |
| `--attach PATH` | 无 | 上传附件（可多次指定） |
| `--model NAME` | 无 | 尽力切换 Gemini Web 页面上的模型选项（失败仅告警不中断） |
| `--chat-url URL` | 无 | 复用已有会话 URL，与 `--new-chat` 互斥 |
| `--new-chat` | `false` | 强制新开对话 |
| `--extract-json` | `false` | 从回复中自动提取 ` ```json ` 代码块或最外层 JSON 对象/数组 |
| `--out-file PATH` | 无 | 将回复正文（或开启 `--extract-json` 时的 JSON）落盘 |
| `--headed` / `--headless` | `--headed` | 默认使用可见真实 Chrome 以降低 Google 风控概率 |
| `--timeout INT` | `180` | 等待流式生成的超时秒数 |
| `--json` | `false` | 标准输出打印结构化 JSON |

---

## 3. 高清生图与参考图编辑（`gflow gemini image`）

### 常用示例

```bash
# 文生图（16:9 横屏）
gflow gemini image "手绘2D扁平绘本风格，两只小熊在温暖客厅里分享草莓蛋糕" \
  --aspect 16:9 \
  --out-file ./out/seg01.png \
  --profile gemini_ultra \
  --json

# 单参考图 / 多参考图一致性生图（--ref 可重复传入）
gflow gemini image --prompt-file ./prompts/seg02.txt \
  --ref ./refs/yierbubu_2d_master_ref.jpg \
  --ref ./out/seg01.png \
  --aspect 16:9 \
  --out-file ./out/seg02.png \
  --profile gemini_ultra \
  --json

# 如需裁去右下角 Gemini 星标水印（保持比例）
gflow gemini image --prompt-file ./prompts/seg02.txt \
  --ref ./refs/yierbubu_2d_master_ref.jpg \
  --strip-watermark \
  --out-file ./out/seg02_clean.png \
  --json
```

### 下载优先级与图片校验

1. 优先触发 Gemini 原生 **Download full size** 按钮捕获真原图下载；
2. 兜底提取生成图片 `img[src]`（自动升级 Google `googleusercontent.com` 参数为 `=s0`）并通过已登录浏览器上下文下载；
3. 自动校验图片文件头魔数与最小分辨率（默认 `>= 512px`），防止误保存头像或缩略图；
4. 若未传 `--out-file`，图片会自动保存到 `<output_dir>/gemini/images/<YYYY-MM-DD>/gemini_<timestamp>_<id>.png`。

---

## 3.5 同会话批量生图（`gflow gemini image-batch`）

对于一整集连续分镜（例如 6 张 16:9 锚图），推荐使用 `gflow gemini image-batch`：只启动 **1 次**独立 Chrome，并在 **1 个 Gemini 对话（`--reuse-chat`）** 内按顺序生成多张原图，既避免频繁冷启动浏览器，又保留上下文房间记忆，同时每张仍是独立的 2.7K 原图下载（绝不拼九宫格降清）。

### Manifest 格式

支持顶层 JSON 数组 `[...]` 或 `{"items": [...]}`：

```json
{
  "items": [
    {
      "id": "seg01",
      "prompt": "第 1 镜分镜提示词...",
      "out_file": "/abs/path/seg01.jpg",
      "ref_paths": ["/abs/path/yierbubu_2d_master_ref.jpg"],
      "aspect": "16:9"
    },
    {
      "id": "seg02",
      "prompt": "第 2 镜同场景延续提示词...",
      "out_file": "/abs/path/seg02.jpg",
      "ref_paths": ["/abs/path/seg01.jpg"],
      "aspect": "16:9"
    }
  ]
}
```

### 运行示例

```bash
gflow gemini image-batch \
  --manifest ./batch_manifest.json \
  --profile gemini_ultra \
  --reuse-chat \
  --gap 25-45 \
  --resume \
  --aspect 16:9 \
  --json
```

关键特性：

- **断点续跑（`--resume`，默认开启）**：若某张目标图已存在且尺寸合法，直接记入 `skipped`；若整批全部就绪则**零启动 Chrome** 秒返 `ok: true`。
- **同会话与串行锚点（`--reuse-chat` / `--separate-chats`）**：`seg01.jpg` 在第 1 轮生成落地后，第 2 轮即可直接作为 `seg02` 的 `ref_paths` 上传，全程无需重启浏览器。
- **中途风控优雅停机**：若中途命中 `RATE_LIMITED`（退出码 `4`）、`CHALLENGE_DETECTED` / `RISK_QUARANTINED`（退出码 `10`）或 `AUTH_REQUIRED`（退出码 `3`），会立即停止剩余条目（标记为 `deferred`）、关闭浏览器，并在 `stdout` 返回包含 `completed` / `skipped` / `failed` / `deferred` / `stop_error_code` / `stop_exit_code` / `hard_halt` / `retry_after_seconds` / `chat_url` 的完整部分回执 JSON，方便上游池守护进程精准退让或停机。

---

## 3.6 异步云端任务模式（`gflow gemini task-submit` / `task-harvest`）

Gemini Ultra 的 **「任务」（Tasks / Neon 云端代理）** 支持在云端沙盒内异步执行 Python 校验与多图生成，关闭本地 Chrome 后任务仍持续运行，大幅降低单集提交次数与锁占用时长：

### 1) 提交异步任务（立即释放 Chrome 与锁）

```bash
# 批量故事生成（上传历史去重 JSON）
gflow gemini task-submit \
  --prompt-file ./prompts/seed_50_stories.txt \
  --file ./story_memory_export.json \
  --profile gemini_ultra \
  --timeout 120 \
  --json

# 一集多帧锚图生成（上传角色母图）
gflow gemini task-submit \
  --prompt-file ./prompts/ep_anchors_task.txt \
  --file ./refs/yierbubu_2d_master_ref.jpg \
  --profile gemini_ultra \
  --json
```

提交后返回 `chat_url`（例如 `https://gemini.google.com/app/<task_id>`）并立即释放 `GeminiProfileLease`。

### 2) 轮询与收割产物（`task-harvest`）

```bash
# 非阻塞检查进度（--wait-timeout 0：仍在运行返回 completed=false, status_label="running", exit 0）
gflow gemini task-harvest \
  --url "https://gemini.google.com/app/<task_id>" \
  --no-download-images \
  --wait-timeout 0 \
  --json

# 等待完成并提取 JSON 故事库
gflow gemini task-harvest \
  --url "https://gemini.google.com/app/<task_id>" \
  --extract-json \
  --no-download-images \
  --out-file ./out/stories.json \
  --wait-timeout 600 \
  --json

# 等待完成并下载全部 2.7K 原图（单张下载超时默认 50s，失败自动重试 1 次）
gflow gemini task-harvest \
  --url "https://gemini.google.com/app/<task_id>" \
  --out-dir ./out/anchors \
  --download-images \
  --expected-images 6 \
  --download-timeout 50 \
  --wait-timeout 600 \
  --json
```

`task-harvest` 退出码约定：
- `0`：任务完成且满足预期（`ok=true`），或非阻塞轮询 `--wait-timeout 0` 时任务仍在运行（`completed=false`, `status_label="running"`）；
- `1`：任务已完成但产物不完整（`status_label="incomplete"`，详见 `missing_image_indices` 以便定点补帧）；
- `9`：等待超过 `--wait-timeout` 仍未完成（`status_label="timed_out"`）；
- `3` / `4` / `10` / `11` / `23`：标准 `GeminiError` 风控/鉴权/锁冲突退出码。

---

## 4. 错误码与退出码约定

配合 `--json` 调用时，失败会在 `stdout` 输出包含 `ok: false`、`status: "fail"`、`error_code`、`exit_code`、`retryable`、`remediation_hint`、`screenshot_path`、`dom_path` 的 JSON：

| 退出码 | `error_code` | 含义与建议动作 |
| --- | --- | --- |
| `0` | — | 成功 |
| `1` | `UNEXPECTED_ERROR` / 状态未就绪 | 通用错误，或 `status` 检测到 profile 未就绪 |
| `2` | Usage / Validation | CLI 参数错误（缺少 prompt、文件不存在、互斥参数冲突等） |
| `3` | `AUTH_REQUIRED` | 登录态失效或跳转登录页；**禁止重试**，请重新执行 `gflow gemini login` |
| `4` | `RATE_LIMITED` | 触发限流/配额提示；**立即退让冷却**，切勿高频连发 |
| `6` | `DOWNLOAD_FAILED` | 图片下载失败，可检查网络/代理后择机重试 |
| `9` | `TIMEOUT` | 等待页面响应或出图超时，可适当调大 `--timeout` |
| `10` | `CHALLENGE_DETECTED` / `RISK_QUARANTINED` | 触发人机验证/风控拦截，或命中断路器/冷却隔离/急停文件；**禁止自动重试**，先停跑并冷却 |
| `11` | `PROFILE_LOCKED` / `INVALID_INPUT` | 该 Gemini profile 正被另一进程占用，或配置/名称非法 |
| `12` | `LOGIN_TIMEOUT` | `gflow gemini login` 在超时前未检测到登录成功 |
| `23` | `UI_DRIFT` | Gemini Web DOM 结构变动导致关键控件定位失败，查看诊断 DOM/截图 |
| `26` | `NO_IMAGE_GENERATED` | 模型仅返回文字或拒绝生图，未产出符合分辨率要求的图片 |
| `130` | `INTERRUPTED` | 用户按下 `Ctrl+C` 中断 |

---

## 5. 与 Flow 并行/共存时的防风控与退让建议

`gflow gemini` 内置持久化防风控账本（`$GFLOW_CLI_HOME/gemini_ledger/state.json`），与 Flow 产线的退让规则保持一致：

1. **配置目录与锁绝对隔离**：始终使用 `gemini_profile_<name>`（默认 `gemini_ultra`）与独立锁目录 `$GFLOW_CLI_HOME/gemini_locks/`，切勿与 Flow 的 `profile_default` / `profile_acc2` / `profile_acc3` 混用目录。
2. **默认节流与突发上限（Burst Cap）**：
   - 全局最小间隔 **8–15 秒**；单 profile 文本对话 **10–25 秒**、生图 **15–35 秒**；
   - 滚动 10 分钟生图突发上限默认 **8 张**（可通过 `GFLOW_GEMINI_MAX_PER_BURST` 调整，`0` 表示关闭）；
   - 触发一次 WAF/Challenge 后，该 profile 隔离 **45 分钟**，解封后额外进入 **2 小时高戒备期（45–90 秒/次）**；30 分钟内触发 2 次 WAF 则全局熔断（`global_halted`）。
3. **遇风控零盲目重试**：
   - `CHALLENGE_DETECTED`（退出码 `10`）：0 重试，立即隔离 45 分钟 + 2 小时高戒备；
   - `AUTH_REQUIRED`（退出码 `3`）：0 重试，进入手动隔离（`quarantined_manual`），重新执行 `gflow gemini login` 成功后自动解除；
   - `RATE_LIMITED`（退出码 `4`）：0 重试，按 30 分钟内连续次数阶梯退让 **120s → 300s → 900s**，连续第 3 次触发 45 分钟隔离；
   - 仅图片 CDN 下载网络抖动（退出码 `6`）会执行最多 3 次纯下载重试（1s → 2s → 4s ±25%），绝不重复提交 prompt。
4. **急停开关与 Flow 账本跨库只读联动**：
   - 在当前工作目录、`$GFLOW_CLI_HOME` 或 Flow 账本目录放置 `STOP_GEMINI` 或 `STOP_FLOW` 文件，都会在启动浏览器前和等待休眠后立即停机；
   - 设置 `GFLOW_FLOW_LEDGER_DIR` 与 `GFLOW_GEMINI_LINKED_FLOW_PROFILE`（或 `GFLOW_LINKED_FLOW_PROFILE`）后，`gflow gemini` 会以**严格只读**方式探测 Flow 账本：若 Flow 全局熔断、对应账号处于冷却隔离或 `profile_<name>.lock` 正在执行视频任务，则自动拒绝并发启动，避免同一 Google Ultra 账号双线撞车。
5. **坚持 `--headed` 真实 Chrome 与固定代理出口**：默认使用可见真实 Chrome（切勿静默降级到无头 Chromium），且 `login` 与 `chat` / `image` 应保持同一 `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` 出口。

---

## 6. 常见问题排查

- **提示 `PROFILE_LOCKED`（退出码 `11`）**：说明已有另一个 `gflow gemini` 进程正在使用同一 profile。先等待其结束，或通过 `gflow gemini status --profile <name>` 查看 `lock_owner_pid`。
- **提示 `AUTH_REQUIRED`（退出码 `3`）**：运行 `gflow gemini login --profile <name>` 重新登录刷新 Cookie。
- **页面结构变更或生图未捕获（退出码 `23` / `26`）**：失败时引擎会在 `tmp/gemini-debug/` 下保存现场截图（`screenshot_path`）与精简 DOM 快照（`dom_path`），先查看这些文件定位具体提示（例如安全策略拒答或按钮文案变动）。
- **网络或代理不通**：确认终端环境中的 `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` 已正确指向可用的本地代理端口（如 Clash 端口），并用 `gflow gemini status --verify-live --headed` 直观验证页面加载情况。
