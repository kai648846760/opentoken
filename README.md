# OpenToken / OpenToken

> 中英双语 README。
> Bilingual README — Chinese first, English follows each section.

把多家 LLM 网页登录态、API key 凭证统一成一个本地 **OpenAI 兼容网关**。<br>
A single local **OpenAI-compatible gateway** that fronts multiple LLM providers — web sessions, browser-harvested cookies, and direct API keys — behind one unified surface.

- 上游：每家 provider 的网页 / 登录态 / API。<br>
  Upstream: each provider's web session, browser-harvested credentials, or direct API.
- 下游：本地 OpenAI 风格接口。<br>
  Downstream: a local OpenAI-style HTTP API.
- 本地暴露：`/v1/models`、`/v1/chat/completions`、`/v1/responses`、`/v1/files`、`/v1/uploads`、`/v1/embeddings`（501，见下）。<br>
  Locally exposed: the routes above; `/v1/embeddings` deliberately returns 501 — see the Embeddings section.

---

## 它能解决什么问题 / What it solves

如果你同时在用 DeepSeek、Qwen、Kimi、Doubao、GLM、Claude、Gemini、ChatGPT、Grok、Mimo 等多家服务，通常会被这些问题困扰：每家登录方式不同、模型名不同、流式协议不一、和 OpenAI 风格客户端对接很麻烦。OpenToken 做四件事：（1）统一管理凭证；（2）暴露一个本地 OpenAI 兼容网关；（3）把不同 provider 的模型映射到统一调用方式；（4）尽量把流式输出、tool calls、响应结构对齐到 OpenAI 风格。

If you use DeepSeek, Qwen, Kimi, Doubao, GLM, Claude, Gemini, ChatGPT, Grok, and Mimo side by side, you've hit it: every provider has a different login, a different model namespace, a different streaming dialect, and a different way of bolting onto an OpenAI-style client. OpenToken (1) manages credentials in one place, (2) exposes a single OpenAI-compatible gateway, (3) maps every provider's models to a uniform call shape, and (4) aligns streaming, tool-calls, and response envelopes to OpenAI's spec.

---

## 当前支持的 provider / Supported providers

**网页登录态 / 浏览器采集（11 个） — Web sessions / browser-harvested (11)**

DeepSeek · Qwen International · Qwen China · Kimi · Claude · Doubao · ChatGPT · Gemini · Grok · GLM International · GLM China · Xiaomi Mimo

**API key 直连（4 个） — Direct API keys (4)**

- **Manus**（官方 API / official API）
- **NVIDIA NIM** — 走 `integrate.api.nvidia.com/v1`，免费 40 RPM，覆盖 DeepSeek R1 / Llama 3.3 70B / Qwen 2.5 72B、Coder 32B / Mixtral 8x22B 等。注册 NVIDIA 账号拿 `nvapi-...` key 就能用，0 信用卡。<br>
  Free 40 RPM via `integrate.api.nvidia.com/v1`, covers DeepSeek R1 / Llama 3.3 70B / Qwen 2.5 72B & Coder 32B / Mixtral 8x22B and more. Just a NVIDIA account → `nvapi-...` key. No card needed.
- **Unified Proxy (LiteLLM)** — 一个 adapter 接 100+ 后端（OpenRouter / Groq / Together / Bedrock / Anthropic / OpenAI / Perplexity / Cohere / Mistral / xAI / Fireworks / DeepInfra / Azure / Ollama / LM Studio …）。LiteLLM 是软依赖：`uv sync --extra unified` 装上。<br>
  One adapter, 100+ backends through LiteLLM. Soft dependency — install with `uv sync --extra unified` only if you need it.

> 实际可用模型取决于你已经登录 / 配置成功的 provider。`/v1/models` 只列出已经登录、可被路由的那些。<br>
> What actually shows up in `/v1/models` is limited to providers you've logged into / configured.

---

## /v1/models 全部实时发现 / `/v1/models` is fully live-discovered

每个 provider 各自的发现路径如下；任何发现器失败都软降级为空。结果缓存在 `~/.opentoken/model-catalog-cache.json`（TTL 6 小时），首次冷启动并发跑所有发现器并尊重 45 s 全局 deadline，所以一个慢 provider 不会拖慢整体。

Each provider discovers its catalog live; any failure soft-degrades to empty. Results cache in `~/.opentoken/model-catalog-cache.json` (6h TTL); cold-start runs every discoverer concurrently under a 45 s deadline, so a slow provider can't hold up the rest.

| Provider | 发现方式 / How |
|----------|---------|
| qwen-intl / qwen-cn / doubao / glm-cn | web 页面 / dialog 抓取（部分走 Camoufox 浏览器） |
| glm-intl | `GET chat.z.ai/api/models`，失败回退抓首页 |
| deepseek | `GET /api/v0/users/current` 校验，返回协议支持的两个 wire 模型 |
| kimi | 抓 kimi.com 首页嵌入的 model metadata |
| nim | `GET integrate.api.nvidia.com/v1/models` Bearer auth |
| manus | `GET api.manus.im/api/v1/agents` |
| chatgpt | `GET /backend-api/models`，失败回退抓首页 |
| claude | `GET /api/organizations` + statsig chat-models 配置 |
| gemini | 抓 gemini.google.com app HTML |
| grok | 抓 grok.com 首页 HTML |
| mimo | 抓 xiaomimo.com 首页 HTML |
| unified | 按凭证里配置的 backend 过滤 `litellm.model_cost` |

**Fallback 楼板 / Fallback floor**：qwen-intl 的目录现在是纯 JS 渲染，kimi 的目录在 gRPC-Connect 后面 —— httpx 抓不到。如果某个 provider 已登录但实时发现返回空，opentoken 会用一个最小的已知 wire 模型清单填底（qwen-intl → `qwen3.6-plus`、`qwen-max-latest`；kimi → `k2`、`k1`），让 chat 仍然可用、不会从 `/v1/models` 静默消失。实时发现一旦恢复就直接覆盖楼板。<br>
qwen-intl's catalog is now JS-rendered; kimi's lives behind a gRPC-Connect endpoint — neither is scrape-able with httpx. When a logged-in provider's live discovery yields nothing, opentoken falls back to a minimal known-wire model list (qwen-intl → `qwen3.6-plus`, `qwen-max-latest`; kimi → `k2`, `k1`) so chat still works and the provider doesn't silently vanish from `/v1/models`. Live discovery wins whenever it returns something.

---

## 环境要求 / Requirements

- Python **>= 3.13**
- 推荐用 [`uv`](https://docs.astral.sh/uv/) — Recommended: `uv`.

---

## 目录与本地状态 / On-disk layout

凭证不进仓库 —— 全部写到 `~/.opentoken/`：<br>
Credentials never enter the repo — everything lives under `~/.opentoken/`:

```
~/.opentoken/
├── config.json              # 本地网关配置（含本地 API key / host / port） / local gateway config
├── providers/<name>.json    # 各 provider 凭证 / per-provider credentials
├── auth-profiles.json       # 跨 provider 的认证 profile / cross-provider auth profiles
├── provider-sessions.json   # 会话上下文（conversation_id 等） / session continuation state
├── responses.json           # /v1/responses 历史 / responses store
├── files/, uploads/         # /v1/files & /v1/uploads 二进制内容 / binary blobs
└── model-catalog-cache.json # 模型发现缓存 / discovery cache
```

**权限 / Permissions**：所有凭证 / cookie / token / 用户上传内容文件 0600（owner-only），目录树 0700（不可列）。`response_store`、`upload_store`、`file_store`、`provider_sessions`、`auth_profiles`、`provider_store`、bridge.py 都走原子写 + sensitive=True 强制 chmod。多用户主机上其它用户既看不到你的 cookie 也看不到你的对话历史。<br>
All credentials/cookies/tokens/uploads are 0600 (owner-only); the directory tree is 0700 (not listable). Every JSON store goes through atomic write + sensitive=True chmod — on a shared host nobody else can read your sessions, conversation history, or uploaded files.

---

## 快速开始 / Quick start

```bash
uv sync
uv run opentoken onboard      # 初始化 state 目录 / scaffolds ~/.opentoken/
uv run opentoken start        # 默认监听 http://127.0.0.1:32117 / binds 127.0.0.1:32117
```

默认 base URL：`http://127.0.0.1:32117/v1`。<br>
Default base URL: `http://127.0.0.1:32117/v1`.

绑定非 loopback 时 / **Binding a non-loopback host**：`opentoken start --host 0.0.0.0` 会在 stderr 打印一条警告，因为这会把你已登录的 provider 会话暴露到本机之外；如果同时没配 API key，警告会升级为 "UNAUTHENTICATED"。<br>
`opentoken start --host 0.0.0.0` prints a stderr warning — that exposes every logged-in provider session beyond this machine. With no API key configured, the warning escalates to **UNAUTHENTICATED**.

---

## 本地 API key / Local API key

OpenToken 用的是**本地网关自己的 API key**（不是上游 provider 的 key）。<br>
The API key OpenToken expects on inbound requests is the **local gateway key**, not any upstream provider key.

```bash
cat ~/.opentoken/config.json
# {"api_key":"...","host":"127.0.0.1","port":32117}
```

如果配置文件被改成 `"api_key": ""` 或纯空白 → 当前为 "keyless 本地模式"，请求免鉴权直接放行。<br>
If `api_key` is empty/whitespace, opentoken treats it as keyless-local mode and accepts requests without auth.

如果 `config.json` 被截断 / 不是合法 JSON → 中间件 **fail closed 返回 503**（不会回退到 keyless）。<br>
If `config.json` is corrupt/unreadable → the middleware **fails closed with 503** instead of falling through to keyless.

---

## Provider 登录 / Logging in providers

统一命令：`uv run opentoken login <provider>`。<br>
Single command: `uv run opentoken login <provider>`.

### 方式 A：浏览器登录 / Browser-based

会打开真实 Firefox（Camoufox）让你登录，凭证保存到 `~/.opentoken/providers/<provider>.json`。<br>
Launches a real (Camoufox) Firefox, you log in, the cookies/headers land in `~/.opentoken/providers/<provider>.json`.

```bash
uv run opentoken login qwen international --browser
uv run opentoken login qwen china         --browser
uv run opentoken login deepseek           --browser
uv run opentoken login kimi               --browser
uv run opentoken login doubao             --browser
uv run opentoken login glm international  --browser
uv run opentoken login glm china          --browser
uv run opentoken login claude             --browser
uv run opentoken login chatgpt            --browser
uv run opentoken login gemini             --browser
uv run opentoken login grok               --browser
uv run opentoken login mimo               --browser
```

> 登录有 dry-run 校验：如果旧凭证仍然有效，新捕获到的凭证必须通过认证 probe 才会覆盖旧的，避免一次失败的 harvest 把可用 cookie 替换成坏 cookie。首次登录跳过 probe。<br>
> Login is dry-run validated: if existing credentials still work, freshly captured ones must pass an authenticated probe before they replace the working pair. First-time login skips the probe.

### 方式 B：手工凭证 / Manual cookie / header

```bash
uv run opentoken login qwen international \
  --cookie 'your_cookie_here' \
  --user-agent 'your user agent'

uv run opentoken login deepseek --header 'authorization=Bearer xxx'

uv run opentoken login some-provider \
  --header 'authorization=Bearer xxx' \
  --header 'x-token=yyy'
```

### 方式 C：API key / Direct API key

```bash
uv run opentoken login manus --api-key YOUR_KEY
uv run opentoken login nim   --api-key nvapi-XXXXXXXXXXXXXXXXXXXX
```

NIM 凭证文件支持可选 `model_chain` 跨模型 fallback —— 被 429 限流的模型自动切到链表下一个，调用方完全无感。

NIM credentials accept an optional `model_chain` for cross-model rate-limit fallback — a 429 on the active model transparently switches to the next id in the list.

```json
{
  "kind": "api_key",
  "metadata": {
    "api_key": "nvapi-XXXXXXXXXXXXXXXXXXXX",
    "model_chain": "[\"deepseek-ai/deepseek-r1\", \"meta/llama-3.3-70b-instruct\", \"qwen/qwen2.5-72b-instruct\"]"
  },
  "status": "valid"
}
```

Fallback 对流式也生效：`stream_with_chain` 会 prime 第一个 chunk，所以即使是 lazy generator 在首字节才检测到 429，也能正确切到链表里下一个模型。

Streaming respects the chain too: the first chunk is primed so a 429 detected lazily on the first byte still triggers the next candidate.

### 方式 D：Unified Proxy (LiteLLM)

```bash
uv sync --extra unified

uv run opentoken login unified \
  --header api_key_openrouter=sk-or-XXXXXXXXX \
  --header api_key_anthropic=sk-ant-XXXXXXXX \
  --header api_key_groq=gsk_XXXXXXXX
```

调用模型形如 `unified/<backend>/<model>`：<br>
Then call with `unified/<backend>/<model>`:

- `unified/openrouter/anthropic/claude-3.5-sonnet`
- `unified/groq/llama-3.3-70b-versatile`
- `unified/together/qwen/qwen2.5-coder-32b-instruct`

> 流式 + 工具调用尚未支持 / **streaming + tool_calls not supported**：unified 的流式接口当前用 `Iterator[str]` 不能承载结构化 tool_call delta；若 backend 在流里发出 tool_calls，opentoken 会显式抛错让你用 `stream=false` 重试，而不是静默吃掉它。<br>
> The unified streaming interface yields plain strings, which can't carry OpenAI-spec structured tool_call deltas. Rather than silently drop them, opentoken raises a clear "retry with stream=false" error.

### 查看状态 / Listing & logout

```bash
uv run opentoken providers
uv run opentoken logout qwen international
```

---

## 启动后验证 / Verifying the gateway

```bash
curl http://127.0.0.1:32117/health      # → {"status":"ok"}
uv run opentoken status                 # 服务状态 / quick service status
uv run opentoken doctor                 # 系统诊断 / system diagnostics
uv run opentoken verify                 # 接口契约验证 / contract checks
```

跨 provider E2E 烟雾测试 —— 另起一个终端：<br>
Cross-provider E2E smoke — open a second terminal:

```bash
uv run python scripts/live_provider_smoke.py
```

会对每个已登录 provider 跑一次非流 + 一次流 + 一次 /v1/responses，结果写到 `live_provider_smoke_report.json`。<br>
Runs non-stream + stream + `/v1/responses` against every logged-in provider; per-provider pass/fail + first-byte latency lands in `live_provider_smoke_report.json`.

---

## OpenAI 兼容调用 / OpenAI-compatible API

下面所有 `Authorization: Bearer ...` 都是**本地网关 API key**。<br>
`Authorization: Bearer ...` is the **local gateway key** below.

假设 / Assume:
- base URL `http://127.0.0.1:32117/v1`
- local api key `YOUR_LOCAL_GATEWAY_KEY`

### 1) `GET /v1/models`

```bash
curl http://127.0.0.1:32117/v1/models \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY'
```

模型 id 形如：<br>
Model ids look like:

- `algae/qwen-intl/qwen3.6-plus`
- `algae/qwen-cn/Qwen3.5-Flash`
- `algae/deepseek/deepseek-chat`
- `algae/nim/deepseek-ai/deepseek-r1`
- `algae/unified/openrouter/anthropic/claude-3.5-sonnet`

> `algae/` 是外部 OpenClaw 客户端的命名空间前缀（保留作为线协议契约）。你直接传整个 id 即可。<br>
> The `algae/` prefix is an external-client namespace tag (kept as a wire-format contract). Just pass the full id.

模型别名是**大小写不敏感**的：`Qwen-3.5-Turbo` 和 `qwen-3.5-turbo` 都会被规范化成 `qwen3.5-flash`。<br>
Alias resolution is **case-insensitive**: both `Qwen-3.5-Turbo` and `qwen-3.5-turbo` normalize to `qwen3.5-flash`.

### 2) `POST /v1/chat/completions` —— 非流式 / Non-streaming

```bash
curl http://127.0.0.1:32117/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "algae/qwen-intl/qwen3.6-plus",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

支持的可选参数 / Honored optional params: `temperature`, `max_tokens`, `top_p`, `tools`, `tool_choice`（流式 + tools 时部分 provider 会自动回退到非流式以保留工具结构化输出）。

### 3) `POST /v1/chat/completions` —— 流式 / Streaming

```bash
curl http://127.0.0.1:32117/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY' \
  -H 'Content-Type: application/json' \
  -N \
  -d '{
    "model": "algae/qwen-intl/qwen3.6-plus",
    "stream": true,
    "messages": [{"role": "user", "content": "来一段3000字自我介绍"}]
  }'
```

流的 SSE 协议：第一条和最后一条 chunk 带 `system_fingerprint`；末尾 chunk 之后接 `[DONE]`；若 provider 输出 `<tool_calls>` 协议块，opentoken 会在流末尾回填一段标准 OpenAI `tool_calls` delta 并把 `finish_reason` 设为 `tool_calls`。<br>
SSE conventions: first/last chunks carry `system_fingerprint`; the final `usage` chunk is followed by `[DONE]`; a `<tool_calls>` protocol block in the provider output is back-filled as a standard OpenAI `tool_calls` delta with `finish_reason="tool_calls"`.

`<think>` 标签：推理模型（含 `reasoner` / `thinking` / `-think` 关键字）在流式里**保留** `<think>...</think>` 标签让客户端实时看到推理过程；非流式响应里**自动剥离**只返回最终答案。这是有意为之、并被测试钉住的行为差异。<br>
`<think>` tags: for reasoner models the streaming path **keeps** `<think>...</think>` markup (clients can show live reasoning), while the non-stream path **strips** it. This is an intentional, test-pinned divergence.

### 4) `POST /v1/responses`（OpenAI Responses API）

```bash
curl http://127.0.0.1:32117/v1/responses \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"model":"algae/qwen-intl/qwen3.6-plus","input":"你好，写一个摘要"}'
```

带 `previous_response_id` 续聊时，新请求里的 `instructions` 会被**置于上下文最前**而不是拼在历史尾部，避免活跃的系统提示被模型忽略。<br>
When continuing via `previous_response_id`, the new request's `instructions` are **hoisted to the front** of the model context — otherwise the active system prompt would land after the entire prior conversation and be largely ignored.

`max_output_tokens` 自动映射成 chat completions 风格的 `max_tokens`。<br>
`max_output_tokens` is mapped onto the unified `max_tokens` field.

### 5) `POST /v1/files`、`POST /v1/uploads` —— 文件上传 / File upload

```bash
# 一次性上传 / single-shot
curl -X POST http://127.0.0.1:32117/v1/files \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY' \
  -F 'purpose=assistants' \
  -F 'file=@./report.pdf'

# 分块上传 / multipart upload
curl -X POST http://127.0.0.1:32117/v1/uploads \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"filename":"big.bin","bytes":52428800,"purpose":"assistants","mime_type":"application/octet-stream"}'

# 上传每个 part / send each part
curl -X POST http://127.0.0.1:32117/v1/uploads/<upload-id>/parts \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY' \
  -F 'data=@./big.bin.part1'

curl -X POST http://127.0.0.1:32117/v1/uploads/<upload-id>/complete \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY' \
  -H 'Content-Type: application/json' \
  -d '{}'
```

**大小上限 / Size limits**：每个 part 100 MiB，整个 upload 声明的 `bytes` 也 ≤ 100 MiB（complete 阶段会把所有 parts 一次性拼到内存）。每个 part 进来时会校验"已有 parts 字节 + 本 part > 声明 bytes"，超了直接 413 拒绝 —— 防止用大量 part 累积绕过单 part 上限把 worker OOM 掉。<br>
Each part is capped at 100 MiB; the declared total `bytes` of an upload is also capped at 100 MiB (complete concatenates all parts in memory). Each incoming part is rejected with 413 if `existing_parts_bytes + new_part > declared bytes` — closing the OOM vector of an unbounded part count.

**Content-Type 服务 / Serving uploaded content**：`GET /v1/files/{id}/content` **始终**回 `application/octet-stream` + `X-Content-Type-Options: nosniff` + `Content-Disposition: attachment`，**不**回显上传时声明的 mime_type，防止上传一份 `text/html` 或 `image/svg+xml` 在浏览器里渲染成 stored XSS。<br>
`GET /v1/files/{id}/content` **always** returns `application/octet-stream` + `nosniff` + `attachment`, never the caller-supplied mime_type, so an uploaded text/html or SVG blob can't execute as stored XSS in a browser-reachable deployment.

### 6) `POST /v1/embeddings` —— 当前 501 / Currently 501

返回 501 `not_implemented`。早期实现是 SHA-256 派生的伪向量（256 维、无归一化、忽略 model 名），把它当 RAG 向量数据源会得到高熵噪声。把 embedding 流量路由到真实 backend（自托管 sentence-transformer / NIM 的 embedding 模型 / OpenAI 等）。<br>
Returns 501 `not_implemented`. The previous implementation produced SHA-256-derived pseudo-vectors that would have polluted any real RAG/vector store. Route embedding traffic to a real backend instead.

---

## 错误分类 / Error classification

`/v1/chat/completions` 与 `/v1/responses` 共用同一个分类器：<br>
Both routes share a single classifier:

| 上游错误 / Upstream error | 网关返回 / Gateway returns |
|--------------------------|-----------------------------|
| 缺少 / 失效凭证、session 过期、re-login 提示、`unauthenticated` (Kimi gRPC)、`no chat id` (Qwen) | **401** `authentication_error` |
| 上游 4xx 业务错（rate limit 除外） | **400** `invalid_request_error` |
| 上游 429 | **429** `rate_limit_error` |
| 上游 5xx / 解析失败 / 网关侧异常 | **502** `api_error` |
| `Unsupported model` / `No route configured` | **400** `invalid_request_error` |

流式时同一分类也作用于 SSE `error` 事件，所以一个上游 502 在流中间发生不会被错误标记为 `invalid_request_error`。<br>
The same classification applies inside the SSE `error` event mid-stream, so an upstream 502 mid-flight isn't mislabeled as `invalid_request_error`.

bare `expired` 字符串不会再单独触发 401 —— 必须配合 `session expired` / `credentials expired` / `token expired` 等明确的 auth subject 才映射为 401，避免 "upstream certificate expired" 之类的上游错被误判成 auth 问题。<br>
A bare `expired` substring no longer triggers 401 — only `session/credentials/token expired` and similar auth phrases do; "upstream certificate expired" and the like correctly stay 502.

---

## 注入到外部 OpenClaw 配置 / Bridge to OpenClaw config

```bash
uv run opentoken config --dry-run                         # 预览 / preview
uv run opentoken config                                   # 写入 / write
uv run opentoken config --opentoken-config /path/to/openclaw-config.json
```

写入是原子的（tmp + os.replace），并 chmod 0600（patch 中包含本地网关 apiKey）。<br>
Atomic write (tmp + os.replace) + chmod 0600 — the patch contains the gateway apiKey.

---

## 常见问题 / Troubleshooting

### `/v1/models` 是空的 / Empty model list

```bash
curl http://127.0.0.1:32117/health
uv run opentoken providers
```

某个 provider 没有有效凭证就不会出现在 `/v1/models`。<br>
A provider without valid credentials simply doesn't appear.

### 一直 401 / Always 401

检查你传的是不是 **本地网关 key**（不是 provider 自己的 key）：<br>
Make sure the `Authorization: Bearer` is the **local gateway key**, not an upstream provider key:

```bash
cat ~/.opentoken/config.json
```

### 流式输出不稳定 / Flaky streams

先用 `curl -N` 直接测，区分网关本地问题（直接失败 / 超时 / 协议不完整）和上游限流（明确的 rate limit / error 事件）。<br>
First isolate with `curl -N` — local gateway issues fail fast/timeout, upstream throttles surface as explicit rate-limit/error events.

### 浏览器 provider 报 `NS_ERROR_PROXY_CONNECTION_REFUSED`

doubao / glm-cn / glm-intl / qwen-cn 走真实 Firefox（Camoufox），它继承**系统级**代理设置（不受 `HTTP_PROXY` env 影响）。如果系统代理当前连不通就会报这个错。<br>
The Camoufox-backed providers read the **OS-level** proxy config (independent of `HTTP_PROXY`), so an unreachable system proxy surfaces this error. Pure-HTTP providers (deepseek/nim/unified/manus/kimi) aren't affected.

排查 / Fix: 关掉系统代理或确保代理可达，重试浏览器 provider。这是环境问题，不是网关 bug。<br>
Disable / fix the system proxy, then retry.

### 跨平台的 composer 清空

浏览器 provider 在每次发送前会 select-all + Backspace 清空草稿。这个 chord 在 macOS 是 `Meta+A`，在 Linux/Windows 是 `Control+A` —— 由 `sys.platform` 自动选择，所以同一 codebase 在 Linux server / Docker 上不会因为草稿残留而把旧消息和新消息拼起来发出去。<br>
Browser providers clear the composer with select-all + Backspace before each send. The chord is `Meta+A` on macOS and `Control+A` elsewhere, picked from `sys.platform` at import — same codebase, no stale-draft concatenation on Linux servers.

### 切换监听地址 / Change bind address

```bash
uv run opentoken start --host 0.0.0.0 --port 32117   # 会有非 loopback 警告 / prints LAN-exposure warning
```

---

## 开发与测试 / Development & tests

```bash
./.venv/bin/pytest                      # 全套 / full suite
./.venv/bin/pytest -k stream            # 流式相关 / streaming tests
./.venv/bin/pytest tests/providers/     # provider 单测 / provider unit tests
./.venv/bin/pytest tests/storage/       # 存储原子性 / atomicity + permission tests
```

当前测试总数：**640+**（包括 5 轮回归审查累计 +80 余条新回归用例）。<br>
Current test count: **640+** including 80+ regression tests added across five review waves.

---

## 提交到 Git 前的安全注意 / Git hygiene

`.gitignore` 已经优先忽略 `.venv/`、`.opentoken/`、本地 agent 规划文件、`tmp/`、`*.log`。但永远不要提交：<br>
`.gitignore` already excludes `.venv/`, `.opentoken/`, planning notes, `tmp/`, `*.log`. Never commit:

- `~/.opentoken/` 任何内容 / anything from `~/.opentoken/`
- 导出的 cookie / header / bearer token 文件 / exported cookies/headers/tokens
- `.env` / `.env.*`
- 含 token 的临时抓包或调试日志 / debug logs containing tokens

提交前先 `git status` 看一眼，发现凭证文件就停下。<br>
Always `git status` first; bail if any credential file is staged.

---

## License

MIT License — 见 `LICENSE` / see `LICENSE`.
