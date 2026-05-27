# OpenToken

一个把多家网页端 / 浏览器登录态 / 少量 API 能力统一封装成 **OpenAI-compatible 本地网关** 的项目。

你可以把它当成一个本地中间层来用：
- 上游还是各家 provider 的网页端 / 登录态 / API
- 下游统一用 OpenAI 风格接口接入
- 本地暴露 `/v1/models`、`/v1/chat/completions`、`/v1/responses`、`/v1/embeddings`

---

## 它能解决什么问题

如果你同时在用 DeepSeek、Qwen、Kimi、Doubao、GLM、Claude、Gemini、ChatGPT 等不同 provider，通常会遇到这些问题：

- 每家登录方式不一样
- 每家模型名不一样
- 每家接口格式不一样
- 流式输出行为不一致
- 本地工具 / OpenToken / OpenAI-compatible 客户端接起来很麻烦

OpenToken 做的就是：

1. 统一管理 provider 凭证
2. 暴露一个本地 OpenAI-compatible 网关
3. 把不同 provider 的模型都映射到统一调用方式
4. 尽量把流式输出、工具调用、响应结构对齐到 OpenAI 风格

---

## 当前支持的 provider

**网页登录态 / 浏览器采集（11 个）**

- DeepSeek
- Qwen International / Qwen China
- Kimi
- Claude
- Doubao
- ChatGPT
- Gemini
- Grok
- GLM International / GLM China
- Xiaomi Mimo

**API key 直连（4 个）**

- Manus（官方 API）
- **NVIDIA NIM** — 走 `integrate.api.nvidia.com/v1`，免费 40 RPM，覆盖 DeepSeek R1、Llama 3.3 70B、Qwen 2.5 72B / Coder 32B、Mixtral 8x22B 等。注册一个 NVIDIA 账号拿 `nvapi-...` key 就能用，0 信用卡。
- **Unified Proxy (LiteLLM)** — 一个 adapter 接 100+ 后端：OpenRouter / Groq / Together / Bedrock / Anthropic / OpenAI / Perplexity / Cohere / Mistral / xAI / Fireworks / DeepInfra / Azure / Ollama / LM Studio …。LiteLLM 是软依赖，要用时 `uv sync --extra unified` 装上。

> 实际可用模型取决于你当前已经登录/配置成功的 provider。`/v1/models` 只会列出你已经登录、可被路由的那些。

---

## /v1/models 不再写死，所有模型实时发现

opentoken 早期版本在 `models/catalog.py` 里写死了一份 33 项的模型清单，作为 `/v1/models` 的数据源。这有两个根本问题：

1. **会过时** —— Qwen3.7-Max、GLM-5.1、GLM-5-Turbo、doubao-thinking 这些上游已经上线的模型出不来，反过来 `glm-4-plus`、`qwen3.5-turbo` 这种早就下线的模型还在响应里
2. **会撒谎** —— 即便用户没登录任何 provider，`/v1/models` 也会展示 33 个模型，所有第一次调用都会 401

现在 `/v1/models` 完全由 `opentoken.models.discovery` 动态产生，每个 provider 独立查询：

| Provider | 发现方式 |
|----------|---------|
| qwen-intl / qwen-cn / doubao / glm-cn | 抓自上游 web 页面或 dialog（含 Camoufox 浏览器抓取） |
| **glm-intl** | `GET chat.z.ai/api/models`，失败回退抓首页 |
| **deepseek** | `GET /api/v0/users/current` 校验凭证有效，返回协议支持的两个 wire 模型 |
| **kimi** | 抓 kimi.com 首页嵌入的 model metadata |
| **nim** | `GET integrate.api.nvidia.com/v1/models` Bearer auth（标准 OpenAI 协议） |
| **manus** | `GET api.manus.im/api/v1/agents` |
| **chatgpt** | `GET /backend-api/models`，失败回退抓首页 |
| **claude** | `GET /api/organizations` + statsig chat-models 动态配置 |
| **gemini** | 抓 gemini.google.com app HTML |
| **grok** | 抓 grok.com 首页 HTML |
| **mimo** | 抓 xiaomimo.com 首页 HTML |
| **unified** | 按凭证里配置的 backend 过滤 `litellm.model_cost` |

每个发现器都是软失败 —— 拿不到就返回 `[]`，那个 provider 的模型在 `/v1/models` 里就不会出现。结果 cache 在 `~/.opentoken/model-catalog-cache.json`，TTL 6 小时。第一次启动后续请求几乎零延迟，凭证变更会自动失效缓存。

**实测对比**（同一台机器，7 个已登录 provider）：

| Provider | 旧硬编码数量 | 现 live 发现数量 | 差异 |
|----------|-------------|----------------|------|
| glm-intl | 2 | 14 | 多出 GLM-5.1 / GLM-5-Turbo / glm-4.7 / glm-4.6v / deep-research 等 |
| qwen-cn | 5 | 6 | 多出 Qwen3.6-千问 / Qwen3.7-Max，对齐了新发布 |
| doubao | 2 | 3 | 多出 doubao-thinking |
| glm-cn | 2 | 1 | 实际只有 glm-5 在线（淘汰了 glm-4 系列） |
| deepseek | 2 | 2 | 一致 |

> 如果某个 provider 的 web 页面改版导致发现器抓不到，模型暂时不展示是**符合预期的诚实行为**。修发现器（更新 regex 或换 API 端点）是后续 PR 的事。

---

## 最近的功能与稳定性改进

- **新 provider：NVIDIA NIM**（`opentoken login nim --api-key nvapi-...`）+ 跨模型自动降级链。在 metadata 里配 `model_chain` 后，被 rate-limit 的模型会自动切到链表里的下一个，调用方完全无感。
- **新 provider：Unified Proxy (LiteLLM)**（`opentoken login unified --header api_key_openrouter=sk-or-...`），model id 走 `unified/<backend>/<model>`。
- **安全修复**：
  - 关掉 `/v1/chat/completions` 附件加载的 SSRF + 任意本地文件读漏洞（block file://、私网/loopback/metadata IP、禁止 301-308 重定向跨域绕过）。
  - 本地 API key 校验改为 `hmac.compare_digest` 常量时间比较。
  - 全局异常处理脱敏：cookie / authorization / 内部堆栈不再泄漏到 500 响应，请求加 `X-Request-Id` 头便于追踪。
  - `/v1/files` 上传走分块读取 + 100 MiB 大小上限，避免单请求打爆内存。
- **正确性**：
  - 修复 Grok 跨用户会话串扰（之前会复用账户里任意一条历史 conversation）。
  - 修复 Qwen `_ensure_chat_id` 在 401 时静默失败导致后续请求带空 chat_id。
  - 修复流式 + tool_calls 完全失效（projector 之前会把 `<tool_calls>` 标签静默吃掉）。
  - 修复 finish_reason 永远硬编码 "stop"。
  - 修复 failover 把 401 / 403 当 retryable（凭证过期靠重试救不回来）。
  - 所有 provider client cache 改成有界 LRU，给 Qwen-Intl / Doubao / GLM-Intl 补上之前缺失的 cache。
- **存储**：所有 JSON 持久化改为 `tmpfile + os.replace + flock` 原子写，response_store 加了 TTL + LRU，upload_store 合并后清理 parts，file_store 校验 file_id 防路径遍历。
- **协议对齐**：
  - `usage` 不再恒为 0，按 ASCII / CJK 字符做轻量估算。
  - 非流式 + 流式响应都加 `system_fingerprint=fp_opentoken_v1`。
  - tool_calls 存在时 content 强制为 null（符合 OpenAI 规范）。
- **`/v1/embeddings` 改成 501 not_implemented**。原实现是 SHA-256 派生的伪向量（256 维、无归一化、忽略 model 名），把它当 RAG / 向量检索数据源会得到高熵噪声。现在直接返回 501，让上层路由到真实 backend（NIM / 你自己的 sentence-transformer）。
- **新增 E2E 烟雾脚本**：`scripts/live_provider_smoke.py`，对每个已登录 provider 跑一次非流 + 流的 chat completion，写一份 JSON 报告。比 200-case suite 快 10 倍以上。

### 第二轮加固（在上述基础上进一步打磨）

- **凭证文件 0600 权限**：`~/.opentoken/config.json`、`~/.opentoken/providers/*.json`、`~/.opentoken/auth-profiles.json` 写入时强制 owner-only，多用户主机上别人没法直接 `cat` 出 cookie / api key。
- **JSON body 大小上限**：`/v1/chat/completions` / `/v1/responses` 等 JSON 端点加 25 MiB 上限（基于 Content-Length 头），拒掉用大 body 打爆 worker 内存的 DoS 尝试。`/v1/uploads/{id}/parts` 单 part 100 MiB cap + 声明 bytes ≤ 8 GiB。
- **401 一致契约**：8 个 HTTP provider（claude、chatgpt、gemini、grok、mimo、kimi、qwen-intl、qwen-cn）的上游 401/403 现在统一映射成 401 `authentication_error`（提示 `opentoken login <provider>` 重登），而不是过去那种 502 `api_error`。
- **错误分类器**：`/v1/chat/completions` 和 `/v1/responses` 共享 `classify_provider_runtime_error`：`Unsupported model` → 400；缺凭证 / session expired → 401；其余上游失败 → 502。
- **请求 body 校验提前**：空 `messages` / 非 list `messages` 在请求归一化阶段就 400 拒绝，不再让它沉到 provider 里返 502。
- **/v1/responses 流式 error 事件平字段**：`{"type":"error","code":..,"message":..,"param":..}`（符合 OpenAI Responses SSE 协议），不再嵌套 `error` 对象。
- **token usage 估算覆盖 responses**：`/v1/responses` 的 `usage` 也从硬编码 0 改成 char-based 估算，和 `/v1/chat/completions` 一致。
- **`monotonic` 时钟**：50+ 处 polling/stream deadline 从 `time.time()` 改为 `time.monotonic()`，NTP / 手动调时不再造成提前超时或永远不超时。
- **NIM 推理透传**：DeepSeek R1 等模型的 `reasoning_content` 字段不再被丢弃，会被包进 `<think>...</think>` 经 projector 路径正常处理。
- **跨模型 fallback 流式可用**：`stream_with_chain` 现在 prime 第一个 chunk —— lazy generator 在首字节出来时才打的 429 也能正确触发链表降级（之前会漏 fallback）。
- **discovery 并发跑**：`/v1/models` 冷缓存从 >30s 超时 → ~16s（瓶颈是 qwen-cn 浏览器启动，HTTP discoverers 并发完成）。缓存写入原子化。
- **`opentoken verify` 并发跑**：每 provider 独立线程，单个慢 provider 不再阻塞整轮验证。
- **登录 dry-run 接线**：浏览器登录时若已有有效凭证，新捕获的凭证必须通过 `verification/credentials_probe` 的 authenticated probe 才能覆盖，避免半成 harvest 破坏现有可用 cookie；首次登录跳过 probe。
- **Camoufox recovery 目录清理**：Firefox 占用错误时新建的 `<name>-recovery-<ms>` profile 目录，超 1 小时未用的会被回收，不再无限堆积。
- **dead code 清理**：删掉 `streaming.py` 里 5 个永远没人调的（且逻辑本身有 bug 的）heartbeat helper。

---

## 环境要求

- Python `>= 3.13`
- 推荐使用 [`uv`](https://docs.astral.sh/uv/)

---

## 目录与本地状态

项目本身不会把登录凭证写进仓库，而是写到用户目录下：

- `~/.opentoken/config.json`：本地网关配置（包含本地 API key / host / port）
- `~/.opentoken/providers/*.json`：各 provider 凭证
- `~/.opentoken/`：其它本地运行状态

这意味着：
- **仓库代码可以提交**
- **本地登录态不会默认进 Git**
- 只要 `.gitignore` 保持正常，就不会把你的 cookie / header / api key 一起传上去

---

## 快速开始

### 1）安装依赖

```bash
uv sync
```

### 2）初始化本地状态目录（可选，但推荐）

```bash
uv run opentoken onboard
```

### 3）启动服务

```bash
uv run opentoken start
```

默认监听：

- `http://127.0.0.1:32117`

默认 OpenAI-compatible base URL：

- `http://127.0.0.1:32117/v1`

---

## 本地 API key 在哪看

OpenToken 使用的是**本地网关自己的 API key**，不是上游 provider 的 key。

默认保存在：

- `~/.opentoken/config.json`

你可以直接查看：

```bash
cat ~/.opentoken/config.json
```

示例内容：

```json
{
  "api_key": "your-local-gateway-key",
  "host": "127.0.0.1",
  "port": 32117
}
```

如果你只想快速取出 API key：

```bash
python3 - <<'PY'
import json, os
path = os.path.expanduser('~/.opentoken/config.json')
with open(path, 'r', encoding='utf-8') as f:
    print(json.load(f)['api_key'])
PY
```

---

## Provider 登录方式

登录命令统一是：

```bash
uv run opentoken login <provider>
```

### 方式 A：浏览器登录（推荐）

适合支持浏览器采集登录态的 provider。

示例：

```bash
uv run opentoken login qwen international --browser
uv run opentoken login qwen china --browser
uv run opentoken login deepseek --browser
uv run opentoken login kimi --browser
uv run opentoken login doubao --browser
uv run opentoken login glm international --browser
uv run opentoken login glm china --browser
```

说明：
- 会打开对应网页登录流程
- 登录成功后，凭证会保存到 `~/.opentoken/providers/*.json`

### 方式 B：手工凭证登录

适合你已经有 cookie / header 的情况。

示例：

```bash
uv run opentoken login qwen international \
  --cookie 'your_cookie_here' \
  --user-agent 'your user agent'
```

或者：

```bash
uv run opentoken login deepseek \
  --header 'authorization=Bearer xxx'
```

也可以多个 header：

```bash
uv run opentoken login some-provider \
  --header 'authorization=Bearer xxx' \
  --header 'x-token=yyy'
```

### 方式 C：API key 登录

适合 provider 本身支持 `--api-key` 的情况，例如 Manus、NIM。

```bash
uv run opentoken login manus --api-key YOUR_KEY
uv run opentoken login nim --api-key nvapi-XXXXXXXXXXXXXXXXXXXX
```

NIM 的凭证文件 `~/.opentoken/providers/nim.json` 里 metadata 还支持可选的 `model_chain`（JSON 编码的字符串数组），用于跨模型 rate-limit fallback：

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

被请求的模型会先试，如果它返回 429 / `ProviderRateLimitError`，自动切到链表里下一个继续试，整个 fallback 对调用方透明。

### 方式 D：Unified Proxy (LiteLLM)

```bash
uv sync --extra unified  # 软依赖：仅在用 unified 时装

uv run opentoken login unified \
  --header api_key_openrouter=sk-or-XXXXXXXXX \
  --header api_key_anthropic=sk-ant-XXXXXXXX \
  --header api_key_groq=gsk_XXXXXXXX
```

调用时 model 形如：

- `unified/openrouter/anthropic/claude-3.5-sonnet`
- `unified/groq/llama-3.3-70b-versatile`
- `unified/together/qwen/qwen2.5-coder-32b-instruct`

`unified/<backend>/<model>` 中 `<backend>` 是 LiteLLM 的 provider 名，后面是上游模型 id。Bearer token 还是本地 gateway key；上游 key 从凭证文件取并临时注入到对应环境变量（OPENROUTER_API_KEY / GROQ_API_KEY / …）执行完即清理。

### 查看当前 provider 状态

```bash
uv run opentoken providers
```

### 登出某个 provider

```bash
uv run opentoken logout qwen international
```

---

## 启动后怎么验证服务正常

### 健康检查

```bash
curl http://127.0.0.1:32117/health
```

期望返回：

```json
{"status":"ok"}
```

### 查看服务状态

```bash
uv run opentoken status
```

### 跑诊断

```bash
uv run opentoken doctor
```

### 跑接口契约验证

```bash
uv run opentoken verify
```

### 跑跨 provider E2E 烟雾测试

启动网关后，另起一个终端：

```bash
uv run python scripts/live_provider_smoke.py
```

会对每个已登录 provider 跑一次非流 + 一次流式 chat completion，并把 per-provider 通过/失败、首字延迟写到 `live_provider_smoke_report.json`。

---

## OpenAI-compatible 调用方式

下面所有示例里的 `Authorization: Bearer ...`，都应该填**本地网关 API key**。

假设：

- base URL：`http://127.0.0.1:32117/v1`
- local api key：`YOUR_LOCAL_GATEWAY_KEY`

---

### 1）获取模型列表

```bash
curl http://127.0.0.1:32117/v1/models \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY'
```

返回结果里模型名格式通常类似：

- `algae/qwen-intl/qwen3.6-plus`
- `algae/qwen-cn/Qwen3.5-Flash`
- `algae/deepseek/deepseek-chat`
- `algae/nim/deepseek-ai/deepseek-r1`
- `algae/unified/openrouter/anthropic/claude-3.5-sonnet`

> `algae/` 前缀是给外部 OpenClaw 客户端用的命名空间标识，并非项目名。普通调用直接传整个 id 即可，opentoken 内部会正确解析多段模型名。

---

### 2）普通 Chat Completions

```bash
curl http://127.0.0.1:32117/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "algae/qwen-intl/qwen3.6-plus",
    "messages": [
      {"role": "user", "content": "你好，介绍一下你自己"}
    ]
  }'
```

---

### 3）流式 Chat Completions

```bash
curl http://127.0.0.1:32117/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY' \
  -H 'Content-Type: application/json' \
  -N \
  -d '{
    "model": "algae/qwen-intl/qwen3.6-plus",
    "stream": true,
    "messages": [
      {"role": "user", "content": "来一个3000字自我介绍"}
    ]
  }'
```

如果你想观察首包是不是实时到达，可以直接看终端里 SSE 持续输出，而不是等完整响应结束后一次性返回。

流式 SSE 中：
- `system_fingerprint` 字段会出现在第一条和最后一条 chunk
- `usage` 在末尾 chunk（`finish_reason` 不为 null 的那条）后续接 `[DONE]`
- 如果 provider 返回 `<tool_calls>` 协议块，opentoken 会在流结束时回填一段标准 OpenAI tool_calls delta，并把 `finish_reason` 改为 `tool_calls`

---

### 4）Embeddings

**当前 `/v1/embeddings` 返回 501 `not_implemented`。** opentoken 不再以伪向量伪装实现 embedding 接口；建议把 embedding 流量路由到一个真实 backend（自托管 sentence-transformer / NIM 的 embedding 模型 / OpenAI 等），等真实代理接入后再恢复。

---

### 5）Responses API

```bash
curl http://127.0.0.1:32117/v1/responses \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "algae/qwen-intl/qwen3.6-plus",
    "input": "你好，写一个摘要"
  }'
```

---

### 6）流式 Responses API

```bash
curl http://127.0.0.1:32117/v1/responses \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY' \
  -H 'Content-Type: application/json' \
  -N \
  -d '{
    "model": "algae/qwen-intl/qwen3.6-plus",
    "stream": true,
    "input": "来一个3000字自我介绍"
  }'
```

---

## 把 opentoken 注入到外部 OpenClaw 配置

`opentoken config` 是一条 bridge 命令，把当前网关写成一段 OpenClaw 格式的 provider patch（provider 名沿用历史的 `algae`），方便外部 OpenClaw 客户端接到这个本地网关上。

先看将要写入的配置：

```bash
uv run opentoken config --dry-run
```

确认后正式写入：

```bash
uv run opentoken config
```

如果你要写到自定义 OpenClaw 配置文件：

```bash
uv run opentoken config --opentoken-config /path/to/openclaw-config.json
```

---

## 推荐的日常使用流程

### 最短路径

```bash
uv sync
uv run opentoken login qwen international --browser
uv run opentoken start
```

然后：

```bash
curl http://127.0.0.1:32117/v1/models \
  -H 'Authorization: Bearer <你的本地网关APIKey>'
```

---

## 常见问题 / 排障

### 1）`/v1/models` 获取不到模型

先检查服务是否启动：

```bash
curl http://127.0.0.1:32117/health
```

再检查 provider 是否已经登录：

```bash
uv run opentoken providers
```

如果某个 provider 没有有效凭证，它对应模型通常不会出现在模型列表里。

### 2）服务能起来，但请求报 401

大概率是你传的 `Authorization Bearer` 不是本地网关 API key。

重新查看：

```bash
cat ~/.opentoken/config.json
```

### 3）某些 provider 流式不稳定

先区分两种情况：

- **网关本地问题**：通常会表现为请求直接失败、超时、或者协议不完整
- **上游限流 / 风控**：通常会返回明确的 rate limit / error 事件

建议先用 `curl -N` 直接测：

```bash
curl http://127.0.0.1:32117/v1/responses \
  -H 'Authorization: Bearer YOUR_LOCAL_GATEWAY_KEY' \
  -H 'Content-Type: application/json' \
  -N \
  -d '{
    "model": "algae/qwen-intl/qwen3.6-plus",
    "stream": true,
    "input": "来一个3000字自我介绍"
  }'
```

### 4）登录态过期了怎么办

重新登录对应 provider：

```bash
uv run opentoken login qwen international --browser
```

### 5）想切换监听地址和端口

```bash
uv run opentoken start --host 0.0.0.0 --port 32117
```

### 6）NIM 一直被 rate-limit

给 `~/.opentoken/providers/nim.json` 的 metadata 加 `model_chain` 字段（见上文 "API key 登录" 一节）。当请求的模型被 429 时，opentoken 会自动切到链表里下一个模型继续试，客户端透明。

### 7）想接 OpenRouter / Groq / Together 等 OpenAI-compatible 后端

用 unified provider：`opentoken login unified --header api_key_<backend>=...`，model 走 `unified/<backend>/<model>` 即可。详见 "方式 D"。

### 8）`/v1/embeddings` 返回 501

预期行为。详见 "OpenAI-compatible 调用方式 → Embeddings"。

---

## 开发与测试

运行全部测试：

```bash
./.venv/bin/pytest
```

运行流式相关验证：

```bash
./.venv/bin/pytest tests/e2e/test_http_e2e.py -k stream
```

运行 qwen 相关验证：

```bash
./.venv/bin/pytest tests/providers/test_http_providers.py -k qwen
```

运行新加的 NIM / unified / model_chain / bounded cache / dry-run 测试：

```bash
./.venv/bin/pytest tests/providers/test_nim.py tests/providers/test_unified_proxy.py tests/providers/test_client_cache.py tests/storage/test_provider_store_validation.py tests/api/test_auth_timing.py -v
```

当前单测总数：**453 个全过**。

---

## 上传到 GitHub 前的安全注意事项

本项目已经在 `.gitignore` 里优先忽略：

- `.venv/`
- `.opentoken/`
- `.opentoken/`
- `tmp/`
- `*.log`
- 本地 agent 规划/记录文件

但你仍然需要注意：

### 不要提交这些内容

- `~/.opentoken/` 整个目录
- 手工导出的 cookie 文件
- header / bearer token 文本文件
- `.env` / `.env.*`
- 临时抓包 / 调试日志（如果里面有 token）

### 提交前建议先看一眼

```bash
git status
```

如果看到任何凭证文件、浏览器导出文件、cookie 文本，请不要提交。

---

## License

MIT License，详见：

- `LICENSE`
