# OpenClaw Algae

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
- 本地工具 / OpenClaw / OpenAI-compatible 客户端接起来很麻烦

OpenClaw Algae 做的就是：

1. 统一管理 provider 凭证
2. 暴露一个本地 OpenAI-compatible 网关
3. 把不同 provider 的模型都映射到统一调用方式
4. 尽量把流式输出、工具调用、响应结构对齐到 OpenAI 风格

---

## 当前支持的 provider

- DeepSeek
- Qwen International
- Qwen China
- Kimi
- Claude
- Doubao
- ChatGPT
- Gemini
- Grok
- GLM China
- GLM International
- Xiaomi Mimo
- Manus

> 实际可用模型取决于你当前已经登录/配置成功的 provider。

---

## 环境要求

- Python `>= 3.13`
- 推荐使用 [`uv`](https://docs.astral.sh/uv/)

---

## 目录与本地状态

项目本身不会把登录凭证写进仓库，而是写到用户目录下：

- `~/.openclaw-algae/config.json`：本地网关配置（包含本地 API key / host / port）
- `~/.openclaw-algae/providers/*.json`：各 provider 凭证
- `~/.openclaw-algae/`：其它本地运行状态

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
uv run openclaw-algae onboard
```

### 3）启动服务

```bash
uv run openclaw-algae start
```

默认监听：

- `http://127.0.0.1:32117`

默认 OpenAI-compatible base URL：

- `http://127.0.0.1:32117/v1`

---

## 本地 API key 在哪看

OpenClaw Algae 使用的是**本地网关自己的 API key**，不是上游 provider 的 key。

默认保存在：

- `~/.openclaw-algae/config.json`

你可以直接查看：

```bash
cat ~/.openclaw-algae/config.json
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
path = os.path.expanduser('~/.openclaw-algae/config.json')
with open(path, 'r', encoding='utf-8') as f:
    print(json.load(f)['api_key'])
PY
```

---

## Provider 登录方式

登录命令统一是：

```bash
uv run openclaw-algae login <provider>
```

### 方式 A：浏览器登录（推荐）

适合支持浏览器采集登录态的 provider。

示例：

```bash
uv run openclaw-algae login qwen international --browser
uv run openclaw-algae login qwen china --browser
uv run openclaw-algae login deepseek --browser
uv run openclaw-algae login kimi --browser
uv run openclaw-algae login doubao --browser
uv run openclaw-algae login glm international --browser
uv run openclaw-algae login glm china --browser
```

说明：
- 会打开对应网页登录流程
- 登录成功后，凭证会保存到 `~/.openclaw-algae/providers/*.json`

### 方式 B：手工凭证登录

适合你已经有 cookie / header 的情况。

示例：

```bash
uv run openclaw-algae login qwen international \
  --cookie 'your_cookie_here' \
  --user-agent 'your user agent'
```

或者：

```bash
uv run openclaw-algae login deepseek \
  --header 'authorization=Bearer xxx'
```

也可以多个 header：

```bash
uv run openclaw-algae login some-provider \
  --header 'authorization=Bearer xxx' \
  --header 'x-token=yyy'
```

### 方式 C：API key 登录

适合 provider 本身支持 `--api-key` 的情况，例如 Manus。

```bash
uv run openclaw-algae login manus --api-key YOUR_KEY
```

### 查看当前 provider 状态

```bash
uv run openclaw-algae providers
```

### 登出某个 provider

```bash
uv run openclaw-algae logout qwen international
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
uv run openclaw-algae status
```

### 跑诊断

```bash
uv run openclaw-algae doctor
```

### 跑接口契约验证

```bash
uv run openclaw-algae verify
```

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

---

### 4）Responses API

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

### 5）流式 Responses API

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

## 给 OpenClaw 使用

如果你是要让 OpenClaw 直接接这个网关，可以先看将要写入的配置：

```bash
uv run openclaw-algae config --dry-run
```

确认后正式写入：

```bash
uv run openclaw-algae config
```

如果你要写到自定义 OpenClaw 配置文件：

```bash
uv run openclaw-algae config --openclaw-config /path/to/openclaw-config.json
```

---

## 推荐的日常使用流程

### 最短路径

```bash
uv sync
uv run openclaw-algae login qwen international --browser
uv run openclaw-algae start
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
uv run openclaw-algae providers
```

如果某个 provider 没有有效凭证，它对应模型通常不会出现在模型列表里。

### 2）服务能起来，但请求报 401

大概率是你传的 `Authorization Bearer` 不是本地网关 API key。

重新查看：

```bash
cat ~/.openclaw-algae/config.json
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
uv run openclaw-algae login qwen international --browser
```

### 5）想切换监听地址和端口

```bash
uv run openclaw-algae start --host 0.0.0.0 --port 32117
```

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

---

## 上传到 GitHub 前的安全注意事项

本项目已经在 `.gitignore` 里优先忽略：

- `.venv/`
- `.openclaw-algae/`
- `.openclaw/`
- `tmp/`
- `*.log`
- 本地 agent 规划/记录文件

但你仍然需要注意：

### 不要提交这些内容

- `~/.openclaw-algae/` 整个目录
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
