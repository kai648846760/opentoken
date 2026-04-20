#!/usr/bin/env python3
"""Kimi 200+ case E2E suite with real credentials and detailed report."""
from __future__ import annotations

import json
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import httpx

BASE_URL = "http://127.0.0.1:32117"
CLIENT = httpx.Client(trust_env=False)
API_KEY = json.loads((Path.home() / ".openclaw-algae" / "config.json").read_text())["api_key"]
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}
MODEL = "algae/kimi/moonshot-v1-32k"
SECONDARY_MODEL = "algae/kimi/moonshot-v1-32k"


@dataclass
class CaseResult:
    category: str
    name: str
    endpoint: str
    status_code: int
    success: bool
    latency_ms: float
    detail: str = ""
    response_preview: str = ""


results: list[CaseResult] = []


def restart_server() -> None:
    subprocess.run("lsof -i :32117 | grep LISTEN | awk '{print $2}' | xargs kill -9 2>/dev/null || true", shell=True, check=False)
    time.sleep(2)
    subprocess.Popen("uv run openclaw-algae start >/tmp/algae-kimi-case.log 2>&1", shell=True)
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            r = CLIENT.get(f"{BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("server not ready")


def run_post(name: str, endpoint: str, payload: dict, *, category: str, timeout: float = 60.0) -> None:
    t0 = time.perf_counter()
    try:
        resp = CLIENT.post(f"{BASE_URL}{endpoint}", headers=HEADERS, json=payload, timeout=timeout)
        latency = (time.perf_counter() - t0) * 1000
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        preview = ""
        success = False
        if isinstance(body, dict) and body.get("error"):
            preview = body["error"].get("message", "")[:120]
            if category == "error" and resp.status_code == 400:
                success = True
        elif isinstance(body, dict) and body.get("object") == "chat.completion":
            preview = str(body.get("choices", [{}])[0].get("message", {}).get("content", ""))[:120]
            success = resp.status_code == 200 and bool(preview)
        elif isinstance(body, dict) and body.get("object") == "response":
            output = body.get("output", [])
            if output and isinstance(output[0], dict):
                content_blocks = output[0].get("content", [])
                if content_blocks and isinstance(content_blocks[0], dict):
                    preview = str(content_blocks[0].get("text", ""))[:120]
            success = resp.status_code == 200 and bool(preview)
        else:
            preview = str(body)[:120]
        results.append(CaseResult(category, name, endpoint, resp.status_code, success, latency, response_preview=preview))
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        results.append(CaseResult(category, name, endpoint, 0, False, latency, detail=f"{type(exc).__name__}: {exc}"))
    time.sleep(3.0)


def run_stream(name: str, endpoint: str, payload: dict, *, category: str, timeout: float = 60.0) -> None:
    t0 = time.perf_counter()
    try:
        chunks = []
        with CLIENT.stream("POST", f"{BASE_URL}{endpoint}", headers=HEADERS, json=payload, timeout=timeout) as resp:
            for line in resp.iter_lines():
                line = line.strip()
                if line.startswith("data:"):
                    chunks.append(line)
            status = resp.status_code
        latency = (time.perf_counter() - t0) * 1000
        success = status == 200 and len(chunks) >= 2
        preview = chunks[0][:120] if chunks else ""
        results.append(CaseResult(category, name, endpoint, status, success, latency, detail=f"{len(chunks)} SSE lines", response_preview=preview))
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        results.append(CaseResult(category, name, endpoint, 0, False, latency, detail=f"{type(exc).__name__}: {exc}"))
    time.sleep(0.3)


def build_cases() -> None:
    prompts = [
        "1+1等于几？", "2+2等于几？", "3*7等于几？", "法国首都是什么？", "中国首都是什么？",
        "写一首四行诗", "解释量子计算", "列出五种编程语言", "把hello翻译成中文", "今天天气怎么样？",
        "什么是递归？", "什么是排序算法？", "太阳系有几颗行星？", "地球为什么是圆的？", "什么是人工智能？",
        "讲个笑话", "写一个Python函数", "写一个JS函数", "什么是HTTP？", "什么是TCP？",
        "什么是数据库索引？", "什么是哈希表？", "什么是二叉树？", "什么是BFS？", "什么是DFS？",
        "什么是动态规划？", "什么是机器学习？", "什么是大语言模型？", "什么是推理？", "什么是向量数据库？",
        "用一句话介绍北京", "用一句话介绍上海", "介绍杭州", "介绍深圳", "介绍成都",
        "2的10次方是多少？", "100以内的质数有哪些？", "把abc倒序", "给我一个正则表达式示例", "如何写单元测试？",
        "如何提升代码质量？", "什么是重构？", "什么是设计模式？", "什么是依赖注入？", "什么是消息队列？",
        "什么是缓存穿透？", "什么是幂等性？", "什么是限流？", "什么是熔断？", "什么是回滚？",
        "写一句鼓励的话", "写一句安慰的话", "写一句祝福的话", "写一句广告语", "写一句口号",
        "什么是云计算？", "什么是容器？", "什么是Kubernetes？", "什么是Linux？", "什么是MacOS？",
        "什么是HTTP状态码404？", "什么是500错误？", "什么是JWT？", "什么是OAuth？", "什么是SSO？",
        "解释微服务", "解释单体架构", "解释事件驱动", "解释函数式编程", "解释面向对象",
        "用一句话概括春天", "用一句话概括夏天", "用一句话概括秋天", "用一句话概括冬天", "写一句励志名言",
        "如何学习编程？", "如何学英语？", "如何写简历？", "如何准备面试？", "如何做时间管理？",
    ]
    for i, prompt in enumerate(prompts, 1):
        run_post(f"single-turn #{i}", "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}, category="single-turn")

    systems = [
        "你是一个数学老师。", "你是一个历史老师。", "你是一个翻译官。", "你是一个程序员。", "你是一个诗人。",
        "你要简洁回答。", "你只能回答中文。", "你只能回答英文。", "你只输出JSON。", "你只输出数字。",
    ]
    users = ["1+1等于几？", "第一次世界大战哪年开始？", "Hello world 翻译成中文", "写个Python hello world", "写一首诗"]
    idx = 0
    for s in systems:
        for u in users[:3]:
            idx += 1
            run_post(f"system #{idx}", "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "system", "content": s}, {"role": "user", "content": u}]}, category="system")

    conversations = [
        [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮你的？"}, {"role": "user", "content": "1+1等于几？"}],
        [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is a programming language."}, {"role": "user", "content": "Is it easy to learn?"}],
        [{"role": "user", "content": "推荐一本书"}, {"role": "assistant", "content": "《百年孤独》值得一读。"}, {"role": "user", "content": "作者是谁？"}],
    ]
    for i in range(30):
        run_post(f"multi-turn #{i+1}", "/v1/chat/completions", {"model": MODEL, "messages": conversations[i % len(conversations)]}, category="multi-turn")

    for i, prompt in enumerate(["说OK", "你好", "1+1=?", "给我一个词"] * 5, 1):
        run_stream(f"chat-stream #{i}", "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": True}, category="chat-stream")

    inputs = ["1+1等于几？", "说OK", "介绍一下你自己", "列出三种水果"] * 5
    for i, inp in enumerate(inputs, 1):
        run_post(f"responses #{i}", "/v1/responses", {"model": MODEL, "input": inp}, category="responses")

    for i, inp in enumerate(["说OK", "你好"] * 5, 1):
        run_stream(f"responses-stream #{i}", "/v1/responses", {"model": MODEL, "input": inp, "stream": True}, category="responses-stream")

    tool_messages = [
        [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "calc", "arguments": '{"a":2,"b":2}'}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "4"}],
        [{"role": "user", "content": "What's the weather in Tokyo?"}, {"role": "assistant", "content": None, "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Tokyo"}'}}]}, {"role": "tool", "tool_call_id": "call_2", "content": '{"temp":22,"unit":"C"}'}],
    ]
    for i in range(20):
        run_post(f"tool #{i+1}", "/v1/chat/completions", {"model": MODEL, "messages": tool_messages[i % len(tool_messages)]}, category="tool-calling")

    error_payloads = [
        {"model": "algae/nonexist/test", "messages": [{"role": "user", "content": "hello"}]},
        {"model": MODEL, "messages": []},
        {"messages": [{"role": "user", "content": "hello"}]},
        {"model": MODEL, "messages": [{"role": "invalid_role", "content": "hello"}]},
        {"model": MODEL, "messages": [{"role": "user", "content": ""}]},
    ]
    for i in range(20):
        run_post(f"error #{i+1}", "/v1/chat/completions", error_payloads[i % len(error_payloads)], category="error")

    for i in range(10):
        run_post(f"secondary-model #{i+1}", "/v1/chat/completions", {"model": SECONDARY_MODEL, "messages": [{"role": "user", "content": f"第{i+1}次测试：1+1等于几？"}]}, category="secondary-model")


def write_report() -> Path:
    report_path = Path("kimi_200_cases_report.md")
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    by_cat = Counter(r.category for r in results)
    by_cat_pass = Counter(r.category for r in results if r.success)
    lines = []
    lines.append("# Kimi 200+ Case E2E Report")
    lines.append("")
    lines.append(f"- Total cases: **{total}**")
    lines.append(f"- Passed: **{passed}**")
    lines.append(f"- Failed: **{failed}**")
    lines.append(f"- Success rate: **{passed/total*100:.1f}%**")
    lines.append("")
    lines.append("## Category Summary")
    lines.append("")
    lines.append("| Category | Total | Passed | Failed |")
    lines.append("|---|---:|---:|---:|")
    for cat, cnt in by_cat.items():
        p = by_cat_pass[cat]
        lines.append(f"| {cat} | {cnt} | {p} | {cnt-p} |")
    lines.append("")
    lines.append("## Failed Cases")
    lines.append("")
    failures = [r for r in results if not r.success]
    if not failures:
        lines.append("None")
    else:
        for r in failures:
            lines.append(f"- **{r.category} / {r.name}** — `{r.endpoint}` — HTTP {r.status_code} — {r.detail or r.response_preview}")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    r = CLIENT.get(f"{BASE_URL}/health", timeout=5)
    if r.status_code != 200:
        raise SystemExit("server not ready")
    build_cases()
    report = write_report()
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Report: {report}")
    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
