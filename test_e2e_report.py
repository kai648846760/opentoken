#!/usr/bin/env python3
"""
Comprehensive E2E Test Runner — generates detailed markdown report.
Tests ALL working providers against ALL endpoints with real credentials.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import httpx


# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL = "http://127.0.0.1:32117"
CONFIG_PATH = Path.home() / ".openclaw-algae" / "config.json"
API_KEY = json.loads(CONFIG_PATH.read_text())["api_key"]
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# Only test providers with valid credentials
PROVIDERS = {
    "deepseek": {
        "models": ["algae/deepseek/deepseek-chat", "algae/deepseek/deepseek-reasoner"],
    },
    "kimi": {
        "models": ["algae/kimi/moonshot-v1-32k"],
    },
    "qwen-cn": {
        "models": ["algae/qwen-cn/Qwen3.5-Plus"],
    },
    "glm-cn": {
        "models": ["algae/glm-cn/glm-4-plus"],
    },
}


@dataclass
class TestResult:
    provider: str
    test_name: str
    endpoint: str
    method: str
    request_payload: dict | None
    status_code: int
    response_body: Any
    success: bool
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class Report:
    tests: list[TestResult] = field(default_factory=list)
    started: str = ""
    finished: str = ""


report = Report()


def run_test(
    provider: str,
    name: str,
    endpoint: str,
    method: str,
    payload: dict | None,
    timeout: float = 30.0,
) -> TestResult:
    """Run a single API test and return the result."""
    t0 = time.perf_counter()
    try:
        if method == "GET":
            resp = httpx.get(
                f"{BASE_URL}{endpoint}",
                headers={"Authorization": HEADERS["Authorization"]},
                timeout=timeout,
            )
        else:
            resp = httpx.post(
                f"{BASE_URL}{endpoint}",
                headers=HEADERS,
                json=payload or {},
                timeout=timeout,
            )
        latency = (time.perf_counter() - t0) * 1000
        # Add delay between API calls to avoid rate limiting
        time.sleep(0.5)
        try:
            body = resp.json()
        except Exception:
            body = resp.text[:500]

        success = resp.status_code == 200 and (
            method == "GET" or
            (isinstance(body, dict) and "error" not in body)
        )

        detail = ""
        if isinstance(body, dict) and "error" in body:
            detail = body["error"].get("message", "")[:120]
        elif method == "GET" and resp.status_code == 401:
            detail = "Authentication required"

        result = TestResult(
            provider=provider,
            test_name=name,
            endpoint=endpoint,
            method=method,
            request_payload=payload,
            status_code=resp.status_code,
            response_body=body,
            success=success,
            detail=detail,
            latency_ms=latency,
        )
        report.tests.append(result)
        return result

    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        result = TestResult(
            provider=provider,
            test_name=name,
            endpoint=endpoint,
            method=method,
            request_payload=payload,
            status_code=0,
            response_body=None,
            success=False,
            detail=f"{type(exc).__name__}: {exc}",
            latency_ms=latency,
        )
        report.tests.append(result)
        return result


def run_streaming_test(
    provider: str,
    name: str,
    model: str,
    payload: dict,
    timeout: float = 60.0,
    endpoint: str = "/v1/chat/completions",
) -> TestResult:
    """Run a streaming test with SSE parsing."""
    t0 = time.perf_counter()
    try:
        chunks = []
        event_count = 0
        with httpx.stream(
            "POST",
            f"{BASE_URL}{endpoint}",
            headers=HEADERS,
            json=payload,
            timeout=timeout,
        ) as resp:
            status_code = resp.status_code
            for line in resp.iter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    event_count += 1
                    chunks.append(line[6:])
                elif line.startswith("data:"):
                    event_count += 1
                    chunks.append(line[5:])
            latency = (time.perf_counter() - t0) * 1000

        success = status_code == 200 and event_count >= 2
        # Check that chunks have valid JSON
        has_valid_chunk = False
        for c in chunks:
            if c == "[DONE]":
                continue
            try:
                d = json.loads(c)
                if isinstance(d, dict):
                    has_valid_chunk = True
                    break
            except Exception:
                pass

        if status_code == 200:
            if not has_valid_chunk:
                success = False

        # Add delay after streaming test
        time.sleep(1.0)

        result = TestResult(
            provider=provider,
            test_name=name,
            endpoint=endpoint,
            method="POST",
            request_payload=payload,
            status_code=status_code,
            response_body={"events": event_count, "chunks": len(chunks)},
            success=success,
            detail=f"{event_count} SSE events" if success else f"HTTP {status_code}, {event_count} events",
            latency_ms=latency,
        )
        report.tests.append(result)
        return result

    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        result = TestResult(
            provider=provider,
            test_name=name,
            endpoint=endpoint,
            method="POST",
            request_payload=payload,
            status_code=0,
            response_body=None,
            success=False,
            detail=f"{type(exc).__name__}: {exc}",
            latency_ms=latency,
        )
        report.tests.append(result)
        return result


# ── Test Suite ────────────────────────────────────────────────────────────────

def run_all_tests() -> None:
    report.started = time.strftime("%Y-%m-%d %H:%M:%S")

    # 0. Health & Models (provider-agnostic)
    print("Testing health...")
    run_test("system", "Health Check", "/health", "GET", None, timeout=5)

    print("Testing models...")
    run_test("system", "Models List", "/v1/models", "GET", None, timeout=5)

    print("Testing auth rejection...")
    try:
        resp = httpx.get(f"{BASE_URL}/v1/models", timeout=5)
        report.tests.append(TestResult(
            provider="system",
            test_name="Auth Rejection (no API key)",
            endpoint="/v1/models",
            method="GET",
            request_payload=None,
            status_code=resp.status_code,
            response_body=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text[:200],
            success=resp.status_code == 401,
            detail="Expected 401" if resp.status_code == 401 else f"Expected 401, got {resp.status_code}",
        ))
    except Exception as exc:
        report.tests.append(TestResult(
            provider="system",
            test_name="Auth Rejection (no API key)",
            endpoint="/v1/models",
            method="GET",
            request_payload=None,
            status_code=0,
            response_body=None,
            success=False,
            detail=str(exc)[:120],
        ))

    # Per-provider tests
    for provider_key, provider_info in PROVIDERS.items():
        models = provider_info["models"]
        default_model = models[0]
        print(f"\nTesting {provider_key} ({default_model})...")

        # ── Basic Chat (10 prompts) ────────────────────────────────
        chat_prompts = [
            "请用一句话回答：1+1等于几？",
            "What is the capital of France?",
            "请写一首关于春天的四行诗。",
            "用三句话解释量子计算。",
            "List 5 programming languages.",
            "请翻译：Hello, how are you? 成中文。",
            "What color is the sky on a clear day?",
            "请列出太阳系八大行星。",
            "Summarize the benefits of exercise in one sentence.",
            "用一句话描述人工智能的未来。",
        ]
        for i, prompt in enumerate(chat_prompts):
            run_test(
                provider_key,
                f"Chat: prompt #{i+1} ({prompt[:30]}...)",
                "/v1/chat/completions",
                "POST",
                {"model": default_model, "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )

        # ── System Prompts (5 roles) ──────────────────────────────
        system_roles = [
            ("你是一个数学老师", "1+1等于几？"),
            ("You are a Python expert. Answer concisely.", "What is a list comprehension?"),
            ("你是一个翻译官，只输出翻译结果。", "Hello, world!"),
            ("你是一个幽默的助手。", "讲个笑话。"),
            ("你是一个历史学家。", "第一次世界大战是哪一年开始的？"),
        ]
        for i, (system, user) in enumerate(system_roles):
            run_test(
                provider_key,
                f"System prompt #{i+1}: {system[:25]}...",
                "/v1/chat/completions",
                "POST",
                {
                    "model": default_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=60,
            )

        # ── Multi-turn Conversations (5 scenarios) ───────────────
        multi_turn_scenarios = [
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮你的？"},
                {"role": "user", "content": "1+1等于几？"},
            ],
            [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a high-level programming language."},
                {"role": "user", "content": "Is it easy to learn?"},
            ],
            [
                {"role": "system", "content": "你是一个程序员。"},
                {"role": "user", "content": "写一个快速排序"},
                {"role": "assistant", "content": "def quicksort(arr): ..."},
                {"role": "user", "content": "时间复杂度是多少？"},
            ],
            [
                {"role": "user", "content": "今天天气怎么样？"},
                {"role": "assistant", "content": "我无法获取实时天气信息。"},
                {"role": "user", "content": "那你能做什么？"},
            ],
            [
                {"role": "user", "content": "推荐一本书"},
                {"role": "assistant", "content": "《百年孤独》是加西亚·马尔克斯的经典之作。"},
                {"role": "user", "content": "作者是谁？"},
            ],
        ]
        for i, messages in enumerate(multi_turn_scenarios):
            run_test(
                provider_key,
                f"Multi-turn #{i+1} ({len(messages)} turns)",
                "/v1/chat/completions",
                "POST",
                {"model": default_model, "messages": messages},
                timeout=60,
            )

        # ── Content Types (5 formats) ───────────────────────────
        content_formats = [
            "请用markdown格式列出Python的优点。",
            "用JSON格式回答：{'question': '1+1=?', 'answer': 2}",
            "请用代码块展示一个Python的hello world程序。",
            "请用表格格式对比Python和JavaScript。",
            "What is 2 * 3 * 5 * 7? Show your work step by step.",
        ]
        for i, prompt in enumerate(content_formats):
            run_test(
                provider_key,
                f"Content format #{i+1}: {prompt[:30]}...",
                "/v1/chat/completions",
                "POST",
                {"model": default_model, "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )

        # ── Streaming Tests (4 variants) ────────────────────────
        streaming_prompts = ["说OK", "Hello", "你好", "1+1=?"]
        for i, prompt in enumerate(streaming_prompts):
            run_streaming_test(
                provider_key,
                f"Streaming #{i+1}: {prompt}",
                default_model,
                {"model": default_model, "messages": [{"role": "user", "content": prompt}], "stream": True},
                timeout=60,
            )

        # ── Responses API (5 tests) ─────────────────────────────
        run_test(
            provider_key, "Responses: basic", "/v1/responses", "POST",
            {"model": default_model, "input": "请用一句话回答：1+1等于几？"}, timeout=60,
        )
        run_test(
            provider_key, "Responses: with instructions", "/v1/responses", "POST",
            {"model": default_model, "instructions": "只回答数字。", "input": "1+1等于几？"}, timeout=60,
        )
        run_test(
            provider_key, "Responses: multi-message input", "/v1/responses", "POST",
            {"model": default_model, "input": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}, {"role": "user", "content": "1+1=?"}]},
            timeout=60,
        )
        run_streaming_test(
            provider_key, "Responses: streaming", default_model,
            {"model": default_model, "input": "说OK", "stream": True},
            timeout=60, endpoint="/v1/responses",
        )
        run_streaming_test(
            provider_key, "Responses: streaming with instructions", default_model,
            {"model": default_model, "instructions": "只回答数字。", "input": "2*2=?", "stream": True},
            timeout=60, endpoint="/v1/responses",
        )

        # ── Tool Calling (4 scenarios) ──────────────────────────
        tool_tests = [
            (
                "Tool: calculator result",
                [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "4"},
                ],
            ),
            (
                "Tool: weather function",
                [
                    {"role": "user", "content": "What's the weather in Tokyo?"},
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Tokyo"}'}}]},
                    {"role": "tool", "tool_call_id": "call_2", "content": '{"temp": 22, "unit": "C"}'},
                ],
            ),
            (
                "Tool: multiple tools",
                [
                    {"role": "user", "content": "What is 3+4 and 5*6?"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a":3,"b":4}'}},
                        {"id": "c2", "type": "function", "function": {"name": "mul", "arguments": '{"a":5,"b":6}'}},
                    ]},
                    {"role": "tool", "tool_call_id": "c1", "content": "7"},
                    {"role": "tool", "tool_call_id": "c2", "content": "30"},
                ],
            ),
            (
                "Tool: tool with error",
                [
                    {"role": "user", "content": "What is 1/0?"},
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_3", "type": "function", "function": {"name": "divide", "arguments": '{"a":1,"b":0}'}}]},
                    {"role": "tool", "tool_call_id": "call_3", "content": "Error: division by zero"},
                ],
            ),
        ]
        for i, (name, messages) in enumerate(tool_tests):
            run_test(
                provider_key, f"Tool calling: {name}",
                "/v1/chat/completions", "POST",
                {"model": default_model, "messages": messages},
                timeout=60,
            )

        # ── Error Handling (8 variants) ─────────────────────────
        error_tests = [
            ("invalid model", {"model": "algae/nonexist/test", "messages": [{"role": "user", "content": "hello"}]}),
            ("missing messages", {"model": default_model, "messages": []}),
            ("missing model", {"messages": [{"role": "user", "content": "hello"}]}),
            ("invalid role", {"model": default_model, "messages": [{"role": "invalid_role", "content": "hello"}]}),
            ("null content", {"model": default_model, "messages": [{"role": "user", "content": None}]}),
            ("empty content", {"model": default_model, "messages": [{"role": "user", "content": ""}]}),
            ("very long prompt", {"model": default_model, "messages": [{"role": "user", "content": "A" * 10000}]}),
            ("unicode prompt", {"model": default_model, "messages": [{"role": "user", "content": "🎉✨🔥 你好世界！"}]}),
        ]
        for i, (name, payload) in enumerate(error_tests):
            result = run_test(
                provider_key, f"Error handling: {name}",
                "/v1/chat/completions", "POST",
                payload,
            )
            # 400 with proper error envelope is correct behavior for errors
            if result.status_code == 400 and isinstance(result.response_body, dict) and "error" in result.response_body:
                result.success = True
                result.detail = f"Correctly returned 400: {result.detail[:80]}"

        # ── Secondary Models ────────────────────────────────────
        if len(models) > 1:
            for secondary in models[1:]:
                run_test(
                    provider_key,
                    f"Secondary model: {secondary.split('/')[-1]}",
                    "/v1/chat/completions",
                    "POST",
                    {"model": secondary, "messages": [{"role": "user", "content": "请用一句话回答：1+1等于几？"}]},
                    timeout=60,
                )
                # Streaming for secondary model
                run_streaming_test(
                    provider_key,
                    f"Secondary model streaming: {secondary.split('/')[-1]}",
                    secondary,
                    {"model": secondary, "messages": [{"role": "user", "content": "说OK"}], "stream": True},
                    timeout=60,
                )

    report.finished = time.strftime("%Y-%m-%d %H:%M:%S")


# ── Report Generation ─────────────────────────────────────────────────────────

def generate_report() -> str:
    lines = []
    lines.append(f"# OpenClaw Algae — E2E Test Report")
    lines.append(f"\n**Date:** {report.started} → {report.finished}")
    lines.append(f"**Base URL:** {BASE_URL}")
    lines.append(f"**Total Tests:** {len(report.tests)}")

    passed = sum(1 for t in report.tests if t.success)
    failed = sum(1 for t in report.tests if not t.success)
    lines.append(f"**Passed:** {passed} | **Failed:** {failed}")
    lines.append(f"**Success Rate:** {passed/len(report.tests)*100:.1f}%")

    # ── Summary by Provider ──
    lines.append(f"\n## Summary by Provider\n")
    lines.append("| Provider | Tests | Passed | Failed | Success Rate |")
    lines.append("|----------|-------|--------|--------|-------------|")

    providers = {}
    for t in report.tests:
        if t.provider not in providers:
            providers[t.provider] = {"total": 0, "passed": 0, "failed": 0}
        providers[t.provider]["total"] += 1
        if t.success:
            providers[t.provider]["passed"] += 1
        else:
            providers[t.provider]["failed"] += 1

    for provider, stats in providers.items():
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] else 0
        emoji = "✅" if rate == 100 else ("⚠️" if rate >= 80 else "❌")
        lines.append(
            f"| {emoji} {provider} | {stats['total']} | {stats['passed']} | "
            f"{stats['failed']} | {rate:.0f}% |"
        )

    # ── Summary by Endpoint ──
    lines.append(f"\n## Summary by Endpoint\n")
    lines.append("| Endpoint | Method | Tests | Passed | Failed |")
    lines.append("|----------|--------|-------|--------|--------|")

    endpoints = {}
    for t in report.tests:
        key = (t.endpoint, t.method)
        if key not in endpoints:
            endpoints[key] = {"total": 0, "passed": 0, "failed": 0}
        endpoints[key]["total"] += 1
        if t.success:
            endpoints[key]["passed"] += 1
        else:
            endpoints[key]["failed"] += 1

    for (endpoint, method), stats in sorted(endpoints.items()):
        emoji = "✅" if stats["failed"] == 0 else "❌"
        lines.append(
            f"| {emoji} `{endpoint}` | {method} | {stats['total']} | "
            f"{stats['passed']} | {stats['failed']} |"
        )

    # ── Detailed Results ──
    lines.append(f"\n## Detailed Results\n")

    current_provider = ""
    for t in report.tests:
        if t.provider != current_provider:
            current_provider = t.provider
            lines.append(f"\n### {current_provider}\n")
            lines.append("")

        status = "✅ PASS" if t.success else "❌ FAIL"
        lines.append(f"#### {status} — {t.test_name}")
        lines.append("")
        lines.append(f"- **Endpoint:** `{t.method} {t.endpoint}`")
        lines.append(f"- **Status Code:** {t.status_code}")
        lines.append(f"- **Latency:** {t.latency_ms:.0f}ms")

        if t.request_payload:
            lines.append(f"- **Request:**")
            lines.append(f"  ```json")
            lines.append(f"  {json.dumps(t.request_payload, indent=2, ensure_ascii=False)}")
            lines.append(f"  ```")

        if t.detail:
            lines.append(f"- **Detail:** `{t.detail}`")

        # Response body
        if isinstance(t.response_body, dict):
            if "error" in t.response_body:
                lines.append(f"- **Response:** `{{\"error\": {{\"message\": \"{t.response_body['error'].get('message', '')[:100]}\", \"type\": \"{t.response_body['error'].get('type', '')}\"}}}}`")
            elif t.response_body.get("object") == "list":
                data = t.response_body.get("data", [])
                lines.append(f"- **Response:** `{len(data)} items, object=list`")
            elif t.response_body.get("object") == "chat.completion":
                choices = t.response_body.get("choices", [])
                content = ""
                if choices:
                    content = str(choices[0].get("message", {}).get("content", ""))[:100]
                lines.append(f"- **Response:** `object=chat.completion, content=\"{content}\"`")
            elif t.response_body.get("object") == "response":
                output = t.response_body.get("output", [])
                content = ""
                if output and isinstance(output, list) and isinstance(output[0], dict):
                    content_list = output[0].get("content", [])
                    if isinstance(content_list, list) and content_list:
                        content = str(content_list[0].get("text", ""))[:100]
                lines.append(f"- **Response:** `object=response, content=\"{content}\"`")
            else:
                lines.append(f"- **Response:** `{json.dumps(t.response_body, ensure_ascii=False)[:150]}`")
        elif t.response_body:
            lines.append(f"- **Response:** `{str(t.response_body)[:150]}`")

        if isinstance(t.response_body, dict) and "events" in t.response_body:
            lines.append(f"- **Streaming:** `{t.response_body['events']} SSE events`")

        lines.append("")

    # ── Failed Tests ──
    failures = [t for t in report.tests if not t.success]
    if failures:
        lines.append(f"\n## Failed Tests ({len(failures)})\n")
        for t in failures:
            lines.append(f"- **{t.provider}** — {t.test_name}")
            lines.append(f"  - Endpoint: `{t.method} {t.endpoint}`")
            lines.append(f"  - Error: `{t.detail}`")
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    # Quick server health check
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        if r.json().get("status") != "ok":
            print(f"Server not healthy: {r.text}")
            sys.exit(1)
    except Exception as exc:
        print(f"Cannot reach server at {BASE_URL}: {exc}")
        print("Start it first: uv run openclaw-algae start")
        sys.exit(1)

    print("Running E2E test suite...")
    run_all_tests()

    md = generate_report()
    report_path = Path(__file__).parent / "e2e_test_report.md"
    report_path.write_text(md, encoding="utf-8")

    # Print summary
    passed = sum(1 for t in report.tests if t.success)
    failed = sum(1 for t in report.tests if not t.success)
    total = len(report.tests)
    print(f"\n{'='*50}")
    print(f"  E2E Test Suite Complete")
    print(f"  Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"  Report: {report_path}")
    print(f"{'='*50}")

    # Also print the report
    print("\n" + md)

    sys.exit(0 if failed == 0 else 1)
