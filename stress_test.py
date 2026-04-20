#!/usr/bin/env python3
"""
Full load test for all logged-in providers.
Each provider: 300-500 conversations.
Results saved to stress_test_report.md
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import httpx

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:32117"
API_KEY = "d26bcd78b9f5b25c03b6158f896f4968"
TARGET_CALLS = int(os.getenv("STRESS_CALLS", "300"))  # override with STRESS_CALLS=50 for quick test

PROVIDERS = {
    "glm-cn": [
        "algae/glm-cn/glm-4-plus",
        "algae/glm-cn/glm-4",
    ],
    "deepseek": [
        "algae/deepseek/deepseek-chat",
        "algae/deepseek/deepseek-reasoner",
    ],
    "kimi": [
        "algae/kimi/moonshot-v1-32k",
    ],
    "qwen-cn": [
        "algae/qwen-cn/Qwen3.5-Plus",
        "algae/qwen-cn/Qwen3.5-Turbo",
    ],
}

# Varied prompts to avoid caching / rate limit triggers
PROMPTS = [
    "What is 1+1?",
    "Name a color.",
    "Say OK.",
    "What is 2*3?",
    "Give me a fruit name.",
    "What year is it?",
    "Name a planet.",
    "Say hello.",
    "What is 5-2?",
    "Name an animal.",
    "Is water wet? One word.",
    "Count to 3.",
    "Name a country.",
    "What is 10/2?",
    "Name a day of the week.",
    "Say goodbye.",
    "What color is the sky?",
    "Name a programming language.",
    "What is 3^2?",
    "Name a car brand.",
]


# ── Data ──────────────────────────────────────────────────────────────────────
@dataclass
class CallResult:
    success: bool
    latency_ms: float
    error: str = ""
    content_len: int = 0


@dataclass
class ModelStats:
    model: str
    results: list[CallResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def ok(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def fail(self) -> int:
        return self.total - self.ok

    @property
    def success_rate(self) -> float:
        return self.ok / self.total * 100 if self.total else 0

    @property
    def avg_latency(self) -> float:
        lats = [r.latency_ms for r in self.results if r.success]
        return statistics.mean(lats) if lats else 0

    @property
    def p95_latency(self) -> float:
        lats = sorted(r.latency_ms for r in self.results if r.success)
        if not lats:
            return 0
        idx = int(len(lats) * 0.95)
        return lats[min(idx, len(lats) - 1)]

    @property
    def p99_latency(self) -> float:
        lats = sorted(r.latency_ms for r in self.results if r.success)
        if not lats:
            return 0
        idx = int(len(lats) * 0.99)
        return lats[min(idx, len(lats) - 1)]

    @property
    def errors_summary(self) -> dict[str, int]:
        errs: dict[str, int] = {}
        for r in self.results:
            if not r.success:
                key = r.error[:100]
                errs[key] = errs.get(key, 0) + 1
        return errs


# ── HTTP ──────────────────────────────────────────────────────────────────────
def call_model(client: httpx.Client, model: str, prompt: str) -> CallResult:
    t0 = time.perf_counter()
    try:
        resp = client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120.0,
        )
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            return CallResult(success=False, latency_ms=latency, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return CallResult(success=False, latency_ms=latency, error=f"Empty content. Raw: {resp.text[:200]}")
        return CallResult(success=True, latency_ms=latency, content_len=len(content))
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        return CallResult(success=False, latency_ms=latency, error=str(exc)[:200])


# ── Test runner ───────────────────────────────────────────────────────────────
def test_model(model: str, n: int) -> ModelStats:
    stats = ModelStats(model=model)
    provider = model.split("/")[1] if "/" in model else model

    # Browser-based providers are slower, use adaptive timeout
    is_browser = provider in ("doubao", "qwen-cn", "qwen-intl", "glm-cn", "glm-intl", "chatgpt", "gemini", "grok")

    with httpx.Client(timeout=180.0) as client:
        for i in range(n):
            prompt = PROMPTS[i % len(PROMPTS)]
            result = call_model(client, model, prompt)
            stats.results.append(result)

            # Progress print
            status = "✅" if result.success else "❌"
            print(
                f"  [{i+1:>3}/{n}] {status} {latency_str(result.latency_ms)}ms"
                + (f" len={result.content_len}" if result.success else f" ERR: {result.error[:60]}")
            )

            # For browser providers: small delay to avoid overloading
            if is_browser and result.success:
                time.sleep(1.0)
            elif is_browser and not result.success:
                # Back off on error
                time.sleep(5.0)

    return stats


def latency_str(ms: float) -> str:
    return f"{ms:>7.0f}"


# ── Report ────────────────────────────────────────────────────────────────────
def render_report(all_stats: list[ModelStats], total_seconds: float) -> str:
    lines = []
    lines.append("# OpenClaw Algae — Full Provider Stress Test Report")
    lines.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Target calls per model:** {TARGET_CALLS}")
    lines.append(f"**Total wall time:** {total_seconds/60:.1f} min\n")

    lines.append("## Summary Table\n")
    lines.append("| Provider | Model | Total | ✅ OK | ❌ Fail | Success% | Avg ms | P95 ms | P99 ms |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for s in all_stats:
        parts = s.model.split("/")
        provider = parts[1] if len(parts) >= 3 else parts[0]
        model_name = parts[2] if len(parts) >= 3 else parts[-1]
        lines.append(
            f"| {provider} | {model_name} | {s.total} | {s.ok} | {s.fail} "
            f"| {s.success_rate:.1f}% | {s.avg_latency:.0f} | {s.p95_latency:.0f} | {s.p99_latency:.0f} |"
        )

    lines.append("\n## Error Details\n")
    for s in all_stats:
        if s.fail == 0:
            continue
        lines.append(f"### {s.model}")
        for msg, cnt in s.errors_summary.items():
            lines.append(f"- ({cnt}x) `{msg}`")
        lines.append("")

    lines.append("\n## Provider Aggregate\n")
    by_provider: dict[str, list[ModelStats]] = {}
    for s in all_stats:
        p = s.model.split("/")[1] if "/" in s.model else s.model
        by_provider.setdefault(p, []).append(s)

    for provider, stats_list in by_provider.items():
        total_ok = sum(s.ok for s in stats_list)
        total_calls = sum(s.total for s in stats_list)
        rate = total_ok / total_calls * 100 if total_calls else 0
        all_lats = [r.latency_ms for s in stats_list for r in s.results if r.success]
        avg = statistics.mean(all_lats) if all_lats else 0
        verdict = "🟢 PASS" if rate >= 95 else ("🟡 WARN" if rate >= 80 else "🔴 FAIL")
        lines.append(f"- **{provider}**: {total_ok}/{total_calls} ({rate:.1f}%) avg={avg:.0f}ms {verdict}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"🚀 Starting stress test — {TARGET_CALLS} calls per model")
    print(f"   Base URL: {BASE_URL}")
    print(f"   Providers: {list(PROVIDERS.keys())}")
    print()

    # Quick health check
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        assert r.json().get("status") == "ok", r.text
        print("✅ Service health: OK\n")
    except Exception as exc:
        print(f"❌ Service not reachable: {exc}")
        sys.exit(1)

    all_stats: list[ModelStats] = []
    t_global_start = time.perf_counter()

    for provider, models in PROVIDERS.items():
        print(f"\n{'='*60}")
        print(f"Provider: {provider.upper()} ({len(models)} model(s))")
        print(f"{'='*60}")

        for model in models:
            print(f"\n▶ Model: {model}")
            print(f"  Calls: {TARGET_CALLS}")
            t0 = time.perf_counter()
            stats = test_model(model, TARGET_CALLS)
            elapsed = time.perf_counter() - t0
            all_stats.append(stats)
            print(
                f"\n  → {stats.ok}/{stats.total} OK ({stats.success_rate:.1f}%) "
                f"avg={stats.avg_latency:.0f}ms p95={stats.p95_latency:.0f}ms "
                f"[{elapsed:.0f}s total]"
            )

    total_seconds = time.perf_counter() - t_global_start

    report = render_report(all_stats, total_seconds)
    report_path = "stress_test_report_final.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"✅ Test complete in {total_seconds/60:.1f} min")
    print(f"📄 Report saved to: {report_path}")
    print(f"{'='*60}\n")
    print(report)


if __name__ == "__main__":
    main()
