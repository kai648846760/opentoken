from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx


def _load_live_provider_suite_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "live_provider_200_suite.py"
    spec = importlib.util.spec_from_file_location("live_provider_200_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_live_runner_upload_attachment_uses_multipart_and_purpose(monkeypatch) -> None:
    module = _load_live_provider_suite_module()
    real_httpx_client = httpx.Client
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization", "")
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.read().decode("utf-8", errors="ignore")
        return httpx.Response(200, json={"id": "file_test_123"})

    transport = httpx.MockTransport(handler)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_httpx_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "Client", client_factory)

    runner = module.LiveProviderRunner(base_url="http://example.test", api_key="sk-test")
    try:
        file_id = runner.upload_attachment(provider="deepseek")
    finally:
        runner.close()

    assert file_id == "file_test_123"
    assert captured["authorization"] == "Bearer sk-test"
    assert captured["content_type"].startswith("multipart/form-data; boundary=")
    assert 'name="purpose"' in captured["body"]
    assert "\r\nassistants\r\n" in captured["body"]
    assert 'name="file"; filename="deepseek-attachment.txt"' in captured["body"]


def test_live_runner_retries_rate_limit_json_errors() -> None:
    module = _load_live_provider_suite_module()

    should_retry, detail = module._should_retry_http_result(
        429,
        {
            "error": {
                "message": "DeepSeek rate limit: 消息发送过于频繁，请稍后重试",
                "type": "rate_limit_error",
            }
        },
    )

    assert should_retry is True
    assert "消息发送过于频繁" in detail


def test_live_runner_retries_rate_limit_stream_errors() -> None:
    module = _load_live_provider_suite_module()

    should_retry, detail = module._should_retry_stream_result(
        200,
        [
            'data: {"error":{"message":"DeepSeek rate limit: 消息发送过于频繁，请稍后重试","type":"rate_limit_error"}}',
            "data: [DONE]",
        ],
    )

    assert should_retry is True
    assert "消息发送过于频繁" in detail


def test_live_runner_caps_retry_delay() -> None:
    module = _load_live_provider_suite_module()

    delay = module._compute_retry_delay(
        module.ProviderExecutionPolicy(
            min_interval_seconds=2.0,
            max_attempts=6,
            retry_base_delay_seconds=10.0,
        ),
        6,
    )

    assert delay == 20.0


def test_live_runner_prefers_flash_models_for_primary_selection() -> None:
    module = _load_live_provider_suite_module()

    model = module._pick_primary_model(
        [
            "algae/qwen-intl/qwen3.6-plus",
            "algae/qwen-intl/qwen3.5-flash",
            "algae/qwen-intl/qwen3.5-max-2026-03-08",
        ]
    )

    assert model == "algae/qwen-intl/qwen3.5-flash"


def test_live_runner_prefers_cn_flash_model_for_primary_selection() -> None:
    module = _load_live_provider_suite_module()

    model = module._pick_primary_model(
        [
            "algae/qwen-cn/Qwen3.5-千问",
            "algae/qwen-cn/Qwen3.5-Flash",
            "algae/qwen-cn/Qwen3-Max",
        ]
    )

    assert model == "algae/qwen-cn/Qwen3.5-Flash"


def test_live_runner_collects_reasoning_text_from_responses_payload() -> None:
    module = _load_live_provider_suite_module()

    text = module._responses_text_from_payload(
        {
            "output": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "先想一想"}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "最终答案"}],
                },
            ]
        }
    )

    assert text == "<think>先想一想</think>最终答案"


def test_live_runner_collects_reasoning_text_from_responses_stream() -> None:
    module = _load_live_provider_suite_module()

    text, delta_count, detail = module._parse_responses_stream(
        [
            "event: response.reasoning_text.delta",
            'data: {"type":"response.reasoning_text.delta","delta":"先想一想"}',
            "event: response.output_text.delta",
            'data: {"type":"response.output_text.delta","delta":"最终答案"}',
        ]
    )

    assert text == "<think>先想一想</think>最终答案"
    assert delta_count == 2
    assert detail == ""


def test_live_runner_rejects_snapshot_like_chat_stream_deltas() -> None:
    module = _load_live_provider_suite_module()

    text, delta_count, detail = module._parse_chat_completion_stream(
        [
            'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"这是今天"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"这是今天开源圈"},"finish_reason":null}]}',
            "data: [DONE]",
        ]
    )

    assert text == ""
    assert delta_count == 1
    assert "snapshot-like duplicated content delta" in detail


def test_live_runner_collects_tool_calls_from_chat_stream() -> None:
    module = _load_live_provider_suite_module()

    text, tool_calls, detail = module._parse_chat_completion_tool_stream(
        [
            'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_weather_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"location\\":\\"Tokyo\\"}"}}]},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
    )

    assert text == ""
    assert detail == ""
    assert tool_calls == [
        {
            "id": "call_weather_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location":"Tokyo"}',
            },
        }
    ]
