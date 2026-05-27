from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from opentoken.api.app import create_app
from opentoken.browser.common import CamoufoxRuntimeStatus, probe_camoufox_runtime
from opentoken.config.app_config import load_or_create_app_config
from opentoken.config.paths import resolve_app_config_path, resolve_providers_dir
from opentoken.models.discovery import load_model_catalog
from opentoken.providers.registry import get_provider_definition, list_supported_providers
from opentoken.storage.provider_store import list_provider_credentials


# Concurrent per-provider verification. The browser worker thread serialises
# within a provider, so we just bound across providers to keep CPU/memory
# manageable.
_VERIFY_MAX_WORKERS = 8


_CAMOUFOX_RUNTIME_PROVIDERS = frozenset(
    {
        "doubao",
        "qwen-intl",
        "qwen-cn",
        "chatgpt",
        "gemini",
        "grok",
        "glm-cn",
        "glm-intl",
    }
)


@dataclass(frozen=True)
class EndpointVerificationResult:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class ProviderVerificationResult:
    provider: str
    display_name: str
    model: str | None
    status: str
    checks: tuple[EndpointVerificationResult, ...]


@dataclass(frozen=True)
class VerificationReport:
    requested_providers: tuple[str, ...]
    results: tuple[ProviderVerificationResult, ...]


def run_verification_suite(*, requested_providers: tuple[str, ...] = ()) -> VerificationReport:
    targets = requested_providers or tuple(provider.key for provider in list_supported_providers())
    provider_records = {
        record.provider: record for record in list_provider_credentials(resolve_providers_dir())
    }
    default_models = _default_provider_models()
    app_config = load_or_create_app_config(resolve_app_config_path())
    headers = {"authorization": f"Bearer {app_config['api_key']}"}
    camoufox_runtime = probe_camoufox_runtime()

    with TestClient(create_app()) as client:
        def verify(provider: str) -> ProviderVerificationResult:
            return _verify_one_provider(
                client=client,
                headers=headers,
                provider=provider,
                model=default_models.get(provider),
                provider_records=provider_records,
                camoufox_runtime=camoufox_runtime,
            )

        # Verify providers concurrently. Each provider's live checks are slow
        # (browser launch, multi-second upstream calls), and they're fully
        # independent — running them serially made `opentoken verify` take
        # minutes and risk appearing to hang. The browser worker threads still
        # serialise calls *within* a provider, so this only parallelises across
        # distinct providers. Results are reassembled in the requested order.
        results_by_provider: dict[str, ProviderVerificationResult] = {}
        max_workers = min(_VERIFY_MAX_WORKERS, max(len(targets), 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(verify, provider): provider for provider in targets}
            for future in concurrent.futures.as_completed(futures):
                provider = futures[future]
                results_by_provider[provider] = future.result()

    results = [results_by_provider[provider] for provider in targets]
    return VerificationReport(
        requested_providers=requested_providers,
        results=tuple(results),
    )


def _verify_one_provider(
    *,
    client: TestClient,
    headers: dict[str, str],
    provider: str,
    model: str | None,
    provider_records: dict[str, object],
    camoufox_runtime: CamoufoxRuntimeStatus,
) -> ProviderVerificationResult:
    definition = get_provider_definition(provider)
    display_name = definition.display_name if definition is not None else provider

    if provider not in provider_records:
        return ProviderVerificationResult(
            provider=provider,
            display_name=display_name,
            model=model,
            status="not_logged_in",
            checks=(),
        )

    if provider in _CAMOUFOX_RUNTIME_PROVIDERS and not camoufox_runtime.browser_installed:
        checks = (
            _verify_models(client, headers, provider, model),
            _failed_detail("chat", _camoufox_runtime_missing_detail(camoufox_runtime.install_hint)),
            _failed_detail("chat_stream", _camoufox_runtime_missing_detail(camoufox_runtime.install_hint)),
            _failed_detail("responses", _camoufox_runtime_missing_detail(camoufox_runtime.install_hint)),
            _failed_detail("responses_stream", _camoufox_runtime_missing_detail(camoufox_runtime.install_hint)),
        )
        return ProviderVerificationResult(
            provider=provider,
            display_name=display_name,
            model=model,
            status="failed",
            checks=checks,
        )

    checks = (
        _verify_models(client, headers, provider, model),
        _verify_chat_completion(client, headers, model),
        _verify_chat_completion_stream(client, headers, model),
        _verify_responses(client, headers, model),
        _verify_responses_stream(client, headers, model),
    )
    status = "passed" if all(check.status == "passed" for check in checks) else "failed"
    return ProviderVerificationResult(
        provider=provider,
        display_name=display_name,
        model=model,
        status=status,
        checks=checks,
    )


def verification_exit_code(report: VerificationReport) -> int:
    if any(result.status == "failed" for result in report.results):
        return 1
    if report.requested_providers and any(result.status != "passed" for result in report.results):
        return 1
    if not report.requested_providers and not any(result.status == "passed" for result in report.results):
        return 1
    return 0


def render_verification_report(report: VerificationReport) -> str:
    lines = ["provider\tstatus\tmodel\tchecks"]
    for result in report.results:
        if result.checks:
            checks_text = ", ".join(
                f"{check.name}={check.status}({check.detail})" for check in result.checks
            )
        else:
            checks_text = "-"
        lines.append(
            "\t".join(
                [
                    result.provider,
                    result.status,
                    result.model or "-",
                    checks_text,
                ]
            )
        )
    return "\n".join(lines)


def _default_provider_models() -> dict[str, str]:
    models: dict[str, str] = {}
    for entry in load_model_catalog():
        provider = entry.id.split("/", 2)[1]
        models.setdefault(provider, entry.id)
    return models


def _verify_models(
    client: TestClient,
    headers: dict[str, str],
    provider: str,
    model: str | None,
) -> EndpointVerificationResult:
    if model is None:
        return EndpointVerificationResult(
            name="models",
            status="failed",
            detail=f"no default model registered for {provider}",
        )

    try:
        response = client.get("/v1/models", headers=headers)
        if response.status_code != 200:
            return _failed_check("models", response)
        payload = response.json()
        if set(payload.keys()) != {"object", "data"}:
            return _failed_detail("models", f"unexpected keys: {sorted(payload.keys())}")
        if payload["object"] != "list":
            return _failed_detail("models", f"unexpected object: {payload['object']}")
        items = payload["data"]
        if not isinstance(items, list):
            return _failed_detail("models", "data is not a list")
        matches = [item for item in items if isinstance(item, dict) and item.get("id") == model]
        if len(matches) != 1:
            return _failed_detail("models", f"model {model} missing from catalog")
        item = matches[0]
        if set(item.keys()) != {"id", "object", "owned_by"}:
            return _failed_detail("models", f"unexpected model keys: {sorted(item.keys())}")
        if item["object"] != "model":
            return _failed_detail("models", f"unexpected model object: {item['object']}")
        return EndpointVerificationResult(
            name="models",
            status="passed",
            detail="model present in catalog",
        )
    except Exception as exc:
        return _failed_detail("models", str(exc))


def _verify_chat_completion(
    client: TestClient,
    headers: dict[str, str],
    model: str | None,
) -> EndpointVerificationResult:
    if model is None:
        return _failed_detail("chat", "missing model")

    try:
        response = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Return a short verification reply.",
                    }
                ],
            },
        )
        if response.status_code != 200:
            return _failed_check("chat", response)
        payload = response.json()
        if set(payload.keys()) != {"id", "object", "created", "model", "choices", "usage"}:
            return _failed_detail("chat", f"unexpected keys: {sorted(payload.keys())}")
        if not str(payload["id"]).startswith("chatcmpl-"):
            return _failed_detail("chat", f"unexpected id: {payload['id']}")
        if payload["object"] != "chat.completion":
            return _failed_detail("chat", f"unexpected object: {payload['object']}")
        if not isinstance(payload["created"], int) or payload["created"] <= 0:
            return _failed_detail("chat", f"unexpected created: {payload['created']}")
        if payload["model"] != model:
            return _failed_detail("chat", f"unexpected model: {payload['model']}")
        choices = payload["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            return _failed_detail("chat", "choices is not a single-item list")
        choice = choices[0]
        if set(choice.keys()) != {"index", "message", "finish_reason"}:
            return _failed_detail("chat", f"unexpected choice keys: {sorted(choice.keys())}")
        if choice["index"] != 0:
            return _failed_detail("chat", f"unexpected choice index: {choice['index']}")
        if choice["finish_reason"] != "stop":
            return _failed_detail("chat", f"unexpected finish_reason: {choice['finish_reason']}")
        message = choice["message"]
        if set(message.keys()) != {"role", "content"}:
            return _failed_detail("chat", f"unexpected message keys: {sorted(message.keys())}")
        if message["role"] != "assistant":
            return _failed_detail("chat", f"unexpected message role: {message['role']}")
        if not str(message["content"]).strip():
            return _failed_detail("chat", "assistant content is empty")
        usage = payload["usage"]
        if set(usage.keys()) != {"prompt_tokens", "completion_tokens", "total_tokens"}:
            return _failed_detail("chat", f"unexpected usage keys: {sorted(usage.keys())}")
        if any(not isinstance(usage[key], int) for key in usage):
            return _failed_detail("chat", "usage values must be integers")
        return EndpointVerificationResult(
            name="chat",
            status="passed",
            detail="assistant message is non-empty",
        )
    except Exception as exc:
        return _failed_detail("chat", str(exc))


def _verify_chat_completion_stream(
    client: TestClient,
    headers: dict[str, str],
    model: str | None,
) -> EndpointVerificationResult:
    if model is None:
        return _failed_detail("chat_stream", "missing model")

    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": "Return a short streamed verification reply.",
                    }
                ],
            },
        ) as response:
            if response.status_code != 200:
                return _failed_check("chat_stream", response)
            lines = [line for line in response.iter_lines() if line]

        if lines[-1] != "data: [DONE]":
            return _failed_detail("chat_stream", f"unexpected terminal line: {lines[-1]}")
        chunks = [_parse_data_line(line, "chat_stream") for line in lines[:-1]]
        if len(chunks) < 3:
            return _failed_detail("chat_stream", f"unexpected chunk count: {len(chunks)}")
        first_chunk = chunks[0]
        final_chunk = chunks[-1]
        first_choices = first_chunk["choices"]
        final_choices = final_chunk["choices"]
        if first_chunk["object"] != "chat.completion.chunk":
            return _failed_detail("chat_stream", f"unexpected object: {first_chunk['object']}")
        if first_chunk["model"] != model or final_chunk["model"] != model:
            return _failed_detail("chat_stream", "streamed model mismatch")
        if first_chunk["id"] != final_chunk["id"]:
            return _failed_detail("chat_stream", "chunk ids do not match")
        if first_choices[0]["delta"].get("role") != "assistant":
            return _failed_detail("chat_stream", "first chunk role is not assistant")
        if first_choices[0]["finish_reason"] is not None:
            return _failed_detail("chat_stream", "first chunk finish_reason must be null")
        streamed_text = "".join(
            str(chunk["choices"][0]["delta"].get("content", ""))
            for chunk in chunks[1:-1]
        )
        if not streamed_text.strip():
            return _failed_detail("chat_stream", "streamed content is empty")
        if final_choices[0]["delta"] != {}:
            return _failed_detail("chat_stream", "final chunk delta must be empty")
        if final_choices[0]["finish_reason"] != "stop":
            return _failed_detail("chat_stream", "final chunk finish_reason must be stop")
        return EndpointVerificationResult(
            name="chat_stream",
            status="passed",
            detail="sse chunks match chat.completion.chunk contract",
        )
    except Exception as exc:
        return _failed_detail("chat_stream", str(exc))


def _verify_responses(
    client: TestClient,
    headers: dict[str, str],
    model: str | None,
) -> EndpointVerificationResult:
    if model is None:
        return _failed_detail("responses", "missing model")

    try:
        response = client.post(
            "/v1/responses",
            headers=headers,
            json={
                "model": model,
                "input": "Return a short verification reply.",
            },
        )
        if response.status_code != 200:
            return _failed_check("responses", response)
        payload = response.json()
        if set(payload.keys()) != {"id", "object", "created_at", "status", "model", "output", "usage"}:
            return _failed_detail("responses", f"unexpected keys: {sorted(payload.keys())}")
        if not str(payload["id"]).startswith("resp-"):
            return _failed_detail("responses", f"unexpected id: {payload['id']}")
        if payload["object"] != "response":
            return _failed_detail("responses", f"unexpected object: {payload['object']}")
        if payload["status"] != "completed":
            return _failed_detail("responses", f"unexpected status: {payload['status']}")
        if payload["model"] != model:
            return _failed_detail("responses", f"unexpected model: {payload['model']}")
        if not isinstance(payload["created_at"], int) or payload["created_at"] <= 0:
            return _failed_detail("responses", f"unexpected created_at: {payload['created_at']}")
        output = payload["output"]
        if not isinstance(output, list) or len(output) != 1:
            return _failed_detail("responses", "output is not a single-item list")
        item = output[0]
        if set(item.keys()) != {"type", "id", "role", "status", "content"}:
            return _failed_detail("responses", f"unexpected output keys: {sorted(item.keys())}")
        if item["type"] != "message" or item["role"] != "assistant" or item["status"] != "completed":
            return _failed_detail("responses", "unexpected output item identity")
        if not str(item["id"]).startswith("msg-"):
            return _failed_detail("responses", f"unexpected output item id: {item['id']}")
        content = item["content"]
        if not isinstance(content, list) or len(content) != 1:
            return _failed_detail("responses", "content is not a single-item list")
        part = content[0]
        if set(part.keys()) != {"type", "text"}:
            return _failed_detail("responses", f"unexpected content keys: {sorted(part.keys())}")
        if part["type"] != "output_text":
            return _failed_detail("responses", f"unexpected content type: {part['type']}")
        if not str(part["text"]).strip():
            return _failed_detail("responses", "output text is empty")
        usage = payload["usage"]
        expected_usage_keys = {
            "input_tokens",
            "input_tokens_details",
            "output_tokens",
            "output_tokens_details",
            "total_tokens",
        }
        if set(usage.keys()) != expected_usage_keys:
            return _failed_detail("responses", f"unexpected usage keys: {sorted(usage.keys())}")
        return EndpointVerificationResult(
            name="responses",
            status="passed",
            detail="response output_text is non-empty",
        )
    except Exception as exc:
        return _failed_detail("responses", str(exc))


def _verify_responses_stream(
    client: TestClient,
    headers: dict[str, str],
    model: str | None,
) -> EndpointVerificationResult:
    if model is None:
        return _failed_detail("responses_stream", "missing model")

    try:
        with client.stream(
            "POST",
            "/v1/responses",
            headers=headers,
            json={
                "model": model,
                "stream": True,
                "input": "Return a short streamed verification reply.",
            },
        ) as response:
            if response.status_code != 200:
                return _failed_check("responses_stream", response)
            events = _parse_sse_events(response.iter_lines())

        event_names = [name for name, _ in events]
        expected_prefix = [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
        ]
        expected_suffix = [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        if event_names[:4] != expected_prefix or event_names[-4:] != expected_suffix:
            return _failed_detail("responses_stream", f"unexpected event sequence: {event_names}")
        if not event_names[4:-4] or any(name != "response.output_text.delta" for name in event_names[4:-4]):
            return _failed_detail("responses_stream", f"unexpected delta sequence: {event_names[4:-4]}")

        created_response = events[0][1]["response"]
        completed_response = events[-1][1]["response"]
        item_added = events[2][1]["item"]
        output_done = events[-2][1]["item"]
        text_delta = "".join(event[1]["delta"] for event in events[4:-4])
        text_done = events[-4][1]["text"]

        if created_response["model"] != model or completed_response["model"] != model:
            return _failed_detail("responses_stream", "streamed model mismatch")
        if created_response["id"] != completed_response["id"]:
            return _failed_detail("responses_stream", "response ids do not match")
        if created_response["status"] != "in_progress":
            return _failed_detail("responses_stream", "created response is not in_progress")
        if completed_response["status"] != "completed":
            return _failed_detail("responses_stream", "completed response is not completed")
        if item_added["id"] != output_done["id"]:
            return _failed_detail("responses_stream", "output item ids do not match")
        if item_added["status"] != "in_progress" or output_done["status"] != "completed":
            return _failed_detail("responses_stream", "output item statuses are invalid")
        if not str(text_delta).strip() or not str(text_done).strip():
            return _failed_detail("responses_stream", "streamed text is empty")
        return EndpointVerificationResult(
            name="responses_stream",
            status="passed",
            detail="sse events match response contract",
        )
    except Exception as exc:
        return _failed_detail("responses_stream", str(exc))


def _parse_sse_events(lines: Any) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    current_event: str | None = None
    for raw_line in lines:
        line = _normalize_line(raw_line)
        if not line:
            continue
        if line.startswith("event: "):
            current_event = line.removeprefix("event: ").strip()
            continue
        if line.startswith("data: "):
            if current_event is None:
                raise ValueError(f"missing event name for line: {line}")
            payload = json.loads(line.removeprefix("data: ").strip())
            events.append((current_event, payload))
            current_event = None
    return events


def _parse_data_line(line: str, name: str) -> dict[str, Any]:
    if not line.startswith("data: "):
        raise ValueError(f"{name} line must start with data:, got {line}")
    return json.loads(line.removeprefix("data: ").strip())


def _normalize_line(raw_line: Any) -> str:
    if isinstance(raw_line, bytes):
        return raw_line.decode("utf-8")
    return str(raw_line)


def _failed_check(name: str, response: Any) -> EndpointVerificationResult:
    return _failed_detail(
        name,
        f"http {response.status_code}: {_response_body(response)}",
    )


def _response_body(response: Any) -> str:
    try:
        body = response.text
    except Exception:
        try:
            body = response.content.decode("utf-8")
        except Exception:
            body = "<unreadable>"
    return body.strip() or "<empty>"


def _failed_detail(name: str, detail: str) -> EndpointVerificationResult:
    return EndpointVerificationResult(name=name, status="failed", detail=detail)


def _camoufox_runtime_missing_detail(install_hint: str) -> str:
    return f"camoufox runtime missing; {install_hint}"
