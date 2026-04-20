#!/usr/bin/env python3
"""通用回归测试脚本"""
import json
import sys
import time
from pathlib import Path

import httpx


def load_api_key() -> str:
    config_path = Path.home() / ".openclaw-algae" / "config.json"
    with config_path.open() as f:
        return json.load(f)["api_key"]


def test_models(base_url: str, api_key: str) -> tuple[bool, list[str]]:
    print("\n🧪 Testing GET /v1/models...")
    try:
        response = httpx.get(
            f"{base_url}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        models = [m["id"] for m in data["data"]]
        print(f"✅ Found {len(models)} models")
        return True, models
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, []


def test_chat(base_url: str, api_key: str, model: str) -> bool:
    print(f"\n🧪 Testing /v1/chat/completions ({model})...")
    try:
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(f"✅ Response: {content[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_chat_stream(base_url: str, api_key: str, model: str) -> bool:
    print(f"\n🧪 Testing /v1/chat/completions stream ({model})...")
    try:
        with httpx.stream(
            "POST", f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": True},
            timeout=60.0,
        ) as response:
            response.raise_for_status()
            chunks = sum(1 for line in response.iter_lines() if line.startswith("data: "))
            print(f"✅ Received {chunks} chunks")
            return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_responses(base_url: str, api_key: str, model: str) -> bool:
    print(f"\n🧪 Testing /v1/responses ({model})...")
    try:
        response = httpx.post(
            f"{base_url}/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": "hi"},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        print(f"✅ Response ID: {data['id']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_responses_stream(base_url: str, api_key: str, model: str) -> bool:
    print(f"\n🧪 Testing /v1/responses stream ({model})...")
    try:
        with httpx.stream(
            "POST", f"{base_url}/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": "hi", "stream": True},
            timeout=60.0,
        ) as response:
            response.raise_for_status()
            events = sum(1 for line in response.iter_lines() if line.startswith("event: "))
            print(f"✅ Received {events} events")
            return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def main():
    base_url = "http://127.0.0.1:32117"
    api_key = load_api_key()

    print("=" * 60)
    print("OpenClaw Algae Regression Tests")
    print("=" * 60)

    models_ok, all_models = test_models(base_url, api_key)
    if not models_ok:
        sys.exit(1)

    providers = {}
    for model_id in all_models:
        if model_id.startswith("algae/"):
            provider = model_id.split("/")[1]
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model_id)

    print(f"\n📦 Found {len(providers)} providers: {', '.join(providers.keys())}")

    results = {"models": models_ok}
    for provider, models in providers.items():
        print(f"\n{'=' * 60}")
        print(f"Testing Provider: {provider}")
        print(f"{'=' * 60}")

        model = models[0]
        time.sleep(2)

        results[f"{provider}_chat"] = test_chat(base_url, api_key, model)
        time.sleep(3)

        results[f"{provider}_chat_stream"] = test_chat_stream(base_url, api_key, model)
        time.sleep(3)

        results[f"{provider}_responses"] = test_responses(base_url, api_key, model)
        time.sleep(3)

        results[f"{provider}_responses_stream"] = test_responses_stream(base_url, api_key, model)
        time.sleep(3)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for name, passed in results.items():
        print(f"{'✅' if passed else '❌'} {name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
