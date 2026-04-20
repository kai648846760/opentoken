#!/usr/bin/env python3
"""Doubao provider regression test script."""
import json
import sys
import time
from pathlib import Path

import httpx


def load_api_key() -> str:
    """Load API key from config."""
    config_path = Path.home() / ".openclaw-algae" / "config.json"
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    with config_path.open() as f:
        config = json.load(f)
    return config["api_key"]


def test_models(base_url: str, api_key: str) -> bool:
    """Test GET /v1/models."""
    print("\n🧪 Testing GET /v1/models...")
    try:
        response = httpx.get(
            f"{base_url}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        assert data["object"] == "list", "object should be 'list'"
        assert isinstance(data["data"], list), "data should be a list"

        doubao_models = [m for m in data["data"] if "doubao" in m["id"]]
        if not doubao_models:
            print("⚠️  No doubao models found")
            return False

        print(f"✅ Found {len(doubao_models)} doubao models")
        for model in doubao_models:
            print(f"   - {model['id']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_chat_completions(base_url: str, api_key: str, model: str) -> bool:
    """Test POST /v1/chat/completions (non-streaming)."""
    print(f"\n🧪 Testing POST /v1/chat/completions (model: {model})...")
    try:
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say 'test ok' in English"}],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

        content = data["choices"][0]["message"]["content"]
        print(f"✅ Response: {content[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_chat_completions_stream(base_url: str, api_key: str, model: str) -> bool:
    """Test POST /v1/chat/completions (streaming)."""
    print(f"\n🧪 Testing POST /v1/chat/completions stream (model: {model})...")
    try:
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say 'stream ok'"}],
                "stream": True,
            },
            timeout=60.0,
        ) as response:
            response.raise_for_status()
            chunks = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    chunks.append(chunk)

            assert len(chunks) > 0, "Should receive chunks"
            print(f"✅ Received {len(chunks)} chunks")
            return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_responses(base_url: str, api_key: str, model: str) -> bool:
    """Test POST /v1/responses (non-streaming)."""
    print(f"\n🧪 Testing POST /v1/responses (model: {model})...")
    try:
        response = httpx.post(
            f"{base_url}/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "input": "Say 'response ok'"},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

        assert data["object"] == "response"
        assert data["status"] == "completed"
        assert "output" in data
        assert len(data["output"]) > 0

        print(f"✅ Response ID: {data['id']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_responses_stream(base_url: str, api_key: str, model: str) -> bool:
    """Test POST /v1/responses (streaming)."""
    print(f"\n🧪 Testing POST /v1/responses stream (model: {model})...")
    try:
        with httpx.stream(
            "POST",
            f"{base_url}/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "input": "Say 'stream response ok'", "stream": True},
            timeout=60.0,
        ) as response:
            response.raise_for_status()
            events = []
            for line in response.iter_lines():
                if line.startswith("event: "):
                    events.append(line[7:])

            assert len(events) > 0, "Should receive events"
            print(f"✅ Received {len(events)} events")
            return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    """Run all regression tests."""
    base_url = "http://127.0.0.1:32117"
    api_key = load_api_key()

    print("=" * 60)
    print("Doubao Provider Regression Tests")
    print("=" * 60)

    results = {}

    # Test /v1/models
    results["models"] = test_models(base_url, api_key)

    # Find doubao model
    doubao_model = None
    try:
        response = httpx.get(
            f"{base_url}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        data = response.json()
        doubao_models = [m for m in data["data"] if "doubao" in m["id"]]
        if doubao_models:
            doubao_model = doubao_models[0]["id"]
    except Exception:
        pass

    if not doubao_model:
        print("\n❌ Cannot find doubao model, skipping provider tests")
        sys.exit(1)

    # Test chat completions with delays to avoid rate limiting
    results["chat_completions"] = test_chat_completions(base_url, api_key, doubao_model)
    time.sleep(5)

    results["chat_completions_stream"] = test_chat_completions_stream(base_url, api_key, doubao_model)
    time.sleep(5)

    # Test responses
    results["responses"] = test_responses(base_url, api_key, doubao_model)
    time.sleep(5)

    results["responses_stream"] = test_responses_stream(base_url, api_key, doubao_model)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} passed")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

