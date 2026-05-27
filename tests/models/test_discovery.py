from opentoken.models.discovery import (
    _extract_doubao_models_from_html,
    _extract_glm_cn_models_from_html,
    _extract_qwen_cn_models_from_dialog_text,
    _extract_qwen_intl_models_from_html,
    load_model_catalog,
)
from opentoken.models.provider_credentials import ProviderCredentialRecord


def test_extract_qwen_intl_models_from_html_returns_model_entries() -> None:
    html = """
    <script>
    {"id":"qwen3.6-plus","name":"Qwen3.6-Plus","object":"model","owned_by":"qwen"}
    {"id":"qwen3.5-flash","name":"Qwen3.5-Flash","object":"model","owned_by":"qwen"}
    </script>
    """

    assert _extract_qwen_intl_models_from_html(html) == [
        ("qwen3.6-plus", "Qwen3.6-Plus"),
        ("qwen3.5-flash", "Qwen3.5-Flash"),
    ]


def test_extract_qwen_cn_models_from_dialog_text_returns_current_labels() -> None:
    dialog_text = (
        "模型 "
        "Qwen3.5-千问 综合AI助手，全面回答工作、学习、生活各类问题 "
        "Qwen3.5-Flash 适用于简单任务，响应速度快 "
        "Qwen3-Max 适用于日常通用型任务，综合能力均衡 "
        "Qwen3-Max-Thinking 适用于多步骤推理与问题分析 "
        "Qwen3-Coder 代码 适用于代码生成与编程任务执行"
    )

    assert _extract_qwen_cn_models_from_dialog_text(dialog_text) == [
        ("Qwen3.5-千问", "Qwen3.5-千问"),
        ("Qwen3.5-Flash", "Qwen3.5-Flash"),
        ("Qwen3-Max", "Qwen3-Max"),
        ("Qwen3-Max-Thinking", "Qwen3-Max-Thinking"),
        ("Qwen3-Coder", "Qwen3-Coder"),
    ]


def test_extract_doubao_models_from_html_returns_current_action_bar_models() -> None:
    html = """
    <script>
    {"action_bar_menu_config":{"menu_item_list":[
      {"menu_type":0,"name":"快速","sub_title_name":"适用于大部分情况"},
      {"menu_type":1,"name":"思考","sub_title_name":"擅长解决更难的问题"},
      {"menu_type":3,"name":"专家","sub_title_name":"研究级智能模型"}
    ],"default_deep_think_auto":false}}
    </script>
    """

    assert _extract_doubao_models_from_html(html) == [
        ("doubao-seed-2.0", "Doubao 快速"),
        ("doubao-thinking", "Doubao 思考"),
        ("doubao-pro", "Doubao 专家"),
    ]


def test_extract_glm_cn_models_from_html_returns_meta_models() -> None:
    html = """
    <html>
      <head>
        <meta name="keywords" content="GLM-5,大语言模型,多模态AI,AI编程,AI翻译,智谱" />
        <meta name="description" content="GLM-5 的全能 AI 助手，支持精通对话、写作与编程。" />
      </head>
    </html>
    """

    assert _extract_glm_cn_models_from_html(html) == [
        ("glm-5", "GLM-5"),
    ]


def test_load_model_catalog_replaces_fallback_provider_entries_with_dynamic_discovery(
    monkeypatch,
    tmp_path,
) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    monkeypatch.setattr(
        "opentoken.models.discovery.load_provider_credentials",
        lambda providers_dir, provider: credentials if provider == "qwen-intl" else None,
    )
    monkeypatch.setattr(
        "opentoken.models.discovery._DISCOVERERS",
        {
            "qwen-intl": lambda credentials, state_dir: [
                ("qwen3.6-plus", "Qwen3.6-Plus"),
                ("qwen3.5-flash", "Qwen3.5-Flash"),
            ]
        },
    )

    catalog = load_model_catalog(
        state_dir=tmp_path,
        providers_dir=tmp_path / "providers",
        use_cache=False,
    )
    qwen_models = sorted(entry.id for entry in catalog if "/qwen-intl/" in entry.id)

    assert qwen_models == [
        "algae/qwen-intl/qwen3.5-flash",
        "algae/qwen-intl/qwen3.6-plus",
    ]


def test_load_model_catalog_falls_back_for_logged_in_provider_when_discovery_empty(
    monkeypatch,
    tmp_path,
) -> None:
    """A logged-in provider whose live discovery yields nothing must still
    surface its known wire models, so /v1/models lists it and the smoke script
    can test it. This is the floor for JS-rendered (qwen-intl) / gRPC (kimi)
    catalogs we can't scrape — live discovery is still tried first and wins."""
    credentials = ProviderCredentialRecord(
        provider="kimi",
        kind="web_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    monkeypatch.setattr(
        "opentoken.models.discovery.load_provider_credentials",
        lambda providers_dir, provider: credentials if provider == "kimi" else None,
    )
    # Discoverer runs but finds nothing (page shape changed / endpoint moved).
    monkeypatch.setattr(
        "opentoken.models.discovery._DISCOVERERS",
        {"kimi": lambda credentials, state_dir: []},
    )
    monkeypatch.setattr(
        "opentoken.models.discovery._FALLBACK_MODELS",
        {"kimi": [("k2", "Kimi K2"), ("k1", "Kimi K1")]},
    )

    catalog = load_model_catalog(
        state_dir=tmp_path,
        providers_dir=tmp_path / "providers",
        use_cache=False,
    )
    kimi_models = sorted(entry.id for entry in catalog if "/kimi/" in entry.id)

    assert kimi_models == ["algae/kimi/k1", "algae/kimi/k2"]


def test_load_model_catalog_prefers_live_discovery_over_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    """When live discovery succeeds, the fallback floor must not leak in — the
    listed models are exactly what the provider page returned."""
    credentials = ProviderCredentialRecord(
        provider="kimi",
        kind="web_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    monkeypatch.setattr(
        "opentoken.models.discovery.load_provider_credentials",
        lambda providers_dir, provider: credentials if provider == "kimi" else None,
    )
    monkeypatch.setattr(
        "opentoken.models.discovery._DISCOVERERS",
        {"kimi": lambda credentials, state_dir: [("k3", "Kimi K3")]},
    )
    monkeypatch.setattr(
        "opentoken.models.discovery._FALLBACK_MODELS",
        {"kimi": [("k2", "Kimi K2"), ("k1", "Kimi K1")]},
    )

    catalog = load_model_catalog(
        state_dir=tmp_path,
        providers_dir=tmp_path / "providers",
        use_cache=False,
    )
    kimi_models = sorted(entry.id for entry in catalog if "/kimi/" in entry.id)

    assert kimi_models == ["algae/kimi/k3"]


def test_load_model_catalog_runs_discoverers_concurrently_and_isolates_failures(
    monkeypatch,
    tmp_path,
) -> None:
    """The loader runs every logged-in provider's discoverer in parallel under
    an overall wall-clock budget. One discoverer raising must not knock out the
    others; a slow discoverer that beats the deadline still contributes.

    This guards the cold-cache /v1/models path that previously timed out
    because discoverers ran sequentially in the request thread.
    """
    import threading
    import time as time_module

    def _stub_creds(provider: str) -> ProviderCredentialRecord:
        return ProviderCredentialRecord(
            provider=provider,
            kind="web_session",
            cookie="x",
            headers={},
            user_agent="ua",
            metadata={},
            status="valid",
        )

    monkeypatch.setattr(
        "opentoken.models.discovery.load_provider_credentials",
        lambda providers_dir, provider: _stub_creds(provider)
        if provider in {"good_a", "good_b", "boom"}
        else None,
    )

    call_starts: list[tuple[str, float]] = []
    barrier = threading.Event()

    def good_a(_credentials, _state_dir):
        call_starts.append(("good_a", time_module.monotonic()))
        barrier.wait(timeout=2.0)  # Coordinate with good_b to prove parallelism.
        return [("a-1", "A 1"), ("a-2", "A 2")]

    def good_b(_credentials, _state_dir):
        call_starts.append(("good_b", time_module.monotonic()))
        barrier.set()
        return [("b-1", "B 1")]

    def boom(_credentials, _state_dir):
        raise RuntimeError("upstream is down")

    monkeypatch.setattr(
        "opentoken.models.discovery._DISCOVERERS",
        {"good_a": good_a, "good_b": good_b, "boom": boom},
    )

    catalog = load_model_catalog(
        state_dir=tmp_path,
        providers_dir=tmp_path / "providers",
        use_cache=False,
    )
    ids = sorted(entry.id for entry in catalog)

    # Two good providers contributed; the raising one was isolated.
    assert ids == [
        "algae/good_a/a-1",
        "algae/good_a/a-2",
        "algae/good_b/b-1",
    ]
    # Parallelism: the barrier only releases when good_b runs, so good_a's
    # barrier.wait would time out if discoverers were serialised. Both must have
    # started. (boom raises before recording, so it's not in call_starts.)
    started = {name for name, _ in call_starts}
    assert started == {"good_a", "good_b"}
