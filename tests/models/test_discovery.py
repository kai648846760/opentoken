from openclaw_algae.models.discovery import (
    _extract_doubao_models_from_html,
    _extract_glm_cn_models_from_html,
    _extract_qwen_cn_models_from_dialog_text,
    _extract_qwen_intl_models_from_html,
    load_model_catalog,
)
from openclaw_algae.models.provider_credentials import ProviderCredentialRecord


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
        "openclaw_algae.models.discovery.load_provider_credentials",
        lambda providers_dir, provider: credentials if provider == "qwen-intl" else None,
    )
    monkeypatch.setattr(
        "openclaw_algae.models.discovery._DISCOVERERS",
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
