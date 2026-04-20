from typer.testing import CliRunner

import openclaw_algae.cli.app as cli_app_module
from openclaw_algae.browser import glm as glm_browser_module
from openclaw_algae.browser import glm_intl as glm_intl_browser_module
from openclaw_algae.cli.app import app
from openclaw_algae.storage.provider_store import load_provider_credentials


def test_login_defaults_to_browser_mode_when_no_manual_credentials_are_passed(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_capture(provider: str, *, state_dir):
        assert provider == "deepseek"
        assert state_dir == tmp_path / ".openclaw-algae"
        return {
            "cookie": "session=value",
            "bearer": "browser-token",
            "user_agent": "browser-ua",
        }

    monkeypatch.setattr(cli_app_module, "capture_provider_browser_credentials", fake_capture)
    runner = CliRunner()

    result = runner.invoke(app, ["login", "deepseek"])

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "deepseek")
    assert loaded is not None
    assert loaded.headers["authorization"] == "Bearer browser-token"


def test_login_deepseek_browser_saves_captured_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_capture(provider: str, *, state_dir):
        assert provider == "deepseek"
        assert state_dir == tmp_path / ".openclaw-algae"
        return {
            "cookie": "session=value",
            "bearer": "browser-token",
            "user_agent": "browser-ua",
        }

    monkeypatch.setattr(cli_app_module, "capture_provider_browser_credentials", fake_capture)
    runner = CliRunner()

    result = runner.invoke(app, ["login", "deepseek", "--browser"])

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "deepseek")
    assert loaded is not None
    assert loaded.cookie == "session=value"
    assert loaded.user_agent == "browser-ua"
    assert loaded.headers["authorization"] == "Bearer browser-token"


def test_login_qwen_intl_browser_saves_captured_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_capture(provider: str, *, state_dir):
        assert provider == "qwen-intl"
        assert state_dir == tmp_path / ".openclaw-algae"
        return {
            "cookie": "session=value",
            "session_token": "session-token",
            "user_agent": "browser-ua",
        }

    monkeypatch.setattr(cli_app_module, "capture_provider_browser_credentials", fake_capture)
    runner = CliRunner()

    result = runner.invoke(app, ["login", "qwen international", "--browser"])

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "qwen-intl")
    assert loaded is not None
    assert loaded.cookie == "session=value"
    assert loaded.user_agent == "browser-ua"
    assert loaded.metadata["session_token"] == "session-token"


def test_login_qwen_cn_browser_saves_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_capture(provider: str, *, state_dir):
        assert provider == "qwen-cn"
        assert state_dir == tmp_path / ".openclaw-algae"
        return {
            "cookie": "session=value",
            "user_agent": "browser-ua",
            "metadata": {"xsrf_token": "xsrf", "ut": "user-1"},
        }

    monkeypatch.setattr(cli_app_module, "capture_provider_browser_credentials", fake_capture)
    runner = CliRunner()

    result = runner.invoke(app, ["login", "qwen china", "--browser"])

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "qwen-cn")
    assert loaded is not None
    assert loaded.metadata["xsrf_token"] == "xsrf"
    assert loaded.metadata["ut"] == "user-1"


def test_login_kimi_browser_saves_cookie_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_capture(provider: str, *, state_dir):
        assert provider == "kimi"
        assert state_dir == tmp_path / ".openclaw-algae"
        return {
            "cookie": "kimi-auth=token-1",
            "user_agent": "browser-ua",
        }

    monkeypatch.setattr(cli_app_module, "capture_provider_browser_credentials", fake_capture)
    runner = CliRunner()

    result = runner.invoke(app, ["login", "kimi", "--browser"])

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "kimi")
    assert loaded is not None
    assert loaded.cookie == "kimi-auth=token-1"
    assert loaded.user_agent == "browser-ua"


def test_login_chatgpt_browser_saves_access_token_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_capture(provider: str, *, state_dir):
        assert provider == "chatgpt"
        assert state_dir == tmp_path / ".openclaw-algae"
        return {
            "cookie": "__Secure-next-auth.session-token=token",
            "access_token": "token",
            "user_agent": "browser-ua",
        }

    monkeypatch.setattr(cli_app_module, "capture_provider_browser_credentials", fake_capture)
    runner = CliRunner()

    result = runner.invoke(app, ["login", "chatgpt", "--browser"])

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "chatgpt")
    assert loaded is not None
    assert loaded.metadata["access_token"] == "token"


def test_login_claude_browser_saves_session_key_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_capture(provider: str, *, state_dir):
        assert provider == "claude"
        assert state_dir == tmp_path / ".openclaw-algae"
        return {
            "cookie": "sessionKey=sk-ant-sid01-test",
            "user_agent": "browser-ua",
            "metadata": {"session_key": "sk-ant-sid01-test"},
        }

    monkeypatch.setattr(cli_app_module, "capture_provider_browser_credentials", fake_capture)
    runner = CliRunner()

    result = runner.invoke(app, ["login", "claude", "--browser"])

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "claude")
    assert loaded is not None
    assert loaded.metadata["session_key"] == "sk-ant-sid01-test"


def test_login_doubao_browser_saves_sessionid_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_capture(provider: str, *, state_dir):
        assert provider == "doubao"
        assert state_dir == tmp_path / ".openclaw-algae"
        return {
            "cookie": "sessionid=session-1; ttwid=ttwid-1",
            "user_agent": "browser-ua",
            "metadata": {"sessionid": "session-1", "ttwid": "ttwid-1"},
        }

    monkeypatch.setattr(cli_app_module, "capture_provider_browser_credentials", fake_capture)
    runner = CliRunner()

    result = runner.invoke(app, ["login", "doubao", "--browser"])

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "doubao")
    assert loaded is not None
    assert loaded.metadata["sessionid"] == "session-1"
    assert loaded.metadata["ttwid"] == "ttwid-1"


def test_login_gemini_browser_saves_cookie_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_capture(provider: str, *, state_dir):
        assert provider == "gemini"
        assert state_dir == tmp_path / ".openclaw-algae"
        return {
            "cookie": "SID=sid; __Secure-1PSID=psid",
            "user_agent": "browser-ua",
        }

    monkeypatch.setattr(cli_app_module, "capture_provider_browser_credentials", fake_capture)
    runner = CliRunner()

    result = runner.invoke(app, ["login", "gemini", "--browser"])

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "gemini")
    assert loaded is not None
    assert loaded.cookie == "SID=sid; __Secure-1PSID=psid"
    assert loaded.user_agent == "browser-ua"


def test_login_grok_browser_saves_cookie_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_capture(provider: str, *, state_dir):
        assert provider == "grok"
        assert state_dir == tmp_path / ".openclaw-algae"
        return {
            "cookie": "sso=sso-token; _ga=ga-cookie",
            "user_agent": "browser-ua",
        }

    monkeypatch.setattr(cli_app_module, "capture_provider_browser_credentials", fake_capture)
    runner = CliRunner()

    result = runner.invoke(app, ["login", "grok", "--browser"])

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "grok")
    assert loaded is not None
    assert loaded.cookie == "sso=sso-token; _ga=ga-cookie"




def test_glm_browser_capture_requires_non_guest_token(monkeypatch) -> None:
    class FakePage:
        def goto(self, url: str, wait_until: str) -> None:
            return None

        def evaluate(self, script: str):
            return "browser-ua"

    class FakeContext:
        def __init__(self) -> None:
            self._calls = 0
            self.pages = [FakePage()]

        def cookies(self, urls):
            self._calls += 1
            if self._calls == 1:
                return [
                    {"name": "chatglm_token", "value": "guest-token"},
                    {"name": "chatglm_user_id", "value": "user-1"},
                ]
            return [
                {"name": "chatglm_token", "value": "real-token"},
                {"name": "chatglm_user_id", "value": "user-1"},
                {"name": "chatglm_refresh_token", "value": "refresh-token"},
            ]

        def close(self) -> None:
            return None

    class FakePlaywright:
        chromium = None

        def __init__(self) -> None:
            self.chromium = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def launch_persistent_context(self, user_data_dir: str, headless: bool):
            return FakeContext()

    monkeypatch.setattr(glm_browser_module, "prepare_browser_state_dir", lambda state_dir, provider: state_dir / provider)
    monkeypatch.setattr(glm_browser_module, "require_sync_playwright", lambda: (lambda: FakePlaywright()))

    timeline = iter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    monkeypatch.setattr(glm_browser_module.time, "time", lambda: next(timeline))
    monkeypatch.setattr(glm_browser_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        glm_browser_module,
        "build_cookie_string",
        lambda cookies: "; ".join(f"{cookie['name']}={cookie['value']}" for cookie in cookies),
    )
    monkeypatch.setattr(
        glm_browser_module,
        "_is_glm_guest_token",
        lambda token: token == "guest-token",
        raising=False,
    )

    captured = glm_browser_module.capture_glm_browser_credentials(state_dir=__import__("pathlib").Path("/tmp/state"))

    assert "real-token" in captured["cookie"]
    assert "guest-token" not in captured["cookie"]


def test_login_glm_cn_browser_saves_captured_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_capture(provider: str, *, state_dir):
        assert provider == "glm-cn"
        assert state_dir == tmp_path / ".openclaw-algae"
        return {
            "cookie": "chatglm_refresh_token=refresh",
            "user_agent": "browser-ua",
        }

    monkeypatch.setattr(cli_app_module, "capture_provider_browser_credentials", fake_capture)
    runner = CliRunner()

    result = runner.invoke(app, ["login", "glm cn", "--browser"])

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "glm-cn")
    assert loaded is not None
    assert loaded.cookie == "chatglm_refresh_token=refresh"


def test_login_glm_intl_browser_saves_cookie_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_capture(provider: str, *, state_dir):
        assert provider == "glm-intl"
        assert state_dir == tmp_path / ".openclaw-algae"
        return {
            "cookie": "refresh_token=refresh",
            "user_agent": "browser-ua",
        }

    monkeypatch.setattr(cli_app_module, "capture_provider_browser_credentials", fake_capture)
    runner = CliRunner()

    result = runner.invoke(app, ["login", "glm international", "--browser"])

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "glm-intl")
    assert loaded is not None
    assert loaded.cookie == "refresh_token=refresh"


def test_glm_intl_browser_capture_waits_for_real_login_instead_of_generic_token(monkeypatch) -> None:
    class FakePage:
        def __init__(self) -> None:
            self._calls = 0

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            return None

        def evaluate(self, script: str):
            if "navigator.userAgent" in script:
                return "browser-ua"
            self._calls += 1
            if self._calls == 1:
                return {
                    "href": "https://chat.z.ai/",
                    "has_input": True,
                    "has_sign_in": True,
                }
            return {
                "href": "https://chat.z.ai/",
                "has_input": True,
                "has_sign_in": False,
            }

    class FakeContext:
        def __init__(self) -> None:
            self._calls = 0
            self.pages = [FakePage()]

        def cookies(self, urls):
            self._calls += 1
            if self._calls == 1:
                return [
                    {"name": "token", "value": "anon-token"},
                    {"name": "_ga", "value": "ga-cookie"},
                ]
            return [
                {"name": "refresh_token", "value": "real-refresh-token"},
                {"name": "_ga", "value": "ga-cookie"},
            ]

        def close(self) -> None:
            return None

    class FakePlaywright:
        chromium = None

        def __init__(self) -> None:
            self.chromium = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def launch_persistent_context(self, user_data_dir: str, headless: bool):
            return FakeContext()

    monkeypatch.setattr(
        glm_intl_browser_module,
        "prepare_browser_state_dir",
        lambda state_dir, provider: state_dir / provider,
    )
    monkeypatch.setattr(
        glm_intl_browser_module,
        "require_sync_playwright",
        lambda: (lambda: FakePlaywright()),
    )
    timeline = iter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    monkeypatch.setattr(glm_intl_browser_module.time, "time", lambda: next(timeline))
    monkeypatch.setattr(glm_intl_browser_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        glm_intl_browser_module,
        "build_cookie_string",
        lambda cookies: "; ".join(f"{cookie['name']}={cookie['value']}" for cookie in cookies),
    )

    captured = glm_intl_browser_module.capture_glm_intl_browser_credentials(
        state_dir=__import__("pathlib").Path("/tmp/state")
    )

    assert "real-refresh-token" in captured["cookie"]
    assert "anon-token" not in captured["cookie"]


def test_login_mimo_browser_saves_cookie_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_capture(provider: str, *, state_dir):
        assert provider == "mimo"
        assert state_dir == tmp_path / ".openclaw-algae"
        return {
            "cookie": "mimo-token=token-1; mimo-user=user-1",
            "user_agent": "browser-ua",
        }

    monkeypatch.setattr(cli_app_module, "capture_provider_browser_credentials", fake_capture)
    runner = CliRunner()

    result = runner.invoke(app, ["login", "xiaomi mimo", "--browser"])

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "mimo")
    assert loaded is not None
    assert loaded.cookie == "mimo-token=token-1; mimo-user=user-1"


def test_login_browser_rejects_provider_without_browser_support(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    result = runner.invoke(app, ["login", "manus", "--browser"])

    assert result.exit_code != 0
    assert "Browser login is not implemented for manus" in result.stderr
