import sys
import types

from camoufox.addons import DefaultAddons
from openclaw_algae.browser.common import ensure_camoufox_runtime, probe_camoufox_runtime
import openclaw_algae.browser.common as common_module


def test_probe_camoufox_runtime_reports_missing_package(monkeypatch) -> None:
    monkeypatch.setattr(common_module.importlib.util, "find_spec", lambda name: None)

    status = probe_camoufox_runtime()

    assert status.package_installed is False
    assert status.browser_installed is False
    assert "uv sync" in status.install_hint


def test_ensure_camoufox_runtime_auto_fetches_missing_browser(monkeypatch) -> None:
    statuses = iter(
        [
            common_module.CamoufoxRuntimeStatus(
                package_installed=True,
                browser_installed=False,
                version="0.4.0",
            ),
            common_module.CamoufoxRuntimeStatus(
                package_installed=True,
                browser_installed=True,
                executable_path="/tmp/camoufox",
                version="0.4.0",
            ),
        ]
    )
    monkeypatch.setattr(common_module, "probe_camoufox_runtime", lambda: next(statuses))
    calls: list[str] = []
    monkeypatch.setattr(common_module, "_run_camoufox_fetch", lambda: calls.append("fetch"))

    status = ensure_camoufox_runtime()

    assert status.browser_installed is True
    assert calls == ["fetch"]


def test_require_camoufox_raises_if_runtime_still_missing_after_fetch(monkeypatch) -> None:
    monkeypatch.setattr(
        common_module,
        "probe_camoufox_runtime",
        lambda: common_module.CamoufoxRuntimeStatus(
            package_installed=True,
            browser_installed=False,
        ),
    )
    monkeypatch.setattr(common_module, "_run_camoufox_fetch", lambda: None)

    try:
        common_module.require_camoufox()
    except RuntimeError as exc:
        assert "camoufox fetch" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_require_camoufox_returns_sync_api_camoufox(monkeypatch) -> None:
    monkeypatch.setattr(
        common_module,
        "probe_camoufox_runtime",
        lambda: common_module.CamoufoxRuntimeStatus(
            package_installed=True,
            browser_installed=True,
            executable_path="/tmp/camoufox",
        ),
    )
    fake_sync_api = types.ModuleType("camoufox.sync_api")
    fake_camoufox = object()
    fake_sync_api.Camoufox = fake_camoufox
    monkeypatch.setitem(sys.modules, "camoufox.sync_api", fake_sync_api)

    assert common_module.require_camoufox() is fake_camoufox


def test_launch_persistent_context_excludes_broken_ubo_addon(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeManager:
        def __enter__(self):
            return "context"

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_camoufox(**kwargs):
        captured.update(kwargs)
        return FakeManager()

    monkeypatch.setattr(common_module, "_resolve_camoufox_launcher", lambda: fake_camoufox)
    session = common_module._CamoufoxPlaywrightSession()

    context = session.launch_persistent_context(tmp_path, headless=False)

    assert context == "context"
    assert captured["exclude_addons"] == [DefaultAddons.UBO]
    assert captured["humanize"] is True
    session.__exit__(None, None, None)


def test_launch_persistent_context_uses_host_os_and_cn_locale_for_cn_provider(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    class FakeManager:
        def __enter__(self):
            return "context"

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_camoufox(**kwargs):
        captured.update(kwargs)
        return FakeManager()

    monkeypatch.setattr(common_module, "_resolve_camoufox_launcher", lambda: fake_camoufox)
    monkeypatch.setattr(common_module.sys, "platform", "darwin")
    session = common_module._CamoufoxPlaywrightSession()

    context = session.launch_persistent_context(tmp_path / "qwen-cn", headless=False)

    assert context == "context"
    assert captured["os"] == "macos"
    assert captured["locale"] == ["zh-CN", "zh", "en-US", "en"]
    session.__exit__(None, None, None)


def test_launch_persistent_context_raises_when_camoufox_launch_fails(
    monkeypatch, tmp_path
) -> None:
    class BrokenManager:
        def __enter__(self):
            raise RuntimeError("camoufox launch failed")

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_camoufox(**kwargs):
        return BrokenManager()

    monkeypatch.setattr(common_module, "_resolve_camoufox_launcher", lambda: fake_camoufox)
    session = common_module._CamoufoxPlaywrightSession()

    try:
        session.launch_persistent_context(tmp_path / "doubao", headless=False)
    except RuntimeError as exc:
        assert "camoufox launch failed" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_probe_camoufox_runtime_normalizes_non_string_version(monkeypatch) -> None:
    fake_version_module = types.ModuleType("camoufox.__version__")
    fake_module = types.ModuleType("camoufox")
    fake_module.__version__ = fake_version_module

    monkeypatch.setattr(common_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(common_module.importlib, "import_module", lambda name: fake_module)
    monkeypatch.setattr(common_module, "_discover_camoufox_executable", lambda: "/tmp/camoufox")
    monkeypatch.setattr(common_module.importlib.metadata, "version", lambda name: "0.4.11")

    status = probe_camoufox_runtime()

    assert status.version == "0.4.11"


def test_discover_camoufox_executable_uses_pkgman_without_triggering_download(
    monkeypatch, tmp_path
) -> None:
    executable = tmp_path / "Camoufox.app" / "Contents" / "MacOS" / "camoufox"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text("", encoding="utf-8")
    fake_pkgman = types.ModuleType("camoufox.pkgman")
    fake_pkgman.OS_NAME = "mac"
    fake_pkgman.LAUNCH_FILE = {"mac": "../MacOS/camoufox"}
    fake_pkgman.INSTALL_DIR = tmp_path

    def fake_camoufox_path(*, download_if_missing: bool = True):
        assert download_if_missing is False
        return tmp_path

    fake_pkgman.camoufox_path = fake_camoufox_path

    def fake_import_module(name: str):
        if name == "camoufox.pkgman":
            return fake_pkgman
        raise AssertionError(f"unexpected import: {name}")

    def fail_subprocess_run(*args, **kwargs):
        raise AssertionError("subprocess probing should not be used when pkgman is available")

    monkeypatch.setattr(common_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(common_module.subprocess, "run", fail_subprocess_run)

    assert common_module._discover_camoufox_executable() == str(executable)


def test_launch_persistent_context_falls_back_to_playwright_when_camoufox_runtime_missing(
    monkeypatch, tmp_path
) -> None:
    launched: dict[str, object] = {}

    class FakeBrowserType:
        executable_path = "/tmp/fake-firefox"

        def launch_persistent_context(self, user_data_dir: str, **kwargs):
            launched["user_data_dir"] = user_data_dir
            launched["kwargs"] = kwargs
            return "fallback-context"

    class FakePlaywright:
        firefox = FakeBrowserType()
        chromium = object()
        webkit = object()

    class FakeManager:
        def __enter__(self):
            return FakePlaywright()

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(
        common_module,
        "_resolve_camoufox_launcher",
        lambda: None,
    )
    monkeypatch.setattr(
        common_module,
        "_resolve_playwright_fallback_browser_name",
        lambda: "firefox",
    )
    monkeypatch.setattr(common_module, "_import_playwright_sync_api", lambda: lambda: FakeManager())

    session = common_module._CamoufoxPlaywrightSession()
    context = session.launch_persistent_context(tmp_path / "glm-cn", headless=True)

    assert context == "fallback-context"
    assert launched["user_data_dir"] == str(tmp_path / "glm-cn")
    assert launched["kwargs"] == {
        "headless": True,
        "locale": "zh-CN",
        "executable_path": "/tmp/fake-firefox",
    }
    session.__exit__(None, None, None)


def test_launch_persistent_context_fast_fails_when_no_camoufox_and_no_playwright_fallback(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        common_module,
        "_resolve_camoufox_launcher",
        lambda: None,
    )
    monkeypatch.setattr(
        common_module,
        "_resolve_playwright_fallback_browser_name",
        lambda: None,
    )
    monkeypatch.setattr(
        common_module,
        "probe_camoufox_runtime",
        lambda: common_module.CamoufoxRuntimeStatus(
            package_installed=True,
            browser_installed=False,
        ),
    )

    session = common_module._CamoufoxPlaywrightSession()

    try:
        session.launch_persistent_context(tmp_path / "doubao", headless=False)
    except RuntimeError as exc:
        assert "camoufox" in str(exc).lower()
        assert "playwright fallback" in str(exc).lower()
    else:
        raise AssertionError("Expected RuntimeError")
