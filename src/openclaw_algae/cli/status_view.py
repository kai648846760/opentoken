from openclaw_algae.browser.common import probe_camoufox_runtime
from openclaw_algae.config.paths import (
    resolve_app_config_path,
    resolve_openclaw_config_path,
    resolve_providers_dir,
    resolve_state_dir,
)
from openclaw_algae.storage.provider_store import list_provider_credentials


def render_status_text() -> str:
    return "\n".join(
        [
            f"state_dir={resolve_state_dir()}",
            f"openclaw_config={resolve_openclaw_config_path()}",
        ]
    )


def render_doctor_text() -> str:
    state_dir = resolve_state_dir()
    app_config = resolve_app_config_path()
    openclaw_config = resolve_openclaw_config_path()
    providers = list_provider_credentials(resolve_providers_dir())
    provider_keys = ",".join(record.provider for record in providers) if providers else "none"
    camoufox = probe_camoufox_runtime()

    return "\n".join(
        [
            f"state_dir={state_dir}",
            f"state_dir_exists={'yes' if state_dir.exists() else 'no'}",
            f"app_config={app_config}",
            f"app_config_exists={'yes' if app_config.exists() else 'no'}",
            f"openclaw_config={openclaw_config}",
            f"openclaw_config_exists={'yes' if openclaw_config.exists() else 'no'}",
            f"providers={len(providers)}",
            f"provider_keys={provider_keys}",
            f"camoufox_package_installed={'yes' if camoufox.package_installed else 'no'}",
            f"camoufox_browser_installed={'yes' if camoufox.browser_installed else 'no'}",
            f"camoufox_version={camoufox.version or 'unknown'}",
            f"camoufox_executable={camoufox.executable_path or 'missing'}",
            f"camoufox_install_hint={camoufox.install_hint}",
        ]
    )
