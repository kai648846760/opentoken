from pathlib import Path

import typer
import uvicorn

from openclaw_algae.browser import capture_provider_browser_credentials
from openclaw_algae.cli.status_view import render_doctor_text, render_status_text
from openclaw_algae.config.app_config import load_or_create_app_config
from openclaw_algae.config.paths import (
    resolve_app_config_path,
    resolve_openclaw_config_path,
    resolve_providers_dir,
    resolve_state_dir,
)
from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.openclaw.bridge import (
    apply_algae_provider_patch,
    build_algae_provider_patch,
)
from openclaw_algae.providers.registry import (
    get_provider_definition,
    list_supported_providers,
    supported_provider_keys,
)
from openclaw_algae.storage.bootstrap import initialize_state_dir
from openclaw_algae.storage.provider_store import (
    delete_provider_credentials,
    list_provider_credentials,
    save_provider_credentials,
)
from openclaw_algae.verification.service import (
    render_verification_report,
    run_verification_suite,
    verification_exit_code,
)

app = typer.Typer(help='OpenClaw Algae CLI')


@app.command()
def onboard() -> None:
    """Initialize the algae state directory."""
    state_dir = initialize_state_dir(resolve_state_dir())
    typer.echo(f'Initialized state directory at {state_dir}')


@app.command()
def login(
    provider: list[str] = typer.Argument(..., metavar='PROVIDER'),
    cookie: str | None = typer.Option(None, '--cookie'),
    header: list[str] | None = typer.Option(None, '--header'),
    api_key: str | None = typer.Option(None, '--api-key'),
    user_agent: str | None = typer.Option(None, '--user-agent'),
    browser: bool = typer.Option(False, '--browser'),
) -> None:
    """Login a provider."""
    provider_raw = ' '.join(provider).strip()
    provider_definition = get_provider_definition(provider_raw)
    if provider_definition is None:
        supported = ', '.join(supported_provider_keys())
        raise typer.BadParameter(f'Unsupported provider: {provider_raw}. Supported providers: {supported}')

    provider_key = provider_definition.key
    state_dir = initialize_state_dir(resolve_state_dir())
    manual_credentials_provided = bool(cookie or header or api_key)
    any_login_options_provided = bool(cookie or header or api_key or user_agent)
    use_browser_login = browser or not any_login_options_provided

    if use_browser_login:
        if 'browser' not in provider_definition.login_modes:
            if browser:
                raise typer.BadParameter(
                    f'Browser login is not implemented for {provider_key} in v1. '
                    'Use manual credentials for now.'
                )
            raise typer.BadParameter(
                f'{provider_key} requires manual credentials. '
                'Provide --cookie, --header, or --api-key.'
            )
        captured = capture_provider_browser_credentials(provider_key, state_dir=state_dir)
        parsed_headers = dict(captured.get('headers', {}))
        bearer = str(captured.get('bearer', '')).strip()
        if bearer:
            parsed_headers['authorization'] = f'Bearer {bearer}'
        metadata = {
            key: str(value)
            for key, value in captured.get('metadata', {}).items()
            if value is not None and str(value) != ''
        }
        for key in ('access_token', 'session_token'):
            value = captured.get(key)
            if value is not None and str(value).strip():
                metadata[key] = str(value).strip()
        record = ProviderCredentialRecord(
            provider=provider_key,
            kind='browser_session',
            cookie=captured.get('cookie') or None,
            headers=parsed_headers,
            user_agent=captured.get('user_agent') or None,
            metadata=metadata,
            status='valid',
        )
        save_provider_credentials(resolve_providers_dir(), record)
        typer.echo(f'Captured browser credentials for {provider_key}')
        return

    if api_key and 'api_key' not in provider_definition.manual_auth:
        raise typer.BadParameter(f'{provider_key} does not support --api-key.')
    if not api_key and 'api_key' in provider_definition.manual_auth and provider_definition.manual_auth == (
        'api_key',
    ):
        raise typer.BadParameter(f'{provider_key} requires --api-key.')
    if not api_key and not cookie and not header:
        raise typer.BadParameter(
            f'{provider_key} requires cookie or header credentials for manual login. '
            'Provide --cookie or --header.'
        )

    parsed_headers: dict[str, str] = {}
    for item in header or []:
        if '=' not in item:
            raise typer.BadParameter(f'Invalid header format: {item}. Expected key=value.')
        key, value = item.split('=', 1)
        parsed_headers[key.strip()] = value.strip()
    if api_key:
        parsed_headers['api_key'] = api_key.strip()

    record = ProviderCredentialRecord(
        provider=provider_key,
        kind='api_key' if api_key else 'web_session',
        cookie=cookie,
        headers=parsed_headers,
        user_agent=user_agent,
        metadata={'api_key': api_key.strip()} if api_key else {},
        status='valid',
    )
    save_provider_credentials(resolve_providers_dir(), record)
    if api_key:
        typer.echo(f'Saved API key credentials for {provider_key}')
        return
    typer.echo(f'Saved credentials for {provider_key}')


@app.command()
def start(
    host: str | None = typer.Option(None, '--host'),
    port: int | None = typer.Option(None, '--port'),
) -> None:
    """Start the algae gateway service."""
    initialize_state_dir(resolve_state_dir())
    config = load_or_create_app_config(resolve_app_config_path())
    bind_host = host or str(config['host'])
    bind_port = port or int(config['port'])
    typer.echo(f'Starting OpenClaw Algae on http://{bind_host}:{bind_port}')
    uvicorn.run(
        'openclaw_algae.api.app:create_app',
        factory=True,
        host=bind_host,
        port=bind_port,
    )


@app.command()
def config(
    dry_run: bool = typer.Option(False, '--dry-run'),
    openclaw_config: Path | None = typer.Option(None, '--openclaw-config'),
) -> None:
    """Bridge OpenClaw to algae."""
    config = load_or_create_app_config(resolve_app_config_path())
    patch = build_algae_provider_patch(
        base_url=f"http://{config['host']}:{config['port']}/v1",
        api_key=str(config['api_key']),
    )
    if dry_run:
        import json

        typer.echo(json.dumps(patch, indent=2))
        return
    target_config = openclaw_config or resolve_openclaw_config_path()
    backup_path = apply_algae_provider_patch(target_config, patch)
    typer.echo(f'Updated OpenClaw config at {target_config}')
    if backup_path is not None:
        typer.echo(f'Backup written to {backup_path}')


@app.command()
def providers() -> None:
    """List provider states."""
    records = {record.provider: record for record in list_provider_credentials(resolve_providers_dir())}
    for provider in list_supported_providers():
        record = records.get(provider.key)
        status = record.status if record is not None else 'not_logged_in'
        modes = ','.join(provider.login_modes)
        typer.echo(f'{provider.key}	{provider.display_name}	{modes}	{status}')


@app.command()
def logout(provider: list[str] = typer.Argument(..., metavar='PROVIDER')) -> None:
    """Remove provider credentials."""
    provider_raw = ' '.join(provider).strip()
    provider_definition = get_provider_definition(provider_raw)
    provider_key = provider_definition.key if provider_definition is not None else provider_raw
    deleted = delete_provider_credentials(resolve_providers_dir(), provider_key)
    if deleted:
        typer.echo(f'Removed credentials for {provider_key}')
    else:
        typer.echo(f'No credentials found for {provider_key}')


@app.command()
def status() -> None:
    """Show service status."""
    typer.echo(render_status_text())


@app.command()
def doctor() -> None:
    """Run diagnostics."""
    typer.echo(render_doctor_text())


@app.command()
def verify(
    provider: list[str] | None = typer.Option(None, "--provider"),
) -> None:
    """Run endpoint contract verification."""
    requested_providers: list[str] = []
    for raw_provider in provider or []:
        provider_definition = get_provider_definition(raw_provider)
        if provider_definition is None:
            supported = ", ".join(supported_provider_keys())
            raise typer.BadParameter(
                f"Unsupported provider: {raw_provider}. Supported providers: {supported}"
            )
        requested_providers.append(provider_definition.key)

    report = run_verification_suite(requested_providers=tuple(requested_providers))
    typer.echo(render_verification_report(report))
    raise typer.Exit(code=verification_exit_code(report))
