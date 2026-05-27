from opentoken.browser.common import CamoufoxRuntimeStatus
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.verification.service import (
    EndpointVerificationResult,
    ProviderVerificationResult,
    VerificationReport,
    render_verification_report,
    run_verification_suite,
    verification_exit_code,
)
import opentoken.api.routes.chat as chat_route_module
import opentoken.api.routes.responses as responses_route_module
import opentoken.verification.service as verification_service_module


def test_verification_exit_code_ignores_unlogged_providers_for_full_matrix() -> None:
    report = VerificationReport(
        requested_providers=(),
        results=(
            ProviderVerificationResult(
                provider="deepseek",
                display_name="DeepSeek",
                model="algae/deepseek/deepseek-chat",
                status="passed",
                checks=(),
            ),
            ProviderVerificationResult(
                provider="claude",
                display_name="Claude",
                model="algae/claude/claude-sonnet-4-6",
                status="not_logged_in",
                checks=(),
            ),
        ),
    )

    assert verification_exit_code(report) == 0


def test_verification_exit_code_fails_when_no_provider_verifies_in_full_matrix() -> None:
    report = VerificationReport(
        requested_providers=(),
        results=(
            ProviderVerificationResult(
                provider="deepseek",
                display_name="DeepSeek",
                model="algae/deepseek/deepseek-chat",
                status="not_logged_in",
                checks=(),
            ),
            ProviderVerificationResult(
                provider="claude",
                display_name="Claude",
                model="algae/claude/claude-sonnet-4-6",
                status="not_logged_in",
                checks=(),
            ),
        ),
    )

    assert verification_exit_code(report) == 1


def test_verification_exit_code_fails_for_requested_unlogged_provider() -> None:
    report = VerificationReport(
        requested_providers=("claude",),
        results=(
            ProviderVerificationResult(
                provider="claude",
                display_name="Claude",
                model="algae/claude/claude-sonnet-4-6",
                status="not_logged_in",
                checks=(),
            ),
        ),
    )

    assert verification_exit_code(report) == 1


def test_render_verification_report_includes_provider_and_check_details() -> None:
    report = VerificationReport(
        requested_providers=("deepseek",),
        results=(
            ProviderVerificationResult(
                provider="deepseek",
                display_name="DeepSeek",
                model="algae/deepseek/deepseek-chat",
                status="passed",
                checks=(
                    EndpointVerificationResult(
                        name="models",
                        status="passed",
                        detail="model present in catalog",
                    ),
                    EndpointVerificationResult(
                        name="chat",
                        status="passed",
                        detail="assistant message is non-empty",
                    ),
                ),
            ),
        ),
    )

    rendered = render_verification_report(report)

    assert "provider\tstatus\tmodel\tchecks" in rendered
    assert "deepseek\tpassed\talgae/deepseek/deepseek-chat" in rendered
    assert "models=passed(model present in catalog)" in rendered
    assert "chat=passed(assistant message is non-empty)" in rendered


def test_run_verification_suite_fails_fast_when_camoufox_runtime_is_missing(monkeypatch) -> None:
    def fail_router():
        raise AssertionError("browser-backed provider routes should not be invoked without runtime")

    monkeypatch.setattr(chat_route_module, "get_default_router", fail_router)
    monkeypatch.setattr(responses_route_module, "get_default_router", fail_router)
    monkeypatch.setattr(
        verification_service_module,
        "probe_camoufox_runtime",
        lambda: CamoufoxRuntimeStatus(
            package_installed=True,
            browser_installed=False,
            install_hint="run camoufox fetch",
        ),
    )
    monkeypatch.setattr(
        verification_service_module,
        "list_provider_credentials",
        lambda _providers_dir: (
            ProviderCredentialRecord(
                provider="doubao",
                kind="browser_session",
                cookie="sessionid=test",
                headers={},
                user_agent="ua",
                metadata={"sessionid": "test"},
                status="valid",
            ),
        ),
    )
    # The catalog now comes from live discovery; seed a doubao entry so the
    # "model present in catalog" assertion can succeed without hitting the
    # network. Both code paths read load_model_catalog — the verification
    # service computes the default model id, and /v1/models reads the registry.
    from opentoken.models.catalog import ModelCatalogEntry

    catalog_entries = [
        ModelCatalogEntry(
            id="algae/doubao/doubao-seed-2.0",
            provider="opentoken",
            name="Doubao Seed 2.0",
        ),
    ]
    monkeypatch.setattr(
        verification_service_module,
        "load_model_catalog",
        lambda: list(catalog_entries),
    )
    import opentoken.gateway.model_registry as model_registry_module

    monkeypatch.setattr(
        model_registry_module,
        "load_model_catalog",
        lambda: list(catalog_entries),
    )

    report = run_verification_suite(requested_providers=("doubao",))

    assert len(report.results) == 1
    result = report.results[0]
    assert result.provider == "doubao"
    assert result.status == "failed"
    assert result.checks[0].name == "models"
    assert result.checks[0].status == "passed"
    assert result.checks[1].detail == "camoufox runtime missing; run camoufox fetch"
    assert result.checks[-1].name == "responses_stream"
