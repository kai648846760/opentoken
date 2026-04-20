from typer.testing import CliRunner

import openclaw_algae.cli.app as cli_app_module
from openclaw_algae.cli.app import app
from openclaw_algae.verification.service import (
    EndpointVerificationResult,
    ProviderVerificationResult,
    VerificationReport,
)


def test_verify_command_returns_zero_when_only_unlogged_providers_are_skipped(monkeypatch) -> None:
    def fake_run_verification_suite(*, requested_providers):
        assert requested_providers == ()
        return VerificationReport(
            requested_providers=(),
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
                    ),
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

    monkeypatch.setattr(cli_app_module, "run_verification_suite", fake_run_verification_suite)
    runner = CliRunner()

    result = runner.invoke(app, ["verify"])

    assert result.exit_code == 0
    assert "deepseek" in result.stdout
    assert "passed" in result.stdout
    assert "claude" in result.stdout
    assert "not_logged_in" in result.stdout


def test_verify_command_returns_nonzero_for_requested_not_logged_in_provider(monkeypatch) -> None:
    def fake_run_verification_suite(*, requested_providers):
        assert requested_providers == ("claude",)
        return VerificationReport(
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

    monkeypatch.setattr(cli_app_module, "run_verification_suite", fake_run_verification_suite)
    runner = CliRunner()

    result = runner.invoke(app, ["verify", "--provider", "claude"])

    assert result.exit_code == 1
    assert "claude" in result.stdout
    assert "not_logged_in" in result.stdout

