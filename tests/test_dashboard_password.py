"""Dashboard password setup: auth mode, password changes, and the CLI token."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import modal
import pytest
from click.testing import CliRunner
from modal.exception import AlreadyExistsError

import modal_dojo
from modal_dojo import _dashboard
from modal_dojo.cli import entrypoint_cli
from modal_dojo.cli import setup as cli_setup
from modal_dojo.common import config


@pytest.fixture
def secret(monkeypatch, tmp_path):
    """Fake the password Secret; returns the ``Secret.objects`` mock."""
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / ".modal-dojo.toml")
    objects = Mock()
    monkeypatch.setattr(modal.Secret, "objects", objects)
    return objects


def _created_env(objects: Mock) -> dict[str, str]:
    return objects.create.call_args.args[1]


def test_password_change_replaces_the_secret(secret):
    token = _dashboard.set_dashboard_password("new")

    assert [call[0] for call in secret.mock_calls] == ["delete", "create"]
    assert _created_env(secret) == {
        "DASHBOARD_PASSWORD": "new",
        "DASHBOARD_CLI_TOKEN": token,
    }
    assert _dashboard.set_dashboard_password("") == ""
    assert _created_env(secret) == {"DASHBOARD_PASSWORD": ""}


def _update(
    password=None, *, configured=False, live_url=None, attended=False, proxy=False
):
    return cli_setup._update_password(
        password,
        configured=configured,
        live_url=live_url,
        attended=attended,
        require_proxy_auth=proxy,
    )


def test_first_deploy_generates_password_when_unattended(secret, capsys):
    assert _update() is True

    env = _created_env(secret)
    assert f"Dashboard password: {env['DASHBOARD_PASSWORD']}" in capsys.readouterr().out
    assert config.get_cli_token_from_config() == env["DASHBOARD_CLI_TOKEN"]


def test_first_deploy_prompts_until_passwords_match(secret, monkeypatch, capsys):
    answers = iter(["mine", "typo", "mine", "mine"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(answers))

    assert _update(attended=True) is True

    assert _created_env(secret)["DASHBOARD_PASSWORD"] == "mine"
    assert "mine" not in capsys.readouterr().out


def test_proxy_auth_alone_means_no_password(secret, monkeypatch):
    monkeypatch.setattr("getpass.getpass", Mock(side_effect=AssertionError))

    assert _update(attended=True, proxy=True) is False

    assert _created_env(secret) == {"DASHBOARD_PASSWORD": ""}


def test_existing_password_is_kept(secret):
    assert _update(configured=True) is None
    secret.create.assert_not_called()


@pytest.mark.parametrize(("proxy", "warns"), [(False, True), (True, False)])
def test_unattended_redeploy_of_open_dashboard_only_warns(secret, capsys, proxy, warns):
    assert _update(live_url="https://dash.test", proxy=proxy) is None

    secret.create.assert_not_called()
    assert ("has no password" in capsys.readouterr().out) is warns


def test_no_password_clears_it(secret):
    config.save_cli_token_to_config("old")

    assert _update(False, configured=True) is False

    assert _created_env(secret) == {"DASHBOARD_PASSWORD": ""}
    assert config.get_cli_token_from_config() == ""


@pytest.mark.parametrize(
    ("live", "creates"), [(False, True), (True, False), (None, False)]
)
def test_password_flag_only_replaces_a_password_that_is_off(
    secret, monkeypatch, live, creates
):
    monkeypatch.setattr(config, "get_dashboard_password_auth", lambda _url: live)

    _update(True, configured=True, live_url="https://dash.test")

    assert secret.create.called is creates


@pytest.mark.parametrize(
    ("answers", "mode"),
    [([""], (False, True)), (["3"], (True, False)), (["9", "4"], (False, False))],
)
def test_choose_auth_mode(monkeypatch, answers, mode):
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(replies))

    assert cli_setup._choose_auth_mode() == mode


@pytest.fixture
def stub_deploy(monkeypatch):
    """Run setup() without Modal; returns the mocks it decides with."""
    monkeypatch.setattr("modal_dojo.cli.skills.warn_legacy_skills", lambda: None)
    monkeypatch.setattr(config, "require_migrated_config", lambda: None)
    monkeypatch.setattr(config, "get_dashboard_url", lambda: "https://saved.test")
    monkeypatch.setattr(config, "get_dashboard_proxy_auth", lambda _url: False)
    monkeypatch.setattr(config, "get_dashboard_trajectory_viewer", lambda: None)
    monkeypatch.setattr(config, "save_dashboard_url", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_setup, "ensure_modal_setup", lambda: None)
    monkeypatch.setattr(cli_setup, "_password_secret_exists", lambda: False)
    mocks = SimpleNamespace(
        live_url=None,
        update=Mock(return_value=None),
        choose=Mock(return_value=(True, True)),
        app=Mock(),
    )
    monkeypatch.setattr(cli_setup, "deployed_dashboard_url", lambda: mocks.live_url)
    monkeypatch.setattr(cli_setup, "_update_password", mocks.update)
    monkeypatch.setattr(cli_setup, "_choose_auth_mode", mocks.choose)
    monkeypatch.setattr(cli_setup, "ensure_proxy_auth", lambda **_k: True)
    dashboard = SimpleNamespace(
        app=mocks.app,
        fastapi_app=Mock(get_web_url=lambda: "https://dash.test"),
        ensure_creds_secret=lambda **_k: True,
        MODAL_CREDS_SECRET_NAME="creds",
    )
    monkeypatch.setattr(
        cli_setup, "_load_dashboard_for_deploy", lambda *_a, **_k: dashboard
    )
    return mocks


def test_stale_saved_url_is_not_a_live_dashboard(stub_deploy):
    cli_setup.setup(interactive=False)

    assert stub_deploy.update.call_args.kwargs["live_url"] is None


def test_first_interactive_setup_asks_for_the_auth_mode(stub_deploy, monkeypatch):
    monkeypatch.setattr(cli_setup.sys.stdin, "isatty", lambda: True)

    cli_setup.setup()

    assert stub_deploy.update.call_args.args == (True,)
    assert stub_deploy.update.call_args.kwargs["require_proxy_auth"] is True


@pytest.mark.parametrize(
    ("changed", "strategy"), [(True, "recreate"), (None, "rolling")]
)
def test_password_change_restarts_containers(stub_deploy, changed, strategy):
    stub_deploy.update.return_value = changed

    cli_setup.setup(interactive=False)

    assert stub_deploy.app.deploy.call_args.kwargs["strategy"] == strategy


@pytest.fixture
def log_secret(secret, monkeypatch):
    monkeypatch.delenv("MODAL_IS_REMOTE", raising=False)
    monkeypatch.setenv("MODAL_TOKEN_ID", "test-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(modal.Secret, "hydrate", Mock())
    monkeypatch.setattr(modal.Secret, "update", secret.update)
    return secret


@pytest.mark.parametrize(
    ("auth_flag", "label", "existing_secret"),
    [
        ("--no-proxy-auth", "View Dojo Dashboard", False),
        ("--proxy-auth", "Dashboard URL (requires Modal proxy authentication)", True),
    ],
)
def test_cli_setup_provisions_log_secret_once_and_shows_next_steps(
    auth_flag, label, existing_secret, log_secret, monkeypatch, tmp_path
):
    if existing_secret:
        log_secret.create.side_effect = AlreadyExistsError("Secret already exists")
    url = "https://my-workspace--modal-dojo-dashboard-fastapi-app.modal.run"
    config_path = tmp_path / ("long-directory-" * 5) / ".modal-dojo.toml"
    config_path.parent.mkdir()
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr(config, "LEGACY_CONFIG_PATH", tmp_path / ".training-gym.toml")
    monkeypatch.setattr(config, "_dashboard_requires_proxy_auth", False)
    monkeypatch.setattr(config, "_dashboard_trajectory_viewer", None)
    monkeypatch.setattr(cli_setup, "deployed_dashboard_url", lambda: None)
    monkeypatch.setattr(cli_setup, "ensure_proxy_auth", lambda **_k: True)
    monkeypatch.setattr(modal.App, "deploy", Mock())
    monkeypatch.setattr(modal.Function, "get_web_url", lambda self: url)
    # Load a fresh dashboard as the real CLI does, then restore the shared module.
    monkeypatch.delitem(sys.modules, "modal_dojo._dashboard")
    monkeypatch.delattr(modal_dojo, "_dashboard")

    result = CliRunner().invoke(entrypoint_cli, ["setup", auth_flag])

    assert result.exit_code == 0, result.output
    name = _dashboard.MODAL_CREDS_SECRET_NAME
    log_secret.create.assert_called_once_with(name, ANY)
    assert log_secret.update.call_count == int(existing_secret)
    assert result.stdout.startswith(
        "Setting up the Dojo dashboard (to monitor training runs and view logs).\n"
    )
    action = "Updated" if existing_secret else "Created"
    assert (
        result.stdout.splitlines().count(
            f"{action} Secret {name} (for dashboard log streaming)"
        )
        == 1
    )
    assert result.stdout.endswith(
        f"\n{label}: {url}\n"
        f"  (Saved to {config_path})\n\n"
        "→ Run `modal-dojo skills install` to install optional agent skills.\n"
    )
    assert config.get_dashboard_url() == url


@pytest.mark.parametrize("is_remote", [False, True])
def test_direct_dashboard_deploy_provisions_log_secret_locally(
    is_remote, log_secret, monkeypatch
):
    if is_remote:
        monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    entrypoint = Path(__file__).resolve().parents[1] / "dashboards" / "app.py"

    assert runpy.run_path(str(entrypoint))["app"] is _dashboard.app
    assert log_secret.create.call_count == (0 if is_remote else 1)
    log_secret.update.assert_not_called()
