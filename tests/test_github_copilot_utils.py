from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest  # noqa: F401
from rich.console import Console

from git_copilot_commit import github_copilot


def make_model(
    model_id: str,
    *,
    vendor: str | None = None,
    family: str | None = None,
    context_window_tokens: int | None = None,
    endpoints: tuple[str, ...] = (),
) -> github_copilot.CopilotModel:
    return github_copilot.CopilotModel(
        id=model_id,
        name=model_id,
        vendor=vendor,
        family=family,
        max_context_window_tokens=context_window_tokens,
        supported_endpoints=endpoints,
    )


def test_normalize_domain_and_url_helpers() -> None:
    assert github_copilot.normalize_domain(" HTTPS://GHE.Example.COM/path ") == (
        "ghe.example.com"
    )
    assert github_copilot.normalize_domain("not-a-host") is None
    assert github_copilot.normalize_domain("bad host.example.com") is None
    assert github_copilot.normalize_domain(None) is None

    urls = github_copilot.get_urls("github.example.com")
    assert urls["device_code_url"] == "https://github.example.com/login/device/code"
    assert urls["access_token_url"] == (
        "https://github.example.com/login/oauth/access_token"
    )
    assert urls["copilot_token_url"] == (
        "https://api.github.example.com/copilot_internal/v2/token"
    )

    assert github_copilot.get_github_api_base_url("github.com") == (
        "https://api.github.com"
    )
    assert github_copilot.get_github_api_base_url("github.example.com") == (
        "https://api.github.example.com"
    )
    assert github_copilot.get_base_url_from_token(
        "token;proxy-ep=proxy.example.com"
    ) == ("https://api.example.com")
    assert github_copilot.get_base_url_from_token("token-without-proxy-host") is None
    assert (
        github_copilot.get_github_copilot_base_url(
            "token;proxy-ep=proxy.enterprise.example.com"
        )
        == "https://api.enterprise.example.com"
    )
    assert (
        github_copilot.get_github_copilot_base_url(None, "github.example.com")
        == "https://copilot-api.github.example.com"
    )
    assert github_copilot.get_github_copilot_base_url() == (
        "https://api.individual.githubcopilot.com"
    )


def test_credentials_path_uses_platformdirs_state_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(
        github_copilot,
        "Settings",
        lambda: SimpleNamespace(
            state_dir=state_dir,
            config_file=tmp_path / "config.json",
        ),
    )

    assert github_copilot.credentials_path() == state_dir / "copilot-auth.json"


def test_save_and_load_credentials_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "state" / "copilot-auth.json"
    monkeypatch.setattr(github_copilot, "credentials_path", lambda: path)
    credentials = github_copilot.CopilotCredentials(
        github_access_token="ghu_123",
        copilot_token="copilot-token",
        copilot_expires_at=2_000_000_000,
        enterprise_domain="github.example.com",
    )

    saved_path = github_copilot.save_credentials(credentials)
    loaded = github_copilot.load_credentials()

    assert saved_path == path
    assert loaded == credentials
    assert oct(saved_path.stat().st_mode & 0o777) == "0o600"


def test_credentials_and_payload_parsers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(github_copilot.time, "time", lambda: 1000.0)

    credentials = github_copilot.CopilotCredentials.from_dict(
        {
            "github_access_token": "ghu_123",
            "copilot_token": "token;proxy-ep=proxy.example.com",
            "copilot_expires_at": 1301,
            "enterprise_domain": "github.example.com",
        }
    )
    assert credentials.to_dict()["github_access_token"] == "ghu_123"
    assert not credentials.is_expired()
    assert credentials.base_url() == "https://api.example.com"

    expired = github_copilot.CopilotCredentials.from_dict(
        {
            "github_access_token": "ghu_123",
            "copilot_token": "plain-token",
            "copilot_expires_at": 1200,
        }
    )
    assert expired.is_expired()

    with pytest.raises(github_copilot.CopilotError):
        github_copilot.CopilotCredentials.from_dict({"github_access_token": ""})

    model = github_copilot.CopilotModel.from_payload(
        {
            "id": "gpt-5.4",
            "name": "GPT-5.4",
            "vendor": "openai",
            "capabilities": {
                "family": "gpt-5",
                "limits": {"max_context_window_tokens": 272000},
            },
            "supported_endpoints": ["/responses", "", 123],
        }
    )
    assert model.family == "gpt-5"
    assert model.max_context_window_tokens == 272000
    assert model.supported_endpoints == ("/responses",)

    with pytest.raises(github_copilot.CopilotError):
        github_copilot.CopilotModel.from_payload({"name": "Missing id"})

    viewer = github_copilot.GitHubViewer.from_payload(
        {
            "login": "kd",
            "name": "KD",
            "html_url": "https://github.com/kd",
            "type": "User",
            "plan": {"name": "Pro"},
        }
    )
    assert viewer.login == "kd"
    assert viewer.plan_name == "Pro"

    with pytest.raises(github_copilot.CopilotError):
        github_copilot.GitHubViewer.from_payload({})


def test_pick_model_covers_requested_defaults_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = [
        make_model("custom-model", vendor="openai"),
        make_model("gpt-4.1", vendor="openai"),
        make_model("claude-sonnet-4.6", vendor="anthropic"),
    ]

    assert github_copilot.pick_model(models, "claude-sonnet-4.6").id == (
        "claude-sonnet-4.6"
    )

    with pytest.raises(github_copilot.ModelSelectionError) as requested_error:
        github_copilot.pick_model(models, "missing-model")
    assert requested_error.value.requested_model == "missing-model"

    monkeypatch.setattr(
        github_copilot,
        "load_config",
        lambda: github_copilot.CopilotConfig(default_model="custom-model"),
    )
    assert github_copilot.pick_model(models).id == "custom-model"

    monkeypatch.setattr(
        github_copilot,
        "load_config",
        lambda: github_copilot.CopilotConfig(default_model="missing-config-model"),
    )
    monkeypatch.setattr(github_copilot, "config_path", lambda: Path("/tmp/config.json"))
    with pytest.raises(github_copilot.ModelSelectionError) as config_error:
        github_copilot.pick_model(models)
    assert config_error.value.configured_default_model == "missing-config-model"
    assert config_error.value.configured_default_model_path == Path("/tmp/config.json")

    monkeypatch.setattr(
        github_copilot,
        "load_config",
        lambda: github_copilot.CopilotConfig(default_model=None),
    )
    assert github_copilot.pick_model(models).id == "claude-sonnet-4.6"
    assert github_copilot.pick_model(
        [make_model("z-model"), make_model("a-model")]
    ).id == ("z-model")


def test_infer_api_surface_and_vendor_filtering() -> None:
    gpt5_model = make_model(
        "gpt-5.4",
        vendor="openai",
        family="gpt-5",
        endpoints=("/responses", "/chat/completions"),
    )
    chat_model = make_model(
        "gpt-4o",
        vendor="openai",
        endpoints=("/chat/completions",),
    )
    anthropic_model = make_model(
        "claude-sonnet-4.6",
        vendor="anthropic",
        endpoints=("/v1/messages",),
    )
    google_model = make_model("gemini-2.5-pro", vendor="google")

    assert github_copilot.infer_api_surface(gpt5_model) == "responses"
    assert github_copilot.infer_api_surface(chat_model) == "chat_completions"
    assert github_copilot.infer_api_surface(anthropic_model) == "anthropic_messages"
    assert github_copilot.infer_api_surface(google_model) == "chat_completions"
    assert github_copilot.format_supported_endpoints(chat_model) == "/chat/completions"
    assert github_copilot.format_supported_endpoints(google_model) == "default"
    assert github_copilot.format_context_window(gpt5_model) == "?"

    assert github_copilot.normalize_vendor_filter(" Gemini ") == "google"
    assert github_copilot.normalize_vendor_filter("claude") == "anthropic"
    assert github_copilot.normalize_vendor_filter("") is None

    models = [gpt5_model, anthropic_model, google_model]
    assert github_copilot.filter_models_by_vendor(models, None) == models
    assert github_copilot.filter_models_by_vendor(models, "gemini") == [google_model]

    with pytest.raises(github_copilot.CopilotError) as vendor_error:
        github_copilot.filter_models_by_vendor(models, "mistral")
    assert "anthropic, google, openai" in str(vendor_error.value)


def test_reauthentication_and_json_cache_helpers(tmp_path) -> None:
    assert github_copilot.should_reauthenticate(
        github_copilot.CopilotHttpError(401, "Unauthorized")
    )
    assert not github_copilot.should_reauthenticate(
        github_copilot.CopilotHttpError(500, "Internal Server Error")
    )
    assert github_copilot.should_reauthenticate(
        github_copilot.CopilotError("No cached Copilot credentials found.")
    )

    assert github_copilot.read_json_object(tmp_path / "missing.json") is None

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    with pytest.raises(github_copilot.CopilotError):
        github_copilot.read_json_object(bad_json)

    wrong_type = tmp_path / "list.json"
    wrong_type.write_text("[]", encoding="utf-8")
    with pytest.raises(github_copilot.CopilotError):
        github_copilot.read_json_object(wrong_type)


def test_extract_completion_text_handles_supported_shapes() -> None:
    assert (
        github_copilot.extract_completion_text(
            {"choices": [{"message": {"content": "  feat: add support  "}}]}
        )
        == "feat: add support"
    )

    assert (
        github_copilot.extract_completion_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"text": "feat: first line"},
                                {"content": "second line"},
                                {"ignored": True},
                            ]
                        }
                    }
                ]
            }
        )
        == "feat: first line\nsecond line"
    )

    with pytest.raises(github_copilot.CopilotError):
        github_copilot.extract_completion_text(
            {"choices": [{"message": {"content": []}}]}
        )


def test_extract_response_text_handles_text_refusals_and_errors() -> None:
    assert github_copilot.extract_response_text(
        {"output_text": "  feat: add support  "}
    ) == ("feat: add support")
    assert (
        github_copilot.extract_response_text(
            {
                "output": [
                    {
                        "content": [
                            {"text": "feat: first line"},
                            {"text": "second line"},
                        ]
                    }
                ]
            }
        )
        == "feat: first line\nsecond line"
    )
    assert (
        github_copilot.extract_response_text(
            {"output": [{"content": [{"refusal": "Refused for safety"}]}]}
        )
        == "Refused for safety"
    )

    with pytest.raises(github_copilot.CopilotError):
        github_copilot.extract_response_text({"output": []})


def test_request_json_retries_rate_limits_and_honors_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleep_calls: list[float] = []

    monkeypatch.setattr(github_copilot.time, "sleep", sleep_calls.append)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "2"},
                text="rate limited",
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"ok": True},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = github_copilot.request_json(
            client,
            "GET",
            "https://example.com/models",
        )

    assert payload == {"ok": True}
    assert attempts == 2
    assert sleep_calls == [2.0]


def test_request_json_retries_transient_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleep_calls: list[float] = []

    monkeypatch.setattr(github_copilot.time, "sleep", sleep_calls.append)
    monkeypatch.setattr(github_copilot.random, "uniform", lambda _a, _b: 0.0)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary network issue", request=request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"ok": True},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = github_copilot.request_json(
            client,
            "GET",
            "https://example.com/models",
        )

    assert payload == {"ok": True}
    assert attempts == 2
    assert sleep_calls == [github_copilot.HTTP_RETRY_BASE_DELAY_SECONDS]


def test_responses_completion_retries_retryable_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleep_calls: list[float] = []

    monkeypatch.setattr(github_copilot.time, "sleep", sleep_calls.append)
    monkeypatch.setattr(github_copilot.random, "uniform", lambda _a, _b: 0.0)

    credentials = github_copilot.CopilotCredentials(
        github_access_token="ghu_123",
        copilot_token="copilot-token",
        copilot_expires_at=2_000_000_000,
    )
    model = make_model(
        "gpt-5.4",
        vendor="openai",
        family="gpt-5",
        endpoints=("/responses",),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, text="service unavailable", request=request)

        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                "event: response.output_text.delta\n"
                'data: {"delta":"feat: add retries"}\n\n'
                "event: response.completed\n"
                'data: {"response":{"status":"completed"}}\n\n'
            ),
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        completion = github_copilot.responses_completion(
            client,
            credentials,
            model=model,
            prompt="Write a commit message",
        )

    assert completion == "feat: add retries"
    assert attempts == 2
    assert sleep_calls == [github_copilot.HTTP_RETRY_BASE_DELAY_SECONDS]


def test_render_model_selection_error_and_time_formatting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = github_copilot.ModelSelectionError(
        models=[
            make_model("gpt-5.4", vendor="openai"),
            make_model("claude-sonnet-4.6", vendor="anthropic"),
        ],
        configured_default_model="missing-model",
        configured_default_model_path=Path("/tmp/config.json"),
    )
    console = Console(record=True, width=120)

    console.print(github_copilot.render_model_selection_error(error))
    rendered = console.export_text()

    assert "Model Selection Error" in rendered
    assert "Config model" in rendered
    assert "missing-model" in rendered
    assert "Available Model IDs" in rendered
    assert "gpt-5.4" in rendered

    monkeypatch.setattr(github_copilot.time, "time", lambda: 1_700_000_000.0)
    assert github_copilot.format_relative_duration(3661) == "in 1h 1m"
    assert github_copilot.format_relative_duration(-59) == "59s ago"
    assert github_copilot.format_unix_timestamp(1_700_000_061).endswith("(in 1m 1s)")
    assert github_copilot.format_unix_timestamp(10**20) == str(10**20)


def test_print_model_table_shows_context_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table_console = Console(record=True, width=140)
    monkeypatch.setattr(github_copilot, "console", table_console)

    github_copilot.print_model_table(
        [
            make_model(
                "gpt-5.4",
                vendor="openai",
                context_window_tokens=272000,
                endpoints=("/responses",),
            )
        ]
    )

    rendered = table_console.export_text()

    assert "Context" in rendered
    assert "272,000" in rendered
