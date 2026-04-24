from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from git_copilot_commit.llms import core as llm
from git_copilot_commit.llms import openai_api
from git_copilot_commit.llms import providers


class FakeSettings:
    def __init__(self, data: dict[str, object] | None = None) -> None:
        self._data = data or {}
        self.config_file = Path("/tmp/config.json")

    def get(self, key: str, default: object = None) -> object:
        return self._data.get(key, default)


def test_resolve_provider_config_infers_openai_from_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(providers, "Settings", lambda: FakeSettings())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    config = providers.resolve_provider_config(
        base_url=" http://127.0.0.1:11434/v1/chat/completions "
    )

    assert config.provider == "openai"
    assert config.base_url == "http://127.0.0.1:11434/v1"
    assert config.api_key is None


def test_resolve_provider_config_uses_nested_llm_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(
        providers,
        "Settings",
        lambda: FakeSettings(
            {
                "llm": {
                    "provider": "openai-compatible",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "api_key": "  ",
                }
            }
        ),
    )

    config = providers.resolve_provider_config()

    assert config.provider == "openai"
    assert config.base_url == "http://127.0.0.1:11434/v1"
    assert config.api_key is None


def test_load_default_model_uses_defaults_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        providers,
        "Settings",
        lambda: FakeSettings({"defaults": {"model": " llama3.2 "}}),
    )

    default_model, config_path = providers.load_default_model()

    assert default_model == "llama3.2"
    assert config_path == Path("/tmp/config.json")


def test_get_available_models_for_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "http://127.0.0.1:11434/v1/models"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"data": [{"id": "llama3.2"}]},
            request=request,
        )

    monkeypatch.setattr(
        openai_api.llm,
        "make_http_client",
        lambda _config=None: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    inventory = providers.get_available_models(
        provider_config=providers.ProviderConfig(
            provider="openai",
            base_url="http://127.0.0.1:11434/v1",
        )
    )

    assert inventory.base_url == "http://127.0.0.1:11434/v1"
    assert [model.id for model in inventory.models] == ["llama3.2"]
    assert inventory.models[0].supported_endpoints == ("/chat/completions",)
    assert llm.infer_api_surface(inventory.models[0]) == "chat_completions"


def test_ask_openai_provider_uses_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(providers, "Settings", lambda: FakeSettings())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            assert request.headers.get("authorization") is None
            request_payload = json.loads(request.content.decode("utf-8"))
            assert request_payload["model"] == "openai/gpt-oss-120b"
            assert request_payload["messages"][0]["content"] == "Write a commit message"
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "choices": [{"message": {"content": "feat: add local llm support"}}]
                },
                request=request,
            )

        raise AssertionError(f"Unexpected request path: {request.url.path}")

    monkeypatch.setattr(
        openai_api.llm,
        "make_http_client",
        lambda _config=None: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = providers.ask(
        "Write a commit message",
        provider_config=providers.ProviderConfig(
            provider="openai",
            base_url="http://127.0.0.1:11434/v1",
        ),
        model="openai/gpt-oss-120b",
    )

    assert response == "feat: add local llm support"


def test_ensure_model_ready_uses_configured_default_without_model_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        providers,
        "load_default_model",
        lambda: ("fallback-model", Path("/tmp/config.json")),
    )
    monkeypatch.setattr(
        openai_api.llm,
        "make_http_client",
        lambda _config=None: (_ for _ in ()).throw(
            AssertionError(
                "Model discovery should not run when a default model is set."
            )
        ),
    )

    model = providers.ensure_model_ready(
        provider_config=providers.ProviderConfig(
            provider="openai",
            base_url="http://127.0.0.1:11434/v1",
        )
    )

    assert model.id == "fallback-model"
    assert model.supported_endpoints == ("/chat/completions",)
