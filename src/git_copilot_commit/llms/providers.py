from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from . import copilot
from . import core as llm
from . import openai_api
from ..settings import Settings

ProviderName = Literal["copilot", "openai"]


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    provider: ProviderName
    base_url: str | None = None
    api_key: str | None = None

    @property
    def display_name(self) -> str:
        if self.provider == "copilot":
            return "GitHub Copilot"
        return "OpenAI-compatible provider"


@dataclass(frozen=True, slots=True)
class ModelInventory:
    provider_config: ProviderConfig
    base_url: str
    models: list[llm.Model]


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None

    trimmed = value.strip()
    if not trimmed:
        return None

    return trimmed


def normalize_provider(value: str | None) -> ProviderName | None:
    normalized = _normalize_optional_string(value)
    if normalized is None:
        return None

    aliases: dict[str, ProviderName] = {
        "copilot": "copilot",
        "github-copilot": "copilot",
        "github": "copilot",
        "openai": "openai",
        "openai-compatible": "openai",
        "openai-compatible-api": "openai",
    }
    return aliases.get(normalized.lower())


def _parse_provider(value: str | None, *, source: str) -> ProviderName | None:
    normalized = _normalize_optional_string(value)
    if normalized is None:
        return None

    provider = normalize_provider(normalized)
    if provider is None:
        raise llm.LLMError(f"{source} is invalid. Expected one of: copilot, openai.")

    return provider


def normalize_openai_base_url(value: str | None) -> str | None:
    normalized = _normalize_optional_string(value)
    if normalized is None:
        return None

    normalized = normalized.rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/models"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    parsed = urlparse(normalized)
    if not parsed.scheme or not parsed.netloc:
        raise llm.LLMError(
            "OpenAI-compatible base URL must include a scheme and host, for example "
            "`http://127.0.0.1:11434/v1`."
        )

    return normalized.rstrip("/")


def _read_config_string(
    value: object,
    *,
    label: str,
    allow_blank: bool = False,
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise llm.LLMError(
            f"Configured {label} in {Settings().config_file} is invalid."
        )

    trimmed = value.strip()
    if not trimmed:
        if allow_blank:
            return None
        raise llm.LLMError(
            f"Configured {label} in {Settings().config_file} is invalid."
        )

    return trimmed


def load_default_model(
    settings: Settings | None = None,
) -> tuple[str | None, Path]:
    settings = settings or Settings()
    value = settings.get("default_model")
    defaults = settings.get("defaults")

    if isinstance(defaults, dict) and "model" in defaults:
        value = defaults.get("model")

    if value is None:
        return None, settings.config_file

    if not isinstance(value, str) or not value.strip():
        raise llm.LLMError(
            f"Configured default model in {settings.config_file} is invalid."
        )

    return value.strip(), settings.config_file


def _load_provider_defaults_from_settings(
    settings: Settings | None = None,
) -> ProviderConfig | None:
    settings = settings or Settings()
    raw_provider = settings.get("provider")
    raw_base_url = settings.get("base_url")
    raw_api_key = settings.get("api_key")
    llm_settings = settings.get("llm")
    defaults = settings.get("defaults")

    if isinstance(llm_settings, dict):
        if "provider" in llm_settings:
            raw_provider = llm_settings.get("provider")
        if "base_url" in llm_settings:
            raw_base_url = llm_settings.get("base_url")
        if "api_key" in llm_settings:
            raw_api_key = llm_settings.get("api_key")

    if isinstance(defaults, dict) and "provider" in defaults:
        raw_provider = defaults.get("provider")

    provider = _parse_provider(
        _read_config_string(raw_provider, label="LLM provider"),
        source=str(settings.config_file),
    )
    base_url = normalize_openai_base_url(
        _read_config_string(raw_base_url, label="LLM base URL")
    )
    api_key = _read_config_string(
        raw_api_key,
        label="LLM API key",
        allow_blank=True,
    )

    if provider is None and base_url is None and api_key is None:
        return None

    return ProviderConfig(
        provider=provider or "copilot", base_url=base_url, api_key=api_key
    )


def resolve_provider_config(
    *,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ProviderConfig:
    cli_provider = _parse_provider(provider, source="`--provider`")
    cli_base_url = normalize_openai_base_url(base_url)
    cli_api_key = _normalize_optional_string(api_key)

    env_provider = _parse_provider(
        os.getenv("GIT_COPILOT_COMMIT_PROVIDER"),
        source="GIT_COPILOT_COMMIT_PROVIDER",
    )
    env_base_url = normalize_openai_base_url(os.getenv("GIT_COPILOT_COMMIT_BASE_URL"))
    env_api_key = _normalize_optional_string(os.getenv("GIT_COPILOT_COMMIT_API_KEY"))

    config_defaults = _load_provider_defaults_from_settings()
    config_provider = config_defaults.provider if config_defaults is not None else None
    config_base_url = config_defaults.base_url if config_defaults is not None else None
    config_api_key = config_defaults.api_key if config_defaults is not None else None

    resolved_provider = cli_provider or env_provider or config_provider
    if resolved_provider is None:
        if (
            cli_base_url is not None
            or env_base_url is not None
            or config_base_url is not None
        ):
            resolved_provider = "openai"
        else:
            resolved_provider = "copilot"

    if resolved_provider == "copilot":
        if cli_base_url is not None or cli_api_key is not None:
            raise llm.LLMError(
                "`--base-url` and `--api-key` can only be used with `--provider openai`."
            )
        return ProviderConfig(provider="copilot")

    resolved_base_url = cli_base_url or env_base_url or config_base_url
    if resolved_base_url is None:
        resolved_base_url = normalize_openai_base_url(os.getenv("OPENAI_BASE_URL"))
    if resolved_base_url is None:
        raise llm.LLMError(
            "OpenAI-compatible provider requires a base URL. Pass "
            "`--base-url http://127.0.0.1:11434/v1` or set "
            "`GIT_COPILOT_COMMIT_BASE_URL`."
        )

    resolved_api_key = cli_api_key or env_api_key or config_api_key
    if resolved_api_key is None:
        resolved_api_key = _normalize_optional_string(os.getenv("OPENAI_API_KEY"))

    return ProviderConfig(
        provider="openai",
        base_url=resolved_base_url,
        api_key=resolved_api_key,
    )


def ensure_model_ready(
    *,
    provider_config: ProviderConfig | None = None,
    model: str | None = None,
    http_client_config: llm.HttpClientConfig | None = None,
) -> llm.Model:
    resolved_provider = provider_config or resolve_provider_config()
    default_model, config_file = load_default_model()

    if resolved_provider.provider == "copilot":
        return copilot.ensure_auth_ready(
            model=model,
            default_model=default_model,
            configured_default_model_path=config_file,
            http_client_config=http_client_config,
        )

    if resolved_provider.base_url is None:
        raise llm.LLMError("OpenAI-compatible provider base URL is missing.")

    return openai_api.ensure_model_ready(
        base_url=resolved_provider.base_url,
        api_key=resolved_provider.api_key,
        model=model,
        default_model_id=default_model,
        configured_default_model_path=config_file,
        provider_label=resolved_provider.display_name,
        http_client_config=http_client_config,
    )


def get_available_models(
    *,
    provider_config: ProviderConfig | None = None,
    vendor: str | None = None,
    http_client_config: llm.HttpClientConfig | None = None,
) -> ModelInventory:
    resolved_provider = provider_config or resolve_provider_config()

    if resolved_provider.provider == "copilot":
        credentials, models = copilot.get_available_models(
            vendor=vendor,
            http_client_config=http_client_config,
        )
        return ModelInventory(
            provider_config=resolved_provider,
            base_url=credentials.base_url(),
            models=models,
        )

    if resolved_provider.base_url is None:
        raise llm.LLMError("OpenAI-compatible provider base URL is missing.")

    models = openai_api.get_available_models(
        base_url=resolved_provider.base_url,
        api_key=resolved_provider.api_key,
        provider_label=resolved_provider.display_name,
        vendor=vendor,
        http_client_config=http_client_config,
    )
    return ModelInventory(
        provider_config=resolved_provider,
        base_url=resolved_provider.base_url,
        models=models,
    )


def ask(
    prompt: str,
    *,
    provider_config: ProviderConfig | None = None,
    model: str | None = None,
    http_client_config: llm.HttpClientConfig | None = None,
    disable_thinking: bool = False,
) -> str:
    resolved_provider = provider_config or resolve_provider_config()
    default_model, config_file = load_default_model()

    if resolved_provider.provider == "copilot":
        return copilot.ask(
            prompt,
            model=model,
            default_model=default_model,
            configured_default_model_path=config_file,
            http_client_config=http_client_config,
            disable_thinking=disable_thinking,
        )

    if resolved_provider.base_url is None:
        raise llm.LLMError("OpenAI-compatible provider base URL is missing.")

    return openai_api.ask(
        prompt,
        base_url=resolved_provider.base_url,
        api_key=resolved_provider.api_key,
        model=model,
        default_model_id=default_model,
        configured_default_model_path=config_file,
        provider_label=resolved_provider.display_name,
        http_client_config=http_client_config,
        disable_thinking=disable_thinking,
    )


def show_summary(
    *,
    provider_config: ProviderConfig | None = None,
    http_client_config: llm.HttpClientConfig | None = None,
) -> None:
    resolved_provider = provider_config or resolve_provider_config()
    default_model, config_file = load_default_model()

    if resolved_provider.provider == "copilot":
        copilot.show_login_summary(
            default_model=default_model,
            configured_default_model_path=config_file,
            http_client_config=http_client_config,
        )
        return

    if resolved_provider.base_url is None:
        raise llm.LLMError("OpenAI-compatible provider base URL is missing.")

    openai_api.show_summary(
        base_url=resolved_provider.base_url,
        api_key=resolved_provider.api_key,
        default_model_id=default_model,
        configured_default_model_path=config_file,
        provider_label=resolved_provider.display_name,
        http_client_config=http_client_config,
    )
