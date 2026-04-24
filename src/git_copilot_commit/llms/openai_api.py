from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import core as llm

DEFAULT_SUPPORTED_ENDPOINTS = ("/chat/completions",)

console = Console()

LLMError = llm.LLMError
Model = llm.Model
HttpClientConfig = llm.HttpClientConfig


def request_headers(
    api_key: str | None,
    *,
    accept: str = "application/json",
) -> dict[str, str]:
    headers = {
        "Accept": accept,
        "Content-Type": "application/json",
        "User-Agent": llm.USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def list_models(
    client,
    *,
    base_url: str,
    api_key: str | None = None,
) -> list[Model]:
    payload = llm.request_json(
        client,
        "GET",
        f"{base_url}/models",
        headers=request_headers(api_key),
    )

    if isinstance(payload, dict):
        entries = payload.get("data")
    elif isinstance(payload, list):
        entries = payload
    else:
        raise LLMError("Models endpoint returned an unexpected payload.")

    if not isinstance(entries, list):
        raise LLMError("Models endpoint did not return a model list.")

    models: list[Model] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        models.append(
            Model.from_payload(
                entry,
                default_supported_endpoints=DEFAULT_SUPPORTED_ENDPOINTS,
            )
        )

    if not models:
        raise LLMError("Models endpoint returned no usable models.")

    return models


def default_model(
    model_id: str,
    *,
    supported_endpoints: tuple[str, ...] = DEFAULT_SUPPORTED_ENDPOINTS,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        supported_endpoints=supported_endpoints,
    )


def ensure_model_ready(
    *,
    base_url: str,
    api_key: str | None = None,
    model: str | None = None,
    default_model_id: str | None = None,
    configured_default_model_path: Path | None = None,
    provider_label: str = "OpenAI-compatible provider",
    http_client_config: HttpClientConfig | None = None,
) -> Model:
    if model is not None:
        return default_model(model)

    if default_model_id is not None:
        return default_model(default_model_id)

    with llm.make_http_client(http_client_config) as client:
        models = list_models(client, base_url=base_url, api_key=api_key)

    return llm.pick_model(
        models,
        provider_label=provider_label,
        configured_default_model_path=configured_default_model_path,
    )


def ask(
    prompt: str,
    *,
    base_url: str,
    api_key: str | None = None,
    model: str | None = None,
    default_model_id: str | None = None,
    configured_default_model_path: Path | None = None,
    provider_label: str = "OpenAI-compatible provider",
    http_client_config: HttpClientConfig | None = None,
    disable_thinking: bool = False,
) -> str:
    selected_model = ensure_model_ready(
        base_url=base_url,
        api_key=api_key,
        model=model,
        default_model_id=default_model_id,
        configured_default_model_path=configured_default_model_path,
        provider_label=provider_label,
        http_client_config=http_client_config,
    )

    api_surface = llm.infer_api_surface(selected_model)
    with llm.make_http_client(http_client_config) as client:
        if api_surface == "responses":
            return llm.responses_completion_request(
                client,
                f"{base_url}/responses",
                request_headers(api_key, accept="text/event-stream"),
                model_id=selected_model.id,
                prompt=prompt,
                disable_thinking=disable_thinking,
            )

        return llm.chat_completion_request(
            client,
            f"{base_url}/chat/completions",
            request_headers(api_key),
            model_id=selected_model.id,
            prompt=prompt,
            disable_thinking=disable_thinking,
        )


def get_available_models(
    *,
    base_url: str,
    api_key: str | None = None,
    vendor: str | None = None,
    provider_label: str = "OpenAI-compatible provider",
    http_client_config: HttpClientConfig | None = None,
) -> list[Model]:
    with llm.make_http_client(http_client_config) as client:
        models = list_models(client, base_url=base_url, api_key=api_key)

    return llm.filter_models_by_vendor(
        models,
        vendor,
        provider_label=provider_label,
    )


def show_summary(
    *,
    base_url: str,
    api_key: str | None = None,
    default_model_id: str | None = None,
    configured_default_model_path: Path | None = None,
    provider_label: str = "OpenAI-compatible provider",
    http_client_config: HttpClientConfig | None = None,
) -> None:
    available_models: list[Model] | None = None
    warning: str | None = None
    try:
        with llm.make_http_client(http_client_config) as client:
            available_models = list_models(client, base_url=base_url, api_key=api_key)
    except LLMError as exc:
        warning = f"Could not fetch model summary: {exc}"

    table = Table.grid(padding=(0, 1))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("Provider", provider_label)
    if configured_default_model_path is not None:
        table.add_row("Config file", str(configured_default_model_path))
    table.add_row("Base URL", base_url)
    table.add_row("API key", "configured" if api_key else "not set")

    if available_models is not None:
        table.add_row("Available models", str(len(available_models)))
        try:
            selected_model = llm.pick_model(
                available_models,
                default_model=default_model_id,
                provider_label=provider_label,
                configured_default_model_path=configured_default_model_path,
            )
        except LLMError as exc:
            table.add_row("Default model", f"Unavailable ({exc})")
        else:
            table.add_row(
                "Default model",
                f"{selected_model.id} ({llm.infer_api_surface(selected_model)})",
            )
    elif default_model_id is not None:
        table.add_row("Default model", f"{default_model_id} (chat_completions)")

    console.print(Panel.fit(table, title="LLM Summary"))
    if warning is not None:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
