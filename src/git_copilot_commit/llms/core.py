from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

USER_AGENT = "git-copilot-commit"
DEFAULT_MODEL_PREFERENCES = (
    "gpt-5.3-codex",
    "gpt-5.4",
    "claude-opus-4.6",
    "claude-opus-4.5",
    "claude-sonnet-4.6",
    "claude-sonnet-4.5",
    "gemini-2.5-pro",
    "gpt-4.1",
    "gpt-4o",
)
HTTP_RETRY_ATTEMPTS = 3
HTTP_RETRY_BASE_DELAY_SECONDS = 0.5
HTTP_RETRY_MAX_DELAY_SECONDS = 4.0
HTTP_RETRY_MAX_JITTER_SECONDS = 0.25
HTTP_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

console = Console()


class LLMError(RuntimeError):
    pass


class LLMHttpError(LLMError):
    def __init__(
        self, status_code: int, reason_phrase: str, detail: str | None = None
    ) -> None:
        self.status_code = status_code
        self.reason_phrase = reason_phrase
        self.detail = detail
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{status_code} {reason_phrase}{suffix}")


class ModelSelectionError(LLMError):
    def __init__(
        self,
        *,
        models: list["Model"],
        provider_label: str = "LLM provider",
        requested_model: str | None = None,
        configured_default_model: str | None = None,
        configured_default_model_path: Path | None = None,
    ) -> None:
        self.models = models
        self.provider_label = provider_label
        self.requested_model = requested_model
        self.configured_default_model = configured_default_model
        self.configured_default_model_path = configured_default_model_path

        if requested_model is not None:
            message = f"Model `{requested_model}` was not returned by {provider_label}."
        else:
            location = (
                f" from {configured_default_model_path}"
                if configured_default_model_path is not None
                else ""
            )
            message = (
                f"Configured default model `{configured_default_model}`{location} "
                f"was not returned by {provider_label}."
            )

        super().__init__(message)


@dataclass(slots=True)
class Model:
    id: str
    name: str
    vendor: str | None = None
    family: str | None = None
    max_context_window_tokens: int | None = None
    supported_endpoints: tuple[str, ...] = ()

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        default_supported_endpoints: tuple[str, ...] = (),
    ) -> "Model":
        model_id = payload.get("id")
        name = payload.get("name")
        vendor = payload.get("vendor")
        capabilities = payload.get("capabilities")
        supported_endpoints = payload.get("supported_endpoints")

        family: str | None = None
        max_context_window_tokens: int | None = None
        if isinstance(capabilities, dict):
            raw_family = capabilities.get("family")
            if isinstance(raw_family, str) and raw_family:
                family = raw_family

            limits = capabilities.get("limits")
            if isinstance(limits, dict):
                raw_context_window = limits.get("max_context_window_tokens")
                if isinstance(raw_context_window, int) and raw_context_window > 0:
                    max_context_window_tokens = raw_context_window

        endpoints: list[str] = []
        if isinstance(supported_endpoints, list):
            for entry in supported_endpoints:
                if isinstance(entry, str) and entry:
                    endpoints.append(entry)

        if not isinstance(model_id, str) or not model_id:
            raise LLMError("Models endpoint returned a model without an id.")

        if not endpoints:
            endpoints = list(default_supported_endpoints)

        return cls(
            id=model_id,
            name=name if isinstance(name, str) and name else model_id,
            vendor=vendor if isinstance(vendor, str) and vendor else None,
            family=family,
            max_context_window_tokens=max_context_window_tokens,
            supported_endpoints=tuple(endpoints),
        )


@dataclass(frozen=True, slots=True)
class HttpClientConfig:
    native_tls: bool = True
    insecure: bool = False
    ca_bundle: str | None = None

    @property
    def use_native_tls(self) -> bool:
        return self.native_tls and not self.insecure and self.ca_bundle is None

    @property
    def verify(self) -> bool | str:
        if self.insecure:
            return False
        if self.ca_bundle:
            return self.ca_bundle
        return True


_NATIVE_TLS_ENABLED = False


def _maybe_enable_native_tls(native_tls: bool) -> None:
    global _NATIVE_TLS_ENABLED
    if not native_tls or _NATIVE_TLS_ENABLED:
        return

    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        return

    _NATIVE_TLS_ENABLED = True


def make_http_client(
    http_client_config: HttpClientConfig | None = None,
) -> httpx.Client:
    config = http_client_config or HttpClientConfig()
    _maybe_enable_native_tls(config.use_native_tls)

    return httpx.Client(
        verify=config.verify,
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, connect=10.0),
    )


def truncate_response_detail(detail: str) -> str:
    detail = detail.strip()
    if len(detail) > 400:
        return f"{detail[:397]}..."
    return detail


def should_retry_status_code(status_code: int) -> bool:
    return status_code in HTTP_RETRYABLE_STATUS_CODES


def compute_retry_delay_seconds(attempt: int, retry_after: str | None = None) -> float:
    if retry_after is not None:
        try:
            delay = float(retry_after.strip())
        except ValueError:
            delay = -1.0
        else:
            if delay >= 0:
                return delay

    backoff = min(
        HTTP_RETRY_MAX_DELAY_SECONDS,
        HTTP_RETRY_BASE_DELAY_SECONDS * (2**attempt),
    )
    return backoff + random.uniform(0.0, HTTP_RETRY_MAX_JITTER_SECONDS)


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, str] | None = None,
    json_body: Any | None = None,
) -> Any:
    for attempt in range(HTTP_RETRY_ATTEMPTS):
        try:
            response = client.request(
                method,
                url,
                headers=headers,
                data=data,
                json=json_body,
            )
        except httpx.TransportError as exc:
            if attempt < HTTP_RETRY_ATTEMPTS - 1:
                time.sleep(compute_retry_delay_seconds(attempt))
                continue
            raise LLMError(f"Request failed for {url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Request failed for {url}: {exc}") from exc

        if response.is_error:
            detail = truncate_response_detail(response.text)
            error = LLMHttpError(response.status_code, response.reason_phrase, detail)
            if (
                should_retry_status_code(response.status_code)
                and attempt < HTTP_RETRY_ATTEMPTS - 1
            ):
                time.sleep(
                    compute_retry_delay_seconds(
                        attempt, response.headers.get("retry-after")
                    )
                )
                continue
            raise error

        break

    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        try:
            return response.json()
        except ValueError as exc:
            detail = truncate_response_detail(response.text)
            raise LLMError(
                f"Expected JSON from {url}, got {content_type or 'unknown content type'}: {detail}"
            ) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise LLMError(f"Invalid JSON response from {url}.") from exc


def iter_sse_events(response: httpx.Response, url: str):
    event_name: str | None = None
    data_lines: list[str] = []

    def decode_event(raw_data: str, current_event: str | None) -> Any:
        if raw_data == "[DONE]":
            return None
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            label = current_event or "message"
            raise LLMError(f"Invalid SSE event payload from {url} ({label}).") from exc
        if isinstance(payload, dict) and current_event and "type" not in payload:
            payload = dict(payload)
            payload["type"] = current_event
        return payload

    for raw_line in response.iter_lines():
        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
        if not line:
            if data_lines:
                current_event = event_name
                raw_data = "\n".join(data_lines)
                event_name = None
                data_lines = []
                payload = decode_event(raw_data, current_event)
                if payload is not None:
                    yield payload
            else:
                event_name = None
            continue

        if line.startswith(":"):
            continue

        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]

        if field == "event":
            event_name = value
            continue
        if field == "data":
            data_lines.append(value)

    if data_lines:
        payload = decode_event("\n".join(data_lines), event_name)
        if payload is not None:
            yield payload


def pick_model(
    models: list[Model],
    requested_model: str | None = None,
    *,
    default_model: str | None = None,
    provider_label: str = "LLM provider",
    configured_default_model_path: Path | None = None,
) -> Model:
    by_id = {model.id: model for model in models}
    if requested_model is not None:
        if requested_model not in by_id:
            raise ModelSelectionError(
                models=models,
                provider_label=provider_label,
                requested_model=requested_model,
            )
        return by_id[requested_model]

    if default_model is not None:
        if default_model not in by_id:
            raise ModelSelectionError(
                models=models,
                provider_label=provider_label,
                configured_default_model=default_model,
                configured_default_model_path=configured_default_model_path,
            )
        return by_id[default_model]

    for preferred_model in DEFAULT_MODEL_PREFERENCES:
        if preferred_model in by_id:
            return by_id[preferred_model]

    return models[0]


def infer_api_surface(model: Model) -> str:
    endpoints = set(model.supported_endpoints)
    family_or_id = (model.family or model.id).lower()

    if "/responses" in endpoints and (
        "/chat/completions" not in endpoints or family_or_id.startswith("gpt-5")
    ):
        return "responses"

    if "/chat/completions" in endpoints:
        return "chat_completions"

    if "/responses" in endpoints:
        return "responses"

    if "/v1/messages" in endpoints:
        return "anthropic_messages"

    if family_or_id.startswith("gpt-5"):
        return "responses"

    return "chat_completions"


def format_supported_endpoints(model: Model) -> str:
    if model.supported_endpoints:
        return ", ".join(model.supported_endpoints)
    return "default"


def format_context_window(model: Model) -> str:
    if model.max_context_window_tokens is None:
        return "?"
    return f"{model.max_context_window_tokens:,}"


def normalize_vendor_filter(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if not normalized:
        return None

    aliases = {
        "anthropic": "anthropic",
        "claude": "anthropic",
        "google": "google",
        "gemini": "google",
        "openai": "openai",
        "gpt": "openai",
    }
    return aliases.get(normalized, normalized)


def filter_models_by_vendor(
    models: list[Model],
    vendor_filter: str | None,
    *,
    provider_label: str = "LLM provider",
) -> list[Model]:
    normalized = normalize_vendor_filter(vendor_filter)
    if normalized is None:
        return models

    filtered = [
        model
        for model in models
        if model.vendor and model.vendor.strip().lower() == normalized
    ]
    if not filtered:
        raise LLMError(
            f"No {provider_label} models matched vendor `{vendor_filter}`. Available vendors: "
            + ", ".join(sorted({model.vendor for model in models if model.vendor}))
        )
    return filtered


def _model_id_matches(model_id: str, keywords: tuple[str, ...]) -> bool:
    normalized = model_id.lower()
    return any(keyword in normalized for keyword in keywords)


def _is_openai_reasoning_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return normalized.startswith(("o1", "o3", "o4")) or "/o" in normalized


def _uses_chat_template_thinking_controls(model_id: str) -> bool:
    return _model_id_matches(
        model_id,
        (
            "qwen",
            "deepseek",
            "granite",
            "glm",
            "hunyuan",
            "magistral",
            "mistral",
            "nemotron",
            "seed",
            "step",
        ),
    )


def disable_thinking_options(
    *,
    model_id: str,
    api_surface: str,
) -> dict[str, Any]:
    normalized = model_id.lower()

    if api_surface == "responses":
        if "codex" in normalized:
            return {"reasoning": {"effort": "none"}}
        if "gpt-5" in normalized:
            return {"reasoning": {"effort": "minimal"}}
        if "gpt-oss" in normalized or _is_openai_reasoning_model(model_id):
            return {"reasoning": {"effort": "low"}}
        if _uses_chat_template_thinking_controls(model_id):
            return {
                "reasoning_effort": "none",
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "thinking": False,
                },
            }
        return {}

    if "gemini" in normalized:
        return {"reasoning_effort": "none"}
    if "codex" in normalized:
        return {"reasoning_effort": "none"}
    if "gpt-5" in normalized:
        return {"reasoning_effort": "minimal"}
    if "gpt-oss" in normalized or _is_openai_reasoning_model(model_id):
        return {"reasoning_effort": "low"}
    if "claude" in normalized or "anthropic" in normalized:
        return {"thinking": {"type": "disabled"}}
    if _uses_chat_template_thinking_controls(model_id):
        return {
            "reasoning_effort": "none",
            "chat_template_kwargs": {
                "enable_thinking": False,
                "thinking": False,
            },
        }

    return {}


def extract_completion_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise LLMError("Chat completion returned an invalid payload.")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMError("Chat completion returned no choices.")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise LLMError("Chat completion returned an invalid choice.")

    message = choice.get("message")
    if not isinstance(message, dict):
        raise LLMError("Chat completion returned no message content.")

    content = message.get("content")
    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            return stripped

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
                continue
            nested = item.get("content")
            if isinstance(nested, str) and nested:
                text_parts.append(nested)
        joined = "\n".join(part.strip() for part in text_parts if part.strip()).strip()
        if joined:
            return joined

    finish_reason = choice.get("finish_reason")
    reasoning = message.get("reasoning")
    has_reasoning = isinstance(reasoning, str) and reasoning.strip()
    finish_reason_detail: str | None = None
    if finish_reason is not None:
        try:
            finish_reason_detail = json.dumps(finish_reason)
        except TypeError:
            finish_reason_detail = repr(finish_reason)
        finish_reason_detail = truncate_response_detail(finish_reason_detail)

    if finish_reason == "length":
        detail = (
            " The response contained reasoning text but no final assistant content."
            if has_reasoning
            else ""
        )
        raise LLMError(
            "Chat completion reached the completion token limit before returning "
            'message content (finish_reason="length").'
            f"{detail} Increase `max_tokens` or reduce the prompt."
        )

    if finish_reason_detail is not None:
        raise LLMError(
            "Chat completion message content was empty "
            f"(finish_reason={finish_reason_detail})."
        )

    if has_reasoning:
        raise LLMError(
            "Chat completion message content was empty. The response contained "
            "reasoning text but no final assistant content."
        )

    raise LLMError("Chat completion message content was empty.")


def chat_completion_request(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    *,
    model_id: str,
    prompt: str,
    disable_thinking: bool = False,
    max_tokens: int | None = None,
) -> str:
    request_body: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else 1024,
        "stream": False,
    }
    if disable_thinking:
        request_body.update(
            disable_thinking_options(
                model_id=model_id,
                api_surface="chat_completions",
            )
        )

    payload = request_json(
        client,
        "POST",
        url,
        headers=headers,
        json_body=request_body,
    )
    return extract_completion_text(payload)


def extract_response_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise LLMError("Responses API returned an invalid payload.")

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output")
    if not isinstance(output, list) or not output:
        raise LLMError("Responses API returned no output.")

    parts: list[str] = []
    refusals: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
                continue
            refusal = block.get("refusal")
            if isinstance(refusal, str) and refusal.strip():
                refusals.append(refusal.strip())

    joined = "\n".join(parts).strip()
    if joined:
        return joined

    joined_refusals = "\n".join(refusals).strip()
    if joined_refusals:
        return joined_refusals

    raise LLMError("Responses API output did not contain text.")


def response_output_contains_reasoning(payload: dict[str, Any]) -> bool:
    output = payload.get("output")
    if not isinstance(output, list):
        return False

    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            return True
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in {
                "reasoning_text",
                "reasoning_summary",
            }:
                return True

    return False


def format_incomplete_response_error(
    *,
    reason: str,
    final_response: dict[str, Any],
) -> str:
    message = f"Responses API response was incomplete: {reason}."
    if reason == "max_output_tokens":
        message += " Increase `--max-tokens` or reduce the prompt."
    if response_output_contains_reasoning(final_response):
        message += (
            " The response contained reasoning output before final text; if this "
            "provider cannot disable reasoning on `/responses`, use its "
            "`/chat/completions` endpoint instead."
        )
    return message


def responses_completion_request(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    *,
    model_id: str,
    prompt: str,
    disable_thinking: bool = False,
    max_tokens: int | None = None,
) -> str:
    request_body: dict[str, Any] = {
        "model": model_id,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            }
        ],
        "stream": True,
        "store": False,
    }
    if max_tokens is not None:
        request_body["max_output_tokens"] = max_tokens
    if disable_thinking:
        request_body.update(
            disable_thinking_options(
                model_id=model_id,
                api_surface="responses",
            )
        )

    for attempt in range(HTTP_RETRY_ATTEMPTS):
        text_parts: list[str] = []
        final_response: dict[str, Any] | None = None

        try:
            with client.stream(
                "POST",
                url,
                headers=headers,
                json=request_body,
            ) as response:
                if response.is_error:
                    detail = truncate_response_detail(
                        response.read().decode("utf-8", errors="replace")
                    )
                    error = LLMHttpError(
                        response.status_code, response.reason_phrase, detail
                    )
                    if (
                        should_retry_status_code(response.status_code)
                        and attempt < HTTP_RETRY_ATTEMPTS - 1
                    ):
                        time.sleep(
                            compute_retry_delay_seconds(
                                attempt, response.headers.get("retry-after")
                            )
                        )
                        continue
                    raise error

                content_type = response.headers.get("content-type", "")
                if "text/event-stream" not in content_type:
                    body = response.read()
                    try:
                        payload = json.loads(body)
                    except ValueError as exc:
                        detail = truncate_response_detail(
                            body.decode("utf-8", errors="replace")
                        )
                        raise LLMError(
                            f"Expected an SSE stream from {url}, got {content_type or 'unknown content type'}: {detail}"
                        ) from exc
                    return extract_response_text(payload)

                for event in iter_sse_events(response, url):
                    if not isinstance(event, dict):
                        continue

                    event_type = event.get("type")
                    if event_type == "response.output_text.delta":
                        delta = event.get("delta")
                        if isinstance(delta, str) and delta:
                            text_parts.append(delta)
                        continue

                    if event_type == "response.output_text.done" and not text_parts:
                        text = event.get("text")
                        if isinstance(text, str) and text.strip():
                            text_parts.append(text)
                        continue

                    if event_type == "error":
                        error = event.get("error")
                        if isinstance(error, dict):
                            message = error.get("message")
                            code = error.get("code")
                            if isinstance(message, str) and message.strip():
                                prefix = (
                                    f"{code}: "
                                    if isinstance(code, str) and code
                                    else ""
                                )
                                raise LLMError(
                                    f"Responses stream error: {prefix}{message.strip()}"
                                )
                        raise LLMError("Responses stream returned an error event.")

                    if event_type in {
                        "response.completed",
                        "response.failed",
                        "response.incomplete",
                    }:
                        response_payload = event.get("response")
                        if isinstance(response_payload, dict):
                            final_response = response_payload
        except httpx.TransportError as exc:
            if attempt < HTTP_RETRY_ATTEMPTS - 1:
                time.sleep(compute_retry_delay_seconds(attempt))
                continue
            raise LLMError(f"Request failed for {url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Request failed for {url}: {exc}") from exc

        text = "".join(text_parts).strip()
        if final_response is None:
            if text:
                return text
            raise LLMError("Responses stream ended without a terminal response event.")

        status = final_response.get("status")
        if status == "failed":
            error = final_response.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                code = error.get("code")
                if isinstance(message, str) and message.strip():
                    prefix = f"{code}: " if isinstance(code, str) and code else ""
                    raise LLMError(
                        f"Responses API request failed: {prefix}{message.strip()}"
                    )
            raise LLMError("Responses API request failed.")

        if status == "incomplete":
            details = final_response.get("incomplete_details")
            reason = "unknown"
            if isinstance(details, dict):
                raw_reason = details.get("reason")
                if isinstance(raw_reason, str) and raw_reason.strip():
                    reason = raw_reason.strip()
            if text:
                return f"{text}\n\n[Response incomplete: {reason}]"
            raise LLMError(
                format_incomplete_response_error(
                    reason=reason,
                    final_response=final_response,
                )
            )

        if text:
            return text

        return extract_response_text(final_response)

    raise AssertionError("Responses completion exhausted retries unexpectedly.")


def print_model_table(models: list[Model], *, title: str = "Available Models") -> None:
    table = Table(title=title)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Vendor", style="blue")
    table.add_column("Context", justify="right", style="bright_cyan")
    table.add_column("Route", style="yellow")
    table.add_column("Endpoints", style="magenta")
    for index, model in enumerate(models, start=1):
        table.add_row(
            str(index),
            model.id,
            model.vendor or "?",
            format_context_window(model),
            infer_api_surface(model),
            format_supported_endpoints(model),
        )
    console.print(table)


def render_model_selection_error(error: ModelSelectionError):
    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="red", no_wrap=True)
    summary.add_column()

    if error.requested_model is not None:
        summary.add_row("Invalid model", error.requested_model)
    elif error.configured_default_model is not None:
        summary.add_row("Config model", error.configured_default_model)
        if error.configured_default_model_path is not None:
            summary.add_row("Config file", str(error.configured_default_model_path))

    summary.add_row("Reason", f"{error.provider_label} did not return that model id.")
    summary.add_row("Available", str(len(error.models)))

    model_labels = [
        Text(model.id, style="green")
        for model in sorted(error.models, key=lambda model: model.id.lower())
    ]

    return Group(
        Panel.fit(summary, title="Model Selection Error", border_style="red"),
        Panel(
            Columns(model_labels, equal=False, expand=True),
            title="Available Model IDs",
            border_style="cyan",
        ),
    )


def print_model_selection_error(error: ModelSelectionError) -> None:
    console.print(render_model_selection_error(error))


def format_relative_duration(delta_seconds: int) -> str:
    remaining = abs(delta_seconds)
    units = (
        ("d", 86_400),
        ("h", 3_600),
        ("m", 60),
        ("s", 1),
    )
    parts: list[str] = []
    for suffix, width in units:
        if remaining < width and suffix != "s":
            continue
        value, remaining = divmod(remaining, width)
        if value == 0 and suffix != "s":
            continue
        parts.append(f"{value}{suffix}")
        if len(parts) == 2:
            break

    if not parts:
        parts.append("0s")

    text = " ".join(parts)
    if delta_seconds < 0:
        return f"{text} ago"
    return f"in {text}"


def format_unix_timestamp(timestamp: int) -> str:
    try:
        formatted = datetime.fromtimestamp(timestamp).astimezone()
    except (OSError, OverflowError, ValueError):
        return str(timestamp)

    delta_seconds = int(timestamp - time.time())
    return (
        f"{formatted.strftime('%Y-%m-%d %H:%M:%S %Z')} "
        f"({format_relative_duration(delta_seconds)})"
    )
