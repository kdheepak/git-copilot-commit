from __future__ import annotations

import base64
import json
import os
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

APP_NAME = "github-copilot-commit"
CLI_AUTH_COMMAND = "git-copilot-commit authenticate"
DEFAULT_GITHUB_DOMAIN = "github.com"
USER_AGENT = "GitHubCopilotChat/0.35.0"
EDITOR_VERSION = "vscode/1.107.0"
EDITOR_PLUGIN_VERSION = "copilot-chat/0.35.0"
COPILOT_INTEGRATION_ID = "vscode-chat"
CLIENT_ID = base64.b64decode("SXYxLmI1MDdhMDhjODdlY2ZlOTg=").decode()

INITIAL_POLL_INTERVAL_MULTIPLIER = 1.2
SLOW_DOWN_POLL_INTERVAL_MULTIPLIER = 1.4
DEFAULT_MODEL_ID = "gpt-5.3-codex"
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

console = Console()
T = TypeVar("T")


class CopilotError(RuntimeError):
    pass


class CopilotHttpError(CopilotError):
    def __init__(
        self, status_code: int, reason_phrase: str, detail: str | None = None
    ) -> None:
        self.status_code = status_code
        self.reason_phrase = reason_phrase
        self.detail = detail
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{status_code} {reason_phrase}{suffix}")


@dataclass(slots=True)
class DeviceCodeResponse:
    device_code: str
    user_code: str
    verification_uri: str
    interval: int
    expires_in: int


@dataclass(slots=True)
class CopilotCredentials:
    github_access_token: str
    copilot_token: str
    copilot_expires_at: int
    enterprise_domain: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CopilotCredentials":
        github_access_token = data.get("github_access_token")
        copilot_token = data.get("copilot_token")
        copilot_expires_at = data.get("copilot_expires_at")
        enterprise_domain = data.get("enterprise_domain")

        if not isinstance(github_access_token, str) or not github_access_token:
            raise CopilotError("Cached GitHub access token is missing or invalid.")
        if not isinstance(copilot_token, str) or not copilot_token:
            raise CopilotError("Cached Copilot token is missing or invalid.")
        if not isinstance(copilot_expires_at, int):
            raise CopilotError(
                "Cached Copilot expiration timestamp is missing or invalid."
            )
        if enterprise_domain is not None and not isinstance(enterprise_domain, str):
            raise CopilotError("Cached enterprise domain is invalid.")

        return cls(
            github_access_token=github_access_token,
            copilot_token=copilot_token,
            copilot_expires_at=copilot_expires_at,
            enterprise_domain=enterprise_domain,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "github_access_token": self.github_access_token,
            "copilot_token": self.copilot_token,
            "copilot_expires_at": self.copilot_expires_at,
            "enterprise_domain": self.enterprise_domain,
        }

    def is_expired(self) -> bool:
        return time.time() >= self.copilot_expires_at - 300

    def base_url(self) -> str:
        return get_github_copilot_base_url(self.copilot_token, self.enterprise_domain)


@dataclass(slots=True)
class CopilotModel:
    id: str
    name: str
    vendor: str | None = None
    family: str | None = None
    supported_endpoints: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CopilotModel":
        model_id = payload.get("id")
        name = payload.get("name")
        vendor = payload.get("vendor")
        capabilities = payload.get("capabilities")
        supported_endpoints = payload.get("supported_endpoints")

        family: str | None = None
        if isinstance(capabilities, dict):
            raw_family = capabilities.get("family")
            if isinstance(raw_family, str) and raw_family:
                family = raw_family

        endpoints: list[str] = []
        if isinstance(supported_endpoints, list):
            for entry in supported_endpoints:
                if isinstance(entry, str) and entry:
                    endpoints.append(entry)

        if not isinstance(model_id, str) or not model_id:
            raise CopilotError("Models endpoint returned a model without an id.")

        return cls(
            id=model_id,
            name=name if isinstance(name, str) and name else model_id,
            vendor=vendor if isinstance(vendor, str) and vendor else None,
            family=family,
            supported_endpoints=tuple(endpoints),
        )


@dataclass(frozen=True, slots=True)
class HttpClientConfig:
    native_tls: bool = False
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


@dataclass(slots=True)
class GitHubViewer:
    login: str
    name: str | None = None
    html_url: str | None = None
    account_type: str | None = None
    plan_name: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GitHubViewer":
        login = payload.get("login")
        name = payload.get("name")
        html_url = payload.get("html_url")
        account_type = payload.get("type")
        plan = payload.get("plan")

        if not isinstance(login, str) or not login:
            raise CopilotError("GitHub user endpoint did not return a login.")

        plan_name: str | None = None
        if isinstance(plan, dict):
            raw_plan_name = plan.get("name")
            if isinstance(raw_plan_name, str) and raw_plan_name:
                plan_name = raw_plan_name

        return cls(
            login=login,
            name=name if isinstance(name, str) and name else None,
            html_url=html_url if isinstance(html_url, str) and html_url else None,
            account_type=(
                account_type if isinstance(account_type, str) and account_type else None
            ),
            plan_name=plan_name,
        )


def xdg_data_home() -> Path:
    value = os.environ.get("XDG_DATA_HOME")
    if value:
        return Path(value).expanduser()
    return Path.home() / ".local" / "share"


def credentials_path() -> Path:
    return xdg_data_home() / APP_NAME / "copilot-auth.json"


def normalize_domain(input_value: str | None) -> str | None:
    if input_value is None:
        return None

    trimmed = input_value.strip()
    if not trimmed:
        return None

    candidate = trimmed
    if "://" in candidate:
        candidate = candidate.split("://", 1)[1]
    candidate = candidate.split("/", 1)[0].strip().lower()

    if not candidate:
        return None
    if re.search(r"\s", candidate):
        return None
    if "." not in candidate:
        return None

    return candidate


def get_urls(domain: str) -> dict[str, str]:
    return {
        "device_code_url": f"https://{domain}/login/device/code",
        "access_token_url": f"https://{domain}/login/oauth/access_token",
        "copilot_token_url": f"https://api.{domain}/copilot_internal/v2/token",
    }


def get_github_api_base_url(domain: str) -> str:
    if domain == DEFAULT_GITHUB_DOMAIN:
        return "https://api.github.com"
    return f"https://api.{domain}"


def get_base_url_from_token(token: str) -> str | None:
    match = re.search(r"proxy-ep=([^;]+)", token)
    if not match:
        return None

    proxy_host = match.group(1)
    api_host = re.sub(r"^proxy\.", "api.", proxy_host)
    return f"https://{api_host}"


def get_github_copilot_base_url(
    token: str | None = None, enterprise_domain: str | None = None
) -> str:
    if token:
        url_from_token = get_base_url_from_token(token)
        if url_from_token:
            return url_from_token

    if enterprise_domain:
        return f"https://copilot-api.{enterprise_domain}"

    return "https://api.individual.githubcopilot.com"


_NATIVE_TLS_ENABLED = False


def _maybe_enable_native_tls(native_tls: bool) -> None:
    """
    Globally switch httpx/requests to use the OS certificate store via truststore.
    Safe to call multiple times; it no-ops after the first.
    """
    global _NATIVE_TLS_ENABLED
    if not native_tls or _NATIVE_TLS_ENABLED:
        return

    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception as _:
        return

    _NATIVE_TLS_ENABLED = True


def make_http_client(http_client_config: HttpClientConfig | None = None) -> httpx.Client:
    config = http_client_config or HttpClientConfig()
    _maybe_enable_native_tls(config.use_native_tls)

    return httpx.Client(
        verify=config.verify,
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, connect=10.0),
    )


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, str] | None = None,
    json_body: Any | None = None,
) -> Any:
    try:
        response = client.request(
            method,
            url,
            headers=headers,
            data=data,
            json=json_body,
        )
    except httpx.HTTPError as exc:
        raise CopilotError(f"Request failed for {url}: {exc}") from exc

    if response.is_error:
        detail = response.text.strip()
        if len(detail) > 400:
            detail = f"{detail[:397]}..."
        raise CopilotHttpError(response.status_code, response.reason_phrase, detail)

    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        try:
            return response.json()
        except ValueError as exc:
            detail = response.text.strip()
            if len(detail) > 400:
                detail = f"{detail[:397]}..."
            raise CopilotError(
                f"Expected JSON from {url}, got {content_type or 'unknown content type'}: {detail}"
            ) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise CopilotError(f"Invalid JSON response from {url}.") from exc


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
            raise CopilotError(
                f"Invalid SSE event payload from {url} ({label})."
            ) from exc
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


def read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CopilotError(
            f"Unable to read cached credentials from {path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise CopilotError(f"Cached credentials in {path} are not a JSON object.")

    return raw


def load_stored_credentials_from_path(path: Path) -> CopilotCredentials | None:
    raw = read_json_object(path)
    if raw is None:
        return None

    return CopilotCredentials.from_dict(raw)


def load_credentials() -> CopilotCredentials | None:
    return load_stored_credentials_from_path(credentials_path())


def save_credentials(credentials: CopilotCredentials) -> Path:
    path = credentials_path().expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(credentials.to_dict(), indent=2, sort_keys=True)
    path.write_text(f"{payload}\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def start_device_flow(client: httpx.Client, domain: str) -> DeviceCodeResponse:
    urls = get_urls(domain)
    payload = request_json(
        client,
        "POST",
        urls["device_code_url"],
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": USER_AGENT,
        },
        data={
            "client_id": CLIENT_ID,
            "scope": "read:user",
        },
    )

    if not isinstance(payload, dict):
        raise CopilotError("GitHub device flow returned an invalid response.")

    device_code = payload.get("device_code")
    user_code = payload.get("user_code")
    verification_uri = payload.get("verification_uri")
    interval = payload.get("interval")
    expires_in = payload.get("expires_in")

    if (
        not isinstance(device_code, str)
        or not isinstance(user_code, str)
        or not isinstance(verification_uri, str)
        or not isinstance(interval, int)
        or not isinstance(expires_in, int)
    ):
        raise CopilotError("GitHub device flow response is missing expected fields.")

    return DeviceCodeResponse(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        interval=interval,
        expires_in=expires_in,
    )


def poll_for_github_access_token(
    client: httpx.Client,
    domain: str,
    device_code: str,
    interval_seconds: int,
    expires_in: int,
) -> str:
    urls = get_urls(domain)
    deadline = time.time() + expires_in
    interval_ms = max(1000, interval_seconds * 1000)
    interval_multiplier = INITIAL_POLL_INTERVAL_MULTIPLIER
    saw_slow_down = False

    while time.time() < deadline:
        remaining_ms = max(0, int((deadline - time.time()) * 1000))
        wait_ms = min(int(interval_ms * interval_multiplier), remaining_ms)
        if wait_ms > 0:
            time.sleep(wait_ms / 1000)

        payload = request_json(
            client,
            "POST",
            urls["access_token_url"],
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": USER_AGENT,
            },
            data={
                "client_id": CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )

        if not isinstance(payload, dict):
            raise CopilotError("GitHub token polling returned an invalid response.")

        access_token = payload.get("access_token")
        if isinstance(access_token, str) and access_token:
            return access_token

        error = payload.get("error")
        description = payload.get("error_description")
        if not isinstance(error, str):
            raise CopilotError("GitHub token polling returned an unexpected response.")

        if error == "authorization_pending":
            continue

        if error == "slow_down":
            saw_slow_down = True
            next_interval = payload.get("interval")
            if isinstance(next_interval, int) and next_interval > 0:
                interval_ms = next_interval * 1000
            else:
                interval_ms = max(1000, interval_ms + 5000)
            interval_multiplier = SLOW_DOWN_POLL_INTERVAL_MULTIPLIER
            continue

        suffix = (
            f": {description}" if isinstance(description, str) and description else ""
        )
        raise CopilotError(f"Device flow failed with {error}{suffix}")

    if saw_slow_down:
        raise CopilotError(
            "Device flow timed out after one or more slow_down responses."
        )

    raise CopilotError("Device flow timed out.")


def refresh_copilot_token(
    client: httpx.Client,
    github_access_token: str,
    enterprise_domain: str | None = None,
) -> CopilotCredentials:
    domain = enterprise_domain or DEFAULT_GITHUB_DOMAIN
    urls = get_urls(domain)
    payload = request_json(
        client,
        "GET",
        urls["copilot_token_url"],
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {github_access_token}",
            "User-Agent": USER_AGENT,
            "Editor-Version": EDITOR_VERSION,
            "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
            "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
        },
    )

    if not isinstance(payload, dict):
        raise CopilotError("Copilot token exchange returned an invalid response.")

    token = payload.get("token")
    expires_at = payload.get("expires_at")
    if not isinstance(token, str) or not token:
        raise CopilotError("Copilot token exchange did not return a token.")
    if not isinstance(expires_at, int):
        raise CopilotError("Copilot token exchange did not return expires_at.")

    return CopilotCredentials(
        github_access_token=github_access_token,
        copilot_token=token,
        copilot_expires_at=expires_at,
        enterprise_domain=enterprise_domain,
    )


def fetch_github_viewer(
    client: httpx.Client,
    github_access_token: str,
    domain: str,
) -> GitHubViewer:
    payload = request_json(
        client,
        "GET",
        f"{get_github_api_base_url(domain)}/user",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_access_token}",
            "User-Agent": USER_AGENT,
        },
    )

    if not isinstance(payload, dict):
        raise CopilotError("GitHub user endpoint returned an invalid response.")

    return GitHubViewer.from_payload(payload)


def ensure_fresh_credentials(client: httpx.Client) -> CopilotCredentials:
    credentials = load_credentials()
    if credentials is None:
        raise CopilotError(
            f"No cached Copilot credentials found. Run `{CLI_AUTH_COMMAND}` first."
        )

    if not credentials.is_expired():
        return credentials

    refreshed = refresh_copilot_token(
        client,
        credentials.github_access_token,
        credentials.enterprise_domain,
    )
    save_credentials(refreshed)
    return refreshed


def should_reauthenticate(exc: CopilotError) -> bool:
    if isinstance(exc, CopilotHttpError):
        return exc.status_code == 401

    message = str(exc)
    retryable_prefixes = (
        "No cached Copilot credentials found.",
        "Cached GitHub access token is missing or invalid.",
        "Cached Copilot token is missing or invalid.",
        "Cached Copilot expiration timestamp is missing or invalid.",
        "Cached enterprise domain is invalid.",
        "Unable to read cached credentials from ",
        "Cached credentials in ",
    )
    return any(message.startswith(prefix) for prefix in retryable_prefixes)


def copilot_request_headers(
    access_token: str,
    *,
    intent: str = "conversation-panel",
    accept: str = "application/json",
) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": accept,
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
        "Editor-Version": EDITOR_VERSION,
        "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
        "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
        "OpenAI-Organization": "github-copilot",
        "OpenAI-Intent": intent,
        "X-Request-Id": str(uuid.uuid4()),
        "Vscode-Sessionid": str(uuid.uuid4()),
        "Vscode-Machineid": secrets.token_hex(32),
        "X-Initiator": "user",
    }


def list_models(
    client: httpx.Client, credentials: CopilotCredentials
) -> list[CopilotModel]:
    payload = request_json(
        client,
        "GET",
        f"{credentials.base_url()}/models",
        headers=copilot_request_headers(
            credentials.copilot_token, intent="conversation-panel"
        ),
    )

    if isinstance(payload, dict):
        entries = payload.get("data")
    elif isinstance(payload, list):
        entries = payload
    else:
        raise CopilotError("Models endpoint returned an unexpected payload.")

    if not isinstance(entries, list):
        raise CopilotError("Models endpoint did not return a model list.")

    models: list[CopilotModel] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        models.append(CopilotModel.from_payload(entry))

    if not models:
        raise CopilotError("Models endpoint returned no usable models.")

    return models


def pick_model(
    models: list[CopilotModel], requested_model: str | None = None
) -> CopilotModel:
    by_id = {model.id: model for model in models}
    if requested_model is not None:
        if requested_model not in by_id:
            raise CopilotError(
                f"Model `{requested_model}` was not returned by Copilot. Available models: {', '.join(model.id for model in models)}"
            )
        return by_id[requested_model]

    for preferred_model in DEFAULT_MODEL_PREFERENCES:
        if preferred_model in by_id:
            return by_id[preferred_model]

    return models[0]


def infer_api_surface(model: CopilotModel) -> str:
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


def format_supported_endpoints(model: CopilotModel) -> str:
    if model.supported_endpoints:
        return ", ".join(model.supported_endpoints)
    return "default"


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
    models: list[CopilotModel], vendor_filter: str | None
) -> list[CopilotModel]:
    normalized = normalize_vendor_filter(vendor_filter)
    if normalized is None:
        return models

    filtered = [
        model
        for model in models
        if model.vendor and model.vendor.strip().lower() == normalized
    ]
    if not filtered:
        raise CopilotError(
            f"No Copilot models matched vendor `{vendor_filter}`. Available vendors: "
            + ", ".join(sorted({model.vendor for model in models if model.vendor}))
        )
    return filtered


def extract_completion_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise CopilotError("Chat completion returned an invalid payload.")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise CopilotError("Chat completion returned no choices.")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise CopilotError("Chat completion returned an invalid choice.")

    message = choice.get("message")
    if not isinstance(message, dict):
        raise CopilotError("Chat completion returned no message content.")

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

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

    raise CopilotError("Chat completion message content was empty.")


def chat_completion(
    client: httpx.Client,
    credentials: CopilotCredentials,
    *,
    model: CopilotModel,
    prompt: str,
) -> str:
    payload = request_json(
        client,
        "POST",
        f"{credentials.base_url()}/chat/completions",
        headers=copilot_request_headers(
            credentials.copilot_token, intent="conversation-edits"
        ),
        json_body={
            "model": model.id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0,
            "max_tokens": 1024,
            "stream": False,
        },
    )
    return extract_completion_text(payload)


def extract_response_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise CopilotError("Responses API returned an invalid payload.")

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output")
    if not isinstance(output, list) or not output:
        raise CopilotError("Responses API returned no output.")

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

    raise CopilotError("Responses API output did not contain text.")


def responses_completion(
    client: httpx.Client,
    credentials: CopilotCredentials,
    *,
    model: CopilotModel,
    prompt: str,
) -> str:
    url = f"{credentials.base_url()}/responses"
    request_body = {
        "model": model.id,
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

    text_parts: list[str] = []
    final_response: dict[str, Any] | None = None

    try:
        with client.stream(
            "POST",
            url,
            headers=copilot_request_headers(
                credentials.copilot_token,
                intent="conversation-edits",
                accept="text/event-stream",
            ),
            json=request_body,
        ) as response:
            if response.is_error:
                detail = response.read().decode("utf-8", errors="replace").strip()
                if len(detail) > 400:
                    detail = f"{detail[:397]}..."
                raise CopilotHttpError(
                    response.status_code, response.reason_phrase, detail
                )

            content_type = response.headers.get("content-type", "")
            if "text/event-stream" not in content_type:
                body = response.read()
                try:
                    payload = json.loads(body)
                except ValueError as exc:
                    detail = body.decode("utf-8", errors="replace").strip()
                    if len(detail) > 400:
                        detail = f"{detail[:397]}..."
                    raise CopilotError(
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
                                f"{code}: " if isinstance(code, str) and code else ""
                            )
                            raise CopilotError(
                                f"Responses stream error: {prefix}{message.strip()}"
                            )
                    raise CopilotError("Responses stream returned an error event.")

                if event_type in {
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                }:
                    response_payload = event.get("response")
                    if isinstance(response_payload, dict):
                        final_response = response_payload
    except httpx.HTTPError as exc:
        raise CopilotError(f"Request failed for {url}: {exc}") from exc

    text = "".join(text_parts).strip()
    if final_response is None:
        if text:
            return text
        raise CopilotError("Responses stream ended without a terminal response event.")

    status = final_response.get("status")
    if status == "failed":
        error = final_response.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            code = error.get("code")
            if isinstance(message, str) and message.strip():
                prefix = f"{code}: " if isinstance(code, str) and code else ""
                raise CopilotError(
                    f"Responses API request failed: {prefix}{message.strip()}"
                )
        raise CopilotError("Responses API request failed.")

    if status == "incomplete":
        details = final_response.get("incomplete_details")
        reason = "unknown"
        if isinstance(details, dict):
            raw_reason = details.get("reason")
            if isinstance(raw_reason, str) and raw_reason.strip():
                reason = raw_reason.strip()
        if text:
            return f"{text}\n\n[Response incomplete: {reason}]"
        raise CopilotError(f"Responses API response was incomplete: {reason}.")

    if text:
        return text

    return extract_response_text(final_response)


def complete_text_prompt(
    client: httpx.Client,
    credentials: CopilotCredentials,
    *,
    model: CopilotModel,
    prompt: str,
) -> str:
    api_surface = infer_api_surface(model)
    if api_surface == "chat_completions":
        return chat_completion(client, credentials, model=model, prompt=prompt)
    if api_surface == "responses":
        return responses_completion(client, credentials, model=model, prompt=prompt)

    raise CopilotError(
        f"Model `{model.id}` uses `{api_surface}`, which this script does not support yet."
    )


def print_model_table(models: list[CopilotModel]) -> None:
    table = Table(title="Available Copilot Models")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Vendor", style="blue")
    table.add_column("Route", style="yellow")
    table.add_column("Endpoints", style="magenta")
    for index, model in enumerate(models, start=1):
        table.add_row(
            str(index),
            model.id,
            model.vendor or "?",
            infer_api_surface(model),
            format_supported_endpoints(model),
        )
    console.print(table)


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


def print_login_summary(
    domain: str,
    credentials: CopilotCredentials,
    github_viewer: GitHubViewer | None = None,
    models: list[CopilotModel] | None = None,
) -> None:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="white")

    table.add_row("GitHub host", domain)

    if github_viewer is not None:
        identity = github_viewer.login
        if github_viewer.name and github_viewer.name != github_viewer.login:
            identity = f"{identity} ({github_viewer.name})"
        table.add_row("GitHub user", identity)
        if github_viewer.account_type:
            table.add_row("Account type", github_viewer.account_type)
        if github_viewer.plan_name:
            table.add_row("GitHub plan", github_viewer.plan_name)
        if github_viewer.html_url:
            table.add_row("GitHub profile", github_viewer.html_url)

    table.add_row("Copilot base URL", credentials.base_url())
    table.add_row(
        "Copilot token expires",
        format_unix_timestamp(credentials.copilot_expires_at),
    )

    if models is not None:
        default_model = pick_model(models)
        table.add_row("Available models", str(len(models)))
        table.add_row(
            "Default model",
            f"{default_model.id} ({infer_api_surface(default_model)})",
        )

    console.print(Panel.fit(table, title="Login Summary"))


def login(
    enterprise_domain: str | None = None,
    force: bool = False,
    *,
    http_client_config: HttpClientConfig | None = None,
) -> None:
    """Authenticate with GitHub and cache Copilot credentials locally."""
    normalized_domain = normalize_domain(enterprise_domain)
    if enterprise_domain and not normalized_domain:
        raise CopilotError("Invalid GitHub Enterprise hostname.")

    existing: CopilotCredentials | None = None
    try:
        existing = load_credentials()
    except CopilotError:
        if not force:
            raise

    if existing and not force:
        raise CopilotError(
            f"Cached credentials already exist at {credentials_path()}. Re-run with --force to replace them."
        )

    domain = normalized_domain or DEFAULT_GITHUB_DOMAIN
    with make_http_client(http_client_config) as client:
        device = start_device_flow(client, domain)

        console.print(
            Panel.fit(
                f"Open [bold]{device.verification_uri}[/bold]\n"
                f"Enter code [bold cyan]{device.user_code}[/bold cyan]",
                title="GitHub Copilot Device Login",
            )
        )
        console.print("Waiting for GitHub authorization...")

        github_access_token = poll_for_github_access_token(
            client,
            domain,
            device.device_code,
            device.interval,
            device.expires_in,
        )
        credentials = refresh_copilot_token(
            client, github_access_token, normalized_domain
        )
        path = save_credentials(credentials)
        github_viewer: GitHubViewer | None = None
        available_models: list[CopilotModel] | None = None
        warnings: list[str] = []

        try:
            github_viewer = fetch_github_viewer(client, github_access_token, domain)
        except CopilotError as exc:
            warnings.append(f"Could not fetch GitHub account details: {exc}")

        try:
            available_models = list_models(client, credentials)
        except CopilotError as exc:
            warnings.append(f"Could not fetch Copilot model summary: {exc}")

    console.print(f"[green]Saved Copilot credentials to[/green] {path}")
    print_login_summary(
        domain,
        credentials,
        github_viewer=github_viewer,
        models=available_models,
    )
    for warning in warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def _ask_once(client: httpx.Client, prompt: str, model: str | None = None) -> str:
    credentials = ensure_fresh_credentials(client)
    all_models = list_models(client, credentials)

    selected_model = pick_model(all_models, model)
    return complete_text_prompt(
        client,
        credentials,
        model=selected_model,
        prompt=prompt,
    )


def _with_reauthentication(
    action: Callable[[httpx.Client], T],
    *,
    http_client_config: HttpClientConfig | None = None,
) -> T:
    try:
        with make_http_client(http_client_config) as client:
            return action(client)
    except CopilotError as exc:
        if not should_reauthenticate(exc):
            raise

    console.print(
        "[yellow]Cached GitHub Copilot credentials are missing or no longer valid. Starting authentication...[/yellow]"
    )
    login(force=True, http_client_config=http_client_config)

    with make_http_client(http_client_config) as client:
        return action(client)


def ensure_auth_ready(
    model: str | None = None,
    *,
    http_client_config: HttpClientConfig | None = None,
) -> None:
    def validate(client: httpx.Client) -> None:
        credentials = ensure_fresh_credentials(client)
        all_models = list_models(client, credentials)
        pick_model(all_models, model)

    _with_reauthentication(validate, http_client_config=http_client_config)


def ask(
    prompt: str,
    model: str | None = None,
    *,
    http_client_config: HttpClientConfig | None = None,
) -> str:
    """Send a prompt to GitHub Copilot and print the reply."""
    return _with_reauthentication(
        lambda client: _ask_once(client, prompt, model),
        http_client_config=http_client_config,
    )
