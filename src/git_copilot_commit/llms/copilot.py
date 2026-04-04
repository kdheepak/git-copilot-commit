from __future__ import annotations

import base64
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, TypeVar

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import core as llm
from ..settings import Settings

APP_NAME = "github-copilot-commit"
CLI_AUTH_COMMAND = "git-copilot-commit authenticate"
DEFAULT_GITHUB_DOMAIN = "github.com"
CREDENTIALS_FILENAME = "copilot-auth.json"
COPILOT_USER_AGENT = "GitHubCopilotChat/0.35.0"
EDITOR_VERSION = "vscode/1.107.0"
EDITOR_PLUGIN_VERSION = "copilot-chat/0.35.0"
COPILOT_INTEGRATION_ID = "vscode-chat"
CLIENT_ID = base64.b64decode("SXYxLmI1MDdhMDhjODdlY2ZlOTg=").decode()

INITIAL_POLL_INTERVAL_MULTIPLIER = 1.2
SLOW_DOWN_POLL_INTERVAL_MULTIPLIER = 1.4

console = Console()
T = TypeVar("T")

LLMError = llm.LLMError
LLMHttpError = llm.LLMHttpError
Model = llm.Model
HttpClientConfig = llm.HttpClientConfig
ModelSelectionError = llm.ModelSelectionError


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
            raise LLMError("Cached GitHub access token is missing or invalid.")
        if not isinstance(copilot_token, str) or not copilot_token:
            raise LLMError("Cached Copilot token is missing or invalid.")
        if not isinstance(copilot_expires_at, int):
            raise LLMError("Cached Copilot expiration timestamp is missing or invalid.")
        if enterprise_domain is not None and not isinstance(enterprise_domain, str):
            raise LLMError("Cached enterprise domain is invalid.")

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
        return get_copilot_base_url(self.copilot_token, self.enterprise_domain)


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
            raise LLMError("GitHub user endpoint did not return a login.")

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


def credentials_path() -> Path:
    return Settings().state_dir / CREDENTIALS_FILENAME


def config_path() -> Path:
    return Settings().config_file


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

    if not candidate or re.search(r"\s", candidate) or "." not in candidate:
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


def get_copilot_base_url(
    token: str | None = None, enterprise_domain: str | None = None
) -> str:
    if token:
        url_from_token = get_base_url_from_token(token)
        if url_from_token:
            return url_from_token

    if enterprise_domain:
        return f"https://copilot-api.{enterprise_domain}"

    return "https://api.individual.githubcopilot.com"


def read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LLMError(f"Unable to read cached credentials from {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise LLMError(f"Cached credentials in {path} are not a JSON object.")

    return raw


def load_credentials() -> CopilotCredentials | None:
    raw = read_json_object(credentials_path())
    if raw is None:
        return None
    return CopilotCredentials.from_dict(raw)


def save_credentials(credentials: CopilotCredentials) -> Path:
    import json

    path = credentials_path().expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(credentials.to_dict(), indent=2, sort_keys=True)
    path.write_text(f"{payload}\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def start_device_flow(
    client,
    domain: str,
) -> DeviceCodeResponse:
    urls = get_urls(domain)
    payload = llm.request_json(
        client,
        "POST",
        urls["device_code_url"],
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": COPILOT_USER_AGENT,
        },
        data={
            "client_id": CLIENT_ID,
            "scope": "read:user",
        },
    )

    if not isinstance(payload, dict):
        raise LLMError("GitHub device flow returned an invalid response.")

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
        raise LLMError("GitHub device flow response is missing expected fields.")

    return DeviceCodeResponse(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        interval=interval,
        expires_in=expires_in,
    )


def poll_for_github_access_token(
    client,
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

        payload = llm.request_json(
            client,
            "POST",
            urls["access_token_url"],
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": COPILOT_USER_AGENT,
            },
            data={
                "client_id": CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )

        if not isinstance(payload, dict):
            raise LLMError("GitHub token polling returned an invalid response.")

        access_token = payload.get("access_token")
        if isinstance(access_token, str) and access_token:
            return access_token

        error = payload.get("error")
        description = payload.get("error_description")
        if not isinstance(error, str):
            raise LLMError("GitHub token polling returned an unexpected response.")

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
        raise LLMError(f"Device flow failed with {error}{suffix}")

    if saw_slow_down:
        raise LLMError("Device flow timed out after one or more slow_down responses.")

    raise LLMError("Device flow timed out.")


def refresh_copilot_token(
    client,
    github_access_token: str,
    enterprise_domain: str | None = None,
) -> CopilotCredentials:
    domain = enterprise_domain or DEFAULT_GITHUB_DOMAIN
    urls = get_urls(domain)
    payload = llm.request_json(
        client,
        "GET",
        urls["copilot_token_url"],
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {github_access_token}",
            "User-Agent": COPILOT_USER_AGENT,
            "Editor-Version": EDITOR_VERSION,
            "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
            "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
        },
    )

    if not isinstance(payload, dict):
        raise LLMError("Copilot token exchange returned an invalid response.")

    token = payload.get("token")
    expires_at = payload.get("expires_at")
    if not isinstance(token, str) or not token:
        raise LLMError("Copilot token exchange did not return a token.")
    if not isinstance(expires_at, int):
        raise LLMError("Copilot token exchange did not return expires_at.")

    return CopilotCredentials(
        github_access_token=github_access_token,
        copilot_token=token,
        copilot_expires_at=expires_at,
        enterprise_domain=enterprise_domain,
    )


def fetch_github_viewer(
    client,
    github_access_token: str,
    domain: str,
) -> GitHubViewer:
    payload = llm.request_json(
        client,
        "GET",
        f"{get_github_api_base_url(domain)}/user",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_access_token}",
            "User-Agent": COPILOT_USER_AGENT,
        },
    )

    if not isinstance(payload, dict):
        raise LLMError("GitHub user endpoint returned an invalid response.")

    return GitHubViewer.from_payload(payload)


def ensure_fresh_credentials(client) -> CopilotCredentials:
    credentials = load_credentials()
    if credentials is None:
        raise LLMError(
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


def should_reauthenticate(exc: LLMError) -> bool:
    if isinstance(exc, LLMHttpError):
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
        "User-Agent": COPILOT_USER_AGENT,
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


def list_models(client, credentials: CopilotCredentials) -> list[Model]:
    payload = llm.request_json(
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
        raise LLMError("Models endpoint returned an unexpected payload.")

    if not isinstance(entries, list):
        raise LLMError("Models endpoint did not return a model list.")

    models: list[Model] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        models.append(Model.from_payload(entry))

    if not models:
        raise LLMError("Models endpoint returned no usable models.")

    return models


def complete_text_prompt(
    client, credentials: CopilotCredentials, *, model: Model, prompt: str
) -> str:
    api_surface = llm.infer_api_surface(model)
    if api_surface == "chat_completions":
        return llm.chat_completion_request(
            client,
            f"{credentials.base_url()}/chat/completions",
            copilot_request_headers(
                credentials.copilot_token, intent="conversation-edits"
            ),
            model_id=model.id,
            prompt=prompt,
        )
    if api_surface == "responses":
        return llm.responses_completion_request(
            client,
            f"{credentials.base_url()}/responses",
            copilot_request_headers(
                credentials.copilot_token,
                intent="conversation-edits",
                accept="text/event-stream",
            ),
            model_id=model.id,
            prompt=prompt,
        )

    raise LLMError(
        f"Model `{model.id}` uses `{api_surface}`, which this script does not support yet."
    )


def print_login_summary(
    domain: str,
    credentials: CopilotCredentials,
    *,
    default_model: str | None = None,
    configured_default_model_path: Path | None = None,
    github_viewer: GitHubViewer | None = None,
    models: list[Model] | None = None,
) -> None:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="white")

    table.add_row("GitHub host", domain)
    table.add_row("Config file", str(configured_default_model_path or config_path()))

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
        llm.format_unix_timestamp(credentials.copilot_expires_at),
    )

    if models is not None:
        table.add_row("Available models", str(len(models)))
        try:
            selected = llm.pick_model(
                models,
                default_model=default_model,
                provider_label="GitHub Copilot",
                configured_default_model_path=configured_default_model_path,
            )
        except LLMError as exc:
            table.add_row("Default model", f"Unavailable ({exc})")
        else:
            table.add_row(
                "Default model",
                f"{selected.id} ({llm.infer_api_surface(selected)})",
            )

    console.print(Panel.fit(table, title="Login Summary"))


def collect_login_summary(
    client,
    credentials: CopilotCredentials,
) -> tuple[str, GitHubViewer | None, list[Model] | None, list[str]]:
    domain = credentials.enterprise_domain or DEFAULT_GITHUB_DOMAIN
    github_viewer: GitHubViewer | None = None
    available_models: list[Model] | None = None
    warnings: list[str] = []

    try:
        github_viewer = fetch_github_viewer(
            client,
            credentials.github_access_token,
            domain,
        )
    except LLMError as exc:
        warnings.append(f"Could not fetch GitHub account details: {exc}")

    try:
        available_models = list_models(client, credentials)
    except LLMError as exc:
        warnings.append(f"Could not fetch Copilot model summary: {exc}")

    return domain, github_viewer, available_models, warnings


def _with_reauthentication(
    action: Callable[[Any], T],
    *,
    http_client_config: HttpClientConfig | None = None,
) -> T:
    try:
        with llm.make_http_client(http_client_config) as client:
            return action(client)
    except LLMError as exc:
        if not should_reauthenticate(exc):
            raise

    console.print(
        "[yellow]Cached GitHub Copilot credentials are missing or no longer valid. Starting authentication...[/yellow]"
    )
    login(force=True, http_client_config=http_client_config)

    with llm.make_http_client(http_client_config) as client:
        return action(client)


def show_login_summary(
    *,
    default_model: str | None = None,
    configured_default_model_path: Path | None = None,
    http_client_config: HttpClientConfig | None = None,
) -> None:
    def run(
        client,
    ) -> tuple[
        str, CopilotCredentials, GitHubViewer | None, list[Model] | None, list[str]
    ]:
        credentials = ensure_fresh_credentials(client)
        domain, github_viewer, available_models, warnings = collect_login_summary(
            client,
            credentials,
        )
        return domain, credentials, github_viewer, available_models, warnings

    domain, credentials, github_viewer, available_models, warnings = (
        _with_reauthentication(
            run,
            http_client_config=http_client_config,
        )
    )

    print_login_summary(
        domain,
        credentials,
        default_model=default_model,
        configured_default_model_path=configured_default_model_path,
        github_viewer=github_viewer,
        models=available_models,
    )
    for warning in warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def login(
    enterprise_domain: str | None = None,
    force: bool = False,
    *,
    http_client_config: HttpClientConfig | None = None,
) -> None:
    normalized_domain = normalize_domain(enterprise_domain)
    if enterprise_domain and not normalized_domain:
        raise LLMError("Invalid GitHub Enterprise hostname.")

    existing: CopilotCredentials | None = None
    try:
        existing = load_credentials()
    except LLMError:
        if not force:
            raise

    if existing and not force:
        raise LLMError(
            "Cached credentials already exist at "
            f"{credentials_path()}. "
            "Re-run with --force to replace them."
        )

    domain = normalized_domain or DEFAULT_GITHUB_DOMAIN
    with llm.make_http_client(http_client_config) as client:
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
        domain, github_viewer, available_models, warnings = collect_login_summary(
            client,
            credentials,
        )

    console.print(f"[green]Saved Copilot credentials to[/green] {path}")
    print_login_summary(
        domain,
        credentials,
        github_viewer=github_viewer,
        models=available_models,
    )
    for warning in warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def ensure_auth_ready(
    *,
    model: str | None = None,
    default_model: str | None = None,
    configured_default_model_path: Path | None = None,
    http_client_config: HttpClientConfig | None = None,
) -> Model:
    def validate(client) -> Model:
        credentials = ensure_fresh_credentials(client)
        models = list_models(client, credentials)
        return llm.pick_model(
            models,
            requested_model=model,
            default_model=default_model,
            provider_label="GitHub Copilot",
            configured_default_model_path=configured_default_model_path,
        )

    return _with_reauthentication(validate, http_client_config=http_client_config)


def get_available_models(
    *,
    vendor: str | None = None,
    http_client_config: HttpClientConfig | None = None,
) -> tuple[CopilotCredentials, list[Model]]:
    def load(client) -> tuple[CopilotCredentials, list[Model]]:
        credentials = ensure_fresh_credentials(client)
        models = list_models(client, credentials)
        return credentials, llm.filter_models_by_vendor(
            models,
            vendor,
            provider_label="GitHub Copilot",
        )

    return _with_reauthentication(load, http_client_config=http_client_config)


def ask(
    prompt: str,
    *,
    model: str | None = None,
    default_model: str | None = None,
    configured_default_model_path: Path | None = None,
    http_client_config: HttpClientConfig | None = None,
) -> str:
    def run(client) -> str:
        credentials = ensure_fresh_credentials(client)
        models = list_models(client, credentials)
        selected_model = llm.pick_model(
            models,
            requested_model=model,
            default_model=default_model,
            provider_label="GitHub Copilot",
            configured_default_model_path=configured_default_model_path,
        )
        return complete_text_prompt(
            client,
            credentials,
            model=selected_model,
            prompt=prompt,
        )

    return _with_reauthentication(run, http_client_config=http_client_config)
