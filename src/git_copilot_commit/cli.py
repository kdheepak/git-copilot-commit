"""
git-copilot-commit - AI-powered Git commit assistant
"""

from dataclasses import dataclass
from pathlib import Path
import os
import re
import sys
from typing import Annotated, Sequence

import rich
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from typer.main import get_command

from .git import GitRepository, GitError, GitStatus, NotAGitRepositoryError
from .split_commits import (
    PatchUnit,
    SplitCommitPlan,
    SplitPlanningError,
    build_split_plan_prompt,
    build_status_for_patch_units,
    extract_patch_units,
    group_patch_units,
    parse_split_plan_response,
)
from .settings import Settings
from .version import __version__
from .llms import copilot
from .llms import core as llm
from .llms import providers

console = Console()
app = typer.Typer(help=__doc__, add_completion=False)

COMMIT_MESSAGE_PROMPT_FILENAME = "commit-message-generator-prompt.md"
SPLIT_COMMIT_PLANNER_PROMPT_FILENAME = "split-commit-planner-prompt.md"
SPLIT_DIFF_ARGS = [
    "--binary",
    "--full-index",
    "--find-renames",
    "--no-color",
    "--no-ext-diff",
    "--src-prefix=a/",
    "--dst-prefix=b/",
    "--unified=3",
]

CA_BUNDLE_HELP = "Path to a custom CA bundle (PEM)"
NATIVE_TLS_HELP = (
    "Use the OS's native certificate store via 'truststore' for httpx instead of "
    "the Python bundle. Ignored if --ca-bundle or --insecure is used."
)

CaBundleOption = Annotated[
    str | None,
    typer.Option("--ca-bundle", metavar="PATH", help=CA_BUNDLE_HELP),
]
InsecureOption = Annotated[
    bool,
    typer.Option("--insecure", help="Disable SSL certificate verification."),
]
NativeTlsOption = Annotated[
    bool,
    typer.Option("--native-tls/--no-native-tls", help=NATIVE_TLS_HELP),
]
ProviderOption = Annotated[
    str | None,
    typer.Option(
        "--provider",
        help="LLM provider to use: copilot or openai.",
    ),
]
BaseUrlOption = Annotated[
    str | None,
    typer.Option(
        "--base-url",
        metavar="URL",
        help=(
            "Base URL for an OpenAI-compatible provider, for example "
            "http://127.0.0.1:11434/v1."
        ),
    ),
]
ApiKeyOption = Annotated[
    str | None,
    typer.Option(
        "--api-key",
        help="API key for an OpenAI-compatible provider. Omit when the server does not require one.",
    ),
]


SplitOption = Annotated[
    bool,
    typer.Option(
        "--split",
        help=(
            "Split staged hunks into multiple commits automatically. Pass "
            "`--split=N` to express a preference for N commits."
        ),
    ),
]
SplitCountOption = Annotated[
    int | None,
    typer.Option(
        "--split-count",
        hidden=True,
        min=1,
    ),
]


@dataclass(frozen=True, slots=True)
class PreparedSplitCommit:
    """A split commit with its generated message and assigned patch units."""

    message: str
    patch_units: tuple[PatchUnit, ...]


@dataclass(frozen=True, slots=True)
class SplitCommitExecutionState:
    """Original HEAD state used to roll back partial split-commit execution."""

    original_head_sha: str | None
    original_head_ref: str | None


CORE_CHANGE_COMMIT_TYPES = frozenset({"feat", "fix", "perf", "refactor", "revert"})
FOLLOW_UP_COMMIT_TYPE_PRIORITY = {
    "test": 2,
    "docs": 3,
    "style": 4,
    "chore": 4,
}
CONVENTIONAL_COMMIT_TYPE_PATTERN = re.compile(
    r"^\s*([a-z]+)(?:\([^)\r\n]*\))?(?:!)?:",
    re.IGNORECASE,
)


def preprocess_cli_args(args: Sequence[str]) -> list[str]:
    """Normalize CLI arguments before Click parses them."""
    processed_args: list[str] = []
    in_commit_command = False
    index = 0

    while index < len(args):
        arg = args[index]

        if not in_commit_command and not arg.startswith("-"):
            processed_args.append(arg)
            if arg == "commit":
                in_commit_command = True
            index += 1
            continue

        if in_commit_command and arg.startswith("--split="):
            split_value = arg.split("=", 1)[1].strip().lower()
            if split_value == "auto":
                processed_args.append("--split")
                index += 1
                continue
            if split_value.isdigit():
                processed_args.extend(["--split-count", split_value])
                index += 1
                continue

            processed_args.append(arg)
            index += 1
            continue

        if in_commit_command and arg == "--split" and index + 1 < len(args):
            split_value = args[index + 1].strip().lower()
            if split_value == "auto":
                processed_args.append("--split")
                index += 2
                continue
            if split_value.isdigit():
                processed_args.extend(["--split-count", split_value])
                index += 2
                continue

        processed_args.append(arg)
        index += 1

    return processed_args


def extract_conventional_commit_type(message: str) -> str | None:
    """Extract the Conventional Commit type from a generated title line."""
    match = CONVENTIONAL_COMMIT_TYPE_PATTERN.match(message.strip())
    if match is None:
        return None

    return match.group(1).lower()


def order_prepared_split_commits(
    prepared_commits: Sequence[PreparedSplitCommit],
) -> list[PreparedSplitCommit]:
    """Order planned commits in a developer-friendly execution sequence."""

    def sort_key(item: tuple[int, PreparedSplitCommit]) -> tuple[int, int]:
        index, prepared_commit = item
        commit_type = extract_conventional_commit_type(prepared_commit.message)

        if commit_type in CORE_CHANGE_COMMIT_TYPES:
            priority = 0
        elif commit_type is None:
            priority = 1
        else:
            priority = FOLLOW_UP_COMMIT_TYPE_PRIORITY.get(commit_type, 1)

        return priority, index

    ordered_items = sorted(enumerate(prepared_commits), key=sort_key)
    return [prepared_commit for _, prepared_commit in ordered_items]


def run(args: Sequence[str] | None = None) -> None:
    """Run the CLI entrypoint with argument normalization."""
    raw_args = list(args) if args is not None else sys.argv[1:]
    command = get_command(app)
    command.main(
        args=preprocess_cli_args(raw_args),
        prog_name=Path(sys.argv[0]).name,
    )


def version_callback(value: bool):
    if value:
        rich.print(f"git-copilot-commit [bold yellow]{__version__}[/]")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    _: bool = typer.Option(
        False, "--version", callback=version_callback, help="Show version and exit"
    ),
):
    """
    Automatically commit changes in the current git repository.
    """
    if ctx.invoked_subcommand is None:
        # Show help when no command is provided
        console.print(ctx.get_help())
        raise typer.Exit()
    else:
        # Don't show version for print command to avoid interfering with pipes
        if ctx.invoked_subcommand != "echo":
            console.print(
                f"[bold]{(__package__ or 'git_copilot_commit').replace('_', '-')}[/] - [bold green]v{__version__}[/]\n"
            )


def get_prompt_locations(filename: str):
    """Get potential prompt file locations in order of preference."""
    import importlib.resources

    return [
        Path(Settings().data_dir) / "prompts" / filename,  # User customizable
        importlib.resources.files("git_copilot_commit")
        / "prompts"
        / filename,  # Packaged version
    ]


def resolve_prompt_file() -> Path | None:
    settings = Settings()
    try:
        configured_prompt_file = settings.default_prompt_file
    except ValueError:
        console.print(
            f"[red]Configured default prompt file in {settings.config_file} is invalid.[/red]"
        )
        raise typer.Exit(1)

    if configured_prompt_file is None:
        return None

    return Path(configured_prompt_file).expanduser()


def load_system_prompt() -> str:
    """Load the system prompt from the markdown file."""
    resolved_prompt_file = resolve_prompt_file()
    if resolved_prompt_file is not None:
        try:
            return resolved_prompt_file.read_text(encoding="utf-8")
        except OSError as exc:
            console.print(
                f"[red]Error reading prompt file {resolved_prompt_file}: {exc}[/red]"
            )
            raise typer.Exit(1)

    return load_named_prompt(COMMIT_MESSAGE_PROMPT_FILENAME)


def load_named_prompt(filename: str) -> str:
    """Load a packaged prompt by filename, optionally overridden via user data."""
    for path in get_prompt_locations(filename):
        try:
            return path.read_text(encoding="utf-8")
        except (FileNotFoundError, AttributeError):
            continue

    console.print(f"[red]Error: Prompt file {filename} not found in any location[/red]")
    raise typer.Exit(1)


def build_http_client_config(
    *,
    ca_bundle: str | None,
    insecure: bool,
    native_tls: bool,
) -> llm.HttpClientConfig:
    if ca_bundle is not None:
        ca_bundle = os.path.expanduser(ca_bundle)
    return llm.HttpClientConfig(
        native_tls=native_tls,
        insecure=insecure,
        ca_bundle=ca_bundle,
    )


def print_llm_error(message: str, exc: llm.LLMError) -> None:
    """Render LLM errors, with rich formatting for model selection issues."""
    if isinstance(exc, llm.ModelSelectionError):
        console.print(f"[red]{message}[/red]")
        llm.print_model_selection_error(exc)
        return

    console.print(f"[red]{message}: {exc}[/red]")


def display_selected_model(model: llm.Model) -> None:
    """Show the resolved model for the current command."""
    details = [llm.infer_api_surface(model)]
    if model.vendor:
        details.insert(0, model.vendor)
    console.print(f"[green]Using model:[/green] {model.id} ({', '.join(details)})")


def build_commit_message_prompt(
    status: GitStatus,
    context: str = "",
    *,
    include_diff: bool = True,
) -> str:
    """Build the prompt used to generate a commit message."""
    if not status.has_staged_changes:
        console.print("[red]No staged changes to commit.[/red]")
        raise typer.Exit()

    prompt_parts = [
        "`git status`:\n",
        f"```\n{status.get_porcelain_output()}\n```",
    ]

    if include_diff:
        prompt_parts.extend(
            [
                "\n\n`git diff --staged`:\n",
                f"```\n{status.staged_diff}\n```",
            ]
        )

    if context.strip():
        prompt_parts.insert(0, f"User-provided context:\n\n{context.strip()}\n\n")

    return "\n".join(prompt_parts)


def normalize_model_name(model: str | None) -> str | None:
    """Normalize model names accepted by the CLI to provider model ids."""
    if model is not None:
        for prefix in (
            "copilot/",
            "openai-compatible/",
        ):
            if model.startswith(prefix):
                return model.replace(prefix, "", 1)
    return model


def ask_llm_with_system_prompt(
    system_prompt: str,
    prompt: str,
    model: str | None = None,
    provider_config: providers.ProviderConfig | None = None,
    http_client_config: llm.HttpClientConfig | None = None,
) -> str:
    """Send a prepared prompt to the selected LLM provider."""
    return providers.ask(
        f"""
# System Prompt

{system_prompt}

# Prompt

{prompt}
            """,
        provider_config=provider_config,
        model=normalize_model_name(model),
        http_client_config=http_client_config,
    )


def generate_commit_message_for_prompt(
    prompt: str,
    model: str | None = None,
    provider_config: providers.ProviderConfig | None = None,
    http_client_config: llm.HttpClientConfig | None = None,
) -> str:
    """Generate a conventional commit message from a prepared prompt."""
    return ask_llm_with_system_prompt(
        load_system_prompt(),
        prompt,
        model=model,
        provider_config=provider_config,
        http_client_config=http_client_config,
    )


def should_retry_with_compact_prompt(exc: llm.LLMError) -> bool:
    message_parts = [str(exc)]
    if isinstance(exc, llm.LLMHttpError) and exc.detail:
        message_parts.append(exc.detail)

    haystack = " ".join(part.strip() for part in message_parts if part).lower()
    indicators = (
        "maximum context length",
        "context_length_exceeded",
        "context window",
        "prompt is too long",
        "input is too long",
        "request is too large",
        "too many tokens",
        "token limit",
        "max_prompt_tokens",
        "max prompt tokens",
        "input tokens",
        "prompt tokens",
        "prompt token count",
    )
    return any(indicator in haystack for indicator in indicators)


def generate_commit_message_for_status(
    status: GitStatus,
    model: str | None = None,
    context: str = "",
    provider_config: providers.ProviderConfig | None = None,
    http_client_config: llm.HttpClientConfig | None = None,
) -> str:
    """Generate a commit message for a staged status snapshot."""
    full_prompt = build_commit_message_prompt(status, context=context)
    try:
        return generate_commit_message_for_prompt(
            full_prompt,
            model=model,
            provider_config=provider_config,
            http_client_config=http_client_config,
        )
    except llm.LLMError as exc:
        if not should_retry_with_compact_prompt(exc):
            raise

    console.print(
        "[yellow]Staged diff exceeded the model context window; retrying with [bold]`git status`[/] only.[/yellow]"
    )
    fallback_prompt = build_commit_message_prompt(
        status,
        context=context,
        include_diff=False,
    )
    return generate_commit_message_for_prompt(
        fallback_prompt,
        model=model,
        provider_config=provider_config,
        http_client_config=http_client_config,
    )


def commit_with_retry_no_verify(
    repo: GitRepository,
    message: str,
    use_editor: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    """Run commit and offer one retry with -n on failure."""
    try:
        return repo.commit(message, use_editor=use_editor, env=env)
    except GitError as e:
        console.print(f"[red]Commit failed: {e}[/red]")
        if not Confirm.ask(
            "Retry commit with [bold]`-n`[/] (skip hooks) using the same commit message?",
            default=True,
        ):
            raise typer.Exit(1)

    try:
        return repo.commit(message, use_editor=use_editor, no_verify=True, env=env)
    except GitError as retry_error:
        console.print(f"[red]Commit with -n failed: {retry_error}[/red]")
        raise typer.Exit(1)


def ensure_copilot_authentication(
    http_client_config: llm.HttpClientConfig,
) -> None:
    """Authenticate if no cached Copilot credentials are available."""
    try:
        existing_credentials = copilot.load_credentials()
    except copilot.LLMError:
        existing_credentials = None

    if existing_credentials is not None:
        return

    try:
        copilot.login(
            force=True,
            http_client_config=http_client_config,
        )
    except copilot.LLMError as exc:
        print_llm_error("Authentication failed", exc)
        raise typer.Exit(1)


def stage_changes_for_commit(
    repo: GitRepository, status: GitStatus, all_files: bool
) -> GitStatus:
    """Stage changes according to the command options and return refreshed status."""
    if all_files:
        repo.stage_files()
        console.print("[green]Staged all files.[/green]")
        return repo.get_status()

    if status.has_unstaged_changes or status.has_untracked_files:
        git_status_output = repo._run_git_command(["status"])
        console.print(git_status_output.stdout)

    if status.has_unstaged_changes:
        if Confirm.ask(
            "Modified files found. Add [bold yellow]all unstaged changes[/] to staging?",
            default=True,
        ):
            repo.stage_modified()
            console.print("[green]Staged modified files.[/green]")

    if status.has_untracked_files:
        if Confirm.ask(
            "Untracked files found. Add [bold yellow]all untracked files and unstaged changes[/] to staging?",
            default=True,
        ):
            repo.stage_files()
            console.print("[green]Staged untracked files.[/green]")

    return repo.get_status()


def request_commit_message(
    status: GitStatus,
    model: str | None = None,
    context: str = "",
    provider_config: providers.ProviderConfig | None = None,
    http_client_config: llm.HttpClientConfig | None = None,
) -> str:
    """Request a commit message for the provided staged state."""
    try:
        with console.status(
            "[yellow]Generating commit message based on [bold]`git diff --staged`[/] ...[/yellow]"
        ):
            return generate_commit_message_for_status(
                status,
                model=model,
                context=context,
                provider_config=provider_config,
                http_client_config=http_client_config,
            )
    except llm.LLMError as exc:
        print_llm_error("Could not generate a commit message", exc)
        raise typer.Exit(1)


def request_split_commit_plan(
    status: GitStatus,
    patch_units: tuple[PatchUnit, ...],
    *,
    preferred_commits: int | None = None,
    model: str | None = None,
    context: str = "",
    provider_config: providers.ProviderConfig | None = None,
    http_client_config: llm.HttpClientConfig | None = None,
) -> SplitCommitPlan:
    """Request and validate a split-commit plan for the staged patch units."""
    planner_system_prompt = load_named_prompt(SPLIT_COMMIT_PLANNER_PROMPT_FILENAME)
    planner_prompt = build_split_plan_prompt(
        status,
        patch_units,
        preferred_commits=preferred_commits,
        context=context,
    )

    try:
        with console.status(
            "[yellow]Planning split commits from [bold]staged hunks[/] ...[/yellow]"
        ):
            response = ask_llm_with_system_prompt(
                planner_system_prompt,
                planner_prompt,
                model=model,
                provider_config=provider_config,
                http_client_config=http_client_config,
            )
    except llm.LLMError as exc:
        if not should_retry_with_compact_prompt(exc):
            print_llm_error("Could not generate a split commit plan", exc)
            raise typer.Exit(1)

        console.print(
            "[yellow]Staged patch units exceeded the model context window; retrying split planning with summaries only.[/yellow]"
        )
    else:
        return parse_split_plan_response(
            response,
            patch_units,
        )

    compact_planner_prompt = build_split_plan_prompt(
        status,
        patch_units,
        preferred_commits=preferred_commits,
        context=context,
        include_patches=False,
    )

    try:
        with console.status(
            "[yellow]Planning split commits from [bold]patch summaries[/] ...[/yellow]"
        ):
            response = ask_llm_with_system_prompt(
                planner_system_prompt,
                compact_planner_prompt,
                model=model,
                provider_config=provider_config,
                http_client_config=http_client_config,
            )
    except llm.LLMError as exc:
        print_llm_error("Could not generate a split commit plan", exc)
        raise typer.Exit(1)

    return parse_split_plan_response(
        response,
        patch_units,
    )


def request_split_commit_messages(
    plan: SplitCommitPlan,
    patch_units: tuple[PatchUnit, ...],
    *,
    model: str | None = None,
    context: str = "",
    provider_config: providers.ProviderConfig | None = None,
    http_client_config: llm.HttpClientConfig | None = None,
) -> list[PreparedSplitCommit]:
    """Generate commit messages for each planned split-commit group."""
    try:
        prepared_commits: list[PreparedSplitCommit] = []
        grouped_units = group_patch_units(patch_units, plan)
        total_commits = len(grouped_units)

        for index, unit_group in enumerate(grouped_units, start=1):
            with console.status(
                f"[yellow]Generating commit message {index}/{total_commits} based on [bold]planned staged diff[/] ...[/yellow]"
            ):
                message = generate_commit_message_for_status(
                    build_status_for_patch_units(unit_group),
                    model=model,
                    context=context,
                    provider_config=provider_config,
                    http_client_config=http_client_config,
                )

            prepared_commits.append(
                PreparedSplitCommit(message=message, patch_units=tuple(unit_group))
            )

        return prepared_commits
    except llm.LLMError as exc:
        print_llm_error("Could not generate split commit messages", exc)
        raise typer.Exit(1)


def confirm_split_commit_count(
    plan: SplitCommitPlan,
    *,
    preferred_commits: int,
    yes: bool = False,
) -> SplitCommitPlan:
    """Ask whether to proceed when the planner exceeds the preferred count."""
    actual_commits = len(plan.commits)
    if actual_commits <= preferred_commits:
        return plan

    console.print(
        "[yellow]Split planning produced "
        f"{actual_commits} commits, exceeding the preferred count of "
        f"{preferred_commits}.[/yellow]"
    )

    if yes:
        return plan

    if Confirm.ask(
        f"Proceed with [bold]{actual_commits} commits[/] anyway?",
        default=False,
    ):
        return plan

    console.print("Split commit plan cancelled.")
    raise typer.Exit()


def display_commit_message(commit_message: str) -> None:
    """Render the generated commit message."""
    console.print("[yellow]Generated commit message.[/yellow]")
    console.print(
        Panel(
            f"[bold]{commit_message}[/]",
            title="Commit Message",
            border_style="cyan",
            width=len(commit_message) + 5,
        )
    )


def display_split_commit_plan(prepared_commits: list[PreparedSplitCommit]) -> None:
    """Render the split-commit plan preview."""
    console.print("[yellow]Generated split commit plan.[/yellow]")

    for index, prepared_commit in enumerate(prepared_commits, start=1):
        paths = list(dict.fromkeys(unit.path for unit in prepared_commit.patch_units))
        file_lines = "\n".join(f"- {path}" for path in paths)
        console.print(
            Panel(
                f"[bold]{prepared_commit.message}[/]\n\nFiles:\n{file_lines}",
                title=f"Commit {index}",
                border_style="cyan",
            )
        )


def execute_commit_action(
    repo: GitRepository, commit_message: str, yes: bool = False
) -> str:
    """Run the chosen commit action using the provided message."""
    if yes:
        return commit_with_retry_no_verify(repo, commit_message)

    choice = typer.prompt(
        "Choose action: (c)ommit, (e)dit message, (q)uit",
        default="c",
        show_default=True,
    ).lower()

    if choice == "q":
        console.print("Commit cancelled.")
        raise typer.Exit()
    if choice == "e":
        console.print("[cyan]Opening git editor...[/cyan]")
        return commit_with_retry_no_verify(repo, commit_message, use_editor=True)
    if choice == "c":
        return commit_with_retry_no_verify(repo, commit_message)

    console.print("Invalid choice. Commit cancelled.")
    raise typer.Exit()


def execute_split_commit_plan(
    repo: GitRepository,
    prepared_commits: list[PreparedSplitCommit],
    *,
    yes: bool = False,
) -> list[str]:
    """Run the split-commit plan against temporary alternate indexes."""
    use_editor = False
    if not yes:
        choice = typer.prompt(
            "Choose action: (c)ommit all, (e)dit each message, (q)uit",
            default="c",
            show_default=True,
        ).lower()

        if choice == "q":
            console.print("Commit cancelled.")
            raise typer.Exit()
        if choice == "e":
            use_editor = True
        elif choice != "c":
            console.print("Invalid choice. Commit cancelled.")
            raise typer.Exit()

    execution_state = SplitCommitExecutionState(
        original_head_sha=repo.get_head_sha() if repo.has_commit("HEAD") else None,
        original_head_ref=repo.get_symbolic_head_ref(),
    )
    commit_shas: list[str] = []
    total_commits = len(prepared_commits)

    try:
        for index, prepared_commit in enumerate(prepared_commits, start=1):
            console.print(
                f"[cyan]Creating commit {index}/{total_commits}:[/cyan] {prepared_commit.message}"
            )

            with repo.temporary_alternate_index() as alternate_index:
                try:
                    for patch_unit in prepared_commit.patch_units:
                        repo.check_patch_for_alternate_index(
                            patch_unit.patch,
                            index=alternate_index,
                        )
                        repo.apply_patch_to_alternate_index(
                            patch_unit.patch,
                            index=alternate_index,
                        )
                except GitError as exc:
                    console.print(
                        f"[red]Failed to apply the planned changes for commit {index}: {exc}[/red]"
                    )
                    raise typer.Exit(1)

                try:
                    commit_shas.append(
                        repo.create_commit_from_index(
                            prepared_commit.message,
                            index=alternate_index,
                            use_editor=use_editor,
                        )
                    )
                except GitError as exc:
                    console.print(f"[red]Failed to create commit {index}: {exc}[/red]")
                    raise typer.Exit(1)
    except BaseException:
        try:
            if execution_state.original_head_sha is not None:
                repo.soft_reset(execution_state.original_head_sha)
            elif execution_state.original_head_ref is not None and repo.has_commit(
                "HEAD"
            ):
                repo.delete_ref(execution_state.original_head_ref)
        except GitError as exc:
            console.print(
                "[red]Failed to restore the original staged changes after split commit creation stopped early: "
                f"{exc}[/red]"
            )
        else:
            console.print(
                "[yellow]Split commit creation did not complete; restored the original staged changes.[/yellow]"
            )
        raise

    return commit_shas


def handle_single_commit_flow(
    repo: GitRepository,
    status: GitStatus,
    *,
    model: str | None = None,
    yes: bool = False,
    context: str = "",
    provider_config: providers.ProviderConfig | None = None,
    http_client_config: llm.HttpClientConfig | None = None,
) -> None:
    """Generate, display, and execute the single-commit flow."""
    commit_message = request_commit_message(
        status,
        model=model,
        context=context,
        provider_config=provider_config,
        http_client_config=http_client_config,
    )
    display_commit_message(commit_message)

    commit_sha = execute_commit_action(repo, commit_message, yes=yes)
    console.print(f"[green]✓ Successfully committed: {commit_sha[:8]}[/green]")


def handle_split_commit_flow(
    repo: GitRepository,
    status: GitStatus,
    *,
    preferred_commits: int | None = None,
    model: str | None = None,
    yes: bool = False,
    context: str = "",
    provider_config: providers.ProviderConfig | None = None,
    http_client_config: llm.HttpClientConfig | None = None,
) -> None:
    """Generate, display, and execute the split-commit flow."""
    patch_units = tuple(
        extract_patch_units(repo.get_staged_diff(extra_args=SPLIT_DIFF_ARGS))
    )

    if not patch_units:
        console.print(
            "[yellow]No split patch units were extracted; falling back to a single commit.[/yellow]"
        )
        handle_single_commit_flow(
            repo,
            status,
            model=model,
            yes=yes,
            context=context,
            provider_config=provider_config,
            http_client_config=http_client_config,
        )
        return

    if len(patch_units) == 1:
        console.print(
            "[yellow]Only one staged patch unit was found; creating a single commit.[/yellow]"
        )
        handle_single_commit_flow(
            repo,
            status,
            model=model,
            yes=yes,
            context=context,
            provider_config=provider_config,
            http_client_config=http_client_config,
        )
        return

    if preferred_commits is None:
        console.print(
            "[yellow]Planning split commits from the staged patch units.[/yellow]"
        )
    else:
        console.print(
            "[yellow]Planning split commits with a preference for "
            f"{preferred_commits} commits.[/yellow]"
        )

    try:
        split_plan = request_split_commit_plan(
            status,
            patch_units,
            preferred_commits=preferred_commits,
            model=model,
            context=context,
            provider_config=provider_config,
            http_client_config=http_client_config,
        )
    except SplitPlanningError as exc:
        console.print(
            "[yellow]Split planning returned an invalid plan; falling back to a single commit.[/yellow]"
        )
        console.print(f"[yellow]Reason:[/yellow] {exc}")
        handle_single_commit_flow(
            repo,
            status,
            model=model,
            yes=yes,
            context=context,
            provider_config=provider_config,
            http_client_config=http_client_config,
        )
        return

    if preferred_commits is not None:
        split_plan = confirm_split_commit_count(
            split_plan,
            preferred_commits=preferred_commits,
            yes=yes,
        )

    prepared_commits = request_split_commit_messages(
        split_plan,
        patch_units,
        model=model,
        context=context,
        provider_config=provider_config,
        http_client_config=http_client_config,
    )
    prepared_commits = order_prepared_split_commits(prepared_commits)

    if len(prepared_commits) == 1:
        console.print(
            "[yellow]Split planning resulted in a single commit; using the standard commit flow.[/yellow]"
        )
        display_commit_message(prepared_commits[0].message)
        commit_sha = execute_commit_action(repo, prepared_commits[0].message, yes=yes)
        console.print(f"[green]✓ Successfully committed: {commit_sha[:8]}[/green]")
        return

    display_split_commit_plan(prepared_commits)
    commit_shas = execute_split_commit_plan(repo, prepared_commits, yes=yes)

    console.print(f"[green]✓ Successfully created {len(commit_shas)} commits.[/green]")
    for commit_sha, prepared_commit in zip(commit_shas, prepared_commits, strict=True):
        console.print(f"[green]{commit_sha[:8]}[/green] {prepared_commit.message}")


@app.command("authenticate")
@app.command("login", hidden=True)
def authenticate(
    enterprise_domain: str | None = typer.Option(
        None,
        "--enterprise-domain",
        help="GitHub Enterprise hostname. Omit for github.com.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Replace cached GitHub Copilot credentials"
    ),
    ca_bundle: CaBundleOption = None,
    insecure: InsecureOption = False,
    native_tls: NativeTlsOption = False,
):
    """Authenticate with GitHub Copilot and cache credentials locally."""
    http_client_config = build_http_client_config(
        ca_bundle=ca_bundle,
        insecure=insecure,
        native_tls=native_tls,
    )
    try:
        copilot.login(
            enterprise_domain=enterprise_domain,
            force=force,
            http_client_config=http_client_config,
        )
    except copilot.LLMError as exc:
        print_llm_error("Authentication failed", exc)
        raise typer.Exit(1)


@app.command("summary")
def summary(
    provider: ProviderOption = None,
    base_url: BaseUrlOption = None,
    api_key: ApiKeyOption = None,
    ca_bundle: CaBundleOption = None,
    insecure: InsecureOption = False,
    native_tls: NativeTlsOption = False,
):
    """Show the configured LLM provider summary."""
    http_client_config = build_http_client_config(
        ca_bundle=ca_bundle,
        insecure=insecure,
        native_tls=native_tls,
    )
    try:
        provider_config = providers.resolve_provider_config(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
        )
        providers.show_summary(
            provider_config=provider_config,
            http_client_config=http_client_config,
        )
    except llm.LLMError as exc:
        print_llm_error("Could not load provider summary", exc)
        raise typer.Exit(1)


@app.command("models")
def models_command(
    provider: ProviderOption = None,
    base_url: BaseUrlOption = None,
    api_key: ApiKeyOption = None,
    vendor: str | None = typer.Option(
        None,
        "--vendor",
        help="Filter listed models by vendor: anthropic, gemini/google, or openai.",
    ),
    ca_bundle: CaBundleOption = None,
    insecure: InsecureOption = False,
    native_tls: NativeTlsOption = False,
):
    """List available models for the configured LLM provider."""
    http_client_config = build_http_client_config(
        ca_bundle=ca_bundle,
        insecure=insecure,
        native_tls=native_tls,
    )

    try:
        provider_config = providers.resolve_provider_config(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
        )
        inventory = providers.get_available_models(
            provider_config=provider_config,
            vendor=vendor,
            http_client_config=http_client_config,
        )

        console.print(f"[green]LLM provider:[/green] {provider_config.display_name}")
        console.print(f"[green]Base URL:[/green] {inventory.base_url}")
        console.print(f"[green]Model count:[/green] {len(inventory.models)}")
        llm.print_model_table(
            inventory.models,
            title=f"Available {provider_config.display_name} Models",
        )
    except llm.LLMError as exc:
        print_llm_error("Could not load models", exc)
        raise typer.Exit(1)


@app.command()
def commit(
    all_files: bool = typer.Option(
        False, "--all", "-a", help="Stage all files before committing"
    ),
    split: SplitOption = False,
    split_count: SplitCountOption = None,
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        metavar="MODEL_ID",
        help="Model to use for generating commit message",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Automatically accept the generated commit message"
    ),
    context: str = typer.Option(
        "",
        "--context",
        "-c",
        help="Optional user-provided context to guide commit message",
    ),
    provider: ProviderOption = None,
    base_url: BaseUrlOption = None,
    api_key: ApiKeyOption = None,
    ca_bundle: CaBundleOption = None,
    insecure: InsecureOption = False,
    native_tls: NativeTlsOption = False,
):
    """
    Generate commit message based on changes in the current git repository and commit them.
    """
    try:
        repo = GitRepository()
    except NotAGitRepositoryError:
        console.print("[red]Error: Not in a git repository[/red]")
        raise typer.Exit(1)

    http_client_config = build_http_client_config(
        ca_bundle=ca_bundle,
        insecure=insecure,
        native_tls=native_tls,
    )
    try:
        provider_config = providers.resolve_provider_config(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
        )
    except llm.LLMError as exc:
        print_llm_error("Could not resolve the LLM provider", exc)
        raise typer.Exit(1)

    if provider_config.provider == "copilot":
        ensure_copilot_authentication(http_client_config)

    # Get initial status
    status = repo.get_status()

    if not status.files:
        console.print("[yellow]No changes to commit.[/yellow]")
        raise typer.Exit()

    status = stage_changes_for_commit(repo, status, all_files=all_files)

    if context:
        console.print(
            Panel(context.strip(), title="User Context", border_style="magenta")
        )

    normalized_model = normalize_model_name(model)
    try:
        selected_model = providers.ensure_model_ready(
            provider_config=provider_config,
            model=normalized_model,
            http_client_config=http_client_config,
        )
    except llm.LLMError as exc:
        print_llm_error("Could not select a model", exc)
        raise typer.Exit(1)

    display_selected_model(selected_model)
    model = selected_model.id

    if split or split_count is not None:
        handle_split_commit_flow(
            repo,
            status,
            preferred_commits=split_count,
            model=model,
            yes=yes,
            context=context,
            provider_config=provider_config,
            http_client_config=http_client_config,
        )
        return

    handle_single_commit_flow(
        repo,
        status,
        model=model,
        yes=yes,
        context=context,
        provider_config=provider_config,
        http_client_config=http_client_config,
    )


if __name__ == "__main__":
    run()
