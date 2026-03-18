"""
git-copilot-commit - AI-powered Git commit assistant
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
import os

import rich
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from .git import GitRepository, GitError, GitStatus, NotAGitRepositoryError
from .split_commits import (
    PatchUnit,
    SplitCommitPlan,
    SplitCommitLimitExceededError,
    SplitPlanningError,
    build_split_plan_prompt,
    build_status_for_patch_units,
    extract_patch_units,
    group_patch_units,
    parse_split_plan_response,
)
from .settings import Settings
from .version import __version__
from . import github_copilot

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


SplitOption = Annotated[
    bool,
    typer.Option(
        "--split",
        help=(
            "Split staged hunks into multiple commits automatically. Pass "
            "`--split=N` to prefer up to N commits."
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
) -> github_copilot.HttpClientConfig:
    if ca_bundle is not None:
        ca_bundle = os.path.expanduser(ca_bundle)
    return github_copilot.HttpClientConfig(
        native_tls=native_tls,
        insecure=insecure,
        ca_bundle=ca_bundle,
    )


def print_copilot_error(message: str, exc: github_copilot.CopilotError) -> None:
    """Render Copilot errors, with rich formatting for model selection issues."""
    if isinstance(exc, github_copilot.ModelSelectionError):
        console.print(f"[red]{message}[/red]")
        github_copilot.print_model_selection_error(exc)
        return

    console.print(f"[red]{message}: {exc}[/red]")


def display_selected_model(model: github_copilot.CopilotModel) -> None:
    """Show the resolved Copilot model for the current command."""
    details = [github_copilot.infer_api_surface(model)]
    if model.vendor:
        details.insert(0, model.vendor)
    console.print(f"[green]Using model:[/green] {model.id} ({', '.join(details)})")


def build_commit_message_prompt(status: GitStatus, context: str = "") -> str:
    """Build the prompt used to generate a commit message."""
    if not status.has_staged_changes:
        console.print("[red]No staged changes to commit.[/red]")
        raise typer.Exit()

    prompt_parts = [
        "`git status`:\n",
        f"```\n{status.get_porcelain_output()}\n```",
        "\n\n`git diff --staged`:\n",
        f"```\n{status.staged_diff}\n```",
    ]

    if context.strip():
        prompt_parts.insert(0, f"User-provided context:\n\n{context.strip()}\n\n")

    return "\n".join(prompt_parts)


def normalize_model_name(model: str | None) -> str | None:
    """Normalize model names accepted by the CLI to Copilot API model ids."""
    if model is not None and model.startswith("github_copilot/"):
        return model.replace("github_copilot/", "", 1)
    return model


def ask_copilot_with_system_prompt(
    system_prompt: str,
    prompt: str,
    model: str | None = None,
    http_client_config: github_copilot.HttpClientConfig | None = None,
) -> str:
    """Send a prepared prompt to Copilot using the provided system prompt."""
    return github_copilot.ask(
        f"""
# System Prompt

{system_prompt}

# Prompt

{prompt}
            """,
        model=normalize_model_name(model),
        http_client_config=http_client_config,
    )


def generate_commit_message_for_prompt(
    prompt: str,
    model: str | None = None,
    http_client_config: github_copilot.HttpClientConfig | None = None,
) -> str:
    """Generate a conventional commit message from a prepared prompt."""
    return ask_copilot_with_system_prompt(
        load_system_prompt(),
        prompt,
        model=model,
        http_client_config=http_client_config,
    )


def generate_commit_message_for_status(
    status: GitStatus,
    model: str | None = None,
    context: str = "",
    http_client_config: github_copilot.HttpClientConfig | None = None,
) -> str:
    """Generate a commit message for a staged status snapshot."""
    prompt = build_commit_message_prompt(status, context=context)
    return generate_commit_message_for_prompt(
        prompt,
        model=model,
        http_client_config=http_client_config,
    )


def generate_commit_message(
    repo: GitRepository,
    model: str | None = None,
    context: str = "",
    http_client_config: github_copilot.HttpClientConfig | None = None,
) -> str:
    """Generate a conventional commit message using the repository's staged diff."""
    return generate_commit_message_for_status(
        repo.get_status(),
        model=model,
        context=context,
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
    http_client_config: github_copilot.HttpClientConfig,
) -> None:
    """Authenticate if no cached Copilot credentials are available."""
    try:
        existing_credentials = github_copilot.load_credentials()
    except github_copilot.CopilotError:
        existing_credentials = None

    if existing_credentials is not None:
        return

    try:
        github_copilot.login(
            force=True,
            http_client_config=http_client_config,
        )
    except github_copilot.CopilotError as exc:
        print_copilot_error("Authentication failed", exc)
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
    http_client_config: github_copilot.HttpClientConfig | None = None,
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
                http_client_config=http_client_config,
            )
    except github_copilot.CopilotError as exc:
        print_copilot_error("Could not generate a commit message", exc)
        raise typer.Exit(1)


def request_split_commit_plan(
    status: GitStatus,
    patch_units: tuple[PatchUnit, ...],
    *,
    max_commits: int,
    model: str | None = None,
    context: str = "",
    http_client_config: github_copilot.HttpClientConfig | None = None,
) -> SplitCommitPlan:
    """Request and validate a split-commit plan for the staged patch units."""
    try:
        planner_prompt = build_split_plan_prompt(
            status,
            patch_units,
            max_commits=max_commits,
            context=context,
        )

        with console.status(
            "[yellow]Planning split commits from [bold]staged hunks[/] ...[/yellow]"
        ):
            response = ask_copilot_with_system_prompt(
                load_named_prompt(SPLIT_COMMIT_PLANNER_PROMPT_FILENAME),
                planner_prompt,
                model=model,
                http_client_config=http_client_config,
            )
        return parse_split_plan_response(
            response,
            patch_units,
            max_commits=max_commits,
        )
    except github_copilot.CopilotError as exc:
        print_copilot_error("Could not generate a split commit plan", exc)
        raise typer.Exit(1)


def request_split_commit_messages(
    plan: SplitCommitPlan,
    patch_units: tuple[PatchUnit, ...],
    *,
    model: str | None = None,
    context: str = "",
    http_client_config: github_copilot.HttpClientConfig | None = None,
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
                    http_client_config=http_client_config,
                )

            prepared_commits.append(
                PreparedSplitCommit(message=message, patch_units=tuple(unit_group))
            )

        return prepared_commits
    except github_copilot.CopilotError as exc:
        print_copilot_error("Could not generate split commit messages", exc)
        raise typer.Exit(1)


def resolve_split_commit_limit(
    exc: SplitCommitLimitExceededError, *, yes: bool = False
) -> SplitCommitPlan:
    """Ask whether to proceed when the planner exceeds the configured limit."""
    console.print(
        f"[yellow]Split planning produced {exc.actual_commits} commits, exceeding --max-commits={exc.max_commits}.[/yellow]"
    )

    if yes:
        console.print(
            "[red]Cannot ask whether to proceed because --yes was used. Re-run without --yes to review the larger plan.[/red]"
        )
        raise typer.Exit(1)

    if Confirm.ask(
        f"Proceed with [bold]{exc.actual_commits} commits[/] anyway?",
        default=False,
    ):
        return exc.plan

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

    commit_shas: list[str] = []
    total_commits = len(prepared_commits)

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

            commit_shas.append(
                commit_with_retry_no_verify(
                    repo,
                    prepared_commit.message,
                    use_editor=use_editor,
                    env=alternate_index.env,
                )
            )

    return commit_shas


def handle_single_commit_flow(
    repo: GitRepository,
    status: GitStatus,
    *,
    model: str | None = None,
    yes: bool = False,
    context: str = "",
    http_client_config: github_copilot.HttpClientConfig | None = None,
) -> None:
    """Generate, display, and execute the single-commit flow."""
    commit_message = request_commit_message(
        status,
        model=model,
        context=context,
        http_client_config=http_client_config,
    )
    display_commit_message(commit_message)

    commit_sha = execute_commit_action(repo, commit_message, yes=yes)
    console.print(f"[green]✓ Successfully committed: {commit_sha[:8]}[/green]")


def handle_split_commit_flow(
    repo: GitRepository,
    status: GitStatus,
    *,
    max_commits: int,
    model: str | None = None,
    yes: bool = False,
    context: str = "",
    prompt_file: Path | None = None,
    http_client_config: github_copilot.HttpClientConfig | None = None,
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
            prompt_file=prompt_file,
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
            prompt_file=prompt_file,
            http_client_config=http_client_config,
        )
        return

    try:
        split_plan = request_split_commit_plan(
            status,
            patch_units,
            max_commits=max_commits,
            model=model,
            context=context,
            http_client_config=http_client_config,
        )
    except SplitCommitLimitExceededError as exc:
        split_plan = resolve_split_commit_limit(exc, yes=yes)
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
            http_client_config=http_client_config,
        )
        return

    prepared_commits = request_split_commit_messages(
        split_plan,
        patch_units,
        model=model,
        context=context,
        http_client_config=http_client_config,
    )

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
        github_copilot.login(
            enterprise_domain=enterprise_domain,
            force=force,
            http_client_config=http_client_config,
        )
    except github_copilot.CopilotError as exc:
        print_copilot_error("Authentication failed", exc)
        raise typer.Exit(1)


@app.command("summary")
def summary(
    ca_bundle: CaBundleOption = None,
    insecure: InsecureOption = False,
    native_tls: NativeTlsOption = False,
):
    """Show the current cached GitHub Copilot login summary."""
    http_client_config = build_http_client_config(
        ca_bundle=ca_bundle,
        insecure=insecure,
        native_tls=native_tls,
    )
    try:
        github_copilot.show_login_summary(http_client_config=http_client_config)
    except github_copilot.CopilotError as exc:
        print_copilot_error("Could not load login summary", exc)
        raise typer.Exit(1)


@app.command("models")
def models_command(
    vendor: str | None = typer.Option(
        None,
        "--vendor",
        help="Filter listed models by vendor: anthropic, gemini/google, or openai.",
    ),
    ca_bundle: CaBundleOption = None,
    insecure: InsecureOption = False,
    native_tls: NativeTlsOption = False,
):
    """List available Copilot models for the current account."""
    http_client_config = build_http_client_config(
        ca_bundle=ca_bundle,
        insecure=insecure,
        native_tls=native_tls,
    )

    try:
        credentials, models = github_copilot.get_available_models(
            vendor=vendor,
            http_client_config=http_client_config,
        )

        console.print(f"[green]Copilot base URL:[/green] {credentials.base_url()}")
        console.print(f"[green]Model count:[/green] {len(models)}")
        github_copilot.print_model_table(models)
    except github_copilot.CopilotError as exc:
        print_copilot_error("Could not load models", exc)
        raise typer.Exit(1)


@app.command()
def commit(
    all_files: bool = typer.Option(
        False, "--all", "-a", help="Stage all files before committing"
    ),
    split: SplitOption = False,
    max_commits: MaxCommitsOption = 10,
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
        selected_model = github_copilot.ensure_auth_ready(
            model=normalized_model,
            http_client_config=http_client_config,
        )
    except github_copilot.CopilotError as exc:
        print_copilot_error("Could not select a model", exc)
        raise typer.Exit(1)

    display_selected_model(selected_model)
    model = selected_model.id

    if split:
        handle_split_commit_flow(
            repo,
            status,
            max_commits=max_commits,
            model=model,
            yes=yes,
            context=context,
            http_client_config=http_client_config,
        )
        return

    handle_single_commit_flow(
        repo,
        status,
        model=model,
        yes=yes,
        context=context,
        http_client_config=http_client_config,
    )


if __name__ == "__main__":
    run()
