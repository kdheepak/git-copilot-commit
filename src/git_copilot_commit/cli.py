"""
git-copilot-commit - AI-powered Git commit assistant
"""

from pathlib import Path
from typing import Annotated

import rich
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from .git import GitRepository, GitError, NotAGitRepositoryError
from .settings import Settings
from .version import __version__
from . import github_copilot

console = Console()
app = typer.Typer(help=__doc__, add_completion=False)

CA_BUNDLE_HELP = (
    "Path to a custom CA bundle (PEM). Use this to test internal / company CAs."
)
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


def get_prompt_locations():
    """Get potential prompt file locations in order of preference."""
    import importlib.resources

    filename = "commit-message-generator-prompt.md"

    return [
        Path(Settings().data_dir) / "prompts" / filename,  # User customizable
        importlib.resources.files("git_copilot_commit")
        / "prompts"
        / filename,  # Packaged version
    ]


def get_active_prompt_path():
    """Get the path of the prompt file that will be used."""
    for path in get_prompt_locations():
        try:
            path.read_text(encoding="utf-8")
            return str(path)
        except (FileNotFoundError, AttributeError):
            continue
    return None


def load_system_prompt() -> str:
    """Load the system prompt from the markdown file."""
    for path in get_prompt_locations():
        try:
            return path.read_text(encoding="utf-8")
        except (FileNotFoundError, AttributeError):
            continue

    console.print("[red]Error: Prompt file not found in any location[/red]")
    raise typer.Exit(1)


def build_http_client_config(
    *,
    ca_bundle: str | None,
    insecure: bool,
    native_tls: bool,
) -> github_copilot.HttpClientConfig:
    return github_copilot.HttpClientConfig(
        native_tls=native_tls,
        insecure=insecure,
        ca_bundle=ca_bundle,
    )


def generate_commit_message(
    repo: GitRepository,
    model: str | None = None,
    context: str = "",
    http_client_config: github_copilot.HttpClientConfig | None = None,
) -> str:
    """Generate a conventional commit message using Copilot API."""

    # Refresh status after staging
    status = repo.get_status()

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

    prompt_parts.append("\nGenerate a conventional commit message:")

    prompt = "\n".join(prompt_parts)

    if model is None:
        model = "gpt-5.1-codex"

    if model.startswith("github_copilot/"):
        model = model.replace("github_copilot/", "")

    return github_copilot.ask(
        f"""
# System Prompt

{load_system_prompt()}

# Prompt

{prompt}
            """,
        model=model,
        http_client_config=http_client_config,
    )


def commit_with_retry_no_verify(
    repo: GitRepository, message: str, use_editor: bool = False
) -> str:
    """Run commit and offer one retry with -n on failure."""
    try:
        return repo.commit(message, use_editor=use_editor)
    except GitError as e:
        console.print(f"[red]Commit failed: {e}[/red]")
        if not Confirm.ask(
            "Retry commit with [bold]`-n`[/] (skip hooks) using the same commit message?",
            default=True,
        ):
            raise typer.Exit(1)

    try:
        return repo.commit(message, use_editor=use_editor, no_verify=True)
    except GitError as retry_error:
        console.print(f"[red]Commit with -n failed: {retry_error}[/red]")
        raise typer.Exit(1)


@app.command("authenticate")
@app.command("login", hidden=True)
def authenticate(
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
            force=force,
            http_client_config=http_client_config,
        )
    except github_copilot.CopilotError as exc:
        console.print(f"[red]Authentication failed: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def commit(
    all_files: bool = typer.Option(
        False, "--all", "-a", help="Stage all files before committing"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model to use for generating commit message"
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

    try:
        existing_credentials = github_copilot.load_credentials()
    except github_copilot.CopilotError:
        existing_credentials = None

    http_client_config = build_http_client_config(
        ca_bundle=ca_bundle,
        insecure=insecure,
        native_tls=native_tls,
    )

    if existing_credentials is None:
        try:
            github_copilot.login(
                force=True,
                http_client_config=http_client_config,
            )
        except github_copilot.CopilotError as exc:
            console.print(f"[red]Authentication failed: {exc}[/red]")
            raise typer.Exit(1)

    # Load settings and use default model if none provided
    settings = Settings()
    if model is None:
        model = settings.default_model

    # Get initial status
    status = repo.get_status()

    if not status.files:
        console.print("[yellow]No changes to commit.[/yellow]")
        raise typer.Exit()

    # Handle staging based on options
    if all_files:
        repo.stage_files()  # Stage all files
        console.print("[green]Staged all files.[/green]")
    else:
        # Show git status once if there are unstaged or untracked files to prompt about
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

    if context:
        console.print(
            Panel(context.strip(), title="User Context", border_style="magenta")
        )

    try:
        github_copilot.ensure_auth_ready(
            model=model,
            http_client_config=http_client_config,
        )

        # Generate or use provided commit message
        with console.status(
            "[yellow]Generating commit message based on [bold]`git diff --staged`[/] ...[/yellow]"
        ):
            commit_message = generate_commit_message(
                repo,
                model,
                context=context,
                http_client_config=http_client_config,
            )
    except github_copilot.CopilotError as exc:
        console.print(f"[red]Could not generate a commit message: {exc}[/red]")
        raise typer.Exit(1)

    console.print("[yellow]Generated commit message.[/yellow]")

    # Display commit message
    console.print(
        Panel(
            f"[bold]{commit_message}[/]",
            title="Commit Message",
            border_style="cyan",
            width=len(commit_message) + 5,
        )
    )

    # Confirm commit or edit message (skip if --yes flag is used)
    if yes:
        # Automatically commit with generated message
        commit_sha = commit_with_retry_no_verify(repo, commit_message)
    else:
        choice = typer.prompt(
            "Choose action: (c)ommit, (e)dit message, (q)uit",
            default="c",
            show_default=True,
        ).lower()

        if choice == "q":
            console.print("Commit cancelled.")
            raise typer.Exit()
        elif choice == "e":
            # Use git's built-in editor with generated message as template
            console.print("[cyan]Opening git editor...[/cyan]")
            commit_sha = commit_with_retry_no_verify(
                repo, commit_message, use_editor=True
            )
        elif choice == "c":
            # Commit with generated message
            commit_sha = commit_with_retry_no_verify(repo, commit_message)
        else:
            console.print("Invalid choice. Commit cancelled.")
            raise typer.Exit()

    # Show success message
    console.print(f"[green]✓ Successfully committed: {commit_sha[:8]}[/green]")


if __name__ == "__main__":
    app()
