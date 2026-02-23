"""
git-copilot-commit - AI-powered Git commit assistant
"""

import rich.terminal_theme
import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel
import rich
from pathlib import Path

from litellm import completion
from .git import GitRepository, GitError, NotAGitRepositoryError
from .settings import Settings
from .version import __version__

console = Console()
app = typer.Typer(help=__doc__, add_completion=False)


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


def ask(prompt, model) -> str:
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": load_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        extra_headers={
            "editor-version": "vscode/1.85.1",
            "Copilot-Integration-Id": "vscode-chat",
        },
    )
    text = response.choices[0].message.content
    text = text.strip()
    # Remove triple backticks if they wrap the entire text
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    # Otherwise remove single backticks if they wrap the entire text
    elif text.startswith("`") and text.endswith("`"):
        text = text[1:-1].strip()
    return text


def generate_commit_message(
    repo: GitRepository, model: str | None = None, context: str = ""
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
        model = "github_copilot/gpt-4"

    if not model.startswith("github_copilot/"):
        model = f"github_copilot/{model}"

    try:
        return ask(prompt, model=model)
    except Exception as _:
        console.print(
            "Prompt failed, falling back to simpler commit message generation."
        )

        fallback_prompt_parts = [
            "`git status`:\n",
            f"```\n{status.get_porcelain_output()}\n```",
        ]

        if context.strip():
            fallback_prompt_parts.insert(
                0, f"User-provided context:\n\n{context.strip()}\n\n"
            )

        fallback_prompt_parts.append(
            "\nGenerate a conventional commit message based on the git status above:"
        )

        fallback_prompt = "\n".join(fallback_prompt_parts)

        return ask(fallback_prompt, model=model)


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
):
    """
    Generate commit message based on changes in the current git repository and commit them.
    """
    try:
        repo = GitRepository()
    except NotAGitRepositoryError:
        console.print("[red]Error: Not in a git repository[/red]")
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

    # Generate or use provided commit message
    with console.status(
        "[yellow]Generating commit message based on [bold]`git diff --staged`[/] ...[/yellow]"
    ):
        commit_message = generate_commit_message(repo, model, context=context)

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


@app.command()
def config(
    set_default_model: str | None = typer.Option(
        None, "--set-default-model", help="Set default model for commit messages"
    ),
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
):
    """Manage application configuration."""
    settings = Settings()

    if set_default_model:
        settings.default_model = set_default_model
        console.print(f"[green]✓ Default model set to: {set_default_model}[/green]")

    if show or (not set_default_model):
        console.print("\n[bold]Current Configuration:[/bold]")
        default_model = settings.default_model
        if default_model:
            console.print(f"Default model: [cyan]{default_model}[/cyan]")
        else:
            console.print("Default model: [dim]not set[/dim]")

        active_prompt = get_active_prompt_path()
        if active_prompt:
            console.print(f"Active prompt file: [cyan]{active_prompt}[/cyan]")
        else:
            console.print("Active prompt file: [red]not found[/red]")

        console.print(f"Config file: [dim]{settings.config_file}[/dim]")


if __name__ == "__main__":
    app()
