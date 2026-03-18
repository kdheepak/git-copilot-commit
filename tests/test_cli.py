from unittest.mock import Mock

import pytest
import typer

import git_copilot_commit.cli as cli
from git_copilot_commit import github_copilot
from git_copilot_commit.cli import (
    PreparedSplitCommit,
    build_commit_message_prompt,
    display_selected_model,
    display_split_commit_plan,
    execute_split_commit_plan,
    generate_commit_message_for_status,
    print_copilot_error,
    resolve_split_commit_limit,
    SPLIT_DIFF_ARGS,
)
from git_copilot_commit.git import GitFile, GitStatus
from git_copilot_commit.split_commits import (
    PatchUnit,
    SplitCommitLimitExceededError,
    SplitCommitPlan,
    SplitPlanCommit,
    extract_patch_units,
)


def make_status(staged_diff: str, unstaged_diff: str = "") -> GitStatus:
    return GitStatus(
        files=[GitFile(path="src/example.py", status=" ", staged_status="M")],
        staged_diff=staged_diff,
        unstaged_diff=unstaged_diff,
    )


def test_build_commit_message_prompt_includes_context_status_and_diff() -> None:
    status = make_status(
        staged_diff="diff --git a/src/example.py b/src/example.py\n+print('hi')\n"
    )

    prompt = build_commit_message_prompt(status, context="Keep the title concise")

    assert "User-provided context" in prompt
    assert "Keep the title concise" in prompt
    assert "`git status`" in prompt
    assert "M  src/example.py" in prompt
    assert "`git diff --staged`" in prompt
    assert "+print('hi')" in prompt


def test_build_commit_message_prompt_requires_staged_changes() -> None:
    status = make_status(staged_diff="   \n")

    with pytest.raises(typer.Exit):
        build_commit_message_prompt(status)


def test_generate_commit_message_for_status_normalizes_model_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = make_status(
        staged_diff="diff --git a/src/example.py b/src/example.py\n+print('hi')\n"
    )
    mock_ask = Mock(return_value="feat: add example")
    monkeypatch.setattr(cli, "load_system_prompt", Mock(return_value="system prompt"))
    monkeypatch.setattr(cli.github_copilot, "ask", mock_ask)

    message = generate_commit_message_for_status(
        status,
        model="github_copilot/gpt-5.4",
        context="Prefer feat scope",
    )

    assert message == "feat: add example"
    assert mock_ask.call_count == 1
    assert mock_ask.call_args.kwargs["model"] == "gpt-5.4"
    rendered_prompt = mock_ask.call_args.args[0]
    assert "system prompt" in rendered_prompt
    assert "Prefer feat scope" in rendered_prompt
    assert "diff --git a/src/example.py b/src/example.py" in rendered_prompt


def test_display_split_commit_plan_shows_files_not_hunk_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_print = Mock()
    monkeypatch.setattr(cli.console, "print", mock_print)
    prepared_commits = [
        PreparedSplitCommit(
            message="feat: add split commits",
            patch_units=(
                PatchUnit(
                    id="u1",
                    order=0,
                    path="src/app.py",
                    staged_status="M",
                    kind="hunk",
                    patch="diff --git a/src/app.py b/src/app.py\n",
                    summary="src/app.py hunk 1/2",
                ),
                PatchUnit(
                    id="u2",
                    order=1,
                    path="src/app.py",
                    staged_status="M",
                    kind="hunk",
                    patch="diff --git a/src/app.py b/src/app.py\n",
                    summary="src/app.py hunk 2/2",
                ),
                PatchUnit(
                    id="u3",
                    order=2,
                    path="README.md",
                    staged_status="A",
                    kind="new_file",
                    patch="diff --git a/README.md b/README.md\n",
                    summary="add README.md",
                ),
            ),
        )
    ]

    display_split_commit_plan(prepared_commits)

    panel = mock_print.call_args_list[1].args[0]
    rendered = str(panel.renderable)
    assert "Files:" in rendered
    assert "- src/app.py" in rendered
    assert "- README.md" in rendered
    assert "hunk 1/2" not in rendered
    assert rendered.count("src/app.py") == 1


def test_resolve_split_commit_limit_can_proceed_with_larger_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.Confirm, "ask", Mock(return_value=True))
    plan = SplitCommitPlan(
        commits=(SplitPlanCommit(("u1",)), SplitPlanCommit(("u2",)), SplitPlanCommit(("u3",)))
    )
    exc = SplitCommitLimitExceededError(plan, 2)

    resolved_plan = resolve_split_commit_limit(exc, yes=False)

    assert resolved_plan == plan


def test_resolve_split_commit_limit_rejects_noninteractive_yes_mode() -> None:
    plan = SplitCommitPlan(
        commits=(SplitPlanCommit(("u1",)), SplitPlanCommit(("u2",)), SplitPlanCommit(("u3",)))
    )
    exc = SplitCommitLimitExceededError(plan, 2)

    with pytest.raises(typer.Exit):
        resolve_split_commit_limit(exc, yes=True)


def test_print_copilot_error_uses_rich_model_selection_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_console_print = Mock()
    mock_print_model_selection_error = Mock()
    monkeypatch.setattr(cli.console, "print", mock_console_print)
    monkeypatch.setattr(
        cli.github_copilot,
        "print_model_selection_error",
        mock_print_model_selection_error,
    )
    exc = github_copilot.ModelSelectionError(
        models=[
            github_copilot.CopilotModel(id="gpt-5.4", name="GPT-5.4"),
            github_copilot.CopilotModel(id="gpt-5.3-codex", name="GPT-5.3 Codex"),
        ],
        requested_model="nope",
    )

    print_copilot_error("Could not generate a commit message", exc)

    mock_console_print.assert_called_once_with(
        "[red]Could not generate a commit message[/red]"
    )
    mock_print_model_selection_error.assert_called_once_with(exc)


def test_display_selected_model_shows_resolved_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_print = Mock()
    monkeypatch.setattr(cli.console, "print", mock_print)
    model = github_copilot.CopilotModel(
        id="gpt-5.4",
        name="GPT-5.4",
        vendor="openai",
        supported_endpoints=("/responses",),
    )

    display_selected_model(model)

    mock_print.assert_called_once_with(
        "[green]Using model:[/green] gpt-5.4 (openai, responses)"
    )


def test_execute_split_commit_plan_creates_multiple_commits(
    git_repo,
    git_repo_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.Confirm, "ask", Mock(return_value=True))
    file_path = git_repo_path / "file.txt"
    file_path.write_text("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n", encoding="utf-8")
    git_repo.stage_files(["file.txt"])
    git_repo.commit("init", no_verify=True)

    file_path.write_text("A\nb\nc\nd\ne\nf\ng\nh\ni\nJ\n", encoding="utf-8")
    git_repo.stage_files(["file.txt"])
    file_path.write_text("A\nb\nc\nd\ne\nf\ng\nhh\ni\nJ\nextra\n", encoding="utf-8")
    original_unstaged_diff = git_repo.get_status().unstaged_diff

    patch_units = tuple(
        extract_patch_units(git_repo.get_staged_diff(extra_args=SPLIT_DIFF_ARGS))
    )
    assert len(patch_units) == 2

    prepared_commits = [
        PreparedSplitCommit(
            message="chore: update first line",
            patch_units=(patch_units[0],),
        ),
        PreparedSplitCommit(
            message="chore: update last line",
            patch_units=(patch_units[1],),
        ),
    ]

    commit_shas = execute_split_commit_plan(git_repo, prepared_commits, yes=True)

    assert len(commit_shas) == 2
    final_status = git_repo.get_status()
    assert not final_status.has_staged_changes
    assert final_status.has_unstaged_changes
    assert final_status.unstaged_diff == original_unstaged_diff

    recent_messages = [message for _, message in git_repo.get_recent_commits(limit=2)]
    assert recent_messages == ["chore: update last line", "chore: update first line"]
