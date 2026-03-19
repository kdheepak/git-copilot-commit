from unittest.mock import Mock
from pathlib import Path

import pytest  # noqa: F401
import typer
from typer.testing import CliRunner

import git_copilot_commit.cli as cli
from git_copilot_commit import github_copilot
from git_copilot_commit.cli import (
    PreparedSplitCommit,
    build_http_client_config,
    build_commit_message_prompt,
    confirm_split_commit_count,
    commit_with_retry_no_verify,
    display_selected_model,
    display_split_commit_plan,
    execute_split_commit_plan,
    generate_commit_message_for_status,
    handle_split_commit_flow,
    load_named_prompt,
    load_system_prompt,
    normalize_model_name,
    print_copilot_error,
    preprocess_cli_args,
    resolve_prompt_file,
    run,
    SPLIT_DIFF_ARGS,
    stage_changes_for_commit,
)
from git_copilot_commit.git import GitFile, GitStatus
from git_copilot_commit.split_commits import (
    PatchUnit,
    SplitCommitPlan,
    SplitPlanCommit,
    extract_patch_units,
)

runner = CliRunner()


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


def test_confirm_split_commit_count_can_proceed_with_larger_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confirm_ask = Mock(return_value=True)
    monkeypatch.setattr(cli.Confirm, "ask", confirm_ask)
    plan = SplitCommitPlan(
        commits=(
            SplitPlanCommit(("u1",)),
            SplitPlanCommit(("u2",)),
            SplitPlanCommit(("u3",)),
        )
    )

    resolved_plan = confirm_split_commit_count(
        plan,
        preferred_commits=2,
        yes=False,
    )

    assert resolved_plan == plan


def test_confirm_split_commit_count_assumes_yes_means_proceed() -> None:
    plan = SplitCommitPlan(
        commits=(
            SplitPlanCommit(("u1",)),
            SplitPlanCommit(("u2",)),
            SplitPlanCommit(("u3",)),
        )
    )

    resolved_plan = confirm_split_commit_count(
        plan,
        preferred_commits=2,
        yes=True,
    )

    assert resolved_plan == plan


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


def test_cli_version_flag_prints_version() -> None:
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert "git-copilot-commit" in result.stdout


def test_build_http_client_config_and_normalize_model_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    config = build_http_client_config(
        ca_bundle="~/certs/custom.pem",
        insecure=False,
        native_tls=True,
    )

    assert config.ca_bundle == str(tmp_path / "certs" / "custom.pem")
    assert not config.use_native_tls
    assert normalize_model_name("github_copilot/gpt-5.4") == "gpt-5.4"
    assert normalize_model_name("gpt-5.4") == "gpt-5.4"
    assert normalize_model_name(None) is None


def test_preprocess_cli_args_rewrites_split_syntax() -> None:
    assert preprocess_cli_args(["commit", "--split=auto", "--yes"]) == [
        "commit",
        "--split",
        "--yes",
    ]
    assert preprocess_cli_args(["commit", "--split", "auto", "--yes"]) == [
        "commit",
        "--split",
        "--yes",
    ]
    assert preprocess_cli_args(["commit", "--split=3", "--yes"]) == [
        "commit",
        "--split-count",
        "3",
        "--yes",
    ]
    assert preprocess_cli_args(["commit", "--split", "3", "--yes"]) == [
        "commit",
        "--split-count",
        "3",
        "--yes",
    ]
    assert preprocess_cli_args(["summary", "--split=auto"]) == [
        "summary",
        "--split=auto",
    ]


def test_run_uses_preprocessed_args(monkeypatch: pytest.MonkeyPatch) -> None:
    command = Mock()
    monkeypatch.setattr(cli, "get_command", Mock(return_value=command))

    run(["commit", "--split=auto", "--yes"])

    command.main.assert_called_once()
    assert command.main.call_args.kwargs["args"] == [
        "commit",
        "--split",
        "--yes",
    ]


def test_load_named_prompt_prefers_first_existing_location(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    user_prompt = tmp_path / "user.md"
    packaged_prompt = tmp_path / "packaged.md"
    packaged_prompt.write_text("packaged prompt", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "get_prompt_locations",
        lambda _filename: [user_prompt, packaged_prompt],
    )

    assert load_named_prompt("ignored.md") == "packaged prompt"

    user_prompt.write_text("user prompt", encoding="utf-8")
    assert load_named_prompt("ignored.md") == "user prompt"


def test_load_named_prompt_and_resolve_prompt_file_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_print = Mock()
    monkeypatch.setattr(cli.console, "print", mock_print)
    monkeypatch.setattr(
        cli,
        "get_prompt_locations",
        lambda _filename: [
            Path("/does/not/exist/one.md"),
            Path("/does/not/exist/two.md"),
        ],
    )

    with pytest.raises(typer.Exit):
        load_named_prompt("missing.md")

    class BrokenSettings:
        config_file = Path("/tmp/config.json")

        @property
        def default_prompt_file(self) -> str | None:
            raise ValueError("bad config")

    monkeypatch.setattr(cli, "Settings", BrokenSettings)

    with pytest.raises(typer.Exit):
        resolve_prompt_file()

    assert mock_print.call_count == 2


def test_load_system_prompt_and_resolve_prompt_file_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    prompt_path = prompt_dir / "default.md"
    prompt_path.write_text("system prompt", encoding="utf-8")

    class FakeSettings:
        config_file = tmp_path / "config.json"

        @property
        def default_prompt_file(self) -> str | None:
            return "~/prompts/default.md"

    monkeypatch.setattr(cli, "Settings", FakeSettings)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_prompt_file() == prompt_path
    assert load_system_prompt() == "system prompt"

    prompt_path.unlink()
    with pytest.raises(typer.Exit):
        load_system_prompt()


def test_commit_with_retry_no_verify_retries_and_can_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Mock()
    repo.commit.side_effect = [
        cli.GitError("pre-commit hook failed"),
        "abc123",
    ]
    monkeypatch.setattr(cli.Confirm, "ask", Mock(return_value=True))

    assert commit_with_retry_no_verify(repo, "feat: add retry") == "abc123"
    assert repo.commit.call_args_list[1].kwargs["no_verify"] is True

    aborting_repo = Mock()
    aborting_repo.commit.side_effect = cli.GitError("pre-commit hook failed")
    monkeypatch.setattr(cli.Confirm, "ask", Mock(return_value=False))

    with pytest.raises(typer.Exit):
        commit_with_retry_no_verify(aborting_repo, "feat: abort retry")


def test_stage_changes_for_commit_stages_requested_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refreshed_status = make_status(
        staged_diff="diff --git a/src/example.py b/src/example.py\n+print('hi')\n"
    )
    repo = Mock()
    repo.get_status.return_value = refreshed_status
    repo._run_git_command.return_value.stdout = "git status output"

    status = GitStatus(
        files=[
            GitFile(path="src/example.py", status="M", staged_status=" "),
            GitFile(path="src/new.py", status="?", staged_status="?"),
        ],
        staged_diff="",
        unstaged_diff="diff --git a/src/example.py b/src/example.py\n",
    )
    monkeypatch.setattr(cli.Confirm, "ask", Mock(side_effect=[True, True]))

    result = stage_changes_for_commit(repo, status, all_files=False)

    assert result == refreshed_status
    repo.stage_modified.assert_called_once_with()
    repo.stage_files.assert_called_once_with()


def test_stage_changes_for_commit_all_files_short_circuit() -> None:
    refreshed_status = make_status(
        staged_diff="diff --git a/src/example.py b/src/example.py\n+print('hi')\n"
    )
    repo = Mock()
    repo.get_status.return_value = refreshed_status

    result = stage_changes_for_commit(repo, refreshed_status, all_files=True)

    assert result == refreshed_status
    repo.stage_files.assert_called_once_with()


def test_commit_command_from_subdirectory_stages_the_entire_repository(
    git_repo_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = cli.GitRepository(git_repo_path)
    frontend_dir = git_repo_path / "frontend"
    backend_dir = git_repo_path / "backend"
    frontend_dir.mkdir()
    backend_dir.mkdir()
    frontend_file = frontend_dir / "component.py"
    backend_file = backend_dir / "service.py"

    frontend_file.write_text("print('frontend v1')\n", encoding="utf-8")
    backend_file.write_text("print('backend v1')\n", encoding="utf-8")
    repo.stage_files()
    repo.commit("chore: init", no_verify=True)

    frontend_file.write_text("print('frontend v2')\n", encoding="utf-8")
    backend_file.write_text("print('backend v2')\n", encoding="utf-8")

    monkeypatch.chdir(frontend_dir)
    monkeypatch.setattr(cli.Confirm, "ask", Mock(return_value=True))
    monkeypatch.setattr(cli, "ensure_copilot_authentication", lambda _config: None)
    monkeypatch.setattr(
        cli.github_copilot,
        "ensure_auth_ready",
        lambda **_kwargs: github_copilot.CopilotModel(
            id="gpt-5.4",
            name="GPT-5.4",
            vendor="openai",
        ),
    )
    monkeypatch.setattr(
        cli,
        "request_commit_message",
        lambda *_args, **_kwargs: "chore: commit nested changes",
    )

    result = runner.invoke(cli.app, ["commit", "--all", "--yes"])

    assert result.exit_code == 0
    assert repo.get_recent_commits(limit=1)[0][1] == "chore: commit nested changes"
    assert not repo.get_status().files


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


def test_handle_split_commit_flow_falls_back_to_single_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Mock()
    repo.get_staged_diff.return_value = ""
    status = make_status(
        staged_diff="diff --git a/src/example.py b/src/example.py\n+print('hi')\n"
    )
    fallback = Mock()
    monkeypatch.setattr(cli, "handle_single_commit_flow", fallback)
    monkeypatch.setattr(cli, "extract_patch_units", lambda _diff: [])

    handle_split_commit_flow(repo, status, model="gpt-5.4")

    fallback.assert_called_once()


def test_handle_split_commit_flow_auto_mode_always_requests_split_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Mock()
    repo.get_staged_diff.return_value = "diff"
    status = make_status(
        staged_diff="diff --git a/src/example.py b/src/example.py\n+print('hi')\n"
    )
    patch_units = (
        PatchUnit(
            id="u1",
            order=0,
            path="src/example.py",
            staged_status="M",
            kind="hunk",
            patch="patch 1",
            summary="summary 1",
        ),
        PatchUnit(
            id="u2",
            order=1,
            path="src/example.py",
            staged_status="M",
            kind="hunk",
            patch="patch 2",
            summary="summary 2",
        ),
    )
    fallback = Mock()
    split_plan = SplitCommitPlan(commits=(SplitPlanCommit(("u1", "u2")),))
    planner = Mock(return_value=split_plan)
    request_messages = Mock(
        return_value=[
            PreparedSplitCommit(
                message="feat: keep both changes together",
                patch_units=patch_units,
            )
        ]
    )
    monkeypatch.setattr(cli, "handle_single_commit_flow", fallback)
    monkeypatch.setattr(cli, "request_split_commit_plan", planner)
    monkeypatch.setattr(cli, "request_split_commit_messages", request_messages)
    monkeypatch.setattr(cli, "extract_patch_units", lambda _diff: patch_units)
    monkeypatch.setattr(cli, "display_commit_message", Mock())
    monkeypatch.setattr(cli, "execute_commit_action", Mock(return_value="deadbeef"))

    handle_split_commit_flow(
        repo,
        status,
        model="gpt-5.4",
    )

    fallback.assert_not_called()
    planner.assert_called_once_with(
        status,
        patch_units,
        preferred_commits=None,
        model="gpt-5.4",
        context="",
        http_client_config=None,
    )


def test_handle_split_commit_flow_handles_invalid_plan_and_single_commit_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = make_status(
        staged_diff="diff --git a/src/example.py b/src/example.py\n+print('hi')\n"
    )
    repo = Mock()
    repo.get_staged_diff.return_value = "diff"
    patch_units = (
        PatchUnit(
            id="u1",
            order=0,
            path="src/example.py",
            staged_status="M",
            kind="hunk",
            patch="patch 1",
            summary="summary 1",
        ),
        PatchUnit(
            id="u2",
            order=1,
            path="README.md",
            staged_status="A",
            kind="new_file",
            patch="patch 2",
            summary="summary 2",
        ),
    )

    fallback = Mock()
    monkeypatch.setattr(cli, "handle_single_commit_flow", fallback)
    monkeypatch.setattr(cli, "extract_patch_units", lambda _diff: patch_units)
    monkeypatch.setattr(
        cli,
        "request_split_commit_plan",
        Mock(side_effect=cli.SplitPlanningError("bad plan")),
    )

    handle_split_commit_flow(repo, status, model="gpt-5.4")

    fallback.assert_called_once()

    monkeypatch.setattr(cli, "request_split_commit_plan", Mock(return_value=Mock()))
    monkeypatch.setattr(
        cli,
        "request_split_commit_messages",
        Mock(
            return_value=[
                PreparedSplitCommit(
                    message="feat: one commit after planning",
                    patch_units=patch_units,
                )
            ]
        ),
    )
    display_message = Mock()
    execute_action = Mock(return_value="deadbeefcafefeed")
    monkeypatch.setattr(cli, "display_commit_message", display_message)
    monkeypatch.setattr(cli, "execute_commit_action", execute_action)
    fallback.reset_mock()

    handle_split_commit_flow(repo, status, model="gpt-5.4")

    fallback.assert_not_called()
    display_message.assert_called_once_with("feat: one commit after planning")
    execute_action.assert_called_once()


def test_handle_split_commit_flow_auto_mode_can_trigger_split_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = make_status(
        staged_diff="diff --git a/src/example.py b/src/example.py\n+print('hi')\n"
    )
    repo = Mock()
    repo.get_staged_diff.return_value = "diff"
    patch_units = (
        PatchUnit(
            id="u1",
            order=0,
            path="src/app.py",
            staged_status="M",
            kind="hunk",
            patch="patch 1",
            summary="summary 1",
        ),
        PatchUnit(
            id="u2",
            order=1,
            path="README.md",
            staged_status="A",
            kind="new_file",
            patch="patch 2",
            summary="summary 2",
        ),
    )
    split_plan = SplitCommitPlan(
        commits=(
            SplitPlanCommit(("u1",)),
            SplitPlanCommit(("u2",)),
        )
    )
    prepared_commits = [
        PreparedSplitCommit(message="feat: code", patch_units=(patch_units[0],)),
        PreparedSplitCommit(message="docs: readme", patch_units=(patch_units[1],)),
    ]
    request_plan = Mock(return_value=split_plan)
    request_messages = Mock(return_value=prepared_commits)
    execute_plan = Mock(return_value=["aaaabbbb", "ccccdddd"])
    display_plan = Mock()
    fallback = Mock()
    monkeypatch.setattr(cli, "extract_patch_units", lambda _diff: patch_units)
    monkeypatch.setattr(cli, "request_split_commit_plan", request_plan)
    monkeypatch.setattr(cli, "request_split_commit_messages", request_messages)
    monkeypatch.setattr(cli, "execute_split_commit_plan", execute_plan)
    monkeypatch.setattr(cli, "display_split_commit_plan", display_plan)
    monkeypatch.setattr(cli, "handle_single_commit_flow", fallback)

    handle_split_commit_flow(
        repo,
        status,
        model="gpt-5.4",
    )

    fallback.assert_not_called()
    request_plan.assert_called_once_with(
        status,
        patch_units,
        preferred_commits=None,
        model="gpt-5.4",
        context="",
        http_client_config=None,
    )
    request_messages.assert_called_once_with(
        split_plan,
        patch_units,
        model="gpt-5.4",
        context="",
        http_client_config=None,
    )
    display_plan.assert_called_once_with(prepared_commits)
    execute_plan.assert_called_once_with(repo, prepared_commits, yes=False)


def test_handle_split_commit_flow_split_limit_can_trigger_split_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = make_status(
        staged_diff="diff --git a/src/example.py b/src/example.py\n+print('hi')\n"
    )
    repo = Mock()
    repo.get_staged_diff.return_value = "diff"
    patch_units = (
        PatchUnit(
            id="u1",
            order=0,
            path="src/app.py",
            staged_status="M",
            kind="hunk",
            patch="patch 1",
            summary="summary 1",
        ),
        PatchUnit(
            id="u2",
            order=1,
            path="README.md",
            staged_status="A",
            kind="new_file",
            patch="patch 2",
            summary="summary 2",
        ),
    )
    split_plan = SplitCommitPlan(
        commits=(
            SplitPlanCommit(("u1",)),
            SplitPlanCommit(("u2",)),
        )
    )
    prepared_commits = [
        PreparedSplitCommit(message="feat: code", patch_units=(patch_units[0],)),
        PreparedSplitCommit(message="docs: readme", patch_units=(patch_units[1],)),
    ]
    request_plan = Mock(return_value=split_plan)
    request_messages = Mock(return_value=prepared_commits)
    execute_plan = Mock(return_value=["aaaabbbb", "ccccdddd"])
    monkeypatch.setattr(cli, "extract_patch_units", lambda _diff: patch_units)
    monkeypatch.setattr(cli, "request_split_commit_plan", request_plan)
    monkeypatch.setattr(cli, "request_split_commit_messages", request_messages)
    monkeypatch.setattr(cli, "execute_split_commit_plan", execute_plan)
    monkeypatch.setattr(cli, "display_split_commit_plan", Mock())
    monkeypatch.setattr(cli, "handle_single_commit_flow", Mock())

    handle_split_commit_flow(
        repo,
        status,
        preferred_commits=2,
        model="gpt-5.4",
    )

    request_plan.assert_called_once_with(
        status,
        patch_units,
        preferred_commits=2,
        model="gpt-5.4",
        context="",
        http_client_config=None,
    )
    execute_plan.assert_called_once_with(repo, prepared_commits, yes=False)


def test_handle_split_commit_flow_prompts_when_plan_exceeds_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = make_status(
        staged_diff="diff --git a/src/example.py b/src/example.py\n+print('hi')\n"
    )
    repo = Mock()
    repo.get_staged_diff.return_value = "diff"
    patch_units = (
        PatchUnit(
            id="u1",
            order=0,
            path="src/app.py",
            staged_status="M",
            kind="hunk",
            patch="patch 1",
            summary="summary 1",
        ),
        PatchUnit(
            id="u2",
            order=1,
            path="src/lib.py",
            staged_status="M",
            kind="hunk",
            patch="patch 2",
            summary="summary 2",
        ),
        PatchUnit(
            id="u3",
            order=2,
            path="README.md",
            staged_status="A",
            kind="new_file",
            patch="patch 3",
            summary="summary 3",
        ),
    )
    split_plan = SplitCommitPlan(
        commits=(
            SplitPlanCommit(("u1",)),
            SplitPlanCommit(("u2",)),
            SplitPlanCommit(("u3",)),
        )
    )
    prepared_commits = [
        PreparedSplitCommit(message="feat: app", patch_units=(patch_units[0],)),
        PreparedSplitCommit(message="refactor: lib", patch_units=(patch_units[1],)),
        PreparedSplitCommit(message="docs: readme", patch_units=(patch_units[2],)),
    ]
    monkeypatch.setattr(cli, "extract_patch_units", lambda _diff: patch_units)
    monkeypatch.setattr(cli, "request_split_commit_plan", Mock(return_value=split_plan))
    confirm_ask = Mock(return_value=True)
    monkeypatch.setattr(cli.Confirm, "ask", confirm_ask)
    request_messages = Mock(return_value=prepared_commits)
    execute_plan = Mock(return_value=["aaaabbbb", "ccccdddd", "eeeeffff"])
    monkeypatch.setattr(cli, "request_split_commit_messages", request_messages)
    monkeypatch.setattr(cli, "execute_split_commit_plan", execute_plan)
    monkeypatch.setattr(cli, "display_split_commit_plan", Mock())
    monkeypatch.setattr(cli, "handle_single_commit_flow", Mock())

    handle_split_commit_flow(
        repo,
        status,
        preferred_commits=2,
        model="gpt-5.4",
    )

    confirm_ask.assert_called_once()
    execute_plan.assert_called_once_with(repo, prepared_commits, yes=False)


def test_handle_split_commit_flow_split_limit_does_not_reject_fewer_patch_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Mock()
    repo.get_staged_diff.return_value = "diff"
    status = make_status(
        staged_diff="diff --git a/src/example.py b/src/example.py\n+print('hi')\n"
    )
    patch_units = (
        PatchUnit(
            id="u1",
            order=0,
            path="src/example.py",
            staged_status="M",
            kind="hunk",
            patch="patch 1",
            summary="summary 1",
        ),
        PatchUnit(
            id="u2",
            order=1,
            path="src/example.py",
            staged_status="M",
            kind="hunk",
            patch="patch 2",
            summary="summary 2",
        ),
    )
    monkeypatch.setattr(cli, "extract_patch_units", lambda _diff: patch_units)
    request_plan = Mock(
        return_value=SplitCommitPlan(commits=(SplitPlanCommit(("u1", "u2")),))
    )
    request_messages = Mock(
        return_value=[
            PreparedSplitCommit(
                message="feat: combine both changes",
                patch_units=patch_units,
            )
        ]
    )
    monkeypatch.setattr(cli, "request_split_commit_plan", request_plan)
    monkeypatch.setattr(cli, "request_split_commit_messages", request_messages)
    monkeypatch.setattr(cli, "display_commit_message", Mock())
    monkeypatch.setattr(cli, "execute_commit_action", Mock(return_value="deadbeef"))

    handle_split_commit_flow(
        repo,
        status,
        preferred_commits=3,
        model="gpt-5.4",
    )

    request_plan.assert_called_once_with(
        status,
        patch_units,
        preferred_commits=3,
        model="gpt-5.4",
        context="",
        http_client_config=None,
    )
