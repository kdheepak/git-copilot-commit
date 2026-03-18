import tempfile
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import typer

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
from git_copilot_commit import github_copilot
from git_copilot_commit.git import GitFile, GitStatus
from git_copilot_commit.git import GitRepository
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


class CommitPromptTests(unittest.TestCase):
    def test_build_commit_message_prompt_includes_context_status_and_diff(self) -> None:
        status = make_status(
            staged_diff="diff --git a/src/example.py b/src/example.py\n+print('hi')\n"
        )

        prompt = build_commit_message_prompt(status, context="Keep the title concise")

        self.assertIn("User-provided context", prompt)
        self.assertIn("Keep the title concise", prompt)
        self.assertIn("`git status`", prompt)
        self.assertIn("M  src/example.py", prompt)
        self.assertIn("`git diff --staged`", prompt)
        self.assertIn("+print('hi')", prompt)

    def test_build_commit_message_prompt_requires_staged_changes(self) -> None:
        status = make_status(staged_diff="   \n")

        with self.assertRaises(typer.Exit):
            build_commit_message_prompt(status)

    @patch("git_copilot_commit.cli.load_system_prompt", return_value="system prompt")
    @patch("git_copilot_commit.cli.github_copilot.ask", return_value="feat: add example")
    def test_generate_commit_message_for_status_normalizes_model_prefix(
        self,
        mock_ask,
        _mock_load_system_prompt,
    ) -> None:
        status = make_status(
            staged_diff="diff --git a/src/example.py b/src/example.py\n+print('hi')\n"
        )

        message = generate_commit_message_for_status(
            status,
            model="github_copilot/gpt-5.4",
            context="Prefer feat scope",
        )

        self.assertEqual(message, "feat: add example")
        self.assertEqual(mock_ask.call_count, 1)
        self.assertEqual(mock_ask.call_args.kwargs["model"], "gpt-5.4")
        rendered_prompt = mock_ask.call_args.args[0]
        self.assertIn("system prompt", rendered_prompt)
        self.assertIn("Prefer feat scope", rendered_prompt)
        self.assertIn("diff --git a/src/example.py b/src/example.py", rendered_prompt)

    @patch("git_copilot_commit.cli.console.print")
    def test_display_split_commit_plan_shows_files_not_hunk_summaries(
        self, mock_print
    ) -> None:
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
        self.assertIn("Files:", rendered)
        self.assertIn("- src/app.py", rendered)
        self.assertIn("- README.md", rendered)
        self.assertNotIn("hunk 1/2", rendered)
        self.assertEqual(rendered.count("src/app.py"), 1)

    @patch("git_copilot_commit.cli.Confirm.ask", return_value=True)
    def test_resolve_split_commit_limit_can_proceed_with_larger_plan(
        self, _mock_confirm
    ) -> None:
        plan = SplitCommitPlan(
            commits=(SplitPlanCommit(("u1",)), SplitPlanCommit(("u2",)), SplitPlanCommit(("u3",)))
        )
        exc = SplitCommitLimitExceededError(plan, 2)

        resolved_plan = resolve_split_commit_limit(exc, yes=False)

        self.assertEqual(resolved_plan, plan)

    def test_resolve_split_commit_limit_rejects_noninteractive_yes_mode(self) -> None:
        plan = SplitCommitPlan(
            commits=(SplitPlanCommit(("u1",)), SplitPlanCommit(("u2",)), SplitPlanCommit(("u3",)))
        )
        exc = SplitCommitLimitExceededError(plan, 2)

        with self.assertRaises(typer.Exit):
            resolve_split_commit_limit(exc, yes=True)

    @patch("git_copilot_commit.cli.github_copilot.print_model_selection_error")
    @patch("git_copilot_commit.cli.console.print")
    def test_print_copilot_error_uses_rich_model_selection_format(
        self,
        mock_console_print,
        mock_print_model_selection_error,
    ) -> None:
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

    @patch("git_copilot_commit.cli.console.print")
    def test_display_selected_model_shows_resolved_model(self, mock_print) -> None:
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


class SplitCommitExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_path = Path(self.tempdir.name)
        self.run_git("init", "-q")
        self.run_git("config", "user.name", "Test User")
        self.run_git("config", "user.email", "test@example.com")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    @patch("git_copilot_commit.cli.Confirm.ask", return_value=True)
    def test_execute_split_commit_plan_creates_multiple_commits(
        self, _mock_confirm
    ) -> None:
        repo = GitRepository(self.repo_path)
        file_path = self.repo_path / "file.txt"
        file_path.write_text("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n", encoding="utf-8")
        repo.stage_files(["file.txt"])
        repo.commit("init", no_verify=True)

        file_path.write_text("A\nb\nc\nd\ne\nf\ng\nh\ni\nJ\n", encoding="utf-8")
        repo.stage_files(["file.txt"])
        file_path.write_text("A\nb\nc\nd\ne\nf\ng\nhh\ni\nJ\nextra\n", encoding="utf-8")
        original_unstaged_diff = repo.get_status().unstaged_diff

        patch_units = tuple(extract_patch_units(repo.get_staged_diff(extra_args=SPLIT_DIFF_ARGS)))
        self.assertEqual(len(patch_units), 2)

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

        commit_shas = execute_split_commit_plan(repo, prepared_commits, yes=True)

        self.assertEqual(len(commit_shas), 2)
        final_status = repo.get_status()
        self.assertFalse(final_status.has_staged_changes)
        self.assertTrue(final_status.has_unstaged_changes)
        self.assertEqual(final_status.unstaged_diff, original_unstaged_diff)

        recent_messages = [message for _, message in repo.get_recent_commits(limit=2)]
        self.assertEqual(
            recent_messages,
            ["chore: update last line", "chore: update first line"],
        )


if __name__ == "__main__":
    unittest.main()
