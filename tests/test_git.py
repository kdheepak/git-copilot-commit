import subprocess
import tempfile
import unittest
from pathlib import Path

from git_copilot_commit.git import GitRepository

FIRST_PATCH = """\
diff --git a/file.txt b/file.txt
index d68dd40..4f11a6d 100644
--- a/file.txt
+++ b/file.txt
@@ -1,4 +1,4 @@
-a
+A
 b
 c
 d
"""

SECOND_PATCH = """\
diff --git a/file.txt b/file.txt
index 4f11a6d..41a082c 100644
--- a/file.txt
+++ b/file.txt
@@ -1,4 +1,4 @@
 A
 b
 c
-d
+D
"""


class GitRepositoryAlternateIndexTests(unittest.TestCase):
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

    def test_get_staged_diff_supports_extra_flags(self) -> None:
        repo = GitRepository(self.repo_path)
        file_path = self.repo_path / "file.txt"
        file_path.write_text("before\n", encoding="utf-8")
        repo.stage_files(["file.txt"])
        repo.commit("init", no_verify=True)

        file_path.write_text("after\n", encoding="utf-8")
        repo.stage_files(["file.txt"])

        diff = repo.get_staged_diff(extra_args=["--full-index"])

        self.assertIn("diff --git", diff)
        self.assertRegex(diff, r"index [0-9a-f]{40}\.\.[0-9a-f]{40}")

    def test_alternate_index_commit_preserves_real_index_and_unstaged_changes(
        self,
    ) -> None:
        repo = GitRepository(self.repo_path)
        file_path = self.repo_path / "file.txt"
        file_path.write_text("a\nb\nc\nd\n", encoding="utf-8")
        repo.stage_files(["file.txt"])
        repo.commit("init", no_verify=True)

        file_path.write_text("A\nb\nc\nD\n", encoding="utf-8")
        repo.stage_files(["file.txt"])
        file_path.write_text("A\nbb\nc\nD\nextra\n", encoding="utf-8")

        initial_status = repo.get_status()
        self.assertTrue(initial_status.has_staged_changes)
        self.assertTrue(initial_status.has_unstaged_changes)
        original_unstaged_diff = initial_status.unstaged_diff

        with repo.temporary_alternate_index() as alternate_index:
            self.assertTrue(alternate_index.path.exists())

            repo.check_patch_for_alternate_index(FIRST_PATCH, index=alternate_index)
            repo.apply_patch_to_alternate_index(FIRST_PATCH, index=alternate_index)
            first_commit = repo.commit(
                "part 1",
                env=alternate_index.env,
                no_verify=True,
            )
            self.assertEqual(len(first_commit), 40)

            after_first_commit = repo.get_status()
            self.assertTrue(after_first_commit.has_staged_changes)
            self.assertEqual(after_first_commit.unstaged_diff, original_unstaged_diff)

            repo.check_patch_for_alternate_index(SECOND_PATCH, index=alternate_index)
            repo.apply_patch_to_alternate_index(SECOND_PATCH, index=alternate_index)
            second_commit = repo.commit(
                "part 2",
                env=alternate_index.env,
                no_verify=True,
            )
            self.assertEqual(len(second_commit), 40)

        self.assertFalse(alternate_index.path.exists())

        final_status = repo.get_status()
        self.assertFalse(final_status.has_staged_changes)
        self.assertTrue(final_status.has_unstaged_changes)
        self.assertEqual(final_status.unstaged_diff, original_unstaged_diff)

        recent_messages = [message for _, message in repo.get_recent_commits(limit=2)]
        self.assertEqual(recent_messages, ["part 2", "part 1"])


if __name__ == "__main__":
    unittest.main()
