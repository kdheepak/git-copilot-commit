import re

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


def test_get_staged_diff_supports_extra_flags(git_repo, git_repo_path) -> None:
    file_path = git_repo_path / "file.txt"
    file_path.write_text("before\n", encoding="utf-8")
    git_repo.stage_files(["file.txt"])
    git_repo.commit("init", no_verify=True)

    file_path.write_text("after\n", encoding="utf-8")
    git_repo.stage_files(["file.txt"])

    diff = git_repo.get_staged_diff(extra_args=["--full-index"])

    assert "diff --git" in diff
    assert re.search(r"index [0-9a-f]{40}\.\.[0-9a-f]{40}", diff)


def test_alternate_index_commit_preserves_real_index_and_unstaged_changes(
    git_repo, git_repo_path
) -> None:
    file_path = git_repo_path / "file.txt"
    file_path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    git_repo.stage_files(["file.txt"])
    git_repo.commit("init", no_verify=True)

    file_path.write_text("A\nb\nc\nD\n", encoding="utf-8")
    git_repo.stage_files(["file.txt"])
    file_path.write_text("A\nbb\nc\nD\nextra\n", encoding="utf-8")

    initial_status = git_repo.get_status()
    assert initial_status.has_staged_changes
    assert initial_status.has_unstaged_changes
    original_unstaged_diff = initial_status.unstaged_diff

    with git_repo.temporary_alternate_index() as alternate_index:
        assert alternate_index.path.exists()

        git_repo.check_patch_for_alternate_index(FIRST_PATCH, index=alternate_index)
        git_repo.apply_patch_to_alternate_index(FIRST_PATCH, index=alternate_index)
        first_commit = git_repo.commit(
            "part 1",
            env=alternate_index.env,
            no_verify=True,
        )
        assert len(first_commit) == 40

        after_first_commit = git_repo.get_status()
        assert after_first_commit.has_staged_changes
        assert after_first_commit.unstaged_diff == original_unstaged_diff

        git_repo.check_patch_for_alternate_index(SECOND_PATCH, index=alternate_index)
        git_repo.apply_patch_to_alternate_index(SECOND_PATCH, index=alternate_index)
        second_commit = git_repo.commit(
            "part 2",
            env=alternate_index.env,
            no_verify=True,
        )
        assert len(second_commit) == 40

    assert not alternate_index.path.exists()

    final_status = git_repo.get_status()
    assert not final_status.has_staged_changes
    assert final_status.has_unstaged_changes
    assert final_status.unstaged_diff == original_unstaged_diff

    recent_messages = [message for _, message in git_repo.get_recent_commits(limit=2)]
    assert recent_messages == ["part 2", "part 1"]
