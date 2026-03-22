import re

import pytest  # noqa: F401

from git_copilot_commit.git import GitFile, GitRepository, GitStatus

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


def test_alternate_index_supports_unborn_head(git_repo, git_repo_path) -> None:
    file_path = git_repo_path / "README.md"
    file_path.write_text("# Title\n", encoding="utf-8")
    git_repo.stage_files(["README.md"])

    staged_diff = git_repo.get_staged_diff(
        extra_args=["--src-prefix=a/", "--dst-prefix=b/"]
    )
    assert "new file mode" in staged_diff

    with git_repo.temporary_alternate_index() as alternate_index:
        git_repo.check_patch_for_alternate_index(staged_diff, index=alternate_index)
        git_repo.apply_patch_to_alternate_index(staged_diff, index=alternate_index)
        commit_sha = git_repo.commit(
            "docs: add readme",
            env=alternate_index.env,
            no_verify=True,
        )

    assert len(commit_sha) == 40
    assert not alternate_index.path.exists()

    final_status = git_repo.get_status()
    assert not final_status.has_staged_changes
    assert not final_status.has_unstaged_changes
    assert git_repo.get_recent_commits(limit=1)[0][1] == "docs: add readme"


def test_git_file_and_status_helper_properties() -> None:
    staged = GitFile(path="staged.py", status=" ", staged_status="M")
    unstaged = GitFile(path="unstaged.py", status="M", staged_status=" ")
    untracked = GitFile(path="new.py", status="?", staged_status="?")
    status = GitStatus(
        files=[staged, unstaged, untracked],
        staged_diff="diff --git a/staged.py b/staged.py\n",
        unstaged_diff="diff --git a/unstaged.py b/unstaged.py\n",
    )

    assert staged.is_staged
    assert staged.is_modified
    assert not staged.is_untracked
    assert unstaged.is_modified
    assert untracked.is_untracked
    assert status.has_staged_changes
    assert status.has_unstaged_changes
    assert status.has_untracked_files
    assert [file.path for file in status.staged_files] == ["staged.py"]
    assert [file.path for file in status.unstaged_files] == ["unstaged.py"]
    assert [file.path for file in status.untracked_files] == ["new.py"]
    assert status.get_porcelain_output() == "M  staged.py\n M unstaged.py\n?? new.py"


def test_stage_files_from_subdirectory_stages_the_entire_repository(
    git_repo_path,
) -> None:
    repo = GitRepository(git_repo_path)
    frontend_dir = git_repo_path / "frontend"
    backend_dir = git_repo_path / "backend"
    frontend_dir.mkdir()
    backend_dir.mkdir()
    frontend_file = frontend_dir / "component.py"
    backend_file = backend_dir / "service.py"

    frontend_file.write_text("print('frontend v1')\n", encoding="utf-8")
    backend_file.write_text("print('backend v1')\n", encoding="utf-8")
    repo.stage_files()
    repo.commit("init", no_verify=True)

    frontend_file.write_text("print('frontend v2')\n", encoding="utf-8")
    backend_file.write_text("print('backend v2')\n", encoding="utf-8")

    nested_repo = GitRepository(frontend_dir)
    nested_repo.stage_files()

    status = nested_repo.get_status()

    assert [file.path for file in status.staged_files] == [
        "backend/service.py",
        "frontend/component.py",
    ]
    assert not status.has_unstaged_changes


def test_stage_modified_from_subdirectory_updates_the_entire_repository(
    git_repo_path,
) -> None:
    repo = GitRepository(git_repo_path)
    frontend_dir = git_repo_path / "frontend"
    backend_dir = git_repo_path / "backend"
    frontend_dir.mkdir()
    backend_dir.mkdir()
    frontend_file = frontend_dir / "component.py"
    backend_file = backend_dir / "service.py"

    frontend_file.write_text("print('frontend v1')\n", encoding="utf-8")
    backend_file.write_text("print('backend v1')\n", encoding="utf-8")
    repo.stage_files()
    repo.commit("init", no_verify=True)

    frontend_file.write_text("print('frontend v2')\n", encoding="utf-8")
    backend_file.write_text("print('backend v2')\n", encoding="utf-8")

    nested_repo = GitRepository(frontend_dir)
    nested_repo.stage_modified()

    status = nested_repo.get_status()

    assert [file.path for file in status.staged_files] == [
        "backend/service.py",
        "frontend/component.py",
    ]
    assert not status.has_unstaged_changes


def test_parse_status_output_build_env_and_commit_validation(git_repo) -> None:
    parsed = git_repo._parse_status_output("M  staged.py\n M unstaged.py\n?? new.py\n")

    assert [(file.staged_status, file.status, file.path) for file in parsed] == [
        ("M", " ", "staged.py"),
        (" ", "M", "unstaged.py"),
        ("?", "?", "new.py"),
    ]

    merged_env = git_repo._build_env({"CUSTOM_ENV": "1"})
    assert merged_env is not None
    assert merged_env["CUSTOM_ENV"] == "1"
    assert git_repo._build_env(None) is None

    with pytest.raises(ValueError):
        git_repo.commit(None)


def test_parse_status_output_preserves_leading_space_on_first_line(git_repo) -> None:
    parsed = git_repo._parse_status_output(" M backend/service.py\nM  frontend.py\n")

    assert [(file.staged_status, file.status, file.path) for file in parsed] == [
        (" ", "M", "backend/service.py"),
        ("M", " ", "frontend.py"),
    ]
