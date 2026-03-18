import pytest
from git_copilot_commit.git import GitFile, GitStatus
from git_copilot_commit.split_commits import (
    FilePatch,
    PatchUnit,
    SplitCommitLimitExceededError,
    SplitPlanningError,
    build_split_plan_prompt,
    build_status_for_patch_units,
    classify_file_patch,
    count_patch_changes,
    evaluate_auto_split,
    extract_patch_units,
    find_duplicates,
    group_patch_units,
    parse_diff_paths,
    parse_file_patch,
    parse_json_payload,
    parse_split_plan_response,
    should_split_file_patch,
    summarize_file_patch,
    summarize_hunk,
    SplitCommitPlan,
    SplitPlanCommit,
)

MULTI_FILE_DIFF = """\
diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1,4 +1,4 @@
-old
+new
 keep
 same
 lines
@@ -10,4 +10,4 @@ keep
 more
 context
 here
-tail
+done
diff --git a/README.md b/README.md
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/README.md
@@ -0,0 +1,2 @@
+# Title
+Text
"""


def make_status() -> GitStatus:
    return GitStatus(
        files=[
            GitFile(path="src/app.py", status=" ", staged_status="M"),
            GitFile(path="README.md", status=" ", staged_status="A"),
        ],
        staged_diff=MULTI_FILE_DIFF,
        unstaged_diff="",
    )


def test_extract_patch_units_splits_multihunk_files_and_keeps_new_file_atomic() -> None:
    units = extract_patch_units(MULTI_FILE_DIFF)

    assert len(units) == 3
    assert [unit.kind for unit in units] == ["hunk", "hunk", "new_file"]
    assert [unit.path for unit in units] == ["src/app.py", "src/app.py", "README.md"]
    assert "hunk 1/2" in units[0].summary
    assert "hunk 2/2" in units[1].summary
    assert "add README.md" in units[2].summary


def test_build_status_for_patch_units_uses_synthetic_staged_state() -> None:
    units = extract_patch_units(MULTI_FILE_DIFF)

    status = build_status_for_patch_units((units[0], units[2]))

    assert status.has_staged_changes
    assert not status.has_unstaged_changes
    assert status.get_porcelain_output() == "M  src/app.py\nA  README.md"
    assert "diff --git a/src/app.py b/src/app.py" in status.staged_diff
    assert "diff --git a/README.md b/README.md" in status.staged_diff


def test_build_split_plan_prompt_includes_unit_details() -> None:
    units = extract_patch_units(MULTI_FILE_DIFF)

    prompt = build_split_plan_prompt(
        make_status(),
        units,
        max_commits=3,
        context="Keep docs separate from code",
    )

    assert "Maximum commits: 3" in prompt
    assert "Keep docs separate from code" in prompt
    assert "### u1" in prompt
    assert "Kind: hunk" in prompt
    assert "### u3" in prompt
    assert "Kind: new_file" in prompt


def test_build_split_plan_prompt_supports_preferred_commit_count() -> None:
    units = extract_patch_units(MULTI_FILE_DIFF)

    prompt = build_split_plan_prompt(
        make_status(),
        units,
        max_commits=2,
        preferred_commits=2,
    )

    assert "Preferred commits: 2" in prompt
    assert "Maximum commits: 2" in prompt


def test_parse_split_plan_response_validates_complete_assignment() -> None:
    units = extract_patch_units(MULTI_FILE_DIFF)

    plan = parse_split_plan_response(
        """
        {
          "commits": [
            { "unit_ids": ["u2", "u1"] },
            { "unit_ids": ["u3"] }
          ]
        }
        """,
        units,
        max_commits=3,
    )

    assert [commit.unit_ids for commit in plan.commits] == [("u1", "u2"), ("u3",)]


def test_parse_split_plan_response_rejects_duplicate_or_missing_units() -> None:
    units = extract_patch_units(MULTI_FILE_DIFF)

    with pytest.raises(SplitPlanningError):
        parse_split_plan_response(
            '{"commits":[{"unit_ids":["u1","u1"]},{"unit_ids":["u3"]}]}',
            units,
            max_commits=3,
        )

    with pytest.raises(SplitPlanningError):
        parse_split_plan_response(
            '{"commits":[{"unit_ids":["u1"]},{"unit_ids":["u3"]}]}',
            units,
            max_commits=3,
        )


def test_parse_split_plan_response_raises_limit_error_with_validated_plan() -> None:
    units = extract_patch_units(MULTI_FILE_DIFF)

    with pytest.raises(SplitCommitLimitExceededError) as context:
        parse_split_plan_response(
            '{"commits":[{"unit_ids":["u1"]},{"unit_ids":["u2"]},{"unit_ids":["u3"]}]}',
            units,
            max_commits=2,
        )

    assert context.value.actual_commits == 3
    assert context.value.max_commits == 2
    assert [commit.unit_ids for commit in context.value.plan.commits] == [
        ("u1",),
        ("u2",),
        ("u3",),
    ]


def test_parse_split_plan_response_allows_fewer_than_preferred_commits() -> None:
    units = extract_patch_units(MULTI_FILE_DIFF)

    plan = parse_split_plan_response(
        '{"commits":[{"unit_ids":["u1","u2"]},{"unit_ids":["u3"]}]}',
        units,
        max_commits=3,
    )

    assert [commit.unit_ids for commit in plan.commits] == [("u1", "u2"), ("u3",)]


def test_parse_json_payload_supports_code_fences_and_embedded_objects() -> None:
    fenced = parse_json_payload(
        """```json
        {"commits": [{"unit_ids": ["u1"]}]}
        ```"""
    )
    embedded = parse_json_payload(
        'Planner output:\n{"commits":[{"unit_ids":["u1","u2"]}]}\nThanks.'
    )

    assert fenced == {"commits": [{"unit_ids": ["u1"]}]}
    assert embedded == {"commits": [{"unit_ids": ["u1", "u2"]}]}

    with pytest.raises(SplitPlanningError):
        parse_json_payload("")


def test_parse_diff_paths_and_parse_file_patch_support_quoted_paths() -> None:
    header = 'diff --git "a/docs/my file.md" "b/docs/my file.md"'

    assert parse_diff_paths(header) == ("docs/my file.md", "docs/my file.md")

    file_patch = parse_file_patch(
        """\
diff --git "a/docs/my file.md" "b/docs/my file.md"
index 1111111..2222222 100644
--- "a/docs/my file.md"
+++ "b/docs/my file.md"
@@ -1 +1 @@
-old
+new
"""
    )
    assert file_patch.display_path == "docs/my file.md"
    assert file_patch.kind == "file"
    assert len(file_patch.hunks) == 1
    assert not should_split_file_patch(file_patch)

    with pytest.raises(SplitPlanningError):
        parse_diff_paths("diff --git only-one-path")


def test_classify_and_summarize_special_patch_kinds() -> None:
    assert classify_file_patch(
        ["diff --git a/file.bin b/file.bin\n", "GIT binary patch\n"],
        "file.bin",
        "file.bin",
        [],
    ) == ("M", "binary")
    assert classify_file_patch(
        ["rename from old.txt\n", "rename to new.txt\n"],
        "old.txt",
        "new.txt",
        [],
    ) == ("R", "rename")
    assert classify_file_patch(
        ["old mode 100644\n", "new mode 100755\n"],
        "script.sh",
        "script.sh",
        [],
    ) == ("M", "mode_change")
    assert classify_file_patch(
        ["diff --git a/file.txt b/file.txt\n"],
        "file.txt",
        "file.txt",
        [],
    ) == ("M", "metadata")

    rename_patch = FilePatch(
        old_path="old.txt",
        new_path="new.txt",
        display_path="new.txt",
        staged_status="R",
        kind="rename",
        raw_patch="diff --git a/old.txt b/new.txt\n",
        header="diff --git a/old.txt b/new.txt\n",
        hunks=(),
    )
    mode_patch = FilePatch(
        old_path="script.sh",
        new_path="script.sh",
        display_path="script.sh",
        staged_status="M",
        kind="mode_change",
        raw_patch="diff --git a/script.sh b/script.sh\nold mode 100644\nnew mode 100755\n",
        header="diff --git a/script.sh b/script.sh\n",
        hunks=(),
    )

    assert summarize_file_patch(rename_patch) == "rename old.txt -> new.txt"
    assert summarize_file_patch(mode_patch) == "mode change script.sh"
    assert summarize_hunk(
        "src/app.py",
        2,
        3,
        "@@ -1 +1 @@\n-old\n+new\n",
    ) == "src/app.py hunk 2/3 @@ -1 +1 @@ (+1/-1)"


def test_count_patch_changes_find_duplicates_and_group_patch_units() -> None:
    assert count_patch_changes(
        """\
--- a/file.txt
+++ b/file.txt
+added
-removed
 context
"""
    ) == (1, 1)
    assert find_duplicates(["u1", "u2", "u1", "u3", "u2"]) == {"u1", "u2"}

    units = extract_patch_units(MULTI_FILE_DIFF)
    grouped = group_patch_units(
        units,
        SplitCommitPlan(
            commits=(
                SplitPlanCommit(("u3",)),
                SplitPlanCommit(("u1", "u2")),
            )
        ),
    )

    assert [unit.id for unit in grouped[0]] == ["u3"]
    assert [unit.id for unit in grouped[1]] == ["u1", "u2"]


def test_build_split_plan_prompt_requires_patch_units() -> None:
    with pytest.raises(SplitPlanningError):
        build_split_plan_prompt(make_status(), [], max_commits=2)


def test_evaluate_auto_split_is_conservative_for_small_similar_changes() -> None:
    units = (
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
            path="src/app.py",
            staged_status="M",
            kind="hunk",
            patch="patch 2",
            summary="summary 2",
        ),
    )

    should_split, reason = evaluate_auto_split(units)

    assert not should_split
    assert reason == "found only 2 similar patch units"


def test_evaluate_auto_split_detects_large_or_mixed_changes() -> None:
    many_units = extract_patch_units(MULTI_FILE_DIFF)
    should_split_many, reason_many = evaluate_auto_split(many_units)
    assert should_split_many
    assert reason_many == "found 3 independent patch units"

    mixed_units = (
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
    should_split_mixed, reason_mixed = evaluate_auto_split(mixed_units)
    assert should_split_mixed
    assert reason_mixed == "found mixed change kinds (hunk, new_file)"
