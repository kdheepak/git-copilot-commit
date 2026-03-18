import pytest
from git_copilot_commit.git import GitFile, GitStatus
from git_copilot_commit.split_commits import (
    SplitCommitLimitExceededError,
    SplitPlanningError,
    build_split_plan_prompt,
    build_status_for_patch_units,
    extract_patch_units,
    parse_split_plan_response,
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
