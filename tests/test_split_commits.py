import unittest

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


class SplitCommitUnitTests(unittest.TestCase):
    def test_extract_patch_units_splits_multihunk_files_and_keeps_new_file_atomic(
        self,
    ) -> None:
        units = extract_patch_units(MULTI_FILE_DIFF)

        self.assertEqual(len(units), 3)
        self.assertEqual([unit.kind for unit in units], ["hunk", "hunk", "new_file"])
        self.assertEqual([unit.path for unit in units], ["src/app.py", "src/app.py", "README.md"])
        self.assertIn("hunk 1/2", units[0].summary)
        self.assertIn("hunk 2/2", units[1].summary)
        self.assertIn("add README.md", units[2].summary)

    def test_build_status_for_patch_units_uses_synthetic_staged_state(self) -> None:
        units = extract_patch_units(MULTI_FILE_DIFF)

        status = build_status_for_patch_units((units[0], units[2]))

        self.assertTrue(status.has_staged_changes)
        self.assertFalse(status.has_unstaged_changes)
        self.assertEqual(status.get_porcelain_output(), "M  src/app.py\nA  README.md")
        self.assertIn("diff --git a/src/app.py b/src/app.py", status.staged_diff)
        self.assertIn("diff --git a/README.md b/README.md", status.staged_diff)

    def test_build_split_plan_prompt_includes_unit_details(self) -> None:
        units = extract_patch_units(MULTI_FILE_DIFF)

        prompt = build_split_plan_prompt(
            make_status(),
            units,
            max_commits=3,
            context="Keep docs separate from code",
        )

        self.assertIn("Maximum commits: 3", prompt)
        self.assertIn("Keep docs separate from code", prompt)
        self.assertIn("### u1", prompt)
        self.assertIn("Kind: hunk", prompt)
        self.assertIn("### u3", prompt)
        self.assertIn("Kind: new_file", prompt)

    def test_parse_split_plan_response_validates_complete_assignment(self) -> None:
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

        self.assertEqual(
            [commit.unit_ids for commit in plan.commits],
            [("u1", "u2"), ("u3",)],
        )

    def test_parse_split_plan_response_rejects_duplicate_or_missing_units(self) -> None:
        units = extract_patch_units(MULTI_FILE_DIFF)

        with self.assertRaises(SplitPlanningError):
            parse_split_plan_response(
                '{"commits":[{"unit_ids":["u1","u1"]},{"unit_ids":["u3"]}]}',
                units,
                max_commits=3,
            )

    def test_parse_split_plan_response_raises_limit_error_with_validated_plan(self) -> None:
        units = extract_patch_units(MULTI_FILE_DIFF)

        with self.assertRaises(SplitCommitLimitExceededError) as context:
            parse_split_plan_response(
                '{"commits":[{"unit_ids":["u1"]},{"unit_ids":["u2"]},{"unit_ids":["u3"]}]}',
                units,
                max_commits=2,
            )

        self.assertEqual(context.exception.actual_commits, 3)
        self.assertEqual(context.exception.max_commits, 2)
        self.assertEqual(
            [commit.unit_ids for commit in context.exception.plan.commits],
            [("u1",), ("u2",), ("u3",)],
        )

        with self.assertRaises(SplitPlanningError):
            parse_split_plan_response(
                '{"commits":[{"unit_ids":["u1"]},{"unit_ids":["u3"]}]}',
                units,
                max_commits=3,
            )


if __name__ == "__main__":
    unittest.main()
