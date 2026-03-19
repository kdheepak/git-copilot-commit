from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, cast

from .git import GitFile, GitStatus

DIFF_HEADER_PREFIX = "diff --git "
AUTO_SPLIT_UNIT_THRESHOLD = 3
AUTO_SPLIT_PATH_THRESHOLD = 4


class SplitPlanningError(ValueError):
    """Raised when a split-commit plan cannot be built or validated."""


@dataclass(frozen=True, slots=True)
class FilePatch:
    """Structured representation of a single file-level diff patch."""

    old_path: str
    new_path: str
    display_path: str
    staged_status: str
    kind: str
    raw_patch: str
    header: str
    hunks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PatchUnit:
    """A patch fragment that can be assigned to a planned commit."""

    id: str
    order: int
    path: str
    staged_status: str
    kind: str
    patch: str
    summary: str


@dataclass(frozen=True, slots=True)
class SplitPlanCommit:
    """A validated planned commit represented by ordered patch-unit ids."""

    unit_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SplitCommitPlan:
    """A validated split-commit plan."""

    commits: tuple[SplitPlanCommit, ...]


def evaluate_auto_split(patch_units: Sequence[PatchUnit]) -> tuple[bool, str]:
    """Return whether automatic split planning should run for the patch set."""
    if len(patch_units) < 2:
        return False, "fewer than two independent patch units were found"

    kinds = {unit.kind for unit in patch_units}
    staged_statuses = {unit.staged_status for unit in patch_units}
    paths = {unit.path for unit in patch_units}

    if len(patch_units) >= AUTO_SPLIT_UNIT_THRESHOLD:
        return True, f"found {len(patch_units)} independent patch units"

    if len(kinds) >= 2:
        return True, f"found mixed change kinds ({', '.join(sorted(kinds))})"

    if len(staged_statuses) >= 2:
        return True, (
            f"found mixed staged change types ({', '.join(sorted(staged_statuses))})"
        )

    if len(paths) >= AUTO_SPLIT_PATH_THRESHOLD:
        return True, f"found changes across {len(paths)} files"

    return False, f"found only {len(patch_units)} similar patch units"


def extract_patch_units(diff_text: str) -> list[PatchUnit]:
    """Convert a staged diff into patch units suitable for split planning."""
    units: list[PatchUnit] = []

    for file_patch in parse_file_patches(diff_text):
        if should_split_file_patch(file_patch):
            for hunk_index, hunk in enumerate(file_patch.hunks, start=1):
                patch = f"{file_patch.header}{hunk}"
                units.append(
                    PatchUnit(
                        id=f"u{len(units) + 1}",
                        order=len(units),
                        path=file_patch.display_path,
                        staged_status=file_patch.staged_status,
                        kind="hunk",
                        patch=patch,
                        summary=summarize_hunk(
                            file_patch.display_path,
                            hunk_index,
                            len(file_patch.hunks),
                            hunk,
                        ),
                    )
                )
            continue

        units.append(
            PatchUnit(
                id=f"u{len(units) + 1}",
                order=len(units),
                path=file_patch.display_path,
                staged_status=file_patch.staged_status,
                kind=file_patch.kind,
                patch=file_patch.raw_patch,
                summary=summarize_file_patch(file_patch),
            )
        )

    return units


def parse_file_patches(diff_text: str) -> list[FilePatch]:
    """Parse a unified diff into per-file patch structures."""
    if not diff_text.strip():
        return []

    file_blocks: list[str] = []
    current_block: list[str] = []

    for line in diff_text.splitlines(keepends=True):
        if line.startswith(DIFF_HEADER_PREFIX):
            if current_block:
                file_blocks.append("".join(current_block))
            current_block = [line]
            continue

        if current_block:
            current_block.append(line)

    if current_block:
        file_blocks.append("".join(current_block))

    return [parse_file_patch(block) for block in file_blocks]


def parse_file_patch(raw_patch: str) -> FilePatch:
    """Parse a single file-level patch block."""
    lines = raw_patch.splitlines(keepends=True)
    if not lines or not lines[0].startswith(DIFF_HEADER_PREFIX):
        raise SplitPlanningError("Diff block is missing a `diff --git` header.")

    old_path, new_path = parse_diff_paths(lines[0].rstrip("\n"))
    display_path = new_path if new_path != "/dev/null" else old_path

    header_lines: list[str] = []
    hunks: list[str] = []
    current_hunk: list[str] | None = None

    for line in lines:
        if line.startswith("@@ "):
            if current_hunk is not None:
                hunks.append("".join(current_hunk))
            current_hunk = [line]
            continue

        if current_hunk is not None:
            current_hunk.append(line)
        else:
            header_lines.append(line)

    if current_hunk is not None:
        hunks.append("".join(current_hunk))

    staged_status, kind = classify_file_patch(lines, old_path, new_path, hunks)

    return FilePatch(
        old_path=old_path,
        new_path=new_path,
        display_path=display_path,
        staged_status=staged_status,
        kind=kind,
        raw_patch=raw_patch,
        header="".join(header_lines),
        hunks=tuple(hunks),
    )


def should_split_file_patch(file_patch: FilePatch) -> bool:
    """Return whether a file patch should be split into per-hunk units."""
    return file_patch.kind == "file" and len(file_patch.hunks) > 1


def build_split_plan_prompt(
    status: GitStatus,
    patch_units: Sequence[PatchUnit],
    *,
    preferred_commits: int | None = None,
    context: str = "",
) -> str:
    """Build the user prompt used to request a split-commit plan."""
    if not patch_units:
        raise SplitPlanningError(
            "Cannot plan split commits without staged patch units."
        )

    prompt_parts = []
    if context.strip():
        prompt_parts.append(f"User-provided context:\n\n{context.strip()}\n")

    if preferred_commits is not None:
        prompt_parts.append(f"Preferred commits: {preferred_commits}")

    prompt_parts.extend(
        [
            "",
            "`git status`:",
            f"```\n{status.get_porcelain_output()}\n```",
            "",
            "Patch units:",
        ]
    )

    for unit in patch_units:
        prompt_parts.extend(
            [
                f"### {unit.id}",
                f"Path: {unit.path}",
                f"Kind: {unit.kind}",
                f"Summary: {unit.summary}",
                "Patch:",
                f"```diff\n{unit.patch}\n```",
            ]
        )

    return "\n".join(prompt_parts)


def parse_split_plan_response(
    response_text: str,
    patch_units: Sequence[PatchUnit],
) -> SplitCommitPlan:
    """Parse and validate the planner model's JSON response."""
    payload = parse_json_payload(response_text)

    commits_data: Any
    if isinstance(payload, dict):
        commits_data = payload.get("commits")
    else:
        commits_data = payload

    if not isinstance(commits_data, list) or not commits_data:
        raise SplitPlanningError(
            "Planner response did not include a non-empty commits list."
        )

    units_by_id = {unit.id: unit for unit in patch_units}
    expected_ids = set(units_by_id)
    assigned_ids: list[str] = []
    validated_commits: list[SplitPlanCommit] = []

    for index, commit_data in enumerate(commits_data, start=1):
        if not isinstance(commit_data, dict):
            raise SplitPlanningError(f"Commit {index} in the plan was not an object.")

        commit_object = cast(dict[str, Any], commit_data)
        raw_unit_ids = commit_object.get("unit_ids")
        if not isinstance(raw_unit_ids, list) or not raw_unit_ids:
            raise SplitPlanningError(
                f"Commit {index} must contain a non-empty `unit_ids` list."
            )

        unit_ids: list[str] = []
        seen_in_commit: set[str] = set()
        for unit_id in raw_unit_ids:
            if not isinstance(unit_id, str) or not unit_id:
                raise SplitPlanningError(
                    f"Commit {index} contained an invalid patch-unit id: {unit_id!r}."
                )
            if unit_id not in units_by_id:
                raise SplitPlanningError(
                    f"Commit {index} referenced unknown patch unit `{unit_id}`."
                )
            if unit_id in seen_in_commit:
                raise SplitPlanningError(
                    f"Commit {index} referenced patch unit `{unit_id}` more than once."
                )
            seen_in_commit.add(unit_id)
            unit_ids.append(unit_id)

        ordered_unit_ids = tuple(
            sorted(unit_ids, key=lambda unit_id: units_by_id[unit_id].order)
        )
        validated_commits.append(SplitPlanCommit(unit_ids=ordered_unit_ids))
        assigned_ids.extend(unit_ids)

    if set(assigned_ids) != expected_ids or len(assigned_ids) != len(expected_ids):
        missing_ids = sorted(expected_ids - set(assigned_ids))
        extra_ids = sorted(set(assigned_ids) - expected_ids)
        duplicates = sorted(find_duplicates(assigned_ids))
        details = []
        if missing_ids:
            details.append(f"missing: {', '.join(missing_ids)}")
        if extra_ids:
            details.append(f"unknown: {', '.join(extra_ids)}")
        if duplicates:
            details.append(f"duplicated: {', '.join(duplicates)}")
        detail_text = "; ".join(details) if details else "unit assignment mismatch"
        raise SplitPlanningError(
            f"Planner must assign each patch unit exactly once ({detail_text})."
        )

    plan = SplitCommitPlan(commits=tuple(validated_commits))

    return plan


def group_patch_units(
    patch_units: Sequence[PatchUnit], plan: SplitCommitPlan
) -> tuple[tuple[PatchUnit, ...], ...]:
    """Resolve a plan's unit ids into ordered patch-unit groups."""
    units_by_id = {unit.id: unit for unit in patch_units}
    return tuple(
        tuple(units_by_id[unit_id] for unit_id in commit.unit_ids)
        for commit in plan.commits
    )


def build_status_for_patch_units(patch_units: Sequence[PatchUnit]) -> GitStatus:
    """Create a synthetic staged status for a patch-unit group."""
    ordered_units = sorted(patch_units, key=lambda unit: unit.order)
    files: list[GitFile] = []
    seen_paths: set[tuple[str, str]] = set()

    for unit in ordered_units:
        file_key = (unit.path, unit.staged_status)
        if file_key in seen_paths:
            continue
        seen_paths.add(file_key)
        files.append(
            GitFile(path=unit.path, status=" ", staged_status=unit.staged_status)
        )

    return GitStatus(
        files=files,
        staged_diff="".join(unit.patch for unit in ordered_units),
        unstaged_diff="",
    )


def parse_json_payload(response_text: str) -> Any:
    """Parse a JSON object from a model response that may include code fences."""
    stripped = response_text.strip()
    if not stripped:
        raise SplitPlanningError("Planner response was empty.")

    candidates = [stripped]

    if stripped.startswith("```"):
        fence_lines = stripped.splitlines()
        if len(fence_lines) >= 3 and fence_lines[-1].startswith("```"):
            candidates.append("\n".join(fence_lines[1:-1]).strip())

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise SplitPlanningError("Planner response did not contain valid JSON.")


def parse_diff_paths(diff_header_line: str) -> tuple[str, str]:
    """Parse `diff --git` paths, including quoted filenames with spaces."""
    try:
        parts = shlex.split(diff_header_line)
    except ValueError as exc:
        raise SplitPlanningError(
            f"Could not parse diff header: {diff_header_line}"
        ) from exc

    if len(parts) < 4:
        raise SplitPlanningError(f"Unexpected diff header: {diff_header_line}")

    return strip_diff_prefix(parts[2]), strip_diff_prefix(parts[3])


def strip_diff_prefix(path: str) -> str:
    """Strip the standard `a/` or `b/` prefixes from diff paths."""
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def classify_file_patch(
    lines: Sequence[str],
    old_path: str,
    new_path: str,
    hunks: Sequence[str],
) -> tuple[str, str]:
    """Infer the staged status and planning kind for a file patch."""
    if any(
        line.startswith("GIT binary patch") or line.startswith("Binary files ")
        for line in lines
    ):
        return "M", "binary"

    if any(
        line.startswith("rename from ") or line.startswith("rename to ")
        for line in lines
    ):
        return "R", "rename"

    if any(line.startswith("new file mode ") for line in lines):
        return "A", "new_file"

    if any(line.startswith("deleted file mode ") for line in lines):
        return "D", "deleted_file"

    if any(
        line.startswith("old mode ") or line.startswith("new mode ") for line in lines
    ):
        return "M", "mode_change"

    if old_path == "/dev/null":
        return "A", "new_file"
    if new_path == "/dev/null":
        return "D", "deleted_file"

    if hunks:
        return "M", "file"

    return "M", "metadata"


def summarize_file_patch(file_patch: FilePatch) -> str:
    """Build a concise human-readable summary for a file-level patch unit."""
    additions, deletions = count_patch_changes(file_patch.raw_patch)
    kind = file_patch.kind
    path = file_patch.display_path

    if kind == "rename":
        return f"rename {file_patch.old_path} -> {file_patch.new_path}"
    if kind == "binary":
        return f"binary update {path}"
    if kind == "new_file":
        return f"add {path} (+{additions}/-{deletions})"
    if kind == "deleted_file":
        return f"delete {path} (+{additions}/-{deletions})"
    if kind == "mode_change":
        return f"mode change {path}"
    if kind == "metadata":
        return f"metadata update {path}"
    return f"update {path} (+{additions}/-{deletions})"


def summarize_hunk(path: str, index: int, total: int, hunk: str) -> str:
    """Build a concise human-readable summary for a per-hunk patch unit."""
    additions, deletions = count_patch_changes(hunk)
    header = hunk.splitlines()[0] if hunk.splitlines() else "hunk"
    return f"{path} hunk {index}/{total} {header} (+{additions}/-{deletions})"


def count_patch_changes(text: str) -> tuple[int, int]:
    """Count added and deleted lines in a unified diff fragment."""
    additions = 0
    deletions = 0

    for line in text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1

    return additions, deletions


def find_duplicates(values: Iterable[str]) -> set[str]:
    """Return the duplicated strings from an iterable."""
    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
            continue
        seen.add(value)

    return duplicates
