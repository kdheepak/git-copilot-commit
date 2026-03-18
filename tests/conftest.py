from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from git_copilot_commit.git import GitRepository


def run_git(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.fixture
def git_repo_path(tmp_path: Path) -> Path:
    run_git(tmp_path, "init", "-q")
    run_git(tmp_path, "config", "user.name", "Test User")
    run_git(tmp_path, "config", "user.email", "test@example.com")
    return tmp_path


@pytest.fixture
def git_repo(git_repo_path: Path) -> GitRepository:
    return GitRepository(git_repo_path)
