import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Tuple


class GitError(Exception):
    """Base exception for git-related errors."""

    pass


class NotAGitRepositoryError(GitError):
    """Raised when not in a git repository."""

    pass


class GitCommandError(GitError):
    """Raised when a git command fails."""

    pass


@dataclass(frozen=True, slots=True)
class AlternateGitIndex:
    """Represents a temporary git index file."""

    path: Path

    @property
    def env(self) -> dict[str, str]:
        """Environment variables needed to use this index."""
        return {"GIT_INDEX_FILE": str(self.path)}


@dataclass
class GitFile:
    """Represents a file in git status."""

    path: str
    status: str
    staged_status: str

    @property
    def is_staged(self) -> bool:
        """Check if file has staged changes."""
        return self.staged_status != " " and self.staged_status != "?"

    @property
    def is_modified(self) -> bool:
        """Check if file is modified."""
        return self.status == "M" or self.staged_status == "M"

    @property
    def is_untracked(self) -> bool:
        """Check if file is untracked."""
        return self.staged_status == "?" and self.status == "?"


@dataclass
class GitStatus:
    """Structured representation of git status."""

    files: list[GitFile]
    staged_diff: str
    unstaged_diff: str

    @property
    def has_staged_changes(self) -> bool:
        """Check if there are any staged changes."""
        return bool(self.staged_diff.strip())

    @property
    def has_unstaged_changes(self) -> bool:
        """Check if there are any unstaged changes."""
        return bool(self.unstaged_diff.strip())

    @property
    def has_untracked_files(self) -> bool:
        """Check if there are any untracked files."""
        return any(f.is_untracked for f in self.files)

    @property
    def staged_files(self) -> list[GitFile]:
        """Get list of files with staged changes."""
        return [f for f in self.files if f.is_staged]

    @property
    def unstaged_files(self) -> list[GitFile]:
        """Get list of files with unstaged changes."""
        return [f for f in self.files if not f.is_staged and not f.is_untracked]

    @property
    def untracked_files(self) -> list[GitFile]:
        """Get list of untracked files."""
        return [f for f in self.files if f.is_untracked]

    def get_porcelain_output(self) -> str:
        """Get the original porcelain output format."""
        lines = []
        for file in self.files:
            lines.append(f"{file.staged_status}{file.status} {file.path}")
        return "\n".join(lines)


class GitRepository:
    """Encapsulates git repository operations."""

    def __init__(self, repo_path: Path | None = None, timeout: int = 30):
        """
        Initialize GitRepository.

        Args:
            repo_path: Path to git repository. Defaults to current directory.
            timeout: Timeout for git commands in seconds.

        Raises:
            NotAGitRepositoryError: If the path is not a git repository.
        """
        self.cwd = (repo_path or Path.cwd()).resolve()
        self.timeout = timeout
        self.repo_path = self._resolve_repo_root()

    def _resolve_repo_root(self) -> Path:
        """Resolve and cache the repository top-level path."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=True,
            )
        except subprocess.CalledProcessError:
            raise NotAGitRepositoryError(f"{self.cwd} is not a git repository")
        except subprocess.TimeoutExpired:
            raise GitCommandError(
                "Git command timed out: git rev-parse --show-toplevel"
            )

        repo_root = result.stdout.strip()
        if not repo_root:
            raise NotAGitRepositoryError(f"{self.cwd} is not a git repository")

        return Path(repo_root)

    def _run_git_command(
        self,
        args: list[str],
        check: bool = True,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Run a git command and return the result.

        Args:
            args: Git command arguments (without 'git' prefix).
            check: Whether to raise exception on non-zero exit code.
            env: Environment variables to merge into the git process.
            input_text: Optional text piped to stdin.
            capture_output: Whether to capture stdout/stderr.

        Returns:
            CompletedProcess instance.

        Raises:
            GitCommandError: If command fails and check=True.
        """
        cmd = ["git"] + args
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=capture_output,
                text=True,
                timeout=self.timeout,
                check=check,
                env=self._build_env(env),
                input=input_text,
            )
            return result
        except subprocess.CalledProcessError as e:
            error_output = e.stderr or e.stdout or ""
            if error_output:
                raise GitCommandError(
                    f"Git command failed: {' '.join(cmd)}\n{error_output}"
                )
            raise GitCommandError(f"Git command failed: {' '.join(cmd)}")
        except subprocess.TimeoutExpired:
            raise GitCommandError(f"Git command timed out: {' '.join(cmd)}")

    def _build_env(self, env: Mapping[str, str] | None = None) -> dict[str, str] | None:
        """Merge extra environment variables into the current process environment."""
        if env is None:
            return None

        merged_env = os.environ.copy()
        merged_env.update(env)
        return merged_env

    def _normalize_paths(self, paths: list[str]) -> list[str]:
        """Normalize user paths relative to the repository root."""
        normalized_paths: list[str] = []
        for path in paths:
            path_obj = Path(path)
            if path_obj.is_absolute():
                normalized_paths.append(str(path_obj))
                continue

            normalized_paths.append(
                os.path.relpath(self.cwd / path_obj, start=self.repo_path)
            )

        return normalized_paths

    def get_status(self, env: Mapping[str, str] | None = None) -> GitStatus:
        """
        Get comprehensive git status information.

        Returns:
            GitStatus object with all status information.
        """
        # Get porcelain status
        status_result = self._run_git_command(["status", "--porcelain"], env=env)

        # Parse status output into GitFile objects
        files = self._parse_status_output(status_result.stdout)

        return GitStatus(
            files=files,
            staged_diff=self.get_staged_diff(env=env),
            unstaged_diff=self.get_unstaged_diff(env=env),
        )

    def get_staged_diff(
        self,
        extra_args: list[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        """Get the staged diff, optionally with extra diff flags."""
        args = ["diff", "--staged"]
        if extra_args:
            args.extend(extra_args)
        return self._run_git_command(args, env=env).stdout

    def get_unstaged_diff(
        self,
        extra_args: list[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        """Get the unstaged diff, optionally with extra diff flags."""
        args = ["diff"]
        if extra_args:
            args.extend(extra_args)
        return self._run_git_command(args, env=env).stdout

    def get_head_sha(self, ref: str = "HEAD") -> str:
        """Resolve a git ref to a commit SHA."""
        result = self._run_git_command(["rev-parse", ref])
        return result.stdout.strip()

    def has_commit(self, ref: str = "HEAD") -> bool:
        """Return whether the provided ref resolves to a commit."""
        result = self._run_git_command(
            ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            check=False,
        )
        return result.returncode == 0

    def get_symbolic_head_ref(self) -> str | None:
        """Return the symbolic ref for HEAD when attached to a branch."""
        result = self._run_git_command(["symbolic-ref", "-q", "HEAD"], check=False)
        if result.returncode != 0:
            return None

        ref = result.stdout.strip()
        return ref or None

    def _parse_status_output(self, status_output: str) -> list[GitFile]:
        """Parse git status --porcelain output into GitFile objects."""
        files = []
        for line in status_output.splitlines():
            if not line:
                continue

            # Git status format: XY filename
            # X = staged status, Y = unstaged status
            if len(line) < 3:
                continue

            staged_status = line[0]
            unstaged_status = line[1]
            filename = line[3:]  # Skip the space

            files.append(
                GitFile(
                    path=filename, status=unstaged_status, staged_status=staged_status
                )
            )

        return files

    def stage_files(self, paths: list[str] | None = None) -> None:
        """
        Stage files for commit.

        Args:
          paths: List of file paths to stage. If None, stages all files.
        """
        if paths is None:
            self._run_git_command(["add", "--all"])
        else:
            self._run_git_command(["add"] + self._normalize_paths(paths))

    def stage_modified(self) -> None:
        """Stage all modified tracked files."""
        self._run_git_command(["add", "--update"])

    def unstage_files(self, paths: list[str] | None = None) -> None:
        """
        Unstage files.

        Args:
            paths: List of file paths to unstage. If None, unstages all files.
        """
        if paths is None:
            self._run_git_command(["reset", "HEAD"])
        else:
            self._run_git_command(["reset", "HEAD"] + self._normalize_paths(paths))

    def soft_reset(self, ref: str) -> None:
        """Move HEAD to ref while preserving the working tree and index."""
        self._run_git_command(["reset", "--soft", ref])

    def delete_ref(self, ref: str, *, missing_ok: bool = False) -> None:
        """Delete a ref, optionally ignoring missing refs."""
        result = self._run_git_command(["update-ref", "-d", ref], check=False)
        if result.returncode == 0 or missing_ok:
            return

        error_output = result.stderr or result.stdout or ""
        if error_output:
            raise GitCommandError(
                f"Git command failed: git update-ref -d {ref}\n{error_output}"
            )
        raise GitCommandError(f"Git command failed: git update-ref -d {ref}")

    def create_alternate_index(self, from_ref: str = "HEAD") -> AlternateGitIndex:
        """Create a temporary git index initialized from the provided ref."""
        fd, index_path = tempfile.mkstemp(prefix="git-copilot-commit-", suffix=".index")
        os.close(fd)
        alternate_index = AlternateGitIndex(Path(index_path))
        if from_ref == "HEAD" and not self.has_commit(from_ref):
            self.read_empty_tree(index=alternate_index)
        else:
            self.read_tree(from_ref, index=alternate_index)
        return alternate_index

    @contextmanager
    def temporary_alternate_index(
        self, from_ref: str = "HEAD"
    ) -> Iterator[AlternateGitIndex]:
        """Yield a temporary alternate index and delete it afterwards."""
        alternate_index = self.create_alternate_index(from_ref=from_ref)
        try:
            yield alternate_index
        finally:
            alternate_index.path.unlink(missing_ok=True)

    def read_tree(self, ref: str, *, index: AlternateGitIndex) -> None:
        """Populate an alternate index from the provided ref."""
        self._run_git_command(["read-tree", ref], env=index.env)

    def read_empty_tree(self, *, index: AlternateGitIndex) -> None:
        """Initialize an alternate index with an empty tree."""
        self._run_git_command(["read-tree", "--empty"], env=index.env)

    def apply_patch(
        self,
        patch: str,
        *,
        cached: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Apply a patch, optionally to the index only."""
        args = ["apply"]
        if cached:
            args.append("--cached")
        args.append("-")
        self._run_git_command(args, env=env, input_text=patch)

    def check_patch(
        self,
        patch: str,
        *,
        cached: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Validate whether a patch can be applied."""
        args = ["apply"]
        if cached:
            args.append("--cached")
        args.extend(["--check", "-"])
        self._run_git_command(args, env=env, input_text=patch)

    def apply_patch_to_alternate_index(
        self, patch: str, *, index: AlternateGitIndex
    ) -> None:
        """Apply a cached patch to an alternate index."""
        self.apply_patch(patch, cached=True, env=index.env)

    def check_patch_for_alternate_index(
        self, patch: str, *, index: AlternateGitIndex
    ) -> None:
        """Validate whether a cached patch can be applied to an alternate index."""
        self.check_patch(patch, cached=True, env=index.env)

    def commit(
        self,
        message: str | None = None,
        use_editor: bool = False,
        no_verify: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> str:
        """
        Create a commit with the given message or using git's editor.

        Args:
            message: Commit message. Used as template if use_editor is True.
            use_editor: Whether to use git's configured editor.
            no_verify: Skip pre-commit and commit-msg hooks (git commit -n).

        Returns:
            Commit SHA.

        Raises:
            GitCommandError: If commit fails.
        """
        if use_editor:
            # Create temp file with message as starting point
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                if message:
                    f.write(message)
                temp_file = f.name

            try:
                args = ["commit"]
                if no_verify:
                    args.append("-n")
                args.extend(["-e", "-F", temp_file])

                # Run interactively without capturing output
                cmd = ["git"] + args
                subprocess.run(
                    cmd,
                    cwd=self.repo_path,
                    timeout=self.timeout,
                    check=True,
                    env=self._build_env(env),
                )
            except subprocess.CalledProcessError:
                raise GitCommandError(f"Git commit failed: {' '.join(cmd)}")
            except subprocess.TimeoutExpired:
                raise GitCommandError(f"Git commit timed out: {' '.join(cmd)}")
            finally:
                # Clean up temp file
                os.unlink(temp_file)
        else:
            if message is None:
                raise ValueError("message is required when use_editor is False")
            args = ["commit"]
            if no_verify:
                args.append("-n")
            args.extend(["-m", message])

            self._run_git_command(args, env=env)

        # Extract commit SHA from output
        return self.get_head_sha()

    def get_recent_commits(self, limit: int = 10) -> list[Tuple[str, str]]:
        """
        Get recent commit history.

        Args:
            limit: Number of commits to retrieve.

        Returns:
            List of tuples (sha, message).
        """
        result = self._run_git_command(
            ["log", f"--max-count={limit}", "--pretty=format:%H|%s"]
        )

        commits = []
        for line in result.stdout.strip().split("\n"):
            if "|" in line:
                sha, message = line.split("|", 1)
                commits.append((sha, message))

        return commits
