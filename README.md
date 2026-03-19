# `git-copilot-commit`

[![CI](https://img.shields.io/github/actions/workflow/status/kdheepak/git-copilot-commit/ci.yml?branch=main&label=CI)](https://github.com/kdheepak/git-copilot-commit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/git-copilot-commit)](https://pypi.org/project/git-copilot-commit/)
[![License](https://img.shields.io/github/license/kdheepak/git-copilot-commit)](https://github.com/kdheepak/git-copilot-commit/blob/main/LICENSE)

AI-powered Git commit assistant that generates conventional commit messages using GitHub Copilot.

![Screenshot of git-copilot-commit in action](https://github.com/user-attachments/assets/6a6d70a6-6060-44e6-8cf4-a6532e9e9142)

## Features

- Generates commit messages based on your staged changes
- Supports multiple LLM models: GPT-4, Claude, Gemini, and more
- Allows editing of generated messages before committing
- Follows the [Conventional Commits](https://www.conventionalcommits.org/) standard

## Installation

### Install the tool using [`uv`]

Install [`uv`]:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

You can run the latest version of tool directly every time by invoking this one command:

```bash
# Every invocation installs latest version into temporary environment and runs --help
uvx git-copilot-commit --help
```

Alternatively, you can install the tool once into a global isolated environment 
and run `git-copilot-commit` to invoke it:

```bash
# Install into global isolated environment
uv tool install git-copilot-commit

# Run --help to see available commands
git-copilot-commit --help
```

[`uv`]: https://github.com/astral-sh/uv

## Prerequisites

- Active GitHub Copilot subscription

## Quick Start

1. Authenticate with GitHub Copilot:

   ```bash
   uvx git-copilot-commit authenticate
   ```

   If your cached GitHub token is revoked or expires, refresh it with:

   ```bash
   uvx git-copilot-commit authenticate --force
   ```

2. Make changes in your repository.

3. Generate and commit:

   ```bash
   uvx git-copilot-commit commit
   # Or, if you want to stage all files and accept the generated commit message, use:
   uvx git-copilot-commit commit --all --yes
   ```

## Usage

### Commit changes

```bash
$ uvx git-copilot-commit commit --help

 Usage: git-copilot-commit commit [OPTIONS]

 Generate commit message based on changes in the current git repository and commit them.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --all         -a                               Stage all files before committing                         │
│ --split                                        Split staged hunks into multiple commits automatically.   │
│                                                Pass `--split=N` to express a preference for N commits.   │
│ --model       -m                     MODEL_ID  Model to use for generating commit message                │
│ --yes         -y                               Automatically accept the generated commit message         │
│ --context     -c                     TEXT      Optional user-provided context to guide commit message    │
│ --ca-bundle                          PATH      Path to a custom CA bundle (PEM)                          │
│ --insecure                                     Disable SSL certificate verification.                     │
│ --native-tls      --no-native-tls              Use the OS's native certificate store via 'truststore'    │
│                                                for httpx instead of the Python bundle. Ignored if        │
│                                                --ca-bundle or --insecure is used.                        │
│                                                [default: no-native-tls]                                  │
│ --help                                         Show this message and exit.                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Examples

Commit all changes:

```bash
uvx git-copilot-commit commit --all
```

Accept the generated commit message without editing:

```bash
uvx git-copilot-commit commit --yes
```

Use a specific model:

```bash
uvx git-copilot-commit commit --model claude-3.5-sonnet
```

Split staged hunks into separate commits:

```bash
uvx git-copilot-commit commit --split
```

Prefer two commits:

```bash
uvx git-copilot-commit commit --split 2
```

## Commit Message Format

Follows [Conventional Commits](https://www.conventionalcommits.org/):

```plaintext
<type>[optional scope]: <description>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting only
- `refactor`: Code restructure
- `perf`: Performance
- `test`: Tests
- `chore`: Maintenance
- `revert`: Revert changes

## Git Configuration

Add a git alias by adding the following to your `~/.gitconfig`:

```ini
[alias]
    ai-commit = "!f() { uvx git-copilot-commit commit $@; }; f"
```

Now you can run to review the message before committing:

```bash
git ai-commit
```

Alternatively, you can stage all files and auto accept the commit message and
specify which model should be used to generate the commit in one CLI invocation.

```bash
git ai-commit --all --yes --model claude-3.5-sonnet
```

> [!TIP]
>
> Show more context in diffs by running the following command:
>
> ```bash
> git config --global diff.context 3
> ```
>
> This may be useful because this tool sends the diffs with surrounding context 
> to the LLM for generating a commit message
