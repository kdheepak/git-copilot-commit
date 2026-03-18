from __future__ import annotations

import json

import pytest

import git_copilot_commit.settings as settings_module


@pytest.fixture
def settings_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path):
    dirs = {
        "config": tmp_path / "config",
        "data": tmp_path / "data",
        "cache": tmp_path / "cache",
        "state": tmp_path / "state",
    }

    monkeypatch.setattr(
        settings_module, "user_config_dir", lambda _app_name: str(dirs["config"])
    )
    monkeypatch.setattr(
        settings_module, "user_data_dir", lambda _app_name: str(dirs["data"])
    )
    monkeypatch.setattr(
        settings_module, "user_cache_dir", lambda _app_name: str(dirs["cache"])
    )
    monkeypatch.setattr(
        settings_module, "user_state_dir", lambda _app_name: str(dirs["state"])
    )

    return dirs


def test_settings_create_directories_and_persist_values(settings_dirs) -> None:
    settings = settings_module.Settings()

    assert settings.config_dir == settings_dirs["config"]
    assert settings.data_dir == settings_dirs["data"]
    assert settings.cache_dir == settings_dirs["cache"]
    assert settings.state_dir == settings_dirs["state"]
    assert settings.config_file == settings_dirs["config"] / "config.json"
    assert settings.get("missing", "fallback") == "fallback"

    settings.default_model = "gpt-5.4"
    settings.default_prompt_file = " prompts/custom.md "
    settings.set("theme", "dark")

    reloaded = settings_module.Settings()
    assert reloaded.default_model == "gpt-5.4"
    assert reloaded.default_prompt_file == "prompts/custom.md"
    assert reloaded.get("theme") == "dark"

    reloaded.delete("theme")
    assert settings_module.Settings().get("theme") is None


def test_settings_load_invalid_json_as_empty_config(settings_dirs) -> None:
    config_file = settings_dirs["config"] / "config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("{invalid json", encoding="utf-8")

    settings = settings_module.Settings()

    assert settings.get("anything") is None


def test_default_prompt_file_prefers_nested_defaults(settings_dirs) -> None:
    config_file = settings_dirs["config"] / "config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps(
            {
                "default_prompt_file": "legacy.md",
                "defaults": {"prompt_file": " prompts/from-defaults.md "},
            }
        ),
        encoding="utf-8",
    )

    settings = settings_module.Settings()

    assert settings.default_prompt_file == "prompts/from-defaults.md"


def test_default_prompt_file_rejects_invalid_values(settings_dirs) -> None:
    config_file = settings_dirs["config"] / "config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps({"defaults": {"prompt_file": "   "}}),
        encoding="utf-8",
    )

    settings = settings_module.Settings()

    with pytest.raises(ValueError):
        _ = settings.default_prompt_file


def test_clear_cache_removes_files_and_directories(settings_dirs) -> None:
    settings = settings_module.Settings()
    cached_file = settings.cache_dir / "cached.txt"
    cached_file.write_text("cached", encoding="utf-8")
    nested_dir = settings.cache_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "item.txt").write_text("nested", encoding="utf-8")

    settings.clear_cache()

    assert list(settings.cache_dir.iterdir()) == []
