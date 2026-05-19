from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - dependency is declared.
    yaml = None


def repo_root() -> Path:
    """Return the project root, preferring the current working tree."""

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "AGENTS.md").exists() and (candidate / "configs").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return repo_root() / value


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read configuration files.")
    config_path = resolve_project_path(path)
    if not config_path.exists():
        return {}
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return loaded or {}


def app_config() -> dict[str, Any]:
    return load_yaml("configs/app.yaml")


def database_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path:
        return resolve_project_path(explicit_path)
    config = app_config()
    return resolve_project_path(config.get("paths", {}).get("database", "data/processed/notes.db"))


def raw_notes_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path:
        return resolve_project_path(explicit_path)
    config = app_config()
    return resolve_project_path(config.get("paths", {}).get("raw_notes", "data/raw/apple_notes_export"))


def load_categories() -> list[str]:
    config = load_yaml("configs/categories.yaml")
    categories = config.get("categories", [])
    return [str(item) for item in categories]


def load_model_config() -> dict[str, list[dict[str, str]]]:
    config = load_yaml("configs/models.yaml")
    return config.get("models", {})


def load_model_defaults() -> dict[str, str]:
    config = load_yaml("configs/models.yaml")
    defaults = config.get("defaults", {})
    return {str(key): str(value) for key, value in defaults.items()}
