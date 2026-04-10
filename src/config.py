"""Configuration management utilities.

Handles parsing, merging, and retrieving values from YAML configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required for config loading. Install it with: pip install pyyaml"
    ) from exc


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries.

    Args:
        base: The base dictionary.
        override: The dictionary containing overrides to apply.

    Returns:
        A new dictionary containing the merged result.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration from a YAML file and merge with defaults.

    Args:
        config_path: Path to the configuration file.

    Returns:
        The parsed and merged configuration dictionary.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If the parsed configuration root is not a mapping.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file_handle:
        parsed = yaml.safe_load(file_handle) or {}

    if not isinstance(parsed, dict):
        raise ValueError("Config root must be a mapping/object.")

    # Always merge with default config so missing keys still have safe defaults.
    if path != DEFAULT_CONFIG_PATH and DEFAULT_CONFIG_PATH.exists():
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as default_file:
            default_parsed = yaml.safe_load(default_file) or {}
        if not isinstance(default_parsed, dict):
            raise ValueError("Default config root must be a mapping/object.")
        return _deep_merge(default_parsed, parsed)

    return parsed


def get_config_value(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Retrieve a nested value from the configuration dictionary.

    Args:
        config: The configuration dictionary.
        *keys: Arguments representing the nested path of keys.
        default: The value to return if the path is not found.

    Returns:
        The resolved value or the default.
    """
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def resolve_path(config: Dict[str, Any], key: str, fallback: Path) -> Path:
    """Resolve a path from the 'paths' section of the configuration.

    Args:
        config: The configuration dictionary.
        key: The key within the 'paths' section.
        fallback: The default path to use if the key is not found.

    Returns:
        An absolute path object.
    """
    raw_value = get_config_value(config, "paths", key)
    if raw_value is None:
        return fallback
    candidate = Path(str(raw_value))
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate


def cli_or_config(cli_value: Any, config: Dict[str, Any], section: str, key: str) -> Any:
    """Select a value from the CLI override or the configuration file.

    Args:
        cli_value: The value provided via command-line arguments.
        config: The configuration dictionary.
        section: The section key in the configuration.
        key: The specific key within the section.

    Returns:
        The CLI value if present, otherwise the configuration value.
    """
    if cli_value is not None:
        return cli_value
    return get_config_value(config, section, key)
