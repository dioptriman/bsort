"""Configuration management for bsort."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract model configuration from main config.

    Args:
        config: Main configuration dictionary.

    Returns:
        Model configuration dictionary.
    """
    return config.get("model", {})


def get_train_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract training configuration from main config.

    Args:
        config: Main configuration dictionary.

    Returns:
        Training configuration dictionary.
    """
    return config.get("train", {})


def get_inference_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract inference configuration from main config.

    Args:
        config: Main configuration dictionary.

    Returns:
        Inference configuration dictionary.
    """
    return config.get("inference", {})


def get_wandb_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract W&B configuration from main config.

    Args:
        config: Main configuration dictionary.

    Returns:
        W&B configuration dictionary.
    """
    return config.get("wandb", {})


def get_export_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract export configuration from main config.

    Args:
        config: Main configuration dictionary.

    Returns:
        Export configuration dictionary.
    """
    return config.get("export", {})
