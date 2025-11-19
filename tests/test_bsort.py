"""Tests for bsort package."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from bsort.config import (
    get_export_config,
    get_inference_config,
    get_model_config,
    get_train_config,
    get_wandb_config,
    load_config,
)


class TestConfig:
    """Tests for configuration loading."""

    def test_load_config_valid(self):
        """Test loading a valid config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            config_data = {
                "model": {"weights": "test.pt", "imgsz": 640},
                "train": {"epochs": 100},
            }
            yaml.dump(config_data, f)
            f.flush()

            config = load_config(f.name)
            assert config["model"]["weights"] == "test.pt"
            assert config["model"]["imgsz"] == 640
            assert config["train"]["epochs"] == 100

        os.unlink(f.name)

    def test_load_config_not_found(self):
        """Test loading a non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_get_model_config(self):
        """Test extracting model config."""
        config = {
            "model": {"weights": "best.pt", "imgsz": 192},
            "train": {"epochs": 100},
        }
        model_config = get_model_config(config)
        assert model_config["weights"] == "best.pt"
        assert model_config["imgsz"] == 192

    def test_get_model_config_empty(self):
        """Test extracting model config from empty dict."""
        config = {}
        model_config = get_model_config(config)
        assert model_config == {}

    def test_get_train_config(self):
        """Test extracting train config."""
        config = {
            "train": {
                "epochs": 200,
                "batch": 16,
                "lr0": 0.001,
            }
        }
        train_config = get_train_config(config)
        assert train_config["epochs"] == 200
        assert train_config["batch"] == 16
        assert train_config["lr0"] == 0.001

    def test_get_inference_config(self):
        """Test extracting inference config."""
        config = {
            "inference": {
                "save": True,
                "show": False,
            }
        }
        inference_config = get_inference_config(config)
        assert inference_config["save"] is True
        assert inference_config["show"] is False

    def test_get_wandb_config(self):
        """Test extracting W&B config."""
        config = {
            "wandb": {
                "project": "test-project",
                "enabled": True,
            }
        }
        wandb_config = get_wandb_config(config)
        assert wandb_config["project"] == "test-project"
        assert wandb_config["enabled"] is True

    def test_get_export_config(self):
        """Test extracting export config."""
        config = {
            "export": {
                "format": "ncnn",
                "imgsz": 192,
                "half": True,
            }
        }
        export_config = get_export_config(config)
        assert export_config["format"] == "ncnn"
        assert export_config["imgsz"] == 192
        assert export_config["half"] is True


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_import(self):
        """Test that CLI can be imported."""
        from bsort.cli import main
        assert main is not None

    def test_cli_version(self):
        """Test version is accessible."""
        from bsort import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)


class TestInference:
    """Tests for inference module."""

    def test_infer_import(self):
        """Test that inference module can be imported."""
        from bsort.infer import (
            benchmark_inference,
            run_batch_inference,
            run_inference,
            visualize_result,
        )
        assert run_inference is not None
        assert run_batch_inference is not None
        assert visualize_result is not None
        assert benchmark_inference is not None


class TestTrain:
    """Tests for training module."""

    def test_train_import(self):
        """Test that training module can be imported."""
        from bsort.train import export_model, train_model
        assert train_model is not None
        assert export_model is not None
