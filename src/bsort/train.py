"""Training module for bsort."""

from pathlib import Path
from typing import Any, Optional

import wandb
from ultralytics import YOLO

from bsort.config import get_model_config, get_train_config, get_wandb_config


def train_model(
    config: dict[str, Any],
    project: str = "runs/train",
    name: str = "bottle_cap",
) -> Path:
    """
    Train a YOLO model for bottle cap detection.

    Args:
        config: Configuration dictionary loaded from YAML.
        project: Project directory for saving results.
        name: Run name for this training session.

    Returns:
        Path to the best model weights.
    """
    model_config = get_model_config(config)
    train_config = get_train_config(config)
    wandb_config = get_wandb_config(config)

    # Initialize W&B if enabled
    if wandb_config.get("enabled", False):
        wandb.init(
            project=wandb_config.get("project", "bottle-cap-detection"),
            entity=wandb_config.get("entity"),
            name=name,
            config=config,
        )

    # Load model
    model_weights = model_config.get("weights", "yolo11n.pt")
    model = YOLO(model_weights)

    # Train
    results = model.train(
        data=train_config.get("data", "data/data.yaml"),
        epochs=train_config.get("epochs", 100),
        imgsz=model_config.get("imgsz", 640),
        batch=train_config.get("batch", 16),
        patience=train_config.get("patience", 50),
        optimizer=train_config.get("optimizer", "Adam"),
        lr0=train_config.get("lr0", 0.001),
        lrf=train_config.get("lrf", 0.01),
        momentum=train_config.get("momentum", 0.937),
        weight_decay=train_config.get("weight_decay", 0.0005),
        warmup_epochs=train_config.get("warmup_epochs", 3),
        augment=train_config.get("augment", True),
        hsv_h=train_config.get("hsv_h", 0.015),
        hsv_s=train_config.get("hsv_s", 0.7),
        hsv_v=train_config.get("hsv_v", 0.4),
        degrees=train_config.get("degrees", 0.0),
        translate=train_config.get("translate", 0.1),
        scale=train_config.get("scale", 0.5),
        fliplr=train_config.get("fliplr", 0.5),
        mosaic=train_config.get("mosaic", 1.0),
        mixup=train_config.get("mixup", 0.0),
        copy_paste=train_config.get("copy_paste", 0.0),
        device=model_config.get("device", "cpu"),
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
    )

    # Get best model path
    best_model_path = Path(project) / name / "weights" / "best.pt"

    # Log to W&B
    if wandb_config.get("enabled", False):
        artifact = wandb.Artifact("bottle-cap-model", type="model")
        artifact.add_file(str(best_model_path))
        wandb.log_artifact(artifact)
        wandb.finish()

    return best_model_path


def export_model(
    model_path: str,
    export_format: str = "onnx",
    imgsz: int = 192,
    half: bool = True,
    int8: bool = False,
) -> str:
    """
    Export model to specified format.

    Args:
        model_path: Path to the model weights.
        export_format: Export format (onnx, tflite, ncnn).
        imgsz: Image size for export.
        half: Use FP16 quantization.
        int8: Use INT8 quantization.

    Returns:
        Path to exported model.
    """
    model = YOLO(model_path)
    
    export_path = model.export(
        format=export_format,
        imgsz=imgsz,
        half=half,
        int8=int8,
    )
    
    return str(export_path)
