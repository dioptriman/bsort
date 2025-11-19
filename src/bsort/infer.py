"""Inference module for bsort."""

import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from bsort.config import get_inference_config, get_model_config


def run_inference(
    config: dict[str, Any],
    image_path: str,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run inference on a single image.

    Args:
        config: Configuration dictionary loaded from YAML.
        image_path: Path to the input image.
        output_dir: Directory to save results (optional).

    Returns:
        Dictionary containing detection results and timing.
    """
    model_config = get_model_config(config)
    inference_config = get_inference_config(config)

    # Load model
    model_weights = model_config.get("weights", "models/best.pt")
    model = YOLO(model_weights)

    # Run inference with timing
    start_time = time.time()
    results = model.predict(
        source=image_path,
        imgsz=model_config.get("imgsz", 192),
        conf=model_config.get("conf", 0.5),
        iou=model_config.get("iou", 0.45),
        device=model_config.get("device", "cpu"),
        save=inference_config.get("save", False),
        save_txt=inference_config.get("save_txt", False),
        save_conf=inference_config.get("save_conf", False),
        show=inference_config.get("show", False),
        project=output_dir,
        verbose=False,
    )
    inference_time = (time.time() - start_time) * 1000  # Convert to ms

    # Process results
    result = results[0]
    detections = []

    class_names = config.get("classes", {0: "light_blue", 1: "dark_blue", 2: "others"})

    for box in result.boxes:
        detection = {
            "class_id": int(box.cls[0]),
            "class_name": class_names.get(int(box.cls[0]), "unknown"),
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
        }
        detections.append(detection)

    return {
        "image_path": image_path,
        "inference_time_ms": inference_time,
        "num_detections": len(detections),
        "detections": detections,
    }


def run_batch_inference(
    config: dict[str, Any],
    image_dir: str,
    output_dir: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Run inference on all images in a directory.

    Args:
        config: Configuration dictionary loaded from YAML.
        image_dir: Directory containing input images.
        output_dir: Directory to save results (optional).

    Returns:
        List of detection results for each image.
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(f"*{ext}"))
        image_paths.extend(Path(image_dir).glob(f"*{ext.upper()}"))

    results = []
    for img_path in image_paths:
        result = run_inference(config, str(img_path), output_dir)
        results.append(result)

    return results


def visualize_result(
    image_path: str,
    detections: list[dict[str, Any]],
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Visualize detection results on image.

    Args:
        image_path: Path to the input image.
        detections: List of detection dictionaries.
        output_path: Path to save visualization (optional).

    Returns:
        Image with drawn bounding boxes.
    """
    img = cv2.imread(image_path)

    # Color mapping for classes
    colors = {
        0: (255, 200, 100),  # Light blue (BGR)
        1: (255, 100, 50),  # Dark blue (BGR)
        2: (100, 100, 100),  # Others (gray)
    }

    for det in detections:
        class_id = det["class_id"]
        class_name = det["class_name"]
        conf = det["confidence"]
        bbox = det["bbox"]

        x1, y1, x2, y2 = map(int, bbox)
        color = colors.get(class_id, (0, 255, 0))

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_name} {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(
            img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    if output_path:
        cv2.imwrite(output_path, img)

    return img


def benchmark_inference(
    config: dict[str, Any],
    image_path: str,
    n_runs: int = 100,
    warmup: int = 10,
) -> dict[str, float]:
    """
    Benchmark inference speed.

    Args:
        config: Configuration dictionary loaded from YAML.
        image_path: Path to test image.
        n_runs: Number of inference runs.
        warmup: Number of warmup runs.

    Returns:
        Dictionary with timing statistics.
    """
    model_config = get_model_config(config)
    model = YOLO(model_config.get("weights", "models/best.pt"))

    # Warmup
    for _ in range(warmup):
        model.predict(
            source=image_path,
            imgsz=model_config.get("imgsz", 192),
            device=model_config.get("device", "cpu"),
            verbose=False,
        )

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.time()
        model.predict(
            source=image_path,
            imgsz=model_config.get("imgsz", 192),
            device=model_config.get("device", "cpu"),
            verbose=False,
        )
        times.append((time.time() - start) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
    }
