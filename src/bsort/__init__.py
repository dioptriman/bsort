"""Bottle cap color detection and sorting using YOLOv11."""

__version__ = "0.1.0"
__author__ = "Your Name"

from bsort.config import load_config
from bsort.infer import run_inference
from bsort.train import train_model

__all__ = ["load_config", "train_model", "run_inference", "__version__"]
