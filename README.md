# bsort - Bottle Cap Color Detection

A real-time computer vision system for detecting and classifying bottle caps by color using YOLOv11.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![CI/CD](https://github.com/yourusername/bsort/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/bsort/actions)

## Overview

This project implements a bottle cap detection and color classification system designed for edge deployment on Raspberry Pi 5. The system detects bottle caps and classifies them into three categories:

- **Light Blue** (Class 0)
- **Dark Blue** (Class 1)
- **Others** (Class 2) - includes yellow, green, white, etc.

## Features

- 🚀 Fast inference optimized for edge devices (RPi5)
- 🎯 YOLOv11 nano model for best speed-accuracy tradeoff
- 📊 W&B integration for experiment tracking
- 🐳 Docker support for easy deployment
- ⚡ Multiple export formats (ONNX, TFLite, NCNN)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/bsort.git
cd bsort

# Install the package
pip install -e .

# For development
pip install -e ".[dev]"
```

### Using Docker

```bash
# Build the image
docker build -t bsort .

# Run inference
docker run -v $(pwd)/images:/images bsort infer --config settings.yaml --image /images/sample.jpg
```

## Usage

### Training

```bash
bsort train --config settings.yaml --name experiment_1
```

### Inference

```bash
# Single image
bsort infer --config settings.yaml --image sample.jpg

# Batch inference on directory
bsort infer --config settings.yaml --image images/ --json

# With custom output directory
bsort infer --config settings.yaml --image sample.jpg --output results/
```

### Benchmarking

```bash
bsort benchmark --config settings.yaml --image sample.jpg --runs 100
```

### Export Model

```bash
# Export to NCNN (recommended for RPi5)
bsort export --model models/best.pt --format ncnn --imgsz 192 --half

# Export to ONNX
bsort export --model models/best.pt --format onnx --imgsz 192

# Export to TFLite with INT8
bsort export --model models/best.pt --format tflite --imgsz 192 --int8
```

## Configuration

All settings are controlled via `settings.yaml`:

```yaml
model:
  weights: "models/best.pt"
  imgsz: 192
  conf: 0.5
  device: "cpu"

train:
  epochs: 300
  batch: 8
  patience: 100
  # ... see settings.yaml for full options

wandb:
  project: "bottle-cap-color-detection"
  enabled: true
```

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| mAP50 | TBD |
| mAP50-95 | TBD |
| Precision | TBD |
| Recall | TBD |

### Inference Speed

**Benchmarked on:** Google Colab CPU (Intel Xeon)

| Input Size | Colab CPU | Estimated RPi5* |
|------------|-----------|-----------------|
| 320x320 | 49.9ms | ~15-25ms |
| 256x256 | 110.3ms | ~30-50ms |
| 192x192 | 26.7ms | ~8-15ms ✅ |

*Estimated with NCNN runtime and INT8 quantization

### Class Distribution

- Others: 49 samples (62.0%)
- Dark Blue: 17 samples (21.5%)
- Light Blue: 13 samples (16.5%)

## W&B Experiment Tracking

View all experiments and model artifacts:

🔗 **[W&B Project Dashboard](https://wandb.ai/your-username/bottle-cap-color-detection)**

## Project Structure

```
bsort/
├── src/bsort/
│   ├── __init__.py
│   ├── cli.py          # CLI commands
│   ├── config.py       # Configuration management
│   ├── train.py        # Training logic
│   └── infer.py        # Inference logic
├── tests/
│   └── test_bsort.py   # Unit tests
├── models/             # Model weights
├── notebooks/          # Jupyter notebooks
├── .github/workflows/  # CI/CD
├── Dockerfile
├── pyproject.toml
├── settings.yaml
└── README.md
```

## Model Selection Justification

### Selected: YOLOv11n

| Model | Parameters | FLOPs | Pros | Cons |
|-------|------------|-------|------|------|
| **YOLOv11n** | 2.6M | 6.5G | Latest architecture, TTA support, best efficiency | Newer |
| YOLOv10n | 2.3M | 6.7G | NMS-free, fast | No TTA support |
| YOLOv8n | 3.2M | 8.7G | Mature ecosystem | More parameters |

**Key reasons for YOLOv11n:**
1. Test-Time Augmentation (TTA) support for handling dirty/shadowed caps
2. Latest architectural improvements
3. Better small object detection
4. Optimized for edge deployment

## Known Issues & Limitations

1. **Small dataset** - Only 79 samples, may not generalize well
2. **Class imbalance** - "Others" class dominates (62%)
3. **Shadow sensitivity** - Blue caps with shadows may be misclassified
4. **Dirty caps** - Contamination affects color detection

## Recommendations for Production

1. **Data collection** - Add more light_blue and dark_blue samples
2. **Preprocessing** - Consider shadow removal or histogram equalization
3. **Two-stage approach** - Detect caps first, then classify color separately
4. **Continuous learning** - Log misclassifications for model improvement

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

### Linting

```bash
pylint src/bsort
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [Weights & Biases](https://wandb.ai) for experiment tracking
