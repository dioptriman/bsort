"""Command-line interface for bsort."""

import json
import sys
from pathlib import Path

import click

from bsort import __version__
from bsort.config import load_config
from bsort.infer import benchmark_inference, run_batch_inference, run_inference
from bsort.train import export_model, train_model


@click.group()
@click.version_option(version=__version__, prog_name="bsort")
def main():
    """Bottle cap color detection and sorting CLI.

    A tool for training and running inference with YOLOv11 models
    to detect and classify bottle caps by color (light blue, dark blue, others).
    """
    pass


@main.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--project",
    "-p",
    default="runs/train",
    help="Project directory for saving results.",
)
@click.option(
    "--name",
    "-n",
    default="bottle_cap",
    help="Run name for this training session.",
)
def train(config: str, project: str, name: str):
    """Train a model for bottle cap detection.

    Example:
        bsort train --config settings.yaml --name experiment_1
    """
    click.echo(f"Loading configuration from: {config}")
    cfg = load_config(config)

    click.echo(f"Starting training...")
    click.echo(f"Project: {project}")
    click.echo(f"Run name: {name}")

    try:
        best_model_path = train_model(cfg, project=project, name=name)
        click.echo(f"\nTraining complete!")
        click.echo(f"Best model saved to: {best_model_path}")
    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--image",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Path to input image or directory.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Output directory for results.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON.",
)
def infer(config: str, image: str, output: str, output_json: bool):
    """Run inference on an image.

    Example:
        bsort infer --config settings.yaml --image sample.jpg
        bsort infer --config settings.yaml --image images/ --json
    """
    cfg = load_config(config)
    image_path = Path(image)

    if image_path.is_dir():
        # Batch inference
        results = run_batch_inference(cfg, str(image_path), output)
        
        if output_json:
            click.echo(json.dumps(results, indent=2))
        else:
            click.echo(f"Processed {len(results)} images\n")
            for result in results:
                _print_result(result)
    else:
        # Single image inference
        result = run_inference(cfg, str(image_path), output)
        
        if output_json:
            click.echo(json.dumps(result, indent=2))
        else:
            _print_result(result)


def _print_result(result: dict):
    """Print inference result in human-readable format."""
    click.echo(f"Image: {result['image_path']}")
    click.echo(f"Inference time: {result['inference_time_ms']:.2f} ms")
    click.echo(f"Detections: {result['num_detections']}")
    
    for det in result["detections"]:
        bbox = det["bbox"]
        click.echo(
            f"  - {det['class_name']} ({det['confidence']:.2f}): "
            f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
        )
    click.echo()


@main.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration YAML file.",
)
@click.option(
    "--image",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Path to test image.",
)
@click.option(
    "--runs",
    "-r",
    default=100,
    help="Number of inference runs.",
)
def benchmark(config: str, image: str, runs: int):
    """Benchmark inference speed.

    Example:
        bsort benchmark --config settings.yaml --image sample.jpg --runs 50
    """
    click.echo(f"Loading configuration from: {config}")
    cfg = load_config(config)

    click.echo(f"Running benchmark with {runs} iterations...")
    
    stats = benchmark_inference(cfg, image, n_runs=runs)
    
    click.echo(f"\nBenchmark Results:")
    click.echo(f"  Mean:   {stats['mean_ms']:.2f} ms")
    click.echo(f"  Std:    {stats['std_ms']:.2f} ms")
    click.echo(f"  Min:    {stats['min_ms']:.2f} ms")
    click.echo(f"  Max:    {stats['max_ms']:.2f} ms")
    click.echo(f"  Median: {stats['median_ms']:.2f} ms")


@main.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to model weights (.pt file).",
)
@click.option(
    "--format",
    "-f",
    default="onnx",
    type=click.Choice(["onnx", "tflite", "ncnn"]),
    help="Export format.",
)
@click.option(
    "--imgsz",
    default=192,
    help="Image size for export.",
)
@click.option(
    "--half",
    is_flag=True,
    help="Use FP16 quantization.",
)
@click.option(
    "--int8",
    is_flag=True,
    help="Use INT8 quantization.",
)
def export(model: str, format: str, imgsz: int, half: bool, int8: bool):
    """Export model to deployment format.

    Example:
        bsort export --model best.pt --format ncnn --imgsz 192 --half
    """
    click.echo(f"Exporting model: {model}")
    click.echo(f"Format: {format}")
    click.echo(f"Image size: {imgsz}")
    
    try:
        export_path = export_model(
            model_path=model,
            export_format=format,
            imgsz=imgsz,
            half=half,
            int8=int8,
        )
        click.echo(f"\nExport complete!")
        click.echo(f"Exported model: {export_path}")
    except Exception as e:
        click.echo(f"Error during export: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
