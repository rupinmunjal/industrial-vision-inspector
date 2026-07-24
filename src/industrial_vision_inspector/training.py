"""Minimal YOLOv8 classification training entry point."""

import shutil
from pathlib import Path


def train_classifier(
    data_dir: str | Path,
    output_path: str | Path = "models/casting_yolov8n_cls.pt",
    *,
    model_name: str = "yolov8n-cls.pt",
    epochs: int = 10,
    image_size: int = 224,
    batch_size: int = 16,
    seed: int = 42,
    device: str | None = None,
    run_dir: str | Path = "reports/training",
) -> Path:
    """Train one small classifier and copy its best weights to ``models/``."""
    dataset = Path(data_dir)
    for split in ("train", "val", "test"):
        if not (dataset / split).is_dir():
            raise NotADirectoryError(f"dataset split does not exist: {dataset / split}")
    if epochs <= 0 or image_size <= 0 or batch_size <= 0:
        raise ValueError("epochs, image size, and batch size must be positive")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required for training. Install it with: pip install ultralytics"
        ) from error

    model = YOLO(model_name)
    train_arguments = {
        "data": str(dataset),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "seed": seed,
        "patience": 3,
        "workers": 4,
        "project": str(Path(run_dir)),
        "name": "casting_yolov8n_cls",
        "exist_ok": True,
    }
    if device is not None:
        train_arguments["device"] = device
    model.train(**train_arguments)

    best_weights = Path(model.trainer.best)
    if not best_weights.is_file():
        raise RuntimeError(f"training did not produce best weights: {best_weights}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, destination)
    return destination
