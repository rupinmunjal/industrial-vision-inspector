#!/usr/bin/env python3
"""Train the YOLOv8n casting classifier with demo-scale defaults."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from industrial_vision_inspector.training import train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=REPO_ROOT / "data/processed/casting_cls"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "models/casting_yolov8n_cls.pt"
    )
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="Ultralytics device, for example cpu, 0, or mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = train_classifier(
        args.data,
        args.output,
        model_name=args.model,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        run_dir=REPO_ROOT / "reports/training",
    )
    print(f"saved best weights to {output}")


if __name__ == "__main__":
    main()
