#!/usr/bin/env python3
"""Evaluate trained casting weights on the held-out test split."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from industrial_vision_inspector.evaluation import (
    evaluate_test_directory,
    save_confusion_matrix,
    save_metrics,
    save_prediction_samples,
)
from industrial_vision_inspector.inspection import Inspector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=REPO_ROOT / "data/processed/casting_cls/test",
    )
    parser.add_argument(
        "--model", type=Path, default=REPO_ROOT / "models/casting_yolov8n_cls.pt"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "reports/metrics"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, samples = evaluate_test_directory(Inspector(args.model), args.test_dir)
    save_metrics(metrics, args.output_dir / "metrics.json")
    save_confusion_matrix(metrics, args.output_dir / "confusion_matrix.png")
    save_prediction_samples(samples, args.output_dir / "prediction_samples.png")
    print(json.dumps(asdict(metrics), indent=2))


if __name__ == "__main__":
    main()
