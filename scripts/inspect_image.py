#!/usr/bin/env python3
"""Inspect one image through the same core used by the desktop UI."""

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from industrial_vision_inspector.ingestion import load_image
from industrial_vision_inspector.inspection import Inspector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model", type=Path, default=REPO_ROOT / "models/casting_yolov8n_cls.pt"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "reports/latest_inspection.jpg"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = Inspector(args.model).inspect(load_image(args.image))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result.annotated_image):
        raise RuntimeError(f"could not save annotated image: {args.output}")
    print(
        json.dumps(
            {
                "classification": result.classification,
                "result": result.result,
                "confidence": result.confidence,
                "defect_region": result.defect_region,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
