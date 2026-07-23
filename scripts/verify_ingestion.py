#!/usr/bin/env python3
"""Load sample images and save a contact sheet for visual verification."""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from industrial_vision_inspector.ingestion import load_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder", nargs="?", type=Path, default=REPO_ROOT / "tests/fixtures"
    )
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "reports/ingestion_samples.jpg"
    )
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def make_contact_sheet(images: list[tuple[Path, np.ndarray]], columns: int = 3) -> np.ndarray:
    if not images:
        raise ValueError("no supported images were found")

    tiles = []
    for path, image in images:
        tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        cv2.putText(
            tile,
            path.name,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    rows = math.ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    tiles.extend(blank.copy() for _ in range(rows * columns - len(tiles)))
    return cv2.vconcat(
        [cv2.hconcat(tiles[index : index + columns]) for index in range(0, len(tiles), columns)]
    )


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    loaded = load_folder(
        args.folder,
        size=(args.width, args.height),
        grayscale=args.grayscale,
    )[: args.count]
    sheet = make_contact_sheet(loaded)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"could not write contact sheet: {args.output}")
    print(f"saved {len(loaded)} images to {args.output}")

    if args.show:
        cv2.imshow("Ingestion samples", sheet)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

