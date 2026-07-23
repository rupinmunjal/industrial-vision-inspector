#!/usr/bin/env python3
"""Download and prepare the casting classification dataset."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from industrial_vision_inspector.dataset import download_dataset, prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache the Kaggle casting dataset and make deterministic splits."
    )
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/cache")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw/casting")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "data/processed/casting_cls"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Prepare an already downloaded dataset and skip the Kaggle download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_dir or download_dataset(args.cache_dir, args.raw_dir)
    counts = prepare_dataset(source, args.output_dir, seed=args.seed)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

