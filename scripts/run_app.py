#!/usr/bin/env python3
"""Launch the Industrial Vision Inspector desktop application."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from industrial_vision_inspector.ui import run_app


def default_paths() -> tuple[Path, Path]:
    """Return model and database defaults for source and frozen launches."""
    if getattr(sys, "frozen", False):
        resource_root = Path(getattr(sys, "_MEIPASS"))
        data_root = Path(sys.executable).resolve().parent
    else:
        resource_root = REPO_ROOT
        data_root = REPO_ROOT
    return (
        resource_root / "models/casting_yolov8n_cls.pt",
        data_root / "data/inspections.db",
    )


def parse_args() -> argparse.Namespace:
    model_path, database_path = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=model_path,
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=database_path,
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="show the window briefly and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_app(
        args.model,
        args.database,
        camera_index=args.camera_index,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    raise SystemExit(main())
