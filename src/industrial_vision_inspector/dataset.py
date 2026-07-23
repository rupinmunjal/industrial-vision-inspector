"""Kaggle download and deterministic classification dataset preparation."""

import json
import os
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

KAGGLE_SLUG = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"
ARCHIVE_NAME = "real-life-industrial-dataset-of-casting-product.zip"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_ALIASES = {
    "def_front": "defective",
    "defective": "defective",
    "ok_front": "ok_front",
}


def _kaggle_credentials_path() -> Path:
    configured = os.environ.get("KAGGLE_CONFIG_DIR")
    return Path(configured) / "kaggle.json" if configured else Path.home() / ".kaggle" / "kaggle.json"


def download_dataset(
    cache_dir: str | Path = "data/cache",
    extract_dir: str | Path = "data/raw/casting",
) -> Path:
    """Download the Kaggle archive once and extract it into the local cache."""
    cache_path = Path(cache_dir)
    extracted_path = Path(extract_dir)
    complete_marker = extracted_path / ".complete"
    if complete_marker.is_file():
        return extracted_path

    archive_path = cache_path / ARCHIVE_NAME
    if not archive_path.is_file():
        has_environment_credentials = bool(
            os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
        )
        credentials_path = _kaggle_credentials_path()
        if not has_environment_credentials and not credentials_path.is_file():
            raise FileNotFoundError(
                "Kaggle credentials not found. Place kaggle.json at "
                f"{credentials_path} or set KAGGLE_USERNAME and KAGGLE_KEY."
            )
        if shutil.which("kaggle") is None:
            raise RuntimeError("Kaggle CLI not found. Install it with: pip install kaggle")

        cache_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(cache_path)],
            check=True,
        )
        if not archive_path.is_file():
            raise RuntimeError(f"Kaggle download completed but {archive_path} was not created")

    extracted_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted_path)
    complete_marker.write_text("downloaded and extracted\n", encoding="utf-8")
    return extracted_path


def discover_images(source_dir: str | Path) -> dict[str, list[Path]]:
    """Find supported images and map Kaggle folder names to two honest labels."""
    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise NotADirectoryError(f"dataset source is not a folder: {source_path}")

    found: dict[str, list[Path]] = {"defective": [], "ok_front": []}
    for path in source_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        labels = {
            CLASS_ALIASES[part.casefold()]
            for part in path.parts
            if part.casefold() in CLASS_ALIASES
        }
        if len(labels) == 1:
            found[labels.pop()].append(path)

    for paths in found.values():
        paths.sort()
    return found


def prepare_dataset(
    source_dir: str | Path,
    output_dir: str | Path = "data/processed/casting_cls",
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """Create deterministic train/val/test folders for YOLO classification."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train, validation, and test ratios must sum to 1.0")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split ratios cannot be negative")

    output_path = Path(output_dir)
    manifest_path = output_path / "split_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest["counts"]
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"output folder is not empty and has no split manifest: {output_path}"
        )

    images = discover_images(source_dir)
    missing = [label for label, paths in images.items() if not paths]
    if missing:
        raise ValueError(f"no images found for class(es): {', '.join(missing)}")

    rng = random.Random(seed)
    split_images: dict[str, dict[str, list[Path]]] = {
        "train": {},
        "val": {},
        "test": {},
    }
    for label in sorted(images):
        shuffled = images[label].copy()
        rng.shuffle(shuffled)
        train_end = int(len(shuffled) * train_ratio)
        val_end = train_end + int(len(shuffled) * val_ratio)
        split_images["train"][label] = shuffled[:train_end]
        split_images["val"][label] = shuffled[train_end:val_end]
        split_images["test"][label] = shuffled[val_end:]

    counts: dict[str, dict[str, int]] = {}
    for split, classes in split_images.items():
        counts[split] = {}
        for label, paths in classes.items():
            destination = output_path / split / label
            destination.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(paths):
                shutil.copy2(source, destination / f"{index:05d}_{source.name}")
            counts[split][label] = len(paths)

    manifest = {
        "source": str(Path(source_dir).resolve()),
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": counts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return counts

