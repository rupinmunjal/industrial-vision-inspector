"""OpenCV image and stream ingestion utilities."""

from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".ppm", ".tif", ".tiff", ".webp"}
)

ImageArray = NDArray[np.uint8] | NDArray[np.float32]


def preprocess_image(
    image: NDArray[np.generic],
    *,
    size: tuple[int, int] | None = None,
    normalize: bool = False,
    grayscale: bool = False,
) -> ImageArray:
    """Apply optional grayscale conversion, resize, and [0, 1] normalization.

    ``size`` is expressed as ``(width, height)``, matching OpenCV.
    """
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")

    processed = image
    if grayscale and processed.ndim == 3:
        if processed.shape[2] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        elif processed.shape[2] == 4:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"unsupported channel count: {processed.shape[2]}")
    elif processed.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D image, got {processed.ndim} dimensions")

    if size is not None:
        if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            raise ValueError("size must contain positive (width, height) values")
        processed = cv2.resize(processed, size, interpolation=cv2.INTER_AREA)

    if normalize:
        processed = processed.astype(np.float32) / 255.0

    return processed


def load_image(
    path: str | Path,
    *,
    size: tuple[int, int] | None = None,
    normalize: bool = False,
    grayscale: bool = False,
) -> ImageArray:
    """Load one image from disk in OpenCV BGR channel order."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"image does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    if image.ndim == 2 and not grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4 and not grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return preprocess_image(
        image, size=size, normalize=normalize, grayscale=grayscale
    )


def load_folder(
    folder: str | Path,
    *,
    size: tuple[int, int] | None = None,
    normalize: bool = False,
    grayscale: bool = False,
    extensions: Sequence[str] = tuple(IMAGE_EXTENSIONS),
) -> list[tuple[Path, ImageArray]]:
    """Load supported images directly inside a folder in filename order."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"not a folder: {folder_path}")

    allowed = {extension.lower() for extension in extensions}
    paths = sorted(
        (
            path
            for path in folder_path.iterdir()
            if path.is_file() and path.suffix.lower() in allowed
        ),
        key=lambda path: path.name.casefold(),
    )
    return [
        (
            path,
            load_image(
                path, size=size, normalize=normalize, grayscale=grayscale
            ),
        )
        for path in paths
    ]


def iter_frames(
    source: str | Path | int,
    *,
    size: tuple[int, int] | None = None,
    normalize: bool = False,
    grayscale: bool = False,
    max_frames: int | None = None,
) -> Iterator[ImageArray]:
    """Yield frames from a video path or webcam index and release it afterward."""
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames cannot be negative")

    capture_source = str(source) if isinstance(source, Path) else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open video source: {source}")

    emitted = 0
    try:
        while max_frames is None or emitted < max_frames:
            success, frame = capture.read()
            if not success:
                break
            yield preprocess_image(
                frame, size=size, normalize=normalize, grayscale=grayscale
            )
            emitted += 1
    finally:
        capture.release()

