from pathlib import Path

import cv2
import numpy as np
import pytest

from industrial_vision_inspector.ingestion import iter_frames, load_folder, load_image

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_image_resizes_and_normalizes() -> None:
    image = load_image(
        FIXTURES / "ok_sample.ppm", size=(10, 6), normalize=True
    )

    assert image.shape == (6, 10, 3)
    assert image.dtype == np.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0


def test_load_image_can_return_grayscale() -> None:
    image = load_image(FIXTURES / "defective_sample.ppm", grayscale=True)

    assert image.shape == (4, 4)
    assert image.dtype == np.uint8


def test_load_folder_is_non_recursive_and_sorted(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    for name in ("ok_sample.ppm", "defective_sample.ppm"):
        (tmp_path / name).write_bytes((FIXTURES / name).read_bytes())
    (tmp_path / "ignored.txt").write_text("not an image", encoding="utf-8")
    (tmp_path / "nested" / "nested.ppm").write_bytes(
        (FIXTURES / "ok_sample.ppm").read_bytes()
    )

    loaded = load_folder(tmp_path)

    assert [path.name for path, _ in loaded] == [
        "defective_sample.ppm",
        "ok_sample.ppm",
    ]


def test_load_image_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")


def test_iter_frames_preprocesses_and_releases_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [
        np.full((4, 4, 3), 64, dtype=np.uint8),
        np.full((4, 4, 3), 128, dtype=np.uint8),
    ]

    class FakeCapture:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            return (True, frames.pop(0)) if frames else (False, None)

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)

    loaded = list(iter_frames(0, size=(2, 2), normalize=True, max_frames=1))

    assert len(loaded) == 1
    assert loaded[0].shape == (2, 2, 3)
    assert loaded[0].dtype == np.float32
    assert capture.released

