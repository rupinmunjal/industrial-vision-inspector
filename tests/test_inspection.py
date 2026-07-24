import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from industrial_vision_inspector.ingestion import load_image
from industrial_vision_inspector.inspection import Inspector

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_predictor(image: np.ndarray) -> tuple[str, float]:
    red_mean = float(image[:, :, 2].mean())
    green_mean = float(image[:, :, 1].mean())
    return ("def_front", 0.96) if red_mean > green_mean else ("ok_front", 0.93)


def test_inspector_returns_fail_and_highlight_for_defective_fixture() -> None:
    image = load_image(FIXTURES / "defective_sample.ppm")

    result = Inspector(predictor=fixture_predictor).inspect(image)

    assert result.classification == "defective"
    assert result.result == "fail"
    assert result.confidence == pytest.approx(0.96)
    assert result.defect_region is not None
    assert not np.array_equal(result.annotated_image, image)


def test_inspector_returns_pass_without_highlight_for_ok_fixture() -> None:
    image = load_image(FIXTURES / "ok_sample.ppm")

    result = Inspector(predictor=fixture_predictor).inspect(image)

    assert result.classification == "ok_front"
    assert result.result == "pass"
    assert result.confidence == pytest.approx(0.93)
    assert result.defect_region is None
    assert np.array_equal(result.annotated_image, image)


def test_inspector_rejects_unknown_model_class() -> None:
    image = load_image(FIXTURES / "ok_sample.ppm")
    inspector = Inspector(predictor=lambda _image: ("scratch", 0.8))

    with pytest.raises(ValueError, match="unsupported model class"):
        inspector.inspect(image)


def test_inspector_requires_weights_without_injected_predictor(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Inspector(tmp_path / "missing.pt")


def test_inspector_parses_ultralytics_classification_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fake weights")

    class FakeConfidence:
        def item(self) -> float:
            return 0.87

    class FakeModel:
        names = {0: "defective", 1: "ok_front"}

        def __init__(self, model_path: str) -> None:
            assert model_path == str(weights)

        def predict(self, *, source: np.ndarray, verbose: bool) -> list[object]:
            assert source.shape == (4, 4, 3)
            assert verbose is False
            probabilities = SimpleNamespace(top1=1, top1conf=FakeConfidence())
            return [SimpleNamespace(probs=probabilities, names=self.names)]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    image = load_image(FIXTURES / "ok_sample.ppm")

    result = Inspector(weights).inspect(image)

    assert result.classification == "ok_front"
    assert result.result == "pass"
    assert result.confidence == pytest.approx(0.87)
