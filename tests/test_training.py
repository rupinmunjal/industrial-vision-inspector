import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from industrial_vision_inspector.training import train_classifier


def test_train_classifier_forwards_defaults_and_copies_best_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (dataset / split).mkdir(parents=True)

    calls: dict[str, object] = {}

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            calls["model_name"] = model_name
            self.trainer = SimpleNamespace(best=None)

        def train(self, **arguments: object) -> None:
            calls["arguments"] = arguments
            weights = (
                Path(str(arguments["project"]))
                / str(arguments["name"])
                / "weights"
                / "best.pt"
            )
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"fixture weights")
            self.trainer.best = weights

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    output = train_classifier(
        dataset,
        tmp_path / "models/model.pt",
        run_dir=tmp_path / "runs",
    )

    assert output.read_bytes() == b"fixture weights"
    assert calls["model_name"] == "yolov8n-cls.pt"
    arguments = calls["arguments"]
    assert isinstance(arguments, dict)
    assert arguments["epochs"] == 10
    assert arguments["imgsz"] == 224
    assert arguments["batch"] == 16
    assert arguments["seed"] == 42


def test_train_classifier_requires_all_dataset_splits(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError, match="dataset split"):
        train_classifier(tmp_path)
