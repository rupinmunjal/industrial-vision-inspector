from pathlib import Path

import pytest

from industrial_vision_inspector.evaluation import (
    compute_metrics,
    evaluate_test_directory,
    save_confusion_matrix,
    save_metrics,
    save_prediction_samples,
)
from industrial_vision_inspector.inspection import Inspector
from tests.test_inspection import FIXTURES, fixture_predictor


def test_compute_metrics_uses_defective_as_positive_class() -> None:
    metrics = compute_metrics(
        ["defective", "defective", "defective", "ok_front", "ok_front"],
        ["defective", "defective", "ok_front", "defective", "ok_front"],
    )

    assert metrics.accuracy == pytest.approx(0.6)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.confusion_matrix == ((1, 1), (1, 2))


def test_evaluation_saves_metrics_and_plots(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    for class_name, fixture_name in (
        ("defective", "defective_sample.ppm"),
        ("ok_front", "ok_sample.ppm"),
    ):
        class_dir = test_dir / class_name
        class_dir.mkdir(parents=True)
        (class_dir / fixture_name).write_bytes((FIXTURES / fixture_name).read_bytes())

    metrics, samples = evaluate_test_directory(
        Inspector(predictor=fixture_predictor), test_dir
    )
    metrics_path = save_metrics(metrics, tmp_path / "metrics.json")
    matrix_path = save_confusion_matrix(metrics, tmp_path / "confusion_matrix.png")
    samples_path = save_prediction_samples(samples, tmp_path / "samples.png")

    assert metrics.accuracy == 1.0
    assert metrics.sample_count == 2
    assert metrics_path.stat().st_size > 100
    assert matrix_path.stat().st_size > 1_000
    assert samples_path.stat().st_size > 1_000
