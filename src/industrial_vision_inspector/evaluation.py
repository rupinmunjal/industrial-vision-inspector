"""Held-out classification metrics and report plots."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from industrial_vision_inspector.ingestion import IMAGE_EXTENSIONS, load_image
from industrial_vision_inspector.inspection import InspectionResult, Inspector

CLASS_ORDER = ("ok_front", "defective")


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    sample_count: int


@dataclass(frozen=True)
class EvaluatedSample:
    path: Path
    expected_class: str
    inspection: InspectionResult


def compute_metrics(
    expected: list[str], predicted: list[str]
) -> ClassificationMetrics:
    """Compute binary metrics with ``defective`` as the positive class."""
    if not expected or len(expected) != len(predicted):
        raise ValueError("expected and predicted must be non-empty and equally sized")
    unknown = (set(expected) | set(predicted)) - set(CLASS_ORDER)
    if unknown:
        raise ValueError(f"unsupported classes: {sorted(unknown)}")

    true_negative = false_positive = false_negative = true_positive = 0
    for actual, guess in zip(expected, predicted, strict=True):
        if actual == "ok_front" and guess == "ok_front":
            true_negative += 1
        elif actual == "ok_front":
            false_positive += 1
        elif guess == "ok_front":
            false_negative += 1
        else:
            true_positive += 1

    total = len(expected)
    accuracy = (true_positive + true_negative) / total
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=(
            (true_negative, false_positive),
            (false_negative, true_positive),
        ),
        sample_count=total,
    )


def evaluate_test_directory(
    inspector: Inspector, test_dir: str | Path
) -> tuple[ClassificationMetrics, list[EvaluatedSample]]:
    """Run one inspector over ``test/<class>`` folders."""
    test_path = Path(test_dir)
    samples: list[EvaluatedSample] = []
    for class_name in CLASS_ORDER:
        class_dir = test_path / class_name
        if not class_dir.is_dir():
            raise NotADirectoryError(f"test class folder does not exist: {class_dir}")
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"test class folder contains no supported images: {class_dir}")
        for image_path in image_paths:
            inspection = inspector.inspect(load_image(image_path))
            samples.append(EvaluatedSample(image_path, class_name, inspection))

    metrics = compute_metrics(
        [sample.expected_class for sample in samples],
        [sample.inspection.classification for sample in samples],
    )
    return metrics, samples


def save_metrics(metrics: ClassificationMetrics, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = asdict(metrics)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_confusion_matrix(
    metrics: ClassificationMetrics, output_path: str | Path
) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(5, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    matrix = metrics.confusion_matrix
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Casting classification confusion matrix",
        xlabel="Predicted class",
        ylabel="Actual class",
        xticks=(0, 1),
        yticks=(0, 1),
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
    )
    threshold = max(max(row) for row in matrix) / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.savefig(path, dpi=150)
    return path


def save_prediction_samples(
    samples: list[EvaluatedSample], output_path: str | Path, *, per_group: int = 4
) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if per_group <= 0:
        raise ValueError("per_group must be positive")
    correct = [
        sample
        for sample in samples
        if sample.expected_class == sample.inspection.classification
    ][:per_group]
    incorrect = [
        sample
        for sample in samples
        if sample.expected_class != sample.inspection.classification
    ][:per_group]
    selected = correct + incorrect
    if not selected:
        raise ValueError("at least one evaluated sample is required")

    columns = min(4, len(selected))
    rows = (len(selected) + columns - 1) // columns
    figure = Figure(figsize=(4 * columns, 3.5 * rows), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, sample in zip(axes.flat, selected, strict=False):
        rgb = cv2.cvtColor(sample.inspection.annotated_image, cv2.COLOR_BGR2RGB)
        correct_label = sample.expected_class == sample.inspection.classification
        axis.imshow(rgb)
        axis.set_title(
            f"{'correct' if correct_label else 'incorrect'} | "
            f"actual={sample.expected_class}\n"
            f"predicted={sample.inspection.classification} "
            f"({sample.inspection.confidence:.1%})"
        )
        axis.axis("off")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    return path


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
