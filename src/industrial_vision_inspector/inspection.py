"""Classification-first defect inspection with approximate CV highlighting."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

Prediction = Callable[[NDArray[np.uint8]], tuple[str, float]]
DefectRegion = tuple[int, int, int, int]


@dataclass(frozen=True)
class InspectionResult:
    """One classification decision and its display-ready image."""

    classification: Literal["defective", "ok_front"]
    result: Literal["fail", "pass"]
    confidence: float
    annotated_image: NDArray[np.uint8]
    defect_region: DefectRegion | None


class Inspector:
    """Run YOLO classification and convert its output into a QC decision."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        predictor: Prediction | None = None,
    ) -> None:
        if predictor is not None:
            self._predict = predictor
            self._model = None
            return

        if model_path is None:
            raise ValueError("model_path is required when no predictor is supplied")
        weights = Path(model_path)
        if not weights.is_file():
            raise FileNotFoundError(f"model weights do not exist: {weights}")

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required for model inference. Install it with: "
                "pip install ultralytics"
            ) from error

        self._model = YOLO(str(weights))
        self._predict = self._predict_with_ultralytics

    def inspect(self, image: NDArray[np.generic]) -> InspectionResult:
        """Classify one OpenCV image and prepare the image shown by the UI."""
        display_image = _as_bgr_uint8(image)
        raw_class, confidence = self._predict(display_image)
        classification = _canonical_class(raw_class)
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

        if classification == "defective":
            annotated, region = highlight_defect_region(display_image)
            result = "fail"
        else:
            annotated = display_image.copy()
            region = None
            result = "pass"

        return InspectionResult(
            classification=classification,
            result=result,
            confidence=confidence,
            annotated_image=annotated,
            defect_region=region,
        )

    def _predict_with_ultralytics(
        self, image: NDArray[np.uint8]
    ) -> tuple[str, float]:
        predictions = self._model.predict(source=image, verbose=False)
        if len(predictions) != 1 or predictions[0].probs is None:
            raise RuntimeError("Ultralytics did not return one classification result")

        prediction = predictions[0]
        class_index = int(prediction.probs.top1)
        confidence = float(prediction.probs.top1conf.item())
        names = prediction.names or self._model.names
        class_name = names[class_index]
        return str(class_name), confidence


def highlight_defect_region(
    image: NDArray[np.generic],
) -> tuple[NDArray[np.uint8], DefectRegion | None]:
    """Draw a rough box around the strongest enclosed dark region.

    This is visual feedback only. It is not a learned localization result.
    """
    annotated = _as_bgr_uint8(image)
    gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
    if min(gray.shape) >= 5:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape
    image_area = height * width
    candidates: list[tuple[int, DefectRegion]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area = box_width * box_height
        touches_border = (
            x == 0
            or y == 0
            or x + box_width == width
            or y + box_height == height
        )
        if touches_border or box_area < max(1, image_area * 0.002):
            continue
        if box_area > image_area * 0.5:
            continue
        candidates.append((box_area, (x, y, box_width, box_height)))

    if not candidates:
        return annotated.copy(), None

    _, region = max(candidates, key=lambda candidate: candidate[0])
    x, y, box_width, box_height = region
    result = annotated.copy()
    thickness = max(1, round(min(width, height) / 150))
    cv2.rectangle(
        result,
        (x, y),
        (x + box_width - 1, y + box_height - 1),
        (0, 0, 255),
        thickness,
    )
    return result, region


def _canonical_class(raw_class: str) -> Literal["defective", "ok_front"]:
    normalized = raw_class.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"defective", "def_front"}:
        return "defective"
    if normalized in {"ok", "ok_front"}:
        return "ok_front"
    raise ValueError(f"unsupported model class: {raw_class!r}")


def _as_bgr_uint8(image: NDArray[np.generic]) -> NDArray[np.uint8]:
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")

    converted = image
    if converted.dtype != np.uint8:
        converted = converted.astype(np.float32)
        if float(converted.max()) <= 1.0:
            converted = converted * 255.0
        converted = np.clip(converted, 0, 255).astype(np.uint8)

    if converted.ndim == 2:
        return cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR)
    if converted.ndim != 3:
        raise ValueError(f"expected a 2D or 3D image, got {converted.ndim} dimensions")
    if converted.shape[2] == 3:
        return converted.copy()
    if converted.shape[2] == 4:
        return cv2.cvtColor(converted, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"unsupported channel count: {converted.shape[2]}")
