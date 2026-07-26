"""CSV and PDF reporting for persisted inspection history."""

import csv
from collections.abc import Iterable
from datetime import timezone
from pathlib import Path

from industrial_vision_inspector.storage import InspectionRecord

CSV_FIELDS = ("id", "timestamp", "image_path", "result", "confidence", "notes")


def write_inspections_csv(
    records: Iterable[InspectionRecord], output_path: str | Path
) -> Path:
    """Write inspection records in their supplied order and return the output path."""
    path = Path(output_path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record.id,
                    "timestamp": _utc_timestamp(record),
                    "image_path": record.image_path,
                    "result": record.result,
                    "confidence": f"{record.confidence:.6f}",
                    "notes": record.notes or "",
                }
            )
    return path


def _utc_timestamp(record: InspectionRecord) -> str:
    return (
        record.timestamp.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
