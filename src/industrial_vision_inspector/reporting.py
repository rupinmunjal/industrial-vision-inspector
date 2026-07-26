"""CSV and PDF reporting for persisted inspection history."""

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from industrial_vision_inspector.storage import InspectionRecord

CSV_FIELDS = ("id", "timestamp", "image_path", "result", "confidence", "notes")


@dataclass(frozen=True)
class ReportData:
    """Aggregated inspection data consumed by the single PDF template."""

    start_time: datetime
    end_time: datetime
    total_count: int
    pass_count: int
    fail_count: int
    defect_rate: float
    examples: tuple[InspectionRecord, ...]


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


def prepare_report_data(
    records: Iterable[InspectionRecord], *, max_examples: int = 4
) -> ReportData:
    """Aggregate records and choose deterministic examples with existing images."""
    if max_examples < 0:
        raise ValueError("max_examples cannot be negative")
    record_list = list(records)
    if not record_list:
        raise ValueError("cannot create a report without inspections")

    pass_count = sum(record.result == "pass" for record in record_list)
    fail_count = len(record_list) - pass_count
    available = [
        record for record in record_list if Path(record.image_path).is_file()
    ]

    examples: list[InspectionRecord] = []
    for outcome in ("fail", "pass"):
        match = next(
            (record for record in available if record.result == outcome), None
        )
        if match is not None and len(examples) < max_examples:
            examples.append(match)
    for record in available:
        if len(examples) >= max_examples:
            break
        if record not in examples:
            examples.append(record)

    return ReportData(
        start_time=min(record.timestamp for record in record_list),
        end_time=max(record.timestamp for record in record_list),
        total_count=len(record_list),
        pass_count=pass_count,
        fail_count=fail_count,
        defect_rate=fail_count / len(record_list),
        examples=tuple(examples),
    )


def _utc_timestamp(record: InspectionRecord) -> str:
    return (
        record.timestamp.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
