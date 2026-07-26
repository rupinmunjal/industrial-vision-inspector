import csv
from datetime import datetime, timezone
from pathlib import Path

from industrial_vision_inspector.reporting import CSV_FIELDS, write_inspections_csv
from industrial_vision_inspector.storage import InspectionDatabase


def test_csv_export_writes_storage_rows_in_history_order(tmp_path: Path) -> None:
    database = InspectionDatabase(tmp_path / "inspections.db")
    database.insert_inspection(
        "images/accepted.jpg",
        "pass",
        0.9123456,
        timestamp=datetime(2026, 7, 26, 8, tzinfo=timezone.utc),
    )
    database.insert_inspection(
        "images/defective.jpg",
        "fail",
        0.9876543,
        notes="Surface void",
        timestamp=datetime(2026, 7, 26, 9, 30, tzinfo=timezone.utc),
    )
    output = tmp_path / "history.csv"

    returned_path = write_inspections_csv(database.list_inspections(), output)

    with output.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert returned_path == output
    assert tuple(reader.fieldnames or ()) == CSV_FIELDS
    assert rows == [
        {
            "id": "2",
            "timestamp": "2026-07-26T09:30:00.000000Z",
            "image_path": "images/defective.jpg",
            "result": "fail",
            "confidence": "0.987654",
            "notes": "Surface void",
        },
        {
            "id": "1",
            "timestamp": "2026-07-26T08:00:00.000000Z",
            "image_path": "images/accepted.jpg",
            "result": "pass",
            "confidence": "0.912346",
            "notes": "",
        },
    ]


def test_csv_export_with_no_records_writes_only_header(tmp_path: Path) -> None:
    output = write_inspections_csv([], tmp_path / "empty.csv")

    assert output.read_text(encoding="utf-8").strip() == ",".join(CSV_FIELDS)
