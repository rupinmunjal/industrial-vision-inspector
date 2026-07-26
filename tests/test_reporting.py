import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from industrial_vision_inspector.reporting import (
    CSV_FIELDS,
    prepare_report_data,
    write_inspections_csv,
)
from industrial_vision_inspector.storage import InspectionDatabase

FIXTURES = Path(__file__).parent / "fixtures"


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


def test_report_data_aggregates_counts_range_and_existing_examples(
    tmp_path: Path,
) -> None:
    database = InspectionDatabase(tmp_path / "inspections.db")
    base = datetime(2026, 7, 26, 9, tzinfo=timezone.utc)
    database.insert_inspection(
        FIXTURES / "ok_sample.ppm", "pass", 0.91, timestamp=base
    )
    database.insert_inspection(
        tmp_path / "missing.jpg",
        "fail",
        0.82,
        timestamp=base + timedelta(hours=1),
    )
    database.insert_inspection(
        FIXTURES / "defective_sample.ppm",
        "fail",
        0.97,
        timestamp=base + timedelta(hours=2),
    )

    report = prepare_report_data(database.list_inspections())

    assert report.start_time == base
    assert report.end_time == base + timedelta(hours=2)
    assert report.total_count == 3
    assert report.pass_count == 1
    assert report.fail_count == 2
    assert report.defect_rate == pytest.approx(2 / 3)
    assert [Path(record.image_path).name for record in report.examples] == [
        "defective_sample.ppm",
        "ok_sample.ppm",
    ]


def test_report_data_validates_empty_input_and_example_limit() -> None:
    with pytest.raises(ValueError, match="without inspections"):
        prepare_report_data([])
    with pytest.raises(ValueError, match="max_examples"):
        prepare_report_data([], max_examples=-1)
