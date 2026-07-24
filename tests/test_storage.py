import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from industrial_vision_inspector.storage import InspectionDatabase


def test_database_initialization_is_idempotent_and_creates_expected_schema(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inspections.db"

    InspectionDatabase(database_path)
    InspectionDatabase(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = connection.execute("PRAGMA table_info(inspections)").fetchall()
    assert [column[1] for column in columns] == [
        "id",
        "timestamp",
        "image_path",
        "result",
        "confidence",
        "notes",
    ]


def test_insert_round_trip_preserves_fields_and_normalizes_timestamp(
    tmp_path: Path,
) -> None:
    database = InspectionDatabase(tmp_path / "inspections.db")
    eastern = timezone(timedelta(hours=-4))
    timestamp = datetime(2026, 7, 25, 10, 30, tzinfo=eastern)

    inserted = database.insert_inspection(
        "images/part-001.jpg",
        "fail",
        0.987,
        notes="Surface void near outer edge",
        timestamp=timestamp,
    )
    stored = database.list_inspections()[0]

    assert inserted.id == 1
    assert stored == inserted
    assert stored.timestamp == datetime(2026, 7, 25, 14, 30, tzinfo=timezone.utc)


def test_default_timestamp_is_utc_and_record_persists_after_reopen(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inspections.db"
    before = datetime.now(timezone.utc)
    inserted = InspectionDatabase(database_path).insert_inspection(
        Path("images/part-002.jpg"), "pass", 0.91
    )
    after = datetime.now(timezone.utc)

    stored = InspectionDatabase(database_path).list_inspections()[0]

    assert before <= inserted.timestamp <= after
    assert inserted.timestamp.tzinfo is timezone.utc
    assert stored == inserted
    assert stored.notes is None


def test_list_orders_newest_first_and_applies_result_and_date_filters(
    tmp_path: Path,
) -> None:
    database = InspectionDatabase(tmp_path / "inspections.db")
    base = datetime(2026, 7, 25, 9, tzinfo=timezone.utc)
    oldest = database.insert_inspection("a.jpg", "pass", 0.8, timestamp=base)
    middle = database.insert_inspection(
        "b.jpg", "fail", 0.9, timestamp=base + timedelta(hours=1)
    )
    newest = database.insert_inspection(
        "c.jpg", "pass", 0.95, timestamp=base + timedelta(hours=2)
    )

    assert database.list_inspections() == [newest, middle, oldest]
    assert database.list_inspections(result="pass") == [newest, oldest]
    assert database.list_inspections(
        start=base + timedelta(hours=1),
        end=base + timedelta(hours=2),
    ) == [middle]


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_insert_rejects_invalid_confidence(
    tmp_path: Path, confidence: float
) -> None:
    database = InspectionDatabase(tmp_path / "inspections.db")

    with pytest.raises(ValueError, match="confidence"):
        database.insert_inspection("part.jpg", "pass", confidence)


def test_database_rejects_invalid_values_and_date_ranges(tmp_path: Path) -> None:
    database = InspectionDatabase(tmp_path / "inspections.db")
    aware = datetime(2026, 7, 25, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="image_path"):
        database.insert_inspection("   ", "pass", 0.9)
    with pytest.raises(ValueError, match="result"):
        database.insert_inspection("part.jpg", "unknown", 0.9)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timezone"):
        database.insert_inspection(
            "part.jpg", "pass", 0.9, timestamp=datetime(2026, 7, 25)
        )
    with pytest.raises(ValueError, match="start"):
        database.list_inspections(start=aware, end=aware)
