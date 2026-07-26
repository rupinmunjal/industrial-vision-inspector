import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QDate
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from industrial_vision_inspector.ingestion import load_image
from industrial_vision_inspector.inspection import Inspector
from industrial_vision_inspector.storage import InspectionDatabase
from industrial_vision_inspector.ui import HistoryView, InspectionWindow

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_predictor(image: np.ndarray) -> tuple[str, float]:
    red_mean = float(image[:, :, 2].mean())
    green_mean = float(image[:, :, 1].mean())
    return ("defective", 0.96) if red_mean > green_mean else ("ok_front", 0.93)


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp: QApplication, tmp_path: Path) -> InspectionWindow:
    inspection_window = InspectionWindow(
        Inspector(predictor=fixture_predictor),
        InspectionDatabase(tmp_path / "inspections.db"),
        tmp_path / "captures",
    )
    yield inspection_window
    inspection_window.close()
    qapp.processEvents()


def wait_for_batch(spy: QSignalSpy, qapp: QApplication) -> None:
    deadline = time.monotonic() + 5
    while spy.count() == 0 and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert spy.count() == 1
    qapp.processEvents()


def test_window_launches_with_two_views(
    window: InspectionWindow, qapp: QApplication
) -> None:
    window.show()
    qapp.processEvents()

    assert window.isVisible()
    assert window.tabs.count() == 2
    assert window.tabs.tabText(0) == "Inspect"
    assert window.tabs.tabText(1) == "History"


def test_single_inspection_updates_ui_and_history(
    window: InspectionWindow, qapp: QApplication
) -> None:
    window.inspect_view.notes_input.setText("Check outer edge")
    finished = QSignalSpy(window.inspect_view.batch_finished)

    window.inspect_view.inspect_path(FIXTURES / "defective_sample.ppm")
    wait_for_batch(finished, qapp)

    records = window.inspect_view.database.list_inspections()
    assert len(records) == 1
    assert records[0].result == "fail"
    assert records[0].notes == "Check outer edge"
    assert window.inspect_view.badge.text() == "FAIL"
    assert window.inspect_view.confidence_label.text() == "Confidence: 96.0%"
    assert window.history_view.table.rowCount() == 1
    assert window.inspect_view.load_image_button.isEnabled()


def test_folder_batch_inspects_every_fixture(
    window: InspectionWindow, qapp: QApplication
) -> None:
    finished = QSignalSpy(window.inspect_view.batch_finished)

    window.inspect_view.inspect_folder(FIXTURES)
    wait_for_batch(finished, qapp)

    records = window.inspect_view.database.list_inspections()
    assert len(records) == 2
    assert {record.result for record in records} == {"pass", "fail"}
    assert window.inspect_view.progress.maximum() == 2
    assert window.inspect_view.progress.value() == 2
    assert window.history_view.table.rowCount() == 2


def test_worker_error_is_visible_and_restores_controls(
    window: InspectionWindow, qapp: QApplication, tmp_path: Path
) -> None:
    finished = QSignalSpy(window.inspect_view.batch_finished)

    window.inspect_view.inspect_path(tmp_path / "missing.jpg")
    wait_for_batch(finished, qapp)

    assert "does not exist" in window.inspect_view.message_label.text()
    assert window.inspect_view.load_image_button.isEnabled()
    assert window.inspect_view.database.list_inspections() == []


def test_webcam_frame_is_saved_before_inspection(
    window: InspectionWindow, qapp: QApplication
) -> None:
    finished = QSignalSpy(window.inspect_view.batch_finished)
    frame = load_image(FIXTURES / "ok_sample.ppm")

    capture_path = window.inspect_view.inspect_webcam_frame(frame)
    wait_for_batch(finished, qapp)

    records = window.inspect_view.database.list_inspections()
    assert capture_path.is_file()
    assert records[0].image_path == str(capture_path)
    assert records[0].result == "pass"


def test_history_filters_by_result_and_inclusive_utc_dates(
    qapp: QApplication, tmp_path: Path
) -> None:
    database = InspectionDatabase(tmp_path / "inspections.db")
    database.insert_inspection(
        "older.jpg",
        "pass",
        0.9,
        timestamp=datetime(2026, 7, 23, 12, tzinfo=timezone.utc),
    )
    database.insert_inspection(
        "failure.jpg",
        "fail",
        0.8,
        timestamp=datetime(2026, 7, 24, 10, tzinfo=timezone.utc),
    )
    database.insert_inspection(
        "accepted.jpg",
        "pass",
        0.95,
        timestamp=datetime(2026, 7, 24, 20, tzinfo=timezone.utc),
    )
    view = HistoryView(database)

    view.result_filter.setCurrentIndex(view.result_filter.findData("pass"))
    view.date_filter.setChecked(True)
    view.from_date.setDate(QDate(2026, 7, 24))
    view.to_date.setDate(QDate(2026, 7, 24))
    view.refresh()
    qapp.processEvents()

    assert view.table.rowCount() == 1
    assert view.table.item(0, 2).text() == "accepted.jpg"
    assert view.table.item(0, 3).text() == "PASS"
    view.close()


def test_history_exports_current_filter_to_csv_and_pdf(
    qapp: QApplication, tmp_path: Path
) -> None:
    database = InspectionDatabase(tmp_path / "inspections.db")
    timestamp = datetime(2026, 7, 26, 15, tzinfo=timezone.utc)
    database.insert_inspection(
        FIXTURES / "ok_sample.ppm", "pass", 0.93, timestamp=timestamp
    )
    database.insert_inspection(
        FIXTURES / "defective_sample.ppm", "fail", 0.96, timestamp=timestamp
    )
    view = HistoryView(database)
    view.result_filter.setCurrentIndex(view.result_filter.findData("fail"))

    csv_path = view.export_csv(tmp_path / "filtered")
    pdf_path = view.export_pdf(tmp_path / "filtered")
    qapp.processEvents()

    assert csv_path == tmp_path / "filtered.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["result"] == "fail"
    assert pdf_path == tmp_path / "filtered.pdf"
    assert pdf_path.stat().st_size > 2_000
    view.close()
