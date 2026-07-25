"""PySide6 desktop interface for running and reviewing inspections."""

from __future__ import annotations

import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import cv2
from numpy.typing import NDArray
from PySide6.QtCore import QDate, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from industrial_vision_inspector.ingestion import iter_frames, load_folder, load_image
from industrial_vision_inspector.inspection import InspectionResult, Inspector
from industrial_vision_inspector.storage import InspectionDatabase, InspectionOutcome


class InspectionWorker(QThread):
    """Load and inspect an image or folder without blocking the GUI thread."""

    inspection_ready = Signal(str, object)
    progress_changed = Signal(int, int)
    failed = Signal(str)

    def __init__(
        self,
        inspector: Inspector,
        database: InspectionDatabase,
        source: Path,
        *,
        folder: bool,
        notes: str | None,
    ) -> None:
        super().__init__()
        self.inspector = inspector
        self.database = database
        self.source = source
        self.folder = folder
        self.notes = notes

    def run(self) -> None:
        try:
            if self.folder:
                items = load_folder(self.source)
            else:
                items = [(self.source, load_image(self.source))]
            if not items:
                raise ValueError(f"folder contains no supported images: {self.source}")

            total = len(items)
            for index, (path, image) in enumerate(items, start=1):
                if self.isInterruptionRequested():
                    return
                result = self.inspector.inspect(image)
                self.database.insert_inspection(
                    path,
                    result.result,
                    result.confidence,
                    notes=self.notes,
                )
                self.inspection_ready.emit(str(path), result)
                self.progress_changed.emit(index, total)
        except Exception as error:
            self.failed.emit(str(error))


class ImagePreview(QLabel):
    """A label that keeps an OpenCV image scaled to its available space."""

    def __init__(self) -> None:
        super().__init__("Load an image, folder, or webcam to begin")
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(540, 480)
        self.setStyleSheet(
            "background: #1f2933; color: #d9e2ec; border: 1px solid #52606d;"
        )

    def show_image(self, image: NDArray) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self._update_pixmap()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class InspectView(QWidget):
    """Run inspections from image, folder, or webcam sources."""

    history_changed = Signal()
    batch_finished = Signal()

    def __init__(
        self,
        inspector: Inspector,
        database: InspectionDatabase,
        capture_dir: Path,
        *,
        camera_index: int = 0,
    ) -> None:
        super().__init__()
        self.inspector = inspector
        self.database = database
        self.capture_dir = capture_dir
        self.camera_index = camera_index
        self._worker: InspectionWorker | None = None
        self._frame_iterator = None
        self._current_frame: NDArray | None = None

        self.preview = ImagePreview()
        self.source_label = QLabel("No source selected")
        self.source_label.setWordWrap(True)
        self.badge = QLabel("NOT INSPECTED")
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setStyleSheet(
            "font-size: 20px; font-weight: bold; padding: 12px; "
            "background: #d9e2ec; color: #243b53; border-radius: 4px;"
        )
        self.confidence_label = QLabel("Confidence: --")
        self.notes_input = QLineEdit()
        self.notes_input.setPlaceholderText("Optional operator notes")
        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)

        self.load_image_button = QPushButton("Load Image")
        self.load_folder_button = QPushButton("Load Folder")
        self.webcam_button = QPushButton("Start Webcam")
        self.inspect_frame_button = QPushButton("Inspect Frame")
        self.inspect_frame_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

        self.load_image_button.clicked.connect(self._choose_image)
        self.load_folder_button.clicked.connect(self._choose_folder)
        self.webcam_button.clicked.connect(self._toggle_webcam)
        self.inspect_frame_button.clicked.connect(self._inspect_current_frame)

        controls = QGroupBox("Inspection controls")
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(self.source_label)
        controls_layout.addWidget(self.badge)
        controls_layout.addWidget(self.confidence_label)
        controls_layout.addWidget(self.notes_input)

        button_grid = QGridLayout()
        button_grid.addWidget(self.load_image_button, 0, 0)
        button_grid.addWidget(self.load_folder_button, 0, 1)
        button_grid.addWidget(self.webcam_button, 1, 0)
        button_grid.addWidget(self.inspect_frame_button, 1, 1)
        controls_layout.addLayout(button_grid)
        controls_layout.addWidget(self.progress)
        controls_layout.addWidget(self.message_label)
        controls_layout.addStretch()

        layout = QHBoxLayout(self)
        layout.addWidget(self.preview, stretch=3)
        layout.addWidget(controls, stretch=1)

        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(33)
        self._frame_timer.timeout.connect(self._read_webcam_frame)

    def inspect_path(self, path: str | Path) -> None:
        self.stop_webcam()
        self._start_worker(Path(path).resolve(), folder=False)

    def inspect_folder(self, folder: str | Path) -> None:
        self.stop_webcam()
        self._start_worker(Path(folder).resolve(), folder=True)

    def inspect_webcam_frame(self, frame: NDArray) -> Path:
        """Save and inspect one webcam frame, returning its durable path."""
        self.stop_webcam()
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = (self.capture_dir / f"webcam_{timestamp}.jpg").resolve()
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"could not save webcam capture: {path}")
        self._start_worker(path, folder=False)
        return path

    def start_webcam(self) -> None:
        if self._worker is not None:
            self._show_error("Wait for the current inspection to finish")
            return
        if self._frame_iterator is not None:
            return
        self._frame_iterator = iter_frames(self.camera_index)
        self._current_frame = None
        self.webcam_button.setText("Stop Webcam")
        self.inspect_frame_button.setEnabled(False)
        self.message_label.setText("Opening webcam...")
        self._frame_timer.start()

    def stop_webcam(self) -> None:
        self._frame_timer.stop()
        if self._frame_iterator is not None:
            self._frame_iterator.close()
        self._frame_iterator = None
        self._current_frame = None
        self.webcam_button.setText("Start Webcam")
        self.inspect_frame_button.setEnabled(False)

    def shutdown(self) -> None:
        self.stop_webcam()
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait()

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose component image",
            "",
            "Images (*.bmp *.jpeg *.jpg *.pgm *.png *.ppm *.tif *.tiff *.webp)",
        )
        if path:
            self.inspect_path(path)

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose inspection folder")
        if folder:
            self.inspect_folder(folder)

    def _toggle_webcam(self) -> None:
        if self._frame_iterator is None:
            self.start_webcam()
        else:
            self.stop_webcam()
            self.message_label.setText("Webcam stopped")

    def _read_webcam_frame(self) -> None:
        try:
            frame = next(self._frame_iterator)
        except StopIteration:
            self.stop_webcam()
            self._show_error("Webcam stream ended")
            return
        except Exception as error:
            self.stop_webcam()
            self._show_error(str(error))
            return

        self._current_frame = frame.copy()
        self.preview.show_image(frame)
        self.source_label.setText(f"Webcam {self.camera_index} preview")
        self.inspect_frame_button.setEnabled(True)
        self.message_label.setText("Previewing webcam. Capture a frame to inspect it.")

    def _inspect_current_frame(self) -> None:
        if self._current_frame is None:
            self._show_error("No webcam frame is available")
            return
        frame = self._current_frame.copy()
        try:
            self.inspect_webcam_frame(frame)
        except Exception as error:
            self._show_error(str(error))

    def _start_worker(self, source: Path, *, folder: bool) -> None:
        if self._worker is not None:
            self._show_error("Wait for the current inspection to finish")
            return

        notes = self.notes_input.text().strip() or None
        worker = InspectionWorker(
            self.inspector,
            self.database,
            source,
            folder=folder,
            notes=notes,
        )
        worker.inspection_ready.connect(self._show_result)
        worker.progress_changed.connect(self._show_progress)
        worker.failed.connect(self._show_error)
        worker.finished.connect(self._worker_finished)
        self._worker = worker

        self._set_busy(True)
        self.source_label.setText(str(source))
        self.message_label.setText("Inspecting folder..." if folder else "Inspecting image...")
        self.progress.setRange(0, 0)
        worker.start()

    def _show_result(self, path: str, result: InspectionResult) -> None:
        self.preview.show_image(result.annotated_image)
        self.source_label.setText(path)
        self.confidence_label.setText(f"Confidence: {result.confidence:.1%}")
        if result.result == "pass":
            self.badge.setText("PASS")
            self.badge.setStyleSheet(
                "font-size: 20px; font-weight: bold; padding: 12px; "
                "background: #d1fae5; color: #065f46; border-radius: 4px;"
            )
        else:
            self.badge.setText("FAIL")
            self.badge.setStyleSheet(
                "font-size: 20px; font-weight: bold; padding: 12px; "
                "background: #fee2e2; color: #991b1b; border-radius: 4px;"
            )
        self.message_label.setStyleSheet("")
        self.message_label.setText("Inspection saved to history")
        self.history_changed.emit()

    def _show_progress(self, completed: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(completed)

    def _show_error(self, message: str) -> None:
        self.message_label.setStyleSheet("color: #b91c1c;")
        self.message_label.setText(message)

    def _worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        self._set_busy(False)
        if worker is not None:
            worker.deleteLater()
        self.batch_finished.emit()

    def _set_busy(self, busy: bool) -> None:
        self.load_image_button.setEnabled(not busy)
        self.load_folder_button.setEnabled(not busy)
        self.webcam_button.setEnabled(not busy)
        self.notes_input.setEnabled(not busy)
        self.inspect_frame_button.setEnabled(False)


class HistoryView(QWidget):
    """Display and filter persisted inspection records."""

    def __init__(self, database: InspectionDatabase) -> None:
        super().__init__()
        self.database = database

        self.result_filter = QComboBox()
        self.result_filter.addItem("All results", None)
        self.result_filter.addItem("Pass", "pass")
        self.result_filter.addItem("Fail", "fail")

        self.date_filter = QCheckBox("Limit to UTC date range")
        today = QDate.currentDate()
        self.from_date = QDateEdit(today.addMonths(-1))
        self.to_date = QDateEdit(today)
        for date_edit in (self.from_date, self.to_date):
            date_edit.setCalendarPopup(True)
            date_edit.setDisplayFormat("yyyy-MM-dd")
            date_edit.setEnabled(False)
        self.date_filter.toggled.connect(self.from_date.setEnabled)
        self.date_filter.toggled.connect(self.to_date.setEnabled)

        self.apply_button = QPushButton("Apply Filters")
        self.clear_button = QPushButton("Clear Filters")
        self.refresh_button = QPushButton("Refresh")
        self.apply_button.clicked.connect(self.refresh)
        self.clear_button.clicked.connect(self.clear_filters)
        self.refresh_button.clicked.connect(self.refresh)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Result:"))
        filters.addWidget(self.result_filter)
        filters.addWidget(self.date_filter)
        filters.addWidget(QLabel("From:"))
        filters.addWidget(self.from_date)
        filters.addWidget(QLabel("To:"))
        filters.addWidget(self.to_date)
        filters.addWidget(self.apply_button)
        filters.addWidget(self.clear_button)
        filters.addWidget(self.refresh_button)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Timestamp (UTC)", "Image", "Result", "Confidence", "Notes"]
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        self.message_label = QLabel("")
        layout = QVBoxLayout(self)
        layout.addLayout(filters)
        layout.addWidget(self.table)
        layout.addWidget(self.message_label)
        self.refresh()

    def refresh(self) -> None:
        result: InspectionOutcome | None = self.result_filter.currentData()
        start = None
        end = None
        if self.date_filter.isChecked():
            start_date = self.from_date.date().toPython()
            end_date = self.to_date.date().toPython()
            if start_date > end_date:
                self.message_label.setStyleSheet("color: #b91c1c;")
                self.message_label.setText("From date must not be after To date")
                return
            start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
            end = datetime.combine(
                end_date + timedelta(days=1), time.min, tzinfo=timezone.utc
            )

        try:
            records = self.database.list_inspections(
                result=result,
                start=start,
                end=end,
            )
        except Exception as error:
            self.message_label.setStyleSheet("color: #b91c1c;")
            self.message_label.setText(str(error))
            return

        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = (
                str(record.id),
                record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                record.image_path,
                record.result.upper(),
                f"{record.confidence:.1%}",
                record.notes or "",
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))

        self.message_label.setStyleSheet("")
        self.message_label.setText(f"{len(records)} inspection(s)")

    def clear_filters(self) -> None:
        self.result_filter.setCurrentIndex(0)
        self.date_filter.setChecked(False)
        self.refresh()


class InspectionWindow(QMainWindow):
    """Main application window containing inspection and history views."""

    def __init__(
        self,
        inspector: Inspector,
        database: InspectionDatabase,
        capture_dir: Path,
        *,
        camera_index: int = 0,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Industrial Vision Inspector")
        self.resize(1120, 720)

        self.inspect_view = InspectView(
            inspector,
            database,
            capture_dir,
            camera_index=camera_index,
        )
        self.history_view = HistoryView(database)
        self.inspect_view.history_changed.connect(self.history_view.refresh)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.inspect_view, "Inspect")
        self.tabs.addTab(self.history_view, "History")
        self.setCentralWidget(self.tabs)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.inspect_view.shutdown()
        super().closeEvent(event)


def run_app(
    model_path: str | Path,
    database_path: str | Path,
    *,
    camera_index: int = 0,
    smoke_test: bool = False,
) -> int:
    """Create and run the desktop application."""
    database_file = Path(database_path)
    database_file.parent.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv)
    window = InspectionWindow(
        Inspector(model_path),
        InspectionDatabase(database_file),
        database_file.parent / "captures",
        camera_index=camera_index,
    )
    window.show()
    if smoke_test:
        QTimer.singleShot(250, window.close)
    return app.exec()
