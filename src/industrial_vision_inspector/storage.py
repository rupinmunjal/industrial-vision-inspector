"""SQLite persistence for inspection history."""

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

InspectionOutcome = Literal["pass", "fail"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    image_path TEXT NOT NULL CHECK (length(trim(image_path)) > 0),
    result TEXT NOT NULL CHECK (result IN ('pass', 'fail')),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    notes TEXT
)
"""


@dataclass(frozen=True)
class InspectionRecord:
    """One persisted quality-control inspection."""

    id: int
    timestamp: datetime
    image_path: str
    result: InspectionOutcome
    confidence: float
    notes: str | None


class InspectionDatabase:
    """Thin data-access layer for the local inspection history database."""

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self._initialize()

    def insert_inspection(
        self,
        image_path: str | Path,
        result: InspectionOutcome,
        confidence: float,
        *,
        notes: str | None = None,
        timestamp: datetime | None = None,
    ) -> InspectionRecord:
        """Insert one inspection and return it with its generated ID."""
        stored_path = str(image_path)
        if not stored_path.strip():
            raise ValueError("image_path cannot be empty")
        _validate_result(result)
        stored_confidence = float(confidence)
        if not math.isfinite(stored_confidence) or not 0.0 <= stored_confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")

        stored_timestamp = _normalize_timestamp(timestamp or datetime.now(timezone.utc))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO inspections (timestamp, image_path, result, confidence, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    _serialize_timestamp(stored_timestamp),
                    stored_path,
                    result,
                    stored_confidence,
                    notes,
                ),
            )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return an inspection ID")

        return InspectionRecord(
            id=cursor.lastrowid,
            timestamp=stored_timestamp,
            image_path=stored_path,
            result=result,
            confidence=stored_confidence,
            notes=notes,
        )

    def list_inspections(
        self,
        *,
        result: InspectionOutcome | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[InspectionRecord]:
        """Return matching inspections newest-first using a ``[start, end)`` range."""
        clauses: list[str] = []
        parameters: list[str] = []
        if result is not None:
            _validate_result(result)
            clauses.append("result = ?")
            parameters.append(result)

        normalized_start = _normalize_timestamp(start) if start is not None else None
        normalized_end = _normalize_timestamp(end) if end is not None else None
        if normalized_start is not None and normalized_end is not None:
            if normalized_start >= normalized_end:
                raise ValueError("start must be earlier than end")
        if normalized_start is not None:
            clauses.append("timestamp >= ?")
            parameters.append(_serialize_timestamp(normalized_start))
        if normalized_end is not None:
            clauses.append("timestamp < ?")
            parameters.append(_serialize_timestamp(normalized_end))

        query = "SELECT id, timestamp, image_path, result, confidence, notes FROM inspections"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp DESC, id DESC"

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_record_from_row(row) for row in rows]

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


def _validate_result(result: str) -> None:
    if result not in {"pass", "fail"}:
        raise ValueError("result must be 'pass' or 'fail'")


def _normalize_timestamp(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must include timezone information")
    return timestamp.astimezone(timezone.utc)


def _serialize_timestamp(timestamp: datetime) -> str:
    return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _record_from_row(row: sqlite3.Row) -> InspectionRecord:
    return InspectionRecord(
        id=row["id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        image_path=row["image_path"],
        result=row["result"],
        confidence=row["confidence"],
        notes=row["notes"],
    )
