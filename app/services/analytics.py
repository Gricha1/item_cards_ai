from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


class AnalyticsStore:
    """Small local audit store for bot usage; no message text or images are retained."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    telegram_id INTEGER PRIMARY KEY,
                    username TEXT,
                    display_name TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    starts_count INTEGER NOT NULL DEFAULT 0,
                    generations_count INTEGER NOT NULL DEFAULT 0,
                    last_action TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    telegram_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    detail TEXT,
                    FOREIGN KEY (telegram_id) REFERENCES users(telegram_id)
                );
                CREATE INDEX IF NOT EXISTS events_created_at_idx ON events(created_at DESC);
                CREATE INDEX IF NOT EXISTS events_telegram_id_idx ON events(telegram_id);
                """
            )

    def record(self, user: Any, event_type: str, detail: str | None = None) -> None:
        if user is None:
            return
        timestamp = _now()
        display_name = " ".join(part for part in (user.first_name, user.last_name) if part).strip()
        display_name = display_name or "Без имени"
        username = user.username or None
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    telegram_id, username, display_name, first_seen, last_seen,
                    starts_count, generations_count, last_action
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(telegram_id) DO UPDATE SET
                    username = excluded.username,
                    display_name = excluded.display_name,
                    last_seen = excluded.last_seen,
                    starts_count = users.starts_count + excluded.starts_count,
                    generations_count = users.generations_count + excluded.generations_count,
                    last_action = excluded.last_action
                """,
                (
                    user.id,
                    username,
                    display_name,
                    timestamp,
                    timestamp,
                    int(event_type == "start"),
                    int(event_type == "generation_completed"),
                    event_type,
                ),
            )
            connection.execute(
                "INSERT INTO events (created_at, telegram_id, event_type, detail) VALUES (?, ?, ?, ?)",
                (timestamp, user.id, event_type, detail),
            )

    def snapshot(self) -> dict[str, Any]:
        with self._connect() as connection:
            totals = connection.execute(
                """
                SELECT COUNT(*) AS users, COALESCE(SUM(generations_count), 0) AS generations
                FROM users
                """
            ).fetchone()
            users = connection.execute(
                """
                SELECT telegram_id, username, display_name, first_seen, last_seen,
                       starts_count, generations_count, last_action
                FROM users
                ORDER BY last_seen DESC
                """
            ).fetchall()
            events = connection.execute(
                """
                SELECT created_at, telegram_id, event_type, detail
                FROM events
                ORDER BY id DESC
                LIMIT 100
                """
            ).fetchall()
        return {"totals": totals, "users": users, "events": events}

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection
