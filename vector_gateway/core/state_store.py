"""SQLite-backed migration runtime state."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


class MigrationStateStore:
    def __init__(self, state_dir: str):
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._state_dir / "migration_state.db"
        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logical_collection_state (
                    logical_name TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    read_target TEXT,
                    write_targets TEXT NOT NULL,
                    shadow_read_targets TEXT NOT NULL,
                    rollback_target TEXT,
                    task_id TEXT,
                    last_verify_at TEXT,
                    last_verify_result TEXT,
                    last_cutover_at TEXT,
                    note TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logical_collection_event (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    logical_name TEXT NOT NULL,
                    event TEXT NOT NULL,
                    state TEXT NOT NULL,
                    task_id TEXT,
                    note TEXT,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def ensure_state(
        self,
        logical_name: str,
        *,
        state: str,
        read_target: str | None,
        write_targets: list[str],
        rollback_target: str | None,
    ) -> dict:
        existing = self.get_state(logical_name)
        if existing is not None:
            return existing
        now = _utc_now()
        payload = {
            "logical_name": logical_name,
            "state": state,
            "read_target": read_target,
            "write_targets": list(write_targets),
            "shadow_read_targets": [],
            "rollback_target": rollback_target,
            "task_id": None,
            "last_verify_at": None,
            "last_verify_result": None,
            "last_cutover_at": None,
            "note": None,
            "updated_at": now,
        }
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO logical_collection_state
                (logical_name, state, read_target, write_targets, shadow_read_targets,
                 rollback_target, task_id, last_verify_at, last_verify_result,
                 last_cutover_at, note, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    logical_name,
                    payload["state"],
                    payload["read_target"],
                    json.dumps(payload["write_targets"]),
                    json.dumps(payload["shadow_read_targets"]),
                    payload["rollback_target"],
                    payload["task_id"],
                    payload["last_verify_at"],
                    payload["last_verify_result"],
                    payload["last_cutover_at"],
                    payload["note"],
                    payload["updated_at"],
                ),
            )
            self._insert_event(
                conn,
                logical_name=logical_name,
                event="bootstrap",
                state=payload["state"],
                task_id=payload["task_id"],
                note=payload["note"],
                metadata={
                    "read_target": payload["read_target"],
                    "write_targets": payload["write_targets"],
                    "rollback_target": payload["rollback_target"],
                },
            )
            conn.commit()
        return payload

    def get_state(self, logical_name: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT logical_name, state, read_target, write_targets, shadow_read_targets,
                       rollback_target, task_id, last_verify_at, last_verify_result,
                       last_cutover_at, note, updated_at
                  FROM logical_collection_state
                 WHERE logical_name = ?
                """,
                (logical_name,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_state(row)

    def list_states(self) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT logical_name, state, read_target, write_targets, shadow_read_targets,
                       rollback_target, task_id, last_verify_at, last_verify_result,
                       last_cutover_at, note, updated_at
                  FROM logical_collection_state
                 ORDER BY logical_name
                """
            ).fetchall()
        return [_row_to_state(row) for row in rows]

    def update_state(self, logical_name: str, **changes) -> dict:
        current = self.get_state(logical_name)
        if current is None:
            raise KeyError(f"Unknown logical collection state: {logical_name}")
        event = changes.pop("event", "update")
        metadata = changes.pop("metadata", {})
        payload = dict(current)
        payload.update({key: value for key, value in changes.items() if value is not None or key in {"read_target", "task_id", "note"}})
        payload["updated_at"] = _utc_now()
        with closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE logical_collection_state
                   SET state = ?,
                       read_target = ?,
                       write_targets = ?,
                       shadow_read_targets = ?,
                       rollback_target = ?,
                       task_id = ?,
                       last_verify_at = ?,
                       last_verify_result = ?,
                       last_cutover_at = ?,
                       note = ?,
                       updated_at = ?
                 WHERE logical_name = ?
                """,
                (
                    payload["state"],
                    payload["read_target"],
                    json.dumps(payload["write_targets"]),
                    json.dumps(payload["shadow_read_targets"]),
                    payload["rollback_target"],
                    payload.get("task_id"),
                    payload.get("last_verify_at"),
                    payload.get("last_verify_result"),
                    payload.get("last_cutover_at"),
                    payload.get("note"),
                    payload["updated_at"],
                    logical_name,
                ),
            )
            self._insert_event(
                conn,
                logical_name=logical_name,
                event=event,
                state=payload["state"],
                task_id=payload.get("task_id"),
                note=payload.get("note"),
                metadata=metadata,
            )
            conn.commit()
        return payload

    def list_events(self, logical_name: str, *, limit: int = 20) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT id, logical_name, event, state, task_id, note, metadata, created_at
                  FROM logical_collection_event
                 WHERE logical_name = ?
                 ORDER BY id DESC
                 LIMIT ?
                """,
                (logical_name, limit),
            ).fetchall()
        return [_row_to_event(row) for row in rows]

    def _insert_event(
        self,
        conn: sqlite3.Connection,
        *,
        logical_name: str,
        event: str,
        state: str,
        task_id: str | None,
        note: str | None,
        metadata: dict,
    ) -> None:
        conn.execute(
            """
            INSERT INTO logical_collection_event
            (logical_name, event, state, task_id, note, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                logical_name,
                event,
                state,
                task_id,
                note,
                json.dumps(metadata or {}),
                _utc_now(),
            ),
        )


def _row_to_state(row: sqlite3.Row | tuple) -> dict:
    (
        logical_name,
        state,
        read_target,
        write_targets,
        shadow_read_targets,
        rollback_target,
        task_id,
        last_verify_at,
        last_verify_result,
        last_cutover_at,
        note,
        updated_at,
    ) = row
    return {
        "logical_name": logical_name,
        "state": state,
        "read_target": read_target,
        "write_targets": json.loads(write_targets or "[]"),
        "shadow_read_targets": json.loads(shadow_read_targets or "[]"),
        "rollback_target": rollback_target,
        "task_id": task_id,
        "last_verify_at": last_verify_at,
        "last_verify_result": last_verify_result,
        "last_cutover_at": last_cutover_at,
        "note": note,
        "updated_at": updated_at,
    }


def _row_to_event(row: sqlite3.Row | tuple) -> dict:
    event_id, logical_name, event, state, task_id, note, metadata, created_at = row
    return {
        "id": int(event_id),
        "logical_name": logical_name,
        "event": event,
        "state": state,
        "task_id": task_id,
        "note": note,
        "metadata": json.loads(metadata or "{}"),
        "created_at": created_at,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
