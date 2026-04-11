"""Queue-backed migration runner integrated into vector-db-gateway."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib import error, parse, request

from vector_gateway.config import DoMigConfig, ServiceEndpointConfig
from vector_gateway.core.logical_registry import LogicalCollectionRegistry
from vector_gateway.models.api import DoMigQueueItem


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_request(
    method: str,
    base_url: str,
    path: str,
    api_key: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: int = 20,
) -> Any:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    req = request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=payload,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:  # pragma: no cover - exercised via message surface
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: {exc.code} {detail}") from exc
    except error.URLError as exc:  # pragma: no cover - exercised via message surface
        raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc
    if not raw:
        return {}
    return json.loads(raw)


class WriteDiskQueueStore:
    def __init__(self, endpoint: ServiceEndpointConfig, channel: str, *, batch_limit: int = 200):
        self._endpoint = endpoint
        self._channel = channel
        self._batch_limit = max(1, batch_limit)

    def list_items(self) -> list[DoMigQueueItem]:
        items: list[DoMigQueueItem] = []
        offset = 0
        while True:
            data = _json_request(
                "GET",
                self._endpoint.url,
                f"/list/{self._channel}?limit={self._batch_limit}&offset={offset}",
                self._endpoint.api_key,
                timeout=self._endpoint.timeout,
            )
            records = data.get("records") or []
            for record in records:
                meta = dict(record.get("meta") or {})
                meta["id"] = record.get("id")
                meta.setdefault("created_at", record.get("created_at"))
                meta.setdefault("updated_at", record.get("updated_at"))
                items.append(DoMigQueueItem.model_validate(meta))
            total = int(data.get("total") or 0)
            offset += len(records)
            if offset >= total or not records:
                break
        items.sort(key=lambda item: (item.sequence, item.created_at or "", item.id))
        return items

    def upsert_item(self, item: DoMigQueueItem) -> DoMigQueueItem:
        meta = item.model_dump(exclude={"id"}, exclude_none=True)
        payload = {
            "id": item.id,
            "format": "json",
            "content": "",
            "meta": meta,
        }
        record = _json_request(
            "POST",
            self._endpoint.url,
            f"/write/{self._channel}",
            self._endpoint.api_key,
            body=payload,
            timeout=self._endpoint.timeout,
        )
        stored = dict(record.get("meta") or {})
        stored["id"] = record.get("id")
        stored.setdefault("created_at", record.get("created_at"))
        stored.setdefault("updated_at", record.get("updated_at"))
        return DoMigQueueItem.model_validate(stored)


class DBMigratorClient:
    def __init__(self, endpoint: ServiceEndpointConfig):
        self._endpoint = endpoint

    def create_task(self, config: dict[str, Any]) -> str:
        body = config if "config" in config else {"config": config}
        resp = _json_request(
            "POST",
            self._endpoint.url,
            "/tasks",
            self._endpoint.api_key,
            body=body,
            timeout=self._endpoint.timeout,
        )
        task_id = str(resp.get("id") or "")
        if not task_id:
            raise RuntimeError("migration task response missing id")
        return task_id

    def get_task(self, task_id: str) -> dict[str, Any]:
        return _json_request(
            "GET",
            self._endpoint.url,
            f"/tasks/{parse.quote(task_id)}",
            self._endpoint.api_key,
            timeout=self._endpoint.timeout,
        )

    def start_task(self, task_id: str, shards: list[str]) -> None:
        body: dict[str, Any] = {}
        if shards:
            body["shards"] = shards
        _json_request(
            "POST",
            self._endpoint.url,
            f"/tasks/{parse.quote(task_id)}/start",
            self._endpoint.api_key,
            body=body,
            timeout=self._endpoint.timeout,
        )

    def pause_task(self, task_id: str) -> None:
        _json_request(
            "POST",
            self._endpoint.url,
            f"/tasks/{parse.quote(task_id)}/pause",
            self._endpoint.api_key,
            body={},
            timeout=self._endpoint.timeout,
        )

    def resume_task(self, task_id: str) -> None:
        _json_request(
            "POST",
            self._endpoint.url,
            f"/tasks/{parse.quote(task_id)}/resume",
            self._endpoint.api_key,
            body={},
            timeout=self._endpoint.timeout,
        )


@dataclass
class DoMigRunResult:
    action: str
    items: list[DoMigQueueItem]


class DoMigRunner:
    def __init__(
        self,
        config: DoMigConfig,
        queue_store: WriteDiskQueueStore,
        migrator: DBMigratorClient,
        logical_registry: LogicalCollectionRegistry,
    ):
        self._config = config
        self._queue_store = queue_store
        self._migrator = migrator
        self._logical_registry = logical_registry
        self._lock = threading.Lock()

    def list_items(self) -> list[DoMigQueueItem]:
        return self._queue_store.list_items()

    def import_items(self, items: list[DoMigQueueItem]) -> list[DoMigQueueItem]:
        stored: list[DoMigQueueItem] = []
        for item in items:
            stored.append(self._queue_store.upsert_item(item))
        return stored

    def run_once(self, now: datetime | None = None) -> DoMigRunResult:
        if not self._config.enabled:
            raise RuntimeError("do-mig is disabled in config")
        with self._lock:
            return self._run_once_locked(now or datetime.now().astimezone())

    def _run_once_locked(self, now: datetime) -> DoMigRunResult:
        items = self._queue_store.list_items()

        for item in items:
            task_status = self._refresh_item(item)
            if item.status == "running" and _window_passed(now, item.window.pause_at) and task_status == "running":
                self._migrator.pause_task(item.task_id or "")
                item.status = "paused"
                item.last_error = None
                item.updated_at = _utc_now()
                self._queue_store.upsert_item(item)
                return DoMigRunResult(f"Paused {item.id} ({item.task_id}) at window boundary", items)
            if item.status == "running":
                self._queue_store.upsert_item(item)
                return DoMigRunResult(f"Task still running: {item.id} ({item.task_id})", items)

        for item in items:
            if item.status == "completed":
                continue
            if not _within_dispatch_window(now, item.window):
                continue
            if not item.task_id:
                if not item.task_config:
                    item.status = "failed"
                    item.last_error = "queue item missing task_config"
                    item.updated_at = _utc_now()
                    self._queue_store.upsert_item(item)
                    raise RuntimeError(f"{item.id} missing task_config")
                item.task_id = self._migrator.create_task(item.task_config)
                item.attempts += 1
                item.created_at = item.created_at or _utc_now()
            self._ensure_logical_backfill_state(item)
            task = self._migrator.get_task(item.task_id)
            task_status = str(task.get("status") or "unknown")
            if task_status == "paused":
                self._migrator.resume_task(item.task_id)
                item.status = "running"
                item.last_error = None
                item.updated_at = _utc_now()
                self._queue_store.upsert_item(item)
                return DoMigRunResult(f"Resumed {item.id} ({item.task_id})", items)
            if task_status == "pending":
                self._migrator.start_task(item.task_id, item.shards)
                item.status = "running"
                item.last_error = None
                item.updated_at = _utc_now()
                self._queue_store.upsert_item(item)
                return DoMigRunResult(f"Started {item.id} ({item.task_id})", items)
            if task_status == "completed":
                item.status = "completed"
                item.last_error = None
                item.updated_at = _utc_now()
                self._queue_store.upsert_item(item)
                continue
            if task_status == "running":
                item.status = "running"
                item.last_error = None
                item.updated_at = _utc_now()
                self._queue_store.upsert_item(item)
                return DoMigRunResult(f"Task already running: {item.id} ({item.task_id})", items)
            if task_status == "failed":
                item.status = "failed"
                item.last_error = f"task {item.task_id} failed"
                item.updated_at = _utc_now()
                self._queue_store.upsert_item(item)
                return DoMigRunResult(f"Task failed: {item.id} ({item.task_id})", items)
            item.status = task_status
            item.updated_at = _utc_now()
            self._queue_store.upsert_item(item)
            return DoMigRunResult(f"No action for {item.id} ({item.task_id})", items)

        return DoMigRunResult("No queue item due in the current window", items)

    def _refresh_item(self, item: DoMigQueueItem) -> str:
        if not item.task_id:
            item.status = item.status or "queued"
            item.updated_at = _utc_now()
            return item.status
        task = self._migrator.get_task(item.task_id)
        status = str(task.get("status") or "unknown")
        if status == "completed":
            item.status = "completed"
        elif status == "failed":
            item.status = "failed"
        elif status == "paused":
            if item.status in {"", "running"}:
                item.status = "paused"
        elif status == "running":
            item.status = "running"
        elif status == "pending":
            if not item.status:
                item.status = "queued"
        else:
            item.status = status
        item.updated_at = _utc_now()
        return status

    def _ensure_logical_backfill_state(self, item: DoMigQueueItem) -> None:
        info = self._logical_registry.get_state(item.logical_collection)
        metadata = dict(item.metadata)
        metadata.update(
            {
                "queue_item_id": item.id,
                "shards": list(item.shards),
                "window": item.window.model_dump(),
            }
        )
        if info.get("state") in {"", "idle", None}:
            self._logical_registry.prepare(
                item.logical_collection,
                task_id=item.task_id,
                note=f"do-mig {item.id} prepare",
                metadata=metadata,
            )
            self._logical_registry.dual_write(
                item.logical_collection,
                task_id=item.task_id,
                note=f"do-mig {item.id} dual-write",
                metadata=metadata,
            )
        self._logical_registry.backfill(
            item.logical_collection,
            task_id=item.task_id,
            note=f"do-mig {item.id} backfill",
            metadata=metadata,
        )


def _clock_at(now: datetime, hhmm: str) -> datetime:
    parsed = datetime.strptime(hhmm, "%H:%M")
    return now.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)


def _within_dispatch_window(now: datetime, window: Any) -> bool:
    start = _clock_at(now, window.start)
    stop_dispatch = _clock_at(now, window.stop_dispatch)
    return start <= now < stop_dispatch


def _window_passed(now: datetime, hhmm: str) -> bool:
    return now >= _clock_at(now, hhmm)
