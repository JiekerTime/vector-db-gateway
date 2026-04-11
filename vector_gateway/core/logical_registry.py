"""Logical collection registry and migration control plane."""

from __future__ import annotations

from datetime import datetime, timezone

from vector_gateway.config import CollectionConfig, GatewayConfig, LogicalCollectionConfig
from vector_gateway.core.state_store import MigrationStateStore
from vector_gateway.models.api import CollectionInfo, LogicalCollectionInfo, LogicalCollectionMigrationState


class LogicalCollectionRegistry:
    def __init__(self, config: GatewayConfig, state_store: MigrationStateStore):
        self._config = config
        self._state_store = state_store

    def bootstrap(self) -> None:
        for logical_name, meta in self._config.logical_collections.items():
            read_target = meta.read_targets[0] if meta.read_targets else None
            self._state_store.ensure_state(
                logical_name,
                state="idle",
                read_target=read_target,
                write_targets=list(meta.write_targets or meta.read_targets),
                rollback_target=read_target,
            )

    def is_logical(self, collection_name: str) -> bool:
        return collection_name in self._config.logical_collections

    def get_logical_config(self, logical_name: str) -> LogicalCollectionConfig:
        try:
            return self._config.logical_collections[logical_name]
        except KeyError as exc:
            raise KeyError(f"Unknown logical collection: {logical_name}") from exc

    def get_state(self, logical_name: str) -> dict:
        state = self._state_store.get_state(logical_name)
        if state is None:
            raise KeyError(f"Unknown logical collection state: {logical_name}")
        return state

    def read_target(self, collection_name: str) -> str:
        if not self.is_logical(collection_name):
            return collection_name
        state = self.get_state(collection_name)
        target = state.get("read_target")
        if not target:
            raise ValueError(f"Logical collection '{collection_name}' has no read target")
        return str(target)

    def write_targets(self, collection_name: str) -> list[str]:
        if not self.is_logical(collection_name):
            return [collection_name]
        state = self.get_state(collection_name)
        targets = [str(item) for item in state.get("write_targets") or []]
        if not targets:
            raise ValueError(f"Logical collection '{collection_name}' has no write targets")
        return targets

    def model_for(self, collection_name: str, *, purpose: str = "query") -> str | None:
        physical_name = self.read_target(collection_name) if purpose == "query" else self.write_targets(collection_name)[0]
        physical = self._config.collections.get(physical_name)
        logical = self._config.logical_collections.get(collection_name) if self.is_logical(collection_name) else None
        if logical:
            if purpose == "write" and logical.write_model:
                return logical.write_model
            if purpose == "query" and logical.query_model:
                return logical.query_model
        if physical is None:
            return None
        if purpose == "write" and physical.write_model:
            return physical.write_model
        if purpose == "query" and physical.query_model:
            return physical.query_model
        return physical.model

    def current_info(self, logical_name: str, physical_infos: dict[str, CollectionInfo]) -> LogicalCollectionInfo:
        meta = self.get_logical_config(logical_name)
        state = self.get_state(logical_name)
        read_target = str(state.get("read_target") or "")
        write_targets = [str(item) for item in state.get("write_targets") or []]
        recent_events = self._state_store.list_events(logical_name, limit=10)
        return LogicalCollectionInfo(
            name=logical_name,
            alias_name=meta.alias_name or logical_name,
            default_query_mode=meta.default_query_mode,
            configured_read_targets=list(meta.read_targets),
            configured_write_targets=list(meta.write_targets or meta.read_targets),
            current_read_target=read_target,
            current_write_targets=write_targets,
            read_collection=physical_infos.get(read_target),
            write_collections=[physical_infos[item] for item in write_targets if item in physical_infos],
            migration=LogicalCollectionMigrationState(
                state=str(state.get("state") or "idle"),
                next_target=meta.migration.next_target,
                rollback_target=state.get("rollback_target"),
                task_id=state.get("task_id"),
                shadow_read_targets=[str(item) for item in state.get("shadow_read_targets") or []],
                last_verify_at=state.get("last_verify_at"),
                last_verify_result=state.get("last_verify_result"),
                last_cutover_at=state.get("last_cutover_at"),
                note=state.get("note"),
                updated_at=state.get("updated_at"),
                recent_events=recent_events,
            ),
        )

    def list_infos(self, physical_infos: dict[str, CollectionInfo]) -> list[LogicalCollectionInfo]:
        return [self.current_info(name, physical_infos) for name in sorted(self._config.logical_collections.keys())]

    def get_info(self, logical_name: str, physical_infos: dict[str, CollectionInfo]) -> LogicalCollectionInfo:
        return self.current_info(logical_name, physical_infos)

    def ensure_target(self, logical_name: str) -> str:
        logical = self.get_logical_config(logical_name)
        if not logical.migration.next_target:
            raise ValueError(f"Logical collection '{logical_name}' has no configured next_target")
        return logical.migration.next_target

    def list_events(self, logical_name: str, *, limit: int = 20) -> list[dict]:
        self.get_logical_config(logical_name)
        return self._state_store.list_events(logical_name, limit=limit)

    def prepare(
        self,
        logical_name: str,
        *,
        task_id: str | None = None,
        note: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        self.ensure_target(logical_name)
        current = self.get_state(logical_name)
        return self._state_store.update_state(
            logical_name,
            event="prepare",
            state="prepare",
            read_target=current["read_target"],
            write_targets=current["write_targets"],
            rollback_target=current.get("rollback_target") or current["read_target"],
            task_id=task_id or current.get("task_id"),
            note=note,
            metadata=metadata or {},
        )

    def dual_write(
        self,
        logical_name: str,
        *,
        task_id: str | None = None,
        note: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        next_target = self.ensure_target(logical_name)
        current = self.get_state(logical_name)
        write_targets = list(dict.fromkeys([*(current.get("write_targets") or []), next_target]))
        return self._state_store.update_state(
            logical_name,
            event="dual_write",
            state="dual_write",
            write_targets=write_targets,
            rollback_target=current.get("rollback_target") or current["read_target"],
            task_id=task_id or current.get("task_id"),
            note=note,
            metadata=metadata or {"next_target": next_target},
        )

    def backfill(
        self,
        logical_name: str,
        *,
        task_id: str | None = None,
        note: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        current = self.get_state(logical_name)
        return self._state_store.update_state(
            logical_name,
            event="backfill",
            state="backfill",
            task_id=task_id or current.get("task_id"),
            note=note,
            metadata=metadata or {},
        )

    def mark_verify(
        self,
        logical_name: str,
        *,
        result: str | None = None,
        note: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        current = self.get_state(logical_name)
        return self._state_store.update_state(
            logical_name,
            event="verify",
            state="verify",
            last_verify_at=_utc_now(),
            last_verify_result=result or "pending",
            note=note,
            task_id=current.get("task_id"),
            metadata=metadata or {"result": result or "pending"},
        )

    def cutover(self, logical_name: str, *, note: str | None = None, metadata: dict | None = None) -> dict:
        next_target = self.ensure_target(logical_name)
        current = self.get_state(logical_name)
        return self._state_store.update_state(
            logical_name,
            event="cutover",
            state="live",
            read_target=next_target,
            write_targets=[next_target],
            rollback_target=current.get("rollback_target") or current["read_target"],
            last_cutover_at=_utc_now(),
            note=note,
            metadata=metadata or {"read_target": next_target},
        )

    def rollback(self, logical_name: str, *, note: str | None = None, metadata: dict | None = None) -> dict:
        current = self.get_state(logical_name)
        rollback_target = current.get("rollback_target") or current.get("read_target")
        if not rollback_target:
            raise ValueError(f"Logical collection '{logical_name}' has no rollback target")
        return self._state_store.update_state(
            logical_name,
            event="rollback",
            state="rollback",
            read_target=rollback_target,
            write_targets=[rollback_target],
            note=note,
            metadata=metadata or {"read_target": rollback_target},
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
