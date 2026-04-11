from __future__ import annotations

import tempfile
import unittest
from datetime import datetime

from vector_gateway.config import CollectionConfig, DoMigConfig, GatewayConfig, LogicalCollectionConfig, LogicalCollectionMigrationConfig
from vector_gateway.core.do_mig import DoMigRunner
from vector_gateway.core.logical_registry import LogicalCollectionRegistry
from vector_gateway.core.state_store import MigrationStateStore
from vector_gateway.models.api import DoMigQueueItem, DoMigWindow


def _gateway_config(tmpdir: str) -> GatewayConfig:
    return GatewayConfig.model_validate(
        {
            "api_key": "test",
            "state_dir": tmpdir,
            "queues": {"interactive": {"max_batch_size": 8, "max_wait_ms": 10}},
            "routing_rules": [{"caller_pattern": "*", "queue": "interactive"}],
            "operation_priority": {"search": 1},
            "collections": {
                "knowledge_base_v2": CollectionConfig(vector_size=1024, owner="gp"),
                "knowledge_base_v3": CollectionConfig(vector_size=1024, owner="gp"),
            },
            "logical_collections": {
                "knowledge": LogicalCollectionConfig(
                    read_targets=["knowledge_base_v2"],
                    write_targets=["knowledge_base_v2"],
                    migration=LogicalCollectionMigrationConfig(next_target="knowledge_base_v3"),
                )
            },
        }
    )


class FakeQueueStore:
    def __init__(self, items: list[DoMigQueueItem]):
        self.items = {item.id: item for item in items}

    def list_items(self) -> list[DoMigQueueItem]:
        return sorted(self.items.values(), key=lambda item: (item.sequence, item.id))

    def upsert_item(self, item: DoMigQueueItem) -> DoMigQueueItem:
        self.items[item.id] = item
        return item


class FakeMigrator:
    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self._counter = 0

    def create_task(self, config: dict) -> str:
        self._counter += 1
        task_id = f"task-{self._counter}"
        self.tasks[task_id] = {"status": "pending", "config": config}
        return task_id

    def get_task(self, task_id: str) -> dict:
        return dict(self.tasks[task_id])

    def start_task(self, task_id: str, shards: list[str]) -> None:
        self.tasks[task_id]["status"] = "running"
        self.tasks[task_id]["shards"] = list(shards)

    def pause_task(self, task_id: str) -> None:
        self.tasks[task_id]["status"] = "paused"

    def resume_task(self, task_id: str) -> None:
        self.tasks[task_id]["status"] = "running"


class DoMigRunnerTest(unittest.TestCase):
    def test_due_item_creates_and_starts_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            store = MigrationStateStore(tmpdir)
            registry = LogicalCollectionRegistry(config, store)
            registry.bootstrap()
            queue = FakeQueueStore(
                [
                    DoMigQueueItem(
                        id="knowledge-batch-1",
                        logical_collection="knowledge",
                        task_config={"source": "a", "sink": "b"},
                        shards=["karpathy"],
                        window=DoMigWindow(start="00:30", stop_dispatch="01:05", pause_at="01:10"),
                    )
                ]
            )
            migrator = FakeMigrator()
            runner = DoMigRunner(DoMigConfig(enabled=True), queue, migrator, registry)

            result = runner.run_once(datetime(2026, 4, 12, 0, 40).astimezone())

            item = queue.items["knowledge-batch-1"]
            self.assertIn("Started", result.action)
            self.assertEqual(item.status, "running")
            self.assertEqual(item.task_id, "task-1")
            self.assertEqual(migrator.tasks["task-1"]["status"], "running")
            self.assertEqual(registry.get_state("knowledge")["state"], "backfill")

    def test_running_item_pauses_after_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            store = MigrationStateStore(tmpdir)
            registry = LogicalCollectionRegistry(config, store)
            registry.bootstrap()
            queue = FakeQueueStore(
                [
                    DoMigQueueItem(
                        id="knowledge-batch-2",
                        logical_collection="knowledge",
                        task_config={"source": "a", "sink": "b"},
                        task_id="task-9",
                        status="running",
                        window=DoMigWindow(start="00:30", stop_dispatch="01:05", pause_at="01:10"),
                    )
                ]
            )
            migrator = FakeMigrator()
            migrator.tasks["task-9"] = {"status": "running"}
            runner = DoMigRunner(DoMigConfig(enabled=True), queue, migrator, registry)

            result = runner.run_once(datetime(2026, 4, 12, 1, 12).astimezone())

            item = queue.items["knowledge-batch-2"]
            self.assertIn("Paused", result.action)
            self.assertEqual(item.status, "paused")
            self.assertEqual(migrator.tasks["task-9"]["status"], "paused")


if __name__ == "__main__":
    unittest.main()
