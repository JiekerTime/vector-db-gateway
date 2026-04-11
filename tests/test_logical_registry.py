from __future__ import annotations

import tempfile
import unittest

from vector_gateway.config import CollectionConfig, GatewayConfig, LogicalCollectionConfig, LogicalCollectionMigrationConfig
from vector_gateway.core.logical_registry import LogicalCollectionRegistry
from vector_gateway.core.state_store import MigrationStateStore
from vector_gateway.models.api import CollectionInfo


def _gateway_config(tmpdir: str) -> GatewayConfig:
    return GatewayConfig.model_validate(
        {
            "api_key": "test",
            "state_dir": tmpdir,
            "queues": {
                "interactive": {"max_batch_size": 8, "max_wait_ms": 10},
            },
            "routing_rules": [{"caller_pattern": "*", "queue": "interactive"}],
            "operation_priority": {"search": 1},
            "collections": {
                "knowledge_base_v2": CollectionConfig(vector_size=1024, owner="gp"),
                "knowledge_base_v3": CollectionConfig(
                    vector_size=1024,
                    owner="gp",
                    vector_name="dense",
                    sparse_vector_name="sparse",
                ),
            },
            "logical_collections": {
                "knowledge": LogicalCollectionConfig(
                    read_targets=["knowledge_base_v2"],
                    write_targets=["knowledge_base_v2"],
                    default_query_mode="hybrid",
                    alias_name="knowledge",
                    migration=LogicalCollectionMigrationConfig(next_target="knowledge_base_v3"),
                )
            },
        }
    )


class LogicalRegistryTest(unittest.TestCase):
    def test_bootstrap_and_cutover_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            store = MigrationStateStore(tmpdir)
            registry = LogicalCollectionRegistry(config, store)

            registry.bootstrap()
            self.assertEqual(registry.read_target("knowledge"), "knowledge_base_v2")
            self.assertEqual(registry.write_targets("knowledge"), ["knowledge_base_v2"])

            registry.prepare("knowledge", task_id="job-1")
            registry.dual_write("knowledge", task_id="job-1")
            registry.backfill("knowledge", task_id="job-1")
            registry.mark_verify("knowledge", result="ok")
            registry.cutover("knowledge")

            self.assertEqual(registry.read_target("knowledge"), "knowledge_base_v3")
            self.assertEqual(registry.write_targets("knowledge"), ["knowledge_base_v3"])

    def test_rollback_restores_previous_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            store = MigrationStateStore(tmpdir)
            registry = LogicalCollectionRegistry(config, store)

            registry.bootstrap()
            registry.dual_write("knowledge")
            registry.cutover("knowledge")
            registry.rollback("knowledge")

            self.assertEqual(registry.read_target("knowledge"), "knowledge_base_v2")
            self.assertEqual(registry.write_targets("knowledge"), ["knowledge_base_v2"])

    def test_info_contains_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            store = MigrationStateStore(tmpdir)
            registry = LogicalCollectionRegistry(config, store)
            registry.bootstrap()
            physical_infos = {
                "knowledge_base_v2": CollectionInfo(
                    name="knowledge_base_v2",
                    vector_size=1024,
                    distance="Cosine",
                    owner="gp",
                ),
                "knowledge_base_v3": CollectionInfo(
                    name="knowledge_base_v3",
                    vector_size=1024,
                    distance="Cosine",
                    owner="gp",
                    vector_name="dense",
                    sparse_vector_name="sparse",
                ),
            }

            info = registry.get_info("knowledge", physical_infos)

            self.assertEqual(info.name, "knowledge")
            self.assertEqual(info.current_read_target, "knowledge_base_v2")
            self.assertEqual(info.migration.next_target, "knowledge_base_v3")

    def test_events_accumulate_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            store = MigrationStateStore(tmpdir)
            registry = LogicalCollectionRegistry(config, store)
            registry.bootstrap()

            registry.prepare(
                "knowledge",
                task_id="job-42",
                metadata={
                    "partition_key": "expert_id",
                    "partition_values": ["karpathy"],
                    "window": "01:00-02:00",
                    "attempt": 1,
                },
            )
            registry.backfill(
                "knowledge",
                task_id="job-42",
                metadata={
                    "partition_key": "expert_id",
                    "partition_values": ["karpathy"],
                    "checkpoint": {"last_id": "p-100"},
                },
            )

            events = registry.list_events("knowledge", limit=10)

            self.assertEqual(events[0]["event"], "backfill")
            self.assertEqual(events[0]["task_id"], "job-42")
            self.assertEqual(events[0]["metadata"]["checkpoint"]["last_id"], "p-100")
            self.assertEqual(events[1]["event"], "prepare")
            self.assertEqual(events[1]["metadata"]["partition_key"], "expert_id")


if __name__ == "__main__":
    unittest.main()
