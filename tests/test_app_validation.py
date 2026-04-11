from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from fastapi import HTTPException

import vector_gateway.app as app_module
from vector_gateway.config import CollectionConfig, EmbeddingModelConfig, GatewayConfig, LogicalCollectionConfig
from vector_gateway.core.logical_registry import LogicalCollectionRegistry
from vector_gateway.core.state_store import MigrationStateStore
from vector_gateway.models.api import UpsertPoint


def _gateway_config(tmpdir: str) -> GatewayConfig:
    return GatewayConfig.model_validate(
        {
            "api_key": "test",
            "state_dir": tmpdir,
            "queues": {"interactive": {"max_batch_size": 8, "max_wait_ms": 10}},
            "routing_rules": [{"caller_pattern": "*", "queue": "interactive"}],
            "operation_priority": {"search": 1, "upsert": 2},
            "models": {
                "default": EmbeddingModelConfig(model_name="BAAI/bge-m3", vector_size=1024),
                "legacy_small": EmbeddingModelConfig(model_name="MiniLM", vector_size=384),
            },
            "collections": {
                "decision_memory_v2": CollectionConfig(
                    vector_size=1024,
                    distance="Cosine",
                    owner="gp",
                    vector_name="dense",
                    query_model="default",
                    write_model="default",
                )
            },
            "logical_collections": {
                "decision_memory": LogicalCollectionConfig(
                    read_targets=["decision_memory_v2"],
                    write_targets=["decision_memory_v2"],
                    query_model="default",
                    write_model="default",
                )
            },
        }
    )


class AppValidationTest(unittest.TestCase):
    def test_validate_dense_request_vector_rejects_wrong_size(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            app_module._config = config

            with self.assertRaises(HTTPException) as ctx:
                app_module._validate_dense_request_vector("decision_memory_v2", [0.1] * 384)

            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("expects dense vector size 1024, got 384", ctx.exception.detail)

    def test_validate_model_for_collection_rejects_wrong_size(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            state_store = MigrationStateStore(tmpdir)
            registry = LogicalCollectionRegistry(config, state_store)
            registry.bootstrap()
            app_module._config = config
            app_module._model_registry = config.models
            app_module._logical_registry = registry

            with self.assertRaises(HTTPException) as ctx:
                app_module._validate_model_for_collection("decision_memory_v2", "legacy_small", purpose="query")

            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("produces 384-dim vectors", ctx.exception.detail)

    def test_validate_upsert_points_requires_dense_vector_payload(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            app_module._config = config

            with self.assertRaises(HTTPException) as ctx:
                app_module._validate_upsert_points_for_targets(
                    ["decision_memory_v2"],
                    [
                        UpsertPoint(
                            id="d1",
                            vector={"sparse": {"indices": [1, 3], "values": [0.2, 0.4]}},
                            payload={},
                        )
                    ],
                )

            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("requires a dense vector payload", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
