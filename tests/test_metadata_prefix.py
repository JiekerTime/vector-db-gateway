from __future__ import annotations

import asyncio
import unittest
from tempfile import TemporaryDirectory

from fastapi import HTTPException

import vector_gateway.app as app_module
from vector_gateway.config import CollectionConfig, EmbeddingModelConfig, GatewayConfig, LogicalCollectionConfig
from vector_gateway.core.logical_registry import LogicalCollectionRegistry
from vector_gateway.core.metadata_prefix import apply_metadata_prefix, render_metadata_prefix
from vector_gateway.core.state_store import MigrationStateStore
from vector_gateway.models.api import TransformMetadataPrefixRequest, UpsertChunk


def _gateway_config(tmpdir: str) -> GatewayConfig:
    return GatewayConfig.model_validate(
        {
            "api_key": "test",
            "state_dir": tmpdir,
            "queues": {"interactive": {"max_batch_size": 8, "max_wait_ms": 10}},
            "routing_rules": [{"caller_pattern": "*", "queue": "interactive"}],
            "operation_priority": {"search": 1, "upsert": 2},
            "models": {
                "default": EmbeddingModelConfig(model_name="BAAI/bge-m3", vector_size=2),
            },
            "collections": {
                "knowledge_base_v3": CollectionConfig(
                    vector_size=2,
                    distance="Cosine",
                    owner="gp",
                    vector_name="dense",
                    sparse_vector_name="sparse",
                    query_model="default",
                    write_model="default",
                )
            },
            "logical_collections": {
                "knowledge": LogicalCollectionConfig.model_validate(
                    {
                        "read_targets": ["knowledge_base_v3"],
                        "write_targets": ["knowledge_base_v3"],
                        "default_query_mode": "hybrid",
                        "query_model": "default",
                        "write_model": "default",
                        "metadata_prefix": {
                            "enabled": True,
                            "parts": [
                                {"payload_key": "expert_name"},
                                {"payload_key": "source", "label": "Source"},
                                {"payload_key": "topic", "label": "Topic"},
                            ],
                        },
                    }
                )
            },
        }
    )


class MetadataPrefixTest(unittest.TestCase):
    def test_render_and_apply_prefix(self) -> None:
        config = _gateway_config("/tmp")
        policy = config.logical_collections["knowledge"].metadata_prefix
        payload = {"expert_name": "Alice", "source": "https://x", "topic": "routing"}
        prefix = render_metadata_prefix(payload, policy)
        self.assertEqual(prefix, "[Alice | Source: https://x | Topic: routing]")

        text, updated_payload, applied = apply_metadata_prefix(
            text="Chunk body",
            payload=payload,
            policy=policy,
        )
        self.assertEqual(applied, prefix)
        self.assertEqual(text, f"{prefix}\nChunk body")
        self.assertEqual(updated_payload["text"], f"{prefix}\nChunk body")
        self.assertEqual(updated_payload["text_raw"], "Chunk body")
        self.assertEqual(updated_payload["metadata_prefix"], prefix)

    def test_prepare_upsert_points_applies_metadata_prefix(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            state_store = MigrationStateStore(tmpdir)
            registry = LogicalCollectionRegistry(config, state_store)
            registry.bootstrap()
            app_module._config = config
            app_module._model_registry = config.models
            app_module._logical_registry = registry

            points, _ = asyncio.run(
                app_module._prepare_upsert_points(
                    collection_name="knowledge",
                    chunks=[
                        UpsertChunk(
                            id="k1",
                            text="Original chunk",
                            vector=[0.1, 0.2],
                            payload={"expert_name": "Alice", "source": "https://x", "topic": "routing"},
                        )
                    ],
                    route=None,
                    model_name=None,
                    device=None,
                    request_id="req-test",
                )
            )

        point = points[0]
        self.assertEqual(point.id, "k1")
        self.assertEqual(point.payload["metadata_prefix"], "[Alice | Source: https://x | Topic: routing]")
        self.assertEqual(point.payload["text_raw"], "Original chunk")
        self.assertEqual(
            point.payload["text"],
            "[Alice | Source: https://x | Topic: routing]\nOriginal chunk",
        )

    def test_transform_metadata_prefix_endpoint(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            state_store = MigrationStateStore(tmpdir)
            registry = LogicalCollectionRegistry(config, state_store)
            registry.bootstrap()
            app_module._config = config
            app_module._logical_registry = registry

            response = asyncio.run(
                app_module.transform_metadata_prefix(
                    TransformMetadataPrefixRequest.model_validate(
                        {
                            "collection": "knowledge",
                            "items": [
                                {
                                    "text": "Chunk body",
                                    "payload": {
                                        "expert_name": "Alice",
                                        "source": "https://x",
                                        "topic": "routing",
                                    },
                                }
                            ],
                        }
                    ),
                    x_api_key="test",
                )
            )

        self.assertEqual(len(response.items), 1)
        self.assertEqual(response.items[0].prefix, "[Alice | Source: https://x | Topic: routing]")
        self.assertEqual(response.items[0].payload["text_raw"], "Chunk body")

    def test_transform_metadata_prefix_rejects_collection_without_policy(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            config.logical_collections["knowledge"].metadata_prefix.enabled = False
            state_store = MigrationStateStore(tmpdir)
            registry = LogicalCollectionRegistry(config, state_store)
            registry.bootstrap()
            app_module._config = config
            app_module._logical_registry = registry

            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(
                    app_module.transform_metadata_prefix(
                        TransformMetadataPrefixRequest.model_validate(
                            {"collection": "knowledge", "items": [{"text": "Chunk body"}]}
                        ),
                        x_api_key="test",
                    )
                )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("has no enabled metadata_prefix policy", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
