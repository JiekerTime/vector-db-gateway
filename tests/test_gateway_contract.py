from __future__ import annotations

import asyncio
import unittest
from tempfile import TemporaryDirectory

import vector_gateway.app as app_module
from vector_gateway.config import CollectionConfig, EmbeddingModelConfig, GatewayConfig, LogicalCollectionConfig
from vector_gateway.core.router import Router
from vector_gateway.core.logical_registry import LogicalCollectionRegistry
from vector_gateway.core.state_store import MigrationStateStore
from vector_gateway.models.api import AgentActionRequest


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
            },
            "collections": {
                "knowledge_base_v3": CollectionConfig(
                    vector_size=1024,
                    distance="Cosine",
                    owner="wiki",
                    vector_name="dense",
                    sparse_vector_name="sparse",
                    query_model="default",
                    write_model="default",
                )
            },
            "logical_collections": {
                "knowledge": LogicalCollectionConfig(
                    read_targets=["knowledge_base_v3"],
                    write_targets=["knowledge_base_v3"],
                    default_query_mode="hybrid",
                    query_model="default",
                    write_model="default",
                )
            },
        }
    )


class _FakeQdrant:
    async def collection_infos(self):
        return []

    async def count(self, *, collection: str, filter_spec):
        return 7


class _FakeScheduler:
    async def submit(self, *, factory, **_kwargs):
        return await factory(), 0


class _FakeMetrics:
    def observe_request(self, *_args, **_kwargs):
        return None


class GatewayContractTest(unittest.TestCase):
    def test_operational_collection_routes_are_exposed(self) -> None:
        paths = {route.path for route in app_module.app.routes}
        self.assertIn("/count", paths)
        self.assertIn("/retrieve", paths)
        self.assertIn("/scroll", paths)
        self.assertIn("/payload/set", paths)
        self.assertIn("/payload/patch", paths)
        self.assertIn("/collections/ensure", paths)

    def test_capabilities_include_operational_collection_actions(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            state_store = MigrationStateStore(tmpdir)
            registry = LogicalCollectionRegistry(config, state_store)
            registry.bootstrap()
            app_module._config = config
            app_module._model_registry = config.models
            app_module._logical_registry = registry
            app_module._qdrant = _FakeQdrant()

            response = asyncio.run(app_module.capabilities(x_api_key="test"))

        action_names = {action.name for action in response.actions}
        self.assertIn("search", action_names)
        self.assertIn("upsert_chunks", action_names)
        self.assertIn("upsert_points", action_names)
        self.assertIn("transform_metadata_prefix", action_names)
        self.assertIn("count", action_names)
        self.assertIn("retrieve", action_names)
        self.assertIn("scroll", action_names)
        self.assertIn("set_payload", action_names)
        self.assertIn("patch_payload", action_names)
        self.assertIn("ensure_collection", action_names)

    def test_agent_action_supports_count(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = _gateway_config(tmpdir)
            state_store = MigrationStateStore(tmpdir)
            registry = LogicalCollectionRegistry(config, state_store)
            registry.bootstrap()
            app_module._config = config
            app_module._router = Router.from_config(config)
            app_module._metrics = _FakeMetrics()
            app_module._job_scheduler = _FakeScheduler()
            app_module._logical_registry = registry
            app_module._qdrant = _FakeQdrant()

            response = asyncio.run(
                app_module.agent_action(
                    AgentActionRequest(
                        action="count",
                        payload={
                            "caller": "interactive/test",
                            "operation": "count",
                            "collection": "knowledge",
                        },
                    ),
                    x_api_key="test",
                )
            )

        self.assertEqual(response.count, 7)
        self.assertEqual(response.collection, "knowledge")


if __name__ == "__main__":
    unittest.main()
