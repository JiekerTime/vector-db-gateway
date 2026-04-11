from __future__ import annotations

import asyncio
import unittest

from vector_gateway.backends.qdrant_store import QdrantStore
from vector_gateway.config import CollectionConfig, QdrantConfig


class _FakeDistance:
    COSINE = "Cosine"


class _FakeVectorParams:
    def __init__(self, size: int, distance: str):
        self.size = size
        self.distance = distance


class _FakeModels:
    Distance = _FakeDistance
    VectorParams = _FakeVectorParams


class _FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict] = {}
        self.create_calls: list[tuple[str, object]] = []

    def get_collection(self, collection_name: str):
        if collection_name not in self.collections:
            raise RuntimeError("not found")
        return {
            "result": {
                "config": {
                    "params": {
                        "vectors": self.collections[collection_name],
                    }
                }
            }
        }

    def create_collection(self, collection_name: str, vectors_config):
        self.create_calls.append((collection_name, vectors_config))
        if isinstance(vectors_config, dict):
            vector_name, params = next(iter(vectors_config.items()))
            self.collections[collection_name] = {
                vector_name: {
                    "size": params.size,
                    "distance": params.distance,
                }
            }
            return
        self.collections[collection_name] = {
            "size": vectors_config.size,
            "distance": vectors_config.distance,
        }


class _FakeQdrantStore(QdrantStore):
    def __init__(self, client: _FakeClient, collections: dict[str, CollectionConfig]):
        super().__init__(QdrantConfig(url="http://qdrant:6333", timeout=20), collections)
        self._client = client

    def _models(self):
        return _FakeModels

    def _get_client(self):
        return self._client


class QdrantStoreBootstrapTest(unittest.TestCase):
    def test_bootstrap_creates_missing_collection(self) -> None:
        client = _FakeClient()
        store = _FakeQdrantStore(
            client,
            {
                "documents": CollectionConfig(
                    vector_size=1024,
                    distance="Cosine",
                    owner="default",
                )
            },
        )

        asyncio.run(store.ensure_collections())

        self.assertEqual(len(client.create_calls), 1)
        self.assertIn("documents", client.collections)
        self.assertEqual(client.collections["documents"]["size"], 1024)
        self.assertEqual(client.collections["documents"]["distance"], "Cosine")

    def test_existing_collection_is_not_recreated(self) -> None:
        client = _FakeClient()
        client.collections["memory"] = {
            "memory_vector": {
                "size": 384,
                "distance": "Cosine",
            }
        }
        store = _FakeQdrantStore(
            client,
            {
                "memory": CollectionConfig(
                    vector_size=384,
                    distance="Cosine",
                    owner="default",
                    vector_name="memory_vector",
                )
            },
        )

        asyncio.run(store.ensure_collections())

        self.assertEqual(client.create_calls, [])


if __name__ == "__main__":
    unittest.main()
