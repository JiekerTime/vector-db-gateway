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
        self.query_points_calls: list[dict[str, object]] = []

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

    def query_points(
        self,
        *,
        collection_name: str,
        query,
        using,
        query_filter,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ):
        self.query_points_calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "using": using,
                "query_filter": query_filter,
                "limit": limit,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )
        point = type(
            "Point",
            (),
            {
                "id": "p1",
                "score": 0.9,
                "payload": {"title": "doc"},
                "vector": [0.1, 0.2],
            },
        )()
        return type("QueryResponse", (), {"points": [point]})()


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

    def test_unregistered_collection_shape_is_inferred_from_qdrant(self) -> None:
        client = _FakeClient()
        client.collections["knowledge_base_v2"] = {
            "size": 1024,
            "distance": "Cosine",
        }
        store = _FakeQdrantStore(client, {})

        meta = store._collection_meta("knowledge_base_v2")

        self.assertEqual(meta.vector_size, 1024)
        self.assertEqual(meta.distance, "cosine")
        self.assertEqual(meta.owner, "external")
        self.assertIsNone(meta.vector_name)

    def test_ensure_collection_creates_external_collection(self) -> None:
        client = _FakeClient()
        store = _FakeQdrantStore(client, {})

        created, info = asyncio.run(
            store.ensure_collection(
                collection="knowledge_base_v2",
                meta=CollectionConfig(
                    vector_size=1024,
                    distance="Cosine",
                    owner="external",
                    description="Shared knowledge collection",
                ),
            )
        )

        self.assertTrue(created)
        self.assertEqual(info.name, "knowledge_base_v2")
        self.assertEqual(info.vector_size, 1024)
        self.assertEqual(info.owner, "external")
        self.assertIn("knowledge_base_v2", client.collections)

    def test_search_uses_query_points_when_search_api_is_unavailable(self) -> None:
        client = _FakeClient()
        client.collections["knowledge_base_v2"] = {
            "memory_vector": {
                "size": 2,
                "distance": "Cosine",
            }
        }
        store = _FakeQdrantStore(client, {})

        hits = asyncio.run(
            store.search(
                collection="knowledge_base_v2",
                vector=[0.3, 0.4],
                limit=3,
                filter_spec=None,
                with_payload=True,
                with_vectors=True,
            )
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].id, "p1")
        self.assertEqual(client.query_points_calls[0]["using"], "memory_vector")
        self.assertEqual(client.query_points_calls[0]["query"], [0.3, 0.4])


if __name__ == "__main__":
    unittest.main()
