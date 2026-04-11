from __future__ import annotations

import asyncio
import unittest

from vector_gateway.backends.qdrant_store import QdrantStore
from vector_gateway.config import CollectionConfig, QdrantConfig
from vector_gateway.models.api import UpsertPoint


class _FakeDistance:
    COSINE = "Cosine"


class _FakeModifier:
    IDF = "idf"


class _FakeFusion:
    RRF = "rrf"


class _FakeVectorParams:
    def __init__(self, size: int, distance: str):
        self.size = size
        self.distance = distance


class _FakeSparseVectorParams:
    def __init__(self, modifier=None):
        self.modifier = modifier


class _FakeSparseVector:
    def __init__(self, *, indices, values):
        self.indices = indices
        self.values = values


class _FakePrefetch:
    def __init__(self, *, query, using=None, limit=None):
        self.query = query
        self.using = using
        self.limit = limit


class _FakeFusionQuery:
    def __init__(self, *, fusion):
        self.fusion = fusion


class _FakePointStruct:
    def __init__(self, *, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload


class _FakeDeleteAlias:
    def __init__(self, *, alias_name):
        self.alias_name = alias_name


class _FakeDeleteAliasOperation:
    def __init__(self, *, delete_alias):
        self.delete_alias = delete_alias


class _FakeCreateAlias:
    def __init__(self, *, collection_name, alias_name):
        self.collection_name = collection_name
        self.alias_name = alias_name


class _FakeCreateAliasOperation:
    def __init__(self, *, create_alias):
        self.create_alias = create_alias


class _FakeModels:
    Distance = _FakeDistance
    Modifier = _FakeModifier
    Fusion = _FakeFusion
    VectorParams = _FakeVectorParams
    SparseVectorParams = _FakeSparseVectorParams
    SparseVector = _FakeSparseVector
    Prefetch = _FakePrefetch
    FusionQuery = _FakeFusionQuery
    PointStruct = _FakePointStruct
    DeleteAlias = _FakeDeleteAlias
    DeleteAliasOperation = _FakeDeleteAliasOperation
    CreateAlias = _FakeCreateAlias
    CreateAliasOperation = _FakeCreateAliasOperation


class _FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict] = {}
        self.create_calls: list[tuple[str, object, object]] = []
        self.delete_calls: list[str] = []
        self.alias_calls: list[list[object]] = []
        self.query_points_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.set_payload_calls: list[dict[str, object]] = []
        self.retrieve_calls: list[dict[str, object]] = []

    def get_collections(self):
        collection_items = [type("CollectionRef", (), {"name": name})() for name in self.collections]
        return type("Collections", (), {"collections": collection_items})()

    def get_collection(self, collection_name: str):
        if collection_name not in self.collections:
            raise RuntimeError("not found")
        stored = self.collections[collection_name]
        payload = {"vectors": stored["vectors"]}
        if stored.get("sparse_vectors") is not None:
            payload["sparse_vectors"] = stored["sparse_vectors"]
        return {
            "result": {
                "config": {"params": payload},
                "points_count": stored.get("points_count", 0),
                "indexed_vectors_count": stored.get("indexed_vectors_count", 0),
                "status": stored.get("status", "green"),
            }
        }

    def create_collection(self, collection_name: str, vectors_config, sparse_vectors_config=None):
        self.create_calls.append((collection_name, vectors_config, sparse_vectors_config))
        if isinstance(vectors_config, dict):
            vector_name, params = next(iter(vectors_config.items()))
            stored_vectors = {
                vector_name: {
                    "size": params.size,
                    "distance": params.distance,
                }
            }
        else:
            stored_vectors = {
                "size": vectors_config.size,
                "distance": vectors_config.distance,
            }
        stored_sparse = None
        if sparse_vectors_config:
            sparse_name, params = next(iter(sparse_vectors_config.items()))
            stored_sparse = {sparse_name: {"modifier": params.modifier}}
        self.collections[collection_name] = {
            "vectors": stored_vectors,
            "sparse_vectors": stored_sparse,
            "points_count": 0,
            "indexed_vectors_count": 0,
            "status": "green",
        }

    def delete_collection(self, *, collection_name: str):
        self.delete_calls.append(collection_name)
        self.collections.pop(collection_name, None)

    def update_collection_aliases(self, operations):
        self.alias_calls.append(list(operations))
        return True

    def query_points(
        self,
        *,
        collection_name: str,
        query=None,
        using=None,
        prefetch=None,
        query_filter=None,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
        **_,
    ):
        self.query_points_calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "using": using,
                "prefetch": prefetch,
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

    def count(self, *, collection_name: str, count_filter, exact: bool):
        return type("CountResult", (), {"count": 7})()

    def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ):
        point = type("Point", (), {"id": "p2", "payload": {"x": 1}, "vector": None})()
        return [point], None

    def retrieve(self, *, collection_name: str, ids, with_payload: bool, with_vectors: bool):
        self.retrieve_calls.append(
            {
                "collection_name": collection_name,
                "ids": list(ids),
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )
        return [type("Record", (), {"id": ids[0], "payload": {"ok": True}, "vector": None})()]

    def set_payload(self, *, collection_name: str, payload, points, wait: bool):
        self.set_payload_calls.append(
            {
                "collection_name": collection_name,
                "payload": payload,
                "points": list(points),
                "wait": wait,
            }
        )
        return True

    def upsert(self, *, collection_name: str, points, wait: bool):
        self.upsert_calls.append(
            {
                "collection_name": collection_name,
                "points": list(points),
                "wait": wait,
            }
        )
        return True


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
        self.assertEqual(client.collections["documents"]["vectors"]["size"], 1024)

    def test_bootstrap_creates_sparse_collection(self) -> None:
        client = _FakeClient()
        store = _FakeQdrantStore(
            client,
            {
                "knowledge_base_v3": CollectionConfig(
                    vector_size=1024,
                    distance="Cosine",
                    owner="default",
                    vector_name="dense",
                    sparse_vector_name="sparse",
                    sparse_modifier="idf",
                )
            },
        )

        asyncio.run(store.ensure_collections())

        created = client.collections["knowledge_base_v3"]
        self.assertEqual(created["vectors"]["dense"]["size"], 1024)
        self.assertEqual(created["sparse_vectors"]["sparse"]["modifier"], "idf")

    def test_existing_collection_is_not_recreated(self) -> None:
        client = _FakeClient()
        client.collections["memory"] = {
            "vectors": {
                "memory_vector": {
                    "size": 384,
                    "distance": "Cosine",
                }
            },
            "sparse_vectors": None,
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
        self.assertEqual(client.delete_calls, [])

    def test_empty_collection_shape_mismatch_is_recreated(self) -> None:
        client = _FakeClient()
        client.collections["decision_memory_v2"] = {
            "vectors": {
                "dense": {
                    "size": 1024,
                    "distance": "Cosine",
                }
            },
            "sparse_vectors": None,
            "points_count": 0,
            "indexed_vectors_count": 0,
        }
        store = _FakeQdrantStore(
            client,
            {
                "decision_memory_v2": CollectionConfig(
                    vector_size=1024,
                    distance="Cosine",
                    owner="default",
                    vector_name="dense",
                    sparse_vector_name="sparse",
                    sparse_modifier="idf",
                )
            },
        )

        asyncio.run(store.ensure_collections())

        self.assertEqual(client.delete_calls, ["decision_memory_v2"])
        self.assertEqual(client.collections["decision_memory_v2"]["sparse_vectors"]["sparse"]["modifier"], "idf")

    def test_alias_sync_skips_existing_collection_name(self) -> None:
        client = _FakeClient()
        client.collections["knowledge"] = {
            "vectors": {
                "size": 1024,
                "distance": "Cosine",
            },
            "sparse_vectors": None,
        }
        store = _FakeQdrantStore(client, {})

        asyncio.run(store.ensure_alias("knowledge", "knowledge_base_v3"))

        self.assertEqual(client.alias_calls, [])

    def test_unregistered_collection_shape_is_inferred_from_qdrant(self) -> None:
        client = _FakeClient()
        client.collections["knowledge_base_v2"] = {
            "vectors": {
                "size": 1024,
                "distance": "Cosine",
            },
            "sparse_vectors": {"sparse": {"modifier": "idf"}},
        }
        store = _FakeQdrantStore(client, {})

        meta = store._collection_meta("knowledge_base_v2")

        self.assertEqual(meta.vector_size, 1024)
        self.assertEqual(meta.distance, "cosine")
        self.assertEqual(meta.owner, "external")
        self.assertIsNone(meta.vector_name)
        self.assertEqual(meta.sparse_vector_name, "sparse")

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

    def test_dense_search_uses_query_points(self) -> None:
        client = _FakeClient()
        client.collections["knowledge_base_v2"] = {
            "vectors": {
                "memory_vector": {
                    "size": 2,
                    "distance": "Cosine",
                }
            },
            "sparse_vectors": None,
        }
        store = _FakeQdrantStore(client, {})

        hits = asyncio.run(
            store.search(
                collection="knowledge_base_v2",
                dense_vector=[0.3, 0.4],
                sparse_vector=None,
                query_mode="dense",
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

    def test_hybrid_search_uses_prefetch(self) -> None:
        client = _FakeClient()
        client.collections["knowledge_base_v3"] = {
            "vectors": {
                "dense": {
                    "size": 2,
                    "distance": "Cosine",
                }
            },
            "sparse_vectors": {"sparse": {"modifier": "idf"}},
        }
        store = _FakeQdrantStore(client, {})

        asyncio.run(
            store.search(
                collection="knowledge_base_v3",
                dense_vector=[0.3, 0.4],
                sparse_vector={"indices": [1, 2], "values": [0.5, 0.7]},
                query_mode="hybrid",
                limit=3,
                filter_spec=None,
                with_payload=True,
                with_vectors=False,
            )
        )

        call = client.query_points_calls[0]
        self.assertIsNone(call["using"])
        self.assertEqual(len(call["prefetch"]), 2)
        self.assertEqual(call["prefetch"][0].using, "dense")
        self.assertEqual(call["prefetch"][1].using, "sparse")
        self.assertEqual(call["query"].fusion, "rrf")

    def test_search_rejects_dense_vector_size_mismatch(self) -> None:
        client = _FakeClient()
        client.collections["decision_memory_v2"] = {
            "vectors": {
                "dense": {
                    "size": 1024,
                    "distance": "Cosine",
                }
            },
            "sparse_vectors": None,
        }
        store = _FakeQdrantStore(client, {})

        with self.assertRaisesRegex(ValueError, "expects dense vector size 1024, got 384"):
            asyncio.run(
                store.search(
                    collection="decision_memory_v2",
                    dense_vector=[0.1] * 384,
                    sparse_vector=None,
                    query_mode="dense",
                    limit=3,
                    filter_spec=None,
                    with_payload=True,
                    with_vectors=False,
                )
            )

    def test_retrieve_and_set_payload_are_supported(self) -> None:
        client = _FakeClient()
        client.collections["decision_memory_v2"] = {
            "vectors": {
                "dense": {
                    "size": 1024,
                    "distance": "Cosine",
                }
            },
            "sparse_vectors": None,
        }
        store = _FakeQdrantStore(client, {})

        points = asyncio.run(
            store.retrieve(
                collection="decision_memory_v2",
                ids=["d1"],
                with_payload=True,
                with_vectors=False,
            )
        )
        updated = asyncio.run(
            store.set_payload(
                collection="decision_memory_v2",
                ids=["d1"],
                payload={"status": "done"},
            )
        )

        self.assertEqual(points[0].id, "d1")
        self.assertEqual(updated, 1)
        self.assertEqual(client.set_payload_calls[0]["points"], ["d1"])

    def test_upsert_adds_sparse_vector_from_text_payload(self) -> None:
        client = _FakeClient()
        client.collections["knowledge_base_v3"] = {
            "vectors": {
                "dense": {
                    "size": 2,
                    "distance": "Cosine",
                }
            },
            "sparse_vectors": {"sparse": {"modifier": "idf"}},
        }
        store = _FakeQdrantStore(client, {})

        asyncio.run(
            store.upsert_points(
                collection="knowledge_base_v3",
                points=[
                    UpsertPoint(
                        id="k1",
                        vector=[0.1, 0.2],
                        payload={"text": "hello world hello"},
                    )
                ],
            )
        )

        point = client.upsert_calls[0]["points"][0]
        self.assertIn("dense", point.vector)
        self.assertIn("sparse", point.vector)

    def test_upsert_rejects_dense_vector_size_mismatch(self) -> None:
        client = _FakeClient()
        client.collections["decision_memory_v2"] = {
            "vectors": {
                "dense": {
                    "size": 1024,
                    "distance": "Cosine",
                }
            },
            "sparse_vectors": None,
        }
        store = _FakeQdrantStore(client, {})

        with self.assertRaisesRegex(ValueError, "expects dense vector size 1024, got 384"):
            asyncio.run(
                store.upsert_points(
                    collection="decision_memory_v2",
                    points=[
                        UpsertPoint(
                            id="d1",
                            vector=[0.1] * 384,
                            payload={"text": "legacy vector"},
                        )
                    ],
                )
            )


if __name__ == "__main__":
    unittest.main()
