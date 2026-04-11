"""Qdrant facade with simplified business-neutral request models."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from vector_gateway.config import CollectionConfig, QdrantConfig
from vector_gateway.models.api import CollectionInfo, SearchHit, UpsertPoint

logger = logging.getLogger(__name__)


class QdrantStore:
    """Wrapper around qdrant-client that hides raw driver details from callers."""

    def __init__(self, config: QdrantConfig, collections: dict[str, CollectionConfig]):
        self._config = config
        self._collections = collections
        self._client = None

    async def health(self) -> dict[str, Any]:
        try:
            collections = await asyncio.to_thread(self._client_get_collections)
        except Exception as exc:  # pragma: no cover - exercised indirectly
            return {"status": "down", "detail": str(exc)}
        return {"status": "ok", "collections": len(collections)}

    async def collection_infos(self) -> list[CollectionInfo]:
        infos: list[CollectionInfo] = []
        for name, meta in self._collections.items():
            info = CollectionInfo(
                name=name,
                vector_size=meta.vector_size,
                distance=meta.distance,
                owner=meta.owner,
                vector_name=meta.vector_name,
                model=meta.model,
                query_model=meta.query_model,
                write_model=meta.write_model,
                aliases=meta.aliases,
                description=meta.description,
            )
            try:
                details = await asyncio.to_thread(self._get_collection, name)
            except Exception as exc:  # pragma: no cover - probe failures
                info.status = f"error: {exc}"
            else:
                result = details.get("result") or {}
                info.points_count = _safe_int(result.get("points_count"))
                info.indexed_vectors_count = _safe_int(result.get("indexed_vectors_count"))
                info.status = str(result.get("status") or "unknown")
            infos.append(info)
        return infos

    async def search(
        self,
        *,
        collection: str,
        vector: list[float],
        limit: int,
        filter_spec: dict[str, Any] | None,
        with_payload: bool,
        with_vectors: bool,
    ) -> list[SearchHit]:
        meta = self._collection_meta(collection)
        query_filter = await asyncio.to_thread(self._build_filter, filter_spec)
        raw_hits = await asyncio.to_thread(
            self._search_sync,
            collection,
            meta.vector_name,
            vector,
            limit,
            query_filter,
            with_payload,
            with_vectors,
        )
        hits: list[SearchHit] = []
        for hit in raw_hits:
            hit_id = getattr(hit, "id", None)
            score = float(getattr(hit, "score", 0.0))
            payload = getattr(hit, "payload", None)
            item_vector = getattr(hit, "vector", None) if with_vectors else None
            hits.append(
                SearchHit(
                    id=str(hit_id),
                    score=score,
                    payload=payload,
                    vector=item_vector,
                )
            )
        return hits

    async def count(self, *, collection: str, filter_spec: dict[str, Any] | None) -> int:
        self._collection_meta(collection)
        query_filter = await asyncio.to_thread(self._build_filter, filter_spec)
        result = await asyncio.to_thread(self._count_sync, collection, query_filter)
        return int(result.count)

    async def upsert_points(
        self,
        *,
        collection: str,
        points: list[UpsertPoint],
        wait: bool = True,
    ) -> int:
        self._collection_meta(collection)
        count = await asyncio.to_thread(self._upsert_sync, collection, points, wait)
        return count

    def _client_get_collections(self):
        client = self._get_client()
        result = client.get_collections()
        return result.collections

    def _get_collection(self, collection: str) -> dict[str, Any]:
        client = self._get_client()
        result = client.get_collection(collection_name=collection)
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return dict(result)

    def _search_sync(
        self,
        collection: str,
        vector_name: str | None,
        vector: list[float],
        limit: int,
        query_filter,
        with_payload: bool,
        with_vectors: bool,
    ):
        client = self._get_client()
        query_vector = vector if vector_name is None else (vector_name, vector)
        return client.search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

    def _count_sync(self, collection: str, query_filter):
        client = self._get_client()
        return client.count(
            collection_name=collection,
            count_filter=query_filter,
            exact=True,
        )

    def _upsert_sync(self, collection: str, points: list[UpsertPoint], wait: bool) -> int:
        client = self._get_client()
        models = self._models()
        point_structs = [
            models.PointStruct(
                id=point.id if point.id is not None else None,
                vector=self._point_vector(collection, point.vector),
                payload=point.payload,
            )
            for point in points
        ]
        client.upsert(collection_name=collection, points=point_structs, wait=wait)
        return len(points)

    def _build_filter(self, filter_spec: dict[str, Any] | None):
        if not filter_spec:
            return None
        models = self._models()

        if "must" in filter_spec or "should" in filter_spec or "must_not" in filter_spec:
            return models.Filter(
                must=self._build_conditions(filter_spec.get("must")),
                should=self._build_conditions(filter_spec.get("should")),
                must_not=self._build_conditions(filter_spec.get("must_not")),
            )

        return models.Filter(must=self._build_conditions([{ "key": key, "match": value } for key, value in filter_spec.items()]))

    def _build_conditions(self, conditions: Any) -> list[Any] | None:
        if not conditions:
            return None
        models = self._models()
        built: list[Any] = []
        for item in conditions:
            if isinstance(item, dict) and "key" in item:
                key = item["key"]
                if "range" in item:
                    built.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(**item["range"]),
                        )
                    )
                    continue

                match_value = item.get("match")
                if isinstance(match_value, list):
                    built.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=match_value),
                        )
                    )
                else:
                    built.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=match_value),
                        )
                    )
                continue
            raise ValueError(f"Unsupported filter condition: {item!r}")
        return built

    def _collection_meta(self, collection: str) -> CollectionConfig:
        try:
            return self._collections[collection]
        except KeyError as exc:
            raise ValueError(f"Unknown collection: {collection}") from exc

    def _point_vector(self, collection: str, vector: list[float]) -> Any:
        meta = self._collection_meta(collection)
        if meta.vector_name:
            return {meta.vector_name: vector}
        return vector

    def _models(self):
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required for Qdrant access") from exc
        return models

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("qdrant-client is required for Qdrant access") from exc
            logger.info("Connecting to Qdrant: %s", self._config.url)
            self._client = QdrantClient(
                url=self._config.url,
                timeout=self._config.timeout,
            )
        return self._client


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
