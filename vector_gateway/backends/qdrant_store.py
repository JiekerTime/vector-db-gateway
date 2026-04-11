"""Qdrant facade with simplified business-neutral request models."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from vector_gateway.config import CollectionConfig, QdrantConfig
from vector_gateway.models.api import CollectionInfo, ScrollPoint, SearchHit, UpsertPoint

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

    async def ensure_collections(self) -> None:
        for name, meta in self._collections.items():
            await asyncio.to_thread(self._ensure_collection_sync, name, meta)

    async def collection_infos(self) -> list[CollectionInfo]:
        infos: list[CollectionInfo] = []
        for name, meta in self._collections.items():
            info = self._collection_info(name, meta)
            try:
                details = await asyncio.to_thread(self._get_collection, name)
            except Exception as exc:  # pragma: no cover - probe failures
                info.status = f"error: {exc}"
            else:
                result = self._collection_result(details)
                info.points_count = _safe_int(result.get("points_count"))
                info.indexed_vectors_count = _safe_int(result.get("indexed_vectors_count"))
                info.status = str(result.get("status") or "unknown")
            infos.append(info)
        return infos

    async def live_collection_infos(self) -> list[CollectionInfo]:
        infos: list[CollectionInfo] = []
        collections = await asyncio.to_thread(self._client_get_collections)
        for item in collections:
            name = getattr(item, "name", None)
            if not name:
                continue
            try:
                meta = self._collection_meta(str(name))
                info = self._collection_info(str(name), meta)
                details = await asyncio.to_thread(self._get_collection, str(name))
            except Exception as exc:  # pragma: no cover - probe failures
                info = CollectionInfo(
                    name=str(name),
                    vector_size=0,
                    distance="unknown",
                    owner="external",
                    status=f"error: {exc}",
                )
            else:
                result = self._collection_result(details)
                info.points_count = _safe_int(result.get("points_count"))
                info.indexed_vectors_count = _safe_int(result.get("indexed_vectors_count"))
                info.status = str(result.get("status") or "unknown")
            infos.append(info)
        return infos

    async def ensure_collection(
        self,
        *,
        collection: str,
        meta: CollectionConfig,
    ) -> tuple[bool, CollectionInfo]:
        created = False
        try:
            details = await asyncio.to_thread(self._get_collection, collection)
        except Exception:
            await asyncio.to_thread(self._create_collection, collection, meta)
            created = True
            details = await asyncio.to_thread(self._get_collection, collection)

        info = self._collection_info(collection, meta)
        result = self._collection_result(details)
        info.points_count = _safe_int(result.get("points_count"))
        info.indexed_vectors_count = _safe_int(result.get("indexed_vectors_count"))
        info.status = str(result.get("status") or "unknown")
        return created, info

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
        meta = await asyncio.to_thread(self._collection_meta, collection)
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
        await asyncio.to_thread(self._collection_meta, collection)
        query_filter = await asyncio.to_thread(self._build_filter, filter_spec)
        result = await asyncio.to_thread(self._count_sync, collection, query_filter)
        return int(result.count)

    async def scroll(
        self,
        *,
        collection: str,
        filter_spec: dict[str, Any] | None,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> list[ScrollPoint]:
        await asyncio.to_thread(self._collection_meta, collection)
        query_filter = await asyncio.to_thread(self._build_filter, filter_spec)
        raw_points = await asyncio.to_thread(
            self._scroll_sync,
            collection,
            query_filter,
            limit,
            with_payload,
            with_vectors,
        )
        points: list[ScrollPoint] = []
        for point in raw_points:
            point_id = getattr(point, "id", None)
            payload = getattr(point, "payload", None)
            item_vector = getattr(point, "vector", None) if with_vectors else None
            points.append(ScrollPoint(id=str(point_id), payload=payload, vector=item_vector))
        return points

    async def upsert_points(
        self,
        *,
        collection: str,
        points: list[UpsertPoint],
        wait: bool = True,
    ) -> int:
        await asyncio.to_thread(self._collection_meta, collection)
        count = await asyncio.to_thread(self._upsert_sync, collection, points, wait)
        return count

    def _collection_info(self, name: str, meta: CollectionConfig) -> CollectionInfo:
        return CollectionInfo(
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

    def _client_get_collections(self):
        client = self._get_client()
        result = client.get_collections()
        return result.collections

    def _ensure_collection_sync(self, collection: str, meta: CollectionConfig) -> None:
        try:
            details = self._get_collection(collection)
        except Exception:
            self._create_collection(collection, meta)
            logger.info(
                "Created Qdrant collection collection=%s vector_size=%s distance=%s vector_name=%s",
                collection,
                meta.vector_size,
                meta.distance,
                meta.vector_name,
            )
            return

        actual = self._extract_vector_shape(details)
        expected = {
            "vector_name": meta.vector_name,
            "vector_size": meta.vector_size,
            "distance": meta.distance.lower(),
        }
        if actual != expected:
            logger.warning(
                "Registered collection differs from Qdrant collection=%s expected=%s actual=%s",
                collection,
                expected,
                actual,
            )

    def _create_collection(self, collection: str, meta: CollectionConfig) -> None:
        client = self._get_client()
        models = self._models()
        vector_params = models.VectorParams(
            size=meta.vector_size,
            distance=self._distance_value(meta.distance),
        )
        vectors_config = vector_params if meta.vector_name is None else {meta.vector_name: vector_params}
        client.create_collection(collection_name=collection, vectors_config=vectors_config)

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
        if hasattr(client, "search"):
            query_vector = vector if vector_name is None else (vector_name, vector)
            return client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

        response = client.query_points(
            collection_name=collection,
            query=vector,
            using=vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return getattr(response, "points", response)

    def _count_sync(self, collection: str, query_filter):
        client = self._get_client()
        return client.count(
            collection_name=collection,
            count_filter=query_filter,
            exact=True,
        )

    def _scroll_sync(
        self,
        collection: str,
        query_filter,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ):
        client = self._get_client()
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return points

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

    def _distance_value(self, distance: str):
        models = self._models()
        name = distance.upper()
        try:
            return getattr(models.Distance, name)
        except AttributeError:
            raise ValueError(f"Unsupported Qdrant distance: {distance}") from None

    def _extract_vector_shape(self, details: dict[str, Any]) -> dict[str, Any]:
        result = self._collection_result(details)
        config = result.get("config") or {}
        params = config.get("params") or {}
        vectors = params.get("vectors")
        if isinstance(vectors, dict) and "size" not in vectors:
            vector_name, vector_config = next(iter(vectors.items()))
            return {
                "vector_name": vector_name,
                "vector_size": _safe_int(vector_config.get("size")),
                "distance": str(vector_config.get("distance") or "").lower(),
            }
        return {
            "vector_name": None,
            "vector_size": _safe_int((vectors or {}).get("size")),
            "distance": str((vectors or {}).get("distance") or "").lower(),
        }

    def _collection_result(self, details: dict[str, Any]) -> dict[str, Any]:
        result = details.get("result")
        if isinstance(result, dict):
            return result
        return details

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
        meta = self._collections.get(collection)
        if meta is not None:
            return meta

        details = self._get_collection(collection)
        shape = self._extract_vector_shape(details)
        vector_size = shape.get("vector_size")
        if vector_size is None:
            raise ValueError(f"Unable to infer vector size for collection: {collection}")
        return CollectionConfig(
            vector_size=vector_size,
            distance=str(shape.get("distance") or "Cosine"),
            owner="external",
            vector_name=shape.get("vector_name"),
        )

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
