"""Qdrant facade with simplified business-neutral request models."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from vector_gateway.config import CollectionConfig, QdrantConfig
from vector_gateway.core.sparse import sparse_terms
from vector_gateway.models.api import CollectionInfo, RetrievePoint, ScrollPoint, SearchHit, UpsertPoint

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

    async def ensure_alias(self, alias_name: str, collection: str) -> None:
        await asyncio.to_thread(self._ensure_alias_sync, alias_name, collection)

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
        dense_vector: list[float] | None,
        sparse_vector: dict[str, list[int] | list[float]] | None,
        query_mode: str,
        limit: int,
        filter_spec: dict[str, Any] | None,
        with_payload: bool,
        with_vectors: bool,
    ) -> list[SearchHit]:
        meta = await asyncio.to_thread(self._collection_meta, collection)
        query_filter = await asyncio.to_thread(self._build_filter, filter_spec)
        raw_hits = await asyncio.to_thread(
            self._query_sync,
            collection,
            meta,
            dense_vector,
            sparse_vector,
            query_mode,
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
            hits.append(SearchHit(id=str(hit_id), score=score, payload=payload, vector=item_vector))
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

    async def retrieve(
        self,
        *,
        collection: str,
        ids: list[str | int],
        with_payload: bool,
        with_vectors: bool,
    ) -> list[RetrievePoint]:
        await asyncio.to_thread(self._collection_meta, collection)
        raw_points = await asyncio.to_thread(self._retrieve_sync, collection, ids, with_payload, with_vectors)
        points: list[RetrievePoint] = []
        for point in raw_points:
            point_id = getattr(point, "id", None)
            payload = getattr(point, "payload", None)
            item_vector = getattr(point, "vector", None) if with_vectors else None
            points.append(RetrievePoint(id=str(point_id), payload=payload, vector=item_vector))
        return points

    async def set_payload(
        self,
        *,
        collection: str,
        ids: list[str | int],
        payload: dict[str, Any],
        wait: bool = True,
    ) -> int:
        await asyncio.to_thread(self._collection_meta, collection)
        return await asyncio.to_thread(self._set_payload_sync, collection, ids, payload, wait)

    async def patch_payload(
        self,
        *,
        collection: str,
        point_id: str | int,
        payload: dict[str, Any],
        wait: bool = True,
    ) -> int:
        await asyncio.to_thread(self._collection_meta, collection)
        return await asyncio.to_thread(self._set_payload_sync, collection, [point_id], payload, wait)

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
            sparse_vector_name=meta.sparse_vector_name,
            sparse_modifier=meta.sparse_modifier,
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
                "Created Qdrant collection collection=%s vector_size=%s distance=%s vector_name=%s sparse_vector_name=%s",
                collection,
                meta.vector_size,
                meta.distance,
                meta.vector_name,
                meta.sparse_vector_name,
            )
            return

        actual = self._extract_vector_shape(details)
        expected = {
            "vector_name": meta.vector_name,
            "vector_size": meta.vector_size,
            "distance": meta.distance.lower(),
            "sparse_vector_name": meta.sparse_vector_name,
        }
        if actual != expected:
            result = self._collection_result(details)
            points_count = _safe_int(result.get("points_count")) or 0
            indexed_vectors_count = _safe_int(result.get("indexed_vectors_count")) or 0
            if points_count == 0 and indexed_vectors_count == 0:
                self._recreate_collection(collection, meta)
                logger.info(
                    "Recreated empty Qdrant collection collection=%s expected=%s actual=%s",
                    collection,
                    expected,
                    actual,
                )
                return
            logger.warning(
                "Registered collection differs from Qdrant collection=%s expected=%s actual=%s",
                collection,
                expected,
                actual,
            )

    def _ensure_alias_sync(self, alias_name: str, collection: str) -> None:
        if alias_name == collection:
            return
        client = self._get_client()
        existing_collections = {getattr(item, "name", None) for item in self._client_get_collections()}
        if alias_name in existing_collections:
            logger.warning(
                "Skip alias sync because alias name is already a collection alias=%s target=%s",
                alias_name,
                collection,
            )
            return
        models = self._models()
        delete_op = models.DeleteAliasOperation(
            delete_alias=models.DeleteAlias(alias_name=alias_name),
        )
        create_op = models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias_name),
        )
        try:
            client.update_collection_aliases([delete_op, create_op])
        except Exception:
            client.update_collection_aliases([create_op])

    def _create_collection(self, collection: str, meta: CollectionConfig) -> None:
        client = self._get_client()
        models = self._models()
        vector_params = models.VectorParams(
            size=meta.vector_size,
            distance=self._distance_value(meta.distance),
        )
        vectors_config = vector_params if meta.vector_name is None else {meta.vector_name: vector_params}
        sparse_vectors_config = None
        if meta.sparse_vector_name:
            sparse_params = models.SparseVectorParams(
                modifier=self._sparse_modifier_value(meta.sparse_modifier),
            )
            sparse_vectors_config = {meta.sparse_vector_name: sparse_params}
        client.create_collection(
            collection_name=collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

    def _recreate_collection(self, collection: str, meta: CollectionConfig) -> None:
        client = self._get_client()
        client.delete_collection(collection_name=collection)
        self._create_collection(collection, meta)

    def _get_collection(self, collection: str) -> dict[str, Any]:
        client = self._get_client()
        result = client.get_collection(collection_name=collection)
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return dict(result)

    def _query_sync(
        self,
        collection: str,
        meta: CollectionConfig,
        dense_vector: list[float] | None,
        sparse_vector: dict[str, list[int] | list[float]] | None,
        query_mode: str,
        limit: int,
        query_filter,
        with_payload: bool,
        with_vectors: bool,
    ):
        client = self._get_client()
        models = self._models()
        normalized_mode = (query_mode or "dense").lower()
        sparse_query = self._sparse_query(models, sparse_vector)
        self._validate_dense_vector_size(collection, meta, dense_vector)

        if normalized_mode == "hybrid" and dense_vector and sparse_query and meta.sparse_vector_name:
            response = client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(query=dense_vector, using=meta.vector_name, limit=limit),
                    models.Prefetch(query=sparse_query, using=meta.sparse_vector_name, limit=limit),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            return getattr(response, "points", response)

        if normalized_mode == "sparse" and sparse_query and meta.sparse_vector_name:
            response = client.query_points(
                collection_name=collection,
                query=sparse_query,
                using=meta.sparse_vector_name,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            return getattr(response, "points", response)

        if dense_vector is None:
            raise ValueError(f"Collection '{collection}' requires a dense vector for query mode '{normalized_mode}'")

        if hasattr(client, "search"):
            query_vector = dense_vector if meta.vector_name is None else (meta.vector_name, dense_vector)
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
            query=dense_vector,
            using=meta.vector_name,
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

    def _retrieve_sync(
        self,
        collection: str,
        ids: list[str | int],
        with_payload: bool,
        with_vectors: bool,
    ):
        client = self._get_client()
        return client.retrieve(
            collection_name=collection,
            ids=ids,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

    def _set_payload_sync(
        self,
        collection: str,
        ids: list[str | int],
        payload: dict[str, Any],
        wait: bool,
    ) -> int:
        client = self._get_client()
        client.set_payload(collection_name=collection, payload=payload, points=ids, wait=wait)
        return len(ids)

    def _upsert_sync(self, collection: str, points: list[UpsertPoint], wait: bool) -> int:
        client = self._get_client()
        models = self._models()
        point_structs = [
            models.PointStruct(
                id=point.id if point.id is not None else None,
                vector=self._point_vector(collection, point.vector, point.payload),
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

        return models.Filter(must=self._build_conditions([{"key": key, "match": value} for key, value in filter_spec.items()]))

    def _distance_value(self, distance: str):
        models = self._models()
        name = distance.upper()
        try:
            return getattr(models.Distance, name)
        except AttributeError:
            raise ValueError(f"Unsupported Qdrant distance: {distance}") from None

    def _sparse_modifier_value(self, modifier: str | None):
        if not modifier:
            return None
        models = self._models()
        name = modifier.upper()
        try:
            return getattr(models.Modifier, name)
        except AttributeError:
            raise ValueError(f"Unsupported Qdrant sparse modifier: {modifier}") from None

    def _extract_vector_shape(self, details: dict[str, Any]) -> dict[str, Any]:
        result = self._collection_result(details)
        config = result.get("config") or {}
        params = config.get("params") or {}
        vectors = params.get("vectors")
        sparse_vectors = params.get("sparse_vectors")
        sparse_vector_name = next(iter(sparse_vectors.keys()), None) if isinstance(sparse_vectors, dict) else None
        if isinstance(vectors, dict) and "size" not in vectors:
            vector_name, vector_config = next(iter(vectors.items()))
            return {
                "vector_name": vector_name,
                "vector_size": _safe_int(vector_config.get("size")),
                "distance": str(vector_config.get("distance") or "").lower(),
                "sparse_vector_name": sparse_vector_name,
            }
        return {
            "vector_name": None,
            "vector_size": _safe_int((vectors or {}).get("size")),
            "distance": str((vectors or {}).get("distance") or "").lower(),
            "sparse_vector_name": sparse_vector_name,
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
            sparse_vector_name=shape.get("sparse_vector_name"),
        )

    def _point_vector(self, collection: str, vector: list[float] | dict[str, Any], payload: dict[str, Any]) -> Any:
        meta = self._collection_meta(collection)
        if isinstance(vector, dict):
            built = dict(vector)
            if meta.sparse_vector_name and meta.sparse_vector_name not in built:
                sparse = self._sparse_vector_from_payload(payload)
                if sparse is not None:
                    built[meta.sparse_vector_name] = sparse
            dense_vector = self._dense_vector_value(meta, built)
            if dense_vector is None:
                raise ValueError(f"Collection '{collection}' requires a dense vector payload")
            self._validate_dense_vector_size(collection, meta, dense_vector)
            if meta.vector_name is None and len(built) == 1 and meta.sparse_vector_name not in built:
                return next(iter(built.values()))
            return built

        self._validate_dense_vector_size(collection, meta, vector)
        if meta.vector_name or meta.sparse_vector_name:
            built: dict[str, Any] = {}
            dense_name = meta.vector_name or "dense"
            built[dense_name] = vector
            if meta.sparse_vector_name:
                sparse = self._sparse_vector_from_payload(payload)
                if sparse is not None:
                    built[meta.sparse_vector_name] = sparse
            return built
        return vector

    def _sparse_vector_from_payload(self, payload: dict[str, Any]) -> Any | None:
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        indices, values = sparse_terms(text)
        if not indices:
            return None
        models = self._models()
        return models.SparseVector(indices=indices, values=values)

    def _sparse_query(self, models, sparse_vector: dict[str, list[int] | list[float]] | None):
        if not sparse_vector:
            return None
        indices = sparse_vector.get("indices") or []
        values = sparse_vector.get("values") or []
        if not indices:
            return None
        return models.SparseVector(indices=indices, values=values)

    def _validate_dense_vector_size(
        self,
        collection: str,
        meta: CollectionConfig,
        dense_vector: list[float] | None,
    ) -> None:
        if dense_vector is None:
            return
        actual_size = len(dense_vector)
        if actual_size != meta.vector_size:
            raise ValueError(
                f"Collection '{collection}' expects dense vector size {meta.vector_size}, got {actual_size}"
            )

    def _dense_vector_value(self, meta: CollectionConfig, vector: dict[str, Any]) -> list[float] | None:
        if meta.vector_name:
            dense = vector.get(meta.vector_name)
            return dense if isinstance(dense, list) else None
        if meta.sparse_vector_name:
            for key, value in vector.items():
                if key == meta.sparse_vector_name:
                    continue
                if isinstance(value, list):
                    return value
            return None
        if len(vector) == 1:
            value = next(iter(vector.values()))
            return value if isinstance(value, list) else None
        return None

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
