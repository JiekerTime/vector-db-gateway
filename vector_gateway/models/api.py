"""HTTP request and response models."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, model_validator

_MAX_QUERY_LIMIT = 1000
_MAX_BATCH_ITEMS = 1000


def _is_blank(value: str | None) -> bool:
    return value is None or not value.strip()


def _validate_text_value(value: str | None, field_name: str) -> None:
    if _is_blank(value):
        raise ValueError(f"'{field_name}' must not be blank")


def _validate_text_items(items: list[str] | None, field_name: str) -> None:
    if not items:
        raise ValueError(f"'{field_name}' must not be empty")
    for index, item in enumerate(items):
        if _is_blank(item):
            raise ValueError(f"'{field_name}[{index}]' must not be blank")


def _validate_dense_vector(vector: list[float] | None, field_name: str) -> None:
    if vector is None:
        return
    if not vector:
        raise ValueError(f"'{field_name}' must not be empty")
    for index, value in enumerate(vector):
        if not math.isfinite(value):
            raise ValueError(f"'{field_name}[{index}]' must be a finite number")


def _validate_sparse_vector(value: dict[str, Any], field_name: str) -> None:
    indices = value.get("indices")
    values = value.get("values")
    if not isinstance(indices, list) or not isinstance(values, list):
        raise ValueError(f"'{field_name}' sparse vector must include list 'indices' and 'values'")
    if not indices:
        raise ValueError(f"'{field_name}.indices' must not be empty")
    if len(indices) != len(values):
        raise ValueError(f"'{field_name}' sparse vector requires the same number of indices and values")
    previous = -1
    for index, item in enumerate(indices):
        if not isinstance(item, int):
            raise ValueError(f"'{field_name}.indices[{index}]' must be an integer")
        if item < 0:
            raise ValueError(f"'{field_name}.indices[{index}]' must be non-negative")
        if item <= previous:
            raise ValueError(f"'{field_name}.indices' must be strictly increasing")
        previous = item
    for index, item in enumerate(values):
        if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"'{field_name}.values[{index}]' must be a finite number")


def _validate_named_vector_map(vector: dict[str, Any], field_name: str) -> None:
    if not vector:
        raise ValueError(f"'{field_name}' must not be empty")
    for key, value in vector.items():
        if not key or not key.strip():
            raise ValueError(f"'{field_name}' contains an empty vector name")
        if isinstance(value, list):
            _validate_dense_vector(value, f"{field_name}.{key}")
            continue
        if isinstance(value, dict):
            _validate_sparse_vector(value, f"{field_name}.{key}")
            continue
        raise ValueError(f"'{field_name}.{key}' must be a dense float list or sparse vector object")


def _validate_ids(items: list[str | int], field_name: str) -> None:
    if not items:
        raise ValueError(f"'{field_name}' must not be empty")
    if len(items) > _MAX_BATCH_ITEMS:
        raise ValueError(f"'{field_name}' must contain at most {_MAX_BATCH_ITEMS} items")
    for index, item in enumerate(items):
        if isinstance(item, str) and not item.strip():
            raise ValueError(f"'{field_name}[{index}]' must not be blank")


def _validate_limit(limit: int, field_name: str) -> None:
    if limit < 1 or limit > _MAX_QUERY_LIMIT:
        raise ValueError(f"'{field_name}' must be between 1 and {_MAX_QUERY_LIMIT}")


class EmbedRequest(BaseModel):
    caller: str = "interactive/default"
    operation: str = "query"
    text: str | None = None
    texts: list[str] | None = None
    model: str | None = None
    device: str | None = None
    collection_hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> "EmbedRequest":
        if self.text is not None and self.texts:
            raise ValueError("Provide either 'text' or 'texts', not both")
        if self.text is None and not self.texts:
            raise ValueError("One of 'text' or 'texts' is required")
        if self.text is not None:
            _validate_text_value(self.text, "text")
        if self.texts is not None:
            _validate_text_items(self.texts, "texts")
        return self

    def text_items(self) -> list[str]:
        if self.texts:
            return self.texts
        if self.text is not None:
            return [self.text]
        return []


class EmbedResponse(BaseModel):
    request_id: str
    queue: str
    model: str
    device: str
    vectors: list[list[float]]
    latency_ms: int
    queue_wait_ms: int
    batch_size: int


class TransformEmbedRequest(BaseModel):
    texts: list[str]
    model: str | None = None
    device: str | None = None
    caller: str = "batch/migration"
    operation: str = "backfill"

    @model_validator(mode="after")
    def validate_payload(self) -> "TransformEmbedRequest":
        _validate_text_items(self.texts, "texts")
        return self


class TransformEmbedResponse(BaseModel):
    model: str
    model_name: str
    device: str
    vector_size: int | None = None
    vectors: list[list[float]]


class SearchRequest(BaseModel):
    caller: str = "interactive/default"
    operation: str = "search"
    collection: str
    text: str | None = None
    query_text: str | None = None
    vector: list[float] | None = None
    model: str | None = None
    device: str | None = None
    limit: int = 5
    filter: dict[str, Any] | None = None
    with_payload: bool = True
    with_vectors: bool = False
    search_mode: str = "auto"

    @model_validator(mode="after")
    def validate_payload(self) -> "SearchRequest":
        if self.text is not None:
            _validate_text_value(self.text, "text")
        if self.query_text is not None:
            _validate_text_value(self.query_text, "query_text")
        if self.text is None and self.query_text is None and self.vector is None:
            raise ValueError("One of 'text', 'query_text', or 'vector' is required")
        if self.query_text is None and self.text is not None:
            self.query_text = self.text
        _validate_dense_vector(self.vector, "vector")
        _validate_limit(self.limit, "limit")
        mode = (self.search_mode or "auto").lower()
        if mode not in {"auto", "dense", "sparse", "hybrid"}:
            raise ValueError("'search_mode' must be one of: auto, dense, sparse, hybrid")
        self.search_mode = mode
        if mode == "sparse":
            if self.vector is not None:
                raise ValueError("'vector' is not allowed when 'search_mode' is 'sparse'")
            if _is_blank(self.query_text):
                raise ValueError("'query_text' or 'text' is required when 'search_mode' is 'sparse'")
        return self


class SearchHit(BaseModel):
    id: str
    score: float
    payload: dict[str, Any] | None = None
    vector: list[float] | dict[str, Any] | None = None


class SearchResponse(BaseModel):
    request_id: str
    queue: str
    collection: str
    hits: list[SearchHit]
    latency_ms: int
    queue_wait_ms: int


class ScrollRequest(BaseModel):
    caller: str = "interactive/default"
    operation: str = "scroll"
    collection: str
    filter: dict[str, Any] | None = None
    limit: int = 100
    with_payload: bool = True
    with_vectors: bool = False

    @model_validator(mode="after")
    def validate_payload(self) -> "ScrollRequest":
        _validate_limit(self.limit, "limit")
        return self


class ScrollPoint(BaseModel):
    id: str
    payload: dict[str, Any] | None = None
    vector: list[float] | dict[str, Any] | None = None


class ScrollResponse(BaseModel):
    request_id: str
    queue: str
    collection: str
    points: list[ScrollPoint]
    latency_ms: int
    queue_wait_ms: int


class CountRequest(BaseModel):
    caller: str = "interactive/default"
    operation: str = "count"
    collection: str
    filter: dict[str, Any] | None = None


class CountResponse(BaseModel):
    request_id: str
    queue: str
    collection: str
    count: int
    latency_ms: int
    queue_wait_ms: int


class UpsertChunk(BaseModel):
    id: str | int | None = None
    text: str | None = None
    vector: list[float] | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> "UpsertChunk":
        if self.text is None and self.vector is None:
            raise ValueError("Each chunk requires either 'text' or 'vector'")
        if self.text is not None:
            _validate_text_value(self.text, "text")
        _validate_dense_vector(self.vector, "vector")
        return self


class UpsertChunksRequest(BaseModel):
    caller: str = "batch/default"
    operation: str = "upsert"
    collection: str
    model: str | None = None
    device: str | None = None
    wait: bool = True
    chunks: list[UpsertChunk]

    @model_validator(mode="after")
    def validate_payload(self) -> "UpsertChunksRequest":
        if not self.chunks:
            raise ValueError("'chunks' must not be empty")
        if len(self.chunks) > _MAX_BATCH_ITEMS:
            raise ValueError(f"'chunks' must contain at most {_MAX_BATCH_ITEMS} items")
        return self


class UpsertPoint(BaseModel):
    id: str | int | None = None
    vector: list[float] | dict[str, Any]
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> "UpsertPoint":
        if isinstance(self.vector, list):
            _validate_dense_vector(self.vector, "vector")
        else:
            _validate_named_vector_map(self.vector, "vector")
        return self


class UpsertPointsRequest(BaseModel):
    caller: str = "batch/default"
    operation: str = "upsert"
    collection: str
    wait: bool = True
    points: list[UpsertPoint]

    @model_validator(mode="after")
    def validate_payload(self) -> "UpsertPointsRequest":
        if not self.points:
            raise ValueError("'points' must not be empty")
        if len(self.points) > _MAX_BATCH_ITEMS:
            raise ValueError(f"'points' must contain at most {_MAX_BATCH_ITEMS} items")
        return self


class UpsertResponse(BaseModel):
    request_id: str
    queue: str
    collection: str
    upserted: int
    latency_ms: int
    queue_wait_ms: int


class CollectionInfo(BaseModel):
    name: str
    vector_size: int
    distance: str
    owner: str
    vector_name: str | None = None
    sparse_vector_name: str | None = None
    sparse_modifier: str | None = None
    model: str | None = None
    query_model: str | None = None
    write_model: str | None = None
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    points_count: int | None = None
    indexed_vectors_count: int | None = None
    status: str | None = None


class EnsureCollectionRequest(BaseModel):
    collection: str
    vector_size: int
    distance: str = "Cosine"
    vector_name: str | None = None
    sparse_vector_name: str | None = None
    sparse_modifier: str | None = None
    owner: str = "external"
    model: str | None = None
    query_model: str | None = None
    write_model: str | None = None
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "EnsureCollectionRequest":
        _validate_text_value(self.collection, "collection")
        if self.vector_size < 1:
            raise ValueError("'vector_size' must be greater than 0")
        if self.vector_name is not None:
            _validate_text_value(self.vector_name, "vector_name")
        if self.sparse_vector_name is not None:
            _validate_text_value(self.sparse_vector_name, "sparse_vector_name")
        if self.vector_name and self.sparse_vector_name and self.vector_name == self.sparse_vector_name:
            raise ValueError("'vector_name' and 'sparse_vector_name' must be different")
        for index, alias in enumerate(self.aliases):
            if not alias.strip():
                raise ValueError(f"'aliases[{index}]' must not be blank")
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("'aliases' must not contain duplicates")
        return self


class EnsureCollectionResponse(BaseModel):
    created: bool
    collection: CollectionInfo


class EmbeddingModelInfo(BaseModel):
    name: str
    backend: str
    model_name: str
    vector_size: int | None = None
    distance: str = "Cosine"
    normalize_embeddings: bool | None = None
    device: str | None = None


class CapabilityAction(BaseModel):
    name: str
    endpoint: str
    description: str


class CapabilitiesResponse(BaseModel):
    service: str
    version: str
    actions: list[CapabilityAction]
    queues: list[str]
    models: list[EmbeddingModelInfo]
    collections: list[CollectionInfo]
    logical_collections: list["LogicalCollectionInfo"] = Field(default_factory=list)


class AgentActionRequest(BaseModel):
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)


class QueueSnapshot(BaseModel):
    queue: str
    pending_requests: int
    pending_texts: int
    max_batch_size: int
    max_wait_ms: int
    preferred_device: str | None = None


class StatusResponse(BaseModel):
    status: str
    uptime_s: int
    embedding_backend: dict[str, Any]
    qdrant: dict[str, Any]
    queues: list[QueueSnapshot]
    collections: list[CollectionInfo]
    logical_collections: list["LogicalCollectionInfo"] = Field(default_factory=list)
    router_rules: int
    metrics: dict[str, Any]


class LogicalCollectionMigrationState(BaseModel):
    state: str
    next_target: str | None = None
    rollback_target: str | None = None
    task_id: str | None = None
    shadow_read_targets: list[str] = Field(default_factory=list)
    last_verify_at: str | None = None
    last_verify_result: str | None = None
    last_cutover_at: str | None = None
    note: str | None = None
    updated_at: str | None = None
    recent_events: list["MigrationEvent"] = Field(default_factory=list)


class LogicalCollectionInfo(BaseModel):
    name: str
    alias_name: str | None = None
    default_query_mode: str = "dense"
    configured_read_targets: list[str] = Field(default_factory=list)
    configured_write_targets: list[str] = Field(default_factory=list)
    current_read_target: str | None = None
    current_write_targets: list[str] = Field(default_factory=list)
    read_collection: CollectionInfo | None = None
    write_collections: list[CollectionInfo] = Field(default_factory=list)
    migration: LogicalCollectionMigrationState


class MigrationActionRequest(BaseModel):
    task_id: str | None = None
    verify_result: str | None = None
    note: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MigrationEvent(BaseModel):
    id: int
    logical_name: str
    event: str
    state: str
    task_id: str | None = None
    note: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class DoMigWindow(BaseModel):
    start: str
    stop_dispatch: str
    pause_at: str


class DoMigQueueItem(BaseModel):
    id: str
    logical_collection: str
    task_config: dict[str, Any] | None = None
    shards: list[str] = Field(default_factory=list)
    window: DoMigWindow
    status: str = "queued"
    task_id: str | None = None
    attempts: int = 0
    last_error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    note: str | None = None
    sequence: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DoMigQueueImportRequest(BaseModel):
    items: list[DoMigQueueItem] = Field(default_factory=list)


class DoMigRunResponse(BaseModel):
    action: str
    now: str
    items: list[DoMigQueueItem] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    caller: str = "interactive/default"
    operation: str = "retrieve"
    collection: str
    ids: list[str | int]
    with_payload: bool = True
    with_vectors: bool = False

    @model_validator(mode="after")
    def validate_payload(self) -> "RetrieveRequest":
        _validate_ids(self.ids, "ids")
        return self


class RetrievePoint(BaseModel):
    id: str
    payload: dict[str, Any] | None = None
    vector: list[float] | dict[str, Any] | None = None


class RetrieveResponse(BaseModel):
    request_id: str
    queue: str
    collection: str
    points: list[RetrievePoint]
    latency_ms: int
    queue_wait_ms: int


class PayloadSetRequest(BaseModel):
    caller: str = "batch/default"
    operation: str = "upsert"
    collection: str
    ids: list[str | int]
    payload: dict[str, Any]
    wait: bool = True

    @model_validator(mode="after")
    def validate_payload(self) -> "PayloadSetRequest":
        _validate_ids(self.ids, "ids")
        if not self.payload:
            raise ValueError("'payload' must not be empty")
        return self


class PayloadPatchRequest(BaseModel):
    caller: str = "batch/default"
    operation: str = "upsert"
    collection: str
    id: str | int
    payload: dict[str, Any]
    wait: bool = True

    @model_validator(mode="after")
    def validate_payload(self) -> "PayloadPatchRequest":
        if isinstance(self.id, str) and not self.id.strip():
            raise ValueError("'id' must not be blank")
        if not self.payload:
            raise ValueError("'payload' must not be empty")
        return self


class PayloadUpdateResponse(BaseModel):
    request_id: str
    queue: str
    collection: str
    updated: int
    latency_ms: int
    queue_wait_ms: int


class TransformSparseRequest(BaseModel):
    texts: list[str]

    @model_validator(mode="after")
    def validate_payload(self) -> "TransformSparseRequest":
        _validate_text_items(self.texts, "texts")
        return self


class SparseVectorPayload(BaseModel):
    indices: list[int]
    values: list[float]


class TransformSparseResponse(BaseModel):
    vectors: list[SparseVectorPayload]


class TransformMetadataPrefixItem(BaseModel):
    text: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> "TransformMetadataPrefixItem":
        _validate_text_value(self.text, "text")
        return self


class TransformMetadataPrefixRequest(BaseModel):
    collection: str
    items: list[TransformMetadataPrefixItem]

    @model_validator(mode="after")
    def validate_payload(self) -> "TransformMetadataPrefixRequest":
        _validate_text_value(self.collection, "collection")
        if not self.items:
            raise ValueError("'items' must not be empty")
        if len(self.items) > _MAX_BATCH_ITEMS:
            raise ValueError(f"'items' must contain at most {_MAX_BATCH_ITEMS} items")
        return self


class TransformMetadataPrefixResult(BaseModel):
    text: str
    payload: dict[str, Any] = Field(default_factory=dict)
    prefix: str | None = None


class TransformMetadataPrefixResponse(BaseModel):
    items: list[TransformMetadataPrefixResult]
