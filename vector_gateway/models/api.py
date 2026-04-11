"""HTTP request and response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


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
        if self.text is None and not self.texts:
            raise ValueError("One of 'text' or 'texts' is required")
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
        if self.text is None and self.vector is None:
            raise ValueError("One of 'text' or 'vector' is required")
        if self.query_text is None and self.text is not None:
            self.query_text = self.text
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
        return self


class UpsertChunksRequest(BaseModel):
    caller: str = "batch/default"
    operation: str = "upsert"
    collection: str
    model: str | None = None
    device: str | None = None
    wait: bool = True
    chunks: list[UpsertChunk]


class UpsertPoint(BaseModel):
    id: str | int | None = None
    vector: list[float] | dict[str, Any]
    payload: dict[str, Any] = Field(default_factory=dict)


class UpsertPointsRequest(BaseModel):
    caller: str = "batch/default"
    operation: str = "upsert"
    collection: str
    wait: bool = True
    points: list[UpsertPoint]


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


class PayloadPatchRequest(BaseModel):
    caller: str = "batch/default"
    operation: str = "upsert"
    collection: str
    id: str | int
    payload: dict[str, Any]
    wait: bool = True


class PayloadUpdateResponse(BaseModel):
    request_id: str
    queue: str
    collection: str
    updated: int
    latency_ms: int
    queue_wait_ms: int


class TransformSparseRequest(BaseModel):
    texts: list[str]


class SparseVectorPayload(BaseModel):
    indices: list[int]
    values: list[float]


class TransformSparseResponse(BaseModel):
    vectors: list[SparseVectorPayload]
