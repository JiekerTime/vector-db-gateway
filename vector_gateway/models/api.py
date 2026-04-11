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
    vectors: list[list[float]]
    latency_ms: int
    queue_wait_ms: int
    batch_size: int


class TransformEmbedRequest(BaseModel):
    texts: list[str]
    model: str | None = None
    caller: str = "batch/migration"
    operation: str = "backfill"


class TransformEmbedResponse(BaseModel):
    model: str
    model_name: str
    vector_size: int | None = None
    vectors: list[list[float]]


class SearchRequest(BaseModel):
    caller: str = "interactive/default"
    operation: str = "search"
    collection: str
    text: str | None = None
    vector: list[float] | None = None
    model: str | None = None
    limit: int = 5
    filter: dict[str, Any] | None = None
    with_payload: bool = True
    with_vectors: bool = False

    @model_validator(mode="after")
    def validate_payload(self) -> "SearchRequest":
        if self.text is None and self.vector is None:
            raise ValueError("One of 'text' or 'vector' is required")
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
    wait: bool = True
    chunks: list[UpsertChunk]


class UpsertPoint(BaseModel):
    id: str | int | None = None
    vector: list[float]
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
    model: str | None = None
    query_model: str | None = None
    write_model: str | None = None
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    points_count: int | None = None
    indexed_vectors_count: int | None = None
    status: str | None = None


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


class AgentActionRequest(BaseModel):
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)


class QueueSnapshot(BaseModel):
    queue: str
    pending_requests: int
    pending_texts: int
    max_batch_size: int
    max_wait_ms: int


class StatusResponse(BaseModel):
    status: str
    uptime_s: int
    embedding_backend: dict[str, Any]
    qdrant: dict[str, Any]
    queues: list[QueueSnapshot]
    collections: list[CollectionInfo]
    router_rules: int
    metrics: dict[str, Any]
