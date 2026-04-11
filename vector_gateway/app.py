"""FastAPI application for vector-db-gateway."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse

from vector_gateway.backends import LocalEmbeddingBackend, QdrantStore
from vector_gateway.config import EmbeddingModelConfig, GatewayConfig, load_config
from vector_gateway.core.batching import EmbeddingBatcher
from vector_gateway.core.metrics import MetricsStore
from vector_gateway.core.router import Router
from vector_gateway.core.scheduler import FairSelector, JobScheduler
from vector_gateway.models import (
    AgentActionRequest,
    CapabilitiesResponse,
    CapabilityAction,
    CollectionInfo,
    CountRequest,
    CountResponse,
    EmbeddingModelInfo,
    EmbedRequest,
    EmbedResponse,
    QueueSnapshot,
    SearchRequest,
    SearchResponse,
    StatusResponse,
    TransformEmbedRequest,
    TransformEmbedResponse,
    UpsertChunk,
    UpsertChunksRequest,
    UpsertPoint,
    UpsertPointsRequest,
    UpsertResponse,
)

_config: GatewayConfig
_router: Router
_metrics: MetricsStore
_selector: FairSelector
_embed_backend: LocalEmbeddingBackend
_embed_batcher: EmbeddingBatcher
_job_scheduler: JobScheduler
_qdrant: QdrantStore
_model_registry: dict[str, EmbeddingModelConfig]
_started_at: float = 0.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vector-gateway")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _config, _router, _metrics, _selector
    global _embed_backend, _embed_batcher, _job_scheduler, _qdrant, _model_registry, _started_at

    _started_at = time.monotonic()
    _config = load_config()
    logging.getLogger().setLevel(getattr(logging, _config.log_level.upper(), logging.INFO))

    _router = Router.from_config(_config)
    _metrics = MetricsStore()
    _selector = FairSelector(_config.fairness)
    _model_registry = _build_model_registry(_config)
    _embed_backend = LocalEmbeddingBackend(_config.embedding, _model_registry)
    _qdrant = QdrantStore(_config.qdrant, _config.collections)
    try:
        await _qdrant.ensure_collections()
    except Exception:
        logger.exception("Failed to bootstrap registered collections in Qdrant")
    _embed_batcher = EmbeddingBatcher(
        backend=_embed_backend,
        queue_config=_config.queues,
        selector=_selector,
        metrics=_metrics,
    )
    _job_scheduler = JobScheduler(selector=_selector, metrics=_metrics)
    await _embed_batcher.start()
    await _job_scheduler.start()

    logger.info(
        "vector-db-gateway started port=%s queues=%s collections=%d rules=%d",
        _config.port,
        sorted(_config.queues.keys()),
        len(_config.collections),
        _router.rule_count,
    )
    yield
    await _embed_batcher.stop()
    await _job_scheduler.stop()
    logger.info("vector-db-gateway stopped")


app = FastAPI(title="vector-db-gateway", version="0.1.0", lifespan=lifespan)


def _check_api_key(x_api_key: str = Header(default="")) -> None:
    if x_api_key != _config.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
async def health():
    qdrant_status = await _qdrant.health()
    status = "ok" if qdrant_status.get("status") == "ok" else "degraded"
    return {
        "status": status,
        "uptime_s": int(time.monotonic() - _started_at),
        "qdrant": qdrant_status,
    }


@app.get("/status", response_model=StatusResponse)
async def status(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    queue_states = await _queue_snapshots()
    collections = await _qdrant.collection_infos()
    qdrant_status = await _qdrant.health()
    status_label = "ok" if qdrant_status.get("status") == "ok" else "degraded"
    return StatusResponse(
        status=status_label,
        uptime_s=int(time.monotonic() - _started_at),
        embedding_backend=_embed_backend.status(),
        qdrant=qdrant_status,
        queues=queue_states,
        collections=collections,
        router_rules=_router.rule_count,
        metrics=_metrics.snapshot(),
    )


@app.get("/queues", response_model=list[QueueSnapshot])
async def queues(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    return await _queue_snapshots()


@app.get("/collections", response_model=list[CollectionInfo])
async def collections(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    return await _qdrant.collection_infos()


@app.get("/models", response_model=list[EmbeddingModelInfo])
async def models(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    return _model_infos()


@app.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    return CapabilitiesResponse(
        service="vector-db-gateway",
        version="0.1.0",
        actions=[
            CapabilityAction(name="embed", endpoint="/embed", description="Generate dense embeddings"),
            CapabilityAction(name="transform_embed", endpoint="/transform/embed", description="Migration-safe embedding callback"),
            CapabilityAction(name="search", endpoint="/search", description="Search a registered collection"),
            CapabilityAction(name="count", endpoint="/count", description="Count records in a registered collection"),
            CapabilityAction(name="upsert_chunks", endpoint="/upsert/chunks", description="Upsert chunks with optional embedding"),
            CapabilityAction(name="upsert_points", endpoint="/upsert/points", description="Upsert points with supplied vectors"),
            CapabilityAction(name="agent_action", endpoint="/agent/action", description="Single action endpoint for machine callers"),
        ],
        queues=sorted(_config.queues.keys()),
        models=_model_infos(),
        collections=await _qdrant.collection_infos(),
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    queue_depths = _combined_queue_depths()
    return _metrics.render_prometheus(queue_depths)


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    request_id = _request_id()
    started = time.monotonic()
    route = _router.resolve(request.caller, request.operation)
    device = _resolved_request_device(route.queue_name, request.device)
    model_ref = request.model or _default_model_ref(request.collection_hint, purpose="query")
    result = await _embed_batcher.submit(
        request_id=request_id,
        texts=request.text_items(),
        route=route,
        model_name=model_ref,
        device=device,
    )
    latency_ms = int((time.monotonic() - started) * 1000)
    _metrics.observe_request("embed", latency_ms=latency_ms, queue_wait_ms=result.queue_wait_ms)
    _, profile = _embed_backend.resolve_profile(model_ref, device)
    return EmbedResponse(
        request_id=request_id,
        queue=route.queue_name,
        model=profile.model_name,
        device=profile.device or _config.embedding.device,
        vectors=result.vectors,
        latency_ms=latency_ms,
        queue_wait_ms=result.queue_wait_ms,
        batch_size=result.batch_size,
    )


@app.post("/transform/embed", response_model=TransformEmbedResponse)
async def transform_embed(request: TransformEmbedRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    route = _router.resolve(request.caller, request.operation)
    device = _resolved_request_device(route.queue_name, request.device)
    model_ref = request.model or _default_model_ref(purpose="write")
    result = await _embed_batcher.submit(
        request_id=_request_id(),
        texts=request.texts,
        route=route,
        model_name=model_ref,
        device=device,
    )
    _, profile = _embed_backend.resolve_profile(model_ref, device)
    return TransformEmbedResponse(
        model=model_ref,
        model_name=profile.model_name,
        device=profile.device or _config.embedding.device,
        vector_size=profile.vector_size,
        vectors=result.vectors,
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    request_id = _request_id()
    route = _router.resolve(request.caller, request.operation)
    device = _resolved_request_device(route.queue_name, request.device)
    started = time.monotonic()
    vector = request.vector
    queue_wait_ms = 0
    if vector is None:
        model_ref = request.model or _default_model_ref(request.collection, purpose="query")
        embed_result = await _embed_batcher.submit(
            request_id=request_id,
            texts=[request.text or ""],
            route=route,
            model_name=model_ref,
            device=device,
        )
        vector = embed_result.vectors[0]
        queue_wait_ms += embed_result.queue_wait_ms

    async def run_search():
        return await _qdrant.search(
            collection=request.collection,
            vector=vector or [],
            limit=request.limit,
            filter_spec=request.filter,
            with_payload=request.with_payload,
            with_vectors=request.with_vectors,
        )

    hits, scheduled_wait_ms = await _job_scheduler.submit(
        request_id=request_id,
        endpoint="search",
        route=route,
        factory=run_search,
    )
    total_wait_ms = queue_wait_ms + scheduled_wait_ms
    latency_ms = int((time.monotonic() - started) * 1000)
    return SearchResponse(
        request_id=request_id,
        queue=route.queue_name,
        collection=request.collection,
        hits=hits,
        latency_ms=latency_ms,
        queue_wait_ms=total_wait_ms,
    )


@app.post("/count", response_model=CountResponse)
async def count(request: CountRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    request_id = _request_id()
    started = time.monotonic()
    route = _router.resolve(request.caller, request.operation)

    async def run_count():
        return await _qdrant.count(collection=request.collection, filter_spec=request.filter)

    total, queue_wait_ms = await _job_scheduler.submit(
        request_id=request_id,
        endpoint="count",
        route=route,
        factory=run_count,
    )
    latency_ms = int((time.monotonic() - started) * 1000)
    return CountResponse(
        request_id=request_id,
        queue=route.queue_name,
        collection=request.collection,
        count=total,
        latency_ms=latency_ms,
        queue_wait_ms=queue_wait_ms,
    )


@app.post("/upsert/chunks", response_model=UpsertResponse)
async def upsert_chunks(request: UpsertChunksRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    request_id = _request_id()
    started = time.monotonic()
    route = _router.resolve(request.caller, request.operation)
    device = _resolved_request_device(route.queue_name, request.device)

    points, embed_wait_ms = await _prepare_upsert_points(
        request.chunks,
        route,
        request.model,
        device,
        request_id,
    )

    async def run_upsert():
        return await _qdrant.upsert_points(
            collection=request.collection,
            points=points,
            wait=request.wait,
        )

    upserted, scheduled_wait_ms = await _job_scheduler.submit(
        request_id=request_id,
        endpoint="upsert_chunks",
        route=route,
        factory=run_upsert,
    )
    latency_ms = int((time.monotonic() - started) * 1000)
    return UpsertResponse(
        request_id=request_id,
        queue=route.queue_name,
        collection=request.collection,
        upserted=upserted,
        latency_ms=latency_ms,
        queue_wait_ms=embed_wait_ms + scheduled_wait_ms,
    )


@app.post("/upsert/points", response_model=UpsertResponse)
async def upsert_points(request: UpsertPointsRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    request_id = _request_id()
    started = time.monotonic()
    route = _router.resolve(request.caller, request.operation)
    points = [UpsertPoint(id=point.id, vector=point.vector, payload=point.payload) for point in request.points]

    async def run_upsert():
        return await _qdrant.upsert_points(
            collection=request.collection,
            points=points,
            wait=request.wait,
        )

    upserted, queue_wait_ms = await _job_scheduler.submit(
        request_id=request_id,
        endpoint="upsert_points",
        route=route,
        factory=run_upsert,
    )
    latency_ms = int((time.monotonic() - started) * 1000)
    return UpsertResponse(
        request_id=request_id,
        queue=route.queue_name,
        collection=request.collection,
        upserted=upserted,
        latency_ms=latency_ms,
        queue_wait_ms=queue_wait_ms,
    )


@app.post("/agent/action")
async def agent_action(request: AgentActionRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    if request.action == "embed":
        return await embed(EmbedRequest.model_validate(request.payload), x_api_key)
    if request.action == "transform_embed":
        return await transform_embed(TransformEmbedRequest.model_validate(request.payload), x_api_key)
    if request.action == "search":
        return await search(SearchRequest.model_validate(request.payload), x_api_key)
    if request.action == "count":
        return await count(CountRequest.model_validate(request.payload), x_api_key)
    if request.action == "upsert_chunks":
        return await upsert_chunks(UpsertChunksRequest.model_validate(request.payload), x_api_key)
    if request.action == "upsert_points":
        return await upsert_points(UpsertPointsRequest.model_validate(request.payload), x_api_key)
    raise HTTPException(status_code=400, detail=f"Unsupported action: {request.action}")


async def _prepare_upsert_points(
    chunks: list[UpsertChunk],
    route,
    model_name: str | None,
    device: str | None,
    request_id: str,
) -> tuple[list[UpsertPoint], int]:
    texts = [chunk.text for chunk in chunks if chunk.vector is None and chunk.text is not None]
    embed_wait_ms = 0
    embedded_vectors: list[list[float]] = []
    if texts:
        model_ref = model_name or _default_model_ref(purpose="write")
        embed_result = await _embed_batcher.submit(
            request_id=request_id,
            texts=texts,
            route=route,
            model_name=model_ref,
            device=device,
        )
        embedded_vectors = embed_result.vectors
        embed_wait_ms = embed_result.queue_wait_ms

    embedded_index = 0
    points: list[UpsertPoint] = []
    for chunk in chunks:
        vector = chunk.vector
        if vector is None:
            vector = embedded_vectors[embedded_index]
            embedded_index += 1
        payload = dict(chunk.payload)
        if chunk.text is not None and "text" not in payload:
            payload["text"] = chunk.text
        points.append(
            UpsertPoint(
                id=chunk.id if chunk.id is not None else uuid4().hex,
                vector=vector,
                payload=payload,
            )
        )
    return points, embed_wait_ms


async def _queue_snapshots() -> list[QueueSnapshot]:
    batch_snapshot = _embed_batcher.snapshot()
    job_snapshot = _job_scheduler.snapshot()
    snapshots: list[QueueSnapshot] = []
    for queue_name, queue_cfg in _config.queues.items():
        pending_requests = batch_snapshot.get(queue_name, {}).get("pending_requests", 0) + job_snapshot.get(queue_name, 0)
        pending_texts = batch_snapshot.get(queue_name, {}).get("pending_texts", 0)
        snapshots.append(
            QueueSnapshot(
                queue=queue_name,
                pending_requests=pending_requests,
                pending_texts=pending_texts,
                max_batch_size=queue_cfg.max_batch_size,
                max_wait_ms=queue_cfg.max_wait_ms,
                preferred_device=queue_cfg.preferred_device,
            )
        )
    return snapshots


def _combined_queue_depths() -> dict[str, int]:
    batch_snapshot = _embed_batcher.snapshot()
    job_snapshot = _job_scheduler.snapshot()
    return {
        queue_name: batch_snapshot.get(queue_name, {}).get("pending_requests", 0) + job_snapshot.get(queue_name, 0)
        for queue_name in _config.queues
    }


def _request_id() -> str:
    return uuid4().hex[:12]


def _build_model_registry(config: GatewayConfig) -> dict[str, EmbeddingModelConfig]:
    if config.models:
        return dict(config.models)

    inferred_size = next(
        (collection.vector_size for collection in config.collections.values()),
        None,
    )
    return {
        "default": EmbeddingModelConfig(
            backend=config.embedding.backend,
            model_name=config.embedding.default_model,
            vector_size=inferred_size,
            distance="Cosine",
            normalize_embeddings=config.embedding.normalize_embeddings,
            device=config.embedding.device,
        )
    }


def _default_model_ref(collection_name: str | None = None, purpose: str = "query") -> str:
    if collection_name:
        collection = _config.collections.get(collection_name)
        if collection is not None:
            if purpose == "write" and collection.write_model:
                return collection.write_model
            if purpose == "query" and collection.query_model:
                return collection.query_model
            if collection.model:
                return collection.model
    if "default" in _model_registry:
        return "default"
    return next(iter(_model_registry.keys()), _config.embedding.default_model)


def _resolved_request_device(queue_name: str, request_device: str | None) -> str | None:
    if request_device:
        return request_device
    queue_cfg = _config.queues.get(queue_name)
    if queue_cfg and queue_cfg.preferred_device:
        return queue_cfg.preferred_device
    return None


def _model_infos() -> list[EmbeddingModelInfo]:
    return [
        EmbeddingModelInfo(
            name=name,
            backend=profile.backend,
            model_name=profile.model_name,
            vector_size=profile.vector_size,
            distance=profile.distance,
            normalize_embeddings=profile.normalize_embeddings,
            device=profile.device,
        )
        for name, profile in sorted(_model_registry.items())
    ]
