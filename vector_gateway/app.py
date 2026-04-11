"""FastAPI application for vector-db-gateway."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse

from vector_gateway.backends import LocalEmbeddingBackend, QdrantStore
from vector_gateway.config import CollectionConfig, EmbeddingModelConfig, GatewayConfig, load_config
from vector_gateway.core.batching import EmbeddingBatcher
from vector_gateway.core.do_mig import DBMigratorClient, DoMigRunner, WriteDiskQueueStore
from vector_gateway.core.logical_registry import LogicalCollectionRegistry
from vector_gateway.core.metrics import MetricsStore
from vector_gateway.core.router import Router
from vector_gateway.core.scheduler import FairSelector, JobScheduler
from vector_gateway.core.sparse import sparse_terms
from vector_gateway.core.state_store import MigrationStateStore
from vector_gateway.models import (
    AgentActionRequest,
    CapabilitiesResponse,
    CapabilityAction,
    CollectionInfo,
    CountRequest,
    CountResponse,
    DoMigQueueImportRequest,
    DoMigQueueItem,
    DoMigRunResponse,
    EmbedRequest,
    EmbedResponse,
    EmbeddingModelInfo,
    EnsureCollectionRequest,
    EnsureCollectionResponse,
    LogicalCollectionInfo,
    MigrationActionRequest,
    MigrationEvent,
    PayloadPatchRequest,
    PayloadSetRequest,
    PayloadUpdateResponse,
    QueueSnapshot,
    RetrieveRequest,
    RetrieveResponse,
    ScrollRequest,
    ScrollResponse,
    SearchRequest,
    SearchResponse,
    SparseVectorPayload,
    StatusResponse,
    TransformEmbedRequest,
    TransformEmbedResponse,
    TransformSparseRequest,
    TransformSparseResponse,
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
_state_store: MigrationStateStore
_logical_registry: LogicalCollectionRegistry
_do_mig_runner: DoMigRunner | None = None
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
    global _state_store, _logical_registry, _do_mig_runner

    _started_at = time.monotonic()
    _config = load_config()
    logging.getLogger().setLevel(getattr(logging, _config.log_level.upper(), logging.INFO))

    _router = Router.from_config(_config)
    _metrics = MetricsStore()
    _selector = FairSelector(_config.fairness)
    _model_registry = _build_model_registry(_config)
    _embed_backend = LocalEmbeddingBackend(_config.embedding, _model_registry)
    _qdrant = QdrantStore(_config.qdrant, _config.collections)
    _state_store = MigrationStateStore(_config.state_dir)
    _logical_registry = LogicalCollectionRegistry(_config, _state_store)
    _do_mig_runner = None
    if _config.do_mig.enabled and _config.write_disk and _config.db_migrator:
        _do_mig_runner = DoMigRunner(
            _config.do_mig,
            WriteDiskQueueStore(
                _config.write_disk,
                _config.do_mig.queue_channel,
                batch_limit=_config.do_mig.batch_limit,
            ),
            DBMigratorClient(_config.db_migrator),
            _logical_registry,
        )

    try:
        await _qdrant.ensure_collections()
        _logical_registry.bootstrap()
        await _sync_aliases()
    except Exception:
        logger.exception("Failed to bootstrap registered collections in Qdrant")
    try:
        warmed = await _embed_backend.warmup()
    except Exception:
        logger.exception("Failed to warm embedding runtime")
        raise
    if warmed:
        logger.info("Warmed embedding profiles: %s", warmed)
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
        "vector-db-gateway started port=%s queues=%s collections=%d logical_collections=%d rules=%d state_db=%s",
        _config.port,
        sorted(_config.queues.keys()),
        len(_config.collections),
        len(_config.logical_collections),
        _router.rule_count,
        _state_store.db_path,
    )
    yield
    await _embed_batcher.stop()
    await _job_scheduler.stop()
    logger.info("vector-db-gateway stopped")


app = FastAPI(title="vector-db-gateway", version="0.2.0", lifespan=lifespan)


def _check_api_key(x_api_key: str = Header(default="")) -> None:
    if x_api_key != _config.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _require_do_mig() -> DoMigRunner:
    if _do_mig_runner is None:
        raise HTTPException(status_code=503, detail="do-mig is not configured")
    return _do_mig_runner


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
    physical_infos = await _physical_info_map()
    qdrant_status = await _qdrant.health()
    status_label = "ok" if qdrant_status.get("status") == "ok" else "degraded"
    qdrant_status["migration_state_db"] = str(_state_store.db_path)
    return StatusResponse(
        status=status_label,
        uptime_s=int(time.monotonic() - _started_at),
        embedding_backend=_embed_backend.status(),
        qdrant=qdrant_status,
        queues=queue_states,
        collections=list(physical_infos.values()),
        logical_collections=_logical_registry.list_infos(physical_infos),
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
    return list((await _physical_info_map()).values())


@app.get("/collections/live", response_model=list[CollectionInfo])
async def live_collections(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    return await _qdrant.live_collection_infos()


@app.get("/collections/logical", response_model=list[LogicalCollectionInfo])
async def logical_collections(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    return _logical_registry.list_infos(await _physical_info_map())


@app.get("/collections/logical/{logical_name}", response_model=LogicalCollectionInfo)
async def logical_collection(logical_name: str, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    return await _logical_info(logical_name)


@app.get("/collections/logical/{logical_name}/migration/events", response_model=list[MigrationEvent])
async def logical_migration_events(logical_name: str, limit: int = 20, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    try:
        _logical_registry.get_logical_config(logical_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _logical_registry.list_events(logical_name, limit=max(1, min(limit, 200)))


@app.post("/collections/logical/{logical_name}/migration/{action}", response_model=LogicalCollectionInfo)
async def logical_migration_action(
    logical_name: str,
    action: str,
    request: MigrationActionRequest | None = None,
    x_api_key: str = Header(default=""),
):
    _check_api_key(x_api_key)
    request = request or MigrationActionRequest()
    try:
        _logical_registry.get_logical_config(logical_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        if action == "prepare":
            await _ensure_next_target(logical_name)
            _logical_registry.prepare(logical_name, task_id=request.task_id, note=request.note, metadata=request.metadata)
        elif action == "dual-write":
            await _ensure_next_target(logical_name)
            _logical_registry.dual_write(logical_name, task_id=request.task_id, note=request.note, metadata=request.metadata)
        elif action == "backfill":
            _logical_registry.backfill(logical_name, task_id=request.task_id, note=request.note, metadata=request.metadata)
        elif action == "verify":
            _logical_registry.mark_verify(
                logical_name,
                result=request.verify_result,
                note=request.note,
                metadata=request.metadata,
            )
        elif action == "cutover":
            _logical_registry.cutover(logical_name, note=request.note, metadata=request.metadata)
            await _sync_alias(logical_name)
        elif action == "rollback":
            _logical_registry.rollback(logical_name, note=request.note, metadata=request.metadata)
            await _sync_alias(logical_name)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported migration action: {action}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return await _logical_info(logical_name)


@app.get("/do-mig/queue/items", response_model=list[DoMigQueueItem])
async def do_mig_queue_items(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    return _require_do_mig().list_items()


@app.post("/do-mig/queue/import", response_model=list[DoMigQueueItem])
async def do_mig_queue_import(request: DoMigQueueImportRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    return _require_do_mig().import_items(request.items)


@app.post("/do-mig/queue/run", response_model=DoMigRunResponse)
async def do_mig_queue_run(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    now = datetime.now().astimezone()
    result = _require_do_mig().run_once(now)
    return DoMigRunResponse(action=result.action, now=now.isoformat(), items=result.items)


@app.post("/collections/ensure", response_model=EnsureCollectionResponse)
async def ensure_collection(request: EnsureCollectionRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    created, info = await _qdrant.ensure_collection(
        collection=request.collection,
        meta=CollectionConfig(
            vector_size=request.vector_size,
            distance=request.distance,
            owner=request.owner,
            vector_name=request.vector_name,
            sparse_vector_name=request.sparse_vector_name,
            sparse_modifier=request.sparse_modifier,
            model=request.model,
            query_model=request.query_model,
            write_model=request.write_model,
            aliases=request.aliases,
            description=request.description,
        ),
    )
    return EnsureCollectionResponse(created=created, collection=info)


@app.get("/models", response_model=list[EmbeddingModelInfo])
async def models(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    return _model_infos()


@app.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    physical_infos = await _physical_info_map()
    return CapabilitiesResponse(
        service="vector-db-gateway",
        version="0.2.0",
        actions=[
            CapabilityAction(name="embed", endpoint="/embed", description="Generate dense embeddings"),
            CapabilityAction(name="transform_embed", endpoint="/transform/embed", description="Migration-safe dense embedding callback"),
            CapabilityAction(name="transform_bm25_sparse", endpoint="/transform/bm25_sparse", description="Generate sparse lexical vectors"),
            CapabilityAction(name="search", endpoint="/search", description="Search a physical or logical collection"),
            CapabilityAction(name="retrieve", endpoint="/retrieve", description="Retrieve points by id"),
            CapabilityAction(name="scroll", endpoint="/scroll", description="Scroll records from a collection"),
            CapabilityAction(name="live_collections", endpoint="/collections/live", description="List live collections discovered from Qdrant"),
            CapabilityAction(name="logical_collections", endpoint="/collections/logical", description="List logical collection routes and migration state"),
            CapabilityAction(name="do_mig_queue", endpoint="/do-mig/queue/items", description="Inspect queued migration slices stored in write-disk"),
            CapabilityAction(name="do_mig_run", endpoint="/do-mig/queue/run", description="Advance the queued migration runner by one step"),
            CapabilityAction(name="count", endpoint="/count", description="Count records in a collection"),
            CapabilityAction(name="ensure_collection", endpoint="/collections/ensure", description="Create or validate a collection for vector writes"),
            CapabilityAction(name="set_payload", endpoint="/payload/set", description="Set payload fields for many points"),
            CapabilityAction(name="patch_payload", endpoint="/payload/patch", description="Patch payload fields for a point"),
            CapabilityAction(name="upsert_chunks", endpoint="/upsert/chunks", description="Upsert chunks with optional embedding"),
            CapabilityAction(name="upsert_points", endpoint="/upsert/points", description="Upsert points with supplied vectors"),
            CapabilityAction(name="agent_action", endpoint="/agent/action", description="Single action endpoint for machine callers"),
        ],
        queues=sorted(_config.queues.keys()),
        models=_model_infos(),
        collections=list(physical_infos.values()),
        logical_collections=_logical_registry.list_infos(physical_infos),
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


@app.post("/transform/bm25_sparse", response_model=TransformSparseResponse)
async def transform_bm25_sparse(request: TransformSparseRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    return TransformSparseResponse(
        vectors=[
            SparseVectorPayload(indices=indices, values=values)
            for indices, values in (sparse_terms(text) for text in request.texts)
        ]
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    request_id = _request_id()
    route = _router.resolve(request.caller, request.operation)
    device = _resolved_request_device(route.queue_name, request.device)
    started = time.monotonic()
    physical_collection = _read_collection(request.collection)
    query_mode = _resolved_query_mode(request.collection, physical_collection, request.search_mode)
    vector = request.vector
    queue_wait_ms = 0
    _validate_dense_request_vector(physical_collection, vector)
    if vector is None and query_mode != "sparse":
        model_ref = request.model or _default_model_ref(request.collection, purpose="query")
        _validate_model_for_collection(physical_collection, model_ref, purpose="query")
        embed_result = await _embed_batcher.submit(
            request_id=request_id,
            texts=[request.text or request.query_text or ""],
            route=route,
            model_name=model_ref,
            device=device,
        )
        vector = embed_result.vectors[0]
        queue_wait_ms += embed_result.queue_wait_ms
    sparse_vector = _search_sparse_vector(
        request.collection,
        physical_collection,
        query_text=request.query_text or request.text,
        requested_mode=query_mode,
    )
    if query_mode == "hybrid" and sparse_vector is None:
        query_mode = "dense"
    if query_mode == "hybrid" and vector is None:
        query_mode = "sparse"

    async def run_search():
        return await _qdrant.search(
            collection=physical_collection,
            dense_vector=vector,
            sparse_vector=sparse_vector,
            query_mode=query_mode,
            limit=request.limit,
            filter_spec=request.filter,
            with_payload=request.with_payload,
            with_vectors=request.with_vectors,
        )

    try:
        hits, scheduled_wait_ms = await _job_scheduler.submit(
            request_id=request_id,
            endpoint="search",
            route=route,
            factory=run_search,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    total_wait_ms = queue_wait_ms + scheduled_wait_ms
    latency_ms = int((time.monotonic() - started) * 1000)
    _metrics.observe_request("search", latency_ms=latency_ms, queue_wait_ms=total_wait_ms)
    return SearchResponse(
        request_id=request_id,
        queue=route.queue_name,
        collection=request.collection,
        hits=hits,
        latency_ms=latency_ms,
        queue_wait_ms=total_wait_ms,
    )


@app.post("/scroll", response_model=ScrollResponse)
async def scroll(request: ScrollRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    request_id = _request_id()
    started = time.monotonic()
    route = _router.resolve(request.caller, request.operation)
    physical_collection = _read_collection(request.collection)

    async def run_scroll():
        return await _qdrant.scroll(
            collection=physical_collection,
            filter_spec=request.filter,
            limit=request.limit,
            with_payload=request.with_payload,
            with_vectors=request.with_vectors,
        )

    try:
        points, queue_wait_ms = await _job_scheduler.submit(
            request_id=request_id,
            endpoint="scroll",
            route=route,
            factory=run_scroll,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = int((time.monotonic() - started) * 1000)
    _metrics.observe_request("scroll", latency_ms=latency_ms, queue_wait_ms=queue_wait_ms)
    return ScrollResponse(
        request_id=request_id,
        queue=route.queue_name,
        collection=request.collection,
        points=points,
        latency_ms=latency_ms,
        queue_wait_ms=queue_wait_ms,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    request_id = _request_id()
    started = time.monotonic()
    route = _router.resolve(request.caller, request.operation)
    physical_collection = _read_collection(request.collection)

    async def run_retrieve():
        return await _qdrant.retrieve(
            collection=physical_collection,
            ids=request.ids,
            with_payload=request.with_payload,
            with_vectors=request.with_vectors,
        )

    try:
        points, queue_wait_ms = await _job_scheduler.submit(
            request_id=request_id,
            endpoint="retrieve",
            route=route,
            factory=run_retrieve,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = int((time.monotonic() - started) * 1000)
    _metrics.observe_request("retrieve", latency_ms=latency_ms, queue_wait_ms=queue_wait_ms)
    return RetrieveResponse(
        request_id=request_id,
        queue=route.queue_name,
        collection=request.collection,
        points=points,
        latency_ms=latency_ms,
        queue_wait_ms=queue_wait_ms,
    )


@app.post("/count", response_model=CountResponse)
async def count(request: CountRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    request_id = _request_id()
    started = time.monotonic()
    route = _router.resolve(request.caller, request.operation)
    physical_collection = _read_collection(request.collection)

    async def run_count():
        return await _qdrant.count(collection=physical_collection, filter_spec=request.filter)

    try:
        total, queue_wait_ms = await _job_scheduler.submit(
            request_id=request_id,
            endpoint="count",
            route=route,
            factory=run_count,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = int((time.monotonic() - started) * 1000)
    _metrics.observe_request("count", latency_ms=latency_ms, queue_wait_ms=queue_wait_ms)
    return CountResponse(
        request_id=request_id,
        queue=route.queue_name,
        collection=request.collection,
        count=total,
        latency_ms=latency_ms,
        queue_wait_ms=queue_wait_ms,
    )


@app.post("/payload/set", response_model=PayloadUpdateResponse)
async def set_payload(request: PayloadSetRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    request_id = _request_id()
    started = time.monotonic()
    route = _router.resolve(request.caller, request.operation)
    write_targets = _write_targets(request.collection)

    async def run_set_payload():
        updated = 0
        for target in write_targets:
            updated = await _qdrant.set_payload(
                collection=target,
                ids=request.ids,
                payload=request.payload,
                wait=request.wait,
            )
        return updated

    try:
        updated, queue_wait_ms = await _job_scheduler.submit(
            request_id=request_id,
            endpoint="set_payload",
            route=route,
            factory=run_set_payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = int((time.monotonic() - started) * 1000)
    _metrics.observe_request("set_payload", latency_ms=latency_ms, queue_wait_ms=queue_wait_ms)
    return PayloadUpdateResponse(
        request_id=request_id,
        queue=route.queue_name,
        collection=request.collection,
        updated=updated,
        latency_ms=latency_ms,
        queue_wait_ms=queue_wait_ms,
    )


@app.post("/payload/patch", response_model=PayloadUpdateResponse)
async def patch_payload(request: PayloadPatchRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    request_id = _request_id()
    started = time.monotonic()
    route = _router.resolve(request.caller, request.operation)
    write_targets = _write_targets(request.collection)

    async def run_patch_payload():
        updated = 0
        for target in write_targets:
            updated = await _qdrant.patch_payload(
                collection=target,
                point_id=request.id,
                payload=request.payload,
                wait=request.wait,
            )
        return updated

    try:
        updated, queue_wait_ms = await _job_scheduler.submit(
            request_id=request_id,
            endpoint="patch_payload",
            route=route,
            factory=run_patch_payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = int((time.monotonic() - started) * 1000)
    _metrics.observe_request("patch_payload", latency_ms=latency_ms, queue_wait_ms=queue_wait_ms)
    return PayloadUpdateResponse(
        request_id=request_id,
        queue=route.queue_name,
        collection=request.collection,
        updated=updated,
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
        request.collection,
        request.chunks,
        route,
        request.model,
        device,
        request_id,
    )
    write_targets = _write_targets(request.collection)
    _validate_upsert_points_for_targets(write_targets, points)

    async def run_upsert():
        upserted = 0
        for target in write_targets:
            upserted = await _qdrant.upsert_points(
                collection=target,
                points=points,
                wait=request.wait,
            )
        return upserted

    try:
        upserted, scheduled_wait_ms = await _job_scheduler.submit(
            request_id=request_id,
            endpoint="upsert_chunks",
            route=route,
            factory=run_upsert,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = int((time.monotonic() - started) * 1000)
    total_wait_ms = embed_wait_ms + scheduled_wait_ms
    _metrics.observe_request("upsert_chunks", latency_ms=latency_ms, queue_wait_ms=total_wait_ms)
    return UpsertResponse(
        request_id=request_id,
        queue=route.queue_name,
        collection=request.collection,
        upserted=upserted,
        latency_ms=latency_ms,
        queue_wait_ms=total_wait_ms,
    )


@app.post("/upsert/points", response_model=UpsertResponse)
async def upsert_points(request: UpsertPointsRequest, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    request_id = _request_id()
    started = time.monotonic()
    route = _router.resolve(request.caller, request.operation)
    points = [UpsertPoint(id=point.id, vector=point.vector, payload=point.payload) for point in request.points]
    write_targets = _write_targets(request.collection)
    _validate_upsert_points_for_targets(write_targets, points)

    async def run_upsert():
        upserted = 0
        for target in write_targets:
            upserted = await _qdrant.upsert_points(
                collection=target,
                points=points,
                wait=request.wait,
            )
        return upserted

    try:
        upserted, queue_wait_ms = await _job_scheduler.submit(
            request_id=request_id,
            endpoint="upsert_points",
            route=route,
            factory=run_upsert,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = int((time.monotonic() - started) * 1000)
    _metrics.observe_request("upsert_points", latency_ms=latency_ms, queue_wait_ms=queue_wait_ms)
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
    if request.action == "transform_bm25_sparse":
        return await transform_bm25_sparse(TransformSparseRequest.model_validate(request.payload), x_api_key)
    if request.action == "search":
        return await search(SearchRequest.model_validate(request.payload), x_api_key)
    if request.action == "retrieve":
        return await retrieve(RetrieveRequest.model_validate(request.payload), x_api_key)
    if request.action == "scroll":
        return await scroll(ScrollRequest.model_validate(request.payload), x_api_key)
    if request.action == "count":
        return await count(CountRequest.model_validate(request.payload), x_api_key)
    if request.action == "ensure_collection":
        return await ensure_collection(EnsureCollectionRequest.model_validate(request.payload), x_api_key)
    if request.action == "set_payload":
        return await set_payload(PayloadSetRequest.model_validate(request.payload), x_api_key)
    if request.action == "patch_payload":
        return await patch_payload(PayloadPatchRequest.model_validate(request.payload), x_api_key)
    if request.action == "upsert_chunks":
        return await upsert_chunks(UpsertChunksRequest.model_validate(request.payload), x_api_key)
    if request.action == "upsert_points":
        return await upsert_points(UpsertPointsRequest.model_validate(request.payload), x_api_key)
    raise HTTPException(status_code=400, detail=f"Unsupported action: {request.action}")


async def _prepare_upsert_points(
    collection_name: str,
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
        model_ref = model_name or _default_model_ref(collection_name, purpose="write")
        for target in _write_targets(collection_name):
            _validate_model_for_collection(target, model_ref, purpose="write")
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
        model_name = _logical_registry.model_for(collection_name, purpose=purpose)
        if model_name:
            return model_name
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


def _read_collection(collection_name: str) -> str:
    try:
        return _logical_registry.read_target(collection_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _write_targets(collection_name: str) -> list[str]:
    try:
        return _logical_registry.write_targets(collection_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolved_query_mode(requested_collection: str, physical_collection: str, requested_mode: str) -> str:
    mode = (requested_mode or "auto").lower()
    if mode != "auto":
        return mode
    if _logical_registry.is_logical(requested_collection):
        logical = _logical_registry.get_logical_config(requested_collection)
        return logical.default_query_mode
    meta = _config.collections.get(physical_collection)
    if meta and meta.sparse_vector_name:
        return "hybrid"
    return "dense"


def _search_sparse_vector(
    requested_collection: str,
    physical_collection: str,
    *,
    query_text: str | None,
    requested_mode: str,
) -> dict[str, list[int] | list[float]] | None:
    mode = (requested_mode or "dense").lower()
    if mode not in {"hybrid", "sparse"}:
        return None
    meta = _config.collections.get(physical_collection)
    if meta is None or not meta.sparse_vector_name:
        return None
    if not query_text or not query_text.strip():
        return None
    indices, values = sparse_terms(query_text)
    if not indices:
        return None
    return {"indices": indices, "values": values}


def _collection_dense_vector_size(collection_name: str) -> int | None:
    meta = _config.collections.get(collection_name)
    if meta is None:
        return None
    return meta.vector_size


def _validate_dense_request_vector(collection_name: str, vector: list[float] | None) -> None:
    if vector is None:
        return
    expected_size = _collection_dense_vector_size(collection_name)
    if expected_size is None:
        return
    actual_size = len(vector)
    if actual_size != expected_size:
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{collection_name}' expects dense vector size {expected_size}, got {actual_size}",
        )


def _validate_model_for_collection(collection_name: str, model_ref: str, *, purpose: str) -> None:
    expected_size = _collection_dense_vector_size(collection_name)
    if expected_size is None:
        return
    profile = _model_registry.get(model_ref)
    if profile is None or profile.vector_size is None:
        return
    if profile.vector_size != expected_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_ref}' produces {profile.vector_size}-dim vectors "
                f"but collection '{collection_name}' expects {expected_size} for {purpose}"
            ),
        )


def _validate_upsert_points_for_targets(targets: list[str], points: list[UpsertPoint]) -> None:
    for target in targets:
        for point in points:
            dense_vector = _extract_dense_request_vector(target, point.vector)
            if dense_vector is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Collection '{target}' requires a dense vector payload for upsert",
                )
            _validate_dense_request_vector(target, dense_vector)


def _extract_dense_request_vector(
    collection_name: str,
    vector: list[float] | dict[str, object],
) -> list[float] | None:
    if isinstance(vector, list):
        return vector
    meta = _config.collections.get(collection_name)
    if meta is None:
        return None
    if meta.vector_name:
        candidate = vector.get(meta.vector_name)
        return candidate if isinstance(candidate, list) else None
    if meta.sparse_vector_name:
        for key, value in vector.items():
            if key == meta.sparse_vector_name:
                continue
            if isinstance(value, list):
                return value
        return None
    if len(vector) == 1:
        candidate = next(iter(vector.values()))
        return candidate if isinstance(candidate, list) else None
    return None


async def _physical_info_map() -> dict[str, CollectionInfo]:
    infos = await _qdrant.collection_infos()
    return {info.name: info for info in infos}


async def _logical_info(logical_name: str) -> LogicalCollectionInfo:
    try:
        return _logical_registry.get_info(logical_name, await _physical_info_map())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _ensure_next_target(logical_name: str) -> None:
    next_target = _logical_registry.ensure_target(logical_name)
    meta = _config.collections.get(next_target)
    if meta is None:
        raise HTTPException(status_code=400, detail=f"Missing configured physical collection: {next_target}")
    await _qdrant.ensure_collection(collection=next_target, meta=meta)


async def _sync_aliases() -> None:
    for logical_name in _config.logical_collections:
        await _sync_alias(logical_name)


async def _sync_alias(logical_name: str) -> None:
    logical = _logical_registry.get_logical_config(logical_name)
    alias_name = logical.alias_name or logical_name
    if not alias_name:
        return
    target = _logical_registry.read_target(logical_name)
    try:
        await _qdrant.ensure_alias(alias_name, target)
    except Exception:
        logger.exception("Failed to sync alias logical_name=%s alias=%s target=%s", logical_name, alias_name, target)
