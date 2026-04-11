"""Queue-aware embedding micro-batching."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from vector_gateway.config import QueueConfig
from vector_gateway.core.metrics import MetricsStore
from vector_gateway.core.router import RouteDecision
from vector_gateway.core.scheduler import FairSelector, SchedulerCandidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchKey:
    queue_name: str
    model_name: str
    device: str | None
    service_priority: int
    operation_priority: int


@dataclass
class EmbedTask:
    request_id: str
    texts: list[str]
    route: RouteDecision
    model_name: str
    device: str | None
    created_at: float
    future: asyncio.Future


@dataclass
class EmbedBatchResult:
    vectors: list[list[float]]
    queue_wait_ms: int
    batch_size: int


class EmbeddingBatcher:
    """Collect compatible embedding requests and flush them as micro-batches."""

    def __init__(
        self,
        *,
        backend,
        queue_config: dict[str, QueueConfig],
        selector: FairSelector,
        metrics: MetricsStore,
    ):
        self._backend = backend
        self._queue_config = queue_config
        self._selector = selector
        self._metrics = metrics
        self._buffers: dict[BatchKey, list[EmbedTask]] = {}
        self._condition = asyncio.Condition()
        self._worker: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="vector-embedding-batcher")

    async def stop(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._worker is not None:
            await self._worker

    async def submit(
        self,
        *,
        request_id: str,
        texts: list[str],
        route: RouteDecision,
        model_name: str,
        device: str | None = None,
    ) -> EmbedBatchResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        task = EmbedTask(
            request_id=request_id,
            texts=texts,
            route=route,
            model_name=model_name,
            device=device,
            created_at=time.monotonic(),
            future=future,
        )
        key = BatchKey(
            queue_name=route.queue_name,
            model_name=model_name,
            device=device,
            service_priority=route.service_priority,
            operation_priority=route.operation_priority,
        )
        async with self._condition:
            self._buffers.setdefault(key, []).append(task)
            self._condition.notify_all()
        return await future

    def snapshot(self) -> dict[str, dict[str, int]]:
        per_queue = {
            queue_name: {"pending_requests": 0, "pending_texts": 0}
            for queue_name in self._queue_config
        }
        for key, tasks in self._buffers.items():
            per_queue[key.queue_name]["pending_requests"] += len(tasks)
            per_queue[key.queue_name]["pending_texts"] += sum(len(task.texts) for task in tasks)
        return per_queue

    async def _run(self) -> None:
        while True:
            batch_key: BatchKey | None = None
            batch_tasks: list[EmbedTask] = []
            async with self._condition:
                while not self._closed:
                    ready_keys = self._ready_keys()
                    if ready_keys:
                        batch_key = self._pick_ready_key(ready_keys)
                        batch_tasks = self._take_batch(batch_key)
                        break

                    timeout = self._next_timeout()
                    if timeout is None:
                        await self._condition.wait()
                    else:
                        try:
                            await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                        except TimeoutError:
                            pass

                if self._closed and not self._buffers:
                    return
                if batch_key is None or not batch_tasks:
                    continue

            queue_wait_ms = int(
                (time.monotonic() - min(task.created_at for task in batch_tasks)) * 1000
            )
            flat_texts = [text for task in batch_tasks for text in task.texts]
            try:
                vectors = await self._backend.embed_texts(
                    flat_texts,
                    batch_key.model_name,
                    batch_key.device,
                )
            except Exception as exc:  # pragma: no cover - exercised by integration
                for task in batch_tasks:
                    if not task.future.done():
                        task.future.set_exception(exc)
                logger.exception("Embedding batch failed: queue=%s", batch_key.queue_name)
                continue

            self._metrics.observe_batch(
                batch_key.queue_name,
                batch_count=1,
                item_count=len(flat_texts),
            )
            offset = 0
            for task in batch_tasks:
                size = len(task.texts)
                task_vectors = vectors[offset : offset + size]
                offset += size
                if not task.future.done():
                    task.future.set_result(
                        EmbedBatchResult(
                            vectors=task_vectors,
                            queue_wait_ms=queue_wait_ms,
                            batch_size=len(flat_texts),
                        )
                    )

    def _ready_keys(self) -> list[BatchKey]:
        now = time.monotonic()
        ready: list[BatchKey] = []
        for key, tasks in self._buffers.items():
            if not tasks:
                continue
            queue_cfg = self._queue_config[key.queue_name]
            total_texts = sum(len(task.texts) for task in tasks)
            oldest = min(task.created_at for task in tasks)
            waited_ms = int((now - oldest) * 1000)
            if total_texts >= queue_cfg.max_batch_size or waited_ms >= queue_cfg.max_wait_ms:
                ready.append(key)
        return ready

    def _next_timeout(self) -> float | None:
        if not self._buffers:
            return None
        now = time.monotonic()
        deadlines: list[float] = []
        for key, tasks in self._buffers.items():
            if not tasks:
                continue
            oldest = min(task.created_at for task in tasks)
            queue_cfg = self._queue_config[key.queue_name]
            deadline = oldest + (queue_cfg.max_wait_ms / 1000.0)
            deadlines.append(max(0.0, deadline - now))
        return min(deadlines) if deadlines else None

    def _pick_ready_key(self, ready_keys: list[BatchKey]) -> BatchKey:
        now = time.monotonic()
        candidates = []
        for key in ready_keys:
            oldest = min(task.created_at for task in self._buffers[key])
            candidates.append(
                SchedulerCandidate(
                    queue_name=key.queue_name,
                    service_priority=key.service_priority,
                    operation_priority=key.operation_priority,
                    enqueued_at=oldest,
                    item=key,
                )
            )
        return self._selector.pick(candidates, now=now).item

    def _take_batch(self, key: BatchKey) -> list[EmbedTask]:
        tasks = self._buffers.get(key, [])
        if not tasks:
            return []
        queue_cfg = self._queue_config[key.queue_name]
        selected: list[EmbedTask] = []
        total_texts = 0
        while tasks:
            candidate = tasks[0]
            candidate_texts = len(candidate.texts)
            if selected and total_texts + candidate_texts > queue_cfg.max_batch_size:
                break
            selected.append(tasks.pop(0))
            total_texts += candidate_texts
            if total_texts >= queue_cfg.max_batch_size:
                break
        if not tasks:
            self._buffers.pop(key, None)
        return selected
