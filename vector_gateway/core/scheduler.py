"""Fairness-aware async scheduler for non-batched jobs."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from vector_gateway.config import FairnessConfig
from vector_gateway.core.metrics import MetricsStore
from vector_gateway.core.router import RouteDecision

logger = logging.getLogger(__name__)


@dataclass
class SchedulerCandidate:
    queue_name: str
    service_priority: int
    operation_priority: int
    enqueued_at: float
    item: Any = field(compare=False)


class FairSelector:
    """Select the next queue candidate while protecting low-priority work from starvation."""

    def __init__(self, config: FairnessConfig):
        self._aging_step_ms = max(1, config.aging_step_ms)
        self._max_consecutive_realtime = max(1, config.max_consecutive_realtime_batches)
        self._reserve_batch_share = max(0.0, config.reserve_batch_share)
        self._consecutive_realtime = 0
        self._since_batch = 0

    def pick(self, candidates: list[SchedulerCandidate], now: float | None = None) -> SchedulerCandidate:
        if not candidates:
            raise ValueError("No scheduler candidates")
        now = time.monotonic() if now is None else now

        batch_candidates = [c for c in candidates if c.queue_name == "batch"]
        if batch_candidates and self._should_reserve_batch():
            chosen = self._best_candidate(batch_candidates, now)
            self._mark_served(chosen.queue_name)
            return chosen

        non_realtime = [c for c in candidates if c.queue_name != "realtime"]
        if non_realtime and self._consecutive_realtime >= self._max_consecutive_realtime:
            chosen = self._best_candidate(non_realtime, now)
            self._mark_served(chosen.queue_name)
            return chosen

        chosen = self._best_candidate(candidates, now)
        self._mark_served(chosen.queue_name)
        return chosen

    def _should_reserve_batch(self) -> bool:
        if self._reserve_batch_share <= 0:
            return False
        threshold = max(1, math.ceil(1.0 / self._reserve_batch_share) - 1)
        return self._since_batch >= threshold

    def _best_candidate(self, candidates: list[SchedulerCandidate], now: float) -> SchedulerCandidate:
        return min(
            candidates,
            key=lambda candidate: (
                self._effective_service_priority(candidate, now),
                candidate.operation_priority,
                candidate.enqueued_at,
            ),
        )

    def _effective_service_priority(self, candidate: SchedulerCandidate, now: float) -> int:
        waited_ms = max(0, int((now - candidate.enqueued_at) * 1000))
        boost = waited_ms // self._aging_step_ms
        return max(0, candidate.service_priority - boost)

    def _mark_served(self, queue_name: str) -> None:
        if queue_name == "realtime":
            self._consecutive_realtime += 1
        else:
            self._consecutive_realtime = 0

        if queue_name == "batch":
            self._since_batch = 0
        else:
            self._since_batch += 1


@dataclass
class ScheduledJob:
    request_id: str
    endpoint: str
    route: RouteDecision
    factory: Callable[[], Awaitable[Any]]
    future: asyncio.Future
    created_at: float
    submitted_at: float


class JobScheduler:
    """Fair scheduler for search, count, scroll, and upsert operations."""

    def __init__(self, selector: FairSelector, metrics: MetricsStore, worker_count: int = 1):
        self._selector = selector
        self._metrics = metrics
        self._worker_count = max(1, int(worker_count))
        self._queues: dict[str, list[ScheduledJob]] = {
            "realtime": [],
            "interactive": [],
            "batch": [],
        }
        self._condition = asyncio.Condition()
        self._workers: list[asyncio.Task] = []
        self._closed = False

    async def start(self) -> None:
        if not self._workers:
            self._workers = [
                asyncio.create_task(self._run(), name=f"vector-job-scheduler-{index + 1}")
                for index in range(self._worker_count)
            ]

    async def stop(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
        for worker in self._workers:
            await worker

    async def submit(
        self,
        *,
        request_id: str,
        endpoint: str,
        route: RouteDecision,
        factory: Callable[[], Awaitable[Any]],
    ) -> tuple[Any, int]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        job = ScheduledJob(
            request_id=request_id,
            endpoint=endpoint,
            route=route,
            factory=factory,
            future=future,
            created_at=time.monotonic(),
            submitted_at=time.monotonic(),
        )
        async with self._condition:
            self._queues[route.queue_name].append(job)
            self._condition.notify_all()
        result = await future
        return result

    def snapshot(self) -> dict[str, int]:
        return {queue_name: len(jobs) for queue_name, jobs in self._queues.items()}

    async def _run(self) -> None:
        while True:
            async with self._condition:
                while not self._closed and not any(self._queues.values()):
                    await self._condition.wait()
                if self._closed and not any(self._queues.values()):
                    return

                chosen = self._pick_next()
                queue = self._queues[chosen.route.queue_name]
                queue.remove(chosen)

            queue_wait_ms = int((time.monotonic() - chosen.submitted_at) * 1000)
            request_start = time.monotonic()
            try:
                result = await chosen.factory()
            except Exception as exc:  # pragma: no cover - exercised by integration
                self._metrics.observe_request(
                    f"scheduler.{chosen.endpoint}",
                    latency_ms=int((time.monotonic() - request_start) * 1000),
                    queue_wait_ms=queue_wait_ms,
                    failed=True,
                )
                if not chosen.future.done():
                    chosen.future.set_exception(exc)
                logger.exception("Scheduled job failed: %s", chosen.request_id)
            else:
                self._metrics.observe_request(
                    f"scheduler.{chosen.endpoint}",
                    latency_ms=int((time.monotonic() - request_start) * 1000),
                    queue_wait_ms=queue_wait_ms,
                )
                if not chosen.future.done():
                    chosen.future.set_result((result, queue_wait_ms))

    def _pick_next(self) -> ScheduledJob:
        now = time.monotonic()
        candidates: list[SchedulerCandidate] = []
        for queue_name, jobs in self._queues.items():
            if not jobs:
                continue
            selected = min(
                jobs,
                key=lambda job: (
                    self._effective_service_priority(job, now),
                    job.route.operation_priority,
                    job.created_at,
                ),
            )
            candidates.append(
                SchedulerCandidate(
                    queue_name=queue_name,
                    service_priority=selected.route.service_priority,
                    operation_priority=selected.route.operation_priority,
                    enqueued_at=selected.created_at,
                    item=selected,
                )
            )
        return self._selector.pick(candidates, now=now).item

    def _effective_service_priority(self, job: ScheduledJob, now: float) -> int:
        waited_ms = max(0, int((now - job.created_at) * 1000))
        boost = waited_ms // self._selector._aging_step_ms
        return max(0, job.route.service_priority - boost)
