"""In-memory metrics store with Prometheus-style rendering."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


class MetricsStore:
    """Track endpoint and scheduler metrics without affecting request success."""

    def __init__(self):
        self._lock = threading.Lock()
        self._requests = defaultdict(int)
        self._failures = defaultdict(int)
        self._latency_ms = defaultdict(int)
        self._queue_wait_ms = defaultdict(int)
        self._batches = defaultdict(int)
        self._batch_items = defaultdict(int)

    def observe_request(
        self,
        endpoint: str,
        *,
        latency_ms: int,
        queue_wait_ms: int = 0,
        failed: bool = False,
    ) -> None:
        with self._lock:
            self._requests[endpoint] += 1
            self._latency_ms[endpoint] += latency_ms
            self._queue_wait_ms[endpoint] += queue_wait_ms
            if failed:
                self._failures[endpoint] += 1

    def observe_batch(self, queue_name: str, *, batch_count: int, item_count: int) -> None:
        with self._lock:
            self._batches[queue_name] += batch_count
            self._batch_items[queue_name] += item_count

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            endpoints: dict[str, Any] = {}
            for endpoint, requests in self._requests.items():
                avg_latency = self._latency_ms[endpoint] / requests if requests else 0
                avg_queue_wait = self._queue_wait_ms[endpoint] / requests if requests else 0
                endpoints[endpoint] = {
                    "requests": requests,
                    "failures": self._failures[endpoint],
                    "avg_latency_ms": round(avg_latency, 2),
                    "avg_queue_wait_ms": round(avg_queue_wait, 2),
                }
            batches = {
                queue: {
                    "batches": self._batches[queue],
                    "items": self._batch_items[queue],
                }
                for queue in set(self._batches) | set(self._batch_items)
            }
            return {"endpoints": endpoints, "batches": batches}

    def render_prometheus(self, queue_depths: dict[str, int]) -> str:
        with self._lock:
            lines = [
                "# HELP vector_gateway_requests_total Total HTTP requests by endpoint",
                "# TYPE vector_gateway_requests_total counter",
            ]
            for endpoint in sorted(self._requests):
                lines.append(
                    f'vector_gateway_requests_total{{endpoint="{endpoint}"}} {self._requests[endpoint]}'
                )
            lines.extend(
                [
                    "# HELP vector_gateway_failures_total Total failed requests by endpoint",
                    "# TYPE vector_gateway_failures_total counter",
                ]
            )
            for endpoint in sorted(self._requests):
                lines.append(
                    f'vector_gateway_failures_total{{endpoint="{endpoint}"}} {self._failures[endpoint]}'
                )
            lines.extend(
                [
                    "# HELP vector_gateway_batches_total Total emitted embedding batches by queue",
                    "# TYPE vector_gateway_batches_total counter",
                ]
            )
            for queue_name in sorted(self._batches):
                lines.append(
                    f'vector_gateway_batches_total{{queue="{queue_name}"}} {self._batches[queue_name]}'
                )
            lines.extend(
                [
                    "# HELP vector_gateway_batch_items_total Total embedded text items by queue",
                    "# TYPE vector_gateway_batch_items_total counter",
                ]
            )
            for queue_name in sorted(self._batch_items):
                lines.append(
                    f'vector_gateway_batch_items_total{{queue="{queue_name}"}} {self._batch_items[queue_name]}'
                )
            lines.extend(
                [
                    "# HELP vector_gateway_queue_depth Current queue depth by queue",
                    "# TYPE vector_gateway_queue_depth gauge",
                ]
            )
            for queue_name in sorted(queue_depths):
                lines.append(
                    f'vector_gateway_queue_depth{{queue="{queue_name}"}} {queue_depths[queue_name]}'
                )
            return "\n".join(lines) + "\n"
