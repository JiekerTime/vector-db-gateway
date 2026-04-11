from __future__ import annotations

import asyncio
import unittest

from vector_gateway.config import QueueConfig
from vector_gateway.core.batching import EmbeddingBatcher
from vector_gateway.core.metrics import MetricsStore
from vector_gateway.core.router import RouteDecision
from vector_gateway.core.scheduler import FairSelector
from vector_gateway.config import FairnessConfig


class FakeEmbeddingBackend:
    def __init__(self):
        self.calls: list[tuple[list[str], str]] = []

    async def embed_texts(self, texts: list[str], model_name: str):
        self.calls.append((texts, model_name))
        await asyncio.sleep(0)
        return [[float(len(text))] for text in texts]


class EmbeddingBatcherTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.backend = FakeEmbeddingBackend()
        self.batcher = EmbeddingBatcher(
            backend=self.backend,
            queue_config={
                "realtime": QueueConfig(max_batch_size=8, max_wait_ms=20, max_concurrent_jobs=1),
                "interactive": QueueConfig(max_batch_size=8, max_wait_ms=20, max_concurrent_jobs=1),
                "batch": QueueConfig(max_batch_size=8, max_wait_ms=20, max_concurrent_jobs=1),
            },
            selector=FairSelector(FairnessConfig()),
            metrics=MetricsStore(),
        )
        await self.batcher.start()

    async def asyncTearDown(self) -> None:
        await self.batcher.stop()

    async def test_batches_compatible_requests_together(self) -> None:
        route = RouteDecision(
            queue_name="interactive",
            service_priority=1,
            operation="search",
            operation_priority=1,
        )
        first = asyncio.create_task(
            self.batcher.submit(
                request_id="r1",
                texts=["alpha"],
                route=route,
                model_name="model-a",
            )
        )
        second = asyncio.create_task(
            self.batcher.submit(
                request_id="r2",
                texts=["beta"],
                route=route,
                model_name="model-a",
            )
        )
        first_result, second_result = await asyncio.gather(first, second)

        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.backend.calls[0][0], ["alpha", "beta"])
        self.assertEqual(first_result.vectors, [[5.0]])
        self.assertEqual(second_result.vectors, [[4.0]])


if __name__ == "__main__":
    unittest.main()
