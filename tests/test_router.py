from __future__ import annotations

import unittest

from vector_gateway.config import GatewayConfig
from vector_gateway.core.router import Router


class RouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GatewayConfig.model_validate(
            {
                "queues": {
                    "realtime": {"max_batch_size": 8, "max_wait_ms": 15, "max_concurrent_jobs": 1},
                    "interactive": {"max_batch_size": 32, "max_wait_ms": 50, "max_concurrent_jobs": 1},
                    "batch": {"max_batch_size": 128, "max_wait_ms": 200, "max_concurrent_jobs": 1},
                },
                "routing_rules": [
                    {"caller_pattern": "realtime/*", "queue": "realtime", "service_priority": 0, "operation": "query"},
                    {"caller_pattern": "interactive/*", "queue": "interactive", "service_priority": 1, "operation": "search"},
                    {"caller_pattern": "*", "queue": "batch", "service_priority": 3, "operation": "upsert"},
                ],
                "operation_priority": {"query": 0, "search": 1, "count": 1, "upsert": 2},
                "collections": {"documents": {"vector_size": 1024, "distance": "Cosine", "owner": "default"}},
            }
        )
        self.router = Router.from_config(self.config)

    def test_match_specific_rule(self) -> None:
        decision = self.router.resolve("realtime/demo")
        self.assertEqual(decision.queue_name, "realtime")
        self.assertEqual(decision.service_priority, 0)
        self.assertEqual(decision.operation, "query")
        self.assertEqual(decision.operation_priority, 0)

    def test_operation_override(self) -> None:
        decision = self.router.resolve("interactive/demo", "count")
        self.assertEqual(decision.queue_name, "interactive")
        self.assertEqual(decision.operation, "count")
        self.assertEqual(decision.operation_priority, 1)


if __name__ == "__main__":
    unittest.main()
